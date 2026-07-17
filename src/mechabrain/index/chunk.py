"""Chunking and deterministic Contextual Retrieval (§7.1).

Markdown headings are read as the author's own split *candidates*, and the
character budget decides which ones are taken. The corpus is authored and
atomic, so the typical note yields exactly **one** chunk -- the splitter exists
for the outliers (a long playbook), not for the common case. See :func:`_slice`.

Contextual Retrieval **without an LLM**
=======================================

Anthropic's Contextual Retrieval prepends a short description of the whole
document to every chunk before embedding it, so a fragment stops being an
orphan ("it improved 3%" -> "it" = which release?). The published recipe
generates that description with an LLM at ingestion time.

The kernel never calls an LLM (§13), and here it does not need to. The context
is already **written down**: `scope:`, `title:` and `tags:` are required
frontmatter (§6), and the heading path states where in the note the chunk sits.
So the prefix is *derived*, not *inferred* -- same note in, same bytes out,
forever, offline, at zero cost. On a corpus that is authored and atomic (P1),
LLM extraction at ingestion would mostly paraphrase the frontmatter while
adding cost, latency and a hallucination surface; §7.1 and §13 reject exactly
that trade.

Two texts, two jobs
===================

Each :class:`Chunk` carries both, and they must not be confused:

* ``embed_text`` -- what gets embedded and indexed, by the vector half and the
  BM25 half alike (:mod:`mechabrain.index.lexical`), so that a term in the
  prefix is findable on both sides. Carries the prefix.
* ``raw_text``   -- what a human is shown as the excerpt (R7.1). Never carries
  the prefix; nobody wants to read ``scope: ... | title: ...`` in a hit.

The prefix is not stored a third time on its own: it is a pure function of the
note and the heading path, so a caller that wants it alone calls
:func:`context_prefix`.

When ``retrieval.contextual_retrieval`` is false, ``embed_text == raw_text``.
That is the whole of the switch: it adds a prefix to one field and touches
nothing else.

Frontmatter is *used* but never *chunked*: :class:`~mechabrain.note.Note` has
already split it off, so YAML never lands in a chunk body -- only in its prefix.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from ..note import Note

__all__ = [
    "Chunk",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "HEADING_PATH_SEPARATOR",
    "CONTEXT_FIELD_SEPARATOR",
    "split_sections",
    "context_prefix",
    "chunk_note",
    "chunk_notes",
]

#: Character budget for one chunk's own body text. Roughly 300 tokens: well
#: inside every embedding model's window, and small enough that a hit points at
#: a paragraph a human can actually read.
DEFAULT_MAX_CHARS: Final[int] = 1200

#: Characters of the previous chunk replayed at the head of the next one, so a
#: statement split across a boundary stays retrievable from either side. Small
#: on purpose: headings already carry most of the continuity.
DEFAULT_OVERLAP_CHARS: Final[int] = 120

#: Renders a heading path: ``Deploy > Rollback``.
HEADING_PATH_SEPARATOR: Final[str] = " > "

#: Separates the fields of a context prefix.
CONTEXT_FIELD_SEPARATOR: Final[str] = " | "

# ATX heading: up to 3 leading spaces, 1-6 '#', then whitespace or end of line.
# The mandatory whitespace is what keeps `#tag` (a tag, in most vaults) from
# being read as a heading.
_ATX_HEADING: Final[re.Pattern[str]] = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")

# Fenced code block delimiter: ``` or ~~~ (3+), with an optional info string.
_FENCE: Final[re.Pattern[str]] = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s")


# ══════════════════════════════════════════════════════════════════════
# Chunk
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexable slice of a note.

    Frozen: a chunk is derived state. Anything wrong with it is wrong in the
    note or in the splitter, and is fixed by re-chunking -- never by mutating
    the chunk behind the index's back.

    Attributes:
        note_id: Id of the note this came from (:attr:`Note.note_id`), the join
            back to provenance (R7.1).
        ordinal: 0-based position within the note, in reading order.
        raw_text: The chunk as authored -- what a human is shown.
        embed_text: What is embedded and indexed: ``raw_text``, prefixed with
            :func:`context_prefix` when contextual retrieval is on.
        heading_path: Enclosing headings, outermost first. Empty for text
            before the first heading, or in a note with no headings.
    """

    note_id: str
    ordinal: int
    raw_text: str
    embed_text: str
    heading_path: tuple[str, ...] = ()

    @property
    def chunk_id(self) -> str:
        """Stable primary key for the index: ``<note_id>#<ordinal>``.

        Stable only for a fixed note body: editing a note re-chunks it and
        renumbers from 0, which is why reindexing a note deletes its chunks
        before inserting the new ones.
        """
        return f"{self.note_id}#{self.ordinal}"

    @property
    def section(self) -> str:
        """:attr:`heading_path` rendered for display; ``""`` at note level."""
        return HEADING_PATH_SEPARATOR.join(self.heading_path)


# ══════════════════════════════════════════════════════════════════════
# Fenced code
# ══════════════════════════════════════════════════════════════════════
def _fence_mask(lines: Sequence[str]) -> list[bool]:
    """Mark every line inside a fenced code block, delimiters included.

    Fenced content is opaque: a `# comment` in a shell block is not a heading,
    and a blank line in a diff is not a paragraph break. An unterminated fence
    runs to the end of the note, as CommonMark specifies.
    """
    mask = [False] * len(lines)
    open_char: str | None = None
    open_len = 0

    for i, line in enumerate(lines):
        match = _FENCE.match(line)
        if open_char is None:
            if match is not None:
                open_char = match.group(1)[0]
                open_len = len(match.group(1))
                mask[i] = True
            continue
        mask[i] = True
        # A closing fence matches the opener's character, is at least as long,
        # and carries no info string.
        if (
            match is not None
            and match.group(1)[0] == open_char
            and len(match.group(1)) >= open_len
            and not match.group(2).strip()
        ):
            open_char = None
    return mask


# ══════════════════════════════════════════════════════════════════════
# Sections
# ══════════════════════════════════════════════════════════════════════
def split_sections(body: str) -> list[tuple[tuple[str, ...], str]]:
    """Cut ``body`` into ``(heading_path, text)`` sections at ATX headings.

    Headings are the author's own segmentation -- honouring them beats any
    character heuristic, so the budget only ever applies *within* a section.

    Each section's text keeps its own heading line: it is real content for BM25
    and it is what a reader expects to see above the paragraph. A section with
    a heading but no body is dropped -- an empty container is not worth an
    embedding, and its title survives in its children's ``heading_path``.

    Returns:
        Sections in reading order. Empty for a blank body.
    """
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    mask = _fence_mask(lines)

    sections: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    heading_path: tuple[str, ...] = ()
    heading_line: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip("\n")
        if not text.strip():
            return
        sections.append(
            (heading_path, f"{heading_line}\n{text}" if heading_line else text)
        )

    for line, fenced in zip(lines, mask):
        match = None if fenced else _ATX_HEADING.match(line)
        if match is None:
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        # `## Title ##` -- a closing run of '#' is decoration, not the title.
        title = (match.group(2) or "").rstrip("#").strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        # An empty heading (`###`) still resets the level but names nothing.
        heading_path = tuple(text for _, text in stack if text)
        heading_line = line.strip()
        buffer = []

    flush()
    return sections


# ══════════════════════════════════════════════════════════════════════
# Splitting within a section
# ══════════════════════════════════════════════════════════════════════
def _blocks(text: str) -> list[str]:
    """Split a section into paragraphs, keeping fenced code blocks whole."""
    lines = text.split("\n")
    mask = _fence_mask(lines)
    blocks: list[str] = []
    buffer: list[str] = []

    for line, fenced in zip(lines, mask):
        if not fenced and not line.strip():
            if buffer:
                blocks.append("\n".join(buffer).strip("\n"))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        blocks.append("\n".join(buffer).strip("\n"))
    return [block for block in blocks if block.strip()]


def _hard_split(line: str, max_chars: int) -> list[str]:
    """Wrap one over-long line at word boundaries; mid-word only if forced."""
    if len(line) <= max_chars:
        return [line]
    parts: list[str] = []
    rest = line
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip(" ")
    if rest:
        parts.append(rest)
    return [part for part in parts if part]


def _is_heading_only(lines: Sequence[str]) -> bool:
    """Whether ``lines`` carry headings and blanks but no prose."""
    return bool(lines) and all(
        not line.strip() or _ATX_HEADING.match(line) for line in lines
    )


def _split_oversized(block: str, max_chars: int) -> list[str]:
    """Break a single over-long block on line, then word, then character.

    A section opens with its heading line, so the first line is usually cheap
    and the first prose line usually busts the budget on its own. Flushing
    there would emit `## Steps` as a whole chunk and then replay it as the next
    chunk's overlap -- a heading is a label for what follows, never a chunk. So
    a heading-only buffer refuses to flush and overruns ``max_chars`` instead.
    """
    pieces: list[str] = []
    buffer: list[str] = []
    size = 0

    for line in block.split("\n"):
        for part in _hard_split(line, max_chars):
            cost = len(part) + (1 if buffer else 0)
            if buffer and size + cost > max_chars and not _is_heading_only(buffer):
                pieces.append("\n".join(buffer))
                buffer = []
                cost = len(part)
            buffer.append(part)
            size += cost
    if buffer:
        pieces.append("\n".join(buffer))
    return pieces


def _atoms(text: str, max_chars: int) -> list[str]:
    """The indivisible pieces of a section, each within ``max_chars``."""
    atoms: list[str] = []
    for block in _blocks(text):
        if len(block) <= max_chars:
            atoms.append(block)
        else:
            atoms.extend(_split_oversized(block, max_chars))
    return atoms


def _pack(atoms: Sequence[str], max_chars: int) -> list[str]:
    """Greedily fill chunks with whole atoms, up to ``max_chars`` each."""
    pieces: list[str] = []
    buffer: list[str] = []
    size = 0

    for atom in atoms:
        cost = len(atom) + (2 if buffer else 0)
        if buffer and size + cost > max_chars:
            pieces.append("\n\n".join(buffer))
            buffer = []
            cost = len(atom)
        buffer.append(atom)
        size += cost
    if buffer:
        pieces.append("\n\n".join(buffer))
    return pieces


def _common_prefix(paths: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    """Deepest heading path that encloses every path in ``paths``."""
    if not paths:
        return ()
    prefix = paths[0]
    for path in paths[1:]:
        limit = min(len(prefix), len(path))
        cut = limit
        for i in range(limit):
            if prefix[i] != path[i]:
                cut = i
                break
        prefix = prefix[:cut]
        if not prefix:
            break
    return prefix


def _tail(text: str, overlap: int) -> str:
    """Last ``overlap`` characters of ``text``, snapped forward off a part word."""
    if len(text) <= overlap:
        return text.strip()
    window = text[-overlap:]
    match = _WHITESPACE.search(window)
    if match is not None:
        window = window[match.end() :]
    return window.strip()


def _apply_overlap(pieces: Sequence[str], overlap: int) -> list[str]:
    """Replay each piece's tail at the head of the next one."""
    if overlap <= 0 or len(pieces) < 2:
        return list(pieces)
    out = [pieces[0]]
    for previous, piece in zip(pieces, pieces[1:]):
        tail = _tail(previous, overlap)
        out.append(f"{tail}\n\n{piece}" if tail else piece)
    return out


def _slice(body: str, max_chars: int, overlap: int) -> list[tuple[tuple[str, ...], str]]:
    """Cut ``body`` into ``(heading_path, raw_text)`` pieces within the budget.

    Two passes, because headings are *split candidates* and the budget decides
    which ones are taken:

    1. Split at every heading, then break any section that busts the budget.
    2. Re-merge runs of adjacent sections that fit together in one chunk.

    Pass 2 is what keeps the promise that a typical atomic note is one chunk
    (§7.1): notes here are already the unit of meaning, and a 300-char note with
    three headings is one thought, not three. Splitting it would produce three
    thin embeddings competing for the same query where one strong one belongs --
    the fragmentation that contextual prefixing exists to *repair*. Never
    manufacture a fragment you then have to repair.

    A merged chunk's heading path is the deepest heading enclosing all of it,
    which is ``()`` for sibling sections. Nothing is lost: every heading line is
    still inside ``raw_text``.

    Overlap is applied only *within* a section. Across a heading the author
    already declared a topic change, and replaying the previous topic's tail
    into it would be noise, not continuity.
    """
    units: list[tuple[tuple[str, ...], list[str]]] = [
        (heading_path, _apply_overlap(_pack(_atoms(text, max_chars), max_chars), overlap))
        for heading_path, text in split_sections(body)
    ]

    pieces: list[tuple[tuple[str, ...], str]] = []
    run_paths: list[tuple[str, ...]] = []
    run_texts: list[str] = []
    size = 0

    def flush_run() -> None:
        nonlocal run_paths, run_texts, size
        if run_texts:
            pieces.append((_common_prefix(run_paths), "\n\n".join(run_texts)))
        run_paths, run_texts, size = [], [], 0

    for heading_path, section_pieces in units:
        if len(section_pieces) != 1:
            # Already at the budget; merging it with a neighbour is impossible.
            flush_run()
            pieces.extend((heading_path, piece) for piece in section_pieces)
            continue

        text = section_pieces[0]
        cost = len(text) + (2 if run_texts else 0)
        if run_texts and size + cost > max_chars:
            flush_run()
            cost = len(text)
        run_paths.append(heading_path)
        run_texts.append(text)
        size += cost
    flush_run()

    return [
        (heading_path, text.strip("\n"))
        for heading_path, text in pieces
        if text.strip()
    ]


# ══════════════════════════════════════════════════════════════════════
# Contextual Retrieval
# ══════════════════════════════════════════════════════════════════════
def context_prefix(note: Note, heading_path: Sequence[str] = ()) -> str:
    """Build the deterministic context prefix for a chunk of ``note`` (§7.1).

    Reads only what the author wrote: `scope:`, `title:` and `tags:` from
    frontmatter (§6) plus the chunk's heading path. Pure -- no I/O, no model,
    no clock. Absent fields are omitted rather than filled with a placeholder;
    a note with no frontmatter yields ``""`` and no prefix is applied.

    ``scope`` leads because it is the field that prevents cross-contamination
    (R6.5): it should weigh on the embedding, not just on the filter.

    Returns:
        One line, e.g. ``scope: proj-a | title: Vector search | tags:
        mem/semantic, agent/alpha | section: Steps > Build``, or ``""``.
    """
    fields: list[str] = []

    scope = note.get("scope")
    if scope is not None and str(scope).strip():
        fields.append(f"scope: {str(scope).strip()}")

    title = note.title.strip()
    if title:
        fields.append(f"title: {title}")

    tags = note.tags
    if tags:
        fields.append(f"tags: {', '.join(tags)}")

    if heading_path:
        fields.append(f"section: {HEADING_PATH_SEPARATOR.join(heading_path)}")

    return CONTEXT_FIELD_SEPARATOR.join(fields)


# ══════════════════════════════════════════════════════════════════════
# Entry points
# ══════════════════════════════════════════════════════════════════════
def chunk_note(
    note: Note,
    *,
    contextual: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split ``note`` into indexable chunks.

    Pass ``contextual=manifest.retrieval.contextual_retrieval``; the manifest is
    not read here so that this stays a pure function of a note and three numbers.

    ``max_chars`` bounds the body text a chunk *consumes*. Overlap is replayed
    on top of that, so ``raw_text`` may exceed ``max_chars`` by up to ``overlap``
    characters -- and ``embed_text`` further by the prefix. The budget is a
    splitting rule, not a hard cap on the string.

    Args:
        note: The note to chunk. Its frontmatter is already split off.
        contextual: Prefix ``embed_text`` with :func:`context_prefix`. When
            false, ``embed_text == raw_text``.
        max_chars: Per-chunk character budget.
        overlap: Characters of the previous chunk replayed at the head of the
            next, within a section. 0 disables it.

    Returns:
        Chunks in reading order, ``ordinal`` numbered from 0. Empty for a note
        with no body -- an index entry for nothing helps nobody.

    Raises:
        ValueError: ``max_chars`` is not positive, ``overlap`` is negative, or
            ``overlap >= max_chars`` (which would replay a whole chunk into the
            next one, and never terminate the shrinking).
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= max_chars:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than max_chars ({max_chars})"
        )

    chunks: list[Chunk] = []
    for heading_path, raw_text in _slice(note.body, max_chars, overlap):
        prefix = context_prefix(note, heading_path) if contextual else ""
        chunks.append(
            Chunk(
                note_id=note.note_id,
                ordinal=len(chunks),
                raw_text=raw_text,
                embed_text=f"{prefix}\n\n{raw_text}" if prefix else raw_text,
                heading_path=heading_path,
            )
        )
    return chunks


def chunk_notes(
    notes: Iterable[Note],
    *,
    contextual: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """:func:`chunk_note` over many notes, concatenated in the given order.

    ``ordinal`` restarts at 0 per note; :attr:`Chunk.chunk_id` is what stays
    unique across the corpus.
    """
    chunks: list[Chunk] = []
    for note in notes:
        chunks.extend(
            chunk_note(note, contextual=contextual, max_chars=max_chars, overlap=overlap)
        )
    return chunks
