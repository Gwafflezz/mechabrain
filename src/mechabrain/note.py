"""The note model: Markdown + YAML frontmatter.

Markdown is the source of truth (P1); every index is derived from it. So this
module is deliberately conservative: it parses what a human or another tool may
have written, preserves key order on round-trip, and writes atomically (R7.5).

Robustness contract -- all of these are handled, none raise:

* no frontmatter at all (plain Markdown);
* an empty frontmatter block (``---\\n---``);
* ``---`` occurring inside the body (only the *first* closing fence ends the
  frontmatter);
* CRLF line endings (normalised to LF on parse);
* a leading BOM;
* non-ASCII content (read and written as UTF-8).

Only genuinely malformed frontmatter -- invalid YAML, or YAML that is not a
mapping -- raises :class:`~mechabrain.errors.SchemaViolation`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .contract import FRONTMATTER_FENCE, MARKDOWN_SUFFIX
from .errors import NoteNotFound, SchemaViolation

__all__ = [
    "Note",
    "parse_frontmatter",
    "serialize_frontmatter",
    "render_note",
    "note_id_for",
    "wikilink_for",
    "read_note",
    "iter_notes",
    "scan_notes",
    "write_atomic",
    "normalize_tags",
]

_BOM = "\ufeff"


class _FrontmatterDumper(yaml.SafeDumper):
    """SafeDumper that indents block sequences under their key.

    PyYAML's default emits list items flush with the parent key, which is legal
    YAML but reads badly in an editor showing a human's own vault. Aliases are
    disabled for the same reason: reusing one date object for `created`,
    `modified` and `last_accessed` would otherwise emit `&id001`/`*id001`
    anchors -- legal YAML that diverges from the §6 canonical form and confuses
    frontmatter parsers in host apps.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)

    def ignore_aliases(self, data: Any) -> bool:
        return True


# ══════════════════════════════════════════════════════════════════════
# Frontmatter
# ══════════════════════════════════════════════════════════════════════
def parse_frontmatter(text: str, *, path: Path | None = None) -> tuple[dict[str, Any], str]:
    """Split ``text`` into ``(frontmatter, body)``.

    Key order is preserved: YAML mappings are constructed in document order and
    Python dicts keep insertion order, so parse -> serialize round-trips without
    reshuffling a human's frontmatter.

    A document that opens with ``---`` but never closes it is treated as having
    **no** frontmatter -- a lone ``---`` is a legal Markdown horizontal rule and
    guessing otherwise would swallow the body.

    Args:
        text: Full file content.
        path: Origin, used only to make errors locatable.

    Returns:
        The frontmatter mapping (empty if absent) and the body.

    Raises:
        SchemaViolation: the frontmatter block is invalid YAML or is not a mapping.
    """
    normalized = text.lstrip(_BOM).replace("\r\n", "\n").replace("\r", "\n")

    if not normalized.startswith(FRONTMATTER_FENCE):
        return {}, normalized

    lines = normalized.split("\n")
    if lines[0].strip() != FRONTMATTER_FENCE:
        # e.g. "----" or "--- text": not a fence.
        return {}, normalized

    closing = _find_closing_fence(lines)
    if closing is None:
        return {}, normalized

    raw = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1 :])

    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SchemaViolation(
            f"invalid YAML in frontmatter{_where(path)}: {exc}",
            hint="check indentation, and quote values containing ':' or '[['",
        ) from exc

    if loaded is None:
        return {}, body
    if not isinstance(loaded, Mapping):
        raise SchemaViolation(
            f"frontmatter{_where(path)} must be a mapping, got "
            f"{type(loaded).__name__}",
            hint="frontmatter is a block of `key: value` pairs",
        )
    return {str(key): value for key, value in loaded.items()}, body


def _find_closing_fence(lines: list[str]) -> int | None:
    """Index of the first ``---`` line after the opening fence, or None."""
    for i in range(1, len(lines)):
        if lines[i].strip() == FRONTMATTER_FENCE:
            return i
    return None


def serialize_frontmatter(frontmatter: Mapping[str, Any]) -> str:
    """Render ``frontmatter`` as a fenced YAML block ending in a newline.

    Emits keys in the mapping's own order (never sorted) and UTF-8 as-is.
    Returns ``""`` for an empty mapping -- a note with no frontmatter gets no
    empty fence.
    """
    if not frontmatter:
        return ""
    body = yaml.dump(
        dict(frontmatter),
        Dumper=_FrontmatterDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=4096,
    )
    return f"{FRONTMATTER_FENCE}\n{body}{FRONTMATTER_FENCE}\n"


def render_note(frontmatter: Mapping[str, Any], body: str) -> str:
    """Compose a full note, normalising the body to exactly one trailing newline."""
    text = body.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    head = serialize_frontmatter(frontmatter)
    if not text:
        return head or "\n"
    return f"{head}\n{text}\n" if head else f"{text}\n"


def normalize_tags(value: Any) -> list[str]:
    """Coerce a frontmatter `tags:` value to a list of strings.

    Accepts a YAML list, a single string, a comma/space-separated string, or
    ``None``. Strips a leading ``#`` so ``#mem/semantic`` and ``mem/semantic``
    compare equal against a denylist (R6.1).
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value]
    else:
        parts = [str(value).strip()]
    return [part.lstrip("#") for part in parts if part and part.strip("#")]


# ══════════════════════════════════════════════════════════════════════
# Identity
# ══════════════════════════════════════════════════════════════════════
def note_id_for(path: Path | str) -> str:
    """Stable id of a note: its basename without ``.md``.

    Chosen to match how wikilinks resolve in Markdown vaults, so the id a
    retrieval hit carries (R7.1) is the string an agent can paste into a note
    and have it link. It follows the file if the note is moved between folders,
    and is unique only if basenames are -- which the `naming` templates ensure
    by carrying a date and a slug.
    """
    return Path(path).name.removesuffix(MARKDOWN_SUFFIX)


def wikilink_for(target: Path | str) -> str:
    """Render ``target`` (path or id) as ``[[wikilink]]`` (P7, R7.1)."""
    return f"[[{note_id_for(target)}]]"


# ══════════════════════════════════════════════════════════════════════
# Note
# ══════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Note:
    """One Markdown note: its path, its frontmatter and its body.

    Mutable by design -- consolidation edits frontmatter in place (`status`,
    `last_accessed`) and writes the note back. ``path`` is ``None`` for a note
    that has not been placed on disk yet.
    """

    path: Path | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def parse(cls, text: str, *, path: Path | str | None = None) -> "Note":
        """Build a note from raw file content. See :func:`parse_frontmatter`."""
        note_path = Path(path) if path is not None else None
        frontmatter, body = parse_frontmatter(text, path=note_path)
        return cls(path=note_path, frontmatter=frontmatter, body=body)

    @classmethod
    def load(cls, path: Path | str) -> "Note":
        """Read and parse the note at ``path``.

        Raises:
            NoteNotFound: no readable file at ``path``.
            SchemaViolation: the frontmatter is malformed.
        """
        note_path = Path(path)
        try:
            text = note_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise NoteNotFound(f"no note at {note_path}") from exc
        except OSError as exc:
            raise NoteNotFound(f"cannot read note at {note_path}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise SchemaViolation(
                f"note at {note_path} is not valid UTF-8: {exc}",
                hint="notes are UTF-8 Markdown",
            ) from exc
        return cls.parse(text, path=note_path)

    # ── Identity ────────────────────────────────────────────────────
    @property
    def note_id(self) -> str:
        """Stable id (basename without ``.md``); ``""`` for an unplaced note."""
        return note_id_for(self.path) if self.path is not None else ""

    @property
    def wikilink(self) -> str:
        """``[[id]]`` form, for citing this note as provenance (R7.1)."""
        return f"[[{self.note_id}]]"

    @property
    def title(self) -> str:
        """`title:` from frontmatter, falling back to the note id."""
        value = self.frontmatter.get("title")
        return str(value) if value not in (None, "") else self.note_id

    @property
    def tags(self) -> list[str]:
        """Frontmatter tags, normalised. See :func:`normalize_tags`."""
        return normalize_tags(self.frontmatter.get("tags"))

    def has_tag(self, tag: str) -> bool:
        """Whether the note carries ``tag`` (``#`` optional)."""
        return tag.lstrip("#") in self.tags

    def get(self, key: str, default: Any = None) -> Any:
        """Read a frontmatter key."""
        return self.frontmatter.get(key, default)

    # ── Output ──────────────────────────────────────────────────────
    def to_markdown(self) -> str:
        """Serialize back to file content, preserving frontmatter key order."""
        return render_note(self.frontmatter, self.body)

    def write(self, path: Path | str | None = None) -> Path:
        """Write the note atomically (R7.5) and remember where it went.

        Args:
            path: Destination. Defaults to ``self.path``.

        Returns:
            The path written.

        Raises:
            ValueError: no path given and the note has none.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("cannot write a note with no path: pass path=")
        write_atomic(target, self.to_markdown())
        self.path = target
        return target


# ══════════════════════════════════════════════════════════════════════
# Filesystem
# ══════════════════════════════════════════════════════════════════════
def write_atomic(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    ensure_parents: bool = True,
) -> Path:
    """Write ``text`` to ``path`` atomically (R7.5).

    Writes a temporary file **in the destination directory** -- so the rename
    stays on one filesystem and cannot degrade to a copy -- fsyncs it, then
    ``os.replace``s it over the target. A reader never observes a partial note,
    and a crash mid-write leaves the previous version intact.

    Not a substitute for a lock: it serialises *bytes*, not *decisions*. A
    read-modify-write cycle still needs :func:`mechabrain.locking.file_lock`.

    Returns:
        The path written.
    """
    target = Path(path)
    if ensure_parents:
        target.parent.mkdir(parents=True, exist_ok=True)

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        tmp_name = None
    finally:
        if tmp_name is not None:
            # The replace never happened; do not leave debris in the vault.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return target


def read_note(path: Path | str) -> Note:
    """Read one note. Module-level alias of :meth:`Note.load`."""
    return Note.load(path)


def iter_notes(folder: Path | str, *, recursive: bool = True) -> Iterator[Note]:
    """Yield every ``.md`` note under ``folder``, in sorted path order.

    A missing folder yields nothing: `Research/` legitimately does not exist
    when `zones.research_enabled` is false. Dotfiles and the temporary files of
    :func:`write_atomic` are skipped.

    Raises:
        SchemaViolation: a note on the way has malformed frontmatter. Sorted
            order makes that deterministic and thus reportable.
    """
    root = Path(folder)
    if not root.is_dir():
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.suffix != MARKDOWN_SUFFIX:
            continue
        if path.name.startswith("."):
            continue
        yield Note.load(path)


def scan_notes(folder: Path | str, *, recursive: bool = True) -> list[Note]:
    """Eager :func:`iter_notes`."""
    return list(iter_notes(folder, recursive=recursive))


def _where(path: Path | None) -> str:
    return f" in {path}" if path is not None else ""
