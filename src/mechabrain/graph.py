"""The authored link graph (§7.1).

Graph-lite multi-hop, over the graph **humans and agents wrote** -- never one an
LLM extracted at ingestion. Three edge sources, all authored:

===============  ======================================================
Origin           Where the edge comes from
===============  ======================================================
``wikilink``     ``[[target]]`` in a note body -- what the author linked
``supersedes``   the ``supersedes:`` frontmatter key (§6) -- what
                 replaced what
``authored``     ``_meta/links.jsonl`` -- what ``memory_link`` recorded
===============  ======================================================

Why no extraction: in an authored, atomic corpus the links already *are* the
relations. Entity extraction would re-derive them worse, at ~1.4x ingest cost,
and hallucinate the rest (§13). So the graph improves by curation -- the better
agents link, the better multi-hop recall -- and this module contains no
judgement whatsoever: it resolves, counts and walks.

**Nodes are exactly the notes retrieval can return**: the four memory folders
(§3) plus ``zones.read_only_index``. The generated surfaces -- ``index.md``,
``indices/``, ``hot.md``, ``AGENTS.md`` -- are deliberately *not* nodes. Their
links are derived, not authored (R8.1), and a master MOC listing every note
would make every pair of notes two hops apart, collapsing expansion into "the
whole vault". ``_inbox/`` is out for the same reason it is not searchable: a
proposal is a request, not a memory.

Edges are **undirected for neighbourhood** -- if A cites B, B is relevant to a
query that hit A, and vice versa -- while each :class:`LinkEdge` keeps its
authored direction, so ``supersedes`` still reads new -> old.

Typical use::

    graph = LinkGraph.build(paths, manifest)
    reached = graph.expand([hit.id for hit in hits], hops=1)
    for note_id, via in reached.items():
        ...  # via.hops feeds the retrieval-side decay; via.render() is provenance
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .contract import MARKDOWN_SUFFIX, MemoryType
from .discovery import VaultPaths
from .errors import NoteNotFound, SchemaViolation
from .locking import file_lock
from .manifest import Manifest
from .note import Note, iter_notes, note_id_for, wikilink_for, write_atomic

__all__ = [
    "EdgeOrigin",
    "DropReason",
    "LinkEdge",
    "DroppedLink",
    "ViaChain",
    "LinkGraph",
    "DEFAULT_RELATION",
    "SUPERSEDES_RELATION",
    "WIKILINK_RELATION",
    "extract_wikilinks",
]

#: Relation `memory_link` records when the caller names none -- an authored
#: edge whose meaning the author did not bother to type.
DEFAULT_RELATION: Final[str] = "related"
#: Relation of an edge derived from `supersedes:` (§6). Directed new -> old.
SUPERSEDES_RELATION: Final[str] = "supersedes"
#: Relation of a plain body wikilink: the author cited, without saying why.
WIKILINK_RELATION: Final[str] = "links_to"

#: Lock guarding the read-modify-write of links.jsonl (R7.4 fallback). Lives in
#: the runtime layer: gitignored, per machine -- never next to the tracked file.
_LINKS_LOCK_NAME: Final[str] = "links.lock"

#: Keys one line of links.jsonl may carry. Strict, like the manifest (R5.1): a
#: typo'd key in a git-tracked source of truth is corruption, not a default.
_LINK_KEYS: Final[frozenset[str]] = frozenset({"a", "b", "relation", "created", "agent"})

# `[[target]]`, `[[target|alias]]`, `[[target#heading]]`, `![[embed]]`.
_WIKILINK_RE: Final[re.Pattern[str]] = re.compile(r"!?\[\[([^\[\]]+?)\]\]")
_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^(`{3,}|~{3,})")
_INLINE_CODE_RE: Final[re.Pattern[str]] = re.compile(r"`[^`\n]*`")


class EdgeOrigin(str, Enum):
    """Which authored surface an edge was read from.

    Distinct from the *relation*: the relation is what the author meant, the
    origin is where the kernel found it. Both matter -- `mechabrain check`
    reports by origin, retrieval weighs by relation.
    """

    WIKILINK = "wikilink"
    SUPERSEDES = "supersedes"
    AUTHORED = "authored"

    def __str__(self) -> str:
        return self.value


class DropReason(str, Enum):
    """Why a written link produced no edge.

    Neither is an error: a vault mid-edit has both. They are counted rather
    than raised so `mechabrain check` can report them and search cannot fail
    over one dangling wikilink.
    """

    #: The target resolves to no note the graph knows.
    BROKEN = "broken"
    #: The link points at its own note.
    SELF = "self"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class LinkEdge:
    """One authored relation between two notes, keeping its direction.

    Attributes:
        source: Note id the relation points *from* (the note holding the
            wikilink; the newer note, for ``supersedes``; ``a``, for an
            authored edge).
        target: Note id the relation points *to*.
        relation: What the author meant, e.g. ``links_to``, ``supersedes``.
        origin: Which surface it was read from.
        agent: Runtime that recorded it (authored edges only, P7).
        created: When it was recorded, ISO-8601 (authored edges only).
    """

    source: str
    target: str
    relation: str
    origin: EdgeOrigin
    agent: str | None = None
    created: str | None = None

    def other_end(self, note_id: str) -> str:
        """The end of this edge that is not ``note_id``.

        Raises:
            ValueError: ``note_id`` is not an endpoint of this edge.
        """
        if note_id == self.source:
            return self.target
        if note_id == self.target:
            return self.source
        raise ValueError(f"{note_id!r} is not an endpoint of {self!r}")

    def to_json_line(self) -> str:
        """Serialize as one line of links.jsonl, in the contract's key order."""
        return json.dumps(
            {
                "a": self.source,
                "b": self.target,
                "relation": self.relation,
                "created": self.created,
                "agent": self.agent,
            },
            ensure_ascii=False,
        )

    @property
    def _identity(self) -> tuple[str, str, str, str]:
        """What makes two edges the same edge. Deliberately excludes provenance."""
        return (self.source, self.target, self.relation, self.origin.value)


@dataclass(frozen=True, slots=True)
class DroppedLink:
    """A link that was written but yielded no edge. Counted, never raised.

    Attributes:
        source: Note the link was written in. For an authored edge whose ``a``
            end is the broken one, this is the raw, unresolved ``a``.
        raw_target: The link text as authored, before resolution.
        origin: Which surface it was read from.
        reason: Broken target, or a link to its own note.
    """

    source: str
    raw_target: str
    origin: EdgeOrigin
    reason: DropReason


@dataclass(frozen=True, slots=True)
class ViaChain:
    """How expansion reached a note: the path walked, and how far.

    ``path`` runs seed -> ... -> reached, inclusive of both, so a seed itself
    carries ``path == (seed,)`` and ``hops == 0``. ``relations[i]`` is the
    relation traversed between ``path[i]`` and ``path[i + 1]``.
    """

    path: tuple[str, ...]
    relations: tuple[str, ...]

    @property
    def hops(self) -> int:
        """Edges walked from the seed. ``0`` for a seed."""
        return len(self.path) - 1

    @property
    def seed(self) -> str:
        """The search hit this chain started from."""
        return self.path[0]

    @property
    def note_id(self) -> str:
        """The note this chain reached."""
        return self.path[-1]

    def render(self) -> str:
        """The provenance string a hit carries: ``[[seed]] → [[reached]]`` (§7.1).

        Renders the whole path, seed and reached note included -- a chain that
        named only the intermediates would not say where it started.
        """
        return " → ".join(wikilink_for(note_id) for note_id in self.path)

    def step(self, note_id: str, relation: str) -> "ViaChain":
        """Extend the chain by one hop to ``note_id``."""
        return ViaChain(self.path + (note_id,), self.relations + (relation,))


class LinkGraph:
    """The authored graph of one vault, as a snapshot.

    Build it with :meth:`build`; it reads the vault once and never watches it
    afterwards, so a long-lived caller rebuilds after a write. :meth:`add_edge`
    is the exception: it appends to ``links.jsonl`` *and* updates this snapshot,
    so `memory_link` immediately affects the next `memory_search`.

    Not thread-safe. Cross-process safety is the ``links.jsonl`` lock plus the
    R7.4 single-writer daemon.
    """

    __slots__ = (
        "_paths",
        "_manifest",
        "_note_paths",
        "_lower_ids",
        "_ambiguous_ids",
        "_edges",
        "_adjacency",
        "_dropped",
    )

    def __init__(
        self,
        paths: VaultPaths,
        manifest: Manifest,
        note_paths: Mapping[str, Path],
        ambiguous_ids: Sequence[str],
        edges: Sequence[LinkEdge],
        dropped: Sequence[DroppedLink],
    ) -> None:
        """Not the public constructor: use :meth:`build`, which does the reading."""
        self._paths = paths
        self._manifest = manifest
        self._note_paths = dict(note_paths)
        self._ambiguous_ids = tuple(ambiguous_ids)
        self._lower_ids = _lowercase_index(self._note_paths, self._ambiguous_ids)
        self._edges = tuple(edges)
        self._dropped = tuple(dropped)
        self._adjacency = _build_adjacency(self._edges)

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        paths: VaultPaths,
        manifest: Manifest,
        *,
        notes: Iterable[Note] | None = None,
    ) -> "LinkGraph":
        """Read the vault and resolve every authored link into a graph.

        Args:
            paths: The vault, from :func:`mechabrain.discovery.discover_vault`.
            manifest: Its parsed manifest -- supplies ``zones.read_only_index``,
                ``zones.research_enabled`` and the hop ceiling.
            notes: Already-scanned notes, to avoid a second pass over the vault
                (the indexer has them). Must be the same set :meth:`build`
                would scan; anything outside the memory folders and
                ``read_only_index`` becomes a node it should not be. Defaults to
                scanning.

        Returns:
            A snapshot. Broken and self links are counted, never raised.

        Raises:
            SchemaViolation: ``links.jsonl`` has a malformed line. The message
                names the line number: it is git-tracked authored data, so a
                bad line is corruption to fix, not noise to skip (R5.1).
        """
        scanned = list(cls._scan(paths, manifest) if notes is None else notes)
        note_paths, ambiguous = _index_notes(scanned)
        lower_ids = _lowercase_index(note_paths, ambiguous)

        edges: list[LinkEdge] = []
        dropped: list[DroppedLink] = []
        seen: set[tuple[str, str, str, str]] = set()

        def keep(edge: LinkEdge) -> None:
            if edge._identity not in seen:
                seen.add(edge._identity)
                edges.append(edge)

        for note in scanned:
            note_id = note.note_id
            for raw_target, origin, relation in _authored_links_of(note):
                resolved = _resolve(raw_target, note_paths, lower_ids)
                if resolved is None:
                    dropped.append(DroppedLink(note_id, raw_target, origin, DropReason.BROKEN))
                    continue
                if resolved == note_id:
                    dropped.append(DroppedLink(note_id, raw_target, origin, DropReason.SELF))
                    continue
                keep(LinkEdge(note_id, resolved, relation, origin))

        for record in _read_links_file(paths.links_file):
            edge = record.resolve(note_paths, lower_ids)
            if isinstance(edge, DroppedLink):
                dropped.append(edge)
            else:
                keep(edge)

        return cls(paths, manifest, note_paths, ambiguous, edges, dropped)

    @staticmethod
    def _scan(paths: VaultPaths, manifest: Manifest) -> list[Note]:
        """Every note that is a graph node: memory folders + read_only_index.

        Excludes the generated surfaces on purpose -- see the module docstring.
        Deduplicates by path, so a ``read_only_index`` entry pointing back into
        ``mecha-brain/`` cannot double-count a note.
        """
        folders: list[Path] = [
            paths.folder_for(memory_type)
            for memory_type in MemoryType
            if manifest.is_enabled(memory_type)
        ]
        folders += [paths.resolve(folder) for folder in manifest.zones.read_only_index]

        notes: dict[Path, Note] = {}
        for folder in folders:
            for note in iter_notes(folder):
                assert note.path is not None  # iter_notes always sets it
                notes.setdefault(note.path, note)
        return [notes[path] for path in sorted(notes)]

    # ── Nodes ───────────────────────────────────────────────────────
    @property
    def note_ids(self) -> tuple[str, ...]:
        """Every node, sorted. A node is a note retrieval can return."""
        return tuple(sorted(self._note_paths))

    def has_note(self, note_id: str) -> bool:
        """Whether ``note_id`` is a node of this graph."""
        return note_id in self._note_paths

    def path_of(self, note_id: str) -> Path:
        """Where ``note_id`` lives.

        Raises:
            NoteNotFound: no such node.
        """
        try:
            return self._note_paths[note_id]
        except KeyError:
            raise NoteNotFound(f"no note {note_id!r} in the graph") from None

    @property
    def ambiguous_ids(self) -> tuple[str, ...]:
        """Ids claimed by more than one note, sorted.

        A note id is a basename (:func:`mechabrain.note.note_id_for`), so two
        notes in different folders can collide -- ``read_only_index`` over a
        human folder tree makes that likely. The first in path order owns the
        id and every link to it resolves there; the rest are unreachable by
        wikilink, exactly as they are in a Markdown editor. Reported here so
        `mechabrain check` can surface them instead of the graph guessing
        quietly.
        """
        return self._ambiguous_ids

    # ── Edges ───────────────────────────────────────────────────────
    @property
    def edges(self) -> tuple[LinkEdge, ...]:
        """Every resolved edge, deduplicated, in discovery order.

        Two authored surfaces asserting the same relation between the same pair
        yield one edge; the same pair with different relations yields two.
        """
        return self._edges

    @property
    def dropped_links(self) -> tuple[DroppedLink, ...]:
        """Every link that produced no edge -- broken and self, in order."""
        return self._dropped

    @property
    def broken_links(self) -> tuple[DroppedLink, ...]:
        """Links whose target is no note. Ignored by the walk, counted here."""
        return tuple(d for d in self._dropped if d.reason is DropReason.BROKEN)

    @property
    def self_links(self) -> tuple[DroppedLink, ...]:
        """Links from a note to itself. A note is not its own neighbour."""
        return tuple(d for d in self._dropped if d.reason is DropReason.SELF)

    def neighbors(self, note_id: str) -> tuple[LinkEdge, ...]:
        """Every edge incident to ``note_id``, in either direction.

        Undirected on purpose: if A cites B, B is context for a query that hit
        A, and B being cited by A is just as informative in reverse. Each edge
        still carries its authored direction -- use
        :meth:`LinkEdge.other_end` to get the far side.

        An unknown or isolated node returns ``()``; use :meth:`has_note` to
        tell those apart.
        """
        return self._adjacency.get(note_id, ())

    def neighbor_ids(self, note_id: str) -> tuple[str, ...]:
        """Ids one hop from ``note_id``, deduplicated and sorted."""
        return tuple(
            dict.fromkeys(edge.other_end(note_id) for edge in self.neighbors(note_id))
        )

    # ── Writing (memory_link, §7.2) ─────────────────────────────────
    def add_edge(
        self,
        a: str,
        b: str,
        relation: str = DEFAULT_RELATION,
        agent: str | None = None,
        *,
        created: str | None = None,
    ) -> LinkEdge:
        """Record an authored edge in ``links.jsonl`` and in this snapshot.

        The file is append-only and **git-tracked**: an authored edge is a
        source of truth, not derived state, so it belongs in ``_meta/`` beside
        the manifest and travels with the vault -- unlike ``_meta/index/``,
        which any machine rebuilds. The append is atomic (R7.5) and taken under
        a lock (R7.4), so a concurrent `memory_link` cannot interleave a line.

        Idempotent: adding an edge that is already recorded returns the existing
        one and writes nothing. `memory_link` is a natural thing for an agent to
        repeat, and an append-only log that grows on every repeat is a log that
        gets truncated by hand.

        Args:
            a: Source note -- id, wikilink or path; all resolve the same way.
            b: Target note.
            relation: What the author means by the edge.
            agent: Runtime recording it. Validated against the registry (R6.2);
                ``None`` records an edge with no author, which the CLI may do
                but an agent never should (P7).
            created: ISO-8601 stamp. Defaults to now, UTC.

        Returns:
            The edge, as recorded.

        Raises:
            NoteNotFound: either end resolves to no node.
            ValueError: ``a`` and ``b`` are the same note, or ``relation`` is
                empty.
            ManifestError: ``agent`` is not in the registry (R6.2).
        """
        source = self._require_note(a, "a")
        target = self._require_note(b, "b")
        if source == target:
            raise ValueError(
                f"cannot link {source!r} to itself: an edge relates two notes"
            )
        label = relation.strip()
        if not label:
            raise ValueError(f"relation must not be empty; default is {DEFAULT_RELATION!r}")
        if agent is not None:
            self._manifest.agent(agent)  # R6.2: refuse an unknown author.

        edge = LinkEdge(
            source=source,
            target=target,
            relation=label,
            origin=EdgeOrigin.AUTHORED,
            agent=agent,
            created=created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        with file_lock(
            self._paths.index_dir / _LINKS_LOCK_NAME, purpose="memory_link"
        ):
            raw = _read_text(self._paths.links_file)
            for record in _parse_links(raw, self._paths.links_file):
                existing = record.resolve(self._note_paths, self._lower_ids)
                if isinstance(existing, LinkEdge) and existing._identity == edge._identity:
                    self._remember(existing)
                    return existing
            if raw and not raw.endswith("\n"):
                raw += "\n"
            write_atomic(self._paths.links_file, f"{raw}{edge.to_json_line()}\n")

        self._remember(edge)
        return edge

    def _remember(self, edge: LinkEdge) -> None:
        """Fold ``edge`` into the snapshot, so the next search sees it."""
        if any(known._identity == edge._identity for known in self._edges):
            return
        self._edges = self._edges + (edge,)
        self._adjacency = _build_adjacency(self._edges)

    def _require_note(self, value: str, argument: str) -> str:
        resolved = _resolve(value, self._note_paths, self._lower_ids)
        if resolved is None:
            raise NoteNotFound(
                f"{argument}={value!r} resolves to no note in the graph",
                hint=(
                    "link notes the kernel indexes: the memory folders, or a "
                    "folder listed in zones.read_only_index"
                ),
            )
        return resolved

    # ── Expansion (§7.1) ────────────────────────────────────────────
    def expand(
        self, seed_ids: Iterable[str], hops: int | None = None
    ) -> dict[str, ViaChain]:
        """Walk ``hops`` edges out from ``seed_ids``, keeping the shortest chain.

        Breadth-first over the undirected neighbourhood, so every note is
        reached by its fewest hops and cycles terminate. Where two chains of
        equal length reach the same note, the lexicographically smaller path
        wins -- the result must not depend on iteration order.

        The seeds themselves are in the result at ``hops == 0``: they are the
        search hits, and ``expand(seeds, 0)`` returning exactly the seeds is
        what makes ``link_expansion.default_hops: 0`` mean "off" (§5). A seed
        that is not a node contributes nothing -- it can still be returned as a
        hit, it just has no authored neighbourhood.

        Scoring is *not* done here: this returns reachability and provenance,
        and the retrieval layer applies the per-hop decay and the fusion
        weights it reads from the manifest.

        Args:
            seed_ids: Note ids of the top-k hits.
            hops: How far to walk. Defaults to
                ``retrieval.link_expansion.default_hops``.

        Returns:
            ``note_id -> ViaChain``, including the seeds.

        Raises:
            ValueError: ``hops`` is negative, or above
                ``retrieval.link_expansion.max_hops``. The ceiling is refused,
                not clamped: silently returning a 2-hop neighbourhood to a
                caller who asked for 5 would misreport how far the kernel
                looked (R5.1).
        """
        limit = self._resolve_hops(hops)

        chains: dict[str, ViaChain] = {
            seed: ViaChain((seed,), ())
            for seed in sorted(set(seed_ids))
            if seed in self._note_paths
        }
        frontier = sorted(chains)

        for _ in range(limit):
            if not frontier:
                break
            pending: dict[str, ViaChain] = {}
            for current in frontier:
                chain = chains[current]
                for edge in self.neighbors(current):
                    reached = edge.other_end(current)
                    if reached in chains:
                        continue  # Already reached in fewer hops, or a seed.
                    candidate = chain.step(reached, edge.relation)
                    incumbent = pending.get(reached)
                    if incumbent is None or candidate.path < incumbent.path:
                        pending[reached] = candidate
            chains.update(pending)
            frontier = sorted(pending)

        return chains

    def _resolve_hops(self, hops: int | None) -> int:
        expansion = self._manifest.retrieval.link_expansion
        limit = expansion.default_hops if hops is None else hops
        if limit < 0:
            raise ValueError(f"hops must be >= 0, got {limit}")
        if limit > expansion.max_hops:
            raise ValueError(
                f"hops={limit} exceeds retrieval.link_expansion.max_hops "
                f"({expansion.max_hops}); raise the ceiling in the manifest to walk further"
            )
        return limit

    def __repr__(self) -> str:
        return (
            f"LinkGraph({len(self._note_paths)} notes, {len(self._edges)} edges, "
            f"{len(self.broken_links)} broken)"
        )


# ══════════════════════════════════════════════════════════════════════
# Wikilinks
# ══════════════════════════════════════════════════════════════════════
def extract_wikilinks(body: str) -> list[str]:
    """Every wikilink target written in ``body``, in order, with duplicates.

    Understands the forms a Markdown vault actually contains: ``[[target]]``,
    ``[[target|alias]]``, ``[[target#heading]]``, ``[[target#^block]]`` and the
    embed ``![[target]]`` -- an embed is a citation, so it is an edge.

    Code is not prose: fenced blocks and inline spans are stripped first, so a
    documented example of the syntax does not become an edge. ``[[#heading]]``
    is a jump within the same note and yields nothing.

    Returns:
        Raw target texts, unresolved -- resolution needs the note index.
    """
    targets: list[str] = []
    for match in _WIKILINK_RE.finditer(_strip_code(body)):
        target = match.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            targets.append(target)
    return targets


def _strip_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans, preserving line count."""
    out: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if fence is None:
            opening = _FENCE_RE.match(stripped)
            if opening is not None:
                fence = opening.group(1)[:3]
                out.append("")
                continue
            out.append(_INLINE_CODE_RE.sub(" ", line))
        else:
            if stripped.startswith(fence):
                fence = None
            out.append("")
    return "\n".join(out)


def _supersedes_targets(value: Any) -> list[str]:
    """Targets named by a ``supersedes:`` frontmatter value (§6).

    The spec writes it as one wikilink; humans write a bare id, and a note that
    merges two others names both. All three are read.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [target for item in value for target in _supersedes_targets(item)]
    text = str(value).strip()
    if not text:
        return []
    if "[[" in text:
        return extract_wikilinks(text)
    return [text]


def _authored_links_of(note: Note) -> list[tuple[str, EdgeOrigin, str]]:
    """Every link one note authors: ``(raw_target, origin, relation)``."""
    links: list[tuple[str, EdgeOrigin, str]] = [
        (target, EdgeOrigin.SUPERSEDES, SUPERSEDES_RELATION)
        for target in _supersedes_targets(note.get("supersedes"))
    ]
    links += [
        (target, EdgeOrigin.WIKILINK, WIKILINK_RELATION)
        for target in extract_wikilinks(note.body)
    ]
    return links


# ══════════════════════════════════════════════════════════════════════
# Resolution
# ══════════════════════════════════════════════════════════════════════
def _index_notes(notes: Sequence[Note]) -> tuple[dict[str, Path], tuple[str, ...]]:
    """Map note id -> path, first in path order winning a collision."""
    by_id: dict[str, Path] = {}
    ambiguous: list[str] = []
    for note in sorted(notes, key=lambda n: str(n.path)):
        if note.path is None:
            continue
        note_id = note.note_id
        if note_id in by_id:
            if note_id not in ambiguous:
                ambiguous.append(note_id)
            continue
        by_id[note_id] = note.path
    return by_id, tuple(sorted(ambiguous))


def _lowercase_index(
    note_paths: Mapping[str, Path], ambiguous_ids: Sequence[str]
) -> dict[str, str]:
    """Case-folded id -> id, for the case-insensitive fallback.

    Markdown editors resolve ``[[vector-search]]`` to ``Vector-Search.md``, so
    the kernel does too. Ids that collide once folded are left out: guessing
    between two notes is worse than reporting a broken link.
    """
    buckets: dict[str, list[str]] = {}
    for note_id in note_paths:
        buckets.setdefault(note_id.lower(), []).append(note_id)
    return {
        folded: ids[0]
        for folded, ids in buckets.items()
        if len(ids) == 1 and ids[0] not in ambiguous_ids
    }


def _normalize_target(raw: str) -> str:
    """Reduce a link target to the id it names.

    Absorbs the forms a vault mixes freely: ``[[id]]``, ``id.md``, and the
    path-qualified ``Folder/Sub/id`` an editor writes when basenames collide.
    """
    text = raw.strip()
    if text.startswith("![["):
        text = text[3:]
    elif text.startswith("[["):
        text = text[2:]
    if text.endswith("]]"):
        text = text[:-2]
    text = text.split("|", 1)[0].split("#", 1)[0].strip().strip("/")
    if not text:
        return ""
    return note_id_for(text) if text.endswith(MARKDOWN_SUFFIX) else Path(text).name


def _resolve(
    raw: str, note_paths: Mapping[str, Path], lower_ids: Mapping[str, str]
) -> str | None:
    """Resolve a link target to a note id, or ``None`` if it names none."""
    target = _normalize_target(raw)
    if not target:
        return None
    if target in note_paths:
        return target
    return lower_ids.get(target.lower())


def _build_adjacency(edges: Sequence[LinkEdge]) -> dict[str, tuple[LinkEdge, ...]]:
    """Undirected incidence lists, each sorted for a deterministic walk."""
    lists: dict[str, list[LinkEdge]] = {}
    for edge in edges:
        lists.setdefault(edge.source, []).append(edge)
        lists.setdefault(edge.target, []).append(edge)
    return {
        note_id: tuple(
            sorted(
                incident,
                key=lambda e: (e.other_end(note_id), e.relation, e.origin.value),
            )
        )
        for note_id, incident in lists.items()
    }


# ══════════════════════════════════════════════════════════════════════
# links.jsonl (§7.2)
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class _LinkRecord:
    """One parsed, still-unresolved line of links.jsonl."""

    a: str
    b: str
    relation: str
    created: str | None
    agent: str | None

    def resolve(
        self, note_paths: Mapping[str, Path], lower_ids: Mapping[str, str]
    ) -> LinkEdge | DroppedLink:
        """Turn this record into an edge, or say why it is not one."""
        source = _resolve(self.a, note_paths, lower_ids)
        target = _resolve(self.b, note_paths, lower_ids)
        if source is None or target is None:
            return DroppedLink(
                source=source or self.a,
                raw_target=self.b,
                origin=EdgeOrigin.AUTHORED,
                reason=DropReason.BROKEN,
            )
        if source == target:
            return DroppedLink(source, self.b, EdgeOrigin.AUTHORED, DropReason.SELF)
        return LinkEdge(
            source=source,
            target=target,
            relation=self.relation,
            origin=EdgeOrigin.AUTHORED,
            agent=self.agent,
            created=self.created,
        )


def _read_text(path: Path) -> str:
    """Read ``path``, or ``""`` if it does not exist yet.

    An absent links.jsonl is the normal state of a fresh vault: no agent has
    linked anything. An unreadable one is not.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise SchemaViolation(f"cannot read {path}: {exc}", rule="R7.2") from exc
    except UnicodeDecodeError as exc:
        raise SchemaViolation(f"{path} is not valid UTF-8: {exc}", rule="R7.2") from exc


def _read_links_file(path: Path) -> list[_LinkRecord]:
    """Parse links.jsonl at ``path``. Absent file -> no records."""
    return _parse_links(_read_text(path), path)


def _parse_links(text: str, path: Path) -> list[_LinkRecord]:
    """Parse links.jsonl content, failing loud with a line number (R5.1).

    Strict, because this file is git-tracked authored truth: a line the kernel
    skipped quietly would be an edge an agent believes it recorded and the
    graph never walks. A merge conflict here must stop the kernel, not degrade
    it.
    """
    records: list[_LinkRecord] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        records.append(_parse_link_line(line, path, number))
    return records


def _parse_link_line(line: str, path: Path, number: int) -> _LinkRecord:
    where = f"{path}:{number}"
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(
            f"{where}: not valid JSON: {exc}",
            rule="R7.2",
            hint="each line is one JSON object: "
            '{"a": ..., "b": ..., "relation": ..., "created": ..., "agent": ...}',
        ) from exc
    if not isinstance(data, Mapping):
        raise SchemaViolation(
            f"{where}: expected a JSON object, got {type(data).__name__}", rule="R7.2"
        )
    unknown = sorted(set(map(str, data)) - _LINK_KEYS)
    if unknown:
        raise SchemaViolation(
            f"{where}: unknown key(s): {', '.join(unknown)}",
            rule="R7.2",
            hint=f"valid keys: {', '.join(sorted(_LINK_KEYS))}",
        )
    return _LinkRecord(
        a=_required_str(data, "a", where),
        b=_required_str(data, "b", where),
        relation=_optional_str(data, "relation", where) or DEFAULT_RELATION,
        created=_optional_str(data, "created", where),
        agent=_optional_str(data, "agent", where),
    )


def _required_str(data: Mapping[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaViolation(
            f"{where}: {key!r} must be a non-empty string, got {value!r}", rule="R7.2"
        )
    return value.strip()


def _optional_str(data: Mapping[str, Any], key: str, where: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaViolation(
            f"{where}: {key!r} must be a string or null, got {value!r}", rule="R7.2"
        )
    return value.strip() or None
