"""memory.search: hybrid retrieval with link expansion (spec §7.1).

The read half of the MCP contract. :meth:`Retriever.search` runs the query
through both halves of the index, fuses them with the manifest's weights, keeps
the memory boundaries the deployment declared, walks the authored graph out from
the best hits, and returns provenance-bearing :class:`Hit`\\ s an agent can cite
(R7.1). It never calls an LLM: every step here is mechanically defined and
reproducible on the same inputs (§13).

The pipeline, in order (§7.1)
=============================

1. **Embed + vector search.** The query is embedded (no context prefix -- that
   is a *document* transform, §7.1) and matched against the vector store, which
   returns raw cosine per chunk.
2. **BM25 search.** The same query text is run through the lexical index.
3. **Fusion.** Each list is min-max normalised on its own, then blended with
   ``retrieval.hybrid.vector_weight`` / ``bm25_weight``. Not RRF: the manifest
   specifies weights, and RRF would ignore them. Chunks are then folded to their
   note, keeping each note's best chunk.
4. **Filters** ``{type, agent, profile, scope, tags, status, min_confidence}``.
   The equality/floor filters are pushed *into* both indexes so ``k`` counts
   matching notes; ``tags`` (conjunctive) and ``status`` are finished here. The
   default excludes ``arquivado`` and ``deprecado`` unless ``status`` is named
   explicitly -- archived memories lose weight but stay reachable with an
   explicit filter (§9.3).
5. **Link expansion.** The surviving top-``k`` become seeds; the authored graph
   is walked ``expand_links`` hops (default and ceiling from the manifest, a
   value above the ceiling is a loud error, not a clamp), the reached notes are
   rescored with per-hop decay, and the union is reranked. An expanded hit
   carries its ``via`` chain -- the R6.5 signal that it is context, not local
   truth.
6. **Access.** The returned ids are recorded to the access log (R7.2) for decay.

Why re-read the notes
=====================

The index yields chunk ids and scores; the authoritative frontmatter and the
raw chunk text for the excerpt come from re-reading the Markdown, which is the
source of truth (P1). At personal-vault scale that is a handful of small files
per query. It also means this module depends only on the *scores* an index
returns and on the filter-metadata contract (:data:`STORE_META_KEYS`), never on
an index carrying display fields it might carry differently.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from .access import AccessKind, AccessLog
from .contract import (
    CONFIDENCE_LEVELS,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_DEPRECATED,
    MemoryType,
)
from .discovery import VaultPaths
from .graph import LinkGraph, ViaChain
from .index.chunk import chunk_note
from .index.embed import from_manifest as embed_from_manifest
from .index.lexical import FILTER_KEYS, LEXICAL_DB_FILENAME, LexicalIndex
from .index.store import from_manifest as store_from_manifest
from .manifest import Manifest
from .note import Note, normalize_tags

__all__ = [
    "Hit",
    "Retriever",
    "search",
    "store_metadata",
    "STORE_META_KEYS",
    "HOP_DECAY",
    "ARCHIVED_PENALTY",
    "EXCERPT_CHARS",
]

# ── Tuning constants ────────────────────────────────────────────────
#: Per-hop score multiplier for a link-expanded note (§7.1 "rescore ... com
#: decaimento por hop"). The manifest fixes how *far* to walk but not how fast
#: relevance falls off; 0.5 keeps a one-hop neighbour firmly below its seed
#: while still ahead of noise. A pure function of hops, so expansion never
#: outranks the direct hit that reached it.
HOP_DECAY: Final[float] = 0.5

#: Weight kept by an ``arquivado``/``deprecado`` note when an explicit ``status``
#: filter lets it back into the results. §9.3: archived memories "lose weight in
#: retrieval but stay searchable with an explicit filter" -- this is that lost
#: weight, applied only to a note that would otherwise be excluded by default.
ARCHIVED_PENALTY: Final[float] = 0.5

#: Target length of a hit's excerpt, in characters of the raw chunk (R7.1).
EXCERPT_CHARS: Final[int] = 240

#: Chunks fetched from each index before fusion and filtering. Generous, because
#: the ``status``/``tags`` post-filter and the chunk->note fold both shrink the
#: pool, and expansion needs real seeds to walk from. Bounded work at personal
#: scale: a few dozen dot products and one small SQL query.
_POOL_MULTIPLIER: Final[int] = 5
_MIN_POOL: Final[int] = 40

#: Statuses excluded from retrieval unless the caller names ``status`` (§9.3).
_EXCLUDED_BY_DEFAULT: Final[frozenset[str]] = frozenset(
    {STATUS_ARCHIVED, STATUS_DEPRECATED}
)

#: The known ``status:`` values, for validating a ``status`` filter (R5.1).
_KNOWN_STATUSES: Final[frozenset[str]] = frozenset(
    {STATUS_ACTIVE, STATUS_ARCHIVED, STATUS_DEPRECATED}
)

#: Filter keys pushed down to the vector store and BM25 index as equality/floor
#: constraints, so ``k`` counts matching notes rather than being starved by a
#: non-matching neighbour. ``tags`` and ``status`` are finished in this module.
_PUSHED_KEYS: Final[tuple[str, ...]] = ("type", "agent", "profile", "scope")

#: Metadata keys the vector store must carry per chunk for the filters above to
#: work. The field names are §7.1's own, and match
#: :class:`~mechabrain.index.lexical.LexicalChunk`; a reindexer stores exactly
#: these so that ``search`` and the index agree on what a filter constrains.
STORE_META_KEYS: Final[tuple[str, ...]] = (
    "note_id",
    "type",
    "agent",
    "profile",
    "scope",
    "status",
    "confidence",
    "tags",
)

_TERM_RE: Final[re.Pattern[str]] = re.compile(r"[^\W_]+", re.UNICODE)


# ══════════════════════════════════════════════════════════════════════
# Hit
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieval result, carrying its own provenance (§7.1, R7.1).

    ``path`` and ``wikilink`` are always present so the agent can cite the
    source in its answer (P7). ``path`` is vault-root-relative (R4.2), never
    absolute. ``via`` is set only for a note reached by link expansion; it is
    the chain ``[[seed]] → [[reached]]`` that marks the hit as context rather
    than local truth (R6.5).

    Attributes:
        id: Note id (basename without ``.md``) -- what a wikilink resolves to.
        path: Vault-relative POSIX path to the note.
        wikilink: ``[[id]]`` citation form.
        title: Frontmatter ``title:``, or the id.
        type: Memory type (``semantic`` ...), or ``None`` for a read-only human
            note indexed as context.
        agent: Frontmatter ``agent:`` -- the author runtime (P7).
        confidence: Frontmatter ``confidence:``.
        score: Final rank score: the fused hybrid score, decayed by hop and
            penalised for an archived note. Comparable only within one result
            list -- fusion min-max normalises per query.
        excerpt: A query-relevant window of the matched chunk's *raw* text --
            never the context-prefixed ``embed_text`` (§7.1).
        created: Frontmatter ``created:``, ISO-8601.
        scope: Frontmatter ``scope:`` -- the cross-contamination boundary (R6.5).
        profile: Frontmatter ``profile:`` -- the author persona, if any (R6.6).
        via: Provenance chain for an expanded hit, else ``None``.
        similarity: Raw cosine of the best matched chunk, in ``[-1, 1]``. Not
            part of the §7.1 hit shape; exposed because the write gate's dedup
            check calibrates against ``maintenance.dedup_similarity`` and a fused
            score cannot (see :class:`mechabrain.gate.SearchFn`). ``0.0`` for a
            note reached only by expansion.
        hops: Edges walked from a seed. ``0`` for a direct hit.
    """

    id: str
    path: str
    wikilink: str
    title: str
    type: str | None
    agent: str | None
    confidence: str | None
    score: float
    excerpt: str
    created: str | None
    scope: str | None = None
    profile: str | None = None
    via: str | None = None
    similarity: float = 0.0
    hops: int = 0

    def as_dict(self) -> dict[str, Any]:
        """The §7.1 wire shape, for an MCP response.

        Includes the two documented extensions (``via``, ``similarity``,
        ``hops``); a consumer wanting only the strict R7.1 keys ignores them.
        """
        return {
            "id": self.id,
            "path": self.path,
            "wikilink": self.wikilink,
            "title": self.title,
            "type": self.type,
            "agent": self.agent,
            "confidence": self.confidence,
            "score": self.score,
            "excerpt": self.excerpt,
            "created": self.created,
            "scope": self.scope,
            "profile": self.profile,
            "via": self.via,
            "similarity": self.similarity,
            "hops": self.hops,
        }


# ══════════════════════════════════════════════════════════════════════
# Internal candidate
# ══════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class _Candidate:
    """A note in flight through the pipeline, before it becomes a :class:`Hit`."""

    note_id: str
    note: Note
    score: float
    cosine: float
    best_chunk_id: str | None
    via: ViaChain | None = None


# ══════════════════════════════════════════════════════════════════════
# Retriever
# ══════════════════════════════════════════════════════════════════════
class Retriever:
    """memory.search over one vault (§7.1).

    Opens the vector store and BM25 index (lazily, on first query) and builds
    the authored graph once at construction. A long-lived caller -- the `serve`
    daemon (R7.4) -- rebuilds after a write changes the corpus; a one-shot
    caller uses :func:`search`.

    Not thread-safe: the lexical index holds one SQLite connection. Give each
    thread its own :class:`Retriever`.

    Args:
        paths: The vault, from :func:`mechabrain.discovery.discover_vault`.
        manifest: Its parsed manifest -- the source of the fusion weights, the
            hop ceiling and the store/embedding choice (P6).
        graph: A prebuilt :class:`~mechabrain.graph.LinkGraph`, to avoid a second
            scan when the caller already has one. Defaults to building it.
    """

    __slots__ = ("_paths", "_manifest", "_provider", "_store", "_lexical", "_graph", "_access")

    def __init__(
        self,
        paths: VaultPaths,
        manifest: Manifest,
        *,
        graph: LinkGraph | None = None,
    ) -> None:
        self._paths = paths
        self._manifest = manifest
        self._provider = embed_from_manifest(manifest)
        self._store = store_from_manifest(manifest, paths)
        self._lexical = LexicalIndex(paths.index_dir / LEXICAL_DB_FILENAME)
        self._graph = graph if graph is not None else LinkGraph.build(paths, manifest)
        self._access = AccessLog.for_vault(paths)

    # ── Lifecycle ───────────────────────────────────────────────────
    def close(self) -> None:
        """Release the lexical index's connection. Idempotent."""
        self._lexical.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ── Search (§7.1) ────────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
        expand_links: int | None = None,
    ) -> list[Hit]:
        """Hybrid search with filters and link expansion (§7.1).

        Args:
            query: Natural-language query. Embedded raw for the vector half and
                tokenised for BM25. A blank query returns ``[]``.
            k: Maximum hits. Must be >= 1.
            filters: Any of ``type``, ``agent``, ``profile``, ``scope``, ``tags``,
                ``status``, ``min_confidence``. Scalars take one value or a list
                (OR within a key); ``tags`` requires *all* listed tags;
                ``min_confidence`` is a floor on
                :data:`~mechabrain.contract.CONFIDENCE_LEVELS`. Omitting ``status``
                excludes ``arquivado`` and ``deprecado`` (§9.3).
            expand_links: Hops to walk the authored graph from the top hits.
                Defaults to ``retrieval.link_expansion.default_hops``; a value
                above ``max_hops`` is an error, not a clamp (R5.1).

        Returns:
            Up to ``k`` :class:`Hit`\\ s, best score first, ties broken by id so
            the order is total and reproducible.

        Raises:
            ValueError: ``k`` < 1, an unknown or malformed filter, or
                ``expand_links`` negative or above ``max_hops``.
            mechabrain.errors.MechabrainIndexError: the index is unreadable.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        clean_filters = _validate_filters(filters)
        if not query.strip():
            return []

        explicit_status = _explicit_status(clean_filters)
        required_tags = normalize_tags(clean_filters.get("tags"))

        fused, cosines = self._hybrid_scores(query, k, clean_filters)
        note_score, note_best, note_cos = _fold_chunks_to_notes(fused, cosines)

        seeds = self._seed_candidates(
            note_score, note_best, note_cos, explicit_status, required_tags
        )
        seed_top = seeds[:k]

        results = self._expand(seed_top, seeds, expand_links)
        final = sorted(results.values(), key=lambda c: (-c.score, c.note_id))[:k]

        hits = [self._to_hit(candidate, query) for candidate in final]
        self._access.record([hit.id for hit in hits], AccessKind.SEARCH)
        return hits

    # ── Steps ────────────────────────────────────────────────────────
    def _hybrid_scores(
        self, query: str, k: int, filters: Mapping[str, Any]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Run both indexes, min-max normalise each, blend by the manifest weights.

        Returns ``(fused_by_chunk, cosine_by_chunk)``. The cosine is kept raw and
        unfused so a hit can expose it (:attr:`Hit.similarity`).
        """
        pool = max(k * _POOL_MULTIPLIER, _MIN_POOL)
        query_vector = self._provider.embed_texts([query])[0]

        # A cosine <= 0 is orthogonality, not a match: the chunk shares no
        # feature with the query. Dropping it keeps min-max from stretching pure
        # noise up to a real-looking normalised score, which at a small corpus
        # (few chunks, large pool) is the difference between a hit and a non-hit.
        vector_hits = {
            chunk_id: cosine
            for chunk_id, cosine in self._store.search(
                query_vector, k=pool, filters=_store_filters(filters)
            )
            if cosine > 0.0
        }
        bm25_hits = dict(
            self._lexical.search(query, k=pool, filters=_lexical_filters(filters))
        )

        normalized_vector = _min_max(vector_hits)
        normalized_bm25 = _min_max(bm25_hits)
        weights = self._manifest.retrieval.hybrid

        fused = {
            chunk_id: weights.vector_weight * normalized_vector.get(chunk_id, 0.0)
            + weights.bm25_weight * normalized_bm25.get(chunk_id, 0.0)
            for chunk_id in vector_hits.keys() | bm25_hits.keys()
        }
        return fused, vector_hits

    def _seed_candidates(
        self,
        note_score: Mapping[str, float],
        note_best: Mapping[str, str],
        note_cos: Mapping[str, float],
        explicit_status: frozenset[str] | None,
        required_tags: Sequence[str],
    ) -> list[_Candidate]:
        """Turn scored notes into filtered, ranked seeds.

        Applies the two filters that are not pushed into the index -- conjunctive
        ``tags`` and the §9.3 ``status`` rule -- and penalises an archived note
        that an explicit filter let back in (:data:`ARCHIVED_PENALTY`).
        """
        candidates: list[_Candidate] = []
        for note_id, score in note_score.items():
            # min-max floors the weakest member of each list to 0; a note that
            # scores 0 in both halves contributed to neither ranking and is not
            # a hit. This never drops a top-k result -- the pool is far larger.
            if score <= 0.0:
                continue
            note = self._load(note_id)
            if note is None:
                continue
            if not _passes_tags(note, required_tags):
                continue
            if not _passes_status(note, explicit_status):
                continue
            final_score = score
            if _status_of(note) in _EXCLUDED_BY_DEFAULT:
                final_score *= ARCHIVED_PENALTY
            candidates.append(
                _Candidate(
                    note_id=note_id,
                    note=note,
                    score=final_score,
                    cosine=note_cos.get(note_id, 0.0),
                    best_chunk_id=note_best.get(note_id),
                )
            )
        candidates.sort(key=lambda c: (-c.score, c.note_id))
        return candidates

    def _expand(
        self,
        seed_top: Sequence[_Candidate],
        all_seeds: Sequence[_Candidate],
        expand_links: int | None,
    ) -> dict[str, _Candidate]:
        """Walk the authored graph from ``seed_top`` and rescore what it reaches.

        ``graph.expand`` validates ``expand_links`` (negative or above the
        manifest ceiling raises) and returns the seeds at ``hops == 0`` plus the
        reachable set. A note reached by expansion is scored from its seed with
        per-hop decay and carries its ``via`` chain. Expansion deliberately does
        *not* re-apply the content filters -- reaching a cross-scope note by an
        authored link is the point, and ``via`` is the R6.5 signal that it is
        context. It does honour the default ``status`` exclusion: expansion must
        not resurrect an archived memory.
        """
        results: dict[str, _Candidate] = {c.note_id: c for c in seed_top}
        seed_score = {c.note_id: c.score for c in seed_top}
        direct_score = {c.note_id: c.score for c in all_seeds}

        reached = self._graph.expand([c.note_id for c in seed_top], expand_links)
        for note_id, via in reached.items():
            if via.hops == 0 or note_id in results:
                continue
            note = self._load(note_id)
            if note is None or not _passes_status(note, None):
                continue
            decayed = seed_score.get(via.seed, 0.0) * (HOP_DECAY**via.hops)
            score = max(decayed, direct_score.get(note_id, 0.0))
            results[note_id] = _Candidate(
                note_id=note_id,
                note=note,
                score=score,
                cosine=0.0,
                best_chunk_id=None,
                via=via,
            )
        return results

    def _to_hit(self, candidate: _Candidate, query: str) -> Hit:
        """Build the provenance-bearing hit from a resolved candidate (R7.1)."""
        note = candidate.note
        excerpt = _excerpt(self._raw_text(candidate), query)
        path = self._paths.relative(note.path) if note.path is not None else ""
        return Hit(
            id=candidate.note_id,
            path=path,
            wikilink=note.wikilink,
            title=note.title,
            type=_type_of(note.path, self._paths),
            agent=_str_or_none(note.get("agent")),
            confidence=_str_or_none(note.get("confidence")),
            score=round(candidate.score, 6),
            excerpt=excerpt,
            created=_iso(note.get("created")),
            scope=_str_or_none(note.get("scope")),
            profile=_str_or_none(note.get("profile")),
            via=candidate.via.render() if candidate.via is not None else None,
            similarity=round(candidate.cosine, 6),
            hops=candidate.via.hops if candidate.via is not None else 0,
        )

    def _raw_text(self, candidate: _Candidate) -> str:
        """Raw text of the chunk to excerpt: the matched one, or the note's first.

        Re-chunks the note deterministically (contextual off -- the prefix is
        irrelevant to ``raw_text`` and never shown, §7.1) and selects the ordinal
        the best chunk id names. An expanded note has no matched chunk, so its
        opening chunk stands in.
        """
        chunks = chunk_note(candidate.note, contextual=False)
        if not chunks:
            return ""
        if candidate.best_chunk_id is not None:
            ordinal = _ordinal_of(candidate.best_chunk_id)
            for chunk in chunks:
                if chunk.ordinal == ordinal:
                    return chunk.raw_text
        return chunks[0].raw_text

    def _load(self, note_id: str) -> Note | None:
        """Read a note by id, or ``None`` if the graph no longer knows it.

        A chunk id surviving in the index for a note deleted since the last
        reindex resolves to nothing; skipping it is correct -- the index is
        derived and a stale row is not a reason to fail a query (P1).
        """
        if not self._graph.has_note(note_id):
            return None
        try:
            return Note.load(self._graph.path_of(note_id))
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════
# Module-level convenience
# ══════════════════════════════════════════════════════════════════════
def search(
    paths: VaultPaths,
    manifest: Manifest,
    query: str,
    k: int = 8,
    filters: Mapping[str, Any] | None = None,
    expand_links: int | None = None,
) -> list[Hit]:
    """One-shot :meth:`Retriever.search`, building and closing a retriever.

    For a CLI invocation or a test. A daemon holds a :class:`Retriever` open
    instead, so it does not rebuild the graph per query (R7.4).
    """
    with Retriever(paths, manifest) as retriever:
        return retriever.search(query, k=k, filters=filters, expand_links=expand_links)


# ══════════════════════════════════════════════════════════════════════
# Index metadata contract
# ══════════════════════════════════════════════════════════════════════
def store_metadata(note: Note, memory_type: MemoryType | str) -> dict[str, Any]:
    """The vector-store metadata a chunk of ``note`` must carry (:data:`STORE_META_KEYS`).

    Every chunk of one note shares it. ``type`` is not a frontmatter key -- it
    follows from the note's folder, which the indexer knows -- so it is passed
    in. The field names are §7.1's filter names, so a filter pushed down by
    :func:`search` matches what a reindexer stored.
    """
    return {
        "note_id": note.note_id,
        "type": str(MemoryType.parse(str(memory_type))),
        "agent": _str_or_none(note.get("agent")),
        "profile": _str_or_none(note.get("profile")),
        "scope": _str_or_none(note.get("scope")),
        "status": _status_of(note),
        "confidence": _str_or_none(note.get("confidence")),
        "tags": note.tags,
    }


# ══════════════════════════════════════════════════════════════════════
# Fusion
# ══════════════════════════════════════════════════════════════════════
def _min_max(scores: Mapping[str, float]) -> dict[str, float]:
    """Min-max normalise a score list to ``[0, 1]`` (§7.1 fusion).

    A degenerate list -- one element, or every score equal -- maps to ``1.0``:
    all its members are equally the top of their own ranking, and the fusion
    weight decides how much that half is worth. Returning ``0.0`` instead would
    silently delete one half of the hybrid whenever a query hit a single chunk.
    """
    if not scores:
        return {}
    values = scores.values()
    low, high = min(values), max(values)
    if high == low:
        return {key: 1.0 for key in scores}
    span = high - low
    return {key: (value - low) / span for key, value in scores.items()}


def _fold_chunks_to_notes(
    fused: Mapping[str, float], cosines: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, str], dict[str, float]]:
    """Reduce chunk scores to notes, keeping each note's best chunk.

    A hit is a note, but scoring is per chunk (§7.1). The note's score is its
    strongest chunk's fused score, and that chunk is remembered so the excerpt
    comes from the passage that actually matched. Raw cosine is folded
    separately -- its own max -- because it feeds the dedup gate, not the rank.
    """
    note_score: dict[str, float] = {}
    note_best: dict[str, str] = {}
    note_cos: dict[str, float] = {}
    for chunk_id, score in fused.items():
        note_id = _note_of(chunk_id)
        if note_id not in note_score or score > note_score[note_id]:
            note_score[note_id] = score
            note_best[note_id] = chunk_id
        cosine = cosines.get(chunk_id, 0.0)
        if note_id not in note_cos or cosine > note_cos[note_id]:
            note_cos[note_id] = cosine
    return note_score, note_best, note_cos


# ══════════════════════════════════════════════════════════════════════
# Filters
# ══════════════════════════════════════════════════════════════════════
def _validate_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reject an unknown key or malformed value before it silently widens a scope.

    A dropped ``scope`` typo would search every project and break R6.5, so a bad
    filter is an error, not a no-op (R5.1). Mirrors the lexical index's own
    validation so both halves agree on the language.

    Raises:
        ValueError: unknown filter key, ``min_confidence`` outside
            :data:`~mechabrain.contract.CONFIDENCE_LEVELS`, or ``status`` outside
            the three known values.
    """
    if not filters:
        return {}
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(
            f"unknown search filter(s): {', '.join(sorted(map(str, unknown)))}; "
            f"valid filters are: {', '.join(sorted(FILTER_KEYS))}"
        )
    clean = dict(filters)

    min_confidence = clean.get("min_confidence")
    if min_confidence is not None and str(min_confidence) not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"unknown min_confidence {min_confidence!r}; "
            f"valid levels are: {', '.join(CONFIDENCE_LEVELS)}"
        )

    status = clean.get("status")
    if status is not None:
        for value in _as_list(status):
            if value not in _KNOWN_STATUSES:
                raise ValueError(
                    f"unknown status {value!r}; "
                    f"valid statuses are: {', '.join(sorted(_KNOWN_STATUSES))}"
                )
    return clean


def _explicit_status(filters: Mapping[str, Any]) -> frozenset[str] | None:
    """The status values the caller named, or ``None`` for the §9.3 default."""
    status = filters.get("status")
    if status is None:
        return None
    return frozenset(_as_list(status))


def _store_filters(filters: Mapping[str, Any]) -> dict[str, Any] | None:
    """Lower §7.1 filters into the generic key/value language of the vector store.

    Pushes the equality filters, plus ``min_confidence`` lowered to a
    ``confidence`` allow-list (the store has no notion of an ordered floor -- see
    its docstring). ``status`` is pushed only when named, to keep an explicitly
    requested status in the candidate pool; the default exclusion is finished in
    Python. ``tags`` is not pushed: the store matches a tag list by intersection
    (OR), while §7.1 ``tags`` is conjunctive, so it is enforced by re-read.
    """
    pushed: dict[str, Any] = {
        key: filters[key] for key in _PUSHED_KEYS if filters.get(key) is not None
    }
    min_confidence = filters.get("min_confidence")
    if min_confidence is not None:
        start = CONFIDENCE_LEVELS.index(str(min_confidence))
        pushed["confidence"] = list(CONFIDENCE_LEVELS[start:])
    if filters.get("status") is not None:
        pushed["status"] = filters["status"]
    return pushed or None


def _lexical_filters(filters: Mapping[str, Any]) -> dict[str, Any] | None:
    """The §7.1 filters the BM25 index applies natively.

    Pushes the equality filters and ``min_confidence`` (the lexical index ranks
    confidence itself). ``status`` only when named; ``tags`` is finished by
    re-read, as for the store, so both halves apply one tag semantics.
    """
    pushed: dict[str, Any] = {
        key: filters[key]
        for key in (*_PUSHED_KEYS, "min_confidence")
        if filters.get(key) is not None
    }
    if filters.get("status") is not None:
        pushed["status"] = filters["status"]
    return pushed or None


def _passes_status(note: Note, explicit: frozenset[str] | None) -> bool:
    """Whether ``note`` survives the §9.3 status rule.

    With an explicit ``status`` filter, membership in it; otherwise, anything not
    ``arquivado``/``deprecado``. A note with no ``status:`` counts as active --
    the absence of the key is not the archived state.
    """
    status = _status_of(note)
    if explicit is not None:
        return status in explicit
    return status not in _EXCLUDED_BY_DEFAULT


def _passes_tags(note: Note, required: Sequence[str]) -> bool:
    """Whether ``note`` carries every tag in ``required`` (conjunctive, §7.1)."""
    if not required:
        return True
    return set(required) <= set(note.tags)


# ══════════════════════════════════════════════════════════════════════
# Excerpt
# ══════════════════════════════════════════════════════════════════════
def _excerpt(text: str, query: str, width: int = EXCERPT_CHARS) -> str:
    """A ``width``-ish window of ``text`` around the first query term it contains.

    Whitespace is collapsed so the excerpt is one readable line. When no query
    term appears -- a note reached by expansion, say -- the opening stands in.
    Ellipses mark a window that does not reach an end of the text.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    lowered = collapsed.casefold()
    position = -1
    for term in _TERM_RE.findall(query.casefold()):
        found = lowered.find(term)
        if found != -1 and (position == -1 or found < position):
            position = found
    if position == -1:
        head = collapsed[:width]
        return head + ("…" if len(collapsed) > width else "")
    start = max(0, position - width // 3)
    end = min(len(collapsed), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(collapsed) else ""
    return f"{prefix}{collapsed[start:end]}{suffix}"


# ══════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════
def _note_of(chunk_id: str) -> str:
    """The note id a chunk id belongs to -- ``<note_id>#<ordinal>`` split off."""
    return chunk_id.rsplit("#", 1)[0]


def _ordinal_of(chunk_id: str) -> int:
    """The ordinal a chunk id carries, or ``-1`` if it is malformed."""
    _, _, tail = chunk_id.rpartition("#")
    try:
        return int(tail)
    except ValueError:
        return -1


def _type_of(path: Path | None, paths: VaultPaths) -> str | None:
    """The memory type a note's folder implies, or ``None`` for a human note.

    Derived from the path shape rather than a frontmatter key: ``type`` is a
    function of the §3 folder a note lives in, and a read-only indexed human note
    lives in none of them.
    """
    if path is None:
        return None
    for memory_type in MemoryType:
        folder = paths.folder_for(memory_type)
        if path == folder or path.is_relative_to(folder):
            return str(memory_type)
    return None


def _status_of(note: Note) -> str:
    """Frontmatter ``status:``, defaulting to active when absent (§6)."""
    status = _str_or_none(note.get("status"))
    return status if status is not None else STATUS_ACTIVE


def _as_list(value: Any) -> list[str]:
    """A filter value as a list of strings; a bare string is one value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _str_or_none(value: Any) -> str | None:
    """A frontmatter scalar as trimmed text, or ``None`` when blank/absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso(value: Any) -> str | None:
    """A date-like frontmatter value as an ISO string, else its text."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
