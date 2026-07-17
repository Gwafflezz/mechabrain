"""Orchestration of the derived index: notes -> chunks -> embeddings + BM25.

This is the conductor of the retrieval layer. Every other index module does one
job and does not know the vault: :mod:`~mechabrain.index.chunk` splits a note,
:mod:`~mechabrain.index.embed` turns text into vectors,
:mod:`~mechabrain.index.store` holds those vectors, and
:mod:`~mechabrain.index.lexical` holds their BM25 terms. This module reads the
vault, drives them in order, and keeps the two indexes and the on-disk source of
truth agreeing with each other.

What is indexed
===============

Two corpora, one index:

* the four **memory folders** (§3), minus any type a deployment switched off
  (``zones.research_enabled``); and
* every folder listed in ``zones.read_only_index`` -- human notes indexed as
  *read-only context*. They are searchable, but they are **never** a target of a
  write or of decay (§9.3). The kernel marks them mechanically: each carries a
  ``read_only`` flag in the index state, and :meth:`Indexer.read_only_note_ids`
  hands the write/consolidation paths the exact set to leave alone. The
  protection is also structural -- ``memory_write`` and ``consolidate`` only ever
  touch the memory folders -- but the flag makes it explicit and queryable.

Note identity, and collisions
=============================

A chunk's key is ``<note_id>#<ordinal>`` (:attr:`~mechabrain.index.chunk.Chunk.chunk_id`),
and ``note_id`` is a basename (:func:`~mechabrain.note.note_id_for`). Two notes
in different folders can therefore claim one id -- likely once a human folder
tree is under ``read_only_index``. Exactly as a Markdown editor resolves a
wikilink, the first note in sorted path order wins the id and is indexed; the
rest are reported as :attr:`IndexReport.ambiguous_ids` and left out, so the
index and the authored graph (:class:`~mechabrain.graph.LinkGraph`) agree on
which note owns the id.

Incremental by mtime + content hash
===================================

The index is derived state (P1): gitignored, per machine, always rebuildable. A
small state manifest at ``_meta/index/`` records, per indexed note, its
vault-relative path, its ``st_mtime_ns``, a SHA-256 of its bytes, its chunk
count and its ``read_only`` flag. :meth:`Indexer.reindex` with ``full=False``
diffs the vault against it:

* an unchanged ``mtime`` skips the note untouched (the fast path);
* a changed ``mtime`` but an unchanged hash refreshes the stamp and skips the
  re-embed -- a touch is not an edit;
* a changed hash re-chunks and re-embeds the note, dropping its old chunks
  first (the chunk count may have shrunk);
* a note whose file is gone has its chunks deleted from both indexes.

``full=True`` clears both indexes and rebuilds from scratch. A **full rebuild is
forced** even when ``full=False`` if the state's fingerprint no longer matches
the deployment: a different embedding model (whose vectors share no space with
the old ones), a flipped ``contextual_retrieval``, a different vector store, or a
changed corpus definition (``research_enabled`` / ``read_only_index``). Silently
serving a stale index across any of those returns plausible nonsense (R5.1), so
the fingerprint check is the guard.

The chunk count per note is what lets a single-note update delete exactly the
old chunks from the vector store, which -- unlike the lexical index -- deletes by
chunk id, not by note id.

What lands in the index metadata
================================

The vector store returns only ``(chunk_id, score)`` and the lexical index only
``(chunk_id, bm25)``; neither hands back a note's display fields. So this module
stores in each backend **only what those backends filter on** (§7.1): ``type``,
``agent``, ``profile``, ``scope``, ``status``, ``confidence`` and ``tags``.
Provenance and the excerpt (R7.1) are not duplicated here -- the Markdown is the
source of truth (P1), and the retrieval layer re-derives them from the note the
``chunk_id`` names. ``status`` defaults to ``ativo`` when absent (a note with no
status is not archived); every other field is stored verbatim and omitted when
the note does not carry it, so a filter on an absent field excludes the note --
the safe direction (R6.5).

The context prefix a chunk is embedded behind is the chunker's deterministic
``embed_text`` (scope + title + tags + heading path); both halves index that same
string, so a term in the prefix is findable on either side. That determinism --
no LLM at ingestion -- is the §7.1/§13 choice documented on
:mod:`~mechabrain.index.chunk`.

Concurrency
===========

Every operation that writes the index takes the one index lock (R7.4 fallback),
``_meta/index/index.lock``, for the whole read-modify-write cycle -- the store
takes no lock of its own by design, and the state manifest, both indexes and the
lock must move together. The lock is the same file the daemon and every other
index writer agree on (:data:`~mechabrain.index.store.INDEX_LOCK_FILE`). The
state manifest is written **last** and atomically (R7.5): a crash mid-rebuild
leaves an index that a later run re-derives, never a state that lies about it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from ..contract import MARKDOWN_SUFFIX, STATUS_ACTIVE, MemoryType, type_for_folder
from ..discovery import VaultPaths
from ..errors import NoteNotFound
from ..locking import DEFAULT_LOCK_TIMEOUT, file_lock
from ..manifest import Manifest
from ..note import Note, note_id_for, write_atomic
from .chunk import Chunk, chunk_note
from .embed import EmbeddingProvider
from .embed import from_manifest as embed_from_manifest
from .lexical import LEXICAL_DB_FILENAME, LexicalChunk, LexicalIndex
from .store import INDEX_LOCK_FILE, NumpyStore, VectorStore
from .store import from_manifest as store_from_manifest

__all__ = [
    "Indexer",
    "IndexReport",
    "Progress",
    "STATE_FILE",
    "STATE_VERSION",
]

#: State manifest under ``_meta/index/``: the per-note record an incremental
#: reindex diffs against. Derived state -- gitignored, per machine, rebuildable.
STATE_FILE: Final[str] = "indexer_state.json"

#: Bumped when the state manifest's shape changes. An older or unreadable state
#: is treated as absent -- the next reindex is a full rebuild, which is cheaper
#: and safer than migrating derived state (P1).
STATE_VERSION: Final[int] = 1

#: Chunks buffered before one embed + upsert flush. Bounds peak memory and the
#: size of a single embedding request without changing results -- upserts are
#: ordered and idempotent by chunk id.
_BATCH_CHUNKS: Final[int] = 256

#: A progress sink: called with one human-readable line per note or phase. The
#: kernel never interprets these; they are for a CLI to print.
Progress = Callable[[str], None]


def _noop(_message: str) -> None:
    """The default progress sink: discard."""


# ══════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class IndexReport:
    """The outcome of an index operation -- counts, not side effects.

    Attributes:
        full: Whether this was a from-scratch rebuild. True either because the
            caller asked (``reindex(full=True)``) or because the fingerprint
            forced it -- see :attr:`reason`.
        reason: Why a full rebuild was forced on a ``full=False`` call, or
            ``None`` when the mode was the one requested.
        notes_indexed: Notes freshly chunked and embedded (new or changed).
        notes_removed: Notes whose chunks were dropped (file gone).
        notes_unchanged: Notes the incremental pass skipped as up to date.
        chunks_written: Chunks upserted into both indexes.
        chunks_deleted: Chunks removed from both indexes.
        notes_total: Notes tracked after the operation.
        chunks_total: Chunks in the vector store after the operation.
        read_only_indexed: Of the notes (re)indexed, how many are read-only
            context (from ``zones.read_only_index``).
        ambiguous_ids: Note ids claimed by more than one file; only the first in
            path order was indexed (see the module docstring).
    """

    full: bool
    notes_indexed: int
    notes_removed: int
    notes_unchanged: int
    chunks_written: int
    chunks_deleted: int
    notes_total: int
    chunks_total: int
    read_only_indexed: int = 0
    ambiguous_ids: tuple[str, ...] = ()
    reason: str | None = None


# ══════════════════════════════════════════════════════════════════════
# State manifest
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class _NoteRecord:
    """One indexed note, as the state manifest remembers it."""

    note_id: str
    mtime_ns: int
    hash: str
    chunks: int
    read_only: bool


@dataclass(slots=True)
class _IndexState:
    """The state manifest: a fingerprint plus one record per indexed note.

    Keyed by vault-relative POSIX path, which is unique per file -- ``note_id``
    is not (basenames collide). Mutable: :meth:`Indexer.index_note` folds one
    note into the existing map without rewriting the rest.
    """

    fingerprint: dict[str, Any]
    notes: dict[str, _NoteRecord] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "fingerprint": self.fingerprint,
            "notes": {rel: asdict(rec) for rel, rec in self.notes.items()},
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def load(cls, path: Path) -> "_IndexState | None":
        """Read the state at ``path``, or ``None`` if absent, torn, or stale.

        Every failure resolves to ``None`` -- the caller treats a missing state
        as a signal to rebuild fully, which is always correct for derived state.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return None
        try:
            data = json.loads(raw)
            fingerprint = data["fingerprint"]
            notes = {
                str(rel): _NoteRecord(**record)
                for rel, record in data["notes"].items()
            }
        except (ValueError, KeyError, TypeError):
            return None
        if not isinstance(fingerprint, dict) or fingerprint.get("version") != STATE_VERSION:
            return None
        return cls(fingerprint=fingerprint, notes=notes)


# ══════════════════════════════════════════════════════════════════════
# Scan target
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class _Target:
    """A note file the index should hold, before it is read or chunked."""

    path: Path
    relative: str
    note_id: str
    read_only: bool
    mtime_ns: int


# ══════════════════════════════════════════════════════════════════════
# Indexer
# ══════════════════════════════════════════════════════════════════════
class Indexer:
    """Builds and maintains the derived index of one vault.

    ::

        indexer = Indexer(paths, manifest)
        indexer.reindex(full=True)            # `mechabrain reindex --full`
        indexer.reindex()                     # incremental
        indexer.index_note(note_path)         # after memory_write
        indexer.remove_note(note_id)          # a note was deleted

    One instance is cheap and holds no open handles between calls: each
    operation opens the store and the lexical index, does its work under the
    index lock, and closes them. Not thread-safe; cross-process safety is the
    index lock plus the R7.4 single-writer daemon.

    Args:
        paths: The vault, from :func:`~mechabrain.discovery.discover_vault`.
        manifest: Its parsed manifest -- the source of the embedding backend,
            the vector store, the read-only folders and the enabled types.
        provider: An embedding provider to reuse. Defaults to the one
            ``retrieval.embedding`` names, built lazily on first embed so an
            empty rebuild never loads a model.
        progress: Default progress sink for every operation. A per-call
            ``progress`` argument overrides it.
        lock_timeout: Seconds to wait for the index lock before failing (R7.4).
    """

    __slots__ = (
        "paths",
        "manifest",
        "_progress",
        "_lock_timeout",
        "_provider",
        "_state_path",
    )

    def __init__(
        self,
        paths: VaultPaths,
        manifest: Manifest,
        *,
        provider: EmbeddingProvider | None = None,
        progress: Progress | None = None,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    ) -> None:
        self.paths = paths
        self.manifest = manifest
        self._progress = progress or _noop
        self._lock_timeout = lock_timeout
        self._provider = provider
        self._state_path = paths.index_dir / STATE_FILE

    def __repr__(self) -> str:
        return f"Indexer({str(self.paths.root)!r}, store={self.manifest.retrieval.store!r})"

    # ── Public operations ───────────────────────────────────────────
    def reindex(self, *, full: bool = False, progress: Progress | None = None) -> IndexReport:
        """Bring the index up to date with the vault.

        ``full=False`` diffs the vault against the state manifest and touches
        only what changed; ``full=True`` clears both indexes and rebuilds. A
        ``full=False`` call is upgraded to a full rebuild when the state is
        missing or its fingerprint no longer matches the deployment (see the
        module docstring) -- the returned :attr:`IndexReport.reason` says why.

        Raises:
            SchemaViolation: a note on the way has malformed frontmatter. It is
                raised, not skipped, mirroring
                :func:`~mechabrain.note.iter_notes` (R5.1).
            MechabrainIndexError: the index lock was not free within
                ``lock_timeout`` (R7.4).
        """
        emit = progress or self._progress
        with file_lock(
            self.paths.index_dir / INDEX_LOCK_FILE,
            timeout=self._lock_timeout,
            purpose="reindex --full" if full else "reindex",
        ):
            provider = self._embedder()
            store = self._open_store()
            with self._open_lexical() as lexical:
                state = _IndexState.load(self._state_path)
                reason = self._force_full_reason(state, provider)
                if full or reason is not None:
                    return self._rebuild(store, lexical, provider, emit, reason=reason if not full else None)
                assert state is not None  # a None reason means the state loaded and matched
                return self._incremental(store, lexical, provider, state, emit)

    def index_note(
        self, note: Note | Path | str, *, progress: Progress | None = None
    ) -> IndexReport:
        """(Re)index exactly one note -- the writer's hook after ``memory_write``.

        Drops the note's previous chunks (if any) and inserts the new ones,
        merging one record into the state manifest and leaving every other note
        untouched. If the state's fingerprint no longer matches the deployment,
        a single note cannot be reconciled in isolation, so this falls back to a
        full rebuild.

        Accepts a placed :class:`~mechabrain.note.Note` (``path`` set) or a path,
        so this satisfies the ``index_note`` Protocol
        :func:`mechabrain.writer.write` injects while also serving a path-only
        caller. The note is always re-read from disk -- its current bytes are the
        truth, not whatever a caller happens to hold in memory (P1).

        Args:
            note: The note to index, as a placed ``Note`` or its path. Must exist
                on disk -- use :meth:`remove_note` for a deleted one.

        Raises:
            NoteNotFound: no file at the note's path.
            ValueError: a ``Note`` with no path was given.
            MechabrainIndexError: the index lock was not free (R7.4).
        """
        note_path = self._path_of(note)
        if not note_path.is_file():
            raise NoteNotFound(
                f"cannot index {note_path}: no such file",
                hint="pass an existing note; to drop a deleted note use remove_note()",
            )
        emit = progress or self._progress
        with file_lock(
            self.paths.index_dir / INDEX_LOCK_FILE,
            timeout=self._lock_timeout,
            purpose="index_note",
        ):
            provider = self._embedder()
            store = self._open_store()
            with self._open_lexical() as lexical:
                state = _IndexState.load(self._state_path)
                reason = self._force_full_reason(state, provider)
                if reason is not None:
                    return self._rebuild(store, lexical, provider, emit, reason=reason)

                assert state is not None  # _force_full_reason returns a reason when None
                target = self._target_for(note_path)
                previous = state.notes.get(target.relative)
                deletions = {previous.note_id: previous.chunks} if previous else {}
                deleted = self._delete_chunks(store, lexical, deletions, emit)
                records, written, read_only = self._write_targets(
                    store, lexical, provider, [target], emit
                )
                state.notes.update(records)
                store.flush()
                self._save_state(state)
                return IndexReport(
                    full=False,
                    notes_indexed=1,
                    notes_removed=0,
                    notes_unchanged=len(state.notes) - 1,
                    chunks_written=written,
                    chunks_deleted=deleted,
                    notes_total=len(state.notes),
                    chunks_total=store.count(),
                    read_only_indexed=read_only,
                )

    def remove_note(self, note: Note | Path | str, *, progress: Progress | None = None) -> IndexReport:
        """Drop a note's chunks from both indexes -- a deleted or superseded note.

        Accepts a :class:`~mechabrain.note.Note`, a path, a wikilink or a bare
        id; all reduce to the same ``note_id``. Removing a note that is not
        indexed is not an error -- the requested end state is that it is absent.

        Raises:
            MechabrainIndexError: the index lock was not free (R7.4).
        """
        note_id = self._note_id_of(note)
        emit = progress or self._progress
        with file_lock(
            self.paths.index_dir / INDEX_LOCK_FILE,
            timeout=self._lock_timeout,
            purpose="remove_note",
        ):
            store = self._open_store()
            with self._open_lexical() as lexical:
                state = _IndexState.load(self._state_path)
                paths = (
                    [rel for rel, rec in state.notes.items() if rec.note_id == note_id]
                    if state is not None
                    else []
                )
                deletions = {note_id: state.notes[rel].chunks for rel in paths} if state else {}
                # Even with no state (chunk count unknown) the lexical index can
                # drop the note by id; the vector store cannot, and its orphans
                # wait for the next full rebuild -- honest and rebuildable (P1).
                deleted = self._delete_chunks(store, lexical, deletions, emit, note_id=note_id)
                if state is not None:
                    for rel in paths:
                        del state.notes[rel]
                    store.flush()
                    self._save_state(state)
                return IndexReport(
                    full=False,
                    notes_indexed=0,
                    notes_removed=1 if paths else 0,
                    notes_unchanged=len(state.notes) if state else 0,
                    chunks_written=0,
                    chunks_deleted=deleted,
                    notes_total=len(state.notes) if state else 0,
                    chunks_total=store.count(),
                )

    def read_only_note_ids(self) -> frozenset[str]:
        """Ids of notes indexed as read-only context (``zones.read_only_index``).

        The mechanical marker the write and consolidation paths consult to leave
        human context alone -- it is never a target of a write or of decay (§9.3).
        Empty before the first reindex, or when no read-only folder is set.
        """
        state = _IndexState.load(self._state_path)
        if state is None:
            return frozenset()
        return frozenset(rec.note_id for rec in state.notes.values() if rec.read_only)

    # ── Full rebuild ────────────────────────────────────────────────
    def _rebuild(
        self,
        store: VectorStore,
        lexical: LexicalIndex,
        provider: EmbeddingProvider,
        emit: Progress,
        *,
        reason: str | None,
    ) -> IndexReport:
        """Clear both indexes and index every target note from scratch."""
        emit("full rebuild: clearing the index")
        old_chunks = store.count()
        store.clear()
        lexical.clear()

        targets, ambiguous = self._scan_targets()
        records, written, read_only = self._write_targets(
            store, lexical, provider, targets, emit
        )
        store.flush()
        self._save_state(_IndexState(self._fingerprint(provider), records))
        return IndexReport(
            full=True,
            notes_indexed=len(records),
            notes_removed=0,
            notes_unchanged=0,
            chunks_written=written,
            chunks_deleted=old_chunks,
            notes_total=len(records),
            chunks_total=store.count(),
            read_only_indexed=read_only,
            ambiguous_ids=ambiguous,
            reason=reason,
        )

    # ── Incremental ─────────────────────────────────────────────────
    def _incremental(
        self,
        store: VectorStore,
        lexical: LexicalIndex,
        provider: EmbeddingProvider,
        state: _IndexState,
        emit: Progress,
    ) -> IndexReport:
        """Diff the vault against ``state`` and touch only what changed."""
        current, ambiguous = self._scan_map()

        to_index: list[_Target] = []
        deletions: dict[str, int] = {}  # note_id -> old chunk count to remove
        kept: dict[str, _NoteRecord] = {}
        removed = 0
        unchanged = 0

        for rel, record in state.notes.items():
            if rel not in current:
                deletions[record.note_id] = record.chunks
                removed += 1

        for rel, target in current.items():
            record = state.notes.get(rel)
            if record is None:
                to_index.append(target)
                continue
            if target.mtime_ns == record.mtime_ns:
                kept[rel] = record
                unchanged += 1
                continue
            if _hash_file(target.path) == record.hash:
                # Touched but identical: refresh the stamp, skip the re-embed.
                kept[rel] = replace(record, mtime_ns=target.mtime_ns)
                unchanged += 1
                continue
            deletions[record.note_id] = record.chunks
            to_index.append(target)

        # Deletions first: a note id may move files (its old path vanished while
        # a same-id file became the winner), and deleting before upserting is
        # what makes that handover land the new chunks, not wipe them.
        deleted = self._delete_chunks(store, lexical, deletions, emit)
        records, written, read_only = self._write_targets(
            store, lexical, provider, to_index, emit
        )
        kept.update(records)

        if deleted or written:
            store.flush()
        self._save_state(_IndexState(self._fingerprint(provider), kept))
        return IndexReport(
            full=False,
            notes_indexed=len(to_index),
            notes_removed=removed,
            notes_unchanged=unchanged,
            chunks_written=written,
            chunks_deleted=deleted,
            notes_total=len(kept),
            chunks_total=store.count(),
            read_only_indexed=read_only,
            ambiguous_ids=ambiguous,
        )

    # ── Writing chunks ──────────────────────────────────────────────
    def _write_targets(
        self,
        store: VectorStore,
        lexical: LexicalIndex,
        provider: EmbeddingProvider,
        targets: Iterable[_Target],
        emit: Progress,
    ) -> tuple[dict[str, _NoteRecord], int, int]:
        """Chunk, embed and upsert ``targets``; return their records and counts.

        Buffers up to :data:`_BATCH_CHUNKS` chunks before each embed + upsert, so
        one big vault costs a bounded number of in-flight vectors and one embed
        call per batch rather than per note.
        """
        records: dict[str, _NoteRecord] = {}
        total_written = 0
        read_only_count = 0
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        lexical_chunks: list[LexicalChunk] = []
        contextual = self.manifest.retrieval.contextual_retrieval

        def flush() -> None:
            if not ids:
                return
            store.upsert(ids, provider.embed_texts(texts), metas)
            lexical.upsert(lexical_chunks)
            ids.clear()
            texts.clear()
            metas.clear()
            lexical_chunks.clear()

        ordered = list(targets)
        for position, target in enumerate(ordered, start=1):
            note = Note.load(target.path)
            digest = _hash_file(target.path)
            chunks = chunk_note(note, contextual=contextual)
            memory_type = self._memory_type_of(note)
            meta = self._vector_meta(note, memory_type)
            for chunk in chunks:
                ids.append(chunk.chunk_id)
                texts.append(chunk.embed_text)
                metas.append(meta)
                lexical_chunks.append(self._lexical_chunk(note, chunk, memory_type))

            records[target.relative] = _NoteRecord(
                note_id=target.note_id,
                mtime_ns=target.mtime_ns,
                hash=digest,
                chunks=len(chunks),
                read_only=target.read_only,
            )
            total_written += len(chunks)
            read_only_count += int(target.read_only)
            emit(f"indexed {target.note_id} ({position}/{len(ordered)}, {len(chunks)} chunk(s))")
            if len(ids) >= _BATCH_CHUNKS:
                flush()
        flush()
        return records, total_written, read_only_count

    def _delete_chunks(
        self,
        store: VectorStore,
        lexical: LexicalIndex,
        deletions: dict[str, int],
        emit: Progress,
        *,
        note_id: str | None = None,
    ) -> int:
        """Remove chunks for ``deletions`` (``note_id -> old chunk count``).

        The vector store deletes by chunk id, reconstructed as
        ``<note_id>#<0..count-1>``; the lexical index deletes by note id.
        ``note_id`` forces a lexical delete for that id even when the count is
        unknown (a :meth:`remove_note` with no state), which the vector store
        cannot mirror without the count.
        """
        note_ids = set(deletions)
        if note_id is not None:
            note_ids.add(note_id)
        if not note_ids:
            return 0
        chunk_ids = [
            f"{nid}#{ordinal}"
            for nid, count in deletions.items()
            for ordinal in range(count)
        ]
        if chunk_ids:
            store.delete(chunk_ids)
        lexical.delete(note_ids)
        emit(f"removed {len(note_ids)} note(s), {len(chunk_ids)} chunk(s)")
        return len(chunk_ids)

    # ── Scanning the vault ──────────────────────────────────────────
    def _scan_targets(self) -> tuple[list[_Target], tuple[str, ...]]:
        targets, ambiguous = self._scan_map()
        return list(targets.values()), ambiguous

    def _scan_map(self) -> tuple[dict[str, _Target], tuple[str, ...]]:
        """Every note the index should hold, keyed by vault-relative path.

        Collects the enabled memory folders (as writable memory) then the
        read-only folders (as context), de-duplicating by path so a read-only
        entry pointing back into ``mecha-brain/`` cannot double-count. Then it
        resolves a basename collision the way a Markdown editor does -- first in
        sorted path order wins the id -- and reports the losers.
        """
        sources: dict[Path, bool] = {}  # path -> read_only
        for memory_type in MemoryType:
            if not self.manifest.is_enabled(memory_type):
                continue
            for path in _note_files(self.paths.folder_for(memory_type)):
                sources.setdefault(path, False)
        for folder in self.manifest.zones.read_only_index:
            for path in _note_files(self.paths.resolve(folder)):
                sources.setdefault(path, True)

        winners: dict[str, _Target] = {}
        owner: dict[str, str] = {}  # note_id -> winning relative path
        ambiguous: list[str] = []
        for path in sorted(sources):
            note_id = note_id_for(path)
            if note_id in owner:
                if note_id not in ambiguous:
                    ambiguous.append(note_id)
                continue
            relative = self.paths.relative(path)
            owner[note_id] = relative
            winners[relative] = _Target(
                path=path,
                relative=relative,
                note_id=note_id,
                read_only=sources[path],
                mtime_ns=path.stat().st_mtime_ns,
            )
        return winners, tuple(sorted(ambiguous))

    def _target_for(self, path: Path) -> _Target:
        """A single note's :class:`_Target`, deciding its read-only status."""
        return _Target(
            path=path,
            relative=self.paths.relative(path),
            note_id=note_id_for(path),
            read_only=self._is_read_only(path),
            mtime_ns=path.stat().st_mtime_ns,
        )

    def _is_read_only(self, path: Path) -> bool:
        """Whether ``path`` lives under a ``zones.read_only_index`` folder."""
        resolved = path.resolve()
        for folder in self.manifest.zones.read_only_index:
            base = self.paths.resolve(folder).resolve()
            if resolved == base or base in resolved.parents:
                return True
        return False

    # ── Metadata ────────────────────────────────────────────────────
    def _vector_meta(self, note: Note, memory_type: MemoryType | None) -> dict[str, Any]:
        """The filter metadata one note's chunks carry in the vector store (§7.1).

        Only the fields ``matches_filters`` compares on -- provenance is not
        duplicated here (see the module docstring). Absent optional fields are
        omitted so a filter on them excludes the note, the conservative reading
        of a boundary (R6.5).
        """
        meta: dict[str, Any] = {"status": _status_of(note), "tags": list(note.tags)}
        if memory_type is not None:
            meta["type"] = memory_type.value
        for key in ("scope", "agent", "profile", "confidence"):
            value = _clean(note.get(key))
            if value:
                meta[key] = value
        return meta

    def _lexical_chunk(
        self, note: Note, chunk: Chunk, memory_type: MemoryType | None
    ) -> LexicalChunk:
        """One chunk as the BM25 index's flat, stringly-typed input row."""
        return LexicalChunk(
            chunk_id=chunk.chunk_id,
            note_id=note.note_id,
            text=chunk.embed_text,
            memory_type=memory_type.value if memory_type is not None else None,
            agent=_clean(note.get("agent")) or None,
            profile=_clean(note.get("profile")) or None,
            scope=_clean(note.get("scope")) or None,
            status=_status_of(note),
            confidence=_clean(note.get("confidence")) or None,
            tags=tuple(note.tags),
        )

    def _memory_type_of(self, note: Note) -> MemoryType | None:
        """The note's memory type, from its ``mem/<type>`` tag then its folder.

        Tags first because they are the manifest's own vocabulary and follow a
        moved note; the folder is the fallback. ``None`` for a read-only human
        note that is neither tagged nor in a memory folder -- it then matches no
        ``type`` filter, which is correct: it is context, not a typed memory.

        Mirrors the routing in :mod:`~mechabrain.generate`, which owns the same
        derivation for the maps; both read only public APIs.
        """
        namespaces = self.manifest.frontmatter.tag_namespaces
        for memory_type in MemoryType:
            if note.has_tag(namespaces.memory_tag(memory_type)):
                return memory_type
        if note.path is None:
            return None
        for folder in list(note.path.parents)[:2]:
            try:
                return type_for_folder(folder.name)
            except KeyError:
                continue
        return None

    # ── Fingerprint & backends ──────────────────────────────────────
    def _fingerprint(self, provider: EmbeddingProvider) -> dict[str, Any]:
        """What must match for an incremental reindex to be sound.

        A change to any of these makes the existing index unusable as-is: the
        embedding model changes the vector space, ``contextual_retrieval``
        changes what text is embedded, the store changes where chunks live, and
        the corpus definition changes which notes belong at all.
        """
        retrieval = self.manifest.retrieval
        return {
            "version": STATE_VERSION,
            "embedding": provider.name,
            "contextual": retrieval.contextual_retrieval,
            "store": retrieval.store,
            "research_enabled": self.manifest.zones.research_enabled,
            "read_only_index": sorted(self.manifest.zones.read_only_index),
        }

    def _force_full_reason(
        self, state: _IndexState | None, provider: EmbeddingProvider
    ) -> str | None:
        """Why an incremental pass cannot be trusted, or ``None`` if it can."""
        if state is None:
            return "no index state manifest was found"
        if state.fingerprint != self._fingerprint(provider):
            return (
                "the index fingerprint changed (embedding model, contextual "
                "retrieval, store, or indexed corpus)"
            )
        return None

    def _embedder(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = embed_from_manifest(self.manifest)
        return self._provider

    def _open_store(self) -> VectorStore:
        """The manifest's vector store, in bulk-write mode for the numpy default.

        ``NumpyStore`` holds the whole matrix in one file, so persisting per
        upsert would turn a rebuild into O(n) full-file writes; bulk callers
        disable autosave and :meth:`flush` once. The optional stores commit
        eagerly and take no such flag, so the switch is only meaningful -- and
        only applied -- for the default.
        """
        store = store_from_manifest(self.manifest, self.paths)
        if isinstance(store, NumpyStore):
            store.autosave = False
        return store

    def _open_lexical(self) -> LexicalIndex:
        return LexicalIndex(self.paths.index_dir / LEXICAL_DB_FILENAME)

    def _save_state(self, state: _IndexState) -> None:
        write_atomic(self._state_path, state.to_json())

    def _note_id_of(self, note: Note | Path | str) -> str:
        if isinstance(note, Note):
            if note.path is None:
                raise ValueError("cannot remove a note with no path from the index")
            return note.note_id
        return note_id_for(_strip_wikilink(str(note)))

    @staticmethod
    def _path_of(note: Note | Path | str) -> Path:
        if isinstance(note, Note):
            if note.path is None:
                raise ValueError("cannot index a Note with no path; write it first")
            return note.path
        return Path(note)


# ══════════════════════════════════════════════════════════════════════
# Module helpers
# ══════════════════════════════════════════════════════════════════════
def _note_files(folder: Path) -> Iterator[Path]:
    """Every ``.md`` file under ``folder``, sorted, dotfiles skipped.

    The path-only twin of :func:`~mechabrain.note.iter_notes`: an incremental
    reindex needs mtimes for thousands of notes and must not parse the ones it
    will skip. A missing folder yields nothing -- ``Research/`` legitimately does
    not exist when disabled, and a ``read_only_index`` entry may be stale.
    """
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("**/*")):
        if path.is_file() and path.suffix == MARKDOWN_SUFFIX and not path.name.startswith("."):
            yield path


def _hash_file(path: Path) -> str:
    """SHA-256 of a note's bytes -- the authority on whether content changed."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_of(note: Note) -> str:
    """``status:``, defaulting to active -- a note without one is not archived."""
    value = note.get("status")
    return str(value).strip() if value not in (None, "") else STATUS_ACTIVE


def _clean(value: Any) -> str:
    """A frontmatter value as trimmed text; ``""`` for ``None``/blank."""
    return "" if value is None else str(value).strip()


def _strip_wikilink(text: str) -> str:
    """Reduce ``[[id]]`` to ``id``; leave a bare id or path untouched."""
    stripped = text.strip()
    if stripped.startswith("[[") and stripped.endswith("]]"):
        return stripped[2:-2].strip()
    return stripped
