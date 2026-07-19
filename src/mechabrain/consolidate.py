"""The consolidation pipeline: maintenance, decay, deprecation (spec §9).

`mechabrain consolidate` runs the six §9 steps, in order, as one cycle. The
scheduling of that cycle is a deployment concern (cron, an agent's schedule
skill); the *pipeline* is the kernel, and it is here.

What a maintenance cycle does, and does not, do
===============================================

The through-line, as with the write gate (§8.2), is honesty about what a kernel
with **no LLM** can enforce. The mechanical steps run; the judgement steps are
detected and *reported* for an agent to act on.

======  ===========================================================
Step    What this module does
======  ===========================================================
1       **Flush accesses** (§9.1, R7.3). Aggregates the gitignored
        access log and stamps ``last_accessed:`` into the frontmatter of
        the agent notes that were read -- one commit per cycle, never per
        query. Read-only human notes are never stamped (P4).
2       **Detect duplicates** (§9.2). Same-scope, same-type pairs above
        ``dedup_similarity`` are *listed* as ``merge_candidates`` -- **not
        merged**: fusion preserves detail and needs judgement, so it is a
        `memory_write` with ``supersedes`` an agent performs. Cross-scope
        similar pairs go to a **separate** list and are never merged --
        textual similarity across projects is the R6.5 boundary, not
        redundancy.
3       **Decay** (§9.3, P8). A memory unread for ``decay_days`` becomes
        ``status: arquivado`` -- **never deleted**. Read-only context
        never decays.
4       **Deprecate procedural** (§9.4). A ``PROC`` with a successor that
        names it through ``supersedes`` becomes ``status: deprecado``,
        linked to the successor. The "contradicted by a more recent run"
        case is judgement and cannot be detected here -- see the module
        note on what is *not* covered. What *is* mechanical about ageing
        playbooks is the calendar: an active ``PROC`` untested for
        ``proc_stale_days`` (by ``last_tested:``, else ``created:``) is
        **listed** as stale for an agent to retest -- reported, never
        touched.
5       **Rebuild** (§9.5, R8.1). *Incremental* reindex (vectors + BM25)
        through :class:`~mechabrain.index.indexer.Indexer` -- only the
        notes this cycle touched (decayed, deprecated, stamped) or a human
        edited are re-embedded; the mtime+hash diff finds them. Then
        regeneration of ``index.md``/``indices/`` and ``hot.md`` -- the
        surfaces only the consolidator writes.
6       **Commit** (§9.6). One commit with ``maintenance.commit_prefix``,
        only if the vault is a git repo and something actually changed.
        ``_meta/index/`` is never staged: it is derived, per-machine and
        gitignored (§4).
======  ===========================================================

The whole cycle runs under the index lock (R7.4): a maintenance pass is one
writer. Writes to notes are atomic (R7.5) via :class:`~mechabrain.note.Note`.

What §9 asks for that this does not do
======================================

* **§9.2 fusion.** Detected, never performed -- it is the agent's call
  (a ``supersedes`` write). This module returns the candidates and stops.
* **§9.4 "contradicted by a more recent execution".** There is no mechanical
  signal for "this playbook is now wrong"; only the explicit ``supersedes``
  successor is detectable. The contradiction case is left to an agent, which is
  the faithful reading of "código para o determinístico, LLM para o julgamento".

Fidelity note (CoALA §6): the split above is deliberate. The kernel implements
the mechanically verifiable half of §9 and reports the rest; it does not fake a
judgement with a boolean.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from .access import AccessLog
from .contract import (
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    STATUS_DEPRECATED,
    MemoryType,
)
from .discovery import VaultPaths
from .errors import MechabrainError
from .gate import FULL_GATE_TYPES
from .generate import write_hot, write_index
from .graph import SUPERSEDES_RELATION, LinkGraph
from .index.embed import EmbeddingProvider
from .index.embed import from_manifest as embedder_from_manifest
from .index.indexer import Indexer
from .index.store import INDEX_LOCK_FILE
from .locking import FileLock
from .manifest import Manifest
from .note import Note, iter_notes, wikilink_for, write_atomic

__all__ = [
    "consolidate",
    "ConsolidationReport",
    "SimilarPair",
    "DecayedNote",
    "DeprecatedProcedural",
    "StaleProcedural",
    "DocCitingDead",
    "CONSOLIDATION_REPORT_FILE",
]

#: Where the report is written, under the derived index dir: per-machine and
#: gitignored, so it never enters the maintenance commit. An agent reads it as a
#: plain file; `memory_status` (§7.1) reads its ``generated`` for "last
#: consolidation". A report is derived state -- it carries a timestamp precisely
#: because it is *not* a git-tracked surface (unlike everything `generate` emits).
CONSOLIDATION_REPORT_FILE: Final[str] = "consolidation-report.json"

_LOCK_PURPOSE: Final[str] = "consolidate"


# ══════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SimilarPair:
    """Two notes above ``dedup_similarity``, either same-scope or cross-scope.

    A same-scope pair is a *merge candidate*: an agent decides ``supersedes``,
    fusion, or nothing (§9.2). A cross-scope pair is only ever a *report line*:
    R6.5 makes the same sentence about two projects two facts, so it is never
    merged. ``a``/``b`` are ordered by id so a pair has one identity.
    """

    a: str
    b: str
    memory_type: str
    similarity: float
    scope_a: str
    scope_b: str

    @property
    def cross_scope(self) -> bool:
        return self.scope_a != self.scope_b

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": wikilink_for(self.a),
            "b": wikilink_for(self.b),
            "type": self.memory_type,
            "similarity": round(self.similarity, 4),
            "scope_a": self.scope_a,
            "scope_b": self.scope_b,
        }


@dataclass(frozen=True, slots=True)
class DecayedNote:
    """A memory archived this cycle for lack of access (§9.3). Never deleted (P8)."""

    note_id: str
    memory_type: str
    scope: str
    last_reference: date

    def as_dict(self) -> dict[str, Any]:
        return {
            "note": wikilink_for(self.note_id),
            "type": self.memory_type,
            "scope": self.scope,
            "last_reference": self.last_reference.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class DeprecatedProcedural:
    """A ``PROC`` deprecated this cycle because a successor supersedes it (§9.4)."""

    note_id: str
    successors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "note": wikilink_for(self.note_id),
            "superseded_by": [wikilink_for(s) for s in self.successors],
        }


@dataclass(frozen=True, slots=True)
class StaleProcedural:
    """An active ``PROC`` whose last recorded test is older than
    ``maintenance.proc_stale_days`` (§9.4).

    Reported, never touched: whether the playbook still works is judgement, so
    the cycle lists it for an agent to retest -- exactly the detect-and-report
    split of §9.2. ``last_tested`` is ``None`` when the note predates the field
    and the age was measured from ``created:`` instead.
    """

    note_id: str
    scope: str
    last_tested: date | None
    days_stale: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "note": wikilink_for(self.note_id),
            "scope": self.scope,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "days_stale": self.days_stale,
        }


@dataclass(frozen=True, slots=True)
class DocCitingDead:
    """A read-only doc (Classe B content) whose wikilink points at a memory that
    is archived, deprecated or superseded (Mecha-Scribe Fase 2).

    A mechanical sign the doc lags the memory it cites -- the kind of thing the
    escriba should fix by proposing an edit. Reported, never touched (P4): the
    kernel does not rewrite human/agent-authored docs.
    """

    doc: str
    cited: str
    status: str
    successor: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc": wikilink_for(self.doc),
            "cites": wikilink_for(self.cited),
            "status": self.status,
            "successor": wikilink_for(self.successor) if self.successor else None,
        }


@dataclass(frozen=True, slots=True)
class ConsolidationReport:
    """The result of one :func:`consolidate` cycle -- counts plus the judgement
    items an agent must act on.

    Persisted to :data:`CONSOLIDATION_REPORT_FILE` (unless ``dry_run``) and also
    returned. ``merge_candidates`` and ``cross_scope_similar`` are the two halves
    of §9.2 detection; ``decayed`` and ``deprecated`` are what the mechanical
    steps changed.
    """

    generated: str
    dry_run: bool
    counts: dict[str, int]
    merge_candidates: tuple[SimilarPair, ...] = ()
    cross_scope_similar: tuple[SimilarPair, ...] = ()
    decayed: tuple[DecayedNote, ...] = ()
    deprecated: tuple[DeprecatedProcedural, ...] = ()
    stale_procedurals: tuple[StaleProcedural, ...] = ()
    docs_citing_dead: tuple[DocCitingDead, ...] = ()
    committed: bool = False
    commit: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "dry_run": self.dry_run,
            "counts": dict(self.counts),
            "merge_candidates": [p.as_dict() for p in self.merge_candidates],
            "cross_scope_similar": [p.as_dict() for p in self.cross_scope_similar],
            "decayed": [d.as_dict() for d in self.decayed],
            "deprecated": [d.as_dict() for d in self.deprecated],
            "stale_procedurals": [s.as_dict() for s in self.stale_procedurals],
            "docs_citing_dead": [d.as_dict() for d in self.docs_citing_dead],
            "committed": self.committed,
            "commit": self.commit,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n"


# ══════════════════════════════════════════════════════════════════════
# Internal note view
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class _Typed:
    """A note paired with its resolved memory type and scope, computed once."""

    note: Note
    memory_type: MemoryType | None
    scope: str

    @property
    def note_id(self) -> str:
        return self.note.note_id


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
def consolidate(
    paths: VaultPaths,
    manifest: Manifest,
    *,
    today: date | None = None,
    dry_run: bool = False,
    commit: bool = True,
) -> ConsolidationReport:
    """Run the §9 pipeline over the vault at ``paths`` and return a report.

    Args:
        paths: The discovered vault (:func:`mechabrain.discovery.discover_vault`).
        manifest: Its parsed manifest -- the source of every threshold (P6).
        today: Reference date for decay (§9.3). Defaults to today, UTC. A seam
            for tests; production passes nothing.
        dry_run: Compute the report without touching disk -- no frontmatter
            written, no access log consumed, no reindex, no surfaces, no commit.
            The ``--dry-run`` flag.
        commit: Attempt the §9.6 commit. ``False`` is the ``--no-commit`` flag:
            everything else runs, but no git commit is made.

    Returns:
        A :class:`ConsolidationReport`. On a non-dry run it is also written to
        :data:`CONSOLIDATION_REPORT_FILE` under the index dir.

    Raises:
        MechabrainError: a git command failed (R5.1 -- a broken commit is
            surfaced, never swallowed), or an index rebuild failed.
    """
    reference = today if today is not None else datetime.now(timezone.utc).date()

    memory_notes = _load_typed(_scan_memory(paths, manifest), manifest)
    readonly_notes = _load_typed(_scan_readonly(paths, manifest, memory_notes), manifest)
    memory_by_id = {t.note_id: t for t in memory_notes if t.note_id}

    graph = LinkGraph.build(
        paths, manifest, notes=[t.note for t in (*memory_notes, *readonly_notes)]
    )
    deprecatable = _deprecatable_procedurals(graph, memory_by_id)

    changed: set[Path] = set()
    # One provider for the whole cycle (construction is lazy and offline): the
    # dedup pass and the reindex must share a model load, not repeat it.
    provider = embedder_from_manifest(manifest)
    # One lock *instance* for the whole cycle -- the Indexer reuses it (FileLock
    # is reentrant within an instance), so step 5 nests instead of deadlocking.
    lock = FileLock(paths.index_dir / INDEX_LOCK_FILE, purpose=_LOCK_PURPOSE)

    with lock:
        # 1 -- flush accesses (§9.1, R7.3)
        accesses, accesses_applied = _apply_access(paths, memory_by_id, dry_run, changed)

        # 2 -- detect duplicates (§9.2): report only, never merge
        merge_candidates, cross_scope = _detect_duplicates(memory_notes, manifest, provider)

        # 3 -- decay (§9.3): archive, never delete (P8)
        decayed = _decay(memory_notes, reference, manifest, deprecatable, dry_run, changed)

        # 4 -- deprecate procedural (§9.4)
        deprecated = _deprecate(deprecatable, memory_by_id, dry_run, changed)

        # 4b -- report stale procedurals (§9.4): detect and report, never touch.
        stale_procedurals = _stale_procedurals(
            memory_notes,
            reference,
            manifest,
            excluded={*deprecatable, *(d.note_id for d in decayed)},
        )

        # 4c -- validate read-only docs (Mecha-Scribe Fase 2): report docs that
        # cite a memory now archived/deprecated/superseded. Reuses the graph the
        # cycle already built; detect-and-report only -- a doc is human/agent
        # content, never touched (P4).
        docs_citing_dead = _validate_readonly_docs(
            graph, memory_by_id, readonly_notes, decayed, deprecated
        )

        # 5 -- rebuild (§9.5, R8.1): incremental reindex + regenerate surfaces.
        # Steps 1, 3 and 4 already wrote their frontmatter to disk (note.write()
        # is where each stamp lands), so the mtime+hash diff sees exactly the
        # notes this cycle touched and re-embeds only those.
        chunk_count = 0
        notes_reindexed = 0
        if not dry_run:
            index_report = Indexer(paths, manifest, provider=provider, lock=lock).reindex()
            chunk_count = index_report.chunks_written
            notes_reindexed = index_report.notes_indexed
            active_scopes = _active_scopes(accesses, memory_by_id, manifest)
            write_index(paths, [t.note for t in memory_notes], manifest)
            write_hot(
                paths,
                [t.note for t in memory_notes],
                manifest,
                active_scopes=active_scopes,
            )

        # 6 -- commit (§9.6): one commit, never _meta/index/
        committed, sha = False, None
        if not dry_run and commit:
            committed, sha = _commit(paths, manifest, changed, decayed, deprecated, accesses_applied)

    counts = {
        "notes_scanned": len(memory_notes),
        "readonly_scanned": len(readonly_notes),
        "accesses_applied": accesses_applied,
        "merge_candidates": len(merge_candidates),
        "cross_scope_similar": len(cross_scope),
        "decayed": len(decayed),
        "deprecated": len(deprecated),
        "stale_procedurals": len(stale_procedurals),
        "docs_citing_dead": len(docs_citing_dead),
        "notes_reindexed": notes_reindexed,
        "chunks_indexed": chunk_count,
    }
    report = ConsolidationReport(
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dry_run=dry_run,
        counts=counts,
        merge_candidates=tuple(merge_candidates),
        cross_scope_similar=tuple(cross_scope),
        decayed=tuple(decayed),
        deprecated=tuple(deprecated),
        stale_procedurals=tuple(stale_procedurals),
        docs_citing_dead=docs_citing_dead,
        committed=committed,
        commit=sha,
    )
    if not dry_run:
        _persist_report(paths, report)
    return report


# ══════════════════════════════════════════════════════════════════════
# Scanning and typing
# ══════════════════════════════════════════════════════════════════════
def _scan_memory(paths: VaultPaths, manifest: Manifest) -> list[Note]:
    """Every note in an enabled memory folder (§3), in sorted path order."""
    notes: list[Note] = []
    for memory_type in MemoryType:
        if manifest.is_enabled(memory_type):
            notes.extend(iter_notes(paths.folder_for(memory_type)))
    return notes


def _scan_readonly(
    paths: VaultPaths, manifest: Manifest, memory: Sequence[_Typed]
) -> list[Note]:
    """Notes under ``zones.read_only_index`` -- human context, never mutated (P4).

    Deduplicated by path against the memory folders, so a ``read_only_index``
    entry that points back into ``mecha-brain/`` cannot double-count a note.
    """
    seen = {t.note.path for t in memory}
    notes: list[Note] = []
    for folder in manifest.zones.read_only_index:
        for note in iter_notes(paths.resolve(folder)):
            if note.path not in seen:
                seen.add(note.path)
                notes.append(note)
    return notes


def _load_typed(notes: Iterable[Note], manifest: Manifest) -> list[_Typed]:
    return [_Typed(note, _type_of(note, manifest), _scope_of(note, manifest)) for note in notes]


def _type_of(note: Note, manifest: Manifest) -> MemoryType | None:
    """The note's memory type: from its folder, else its ``mem/`` tag.

    Folder first: a note physically in ``Procedural/`` is procedural whatever a
    human did to its tags. The tag is the fallback for a ``read_only_index`` note
    that happens to carry one; most human notes resolve to ``None``.
    """
    return _type_by_folder(note, manifest) or _type_by_tag(note, manifest)


def _type_by_folder(note: Note, manifest: Manifest) -> MemoryType | None:
    if note.path is None:
        return None
    for memory_type in MemoryType:
        folder_name = manifest.folder_for(memory_type)
        for parent in note.path.parents:
            if parent.name == folder_name:
                return memory_type
    return None


def _type_by_tag(note: Note, manifest: Manifest) -> MemoryType | None:
    namespaces = manifest.frontmatter.tag_namespaces
    for memory_type in MemoryType:
        if note.has_tag(namespaces.memory_tag(memory_type)):
            return memory_type
    return None


def _scope_of(note: Note, manifest: Manifest) -> str:
    value = note.get("scope")
    scope = str(value).strip() if value not in (None, "") else ""
    return scope or manifest.scopes.default


def _status_of(note: Note) -> str:
    value = note.get("status")
    return str(value).strip() if value not in (None, "") else STATUS_ACTIVE


def _as_date(value: Any) -> date | None:
    """Coerce a frontmatter date value, which YAML may hand over already parsed."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# Step 1 -- flush accesses (§9.1, R7.3)
# ══════════════════════════════════════════════════════════════════════
def _apply_access(
    paths: VaultPaths,
    memory_by_id: Mapping[str, _Typed],
    dry_run: bool,
    changed: set[Path],
) -> tuple[dict[str, date], int]:
    """Stamp ``last_accessed:`` on the agent notes that were read (R7.3).

    Read-only human notes are never stamped: they are outside the sandbox (P4),
    so an access to one is dropped here silently -- it simply has no frontmatter
    the kernel may write. A dry run *reads* the log (``aggregate``) without
    consuming it; a real run consumes it under :meth:`AccessLog.flush`, so a
    stamp is applied at most once per cycle.

    Returns:
        ``({note_id: access date}, notes stamped)``.
    """
    log = AccessLog.for_vault(paths)
    if dry_run:
        accesses = log.aggregate()
        applied = sum(
            1
            for note_id, when in accesses.items()
            if note_id in memory_by_id and _needs_last_accessed(memory_by_id[note_id].note, when)
        )
        return accesses, applied

    with log.flush() as accesses:
        # Copy: the mapping is only valid inside the context, but the caller
        # needs it afterwards to pick active scopes for hot.md.
        snapshot = dict(accesses)
        applied = 0
        for note_id, when in snapshot.items():
            typed = memory_by_id.get(note_id)
            if typed is None:
                continue
            if _stamp_last_accessed(typed.note, when):
                typed.note.write()
                changed.add(typed.note.path)  # type: ignore[arg-type]
                applied += 1
    return snapshot, applied


def _needs_last_accessed(note: Note, when: date) -> bool:
    current = _as_date(note.get("last_accessed"))
    return current is None or when > current


def _stamp_last_accessed(note: Note, when: date) -> bool:
    """Set ``last_accessed:`` to ``when`` if that moves it forward. Idempotent."""
    if not _needs_last_accessed(note, when):
        return False
    note.frontmatter["last_accessed"] = when
    return True


# ══════════════════════════════════════════════════════════════════════
# Step 2 -- detect duplicates (§9.2)
# ══════════════════════════════════════════════════════════════════════
def _detect_duplicates(
    memory: Sequence[_Typed], manifest: Manifest, provider: EmbeddingProvider
) -> tuple[list[SimilarPair], list[SimilarPair]]:
    """Find same-type pairs above ``dedup_similarity`` and split by scope.

    Only active ``Semantic``/``Procedural`` notes are compared -- the gated
    types (:data:`~mechabrain.gate.FULL_GATE_TYPES`): ``Episodic`` is an
    append-only diary, ``Research`` is non-atomic, and neither is a fusion
    target. Pairs are compared **within one type**: a ``PROC`` resembling an
    ``INS`` is a related note, not a duplicate (mirrors the §8.2 gate).

    Notes are embedded at document granularity (title + body), one vector each,
    which is enough to *flag* a candidate; the agent that acts on it reads the
    full notes. This is an embedding pass of its own beyond the (incremental)
    reindex -- accepted: it cannot reuse the index's vectors, which are per
    *chunk* and carry the contextual prefix, and the model is shared with the
    reindex via ``provider``.

    Returns:
        ``(merge_candidates, cross_scope_similar)``, each sorted by descending
        similarity then id, deterministically.
    """
    candidates = [
        typed
        for typed in memory
        if typed.memory_type in FULL_GATE_TYPES and _status_of(typed.note) == STATUS_ACTIVE
    ]
    if len(candidates) < 2:
        return [], []

    vectors = provider.embed_texts([_document(typed.note) for typed in candidates])
    threshold = manifest.maintenance.dedup_similarity

    by_type: dict[MemoryType, list[int]] = {}
    for index, typed in enumerate(candidates):
        assert typed.memory_type is not None
        by_type.setdefault(typed.memory_type, []).append(index)

    merge: list[SimilarPair] = []
    cross: list[SimilarPair] = []
    for memory_type, indices in by_type.items():
        if len(indices) < 2:
            continue
        block = np.asarray(vectors)[indices]
        sims = block @ block.T
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                similarity = float(sims[i, j])
                if similarity <= threshold:
                    continue
                left, right = candidates[indices[i]], candidates[indices[j]]
                pair = _make_pair(left, right, memory_type.value, similarity)
                (cross if pair.cross_scope else merge).append(pair)

    merge.sort(key=lambda p: (-p.similarity, p.a, p.b))
    cross.sort(key=lambda p: (-p.similarity, p.a, p.b))
    return merge, cross


def _make_pair(left: _Typed, right: _Typed, memory_type: str, similarity: float) -> SimilarPair:
    """Order the pair by id so it has one identity, keeping scopes aligned."""
    if left.note_id <= right.note_id:
        first, second = left, right
    else:
        first, second = right, left
    return SimilarPair(
        a=first.note_id,
        b=second.note_id,
        memory_type=memory_type,
        similarity=similarity,
        scope_a=first.scope,
        scope_b=second.scope,
    )


def _document(note: Note) -> str:
    """The text embedded for note-level dedup: the claim, title first."""
    body = note.body.strip()
    title = note.title.strip()
    return f"{title}\n\n{body}".strip() or note.note_id


# ══════════════════════════════════════════════════════════════════════
# Step 3 -- decay (§9.3)
# ══════════════════════════════════════════════════════════════════════
def _decay(
    memory: Sequence[_Typed],
    today: date,
    manifest: Manifest,
    deprecatable: Mapping[str, list[str]],
    dry_run: bool,
    changed: set[Path],
) -> list[DecayedNote]:
    """Archive memories unread for ``decay_days`` -- ``status: arquivado`` (§9.3).

    Never deletes (P8): the file stays, searchable behind an explicit ``status``
    filter, out of ``index.md`` and lighter in retrieval. Read-only context is
    not in ``memory`` and so never decays. A memory with no date at all is left
    alone -- archiving without evidence of age would be a guess. A procedural
    with a successor is left for step 4, which gives it the more informative
    ``deprecado`` instead of ``arquivado``.

    "Unread" is measured from ``last_accessed``, falling back to ``modified``
    then ``created`` for a memory the access log never named.
    """
    decay_days = manifest.maintenance.decay_days
    decayed: list[DecayedNote] = []
    for typed in memory:
        note = typed.note
        if _status_of(note) != STATUS_ACTIVE or typed.note_id in deprecatable:
            continue
        reference = _reference_date(note)
        if reference is None or (today - reference).days <= decay_days:
            continue
        if not dry_run:
            note.frontmatter["status"] = STATUS_ARCHIVED
            note.write()
            changed.add(note.path)  # type: ignore[arg-type]
        decayed.append(
            DecayedNote(
                note_id=typed.note_id,
                memory_type=typed.memory_type.value if typed.memory_type else "",
                scope=typed.scope,
                last_reference=reference,
            )
        )
    return decayed


def _reference_date(note: Note) -> date | None:
    for key in ("last_accessed", "modified", "created"):
        parsed = _as_date(note.get(key))
        if parsed is not None:
            return parsed
    return None


# ══════════════════════════════════════════════════════════════════════
# Step 4 -- deprecate procedural (§9.4)
# ══════════════════════════════════════════════════════════════════════
def _deprecatable_procedurals(
    graph: LinkGraph, memory_by_id: Mapping[str, _Typed]
) -> dict[str, list[str]]:
    """``old_id -> [successor_id, ...]`` for active ``PROC`` notes with a successor.

    A supersedes edge runs ``successor -> old`` (new replaces old, §6). The old
    end is the note to deprecate, *if* it is an active procedural memory. Read
    from the authored graph so a supersedes recorded in frontmatter or via
    ``memory_link`` both count.
    """
    superseded: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.relation == SUPERSEDES_RELATION:
            superseded.setdefault(edge.target, []).append(edge.source)

    out: dict[str, list[str]] = {}
    for old_id, successors in superseded.items():
        typed = memory_by_id.get(old_id)
        if (
            typed is not None
            and typed.memory_type is MemoryType.PROCEDURAL
            and _status_of(typed.note) == STATUS_ACTIVE
        ):
            out[old_id] = sorted(dict.fromkeys(successors))
    return out


def _deprecate(
    deprecatable: Mapping[str, list[str]],
    memory_by_id: Mapping[str, _Typed],
    dry_run: bool,
    changed: set[Path],
) -> list[DeprecatedProcedural]:
    """Mark each deprecatable ``PROC`` ``deprecado``, linked to its successor (§9.4).

    The link is an inverse pointer ``superseded_by:`` in the deprecated note's
    frontmatter -- the successor already names it through ``supersedes:``, and
    this lets a reader who lands on the old note find the new one. Idempotent:
    the note is already off the ``active`` list on the next cycle, so it is not
    reprocessed and the report stops listing it.
    """
    deprecated: list[DeprecatedProcedural] = []
    for old_id, successors in sorted(deprecatable.items()):
        typed = memory_by_id[old_id]
        links = [wikilink_for(s) for s in successors]
        if not dry_run:
            typed.note.frontmatter["status"] = STATUS_DEPRECATED
            typed.note.frontmatter["superseded_by"] = links[0] if len(links) == 1 else links
            typed.note.write()
            changed.add(typed.note.path)  # type: ignore[arg-type]
        deprecated.append(
            DeprecatedProcedural(note_id=old_id, successors=tuple(successors))
        )
    return deprecated


# ══════════════════════════════════════════════════════════════════════
# Step 4b -- report stale procedurals (§9.4)
# ══════════════════════════════════════════════════════════════════════
def _stale_procedurals(
    memory: Sequence[_Typed],
    today: date,
    manifest: Manifest,
    excluded: set[str],
) -> list[StaleProcedural]:
    """List active ``PROC`` notes whose last recorded test is older than
    ``maintenance.proc_stale_days`` (§9.4). Detect and report -- never touch.

    Age is measured from ``last_tested:`` (stamped by the writer on every
    procedural write, since §8.2 item 6 evidence attests a run at that date),
    falling back to ``created:`` for a note that predates the field. Never
    ``last_accessed``/``modified``: reading a playbook is not testing it. A note
    with neither date is left alone -- flagging without evidence of age would be
    a guess, the same principle as decay. ``excluded`` carries the ids this
    cycle already deprecated or decayed: their status changed, and a second
    report line would only shout over the first.
    """
    stale_days = manifest.maintenance.proc_stale_days
    if stale_days <= 0:
        return []
    stale: list[StaleProcedural] = []
    for typed in memory:
        if (
            typed.memory_type is not MemoryType.PROCEDURAL
            or typed.note_id in excluded
            or _status_of(typed.note) != STATUS_ACTIVE
        ):
            continue
        last_tested = _as_date(typed.note.get("last_tested"))
        reference = last_tested or _as_date(typed.note.get("created"))
        if reference is None:
            continue
        days = (today - reference).days
        if days <= stale_days:
            continue
        stale.append(
            StaleProcedural(
                note_id=typed.note_id,
                scope=typed.scope,
                last_tested=last_tested,
                days_stale=days,
            )
        )
    stale.sort(key=lambda s: (-s.days_stale, s.note_id))
    return stale


# ══════════════════════════════════════════════════════════════════════
# Step 4c -- validate read-only docs (Mecha-Scribe Fase 2)
# ══════════════════════════════════════════════════════════════════════
def _superseded_by_ids(note: Note) -> list[str]:
    """Note ids named by a note's ``superseded_by:`` frontmatter, wikilinks stripped.

    Empty when the field is absent. Accepts a single value or a list.
    """
    raw = note.get("superseded_by")
    if not raw:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    ids: list[str] = []
    for value in values:
        text = str(value).strip().strip("[]")
        text = text.split("|", 1)[0].split("#", 1)[0].strip()
        if text:
            ids.append(text)
    return ids


def _validate_readonly_docs(
    graph: LinkGraph,
    memory_by_id: Mapping[str, _Typed],
    readonly_notes: Sequence[_Typed],
    decayed: Sequence[DecayedNote],
    deprecated: Sequence[DeprecatedProcedural],
) -> tuple[DocCitingDead, ...]:
    """Report read-only docs that cite a now-dead memory (Fase 2).

    A doc links to a memory that is ``arquivado``, ``deprecado`` or superseded;
    the successor is carried when known so the report can say "cites [[Y]]
    (deprecada → [[Z]])". Read off the ``LinkGraph`` this cycle already built.

    This is the *reliable* half of doc validation: the target is a real memory
    the graph knows, so there is no false positive. A general "broken wikilink"
    check was deliberately dropped -- the graph only spans the memory folders
    plus ``zones.read_only_index``, so it cannot tell a genuinely missing note
    from one that simply lives elsewhere in the vault (a root dashboard, an
    ``MOC_*``, an attachment ``![[img.png]]``); flagging those is noise, and
    Obsidian already surfaces real broken links. Detect-and-report only: a doc
    is human/agent-authored content, never edited (P4) -- an agent proposes the
    fix via ``memory_propose``.
    """
    readonly_ids = {t.note_id for t in readonly_notes if t.note_id}
    if not readonly_ids:
        return ()

    # A memory is "dead" if archived/deprecated/superseded. The note frontmatter
    # already reflects this cycle's decay/deprecation (those steps mutate the
    # same note objects), and the explicit sets add successor provenance.
    dead: dict[str, tuple[str, str | None]] = {}
    for note_id, typed in memory_by_id.items():
        status = _status_of(typed.note)
        successors = _superseded_by_ids(typed.note)
        successor = successors[0] if successors else None
        if status == STATUS_DEPRECATED:
            dead[note_id] = ("deprecado", successor)
        elif status == STATUS_ARCHIVED:
            dead[note_id] = ("superseded" if successor else "arquivado", successor)
    for node in decayed:
        dead.setdefault(node.note_id, ("arquivado", None))
    for proc in deprecated:
        dead[proc.note_id] = ("deprecado", proc.successors[0] if proc.successors else None)

    seen: set[tuple[str, str]] = set()
    citing_dead: list[DocCitingDead] = []
    for edge in graph.edges:
        if edge.source not in readonly_ids:
            continue
        info = dead.get(edge.target)
        if info is None:
            continue
        key = (edge.source, edge.target)
        if key in seen:
            continue
        seen.add(key)
        citing_dead.append(
            DocCitingDead(doc=edge.source, cited=edge.target, status=info[0], successor=info[1])
        )
    citing_dead.sort(key=lambda d: (d.doc, d.cited))
    return tuple(citing_dead)


# ══════════════════════════════════════════════════════════════════════
# Step 5 -- rebuild (§9.5, R8.1)
# ══════════════════════════════════════════════════════════════════════
# The reindex itself is the Indexer's incremental pass (see step 5 in
# `consolidate`): after decay/deprecation have written their frontmatter, the
# mtime+hash diff re-embeds exactly the notes this cycle changed, and the state
# fingerprint upgrades to a full rebuild when the deployment itself changed.
# What remains here is the surface regeneration input below.
def _active_scopes(
    accesses: Mapping[str, date], memory_by_id: Mapping[str, _Typed], manifest: Manifest
) -> Sequence[str] | None:
    """Scopes read this cycle, most recent first -- what hot.md focuses on (R8.2).

    ``None`` when nothing was accessed, which lets `render_hot` fall back to its
    own recency ordering. "Recent access" is the only attention signal the
    kernel can measure without an LLM.
    """
    recency: dict[str, date] = {}
    for note_id, when in accesses.items():
        typed = memory_by_id.get(note_id)
        if typed is None:
            continue
        if typed.scope not in recency or when > recency[typed.scope]:
            recency[typed.scope] = when
    if not recency:
        return None
    return sorted(recency, key=lambda scope: ((date.max - recency[scope]).days, scope))


# ══════════════════════════════════════════════════════════════════════
# Step 6 -- commit (§9.6)
# ══════════════════════════════════════════════════════════════════════
def _commit(
    paths: VaultPaths,
    manifest: Manifest,
    changed: set[Path],
    decayed: Sequence[DecayedNote],
    deprecated: Sequence[DeprecatedProcedural],
    accesses_applied: int,
) -> tuple[bool, str | None]:
    """One commit of the maintenance changes, if the vault is git and anything moved.

    Stages only the surfaces the consolidator owns (``index.md``, ``indices/``,
    ``hot.md``) and the notes it rewrote -- never ``_meta/index/``, which is
    gitignored, per-machine and not portable across embedding models (§4). An
    empty staging area produces no commit, so an idempotent re-run is a no-op.
    """
    if not _is_git_repo(paths.root):
        return False, None

    pathspecs = [paths.index_file, paths.hot_file, *sorted(changed)]
    if paths.indices_dir.is_dir():
        pathspecs.append(paths.indices_dir)
    _git(paths.root, "add", "-A", "--", *[str(p) for p in pathspecs])

    if _git(paths.root, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False, None

    message = (
        f"{manifest.maintenance.commit_prefix} consolidate: "
        f"{len(decayed)} archived, {len(deprecated)} deprecated, "
        f"{accesses_applied} access stamp(s)"
    )
    _git(paths.root, "commit", "-m", message)
    sha = _git(paths.root, "rev-parse", "HEAD").stdout.strip()
    return True, sha or None


def _is_git_repo(root: Path) -> bool:
    result = _git(root, "rev-parse", "--is-inside-work-tree", check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run one git command in ``root``. Fail loud on a non-zero exit (R5.1)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # git is not installed: the vault cannot be a repo we commit to.
        return subprocess.CompletedProcess(args=list(args), returncode=127, stdout="", stderr="git not found")
    if check and result.returncode != 0:
        raise MechabrainError(
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip() or result.stdout.strip()}",
            rule="§9.6",
            hint="the maintenance changes are written; only the commit failed. "
            "Fix the git state and re-run `mechabrain consolidate`",
        )
    return result


# ══════════════════════════════════════════════════════════════════════
# Report persistence
# ══════════════════════════════════════════════════════════════════════
def _persist_report(paths: VaultPaths, report: ConsolidationReport) -> None:
    """Write the report under the index dir -- per-machine, gitignored, agent-readable."""
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(paths.index_dir / CONSOLIDATION_REPORT_FILE, report.to_json())
