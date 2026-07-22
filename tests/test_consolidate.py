"""Tests for the consolidation pipeline (spec §9).

Each of the six steps is exercised in isolation, plus the two invariants the
spec is most emphatic about: **never merge across scopes** (§9.2, R6.5) and
**never delete** (§9.3, P8).

The vault fixtures come from ``conftest`` (``make_vault``, ``write_note``,
``manifest_data_ci``); the CI manifest embeds with the deterministic offline
``hash`` provider and the ``numpy`` store, so a full reindex runs with no model
download and identical vectors on every machine.
"""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mechabrain.access import AccessKind, AccessLog
from mechabrain.consolidate import (
    CONSOLIDATION_REPORT_FILE,
    ConsolidationReport,
    consolidate,
)
from mechabrain.contract import STATUS_ACTIVE, STATUS_ARCHIVED, STATUS_DEPRECATED
from mechabrain.discovery import VaultPaths
from mechabrain.index.embed import from_manifest as embedder_from_manifest
from mechabrain.index.indexer import Indexer
from mechabrain.index.lexical import LEXICAL_DB_FILENAME, LexicalIndex
from mechabrain.index.store import from_manifest as store_from_manifest
from mechabrain.manifest import Manifest, load_manifest
from mechabrain.note import Note

_OLD = date(2026, 1, 15)
_FUTURE = date(2026, 12, 31)


# ══════════════════════════════════════════════════════════════════════
# Builders
# ══════════════════════════════════════════════════════════════════════
def _vault(
    make_vault: Callable[..., VaultPaths],
    base: Mapping[str, Any],
    **maintenance_and_zones: Any,
) -> tuple[VaultPaths, Manifest]:
    """A disposable vault with tweaked maintenance/zones, plus its parsed manifest."""
    data = copy.deepcopy(dict(base))
    for key in ("decay_days", "dedup_similarity", "proc_stale_days"):
        if key in maintenance_and_zones:
            data["maintenance"][key] = maintenance_and_zones[key]
    if "read_only_index" in maintenance_and_zones:
        data["zones"]["read_only_index"] = maintenance_and_zones["read_only_index"]
    paths = make_vault(manifest_data=data)
    return paths, load_manifest(paths.config_file)


def _sem(
    write_note: Callable[..., Note],
    path: Path,
    scope: str,
    body: str,
    *,
    title: str = "A fact",
    agent: str = "alpha",
    status: str = STATUS_ACTIVE,
    created: date = _OLD,
    last_accessed: date | None = None,
    supersedes: str | None = None,
) -> Note:
    frontmatter: dict[str, Any] = {
        "title": title,
        "tags": ["mem/semantic", f"agent/{agent}"],
        "created": created,
        "modified": created,
        "agent": agent,
        "scope": scope,
        "source": "test-session",
        "confidence": "medium",
        "status": status,
    }
    if last_accessed is not None:
        frontmatter["last_accessed"] = last_accessed
    if supersedes is not None:
        frontmatter["supersedes"] = supersedes
    return write_note(path, frontmatter, body)


def _proc(
    write_note: Callable[..., Note],
    path: Path,
    scope: str,
    body: str,
    *,
    title: str = "A playbook",
    status: str = STATUS_ACTIVE,
    created: date = _OLD,
    last_accessed: date | None = None,
    last_tested: date | None = None,
    supersedes: str | None = None,
) -> Note:
    frontmatter: dict[str, Any] = {
        "title": title,
        "tags": ["mem/procedural", "agent/beta"],
        "created": created,
        "modified": created,
        "agent": "beta",
        "scope": scope,
        "source": "test-session",
        "confidence": "medium",
        "status": status,
    }
    if last_accessed is not None:
        frontmatter["last_accessed"] = last_accessed
    if last_tested is not None:
        frontmatter["last_tested"] = last_tested
    if supersedes is not None:
        frontmatter["supersedes"] = supersedes
    return write_note(path, frontmatter, body)


def _git_init(root: Path) -> None:
    for args in (
        ["init"],
        ["config", "user.email", "t@example.test"],
        ["config", "user.name", "Tester"],
        ["add", "-A"],
        ["commit", "-m", "initial"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _commit_count(root: Path) -> int:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--count", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


# ══════════════════════════════════════════════════════════════════════
# Step 1 -- flush accesses (§9.1, R7.3)
# ══════════════════════════════════════════════════════════════════════
def test_flush_stamps_last_accessed_and_consumes_log(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    note = _sem(write_note, paths.semantic_dir / "2026-01-15_INS_a.md", "proj-a", "A body.")

    log = AccessLog.for_vault(paths)
    log.record(note.note_id, AccessKind.SEARCH)
    expected = log.aggregate()[note.note_id]

    report = consolidate(paths, manifest, today=expected)

    reloaded = Note.load(note.path)
    assert reloaded.get("last_accessed") == expected
    assert report.counts["accesses_applied"] == 1
    # The log was rotated and consumed -- one stamp per cycle, not per query.
    assert AccessLog.for_vault(paths).aggregate() == {}


def test_flush_never_stamps_read_only_notes(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(
        make_vault, manifest_data_ci, decay_days=1, read_only_index=["Notes/"]
    )
    human_path = paths.root / "Notes" / "human-note.md"
    human_path.parent.mkdir(parents=True, exist_ok=True)
    human_path.write_text("# Human note\n\nWritten by a person.\n", encoding="utf-8")
    original = human_path.read_text(encoding="utf-8")

    # An access to the human note is recorded, but it lives outside the sandbox.
    AccessLog.for_vault(paths).record("human-note", AccessKind.GET)

    consolidate(paths, manifest, today=_FUTURE)

    # P4: the kernel never writes a human note -- no last_accessed, no decay.
    assert human_path.read_text(encoding="utf-8") == original


# ══════════════════════════════════════════════════════════════════════
# Step 2 -- dedup detection (§9.2)
# ══════════════════════════════════════════════════════════════════════
def test_never_merges_across_scopes(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    text = "Brute-force cosine is fast enough below ten thousand chunks."
    a = _sem(write_note, paths.semantic_dir / "2026-01-15_INS_a.md", "proj-a", text)
    b = _sem(write_note, paths.semantic_dir / "2026-01-15_INS_b.md", "global", text)

    report = consolidate(paths, manifest, today=_OLD)

    # The identical pair is detected as CROSS-scope and never merged.
    assert len(report.cross_scope_similar) == 1
    assert report.merge_candidates == ()
    pair = report.cross_scope_similar[0]
    assert {pair.a, pair.b} == {a.note_id, b.note_id}
    assert pair.cross_scope is True
    # Neither note was touched: cross-scope similarity is the R6.5 boundary.
    assert Note.load(a.path).get("status") == STATUS_ACTIVE
    assert Note.load(b.path).get("status") == STATUS_ACTIVE
    assert Note.load(a.path).body.strip() == text
    assert Note.load(b.path).body.strip() == text


def test_same_scope_pair_is_a_merge_candidate_not_merged(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    text = "Markdown stays the source of truth; every index is derived from it."
    a = _sem(write_note, paths.semantic_dir / "2026-01-15_INS_a.md", "proj-a", text)
    b = _sem(write_note, paths.semantic_dir / "2026-01-15_INS_b.md", "proj-a", text)

    report = consolidate(paths, manifest, today=_OLD)

    assert len(report.merge_candidates) == 1
    assert report.cross_scope_similar == ()
    pair = report.merge_candidates[0]
    assert {pair.a, pair.b} == {a.note_id, b.note_id}
    assert pair.similarity > manifest.maintenance.dedup_similarity
    # Detected only: fusion is a memory_write with supersedes an agent performs.
    assert Note.load(a.path).get("status") == STATUS_ACTIVE
    assert Note.load(b.path).get("status") == STATUS_ACTIVE


def test_dedup_only_compares_within_one_type(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    text = "Exactly the same words, in a semantic note and a procedural note."
    _sem(write_note, paths.semantic_dir / "2026-01-15_INS_a.md", "proj-a", text)
    _proc(write_note, paths.procedural_dir / "PROC_a.md", "proj-a", text)

    report = consolidate(paths, manifest, today=_OLD)

    # A PROC resembling an INS is a related note, not a duplicate (§8.2 mirror).
    assert report.merge_candidates == ()
    assert report.cross_scope_similar == ()


def test_dedup_ignores_archived_notes(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    text = "An archived duplicate must not resurface as a merge candidate."
    _sem(write_note, paths.semantic_dir / "2026-01-15_INS_a.md", "proj-a", text)
    _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_b.md",
        "proj-a",
        text,
        status=STATUS_ARCHIVED,
    )

    report = consolidate(paths, manifest, today=_OLD)

    assert report.merge_candidates == ()
    assert report.cross_scope_similar == ()


# ══════════════════════════════════════════════════════════════════════
# Step 3 -- decay (§9.3, P8)
# ══════════════════════════════════════════════════════════════════════
def test_decay_archives_stale_never_deletes(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    note = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Nobody has read this in a long time.",
        last_accessed=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert note.path.exists()  # P8: archived, never deleted
    assert Note.load(note.path).get("status") == STATUS_ARCHIVED
    assert len(report.decayed) == 1
    assert report.decayed[0].note_id == note.note_id


def test_fresh_note_does_not_decay(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    note = _sem(
        write_note,
        paths.semantic_dir / "2026-12-30_INS_fresh.md",
        "proj-a",
        "Read yesterday.",
        last_accessed=date(2026, 12, 30),
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert Note.load(note.path).get("status") == STATUS_ACTIVE
    assert report.decayed == ()


def test_read_only_context_never_decays(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(
        make_vault, manifest_data_ci, decay_days=1, read_only_index=["Notes/"]
    )
    human_path = paths.root / "Notes" / "ancient.md"
    human_path.parent.mkdir(parents=True, exist_ok=True)
    human_path.write_text(
        "---\ncreated: 2000-01-01\n---\n\nAn ancient human note.\n", encoding="utf-8"
    )
    original = human_path.read_text(encoding="utf-8")

    report = consolidate(paths, manifest, today=_FUTURE)

    assert human_path.read_text(encoding="utf-8") == original
    assert all(d.note_id != "ancient" for d in report.decayed)


# ══════════════════════════════════════════════════════════════════════
# Step 4 -- deprecate procedural (§9.4)
# ══════════════════════════════════════════════════════════════════════
def test_procedural_with_successor_is_deprecated(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    old = _proc(write_note, paths.procedural_dir / "PROC_old.md", "proj-a", "Old steps.")
    new = _proc(
        write_note,
        paths.procedural_dir / "PROC_new.md",
        "proj-a",
        "New steps.",
        supersedes=f"[[{old.note_id}]]",
    )

    report = consolidate(paths, manifest, today=_OLD)

    reloaded_old = Note.load(old.path)
    assert reloaded_old.get("status") == STATUS_DEPRECATED
    assert reloaded_old.get("superseded_by") == f"[[{new.note_id}]]"
    assert Note.load(new.path).get("status") == STATUS_ACTIVE  # successor untouched
    assert old.path.exists()  # deprecated, not deleted
    assert len(report.deprecated) == 1
    assert report.deprecated[0].note_id == old.note_id
    assert report.deprecated[0].successors == (new.note_id,)


def test_superseded_semantic_is_not_deprecated(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=100_000)
    old = _sem(write_note, paths.semantic_dir / "2026-01-15_INS_old.md", "proj-a", "Old fact.")
    _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_new.md",
        "proj-a",
        "New fact.",
        supersedes=f"[[{old.note_id}]]",
    )

    report = consolidate(paths, manifest, today=_OLD)

    # deprecado is procedural-only (§6): a superseded semantic is the agent's
    # to archive at write time, not the consolidator's to deprecate.
    assert Note.load(old.path).get("status") == STATUS_ACTIVE
    assert report.deprecated == ()


def test_superseded_stale_procedural_deprecated_not_archived(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    old = _proc(
        write_note,
        paths.procedural_dir / "PROC_old.md",
        "proj-a",
        "Old steps.",
        last_accessed=_OLD,
    )
    _proc(
        write_note,
        paths.procedural_dir / "PROC_new.md",
        "proj-a",
        "New steps.",
        supersedes=f"[[{old.note_id}]]",
        last_accessed=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    # Stale AND superseded: the more informative deprecado wins over arquivado.
    assert Note.load(old.path).get("status") == STATUS_DEPRECATED
    assert len(report.deprecated) == 1
    assert all(d.note_id != old.note_id for d in report.decayed)


# ══════════════════════════════════════════════════════════════════════
# Step 4b -- report stale procedurals (§9.4)
# ══════════════════════════════════════════════════════════════════════
_RECENT = date(2026, 12, 30)


def test_stale_procedural_is_reported_never_touched(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    """Read often (no decay), tested long ago (stale): report it, change nothing."""
    paths, manifest = _vault(make_vault, manifest_data_ci, proc_stale_days=180)
    note = _proc(
        write_note,
        paths.procedural_dir / "PROC_aging.md",
        "proj-a",
        "Steps that may have rotted.",
        last_accessed=_RECENT,
        last_tested=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert Note.load(note.path).get("status") == STATUS_ACTIVE, "detect-and-report only"
    assert len(report.stale_procedurals) == 1
    stale = report.stale_procedurals[0]
    assert stale.note_id == note.note_id
    assert stale.last_tested == _OLD
    assert stale.days_stale == (_FUTURE - _OLD).days
    assert report.counts["stale_procedurals"] == 1


def test_freshly_tested_procedural_is_not_stale(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, proc_stale_days=180)
    _proc(
        write_note,
        paths.procedural_dir / "PROC_fresh.md",
        "proj-a",
        "Steps retested last month.",
        last_accessed=_RECENT,
        last_tested=date(2026, 12, 1),
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert report.stale_procedurals == ()


def test_stale_falls_back_to_created_when_never_tested(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    """A note predating `last_tested:` is aged by `created:` -- reading is not testing."""
    paths, manifest = _vault(make_vault, manifest_data_ci, proc_stale_days=180)
    _proc(
        write_note,
        paths.procedural_dir / "PROC_pre_field.md",
        "proj-a",
        "Steps from before the field existed.",
        created=_OLD,
        last_accessed=_RECENT,  # read often -- must not count as a test
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert len(report.stale_procedurals) == 1
    assert report.stale_procedurals[0].last_tested is None


def test_proc_stale_days_zero_disables_the_report(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, proc_stale_days=0)
    _proc(
        write_note,
        paths.procedural_dir / "PROC_aging.md",
        "proj-a",
        "Old steps, but the report is off.",
        last_accessed=_RECENT,
        last_tested=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert report.stale_procedurals == ()
    assert report.counts["stale_procedurals"] == 0


def test_deprecated_this_cycle_is_not_also_stale(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    """One report line per note: deprecado already says everything stale would."""
    paths, manifest = _vault(make_vault, manifest_data_ci, proc_stale_days=180)
    old = _proc(
        write_note,
        paths.procedural_dir / "PROC_old.md",
        "proj-a",
        "Old steps.",
        last_accessed=_RECENT,
        last_tested=_OLD,
    )
    _proc(
        write_note,
        paths.procedural_dir / "PROC_new.md",
        "proj-a",
        "New steps.",
        supersedes=f"[[{old.note_id}]]",
        last_accessed=_RECENT,
        last_tested=_RECENT,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert len(report.deprecated) == 1
    assert all(s.note_id != old.note_id for s in report.stale_procedurals)


# ══════════════════════════════════════════════════════════════════════
# Step 5 -- rebuild (§9.5, R8.1)
# ══════════════════════════════════════════════════════════════════════
def test_rebuild_indexes_and_regenerates_surfaces(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    active = _sem(
        write_note,
        paths.semantic_dir / "2026-12-30_INS_active.md",
        "proj-a",
        "A live fact worth indexing.",
        last_accessed=date(2026, 12, 30),
    )
    stale = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "A fact nobody reads.",
        last_accessed=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    # Both are indexed (archived stays searchable, §9.3), so chunk count > 0.
    store = store_from_manifest(manifest, paths)
    assert store.count() == report.counts["chunks_indexed"] > 0

    index_md = paths.index_file.read_text(encoding="utf-8")
    assert active.note_id in index_md          # active memory is mapped
    assert stale.note_id not in index_md       # archived drops out of the MOC
    assert paths.hot_file.exists()


def test_rebuild_reflects_freshly_archived_status_in_index_meta(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    stale = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Archived this very cycle.",
        last_accessed=_OLD,
    )

    consolidate(paths, manifest, today=_FUTURE)

    # The status the rebuild indexed is the one decay just wrote: the archived
    # note is still in the numpy store's metadata, carrying status: arquivado
    # (searchable behind an explicit filter, §9.3).
    metas = (paths.index_dir / "vectors.jsonl").read_text(encoding="utf-8")
    assert stale.note_id in metas
    assert STATUS_ARCHIVED in metas


def test_crash_mid_rebuild_never_destroys_the_index(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§9.5/R7.5: a consolidation killed mid-reindex leaves the old index usable.

    The regression that motivated the atomic rebuild: consolidate used to open
    the store with autosave on, ``clear()`` truncated ``vectors.npy`` /
    ``vectors.jsonl`` on disk immediately, and only then re-embedded -- so the
    OOM a long embedding pass can hit left the vector store empty (0 bytes),
    worse than not consolidating at all.
    """
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    _sem(
        write_note,
        paths.semantic_dir / "2026-12-30_INS_alive.md",
        "proj-a",
        "A fact that must survive a crashed maintenance run.",
        last_accessed=date(2026, 12, 30),
    )
    consolidate(paths, manifest, today=date(2026, 12, 30))
    chunks_before = store_from_manifest(manifest, paths).count()
    assert chunks_before > 0

    # Give the next cycle indexing work, then kill it inside the write pass --
    # after the clear, before any new chunk lands.
    _sem(
        write_note,
        paths.semantic_dir / "2026-12-30_INS_newcomer.md",
        "proj-a",
        "A brand new fact whose indexing dies halfway through.",
        last_accessed=date(2026, 12, 30),
    )

    def boom(self: Indexer, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated OOM during the embedding pass")

    monkeypatch.setattr(Indexer, "_write_targets", boom)
    with pytest.raises(RuntimeError):
        consolidate(paths, manifest, today=date(2026, 12, 30))

    # A fresh process after the crash still sees the whole previous index.
    assert store_from_manifest(manifest, paths).count() == chunks_before
    with LexicalIndex(paths.index_dir / LEXICAL_DB_FILENAME) as lexical:
        assert "2026-12-30_INS_alive" in lexical.note_ids()
    assert (paths.index_dir / "vectors.jsonl").stat().st_size > 0


class _SpyProvider:
    """The real provider behind a counter: records every embed batch size."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.batches: list[int] = []

    def embed_texts(self, texts: Any) -> Any:
        items = list(texts)
        self.batches.append(len(items))
        return self._inner.embed_texts(items)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_consolidate_reembeds_only_what_the_cycle_changed(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§9.5: the rebuild is incremental -- one decayed note, one note re-embedded.

    The regression: consolidate used to re-chunk and re-embed the entire corpus
    every cycle (a >30-minute pass over a few hundred notes with a real model).
    The mtime+hash diff must confine the re-embed to the notes the cycle
    actually changed; the document-level dedup pass is the only full sweep left.
    """
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    stale = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "A fact nobody reads any more, about an abandoned migration.",
        last_accessed=_OLD,
    )
    for ordinal, topic in enumerate(("indexing", "locking", "chunking")):
        _sem(
            write_note,
            paths.semantic_dir / f"2026-12-30_INS_fresh-{ordinal}.md",
            "proj-a",
            f"A live, regularly read fact about {topic} internals.",
            title=f"Fact about {topic}",
            last_accessed=date(2026, 12, 30),
        )

    spy = _SpyProvider(embedder_from_manifest(manifest))
    monkeypatch.setattr("mechabrain.consolidate.embedder_from_manifest", lambda _m: spy)

    first = consolidate(paths, manifest, today=_OLD)  # no state yet: a full build
    assert first.counts["notes_reindexed"] == 4

    spy.batches.clear()
    second = consolidate(paths, manifest, today=_FUTURE)  # decays only the stale note

    assert [d.note_id for d in second.decayed] == [stale.note_id]
    assert second.counts["notes_reindexed"] == 1
    assert 0 < second.counts["chunks_indexed"] < first.counts["chunks_indexed"]
    # Everything embedded this cycle: one dedup pass over the 4 active documents
    # plus the decayed note's chunks. Never the whole corpus again.
    assert sum(spy.batches) == 4 + second.counts["chunks_indexed"]


# ══════════════════════════════════════════════════════════════════════
# Step 6 -- commit (§9.6)
# ══════════════════════════════════════════════════════════════════════
def test_commit_single_and_excludes_meta_index(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Will be archived and committed.",
        last_accessed=_OLD,
    )
    _git_init(paths.root)
    before = _commit_count(paths.root)

    report = consolidate(paths, manifest, today=_FUTURE, commit=True)

    assert report.committed is True
    assert report.commit is not None
    assert _commit_count(paths.root) == before + 1

    message = subprocess.run(
        ["git", "-C", str(paths.root), "log", "-1", "--pretty=%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert message.startswith(manifest.maintenance.commit_prefix)

    tracked = subprocess.run(
        ["git", "-C", str(paths.root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "_meta/index" not in tracked  # derived, per-machine, never committed


def test_no_commit_flag_writes_but_does_not_commit(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    note = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Archived but uncommitted.",
        last_accessed=_OLD,
    )
    _git_init(paths.root)
    before = _commit_count(paths.root)

    report = consolidate(paths, manifest, today=_FUTURE, commit=False)

    assert report.committed is False
    assert _commit_count(paths.root) == before
    assert Note.load(note.path).get("status") == STATUS_ARCHIVED  # write still happened


def test_non_git_vault_runs_without_committing(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "No git here.",
        last_accessed=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    assert report.committed is False
    assert report.commit is None


def test_idempotent_second_run_makes_no_commit(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Archived once.",
        last_accessed=_OLD,
    )
    _git_init(paths.root)

    first = consolidate(paths, manifest, today=_FUTURE, commit=True)
    after_first = _commit_count(paths.root)
    second = consolidate(paths, manifest, today=_FUTURE, commit=True)

    assert first.committed is True
    assert second.committed is False
    assert _commit_count(paths.root) == after_first  # nothing changed, no commit


# ══════════════════════════════════════════════════════════════════════
# Dry run and report persistence
# ══════════════════════════════════════════════════════════════════════
def test_dry_run_changes_nothing_on_disk(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    note = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Would be archived, but this is a preview.",
        last_accessed=_OLD,
    )
    AccessLog.for_vault(paths).record(note.note_id, AccessKind.SEARCH)
    original = note.path.read_text(encoding="utf-8")

    report = consolidate(paths, manifest, today=_FUTURE, dry_run=True)

    # The report says what WOULD happen...
    assert report.dry_run is True
    assert len(report.decayed) == 1
    # ...but the note is untouched, the access log is not consumed, and no
    # report file is written.
    assert note.path.read_text(encoding="utf-8") == original
    assert AccessLog.for_vault(paths).aggregate() != {}
    assert not (paths.index_dir / CONSOLIDATION_REPORT_FILE).exists()


def test_report_is_persisted_and_parseable(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Archived this cycle.",
        last_accessed=_OLD,
    )

    report = consolidate(paths, manifest, today=_FUTURE)

    report_path = paths.index_dir / CONSOLIDATION_REPORT_FILE
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["counts"]["decayed"] == 1
    assert payload["dry_run"] is False
    assert isinstance(report, ConsolidationReport)


def test_never_deletes_any_note_across_the_pipeline(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci, decay_days=90)
    stale = _sem(
        write_note,
        paths.semantic_dir / "2026-01-15_INS_stale.md",
        "proj-a",
        "Stale.",
        last_accessed=_OLD,
    )
    old_proc = _proc(
        write_note,
        paths.procedural_dir / "PROC_old.md",
        "proj-a",
        "Old steps.",
        last_accessed=_OLD,
    )
    new_proc = _proc(
        write_note,
        paths.procedural_dir / "PROC_new.md",
        "proj-a",
        "New steps.",
        supersedes=f"[[{old_proc.note_id}]]",
        last_accessed=_OLD,
    )

    consolidate(paths, manifest, today=_FUTURE)

    # P8: consolidation archives and deprecates -- it never removes a file.
    for note in (stale, old_proc, new_proc):
        assert note.path.exists()


def test_empty_vault_consolidates_cleanly(
    make_vault: Callable[..., VaultPaths],
    manifest_data_ci: dict[str, Any],
) -> None:
    paths, manifest = _vault(make_vault, manifest_data_ci)

    report = consolidate(paths, manifest, today=_FUTURE)

    assert report.counts["notes_scanned"] == 0
    assert report.counts["chunks_indexed"] == 0
    assert report.decayed == ()
    assert paths.index_file.exists()  # surfaces are still (re)generated


# ══════════════════════════════════════════════════════════════════════
# Step 4c -- validate read-only docs (Mecha-Scribe Fase 2)
# ══════════════════════════════════════════════════════════════════════
def test_reports_docs_citing_dead_memories(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    """A read-only doc that cites an archived/superseded memory is reported --
    detect-and-report, never touched (P4)."""
    paths, manifest = _vault(
        make_vault, manifest_data_ci, decay_days=100_000, read_only_index=["Notes/"]
    )
    # An archived memory that was superseded by a live successor.
    write_note(
        paths.semantic_dir / "2026-01-15_INS_dead.md",
        {
            "title": "Superseded fact",
            "tags": ["mem/semantic", "agent/alpha"],
            "created": _OLD,
            "modified": _OLD,
            "agent": "alpha",
            "scope": "proj-a",
            "source": "s",
            "confidence": "medium",
            "status": STATUS_ARCHIVED,
            "superseded_by": "[[2026-06-01_INS_live]]",
        },
        "Old body.",
    )
    _sem(
        write_note,
        paths.semantic_dir / "2026-06-01_INS_live.md",
        "proj-a",
        "New body.",
        title="Live fact",
        created=date(2026, 6, 1),
    )
    # A read-only project doc citing the dead memory, plus a wikilink to a note
    # outside the indexed scope (a root note / attachment): that must NOT be
    # flagged -- only the dead-memory citation is a reliable signal.
    notes_dir = paths.root / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    write_note(
        notes_dir / "2026-06-02_DOC_guide.md",
        {"title": "Guide", "created": date(2026, 6, 2), "modified": date(2026, 6, 2)},
        "Follow [[2026-01-15_INS_dead]]; see the dashboard [[GES_DASHBOARD_CENTRAL]].",
    )

    report = consolidate(paths, manifest, today=_FUTURE, commit=False)

    assert report.counts["docs_citing_dead"] == 1
    dead = report.docs_citing_dead[0]
    assert dead.doc == "2026-06-02_DOC_guide"
    assert dead.cited == "2026-01-15_INS_dead"
    assert dead.status == "superseded"
    assert dead.successor == "2026-06-01_INS_live"

    # No broken-link report exists: a link out of scope is not a defect.
    assert "doc_broken_links" not in report.counts

    # P4: the doc is content, never rewritten by the kernel.
    assert (notes_dir / "2026-06-02_DOC_guide.md").read_text(encoding="utf-8").count("[[") == 2


def test_live_memory_citation_is_not_reported(
    make_vault: Callable[..., VaultPaths],
    write_note: Callable[..., Note],
    manifest_data_ci: dict[str, Any],
) -> None:
    """A doc citing a healthy, active memory is silent -- no false positive."""
    paths, manifest = _vault(
        make_vault, manifest_data_ci, decay_days=100_000, read_only_index=["Notes/"]
    )
    _sem(write_note, paths.semantic_dir / "2026-01-15_INS_ok.md", "proj-a", "Body.")
    notes_dir = paths.root / "Notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    write_note(
        notes_dir / "2026-06-02_DOC_guide.md",
        {"title": "Guide", "created": date(2026, 6, 2), "modified": date(2026, 6, 2)},
        "Follow the live [[2026-01-15_INS_ok]].",
    )

    report = consolidate(paths, manifest, today=_FUTURE, commit=False)

    assert report.counts["docs_citing_dead"] == 0
