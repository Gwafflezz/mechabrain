"""Tests for the index orchestrator (`mechabrain.index.indexer`).

Everything runs on the deterministic, offline stack the suite is built for:
``embedding.provider: hash`` and ``store: numpy`` (see ``manifest_ci``), so the
pipeline is exercised end to end with no model download and no network.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from mechabrain.discovery import VaultPaths
from mechabrain.errors import MechabrainIndexError, NoteNotFound
from mechabrain.index.embed import from_manifest as embed_from_manifest
from mechabrain.index.indexer import STATE_FILE, IndexReport, Indexer
from mechabrain.index.lexical import LEXICAL_DB_FILENAME, LexicalIndex
from mechabrain.index.store import INDEX_LOCK_FILE, NumpyStore
from mechabrain.locking import file_lock
from mechabrain.manifest import Manifest
from mechabrain.note import Note

# ══════════════════════════════════════════════════════════════════════
# Fixtures & helpers
# ══════════════════════════════════════════════════════════════════════
VECTOR_NOTE = "2026-01-15_INS_vector-search"
GLOBAL_NOTE = "2026-01-15_INS_global-fact"
STALE_NOTE = "2026-01-15_INS_stale-fact"


@pytest.fixture
def indexer(tmp_vault: VaultPaths, manifest_ci: Manifest) -> Indexer:
    """An indexer over an empty, initialized vault."""
    return Indexer(tmp_vault, manifest_ci)


def open_store(paths: VaultPaths) -> NumpyStore:
    return NumpyStore(paths.index_dir)


def lexical_ids(paths: VaultPaths) -> set[str]:
    with LexicalIndex(paths.index_dir / LEXICAL_DB_FILENAME) as index:
        return index.note_ids()


def lexical_count(paths: VaultPaths) -> int:
    with LexicalIndex(paths.index_dir / LEXICAL_DB_FILENAME) as index:
        return index.count()


def query_vector(manifest: Manifest, text: str) -> Any:
    return embed_from_manifest(manifest).embed_texts([text])[0]


def read_state(paths: VaultPaths) -> dict[str, Any]:
    return json.loads((paths.index_dir / STATE_FILE).read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════
# Full rebuild
# ══════════════════════════════════════════════════════════════════════
def test_full_rebuild_indexes_every_sample_note(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    report = indexer.reindex(full=True)

    assert report.full is True
    assert report.notes_indexed == len(sample_notes)
    assert report.chunks_written > 0
    # The two indexes must agree on how many chunks exist.
    assert open_store(tmp_vault).count() == report.chunks_written
    assert lexical_count(tmp_vault) == report.chunks_written
    assert report.chunks_total == report.chunks_written


def test_full_rebuild_writes_a_state_manifest(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)
    state = read_state(tmp_vault)

    assert state["fingerprint"]["store"] == "numpy"
    assert state["fingerprint"]["embedding"].startswith("hash:")
    assert len(state["notes"]) == len(sample_notes)
    # Every record carries what an incremental diff needs.
    record = next(iter(state["notes"].values()))
    assert set(record) == {"note_id", "mtime_ns", "hash", "chunks", "read_only"}


def test_empty_vault_rebuild_is_clean_and_offline(indexer: Indexer, tmp_vault: VaultPaths) -> None:
    report = indexer.reindex(full=True)

    assert report.chunks_written == 0
    assert report.notes_indexed == 0
    assert open_store(tmp_vault).count() == 0
    assert read_state(tmp_vault)["notes"] == {}


def test_full_rebuild_replaces_deleted_notes(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)
    (tmp_vault.semantic_dir / f"{VECTOR_NOTE}.md").unlink()

    report = indexer.reindex(full=True)

    assert VECTOR_NOTE not in {rec["note_id"] for rec in read_state(tmp_vault)["notes"].values()}
    assert VECTOR_NOTE not in lexical_ids(tmp_vault)
    assert report.notes_indexed == len(sample_notes) - 1


def test_crash_mid_rebuild_preserves_the_previous_index(
    indexer: Indexer,
    tmp_vault: VaultPaths,
    manifest_ci: Manifest,
    sample_notes: list[Note],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild that dies after the clear must not destroy the old index (R7.5).

    The regression behind the atomic rebuild: the lexical ``clear()`` used to
    commit on its own before the embedding pass, so a process killed mid-embed
    (an OOM) left the BM25 index empty -- a maintenance job destroying the very
    index it meant to refresh. Both halves must survive: the transaction rolls
    the lexical clear back, and the numpy store never flushed.
    """
    indexer.reindex(full=True)
    chunks_before = open_store(tmp_vault).count()
    ids_before = lexical_ids(tmp_vault)
    assert chunks_before > 0

    def boom(self: Indexer, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated OOM during the embedding pass")

    monkeypatch.setattr(Indexer, "_write_targets", boom)
    with pytest.raises(RuntimeError):
        indexer.reindex(full=True)

    # Fresh handles: what a process restarted after the crash sees on disk.
    assert open_store(tmp_vault).count() == chunks_before
    assert lexical_ids(tmp_vault) == ids_before
    query = query_vector(manifest_ci, "brute-force cosine is fast enough")
    assert open_store(tmp_vault).search(query, k=1), "old index no longer searchable"


# ══════════════════════════════════════════════════════════════════════
# Retrieval sanity: the pipeline actually produces a searchable index
# ══════════════════════════════════════════════════════════════════════
def test_indexed_chunks_are_retrievable_by_vector_search(
    indexer: Indexer, tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    query = query_vector(manifest_ci, "brute-force cosine is fast enough below ten thousand chunks")
    hits = open_store(tmp_vault).search(query, k=1)

    assert hits, "expected at least one hit"
    top_id, _score = hits[0]
    assert top_id.startswith(VECTOR_NOTE)


def test_vector_store_honours_scope_filter(
    indexer: Indexer, tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    query = query_vector(manifest_ci, "markdown source of truth derived index")
    hits = open_store(tmp_vault).search(query, k=8, filters={"scope": "global"})

    assert hits
    assert all(chunk_id.startswith(GLOBAL_NOTE) for chunk_id, _ in hits)


def test_lexical_search_honours_filters(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    with LexicalIndex(tmp_vault.index_dir / LEXICAL_DB_FILENAME) as index:
        hits = index.search("fact", k=8, filters={"scope": "proj-b"})

    assert hits
    assert all(chunk_id.startswith(STALE_NOTE) for chunk_id, _ in hits)


def test_type_filter_only_matches_that_type(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    with LexicalIndex(tmp_vault.index_dir / LEXICAL_DB_FILENAME) as index:
        procedural = index.search("build ship steps", k=8, filters={"type": "procedural"})
        semantic = index.search("build ship steps", k=8, filters={"type": "semantic"})

    assert procedural
    assert all(chunk_id.startswith("PROC_deploy-playbook") for chunk_id, _ in procedural)
    assert not any(chunk_id.startswith("PROC_deploy-playbook") for chunk_id, _ in semantic)


# ══════════════════════════════════════════════════════════════════════
# Incremental reindex
# ══════════════════════════════════════════════════════════════════════
def test_incremental_noop_touches_nothing(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    report = indexer.reindex()

    assert report.full is False
    assert report.notes_indexed == 0
    assert report.notes_removed == 0
    assert report.notes_unchanged == len(sample_notes)
    assert report.chunks_written == 0
    assert report.chunks_deleted == 0


def test_incremental_reindexes_a_changed_note_only(
    indexer: Indexer,
    tmp_vault: VaultPaths,
    write_note: Callable[..., Note],
    sample_notes: list[Note],
) -> None:
    indexer.reindex(full=True)
    original = Note.load(tmp_vault.semantic_dir / f"{VECTOR_NOTE}.md")
    write_note(
        tmp_vault.semantic_dir / f"{VECTOR_NOTE}.md",
        original.frontmatter,
        "A completely rewritten body about quantum widgets and flux capacitors.",
    )

    report = indexer.reindex()

    assert report.notes_indexed == 1
    assert report.notes_unchanged == len(sample_notes) - 1
    # The new term is now findable; the store and lexical index stay consistent.
    with LexicalIndex(tmp_vault.index_dir / LEXICAL_DB_FILENAME) as index:
        hits = index.search("flux capacitors", k=5)
    assert any(chunk_id.startswith(VECTOR_NOTE) for chunk_id, _ in hits)
    assert open_store(tmp_vault).count() == lexical_count(tmp_vault)


def test_incremental_drops_a_deleted_note(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)
    before = open_store(tmp_vault).count()
    (tmp_vault.semantic_dir / f"{VECTOR_NOTE}.md").unlink()

    report = indexer.reindex()

    assert report.notes_removed == 1
    assert report.chunks_deleted > 0
    assert VECTOR_NOTE not in lexical_ids(tmp_vault)
    assert open_store(tmp_vault).count() == before - report.chunks_deleted
    assert VECTOR_NOTE not in {rec["note_id"] for rec in read_state(tmp_vault)["notes"].values()}


def test_incremental_skips_a_touched_but_unchanged_note(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)
    note_path = tmp_vault.semantic_dir / f"{VECTOR_NOTE}.md"
    # Rewrite identical bytes: mtime moves, content hash does not.
    note_path.write_bytes(note_path.read_bytes())

    report = indexer.reindex()

    assert report.notes_indexed == 0
    assert report.notes_unchanged == len(sample_notes)


# ══════════════════════════════════════════════════════════════════════
# Single-note operations
# ══════════════════════════════════════════════════════════════════════
def test_index_note_adds_one_new_note(
    indexer: Indexer,
    tmp_vault: VaultPaths,
    write_note: Callable[..., Note],
    sample_notes: list[Note],
) -> None:
    indexer.reindex(full=True)
    before = open_store(tmp_vault).count()
    new_path = tmp_vault.semantic_dir / "2026-01-15_INS_fresh.md"
    write_note(
        new_path,
        {"title": "Fresh", "tags": ["mem/semantic", "agent/alpha"], "scope": "proj-a", "agent": "alpha"},
        "A brand new insight about caching strategies.",
    )

    report = indexer.index_note(new_path)

    assert report.notes_indexed == 1
    assert "2026-01-15_INS_fresh" in lexical_ids(tmp_vault)
    assert open_store(tmp_vault).count() == before + report.chunks_written
    assert report.notes_total == len(sample_notes) + 1


def test_index_note_replaces_an_edited_note(
    indexer: Indexer,
    tmp_vault: VaultPaths,
    write_note: Callable[..., Note],
    sample_notes: list[Note],
) -> None:
    indexer.reindex(full=True)
    note_path = tmp_vault.semantic_dir / f"{VECTOR_NOTE}.md"
    original = Note.load(note_path)
    write_note(note_path, original.frontmatter, "Rewritten to mention holographic storage.")

    indexer.index_note(note_path)

    with LexicalIndex(tmp_vault.index_dir / LEXICAL_DB_FILENAME) as index:
        assert index.search("holographic storage", k=5)
    assert open_store(tmp_vault).count() == lexical_count(tmp_vault)


def test_index_note_requires_an_existing_file(indexer: Indexer, tmp_vault: VaultPaths) -> None:
    with pytest.raises(NoteNotFound):
        indexer.index_note(tmp_vault.semantic_dir / "does-not-exist.md")


def test_index_note_accepts_a_placed_note(
    indexer: Indexer,
    tmp_vault: VaultPaths,
    write_note: Callable[..., Note],
    sample_notes: list[Note],
) -> None:
    # The writer injects `index_note(note)` with a placed Note, not a path.
    indexer.reindex(full=True)
    note = write_note(
        tmp_vault.semantic_dir / "2026-01-15_INS_from-note.md",
        {"title": "From note", "tags": ["mem/semantic", "agent/alpha"], "scope": "proj-a", "agent": "alpha"},
        "Indexed straight from a Note object.",
    )

    report = indexer.index_note(note)

    assert report.notes_indexed == 1
    assert "2026-01-15_INS_from-note" in lexical_ids(tmp_vault)


def test_remove_note_drops_chunks(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    report = indexer.remove_note(VECTOR_NOTE)

    assert report.notes_removed == 1
    assert report.chunks_deleted > 0
    assert VECTOR_NOTE not in lexical_ids(tmp_vault)
    assert VECTOR_NOTE not in {rec["note_id"] for rec in read_state(tmp_vault)["notes"].values()}


def test_remove_note_accepts_a_wikilink(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)

    indexer.remove_note(f"[[{GLOBAL_NOTE}]]")

    assert GLOBAL_NOTE not in lexical_ids(tmp_vault)


# ══════════════════════════════════════════════════════════════════════
# read_only_index context
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def read_only_vault(
    make_vault: Callable[..., VaultPaths],
    manifest_data_ci: dict[str, Any],
    write_note: Callable[..., Note],
) -> tuple[VaultPaths, Manifest]:
    """A vault with a human ``Notes/`` folder indexed read-only."""
    data = copy.deepcopy(manifest_data_ci)
    data["zones"]["read_only_index"] = ["Notes/"]
    paths = make_vault(manifest_data=data, name="ro")
    # A memory note (writable) and a human context note (read-only).
    write_note(
        paths.semantic_dir / "2026-01-15_INS_owned.md",
        {"title": "Owned", "tags": ["mem/semantic", "agent/alpha"], "scope": "proj-a", "agent": "alpha"},
        "An owned memory about pipelines.",
    )
    write_note(
        paths.root / "Notes" / "human-note.md",
        {"title": "Human note"},
        "A human wrote this about pipelines and it is only context.",
    )
    return paths, Manifest.from_mapping(data)


def test_read_only_notes_are_indexed_and_flagged(
    read_only_vault: tuple[VaultPaths, Manifest],
) -> None:
    paths, manifest = read_only_vault
    indexer = Indexer(paths, manifest)

    report = indexer.reindex(full=True)

    assert report.read_only_indexed == 1
    assert "human-note" in lexical_ids(paths)
    assert indexer.read_only_note_ids() == frozenset({"human-note"})
    # The owned memory note is not flagged read-only.
    assert "2026-01-15_INS_owned" not in indexer.read_only_note_ids()


def test_read_only_flag_survives_in_state(
    read_only_vault: tuple[VaultPaths, Manifest],
) -> None:
    paths, manifest = read_only_vault
    Indexer(paths, manifest).reindex(full=True)

    records = read_state(paths)["notes"]
    flags = {rec["note_id"]: rec["read_only"] for rec in records.values()}
    assert flags["human-note"] is True
    assert flags["2026-01-15_INS_owned"] is False


# ══════════════════════════════════════════════════════════════════════
# Basename collisions
# ══════════════════════════════════════════════════════════════════════
def test_colliding_note_ids_report_the_ambiguity(
    make_vault: Callable[..., VaultPaths],
    manifest_data_ci: dict[str, Any],
    write_note: Callable[..., Note],
) -> None:
    data = copy.deepcopy(manifest_data_ci)
    data["zones"]["read_only_index"] = ["Notes/"]
    paths = make_vault(manifest_data=data, name="dup")
    manifest = Manifest.from_mapping(data)
    # Same basename in two indexed folders -> one id, two files.
    write_note(
        paths.semantic_dir / "clash.md",
        {"title": "Clash", "tags": ["mem/semantic", "agent/alpha"], "scope": "proj-a", "agent": "alpha"},
        "The memory copy.",
    )
    write_note(paths.root / "Notes" / "clash.md", {"title": "Clash"}, "The human copy.")

    report = Indexer(paths, manifest).reindex(full=True)

    assert "clash" in report.ambiguous_ids
    # Only one note won the id, so it is tracked once.
    tracked = [rec for rec in read_state(paths)["notes"].values() if rec["note_id"] == "clash"]
    assert len(tracked) == 1


# ══════════════════════════════════════════════════════════════════════
# Fingerprint forces a full rebuild
# ══════════════════════════════════════════════════════════════════════
def test_config_change_forces_a_full_rebuild(
    tmp_vault: VaultPaths, manifest_data_ci: dict[str, Any], sample_notes: list[Note]
) -> None:
    Indexer(tmp_vault, Manifest.from_mapping(manifest_data_ci)).reindex(full=True)

    flipped = copy.deepcopy(manifest_data_ci)
    flipped["retrieval"]["contextual_retrieval"] = False
    report = Indexer(tmp_vault, Manifest.from_mapping(flipped)).reindex()

    assert report.full is True
    assert report.reason is not None
    assert report.notes_indexed == len(sample_notes)


def test_missing_state_forces_a_full_rebuild(
    indexer: Indexer, tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    indexer.reindex(full=True)
    (tmp_vault.index_dir / STATE_FILE).unlink()

    report = indexer.reindex()

    assert report.full is True
    assert report.reason is not None


# ══════════════════════════════════════════════════════════════════════
# Locking and progress
# ══════════════════════════════════════════════════════════════════════
def test_reindex_fails_when_the_index_lock_is_held(
    tmp_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    held = Indexer(tmp_vault, manifest_ci, lock_timeout=0.0)
    with file_lock(tmp_vault.index_dir / INDEX_LOCK_FILE, purpose="test-holder"):
        with pytest.raises(MechabrainIndexError):
            held.reindex(full=True)


def test_progress_callback_receives_lines(
    tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    messages: list[str] = []
    Indexer(tmp_vault, manifest_ci, progress=messages.append).reindex(full=True)

    assert messages
    assert any("indexed" in message for message in messages)


def test_report_is_an_index_report(indexer: Indexer, sample_notes: list[Note]) -> None:
    assert isinstance(indexer.reindex(full=True), IndexReport)
