"""Tests for the pluggable vector store (`mechabrain.index.store`).

``NumpyStore`` is exercised in full: it is the default every deployment gets and
the only backend the suite can rely on being installed. The optional stores are
covered by contract-shaped tests that skip when their extra is absent -- pytest
reports the skip, so a machine that *does* have them runs the same assertions.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mechabrain.discovery import VaultPaths
from mechabrain.errors import MechabrainIndexError
from mechabrain.index.store import (
    LANCEDB_DIR,
    METAS_FILE,
    SQLITE_VEC_FILE,
    VECTORS_FILE,
    LanceDBStore,
    NumpyStore,
    SqliteVecStore,
    VectorStore,
    from_manifest,
    matches_filters,
    open_store,
    require_index_dir,
)
from mechabrain.manifest import Manifest

# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def store(tmp_vault: VaultPaths) -> NumpyStore:
    """An empty default store over a real vault's index dir."""
    return NumpyStore(tmp_vault.index_dir)


def unit(*values: float) -> list[float]:
    return list(values)


#: Four 2-D vectors on the unit circle, so every cosine is known by hand.
EAST = unit(1.0, 0.0)
NORTH = unit(0.0, 1.0)
WEST = unit(-1.0, 0.0)
NORTHEAST = unit(1.0, 1.0)


def fill(store: NumpyStore) -> None:
    """Three rows spanning two scopes and two types, for filter tests."""
    store.upsert(
        ["a", "b", "c"],
        [EAST, NORTH, WEST],
        [
            {"scope": "proj-a", "type": "semantic", "tags": ["mem/semantic", "agent/alpha"]},
            {"scope": "proj-b", "type": "semantic", "tags": ["mem/semantic", "agent/beta"]},
            {"scope": "proj-a", "type": "episodic", "tags": ["mem/episodic", "agent/alpha"]},
        ],
    )


# ══════════════════════════════════════════════════════════════════════
# Protocol and factory
# ══════════════════════════════════════════════════════════════════════
def test_every_store_satisfies_the_protocol(tmp_vault: VaultPaths) -> None:
    for cls in (NumpyStore, LanceDBStore, SqliteVecStore):
        assert isinstance(cls(tmp_vault.index_dir), VectorStore), cls.__name__


def test_from_manifest_builds_the_store_the_manifest_names(
    manifest_ci: Manifest, tmp_vault: VaultPaths
) -> None:
    built = from_manifest(manifest_ci, tmp_vault)
    assert isinstance(built, NumpyStore)
    assert built.index_dir == tmp_vault.index_dir


def test_from_manifest_honours_a_non_default_store(
    manifest_data_ci: dict[str, Any], tmp_vault: VaultPaths
) -> None:
    manifest_data_ci["retrieval"]["store"] = "sqlite-vec"
    built = from_manifest(Manifest.from_mapping(manifest_data_ci), tmp_vault)
    assert isinstance(built, SqliteVecStore)


def test_open_store_rejects_an_unknown_name(tmp_vault: VaultPaths) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        open_store("faiss", tmp_vault.index_dir)
    assert "unknown vector store 'faiss'" in str(excinfo.value)
    assert "numpy" in str(excinfo.value)


def test_missing_extra_names_the_dependency_and_the_way_out(tmp_vault: VaultPaths) -> None:
    for name, module in (("lancedb", "lancedb"), ("sqlite-vec", "sqlite_vec")):
        try:
            __import__(module)
        except ImportError:
            pass
        else:
            pytest.skip(f"{module} is installed; the missing-extra path cannot be exercised")
        with pytest.raises(MechabrainIndexError) as excinfo:
            open_store(name, tmp_vault.index_dir).count()
        message = str(excinfo.value)
        assert module in message
        assert f"mechabrain[{name}]" in message
        assert "retrieval.store: numpy" in message


# ══════════════════════════════════════════════════════════════════════
# Location: always inside _meta/index/ (§4)
# ══════════════════════════════════════════════════════════════════════
def test_a_store_refuses_to_live_outside_the_index_dir(tmp_path: Path) -> None:
    for outside in (tmp_path, tmp_path / "index", tmp_path / "_meta"):
        with pytest.raises(MechabrainIndexError) as excinfo:
            NumpyStore(outside)
        assert "_meta/index/" in str(excinfo.value)


def test_require_index_dir_accepts_the_vaults_index_dir(tmp_vault: VaultPaths) -> None:
    assert require_index_dir(tmp_vault.index_dir) == tmp_vault.index_dir


def test_require_index_dir_does_not_demand_existence(tmp_path: Path) -> None:
    candidate = tmp_path / "nowhere" / "_meta" / "index"
    assert require_index_dir(candidate) == candidate


def test_state_lands_only_in_the_index_dir(store: NumpyStore, tmp_vault: VaultPaths) -> None:
    fill(store)
    assert (tmp_vault.index_dir / VECTORS_FILE).is_file()
    assert (tmp_vault.index_dir / METAS_FILE).is_file()
    outside = [
        path
        for path in tmp_vault.mecha_brain.rglob("*")
        if path.is_file() and tmp_vault.index_dir not in path.parents and path != tmp_vault.config_file
    ]
    assert outside == []


# ══════════════════════════════════════════════════════════════════════
# NumpyStore: basics
# ══════════════════════════════════════════════════════════════════════
def test_an_empty_store_counts_zero_and_finds_nothing(store: NumpyStore) -> None:
    assert store.count() == 0
    assert store.search(EAST, k=8) == []


def test_upsert_reports_rows_written_and_count_follows(store: NumpyStore) -> None:
    assert store.upsert(["a", "b"], [EAST, NORTH], [{}, {}]) == 2
    assert store.count() == 2


def test_upsert_of_nothing_is_a_no_op(store: NumpyStore) -> None:
    assert store.upsert([], np.zeros((0, 2)), []) == 0
    assert store.count() == 0


def test_search_ranks_by_cosine_and_returns_the_similarity(store: NumpyStore) -> None:
    fill(store)
    hits = store.search(EAST, k=3)
    assert [hit_id for hit_id, _ in hits] == ["a", "b", "c"]
    scores = [score for _, score in hits]
    assert scores[0] == pytest.approx(1.0)   # identical
    assert scores[1] == pytest.approx(0.0)   # orthogonal
    assert scores[2] == pytest.approx(-1.0)  # opposite
    assert all(isinstance(score, float) for score in scores)


def test_magnitude_never_changes_a_score(store: NumpyStore) -> None:
    store.upsert(["long"], [[300.0, 0.0]], [{}])
    assert store.search([0.001, 0.0], k=1)[0][1] == pytest.approx(1.0)


def test_search_truncates_to_k(store: NumpyStore) -> None:
    fill(store)
    assert len(store.search(EAST, k=2)) == 2


def test_k_beyond_the_row_count_returns_every_row(store: NumpyStore) -> None:
    fill(store)
    assert len(store.search(EAST, k=99)) == 3


def test_a_non_positive_k_returns_nothing(store: NumpyStore) -> None:
    fill(store)
    assert store.search(EAST, k=0) == []
    assert store.search(EAST, k=-1) == []


def test_ties_break_on_id_so_two_machines_agree(store: NumpyStore) -> None:
    store.upsert(["zulu", "alpha", "mike"], [EAST, EAST, EAST], [{}, {}, {}])
    assert [hit_id for hit_id, _ in store.search(EAST, k=3)] == ["alpha", "mike", "zulu"]


def test_a_zero_vector_scores_zero_rather_than_nan(store: NumpyStore) -> None:
    store.upsert(["zero"], [[0.0, 0.0]], [{}])
    score = store.search(EAST, k=1)[0][1]
    assert score == 0.0
    assert not np.isnan(score)


def test_a_zero_query_scores_zero_rather_than_nan(store: NumpyStore) -> None:
    fill(store)
    assert all(score == 0.0 for _, score in store.search([0.0, 0.0], k=3))


def test_a_numpy_matrix_and_a_list_of_lists_are_equivalent(tmp_vault: VaultPaths) -> None:
    from_lists = NumpyStore(tmp_vault.index_dir)
    from_lists.upsert(["a"], [NORTHEAST], [{}])
    hit_from_lists = from_lists.search(NORTHEAST, k=1)

    from_lists.clear()
    from_lists.upsert(["a"], np.array([NORTHEAST], dtype=np.float64), [{}])
    assert from_lists.search(np.array(NORTHEAST), k=1) == hit_from_lists


# ══════════════════════════════════════════════════════════════════════
# NumpyStore: upsert / delete / clear semantics
# ══════════════════════════════════════════════════════════════════════
def test_re_upserting_an_id_replaces_it_rather_than_duplicating(store: NumpyStore) -> None:
    store.upsert(["a"], [EAST], [{"scope": "proj-a"}])
    store.upsert(["a"], [NORTH], [{"scope": "proj-b"}])
    assert store.count() == 1
    assert store.search(NORTH, k=1)[0] == ("a", pytest.approx(1.0))
    assert store.search(NORTH, k=1, filters={"scope": "proj-a"}) == []
    assert len(store.search(NORTH, k=1, filters={"scope": "proj-b"})) == 1


def test_an_id_repeated_inside_one_batch_collapses_to_the_last(store: NumpyStore) -> None:
    store.upsert(["a", "a"], [EAST, NORTH], [{"n": 1}, {"n": 2}])
    assert store.count() == 1
    assert store.search(NORTH, k=1)[0][1] == pytest.approx(1.0)
    assert len(store.search(NORTH, k=1, filters={"n": 2})) == 1


def test_delete_removes_rows_and_reports_how_many_existed(store: NumpyStore) -> None:
    fill(store)
    assert store.delete(["a", "unknown"]) == 1
    assert store.count() == 2
    assert [hit_id for hit_id, _ in store.search(EAST, k=8)] == ["b", "c"]


def test_deleting_nothing_known_is_not_an_error(store: NumpyStore) -> None:
    fill(store)
    assert store.delete(["nope"]) == 0
    assert store.count() == 3


def test_delete_then_upsert_keeps_ids_and_vectors_aligned(store: NumpyStore) -> None:
    fill(store)
    store.delete(["a"])
    store.upsert(["d"], [EAST], [{"scope": "proj-a"}])
    assert store.search(EAST, k=1)[0] == ("d", pytest.approx(1.0))
    assert store.search(NORTH, k=1)[0] == ("b", pytest.approx(1.0))


def test_clear_empties_the_store_but_leaves_it_usable(store: NumpyStore) -> None:
    fill(store)
    store.clear()
    assert store.count() == 0
    assert store.search(EAST, k=8) == []
    store.upsert(["fresh"], [EAST], [{}])
    assert store.count() == 1


def test_clear_frees_the_dimension_for_a_new_model(store: NumpyStore) -> None:
    store.upsert(["a"], [EAST], [{}])
    store.clear()
    store.upsert(["a"], [[1.0, 0.0, 0.0, 0.0]], [{}])
    assert store.search([1.0, 0.0, 0.0, 0.0], k=1)[0][1] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
# NumpyStore: filters
# ══════════════════════════════════════════════════════════════════════
def test_a_filter_selects_by_equality(store: NumpyStore) -> None:
    fill(store)
    assert [hit_id for hit_id, _ in store.search(EAST, k=8, filters={"scope": "proj-a"})] == ["a", "c"]


def test_a_list_filter_is_an_or(store: NumpyStore) -> None:
    fill(store)
    hits = store.search(EAST, k=8, filters={"scope": ["proj-a", "proj-b"]})
    assert {hit_id for hit_id, _ in hits} == {"a", "b", "c"}


def test_a_filter_matches_inside_a_list_valued_field(store: NumpyStore) -> None:
    fill(store)
    assert [hit_id for hit_id, _ in store.search(EAST, k=8, filters={"tags": "agent/alpha"})] == ["a", "c"]


def test_filters_and_together(store: NumpyStore) -> None:
    fill(store)
    hits = store.search(EAST, k=8, filters={"scope": "proj-a", "type": "episodic"})
    assert [hit_id for hit_id, _ in hits] == ["c"]


def test_a_filter_on_an_absent_key_matches_nothing(store: NumpyStore) -> None:
    fill(store)
    assert store.search(EAST, k=8, filters={"profile": "tutor"}) == []


def test_a_none_filter_is_ignored(store: NumpyStore) -> None:
    fill(store)
    assert len(store.search(EAST, k=8, filters={"scope": None})) == 3


def test_an_empty_filter_sequence_is_a_caller_bug(store: NumpyStore) -> None:
    fill(store)
    with pytest.raises(ValueError) as excinfo:
        store.search(EAST, k=8, filters={"scope": []})
    assert "omit the key" in str(excinfo.value)


def test_filtering_happens_before_the_top_k_cut(store: NumpyStore) -> None:
    """k counts *matching* rows: a nearer non-matching neighbour cannot starve it."""
    fill(store)
    hits = store.search(EAST, k=1, filters={"scope": "proj-b"})
    assert [hit_id for hit_id, _ in hits] == ["b"]


def test_matches_filters_is_the_shared_language(store: NumpyStore) -> None:
    meta = {"scope": "proj-a", "tags": ["mem/semantic"]}
    assert matches_filters(meta, None)
    assert matches_filters(meta, {})
    assert matches_filters(meta, {"scope": "proj-a", "tags": "mem/semantic"})
    assert not matches_filters(meta, {"scope": "proj-b"})
    assert not matches_filters(meta, {"agent": "alpha"})


def test_an_ordered_filter_lowers_to_a_membership_filter(store: NumpyStore) -> None:
    """§7.1 `min_confidence` is the retrieval layer's to lower; the store only ORs."""
    from mechabrain.contract import CONFIDENCE_LEVELS

    store.upsert(
        ["low", "high"],
        [EAST, EAST],
        [{"confidence": "low"}, {"confidence": "high"}],
    )
    at_least_medium = list(CONFIDENCE_LEVELS[CONFIDENCE_LEVELS.index("medium") :])
    hits = store.search(EAST, k=8, filters={"confidence": at_least_medium})
    assert [hit_id for hit_id, _ in hits] == ["high"]


# ══════════════════════════════════════════════════════════════════════
# NumpyStore: metadata
# ══════════════════════════════════════════════════════════════════════
def test_dates_are_stored_as_iso_so_a_json_filter_can_match_them(
    store: NumpyStore, tmp_vault: VaultPaths
) -> None:
    store.upsert(["a"], [EAST], [{"created": date(2026, 1, 15)}])
    assert len(store.search(EAST, k=1, filters={"created": "2026-01-15"})) == 1

    record = json.loads((tmp_vault.index_dir / METAS_FILE).read_text(encoding="utf-8").splitlines()[0])
    assert record == {"id": "a", "meta": {"created": "2026-01-15"}}


def test_unicode_metadata_survives_a_round_trip(store: NumpyStore, tmp_vault: VaultPaths) -> None:
    store.upsert(["a"], [EAST], [{"title": "memória atômica"}])
    assert len(NumpyStore(tmp_vault.index_dir).search(EAST, k=1, filters={"title": "memória atômica"})) == 1


def test_metadata_that_cannot_be_json_fails_loud(store: NumpyStore) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["a"], [EAST], [{"note": object()}])
    assert "not JSON-serialisable" in str(excinfo.value)


def test_a_path_in_metadata_is_refused_by_name(store: NumpyStore) -> None:
    """R4.2: an absolute path must never reach the derived index."""
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["a"], [EAST], [{"path": Path("/tmp/x.md")}])
    assert "vault-relative" in str(excinfo.value)


# ══════════════════════════════════════════════════════════════════════
# NumpyStore: input validation (R5.1)
# ══════════════════════════════════════════════════════════════════════
def test_mismatched_ids_and_vectors_fail_loud(store: NumpyStore) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["a", "b"], [EAST], [{}, {}])
    assert "2 ids but 1 vectors" in str(excinfo.value)


def test_mismatched_ids_and_metas_fail_loud(store: NumpyStore) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["a", "b"], [EAST, NORTH], [{}])
    assert "2 ids but 1 metadata" in str(excinfo.value)


def test_a_flat_vector_list_is_refused_rather_than_guessed(store: NumpyStore) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["a"], EAST, [{}])
    assert "2-D" in str(excinfo.value)


@pytest.mark.parametrize("bad_id", ["", "   ", None, 7])
def test_an_id_must_be_a_non_empty_string(store: NumpyStore, bad_id: Any) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert([bad_id], [EAST], [{}])
    assert "non-empty string" in str(excinfo.value)


@pytest.mark.parametrize("broken", [float("nan"), float("inf")])
def test_a_broken_vector_is_never_indexed(store: NumpyStore, broken: float) -> None:
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["a"], [[broken, 0.0]], [{}])
    assert "NaN or infinity" in str(excinfo.value)


def test_a_broken_query_is_refused(store: NumpyStore) -> None:
    fill(store)
    with pytest.raises(MechabrainIndexError):
        store.search([float("nan"), 0.0], k=1)


def test_a_dimension_change_on_write_names_the_cause(store: NumpyStore) -> None:
    store.upsert(["a"], [EAST], [{}])
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.upsert(["b"], [[1.0, 0.0, 0.0]], [{}])
    assert "does not match the 2" in str(excinfo.value)
    assert "embedding.model" in str(excinfo.value)
    assert "reindex --full" in str(excinfo.value)


def test_a_dimension_change_on_read_names_the_cause(store: NumpyStore) -> None:
    store.upsert(["a"], [EAST], [{}])
    with pytest.raises(MechabrainIndexError) as excinfo:
        store.search([1.0, 0.0, 0.0], k=1)
    assert "reindex --full" in str(excinfo.value)


def test_a_query_shaped_as_one_row_is_accepted(store: NumpyStore) -> None:
    fill(store)
    assert store.search(np.array([EAST]), k=1)[0][0] == "a"


# ══════════════════════════════════════════════════════════════════════
# NumpyStore: persistence
# ══════════════════════════════════════════════════════════════════════
def test_a_second_store_reads_what_the_first_wrote(tmp_vault: VaultPaths) -> None:
    first = NumpyStore(tmp_vault.index_dir)
    fill(first)
    second = NumpyStore(tmp_vault.index_dir)
    assert second.count() == 3
    assert [hit_id for hit_id, _ in second.search(EAST, k=8, filters={"scope": "proj-a"})] == ["a", "c"]


def test_a_delete_reaches_disk(tmp_vault: VaultPaths) -> None:
    first = NumpyStore(tmp_vault.index_dir)
    fill(first)
    first.delete(["a"])
    assert NumpyStore(tmp_vault.index_dir).count() == 2


def test_a_clear_reaches_disk(tmp_vault: VaultPaths) -> None:
    first = NumpyStore(tmp_vault.index_dir)
    fill(first)
    first.clear()
    assert NumpyStore(tmp_vault.index_dir).count() == 0


def test_autosave_off_defers_every_write_until_flush(tmp_vault: VaultPaths) -> None:
    bulk = NumpyStore(tmp_vault.index_dir, autosave=False)
    fill(bulk)
    assert not (tmp_vault.index_dir / VECTORS_FILE).exists()
    assert NumpyStore(tmp_vault.index_dir).count() == 0

    bulk.flush()
    assert NumpyStore(tmp_vault.index_dir).count() == 3


def test_flush_creates_the_index_dir_when_init_has_not(tmp_path: Path) -> None:
    paths = VaultPaths.for_root(tmp_path / "vault")
    store = NumpyStore(paths.index_dir)
    store.upsert(["a"], [EAST], [{}])
    assert paths.index_dir.is_dir()
    assert NumpyStore(paths.index_dir).count() == 1


def test_writes_leave_no_temp_debris(store: NumpyStore, tmp_vault: VaultPaths) -> None:
    fill(store)
    store.delete(["a"])
    assert [path.name for path in tmp_vault.index_dir.iterdir() if path.name.endswith(".tmp")] == []


def test_a_torn_write_is_detected_not_served(tmp_vault: VaultPaths) -> None:
    """Vectors and metadata are two files; a crash between them must not go unnoticed."""
    fill(NumpyStore(tmp_vault.index_dir))
    metas = tmp_vault.index_dir / METAS_FILE
    metas.write_text("\n".join(metas.read_text(encoding="utf-8").splitlines()[:2]) + "\n", encoding="utf-8")

    with pytest.raises(MechabrainIndexError) as excinfo:
        NumpyStore(tmp_vault.index_dir).count()
    assert "corrupt" in str(excinfo.value)
    assert "reindex --full" in str(excinfo.value)


def test_a_malformed_metadata_line_is_reported_with_its_number(tmp_vault: VaultPaths) -> None:
    fill(NumpyStore(tmp_vault.index_dir))
    metas = tmp_vault.index_dir / METAS_FILE
    lines = metas.read_text(encoding="utf-8").splitlines()
    lines[1] = "{not json"
    metas.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MechabrainIndexError) as excinfo:
        NumpyStore(tmp_vault.index_dir).count()
    assert "line 2" in str(excinfo.value)


def test_a_repeated_id_on_disk_is_corruption(tmp_vault: VaultPaths) -> None:
    fill(NumpyStore(tmp_vault.index_dir))
    metas = tmp_vault.index_dir / METAS_FILE
    lines = metas.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    metas.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MechabrainIndexError) as excinfo:
        NumpyStore(tmp_vault.index_dir).count()
    assert "repeats an id" in str(excinfo.value)


def test_a_shredded_vector_file_is_corruption(tmp_vault: VaultPaths) -> None:
    fill(NumpyStore(tmp_vault.index_dir))
    (tmp_vault.index_dir / VECTORS_FILE).write_bytes(b"not a npy file")

    with pytest.raises(MechabrainIndexError) as excinfo:
        NumpyStore(tmp_vault.index_dir).count()
    assert "unreadable" in str(excinfo.value)


def test_half_written_state_is_ignored_rather_than_half_read(tmp_vault: VaultPaths) -> None:
    """Only vectors.npy on disk: nothing was committed, so the store is empty."""
    tmp_vault.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(tmp_vault.index_dir / VECTORS_FILE, np.zeros((3, 2), dtype=np.float32))
    assert NumpyStore(tmp_vault.index_dir).count() == 0


def test_the_matrix_on_disk_is_normalised_float32(tmp_vault: VaultPaths) -> None:
    store = NumpyStore(tmp_vault.index_dir)
    store.upsert(["a"], [[3.0, 4.0]], [{}])
    matrix = np.load(tmp_vault.index_dir / VECTORS_FILE)
    assert matrix.dtype == np.float32
    assert matrix == pytest.approx(np.array([[0.6, 0.8]], dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════
# Optional backends -- skipped unless their extra is installed
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def optional_store(request: pytest.FixtureRequest, tmp_vault: VaultPaths) -> VectorStore:
    """A store from an optional extra, or a skip."""
    name, module = request.param
    pytest.importorskip(module, reason=f"install mechabrain[{name}] to exercise it")
    return open_store(name, tmp_vault.index_dir)


OPTIONAL = [
    pytest.param(("lancedb", "lancedb"), id="lancedb"),
    pytest.param(("sqlite-vec", "sqlite_vec"), id="sqlite-vec"),
]


@pytest.mark.parametrize("optional_store", OPTIONAL, indirect=True)
def test_optional_store_round_trips(optional_store: VectorStore) -> None:
    assert optional_store.count() == 0
    assert optional_store.search(EAST, k=4) == []

    optional_store.upsert(
        ["a", "b", "c"],
        [EAST, NORTH, WEST],
        [{"scope": "proj-a"}, {"scope": "proj-b"}, {"scope": "proj-a"}],
    )
    optional_store.flush()
    assert optional_store.count() == 3

    hits = optional_store.search(EAST, k=3)
    assert [hit_id for hit_id, _ in hits] == ["a", "b", "c"]
    assert [score for _, score in hits] == pytest.approx([1.0, 0.0, -1.0], abs=1e-5)


@pytest.mark.parametrize("optional_store", OPTIONAL, indirect=True)
def test_optional_store_filters_and_replaces(optional_store: VectorStore) -> None:
    optional_store.upsert(
        ["a", "b", "c"],
        [EAST, NORTH, WEST],
        [{"scope": "proj-a"}, {"scope": "proj-b"}, {"scope": "proj-a"}],
    )
    hits = optional_store.search(EAST, k=8, filters={"scope": "proj-a"})
    assert [hit_id for hit_id, _ in hits] == ["a", "c"]

    optional_store.upsert(["a"], [NORTH], [{"scope": "proj-b"}])
    assert optional_store.count() == 3
    assert optional_store.search(EAST, k=8, filters={"scope": "proj-a"})[0][0] == "c"

    assert optional_store.delete(["a", "unknown"]) == 1
    assert optional_store.count() == 2

    optional_store.clear()
    assert optional_store.count() == 0


@pytest.mark.parametrize("optional_store", OPTIONAL, indirect=True)
def test_optional_store_guards_the_dimension(optional_store: VectorStore) -> None:
    """Every backend must catch a changed embedding model, not just the default one."""
    optional_store.upsert(["a"], [EAST], [{}])
    with pytest.raises(MechabrainIndexError, match="does not match the 2"):
        optional_store.upsert(["b"], [[1.0, 0.0, 0.0]], [{}])
    with pytest.raises(MechabrainIndexError, match="does not match the 2"):
        optional_store.search([1.0, 0.0, 0.0], k=1)


@pytest.mark.parametrize("optional_store", OPTIONAL, indirect=True)
def test_optional_store_keeps_its_state_in_the_index_dir(
    optional_store: VectorStore, tmp_vault: VaultPaths
) -> None:
    optional_store.upsert(["a"], [EAST], [{}])
    optional_store.flush()
    expected = LANCEDB_DIR if isinstance(optional_store, LanceDBStore) else SQLITE_VEC_FILE
    assert (tmp_vault.index_dir / expected).exists()
