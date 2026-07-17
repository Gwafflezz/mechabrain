"""Tests for the access log (R7.2/R7.3).

Two properties carry the module and get the most attention here:

* concurrent appenders never corrupt the log -- asserted on the *raw bytes*,
  not through :meth:`AccessLog.aggregate`, which skips bad lines and would
  therefore hide exactly the damage under test;
* an access is never lost -- not to a crashed flush, and not to a record
  written while a flush is in flight.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mechabrain.access import (
    MAX_APPEND_BYTES,
    PENDING_SUFFIX,
    AccessKind,
    AccessLog,
)
from mechabrain.discovery import VaultPaths
from mechabrain.note import Note

# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def log(tmp_vault: VaultPaths) -> AccessLog:
    """The access log of a disposable vault, empty."""
    return AccessLog.for_vault(tmp_vault)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[datetime], None]:
    """Pin the module clock. Returns a setter, so a test can move time forward."""
    current = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)

    def now() -> datetime:
        return current

    def set_now(moment: datetime) -> None:
        nonlocal current
        current = moment

    monkeypatch.setattr("mechabrain.access._now", now)
    return set_now


def read_lines(path: Path) -> list[str]:
    """Raw non-empty lines of ``path``. Deliberately not via ``aggregate``."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


# ══════════════════════════════════════════════════════════════════════
# Recording
# ══════════════════════════════════════════════════════════════════════
def test_record_appends_one_line_per_call(log: AccessLog) -> None:
    log.record(["a", "b"], AccessKind.SEARCH)
    log.record(["c"], AccessKind.GET)

    lines = read_lines(log.path)
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["ids"] == ["a", "b"]
    assert first["kind"] == "search"
    assert second["ids"] == ["c"]
    assert second["kind"] == "get"


def test_record_returns_note_count_and_creates_the_index_dir(tmp_path: Path) -> None:
    # index/ is gitignored, so a fresh clone has no such directory (§4).
    log = AccessLog(tmp_path / "absent" / "access.jsonl")
    assert log.record(["a", "b"], AccessKind.GET) == 2
    assert log.path.is_file()


def test_record_of_nothing_touches_no_disk(log: AccessLog) -> None:
    assert log.record([], AccessKind.SEARCH) == 0
    assert not log.path.exists()


def test_record_takes_a_bare_id_as_one_note_not_as_characters(log: AccessLog) -> None:
    assert log.record("PROC_deploy-playbook", AccessKind.GET) == 1
    assert json.loads(read_lines(log.path)[0])["ids"] == ["PROC_deploy-playbook"]


def test_record_normalizes_paths_and_filenames_to_note_ids(
    log: AccessLog, tmp_vault: VaultPaths
) -> None:
    log.record(
        [tmp_vault.semantic_dir / "2026-01-15_INS_x.md", "2026-01-15_INS_y.md"],
        AccessKind.SEARCH,
    )
    assert json.loads(read_lines(log.path)[0])["ids"] == [
        "2026-01-15_INS_x",
        "2026-01-15_INS_y",
    ]


def test_record_collapses_duplicate_ids(log: AccessLog) -> None:
    assert log.record(["a", "a", "b"], AccessKind.SEARCH) == 2
    assert json.loads(read_lines(log.path)[0])["ids"] == ["a", "b"]


def test_record_accepts_a_string_kind(log: AccessLog) -> None:
    log.record(["a"], "get")
    assert json.loads(read_lines(log.path)[0])["kind"] == "get"


def test_record_rejects_an_unknown_kind_naming_the_valid_ones(log: AccessLog) -> None:
    with pytest.raises(ValueError, match="search, get"):
        log.record(["a"], "peeked")
    assert not log.path.exists()


def test_record_splits_a_large_result_into_atomically_appendable_lines(
    log: AccessLog,
) -> None:
    ids = [f"2026-01-15_INS_note-{i:04d}" for i in range(500)]
    assert log.record(ids, AccessKind.SEARCH) == 500

    lines = read_lines(log.path)
    assert len(lines) > 1, "a 500-hit result should not ride in one oversized write"
    assert all(len(line.encode("utf-8")) + 1 <= MAX_APPEND_BYTES for line in lines)
    # Splitting is lossless: every id survives, exactly once.
    recovered = [i for line in lines for i in json.loads(line)["ids"]]
    assert recovered == ids


# ══════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════
def test_aggregate_of_an_unread_vault_is_empty(log: AccessLog) -> None:
    assert log.aggregate() == {}


def test_aggregate_maps_each_note_to_the_access_date(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    log.record(["a", "b"], AccessKind.SEARCH)
    assert log.aggregate() == {
        "a": date(2026, 7, 16),
        "b": date(2026, 7, 16),
    }


def test_aggregate_keeps_the_most_recent_access(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    log.record(["a"], AccessKind.SEARCH)
    frozen_clock(datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc))
    log.record(["a", "b"], AccessKind.GET)

    assert log.aggregate() == {"a": date(2026, 7, 20), "b": date(2026, 7, 20)}


def test_aggregate_takes_the_maximum_not_the_last_line(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    # Appenders interleave and clocks step backwards; a later *line* is not
    # necessarily a later *access*, and reading a note must never un-read it.
    frozen_clock(datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc))
    log.record(["a"], AccessKind.SEARCH)
    frozen_clock(datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc))
    log.record(["a"], AccessKind.SEARCH)

    assert log.aggregate() == {"a": date(2026, 7, 20)}


def test_aggregate_reports_dates_in_utc(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    # 23:30 in a +05:30 zone is the previous UTC day: the log agrees on one
    # timezone so that machines sharing a vault agree on the decay clock.
    frozen_clock(
        datetime(2026, 7, 17, 2, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    )
    log.record(["a"], AccessKind.SEARCH)

    assert log.aggregate() == {"a": date(2026, 7, 16)}


def test_aggregate_ignores_the_access_kind(log: AccessLog) -> None:
    log.record(["a"], AccessKind.GET)
    log.record(["b"], AccessKind.SEARCH)
    assert set(log.aggregate()) == {"a", "b"}


@pytest.mark.parametrize(
    "junk",
    [
        "not json at all",
        '{"ts":"2026-07-16T12:00:00+00:00","ids":["torn"',  # crashed mid-write
        '{"ts":"nonsense","ids":["a"]}',
        '{"ids":["a"]}',  # no timestamp
        '{"ts":"2026-07-16T12:00:00+00:00"}',  # no ids
        '["not","a","mapping"]',
        "",
        "   ",
    ],
)
def test_aggregate_skips_a_damaged_line_and_reads_the_rest(
    log: AccessLog, frozen_clock: Callable[[datetime], None], junk: str
) -> None:
    log.record(["good"], AccessKind.SEARCH)
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write(f"{junk}\n")
    log.record(["also-good"], AccessKind.GET)

    assert log.aggregate() == {
        "good": date(2026, 7, 16),
        "also-good": date(2026, 7, 16),
    }


def test_aggregate_reads_a_naive_timestamp_as_utc(log: AccessLog) -> None:
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text('{"ts":"2026-07-16T12:00:00","kind":"get","ids":["a"]}\n')
    assert log.aggregate() == {"a": date(2026, 7, 16)}


def test_aggregate_tolerates_unknown_keys_from_a_newer_kernel(log: AccessLog) -> None:
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(
        '{"ts":"2026-07-16T12:00:00+00:00","kind":"browse","ids":["a"],"query":"x"}\n'
    )
    assert log.aggregate() == {"a": date(2026, 7, 16)}


# ══════════════════════════════════════════════════════════════════════
# Concurrency (R7.2: cheap, lock-free, append-only)
# ══════════════════════════════════════════════════════════════════════
def _hammer(args: tuple[str, str, int]) -> int:
    """Worker for the concurrency tests: importable, hence top-level.

    Returns its pid, so the multi-process test can prove it really ran out of
    process instead of quietly degrading to the in-process case it exists to
    rule out.
    """
    path, tag, rounds = args
    log = AccessLog(Path(path))
    for i in range(rounds):
        log.record([f"{tag}-{i:03d}", "shared"], AccessKind.SEARCH)
    return os.getpid()


def test_concurrent_appends_from_threads_do_not_corrupt_the_log(log: AccessLog) -> None:
    workers, rounds = 8, 60
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [
            pool.submit(_hammer, (str(log.path), f"t{w}", rounds))
            for w in range(workers)
        ]
        for job in jobs:
            job.result()  # re-raise anything a worker hit

    lines = read_lines(log.path)
    assert len(lines) == workers * rounds, "lines were lost or torn in two"
    for line in lines:
        json.loads(line)  # every line is whole: no interleaved writes

    assert set(log.aggregate()) == {"shared"} | {
        f"t{w}-{i:03d}" for w in range(workers) for i in range(rounds)
    }


def test_concurrent_appends_from_processes_do_not_corrupt_the_log(
    log: AccessLog,
) -> None:
    # The case R7.4 is actually about: separate processes, separate file
    # descriptions, no shared lock. O_APPEND is the whole guarantee.
    workers, rounds = 4, 60
    with ProcessPoolExecutor(max_workers=workers) as pool:
        args = [(str(log.path), f"p{w}", rounds) for w in range(workers)]
        pids = set(pool.map(_hammer, args))

    assert os.getpid() not in pids, "the workers must not have run in-process"

    lines = read_lines(log.path)
    assert len(lines) == workers * rounds
    for line in lines:
        json.loads(line)

    assert set(log.aggregate()) == {"shared"} | {
        f"p{w}-{i:03d}" for w in range(workers) for i in range(rounds)
    }


def test_concurrent_appends_of_long_results_stay_line_aligned(log: AccessLog) -> None:
    # Near the split threshold is where a naive writer tears: each record here
    # packs ids until it must break, so writes are large and adjacent.
    workers = 6
    per_call = [f"{'x' * 60}-{w}-{i:03d}" for w in range(workers) for i in range(40)]

    def push(worker: int) -> None:
        for _ in range(10):
            log.record(per_call, AccessKind.SEARCH)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(push, range(workers)))

    for line in read_lines(log.path):
        payload = json.loads(line)
        assert payload["kind"] == "search"
        assert all(isinstance(i, str) for i in payload["ids"])


# ══════════════════════════════════════════════════════════════════════
# Rotation
# ══════════════════════════════════════════════════════════════════════
def test_rotate_moves_the_log_aside_and_keeps_its_records(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    log.record(["a"], AccessKind.SEARCH)

    shard = log.rotate()

    assert shard is not None and shard.name.endswith(PENDING_SUFFIX)
    assert not log.path.exists(), "the live log is renamed, never copied"
    assert log.pending_shards() == [shard]
    assert log.aggregate() == {"a": date(2026, 7, 16)}, "a shard is still the log"


def test_rotate_of_an_unread_vault_is_a_noop(log: AccessLog) -> None:
    assert log.rotate() is None
    assert log.pending_shards() == []


def test_record_after_rotate_starts_a_fresh_log(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    log.record(["old"], AccessKind.SEARCH)
    log.rotate()
    log.record(["new"], AccessKind.GET)

    assert read_lines(log.path) and json.loads(read_lines(log.path)[0])["ids"] == ["new"]
    assert set(log.aggregate()) == {"old", "new"}


def test_rotate_twice_keeps_both_shards(log: AccessLog) -> None:
    log.record(["a"], AccessKind.SEARCH)
    first = log.rotate()
    log.record(["b"], AccessKind.SEARCH)
    second = log.rotate()

    assert first != second
    assert log.pending_shards() == sorted([first, second])  # type: ignore[list-item]
    assert set(log.aggregate()) == {"a", "b"}


# ══════════════════════════════════════════════════════════════════════
# Flush (§9.1 step 1)
# ══════════════════════════════════════════════════════════════════════
def test_flush_yields_the_accesses_then_drops_the_log(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    log.record(["a", "b"], AccessKind.SEARCH)

    with log.flush() as accesses:
        assert accesses == {"a": date(2026, 7, 16), "b": date(2026, 7, 16)}

    assert log.aggregate() == {}
    assert log.pending_shards() == []


def test_flush_of_an_unread_vault_yields_nothing(log: AccessLog) -> None:
    with log.flush() as accesses:
        assert accesses == {}


def test_flush_keeps_the_accesses_when_applying_them_fails(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    # At-least-once: a consolidator that dies before writing last_accessed must
    # find the same accesses next cycle, not a note that looks unread.
    log.record(["a"], AccessKind.SEARCH)

    with pytest.raises(RuntimeError):
        with log.flush() as accesses:
            assert accesses == {"a": date(2026, 7, 16)}
            raise RuntimeError("consolidator died mid-cycle")

    assert log.pending_shards(), "the rotated shard survives a failed flush"
    with log.flush() as accesses:
        assert accesses == {"a": date(2026, 7, 16)}
    assert log.aggregate() == {}


def test_flush_does_not_lose_a_record_written_while_it_runs(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    # The reason rotate() renames instead of truncating: a search landing
    # between the aggregate and the reset would otherwise vanish.
    log.record(["before"], AccessKind.SEARCH)

    with log.flush() as accesses:
        assert set(accesses) == {"before"}
        log.record(["during"], AccessKind.GET)

    assert log.aggregate() == {"during": date(2026, 7, 16)}


def test_flush_after_a_crashed_flush_reports_both_cycles(
    log: AccessLog, frozen_clock: Callable[[datetime], None]
) -> None:
    log.record(["a"], AccessKind.SEARCH)
    with pytest.raises(RuntimeError):
        with log.flush():
            raise RuntimeError("boom")

    frozen_clock(datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc))
    log.record(["b"], AccessKind.GET)

    with log.flush() as accesses:
        assert accesses == {"a": date(2026, 7, 16), "b": date(2026, 7, 20)}
    assert log.aggregate() == {}


def test_clear_drops_the_log_and_every_shard(log: AccessLog) -> None:
    log.record(["a"], AccessKind.SEARCH)
    log.rotate()
    log.record(["b"], AccessKind.GET)

    log.clear()

    assert log.aggregate() == {}
    assert not log.path.exists()
    assert log.pending_shards() == []


# ══════════════════════════════════════════════════════════════════════
# Boundaries
# ══════════════════════════════════════════════════════════════════════
def test_the_log_lives_in_the_gitignored_index_dir(tmp_vault: VaultPaths) -> None:
    log = AccessLog.for_vault(tmp_vault)
    assert log.path == tmp_vault.access_file
    assert log.path.parent == tmp_vault.index_dir  # R7.2: derived, per machine


def test_rotation_artifacts_stay_inside_the_index_dir(
    log: AccessLog, tmp_vault: VaultPaths
) -> None:
    log.record(["a"], AccessKind.SEARCH)
    log.rotate()

    for artifact in tmp_vault.index_dir.iterdir():
        assert artifact.parent == tmp_vault.index_dir
    assert list(tmp_vault.meta_dir.glob(f"*{PENDING_SUFFIX}")) == []


def test_recording_and_flushing_never_touch_a_note(
    tmp_vault: VaultPaths, sample_notes: list[Note]
) -> None:
    # R7.3 draws the line here: this module records, the consolidator applies.
    # If a test ever fails, someone taught the log to write last_accessed.
    log = AccessLog.for_vault(tmp_vault)
    before = {
        note.path: (note.path.read_bytes(), note.path.stat().st_mtime_ns)
        for note in sample_notes
        if note.path is not None
    }

    log.record([note.note_id for note in sample_notes], AccessKind.SEARCH)
    with log.flush() as accesses:
        assert len(accesses) == len(sample_notes)

    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert after == before


def test_two_vaults_keep_separate_logs(make_vault: Callable[..., VaultPaths]) -> None:
    first = AccessLog.for_vault(make_vault(name="one"))
    second = AccessLog.for_vault(make_vault(name="two"))

    first.record(["a"], AccessKind.SEARCH)
    second.record(["b"], AccessKind.GET)

    assert set(first.aggregate()) == {"a"}
    assert set(second.aggregate()) == {"b"}


def test_repr_names_the_log_file(log: AccessLog) -> None:
    assert str(log.path) in repr(log)
