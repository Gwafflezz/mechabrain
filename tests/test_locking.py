"""Advisory file locking (R7.4 fallback)."""

from __future__ import annotations

import multiprocessing
import os
import textwrap
import time
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

from mechabrain.errors import MechabrainError, MechabrainIndexError
from mechabrain.locking import FileLock, LockHolder, file_lock


def _hold_lock(path: str, ready: Connection, hold_seconds: float) -> None:
    """Hold ``path`` in a child process, signalling the parent once acquired."""
    with file_lock(path, timeout=10.0, purpose="child-holder"):
        ready.send("acquired")
        ready.close()
        time.sleep(hold_seconds)


def _spawn_holder(lock_path: Path, hold_seconds: float) -> tuple[multiprocessing.process.BaseProcess, Connection]:
    """Start a child holding ``lock_path``; return once it actually holds it.

    Uses a Pipe rather than a raw fd: the `spawn` start method does not inherit
    file descriptors, and `fork` is unsafe to assume.
    """
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe()
    child = context.Process(target=_hold_lock, args=(str(lock_path), child_conn, hold_seconds))
    child.start()
    child_conn.close()
    assert parent_conn.poll(timeout=15), "child never acquired the lock"
    assert parent_conn.recv() == "acquired"
    return child, parent_conn


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "index" / "index.lock"


# ── Basics ──────────────────────────────────────────────────────────
def test_lock_creates_its_file_and_parents(lock_path: Path) -> None:
    with file_lock(lock_path):
        assert lock_path.is_file()


def test_context_manager_acquires_and_releases(lock_path: Path) -> None:
    lock = FileLock(lock_path)
    assert not lock.is_held
    with lock:
        assert lock.is_held
    assert not lock.is_held


def test_lock_is_released_when_the_block_raises(lock_path: Path) -> None:
    lock = FileLock(lock_path)
    with pytest.raises(RuntimeError), lock:
        raise RuntimeError("boom")
    assert not lock.is_held
    with FileLock(lock_path, timeout=0):
        pass  # Free again: the failed block did not leak the lock.


def test_release_is_idempotent(lock_path: Path) -> None:
    lock = FileLock(lock_path)
    lock.acquire()
    lock.release()
    lock.release()
    assert not lock.is_held


def test_lock_is_reentrant_within_one_instance(lock_path: Path) -> None:
    lock = FileLock(lock_path, timeout=0)
    with lock:
        with lock:
            assert lock.is_held
        assert lock.is_held, "inner exit must not release the outer hold"
    assert not lock.is_held


def test_sequential_locks_do_not_block(lock_path: Path) -> None:
    with file_lock(lock_path, purpose="first"):
        pass
    with file_lock(lock_path, timeout=0, purpose="second") as holder:
        assert holder.purpose == "second"


def test_different_paths_do_not_contend(tmp_path: Path) -> None:
    with file_lock(tmp_path / "a.lock", timeout=0), file_lock(tmp_path / "b.lock", timeout=0):
        pass


def test_repr_reports_state(lock_path: Path) -> None:
    lock = FileLock(lock_path, purpose="reindex")
    assert "free" in repr(lock)
    with lock:
        assert "held" in repr(lock)
        assert "reindex" in repr(lock)


# ── Holder metadata ─────────────────────────────────────────────────
def test_acquire_records_the_current_holder(lock_path: Path) -> None:
    with file_lock(lock_path, purpose="consolidate") as holder:
        assert holder.pid == os.getpid()
        assert holder.purpose == "consolidate"
        recorded = LockHolder.read(lock_path)
    assert recorded is not None
    assert recorded.pid == os.getpid()
    assert recorded.purpose == "consolidate"


def test_holder_describe_names_pid_and_purpose() -> None:
    holder = LockHolder(pid=42, host="box", since="2026-01-15T10:00:00+00:00", purpose="reindex")
    described = holder.describe()
    assert "42" in described and "box" in described and "reindex" in described


def test_holder_read_tolerates_garbage(tmp_path: Path) -> None:
    """Holder metadata is diagnostics: unreadable must degrade, never raise."""
    corrupt = tmp_path / "corrupt.lock"
    corrupt.write_text("not json at all", encoding="utf-8")
    assert LockHolder.read(corrupt) is None
    assert LockHolder.read(tmp_path / "missing.lock") is None


def test_holder_survives_a_lock_file_rewrite(lock_path: Path) -> None:
    """A second, shorter purpose must not leave trailing bytes of the first."""
    with file_lock(lock_path, purpose="a-very-long-purpose-string"):
        pass
    with file_lock(lock_path, purpose="short"):
        holder = LockHolder.read(lock_path)
    assert holder is not None
    assert holder.purpose == "short"


# ── Contention across processes (R7.4) ──────────────────────────────
def test_second_process_times_out_and_names_the_holder(lock_path: Path) -> None:
    child, conn = _spawn_holder(lock_path, hold_seconds=5.0)
    try:
        start = time.monotonic()
        with pytest.raises(MechabrainIndexError) as excinfo:
            FileLock(lock_path, timeout=0.3).acquire()
        elapsed = time.monotonic() - start

        assert 0.3 <= elapsed < 3.0, "must wait for the timeout, then give up"
        error = excinfo.value
        assert isinstance(error, MechabrainError)
        assert error.rule == "R7.4"
        assert "timed out" in error.message
        assert "child-holder" in error.message, "the error must name the holder"
        assert str(child.pid) in error.message
        assert error.hint is not None and "serve" in error.hint
    finally:
        conn.close()
        child.terminate()
        child.join(timeout=5)


def test_lock_is_free_once_the_holder_exits(lock_path: Path) -> None:
    """flock is released by the kernel on process death: no stale locks."""
    child, conn = _spawn_holder(lock_path, hold_seconds=0.1)
    conn.close()
    child.join(timeout=10)
    assert child.exitcode == 0
    assert lock_path.is_file(), "the lock file outlives the holder"
    with file_lock(lock_path, timeout=2.0):
        pass  # ...but is inert: the lock itself died with the process.


def test_waiter_acquires_after_the_holder_releases(lock_path: Path) -> None:
    child, conn = _spawn_holder(lock_path, hold_seconds=0.4)
    try:
        with file_lock(lock_path, timeout=10.0, purpose="waiter") as holder:
            assert holder.pid == os.getpid()
    finally:
        conn.close()
        child.join(timeout=5)


# ── Serialization guarantee ─────────────────────────────────────────
def _increment_counter(counter_path: str, lock_file: str, rounds: int) -> None:
    """Read-modify-write under the lock -- what write_atomic alone cannot make safe."""
    for _ in range(rounds):
        with file_lock(lock_file, timeout=30.0, purpose="counter"):
            path = Path(counter_path)
            value = int(path.read_text(encoding="utf-8"))
            time.sleep(0.001)  # Widen the race window.
            path.write_text(str(value + 1), encoding="utf-8")


def test_lock_serializes_read_modify_write_across_processes(tmp_path: Path) -> None:
    counter = tmp_path / "counter.txt"
    counter.write_text("0", encoding="utf-8")
    lock_file = str(tmp_path / "counter.lock")
    rounds, workers = 20, 4

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_increment_counter, args=(str(counter), lock_file, rounds))
        for _ in range(workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    assert int(counter.read_text(encoding="utf-8")) == rounds * workers


def test_module_is_importable_in_a_fresh_interpreter() -> None:
    """The lock is taken by CLI subprocesses, not only inside the test process."""
    import subprocess
    import sys

    code = textwrap.dedent(
        """
        from mechabrain.locking import file_lock, DEFAULT_LOCK_TIMEOUT
        assert DEFAULT_LOCK_TIMEOUT > 0
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "ok"
