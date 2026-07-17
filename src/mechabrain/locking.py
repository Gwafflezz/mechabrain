"""Advisory file locking (R7.4 fallback).

The normative concurrency model is **one writer per machine**: `mechabrain
serve` runs as a local daemon and every MCP client points at it. This module is
the fallback for when there is no daemon -- a CLI run of `reindex` or
`consolidate` while a daemon may be up, two `memory_write` calls racing on
`links.jsonl`. It serialises *decisions*; :func:`mechabrain.note.write_atomic`
only serialises bytes.

Built on ``fcntl.flock``, which the kernel releases automatically when the
holding process dies. There is therefore no stale-lock problem and no lock file
to clean up by hand: a lock file left behind by a crash is inert, and the
holder metadata inside it is best-effort diagnostics, never the lock itself.

POSIX only. Locks live in ``_meta/index/`` -- runtime layer, gitignored, per
machine (§4).
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Final, Iterator

from .errors import MechabrainIndexError

__all__ = ["FileLock", "LockHolder", "file_lock", "DEFAULT_LOCK_TIMEOUT"]

#: Seconds to wait before giving up. Long enough to outlast a reindex commit,
#: short enough that a wedged holder surfaces instead of hanging a session.
DEFAULT_LOCK_TIMEOUT: Final[float] = 30.0

_POLL_INTERVAL: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class LockHolder:
    """Best-effort description of the process holding a lock.

    Written into the lock file by the holder purely so a blocked process can
    name it. Never trusted for correctness -- ``flock`` is the truth.
    """

    pid: int
    host: str
    since: str
    purpose: str = ""

    def describe(self) -> str:
        """Human-readable one-liner for an error message."""
        what = f" ({self.purpose})" if self.purpose else ""
        return f"pid {self.pid} on {self.host}{what}, holding since {self.since}"

    def to_json(self) -> str:
        return json.dumps(
            {
                "pid": self.pid,
                "host": self.host,
                "since": self.since,
                "purpose": self.purpose,
            }
        )

    @classmethod
    def current(cls, purpose: str = "") -> "LockHolder":
        """Describe this process, now."""
        return cls(
            pid=os.getpid(),
            host=socket.gethostname(),
            since=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            purpose=purpose,
        )

    @classmethod
    def read(cls, path: Path) -> "LockHolder | None":
        """Parse the holder recorded at ``path``, or ``None`` if unreadable."""
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return cls(
                pid=int(data["pid"]),
                host=str(data["host"]),
                since=str(data["since"]),
                purpose=str(data.get("purpose", "")),
            )
        except (ValueError, KeyError, TypeError):
            return None


class FileLock:
    """An exclusive advisory lock on ``path``, usable as a context manager.

    ::

        with FileLock(paths.index_dir / "index.lock", purpose="reindex"):
            rebuild(...)

    The lock is *advisory*: it only excludes other processes that take the same
    lock. Reentrant within one instance (nested ``with`` blocks refcount), but
    not across instances in the same process -- ``flock`` is per file
    description, so a second :class:`FileLock` on the same path in the same
    process would deadlock against itself until the timeout.

    Args:
        path: Lock file. Created if absent; its parent is created too.
        timeout: Seconds to wait. ``0`` fails immediately if held.
        purpose: Short label recorded for whoever blocks on this lock.
    """

    __slots__ = ("path", "timeout", "purpose", "_fd", "_depth")

    def __init__(
        self,
        path: Path | str,
        *,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        purpose: str = "",
    ) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.purpose = purpose
        self._fd: int | None = None
        self._depth = 0

    @property
    def is_held(self) -> bool:
        """Whether *this instance* currently holds the lock."""
        return self._fd is not None

    def acquire(self) -> "LockHolder":
        """Take the lock, blocking up to ``timeout``.

        Returns:
            The :class:`LockHolder` record written for this process.

        Raises:
            MechabrainIndexError: the timeout elapsed. The message names the
                current holder when it could be read (R7.4).
        """
        if self._fd is not None:
            self._depth += 1
            return LockHolder.current(self.purpose)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise MechabrainIndexError(
                            f"cannot lock {self.path}: {exc}",
                            rule="R7.4",
                        ) from exc
                    if time.monotonic() >= deadline:
                        raise self._timeout_error() from exc
                    time.sleep(_POLL_INTERVAL)
        except BaseException:
            os.close(fd)
            raise

        holder = LockHolder.current(self.purpose)
        self._fd = fd
        self._depth = 1
        self._record(holder)
        return holder

    def release(self) -> None:
        """Release the lock, or decrement the reentrancy count.

        Idempotent: releasing an unheld lock does nothing.
        """
        if self._fd is None:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        fd = self._fd
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _record(self, holder: LockHolder) -> None:
        """Stamp holder metadata into the lock file. Diagnostics only."""
        if self._fd is None:
            return
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.ftruncate(self._fd, 0)
            os.write(self._fd, holder.to_json().encode("utf-8"))
            os.fsync(self._fd)
        except OSError:
            pass  # Never fail an acquired lock over its own diagnostics.

    def _timeout_error(self) -> MechabrainIndexError:
        holder = LockHolder.read(self.path)
        held_by = holder.describe() if holder else "an unidentified process"
        return MechabrainIndexError(
            f"timed out after {self.timeout:g}s waiting for {self.path}; held by {held_by}",
            rule="R7.4",
            hint=(
                "another mechabrain process is writing. Wait for it, or stop it. "
                "R7.4: run a single `mechabrain serve` daemon per machine and point "
                "MCP clients at it, rather than one kernel process per session."
            ),
        )

    def __enter__(self) -> "LockHolder":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "held" if self.is_held else "free"
        return f"FileLock({str(self.path)!r}, purpose={self.purpose!r}, {state})"


@contextmanager
def file_lock(
    path: Path | str,
    *,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    purpose: str = "",
) -> Iterator[LockHolder]:
    """Hold an exclusive lock on ``path`` for the duration of the block.

    ::

        with file_lock(paths.links_file.with_suffix(".lock"), purpose="memory_link"):
            append_edge(...)

    Yields:
        The :class:`LockHolder` for this process.

    Raises:
        MechabrainIndexError: the lock was not free within ``timeout``.
    """
    lock = FileLock(path, timeout=timeout, purpose=purpose)
    holder = lock.acquire()
    try:
        yield holder
    finally:
        lock.release()
