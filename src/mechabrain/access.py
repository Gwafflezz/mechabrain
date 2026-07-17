"""Access tracking without git noise (§7.3).

`last_accessed:` in frontmatter feeds Ebbinghaus decay (§9.3), but stamping it
on every read would produce one commit per query in a vault with auto-commit.
The spec's answer is to split *recording* from *applying*:

* **R7.2** -- `search`/`get` append to ``_meta/index/access.jsonl``: derived,
  gitignored, per machine. Cheap, lock-free, no note opened.
* **R7.3** -- the consolidation job (§9.1) aggregates the log and only then
  writes `last_accessed:` into frontmatter -- one commit per maintenance cycle.

This module owns the first half **and nothing else**: it never opens, parses or
writes a note. :meth:`AccessLog.flush` hands the consolidator a plain
``{note_id: date}`` mapping; what that mapping does to a note is the
consolidator's business.

Design notes, since three of the choices here are load-bearing:

**Appends are lock-free.** ``record`` is on the hot path of every search and
get; a lock would serialise reads across the whole machine to protect a file
that is disposable by definition. Instead each append is one ``os.write`` to an
``O_APPEND`` descriptor, which POSIX orders against every other appender: the
kernel resolves the offset and writes under one lock, so concurrent writers
interleave *whole lines* rather than corrupting each other. Lines are kept
under :data:`MAX_APPEND_BYTES` (a record over ``ids`` is split across lines) to
stay well inside what a single write delivers in one piece.

**No fsync.** A record lost to a power cut costs one note an early
``status: arquivado``, which §9.3 makes reversible (P8). That is not worth an
fsync per query.

**Rotate, do not truncate.** Aggregating and then truncating would drop every
record written in between. :meth:`rotate` instead ``os.replace``s the live log
onto a uniquely named pending shard: appenders keep writing (to a fresh live
file), and the shard is a stable snapshot the consolidator can read at leisure.
Shards outlive a crashed flush and the next flush picks them up, so the
pipeline is at-least-once -- and applying the same ``last_accessed`` twice is a
no-op, so at-least-once is exactly right here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Final

from .discovery import VaultPaths
from .locking import file_lock
from .note import note_id_for

__all__ = [
    "AccessKind",
    "AccessLog",
    "MAX_APPEND_BYTES",
    "PENDING_SUFFIX",
]

#: Ceiling on one appended line. Concurrent ``O_APPEND`` writers interleave at
#: write boundaries, so a record whose ``ids`` would exceed this is split into
#: several lines rather than risking one oversized write being delivered in
#: pieces. 4 KiB matches the conservative POSIX atomicity bound (``PIPE_BUF``)
#: and is far above a realistic search result.
MAX_APPEND_BYTES: Final[int] = 4096

#: Extension of a rotated log awaiting aggregation. Gitignored with the rest of
#: ``_meta/index/`` -- these are shards of derived state, never source of truth.
PENDING_SUFFIX: Final[str] = ".pending"

_LOCK_PURPOSE: Final[str] = "access-flush"


class AccessKind(str, Enum):
    """Why a note was touched -- the two read paths R7.2 names.

    Inherits :class:`str` so members serialise straight to JSON. Recorded for
    diagnostics: decay cares only *that* a note was read, so
    :meth:`AccessLog.aggregate` ignores this field.
    """

    SEARCH = "search"
    GET = "get"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: "AccessKind | str") -> "AccessKind":
        """Coerce ``value`` to a member, raising ``ValueError`` with the valid set."""
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"unknown access kind {value!r}; valid kinds are: {valid}"
            ) from None


class AccessLog:
    """The append-only access log at ``_meta/index/access.jsonl`` (R7.2).

    ::

        log = AccessLog.for_vault(paths)
        log.record(hit_ids, AccessKind.SEARCH)   # on every search/get

        with log.flush() as accesses:            # in `consolidate`, step 1
            apply_last_accessed(accesses)        # someone else's job

    Cheap to construct and stateless: hold one per vault or build one per call,
    whichever suits. Nothing is created on disk until the first record.

    Args:
        path: The log file. Comes from :class:`~mechabrain.discovery.VaultPaths`
            in real use -- never assembled from a literal (R4.2).
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @classmethod
    def for_vault(cls, paths: VaultPaths) -> "AccessLog":
        """The access log of the vault described by ``paths``."""
        return cls(paths.access_file)

    @property
    def path(self) -> Path:
        """The live log file. May not exist: an unread vault logs nothing."""
        return self._path

    # ── Recording (hot path) ────────────────────────────────────────
    def record(
        self,
        note_ids: Iterable[str | Path] | str | Path,
        kind: AccessKind | str,
    ) -> int:
        """Append one access record for ``note_ids``. Called by `search`/`get`.

        Lock-free and unsynced by design -- see the module docstring. Ids are
        normalised through :func:`~mechabrain.note.note_id_for`, so a path, a
        basename and a bare id are all accepted and stored identically.

        Args:
            note_ids: The notes read. A lone string or path counts as one note,
                not as a sequence of characters. Duplicates collapse; order is
                not preserved.
            kind: Which read path this was. Required: a default would let a
                caller record a `get` as a `search` by omission.

        Returns:
            How many distinct notes were recorded. ``0`` touches no disk.

        Raises:
            ValueError: ``kind`` is not an :class:`AccessKind`.
            OSError: the log could not be written -- the index directory is
                gone or unwritable. Not swallowed: a silently dead access log
                would look like a vault nobody reads and decay the whole thing.
        """
        resolved = AccessKind.parse(kind)
        ids = _normalize_ids(note_ids)
        if not ids:
            return 0
        stamp = _now().isoformat(timespec="seconds")
        self._append(_pack(resolved, stamp, ids))
        return len(ids)

    def _append(self, lines: Sequence[bytes]) -> None:
        """Append pre-encoded ``lines``, one ``os.write`` each (R7.2)."""
        try:
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        except FileNotFoundError:
            # First write of a fresh runtime layer: index/ is gitignored, so a
            # clone has no such directory until something derives state into it.
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            for line in lines:
                os.write(fd, line)
        finally:
            os.close(fd)

    # ── Aggregation (maintenance path) ──────────────────────────────
    def aggregate(self) -> dict[str, date]:
        """Reduce the log to ``{note_id: date of most recent access}`` (R7.3).

        Reads the live log **and** every pending shard, so a flush that crashed
        mid-cycle is not lost. Takes the maximum timestamp per note rather than
        the last line for it: appenders interleave, and a clock that steps
        backwards must not un-read a note.

        Dates are UTC. Decay is measured in tens of days (§9.3), so the local
        midnight a note was read across is noise; agreeing on one timezone
        across the machines that share a vault is not.

        Unparseable lines are skipped rather than raised on -- the one place
        this module bends R5.1. The log is disposable derived state that
        untrusted crashes append to; wedging maintenance over a torn line
        would cost more than the access it holds. A malformed line can only
        ever cost one note one access.

        Returns:
            One entry per note seen. Empty if nothing has been read.
        """
        latest = _scan([self._path, *self.pending_shards()])
        return {
            note_id: moment.astimezone(timezone.utc).date()
            for note_id, moment in latest.items()
        }

    def pending_shards(self) -> list[Path]:
        """Rotated logs awaiting aggregation, oldest name first.

        Non-empty only between a :meth:`rotate` and the flush that consumes it,
        or after a flush died half-way.
        """
        parent = self._path.parent
        if not parent.is_dir():
            return []
        return sorted(parent.glob(f"{self._path.name}.*{PENDING_SUFFIX}"))

    # ── Rotation ────────────────────────────────────────────────────
    def rotate(self) -> Path | None:
        """Move the live log aside to a pending shard, atomically.

        Concurrent appenders are unaffected: their next ``record`` creates a
        fresh live log, and any write racing the rename lands in the shard,
        which is still read. The rename is the whole point -- see the module
        docstring on why this is not a truncate.

        Returns:
            The shard, or ``None`` if there was nothing to rotate.
        """
        with file_lock(self._lock_path, purpose=_LOCK_PURPOSE):
            return self._rotate_locked()

    def _rotate_locked(self) -> Path | None:
        if not self._path.exists():
            return None
        shard = self._new_shard_path()
        os.replace(self._path, shard)
        return shard

    def _new_shard_path(self) -> Path:
        stamp = _now().strftime("%Y%m%dT%H%M%S%f")
        base = f"{self._path.name}.{stamp}-{os.getpid()}"
        shard = self._path.parent / f"{base}{PENDING_SUFFIX}"
        counter = 0
        while shard.exists():
            counter += 1
            shard = self._path.parent / f"{base}-{counter}{PENDING_SUFFIX}"
        return shard

    @contextmanager
    def flush(self) -> Iterator[dict[str, date]]:
        """Rotate, aggregate, and yield the accesses for the caller to apply.

        Step 1 of the consolidation job (§9.1)::

            with log.flush() as accesses:
                for note_id, when in accesses.items():
                    ...  # write last_accessed: -- not this module's business

        The shards are dropped only when the block returns cleanly. If it
        raises, they stay and the next flush re-yields them: an access is
        applied at least once, never zero times. That is safe precisely because
        writing the same ``last_accessed`` twice changes nothing.

        Held under a file lock, so two consolidators cannot interleave halves
        of one cycle (R7.4).

        Yields:
            ``{note_id: date}`` as per :meth:`aggregate`. Empty if unread.
        """
        with file_lock(self._lock_path, purpose=_LOCK_PURPOSE):
            self._rotate_locked()
            shards = self.pending_shards()
            yield {
                note_id: moment.astimezone(timezone.utc).date()
                for note_id, moment in _scan(shards).items()
            }
            for shard in shards:
                shard.unlink(missing_ok=True)

    def clear(self) -> None:
        """Delete the log and every pending shard, dropping unapplied accesses.

        For ``reindex --full`` and for tests. Not part of a flush: a flush that
        cleared instead of rotating would lose whatever arrived while it ran.
        """
        with file_lock(self._lock_path, purpose=_LOCK_PURPOSE):
            self._path.unlink(missing_ok=True)
            for shard in self.pending_shards():
                shard.unlink(missing_ok=True)

    @property
    def _lock_path(self) -> Path:
        return self._path.with_suffix(".lock")

    def __repr__(self) -> str:
        return f"AccessLog({str(self._path)!r})"


# ══════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════
def _normalize_ids(note_ids: Iterable[str | Path] | str | Path) -> list[str]:
    """Coerce ids/paths to a de-duplicated list of note ids, order preserved.

    A bare ``str`` is one id: iterating it into characters is the footgun this
    guard exists for.
    """
    if isinstance(note_ids, (str, Path)):
        items: Iterable[str | Path] = [note_ids]
    else:
        items = note_ids
    seen: dict[str, None] = {}
    for item in items:
        note_id = note_id_for(item)
        if note_id:
            seen.setdefault(note_id, None)
    return list(seen)


def _encode(kind: AccessKind, stamp: str, ids: Sequence[str]) -> bytes:
    """One JSONL record. Compact: this file grows by a line per query."""
    payload = {"ts": stamp, "kind": kind.value, "ids": list(ids)}
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{line}\n".encode("utf-8")


def _pack(kind: AccessKind, stamp: str, ids: Sequence[str]) -> list[bytes]:
    """Split one access into lines that each fit :data:`MAX_APPEND_BYTES`.

    Splitting is lossless for the only consumer that matters: :meth:`aggregate`
    reduces over ``(id, ts)`` pairs, and every line carries the same ``ts``.
    A single id larger than the budget still gets its own (oversized) line --
    truncating an id would corrupt the record it exists to identify.
    """
    budget = MAX_APPEND_BYTES - len(_encode(kind, stamp, []))
    lines: list[bytes] = []
    chunk: list[str] = []
    used = 0
    for note_id in ids:
        cost = len(json.dumps(note_id, ensure_ascii=False).encode("utf-8")) + 1
        if chunk and used + cost > budget:
            lines.append(_encode(kind, stamp, chunk))
            chunk, used = [], 0
        chunk.append(note_id)
        used += cost
    if chunk:
        lines.append(_encode(kind, stamp, chunk))
    return lines


# ══════════════════════════════════════════════════════════════════════
# Decoding
# ══════════════════════════════════════════════════════════════════════
def _scan(files: Iterable[Path]) -> dict[str, datetime]:
    """Reduce ``files`` to ``{note_id: latest timestamp}``, skipping bad lines."""
    latest: dict[str, datetime] = {}
    for path in files:
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            continue
        with handle:
            for line in handle:
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                moment, ids = parsed
                for note_id in ids:
                    current = latest.get(note_id)
                    if current is None or moment > current:
                        latest[note_id] = moment
    return latest


def _parse_line(line: str) -> tuple[datetime, list[str]] | None:
    """Parse one JSONL record into ``(timestamp, ids)``, or ``None`` if unusable.

    Tolerant on purpose: a torn line from a crashed write, or a line from a
    future kernel carrying keys this one has never heard of, must not stop the
    other million from being read.
    """
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    moment = _parse_timestamp(payload.get("ts"))
    if moment is None:
        return None
    raw_ids = payload.get("ids")
    if not isinstance(raw_ids, list):
        return None
    ids = [item for item in raw_ids if isinstance(item, str) and item]
    return (moment, ids) if ids else None


def _parse_timestamp(raw: object) -> datetime | None:
    """Parse an ISO-8601 stamp, assuming UTC when it carries no offset."""
    if not isinstance(raw, str):
        return None
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _now() -> datetime:
    """Current instant, UTC-aware. Seam for tests; the only clock this module reads."""
    return datetime.now(timezone.utc)
