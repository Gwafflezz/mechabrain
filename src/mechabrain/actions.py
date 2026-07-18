"""Kernel action log: what the kernel decided, observable after the fact (v0.2.1).

``_meta/index/actions.jsonl`` records the outcome of every agent-facing action
the MCP service executes -- a write accepted or rejected by the §8.2 gate, a
proposal filed, an authored edge recorded. It exists for **observability**: a
dashboard or an operator can answer "what has the kernel been doing?" without
tailing a daemon.

Same design stance as :mod:`mechabrain.access` (R7.2):

* **Runtime layer.** Per machine, gitignored, disposable. Losing it loses
  history, never state -- every fact it records is derivable from the vault
  (the notes that exist) except the rejections, which are the point.
* **Lock-free appends.** One ``os.write`` of one pre-encoded line per action,
  ``O_APPEND``. Concurrent writers interleave whole lines.
* **Tolerant reads.** A torn line is skipped, not raised on -- the log is a
  feed, and wedging a dashboard over one bad line costs more than the line.

The kernel writes it; nothing in the kernel reads it back for behaviour. It is
never an input to any decision (P1: the Markdown is the source of truth).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from .discovery import VaultPaths

__all__ = ["ActionLog", "ACTION_KINDS"]

#: The actions the service records. A dashboard may rely on this vocabulary.
ACTION_KINDS: Final[tuple[str, ...]] = (
    "write_accepted",
    "write_rejected",
    "proposal",
    "link",
)


class ActionLog:
    """The append-only action log at ``_meta/index/actions.jsonl``.

    ::

        log = ActionLog.for_vault(paths)
        log.record("write_accepted", type="semantic", id=note_id, agent="claude")
        recent = log.tail(50)          # newest last, ready for a feed
    """

    __slots__ = ("_path",)

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @classmethod
    def for_vault(cls, paths: VaultPaths) -> "ActionLog":
        """The action log of the vault described by ``paths``."""
        return cls(paths.actions_file)

    @property
    def path(self) -> Path:
        """The live log file. May not exist: a vault with no actions logs nothing."""
        return self._path

    # ── Recording (hot path) ────────────────────────────────────────
    def record(self, action: str, **fields: Any) -> None:
        """Append one action line: ``{"ts", "action", **fields}``.

        ``action`` should be one of :data:`ACTION_KINDS`; unknown kinds are
        accepted (the log is a feed, not a schema) but a typo will not group
        with anything a dashboard knows. ``fields`` must be JSON-serialisable.

        Raises:
            OSError: the log could not be written. Not swallowed -- a silently
                dead action log would read as a kernel that never acts.
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": str(action),
            **{k: v for k, v in fields.items() if v not in (None, "", [], ())},
        }
        line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            fd = os.open(self._path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        except FileNotFoundError:
            # Fresh runtime layer: index/ is gitignored, a clone has none.
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            # Self-healing: a crash mid-append leaves a torn line with no
            # newline, and a plain append would glue this line onto it --
            # losing both. Starting on a fresh line confines the tear to the
            # one action it already lost.
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                line = b"\n" + line
            os.write(fd, line)
        finally:
            os.close(fd)

    # ── Reading (dashboard path) ────────────────────────────────────
    def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        """The most recent ``limit`` entries, oldest first among those kept.

        Tolerant: a torn or malformed line is skipped. Returns ``[]`` when the
        log does not exist yet.
        """
        return list(self._iter())[-max(0, limit):]

    def _iter(self) -> Iterator[dict[str, Any]]:
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue  # torn line: one lost action, never a wedged feed
            if isinstance(entry, dict):
                yield entry
