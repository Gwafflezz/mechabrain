"""The kernel action log (v0.2.1): observability, never behaviour."""

from __future__ import annotations

from typing import Any

from mechabrain.actions import ACTION_KINDS, ActionLog
from mechabrain.discovery import VaultPaths
from mechabrain.manifest import Manifest
from mechabrain.mcp_server import MemoryService


# ══════════════════════════════════════════════════════════════════════
# The log itself
# ══════════════════════════════════════════════════════════════════════
def test_record_appends_and_tail_reads_in_order(tmp_vault: VaultPaths) -> None:
    log = ActionLog.for_vault(tmp_vault)
    log.record("write_accepted", id="a", agent="alpha")
    log.record("write_rejected", reason="dup", agent="beta")

    entries = log.tail()
    assert [e["action"] for e in entries] == ["write_accepted", "write_rejected"]
    assert entries[0]["id"] == "a"
    assert all("ts" in e for e in entries)


def test_tail_keeps_only_the_most_recent(tmp_vault: VaultPaths) -> None:
    log = ActionLog.for_vault(tmp_vault)
    for i in range(7):
        log.record("link", a=f"n{i}", b="m")
    assert [e["a"] for e in log.tail(3)] == ["n4", "n5", "n6"]


def test_empty_and_none_fields_are_dropped(tmp_vault: VaultPaths) -> None:
    log = ActionLog.for_vault(tmp_vault)
    log.record("proposal", target="x", agent="", id=None, superseded=[])
    entry = log.tail(1)[0]
    assert entry["target"] == "x"
    assert "agent" not in entry and "id" not in entry and "superseded" not in entry


def test_torn_line_is_skipped_not_raised(tmp_vault: VaultPaths) -> None:
    log = ActionLog.for_vault(tmp_vault)
    log.record("link", a="a", b="b")
    with open(log.path, "ab") as fh:
        fh.write(b'{"ts": "2026-07-18T00:00:00+00:00", "action": "wri')  # torn append
    log.record("link", a="c", b="d")
    assert [e["a"] for e in log.tail()] == ["a", "c"]


def test_missing_log_reads_as_empty(tmp_vault: VaultPaths) -> None:
    assert ActionLog.for_vault(tmp_vault).tail() == []


def test_first_record_creates_the_runtime_dir(tmp_vault: VaultPaths) -> None:
    """A fresh clone has no _meta/index/ -- the first action must not crash."""
    import shutil

    shutil.rmtree(tmp_vault.index_dir, ignore_errors=True)
    ActionLog.for_vault(tmp_vault).record("write_accepted", id="x")
    assert ActionLog.for_vault(tmp_vault).tail(1)[0]["id"] == "x"


# ══════════════════════════════════════════════════════════════════════
# Wiring: the MCP service records what it does
# ══════════════════════════════════════════════════════════════════════
def _service_write(tmp_vault: VaultPaths, manifest: Manifest, meta: dict[str, Any]) -> dict[str, Any]:
    with MemoryService(tmp_vault, manifest) as service:
        return service.memory_write(
            "semantic", "Brute force wins at this scale.", meta
        )


def test_accepted_write_is_logged(tmp_vault: VaultPaths, manifest_ci: Manifest) -> None:
    result = _service_write(
        tmp_vault,
        manifest_ci,
        {"title": "A fact", "agent": "alpha", "scope": "proj-a", "source": "s", "confidence": "medium"},
    )
    assert result["rejected"] is False
    entry = ActionLog.for_vault(tmp_vault).tail(1)[0]
    assert entry["action"] == "write_accepted"
    assert entry["id"] == result["id"]
    assert entry["agent"] == "alpha" and entry["scope"] == "proj-a"


def test_rejected_write_is_logged_with_the_reason(
    tmp_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    """The rejection is the one fact only this log holds: no note is written."""
    result = _service_write(
        tmp_vault,
        manifest_ci,
        {"title": "No scope", "agent": "alpha", "scope": "not-a-scope", "source": "s"},
    )
    assert result["rejected"] is True
    entry = ActionLog.for_vault(tmp_vault).tail(1)[0]
    assert entry["action"] == "write_rejected"
    assert "scope" in entry["reason"]


def test_proposal_and_link_are_logged(tmp_vault: VaultPaths, manifest_ci: Manifest) -> None:
    with MemoryService(tmp_vault, manifest_ci) as service:
        first = service.memory_write(
            "semantic", "Fact one.", {"title": "One", "agent": "alpha", "scope": "proj-a", "source": "s"}
        )
        second = service.memory_write(
            "semantic", "An unrelated second fact about lexical search.", {"title": "Two", "agent": "alpha", "scope": "proj-b", "source": "s"}
        )
        service.memory_propose("Notes/human.md", "change it", "because", {"agent": "alpha"})
        service.memory_link(first["id"], second["id"], "related", "alpha")

    actions = [e["action"] for e in ActionLog.for_vault(tmp_vault).tail()]
    assert actions[-2:] == ["proposal", "link"]
    assert set(actions) <= set(ACTION_KINDS)
