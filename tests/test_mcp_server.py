"""Wiring tests for the MCP daemon surface (§7, R7.4).

Regression origin: `mechabrain serve` shipped importing a module that did not
exist (``mechabrain.server``) and the failure was masked by a friendly
fallback message -- a daemon that cannot start must fail loudly, and the CLI
must be wired to the real entry point in :mod:`mechabrain.mcp_server`.
"""

from __future__ import annotations

from typing import Any

import pytest

import mechabrain.mcp_server as mcp_server_module
from mechabrain.cli import _run_daemon
from mechabrain.discovery import VaultPaths
from mechabrain.manifest import Manifest
from mechabrain.mcp_server import _strip_wikilink


# ══════════════════════════════════════════════════════════════════════
# The CLI is wired to the real daemon entry point
# ══════════════════════════════════════════════════════════════════════
def test_the_daemon_entry_point_exists() -> None:
    """The module the CLI imports lazily must expose ``serve`` (R7.4)."""
    assert callable(mcp_server_module.serve)


@pytest.mark.parametrize(("stdio", "transport"), [(False, "sse"), (True, "stdio")])
def test_run_daemon_calls_the_real_serve(
    tmp_vault: VaultPaths,
    manifest_ci: Manifest,
    monkeypatch: pytest.MonkeyPatch,
    stdio: bool,
    transport: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_serve(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(mcp_server_module, "serve", fake_serve)
    code = _run_daemon(
        tmp_vault, manifest_ci, host="127.0.0.1", port=39999, stdio=stdio, emit=None
    )

    assert code == 0
    assert calls == [
        {
            "vault": tmp_vault.root,
            "host": "127.0.0.1",
            "port": 39999,
            "transport": transport,
        }
    ]


# ══════════════════════════════════════════════════════════════════════
# memory.get accepts every wikilink form the authored graph understands
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nota-x", "nota-x"),
        ("[[nota-x]]", "nota-x"),
        ("[[nota-x|um alias]]", "nota-x"),
        ("[[nota-x#Um Heading]]", "nota-x"),
        ("[[nota-x#Um Heading|alias]]", "nota-x"),
        ("  [[ nota-x ]]  ", "nota-x"),
    ],
)
def test_strip_wikilink_handles_alias_and_heading(text: str, expected: str) -> None:
    assert _strip_wikilink(text) == expected


# ══════════════════════════════════════════════════════════════════════
# memory.status warns before brute force stops being the right default
# ══════════════════════════════════════════════════════════════════════
class _CountingStore:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


def _status_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_vault: VaultPaths,
    manifest: Manifest,
    chunks: int,
) -> dict[str, Any]:
    monkeypatch.setattr(
        mcp_server_module, "store_from_manifest", lambda *a, **k: _CountingStore(chunks)
    )
    service = mcp_server_module.MemoryService(tmp_vault, manifest)
    try:
        return service.memory_status()["index"]
    finally:
        service.close()


def test_status_warns_past_the_numpy_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    index = _status_index(
        monkeypatch, tmp_vault, manifest_ci, mcp_server_module.NUMPY_CHUNKS_WARNING + 1
    )
    assert "warning" in index
    assert "lancedb" in index["warning"]


def test_status_is_quiet_below_the_numpy_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    index = _status_index(
        monkeypatch, tmp_vault, manifest_ci, mcp_server_module.NUMPY_CHUNKS_WARNING
    )
    assert "warning" not in index


# ══════════════════════════════════════════════════════════════════════
# serve() warns on boot when the derived docs lag the manifest (§10)
# ══════════════════════════════════════════════════════════════════════
def test_boot_warns_when_derived_docs_are_stale(
    tmp_vault: VaultPaths,
    manifest_ci: Manifest,
    manifest_data_ci: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import copy
    import logging

    drifted = copy.deepcopy(manifest_data_ci)
    drifted["maintenance"] = {**(drifted.get("maintenance") or {}), "decay_days": 45}
    stale_manifest = Manifest.from_mapping(drifted)

    with caplog.at_level(logging.WARNING, logger="mechabrain.mcp_server"):
        mcp_server_module._warn_on_stale_docs(tmp_vault, stale_manifest)
    assert any("mechabrain sync" in record.getMessage() for record in caplog.records)


def test_boot_is_quiet_when_derived_docs_match(
    tmp_vault: VaultPaths, manifest_ci: Manifest, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="mechabrain.mcp_server"):
        mcp_server_module._warn_on_stale_docs(tmp_vault, manifest_ci)
    assert not caplog.records
