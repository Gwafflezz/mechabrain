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
