"""Tests for the command-line interface, :mod:`mechabrain.cli` (§10).

The heavy engines (indexing, consolidation, the MCP daemon) are exercised
through the CLI only enough to prove it wires them correctly; each has its own
test module. ``reindex`` and ``consolidate`` run against ``tmp_vault``, whose
manifest uses the ``hash`` embedder and the ``numpy`` store, so they need no
model download. ``serve`` delegates to an overridable seam, so no real server
is started.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from mechabrain import cli
from mechabrain.cli import build_parser, main
from mechabrain.contract import (
    GITIGNORE_INDEX_ENTRY,
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
)
from mechabrain.discovery import VAULT_ENV_VAR, VaultPaths
from mechabrain.manifest import Manifest
from mechabrain.note import write_atomic


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def edit_manifest(paths: VaultPaths, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Load the on-disk manifest, apply ``mutate`` to the raw mapping, write it back."""
    data = yaml.safe_load(paths.config_file.read_text(encoding="utf-8"))
    mutate(data)
    write_atomic(paths.config_file, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


# ══════════════════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════════════════
def test_parser_exposes_exactly_the_spec_subcommands() -> None:
    parser = build_parser()
    subactions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subactions) == 1
    assert set(subactions[0].choices) == {
        "init",
        "sync",
        "serve",
        "reindex",
        "consolidate",
        "check",
    }


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "mechabrain" in capsys.readouterr().out


def test_no_subcommand_is_an_argparse_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_every_subcommand_accepts_vault(tmp_vault: VaultPaths) -> None:
    parser = build_parser()
    for command in ("init", "sync", "serve", "reindex", "consolidate", "check"):
        args = parser.parse_args([command, "--vault", str(tmp_vault.root)])
        assert args.vault == str(tmp_vault.root)


# ══════════════════════════════════════════════════════════════════════
# init — creation
# ══════════════════════════════════════════════════════════════════════
def test_init_creates_full_skeleton(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "fresh"
    assert main(["init", str(root)]) == 0

    paths = VaultPaths.for_root(root)
    for directory in paths.contract_dirs():
        assert directory.is_dir(), directory
    assert paths.config_file.is_file()
    assert paths.schema_file.is_file()
    assert paths.agents_file.is_file()
    assert paths.index_file.is_file()
    assert paths.hot_file.is_file()

    agents_md = paths.agents_file.read_text(encoding="utf-8")
    assert MANAGED_BLOCK_BEGIN in agents_md
    assert MANAGED_BLOCK_END in agents_md

    # A clean install carries the escriba: the Mecha-Scribe skill is deployed as a
    # git-carried, vault-agnostic artifact (spec §11, Classe B).
    skill = root / ".claude" / "skills" / "mecha-scribe" / "SKILL.md"
    assert skill.is_file()
    assert "name: mecha-scribe" in skill.read_text(encoding="utf-8")

    # The default manifest registers one agent ("exemplo"); its Episodic/ is made.
    manifest = Manifest.load(paths.config_file)
    for agent_id in manifest.agent_ids():
        assert paths.episodic_for(agent_id).is_dir()
    assert manifest.agent_ids()  # a fresh manifest is not empty


def test_init_adds_gitignore_entry(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    assert main(["init", str(root)]) == 0

    gitignore = root / ".gitignore"
    assert gitignore.is_file()
    lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
    assert GITIGNORE_INDEX_ENTRY in lines


def test_init_appends_to_existing_gitignore_without_duplicating(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    root.mkdir()
    (root / ".gitignore").write_text(".obsidian/\n", encoding="utf-8")

    assert main(["init", str(root)]) == 0
    first = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".obsidian/" in first
    assert first.count(GITIGNORE_INDEX_ENTRY) == 1

    assert main(["init", str(root)]) == 0  # second run must not add it again
    second = (root / ".gitignore").read_text(encoding="utf-8")
    assert second.count(GITIGNORE_INDEX_ENTRY) == 1


def test_init_prints_integration_snippet_and_registration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", str(tmp_path / "fresh")]) == 0
    out = capsys.readouterr().out
    # The pasteable snippet points agents at the real contract and the tools.
    assert "AGENTS.md" in out
    assert "memory_search" in out
    # The MCP registration guidance names the daemon and its transport.
    assert "mechabrain serve" in out
    assert "/sse" in out


def test_init_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "fresh"
    assert main(["init", str(root), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["vault"] == str(root)
    assert payload["gitignore_added"] is True
    assert payload["config_kept"] is False
    assert isinstance(payload["agents"], list)


def test_init_resolves_target_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "from-env"
    monkeypatch.setenv(VAULT_ENV_VAR, str(root))
    assert main(["init"]) == 0
    assert VaultPaths.for_root(root).is_initialized


def test_init_positional_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(VAULT_ENV_VAR, str(tmp_path / "env-vault"))
    target = tmp_path / "positional-vault"
    assert main(["init", str(target)]) == 0
    assert VaultPaths.for_root(target).is_initialized
    assert not (tmp_path / "env-vault").exists()


# ══════════════════════════════════════════════════════════════════════
# init — idempotence
# ══════════════════════════════════════════════════════════════════════
def test_init_is_idempotent_and_preserves_edits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "vault"
    assert main(["init", str(root)]) == 0
    paths = VaultPaths.for_root(root)

    # Simulate a running deployment: a human section in AGENTS.md, an edited
    # manifest value, a written memory, and a consolidator-populated index.md.
    human_section = "\n\n## Notas do humano\n\nconteúdo que não pode sumir\n"
    paths.agents_file.write_text(
        paths.agents_file.read_text(encoding="utf-8") + human_section, encoding="utf-8"
    )
    config_text = paths.config_file.read_text(encoding="utf-8").replace(
        "decay_days: 90", "decay_days: 45"
    )
    paths.config_file.write_text(config_text, encoding="utf-8")
    note = paths.semantic_dir / "2026-01-15_INS_kept.md"
    note.write_text("---\ntitle: Kept\n---\n\nbody stays\n", encoding="utf-8")
    paths.index_file.write_text("MAPA CUSTOMIZADO DO CONSOLIDADOR\n", encoding="utf-8")

    capsys.readouterr()  # drop the first run's output
    assert main(["init", str(root)]) == 0

    agents_md = paths.agents_file.read_text(encoding="utf-8")
    assert "## Notas do humano" in agents_md
    assert "conteúdo que não pode sumir" in agents_md
    assert MANAGED_BLOCK_BEGIN in agents_md  # the block is still there and regenerated
    assert "45" in agents_md  # regenerated from the edited manifest

    assert "decay_days: 45" in paths.config_file.read_text(encoding="utf-8")
    assert note.read_text(encoding="utf-8") == "---\ntitle: Kept\n---\n\nbody stays\n"
    assert paths.index_file.read_text(encoding="utf-8") == "MAPA CUSTOMIZADO DO CONSOLIDADOR\n"

    assert "already present" in capsys.readouterr().out  # gitignore not re-added


def test_init_does_not_resurrect_a_disabled_research_folder(tmp_path: Path) -> None:
    """Research/ is the one switchable §3 folder; a re-run honours the manifest."""
    root = tmp_path / "vault"
    assert main(["init", str(root)]) == 0
    paths = VaultPaths.for_root(root)

    config_text = paths.config_file.read_text(encoding="utf-8").replace(
        "research_enabled: true", "research_enabled: false"
    )
    assert "research_enabled: false" in config_text  # the default config carries the key
    paths.config_file.write_text(config_text, encoding="utf-8")
    paths.research_dir.rmdir()

    assert main(["init", str(root)]) == 0
    assert not paths.research_dir.exists()


def test_init_reports_invalid_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "vault"
    assert main(["init", str(root)]) == 0
    paths = VaultPaths.for_root(root)
    paths.config_file.write_text("mecha_brain:\n  bogus_key: 1\n", encoding="utf-8")

    assert main(["init", str(root)]) == 1
    assert "mechabrain:" in capsys.readouterr().err


# ══════════════════════════════════════════════════════════════════════
# sync
# ══════════════════════════════════════════════════════════════════════
def test_sync_regenerates_and_adds_new_episodic_folders(
    tmp_vault: VaultPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    edit_manifest(
        tmp_vault,
        lambda data: data["agents"].append({"id": "gamma", "display_name": "Gamma"}),
    )
    assert not tmp_vault.episodic_for("gamma").exists()

    assert main(["sync", "--vault", str(tmp_vault.root)]) == 0

    assert tmp_vault.episodic_for("gamma").is_dir()
    assert tmp_vault.agents_file.is_file()
    assert tmp_vault.schema_file.is_file()
    assert "gamma" in tmp_vault.agents_file.read_text(encoding="utf-8")
    assert "gamma" in capsys.readouterr().out


def test_sync_preserves_free_form_agents_md(tmp_vault: VaultPaths) -> None:
    # A prior sync writes the managed block; a human adds a section after it.
    assert main(["sync", "--vault", str(tmp_vault.root)]) == 0
    tmp_vault.agents_file.write_text(
        tmp_vault.agents_file.read_text(encoding="utf-8") + "\n\n## Livre\n\ntexto\n",
        encoding="utf-8",
    )
    assert main(["sync", "--vault", str(tmp_vault.root)]) == 0
    assert "## Livre" in tmp_vault.agents_file.read_text(encoding="utf-8")


def test_sync_json(tmp_vault: VaultPaths, capsys: pytest.CaptureFixture[str]) -> None:

    assert main(["sync", "--vault", str(tmp_vault.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["agents"]) == {"alpha", "beta"}


# ══════════════════════════════════════════════════════════════════════
# reindex
# ══════════════════════════════════════════════════════════════════════
def test_reindex_full_indexes_sample_notes(
    tmp_vault: VaultPaths, sample_notes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["reindex", "--full", "--vault", str(tmp_vault.root)]) == 0
    out = capsys.readouterr().out
    assert "full rebuild" in out
    assert (tmp_vault.index_dir).is_dir()


def test_reindex_json_report(
    tmp_vault: VaultPaths, sample_notes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:

    assert main(["reindex", "--vault", str(tmp_vault.root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "chunks_total" in payload
    assert payload["notes_total"] >= 1


# ══════════════════════════════════════════════════════════════════════
# consolidate
# ══════════════════════════════════════════════════════════════════════
def test_consolidate_runs_pipeline(
    tmp_vault: VaultPaths, sample_notes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["consolidate", "--vault", str(tmp_vault.root)]) == 0
    out = capsys.readouterr().out
    assert "consolidate" in out
    # The vault is not a git repo, so nothing is committed but the pass completes.
    assert "committed" in out


def test_consolidate_dry_run_writes_nothing(
    tmp_vault: VaultPaths, sample_notes: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:

    from mechabrain.consolidate import CONSOLIDATION_REPORT_FILE

    assert main(["consolidate", "--vault", str(tmp_vault.root), "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert not (tmp_vault.index_dir / CONSOLIDATION_REPORT_FILE).exists()


# ══════════════════════════════════════════════════════════════════════
# check
# ══════════════════════════════════════════════════════════════════════
def test_check_passes_on_fresh_vault(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "vault"
    assert main(["init", str(root)]) == 0
    capsys.readouterr()
    assert main(["check", "--vault", str(root)]) == 0


def test_check_fails_on_denylisted_note(
    tmp_vault: VaultPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    # manifest_ci denies the key "forbidden-key".
    bad = tmp_vault.semantic_dir / "2026-01-15_INS_bad.md"
    bad.write_text(
        "---\n"
        "title: Bad\n"
        "tags: [mem/semantic, agent/alpha]\n"
        "agent: alpha\n"
        "scope: proj-a\n"
        "forbidden-key: leaked\n"
        "---\n\nbody\n",
        encoding="utf-8",
    )
    assert main(["check", "--vault", str(tmp_vault.root)]) == 1
    assert "denylist" in capsys.readouterr().out.lower()


def test_check_json_reports_exit_code(
    tmp_vault: VaultPaths, capsys: pytest.CaptureFixture[str]
) -> None:

    code = main(["check", "--vault", str(tmp_vault.root), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == code
    assert payload["ok"] is (code == 0)


# ══════════════════════════════════════════════════════════════════════
# serve
# ══════════════════════════════════════════════════════════════════════
class _DaemonRecorder:
    """Stand-in for the MCP daemon that records how ``serve`` invoked it."""

    def __init__(self, returns: int | None = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.returns = returns

    def __call__(
        self,
        paths: VaultPaths,
        manifest: Manifest,
        *,
        host: str,
        port: int,
        stdio: bool,
        emit: Any,
    ) -> int | None:
        self.calls.append(
            {"root": paths.root, "manifest": manifest, "host": host, "port": port, "stdio": stdio}
        )
        return self.returns


def test_serve_delegates_with_defaults(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _DaemonRecorder()
    monkeypatch.setattr(cli, "_run_daemon", recorder)

    assert main(["serve", "--vault", str(tmp_vault.root)]) == 0
    (call,) = recorder.calls
    assert call["root"] == tmp_vault.root
    assert isinstance(call["manifest"], Manifest)
    assert call["host"] == cli.DEFAULT_HOST
    assert call["port"] == cli.DEFAULT_PORT
    assert call["stdio"] is False


def test_serve_port_from_env(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _DaemonRecorder()
    monkeypatch.setattr(cli, "_run_daemon", recorder)
    monkeypatch.setenv(cli.PORT_ENV_VAR, "9191")

    assert main(["serve", "--vault", str(tmp_vault.root)]) == 0
    assert recorder.calls[0]["port"] == 9191


def test_serve_flag_beats_env_port(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _DaemonRecorder()
    monkeypatch.setattr(cli, "_run_daemon", recorder)
    monkeypatch.setenv(cli.PORT_ENV_VAR, "9191")

    assert main(["serve", "--vault", str(tmp_vault.root), "--port", "5050", "--host", "0.0.0.0"]) == 0
    assert recorder.calls[0]["port"] == 5050
    assert recorder.calls[0]["host"] == "0.0.0.0"


def test_serve_stdio_warns(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = _DaemonRecorder()
    monkeypatch.setattr(cli, "_run_daemon", recorder)

    assert main(["serve", "--vault", str(tmp_vault.root), "--stdio"]) == 0
    assert recorder.calls[0]["stdio"] is True
    assert "multiple writers" in capsys.readouterr().err


def test_serve_rejects_non_integer_env_port(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_run_daemon", _DaemonRecorder())
    monkeypatch.setenv(cli.PORT_ENV_VAR, "not-a-port")

    assert main(["serve", "--vault", str(tmp_vault.root)]) == 1
    assert cli.PORT_ENV_VAR in capsys.readouterr().err


def test_serve_reports_missing_daemon_dependency(
    tmp_vault: VaultPaths,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A broken `mcp` install fails loudly with the R7.4 hint, exit 1.

    Regression origin: the CLI once imported a module that did not exist and a
    friendly fallback masked it as "component not available" -- so this also
    guards that the *real* module import is what the fallback protects.
    Setting the ``sys.modules`` entry to ``None`` makes the lazy
    ``from .mcp_server import serve`` raise ``ImportError`` exactly as a
    missing dependency would.
    """
    import sys

    monkeypatch.setitem(sys.modules, "mechabrain.mcp_server", None)
    assert main(["serve", "--vault", str(tmp_vault.root)]) == 1
    err = capsys.readouterr().err
    assert "R7.4" in err and "mcp" in err


# ══════════════════════════════════════════════════════════════════════
# Discovery (R4.3)
# ══════════════════════════════════════════════════════════════════════
def test_command_discovers_vault_by_walking_up(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested = tmp_vault.semantic_dir
    monkeypatch.chdir(nested)
    assert main(["check"]) == 0


def test_command_discovers_vault_from_env(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(VAULT_ENV_VAR, str(tmp_vault.root))
    assert main(["check"]) == 0


def test_missing_vault_is_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "not-a-vault"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert main(["check"]) == 1
    err = capsys.readouterr().err
    assert "no vault found" in err
    assert "mechabrain init" in err  # the hint teaches the fix
