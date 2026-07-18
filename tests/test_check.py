"""Tests for the deployment lint, :mod:`mechabrain.check` (§10).

The git-ignore probe is injected in most tests so they need no real repository;
a handful at the end exercise the real ``git check-ignore`` backend and skip
when git is absent.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from mechabrain.check import (
    CheckReport,
    GitIgnoreProbe,
    Problem,
    Severity,
    check,
)
from mechabrain.discovery import VaultPaths
from mechabrain.note import Note, write_atomic

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def good_git(paths: VaultPaths) -> GitIgnoreProbe:
    """A probe for a correctly-configured repo: index ignored, links not."""

    def probe(path: Path) -> bool | None:
        if path == paths.links_file:
            return False
        return path == paths.index_dir or paths.index_dir in path.parents

    return probe


def no_git(_paths: VaultPaths) -> GitIgnoreProbe:
    """A probe standing in for a vault that is not a git work tree."""

    def probe(_path: Path) -> bool | None:
        return None

    return probe


def check_ids(report: CheckReport) -> set[str]:
    return {problem.check for problem in report.problems}


def only(report: CheckReport, check_id: str) -> Problem:
    """The single problem with ``check_id``; fails if absent or duplicated."""
    matches = [problem for problem in report.problems if problem.check == check_id]
    assert len(matches) == 1, f"expected exactly one {check_id!r}, got {check_ids(report)}"
    return matches[0]


def good_frontmatter(**overrides: Any) -> dict[str, Any]:
    """Valid §6 frontmatter for an ``alpha`` semantic note in ``proj-a``."""
    frontmatter: dict[str, Any] = {
        "title": "A fact",
        "tags": ["mem/semantic", "agent/alpha"],
        "created": date(2026, 1, 15),
        "modified": date(2026, 1, 15),
        "agent": "alpha",
        "scope": "proj-a",
        "source": "test-session",
        "confidence": "medium",
        "status": "ativo",
    }
    frontmatter.update(overrides)
    return frontmatter


@pytest.fixture
def semantic_note(
    tmp_vault: VaultPaths, write_note: Callable[..., Note]
) -> Callable[..., Note]:
    """Write one semantic note into ``tmp_vault`` with the given frontmatter."""

    def _write(frontmatter: Mapping[str, Any], name: str = "2026-01-15_INS_x.md") -> Note:
        return write_note(tmp_vault.semantic_dir / name, dict(frontmatter), "A body.")

    return _write


# ══════════════════════════════════════════════════════════════════════
# Report shape
# ══════════════════════════════════════════════════════════════════════
def test_clean_vault_passes(tmp_vault: VaultPaths, sample_notes: list[Note]) -> None:
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert report.ok
    assert report.exit_code == 0
    assert report.problems == ()
    assert "OK" in report.render()


def test_empty_vault_passes(tmp_vault: VaultPaths) -> None:
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert report.ok
    assert report.problems == ()


# ══════════════════════════════════════════════════════════════════════
# Derived docs (§10, R6.4)
# ══════════════════════════════════════════════════════════════════════
def _edit_config(paths: VaultPaths) -> None:
    """A valid config edit (decay_days) made without running `mechabrain sync`."""
    data = yaml.safe_load(paths.config_file.read_text(encoding="utf-8"))
    data["maintenance"] = {**(data.get("maintenance") or {}), "decay_days": 45}
    write_atomic(paths.config_file, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def test_edited_config_marks_derived_docs_stale(tmp_vault: VaultPaths) -> None:
    """The §10 drift: config.yaml changed, sync not run -- agents read old rules."""
    _edit_config(tmp_vault)
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))

    assert report.ok, "stale docs are a warning, not a failed build"
    assert only(report, "schema_stale").severity is Severity.WARNING
    stale = only(report, "agents_md_stale")
    assert stale.severity is Severity.WARNING
    assert stale.hint is not None and "sync" in stale.hint


def test_missing_schema_reads_as_stale(tmp_vault: VaultPaths) -> None:
    tmp_vault.schema_file.unlink()
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert "schema_stale" in check_ids(report)
    assert "agents_md_stale" not in check_ids(report)


def test_ambiguous_agents_markers_are_an_error(tmp_vault: VaultPaths) -> None:
    """A duplicated marker breaks `mechabrain sync` itself -- that fails the build."""
    agents = tmp_vault.agents_file.read_text(encoding="utf-8")
    write_atomic(tmp_vault.agents_file, agents + "\n<!-- mechabrain:begin -->\n")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))

    assert not report.ok
    problem = only(report, "agents_md_markers_ambiguous")
    assert problem.severity is Severity.ERROR
    assert "agents_md_stale" not in check_ids(report), "ambiguity precludes a staleness verdict"


def test_hand_written_sections_do_not_read_as_drift(tmp_vault: VaultPaths) -> None:
    """§10: the kernel owns the block and nothing else -- a human section is not stale."""
    agents = tmp_vault.agents_file.read_text(encoding="utf-8")
    write_atomic(tmp_vault.agents_file, agents + "\n## Minha seção\n\nTexto humano.\n")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert "agents_md_stale" not in check_ids(report)


def test_ok_ignores_warnings(tmp_vault: VaultPaths) -> None:
    # A warning (git cannot be verified) leaves the deployment passing.
    report = check(tmp_vault, git_ignored=no_git(tmp_vault))
    assert report.ok
    assert report.exit_code == 0
    assert check_ids(report) == {"git_unavailable"}
    assert report.warnings and not report.errors


def test_error_fails_build(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(agent="ghost"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert not report.ok
    assert report.exit_code == 1


# ══════════════════════════════════════════════════════════════════════
# Manifest (R5.1 / R4.5 / R4.3 / R4.2)
# ══════════════════════════════════════════════════════════════════════
def test_manifest_unknown_key_stops(
    make_vault: Callable[..., VaultPaths], manifest_data_ci: dict[str, Any]
) -> None:
    manifest_data_ci["bogus"] = True
    paths = make_vault(manifest_data=manifest_data_ci, name="bad")
    report = check(paths, git_ignored=good_git(paths))
    problem = only(report, "manifest_invalid")
    assert problem.rule == "R5.1"
    assert not report.ok
    # A broken manifest short-circuits: nothing else is even attempted.
    assert len(report.problems) == 1


def test_kernel_too_old_stops(
    make_vault: Callable[..., VaultPaths], manifest_data_ci: dict[str, Any]
) -> None:
    manifest_data_ci["mecha_brain"]["kernel_min_version"] = "99.0.0"
    paths = make_vault(manifest_data=manifest_data_ci, name="future")
    report = check(paths, git_ignored=good_git(paths))
    problem = only(report, "manifest_invalid")
    assert problem.rule == "R4.5"
    assert not report.ok


def test_missing_config_reports_r43(tmp_path: Path) -> None:
    bare = tmp_path / "not-a-vault"
    bare.mkdir()
    report = check(bare)
    problem = only(report, "manifest_invalid")
    assert problem.rule == "R4.3"
    assert not report.ok


def test_manifest_absolute_path_stops(
    make_vault: Callable[..., VaultPaths], manifest_data_ci: dict[str, Any]
) -> None:
    # R4.2 is subsumed by manifest validation: the loader rejects absolute paths,
    # so a manifest that parses holds none.
    manifest_data_ci["zones"]["proposals_dir"] = "/etc/inbox/"
    paths = make_vault(manifest_data=manifest_data_ci, name="abs")
    report = check(paths, git_ignored=good_git(paths))
    problem = only(report, "manifest_invalid")
    assert "/etc/inbox/" in problem.message
    assert not report.ok


# ══════════════════════════════════════════════════════════════════════
# Denylists (R6.1)
# ══════════════════════════════════════════════════════════════════════
def test_denylist_key(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(**{"forbidden-key": "x"}))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "denylist_keys")
    assert problem.rule == "R6.1"
    assert "forbidden-key" in problem.message
    assert problem.path == "mecha-brain/Semantic/2026-01-15_INS_x.md"


def test_denylist_tag(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(tags=["mem/semantic", "agent/alpha", "forbidden/tag"]))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "denylist_tags")
    assert problem.rule == "R6.1"
    assert "forbidden/tag" in problem.message


# ══════════════════════════════════════════════════════════════════════
# Agent and profile (R6.2 / R6.6)
# ══════════════════════════════════════════════════════════════════════
def test_agent_missing(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    fm = good_frontmatter()
    del fm["agent"]
    semantic_note(fm)
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert only(report, "agent_missing").rule == "R6.2"


def test_agent_unknown(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(agent="ghost"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "agent_unknown")
    assert problem.rule == "R6.2"
    assert "ghost" in problem.message


def test_profile_unknown(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    # alpha declares profiles [tutor, planner]; 'ghost' is not one.
    semantic_note(good_frontmatter(profile="ghost"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "profile_unknown")
    assert problem.rule == "R6.6"


def test_profile_of_unknown_agent_is_not_double_reported(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    # An unknown agent must not also trip profile_unknown (profiles_of would raise).
    semantic_note(good_frontmatter(agent="ghost", profile="tutor"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert check_ids(report) == {"agent_unknown"}


def test_valid_profile_passes(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(profile="tutor"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert report.ok


# ══════════════════════════════════════════════════════════════════════
# Scope (R6.5)
# ══════════════════════════════════════════════════════════════════════
def test_scope_missing(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    fm = good_frontmatter()
    del fm["scope"]
    semantic_note(fm)
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert only(report, "scope_missing").rule == "R6.5"


def test_scope_unknown(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(scope="proj-z"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "scope_unknown")
    assert problem.rule == "R6.5"
    assert "proj-z" in problem.message


# ══════════════════════════════════════════════════════════════════════
# Malformed frontmatter (§6)
# ══════════════════════════════════════════════════════════════════════
def test_malformed_frontmatter_reported_not_raised(tmp_vault: VaultPaths) -> None:
    write_atomic(
        tmp_vault.semantic_dir / "2026-01-15_INS_broken.md",
        "---\ntitle: [unclosed\n---\n\nBody.\n",
    )
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "frontmatter_malformed")
    assert problem.rule == "§6"
    assert problem.path.endswith("2026-01-15_INS_broken.md")


# ══════════════════════════════════════════════════════════════════════
# Episodic placement (R6.3 / §3)
# ══════════════════════════════════════════════════════════════════════
def test_episodic_agent_mismatch(
    tmp_vault: VaultPaths, write_note: Callable[..., Note]
) -> None:
    # Registered agent 'beta' filed under alpha's journal.
    fm = good_frontmatter(agent="beta", tags=["mem/episodic", "agent/beta"])
    write_note(tmp_vault.episodic_for("alpha") / "2026-01-15_MEM_x.md", fm, "Happened.")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "episodic_agent_mismatch")
    assert problem.rule == "R6.3"


def test_episodic_unpartitioned(
    tmp_vault: VaultPaths, write_note: Callable[..., Note]
) -> None:
    fm = good_frontmatter(tags=["mem/episodic", "agent/alpha"])
    write_note(tmp_vault.episodic_dir / "2026-01-15_MEM_loose.md", fm, "Happened.")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert only(report, "episodic_unpartitioned").rule == "R6.3"


def test_episodic_correctly_placed_passes(
    tmp_vault: VaultPaths, write_note: Callable[..., Note]
) -> None:
    fm = good_frontmatter(agent="alpha", tags=["mem/episodic", "agent/alpha"])
    write_note(tmp_vault.episodic_for("alpha") / "2026-01-15_MEM_x.md", fm, "Happened.")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert report.ok


# ══════════════════════════════════════════════════════════════════════
# Contract tree (§3) and Episodic subfolders (R6.2)
# ══════════════════════════════════════════════════════════════════════
def test_contract_dir_missing(tmp_vault: VaultPaths) -> None:
    shutil.rmtree(tmp_vault.procedural_dir)
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "contract_dir_missing")
    assert problem.rule == "§3"
    assert "Procedural" in problem.message


def test_research_dir_required_when_enabled(tmp_vault: VaultPaths) -> None:
    # manifest_ci enables research.
    shutil.rmtree(tmp_vault.research_dir)
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert only(report, "contract_dir_missing").path.endswith("Research")


def test_research_dir_optional_when_disabled(
    make_vault: Callable[..., VaultPaths], manifest_data_ci: dict[str, Any]
) -> None:
    manifest_data_ci["zones"]["research_enabled"] = False
    paths = make_vault(manifest_data=manifest_data_ci, name="noresearch")
    shutil.rmtree(paths.research_dir)
    report = check(paths, git_ignored=good_git(paths))
    assert "contract_dir_missing" not in check_ids(report)
    assert report.ok


def test_episodic_subfolder_missing(tmp_vault: VaultPaths) -> None:
    shutil.rmtree(tmp_vault.episodic_for("beta"))
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "episodic_subfolder_missing")
    assert problem.rule == "R6.2"
    assert "beta" in problem.message


# ══════════════════════════════════════════════════════════════════════
# Orphans (§3)
# ══════════════════════════════════════════════════════════════════════
def test_orphan_in_root_is_warning(tmp_vault: VaultPaths) -> None:
    write_atomic(tmp_vault.mecha_brain / "stray.md", "# just sitting here\n")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    problem = only(report, "orphan_note")
    assert problem.severity is Severity.WARNING
    assert problem.rule == "§3"
    assert report.ok  # a warning never fails the build


def test_orphan_in_noncontract_folder(tmp_vault: VaultPaths) -> None:
    write_atomic(tmp_vault.mecha_brain / "Scratch" / "note.md", "text\n")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert only(report, "orphan_note").path.endswith("Scratch/note.md")


def test_generated_files_are_not_orphans(tmp_vault: VaultPaths) -> None:
    for target in (tmp_vault.agents_file, tmp_vault.hot_file, tmp_vault.index_file):
        write_atomic(target, "generated\n")
    write_atomic(tmp_vault.schema_file, "schema\n")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert "orphan_note" not in check_ids(report)


def test_inbox_proposal_is_not_orphan(tmp_vault: VaultPaths) -> None:
    write_atomic(tmp_vault.inbox_dir / "2026-01-15_AI-PROPOSAL_x.md", "proposal\n")
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert "orphan_note" not in check_ids(report)


# ══════════════════════════════════════════════════════════════════════
# Git-ignore (§10), injected probe
# ══════════════════════════════════════════════════════════════════════
def test_index_not_ignored(tmp_vault: VaultPaths) -> None:
    def probe(path: Path) -> bool | None:
        return False  # nothing is ignored

    report = check(tmp_vault, git_ignored=probe)
    problem = only(report, "index_not_ignored")
    assert problem.rule == "§10"
    assert not report.ok


def test_links_ignored(tmp_vault: VaultPaths) -> None:
    def probe(path: Path) -> bool | None:
        return True  # everything is ignored, including links.jsonl

    report = check(tmp_vault, git_ignored=probe)
    problem = only(report, "links_ignored")
    assert problem.rule == "§10"
    assert not report.ok


def test_git_unavailable_is_warning(tmp_vault: VaultPaths) -> None:
    report = check(tmp_vault, git_ignored=no_git(tmp_vault))
    problem = only(report, "git_unavailable")
    assert problem.severity is Severity.WARNING
    assert report.ok


# ══════════════════════════════════════════════════════════════════════
# Git-ignore (§10), real `git check-ignore`
# ══════════════════════════════════════════════════════════════════════
requires_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not installed"
)


def _git_init(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)


@requires_git
def test_real_git_correct_gitignore_passes(tmp_vault: VaultPaths) -> None:
    _git_init(tmp_vault.root)
    write_atomic(tmp_vault.root / ".gitignore", "mecha-brain/_meta/index/\n")
    report = check(tmp_vault)  # default probe -> real git
    assert "git_unavailable" not in check_ids(report)
    assert "index_not_ignored" not in check_ids(report)
    assert "links_ignored" not in check_ids(report)
    assert report.ok


@requires_git
def test_real_git_index_not_ignored(tmp_vault: VaultPaths) -> None:
    _git_init(tmp_vault.root)
    write_atomic(tmp_vault.root / ".gitignore", "# nothing relevant\n")
    report = check(tmp_vault)
    assert "index_not_ignored" in check_ids(report)
    assert not report.ok


@requires_git
def test_real_git_links_ignored(tmp_vault: VaultPaths) -> None:
    _git_init(tmp_vault.root)
    write_atomic(
        tmp_vault.root / ".gitignore",
        "mecha-brain/_meta/index/\nmecha-brain/_meta/links.jsonl\n",
    )
    report = check(tmp_vault)
    assert "links_ignored" in check_ids(report)
    assert not report.ok


@requires_git
def test_real_git_not_a_repo_warns(tmp_vault: VaultPaths) -> None:
    # No `git init`: the vault root is not a work tree.
    report = check(tmp_vault)
    assert only(report, "git_unavailable").severity is Severity.WARNING
    assert report.ok


# ══════════════════════════════════════════════════════════════════════
# Multiple findings and rendering
# ══════════════════════════════════════════════════════════════════════
def test_multiple_findings_accumulate(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(agent="ghost", scope="proj-z"), name="2026-01-15_INS_a.md")
    shutil.rmtree(tmp_vault.procedural_dir)
    report = check(tmp_vault, git_ignored=good_git(tmp_vault))
    assert {"agent_unknown", "scope_unknown", "contract_dir_missing"} <= check_ids(report)
    assert not report.ok


def test_render_lists_rules(
    tmp_vault: VaultPaths, semantic_note: Callable[..., Note]
) -> None:
    semantic_note(good_frontmatter(agent="ghost"))
    rendered = check(tmp_vault, git_ignored=good_git(tmp_vault)).render()
    assert "[R6.2]" in rendered
    assert "error" in rendered.lower()


def test_str_of_problem_carries_rule_and_hint() -> None:
    problem = Problem(
        check="x",
        rule="R6.1",
        message="bad",
        hint="fix it",
        path="mecha-brain/Semantic/a.md",
    )
    text = str(problem)
    assert "[R6.1]" in text
    assert "mecha-brain/Semantic/a.md" in text
    assert "hint: fix it" in text
