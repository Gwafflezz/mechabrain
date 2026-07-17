"""Vault discovery by convention (R4.3) and derived paths (§3)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mechabrain.contract import MemoryType
from mechabrain.discovery import (
    VAULT_ENV_VAR,
    VaultPaths,
    discover_vault,
    find_vault_root,
    is_vault_root,
)
from mechabrain.errors import VaultNotFoundError


# ── Derived paths (§3) ──────────────────────────────────────────────
def test_for_root_derives_the_whole_contract(tmp_path: Path) -> None:
    paths = VaultPaths.for_root(tmp_path)
    assert paths.root == tmp_path
    assert paths.mecha_brain == tmp_path / "mecha-brain"
    assert paths.semantic_dir == tmp_path / "mecha-brain/Semantic"
    assert paths.episodic_dir == tmp_path / "mecha-brain/Episodic"
    assert paths.procedural_dir == tmp_path / "mecha-brain/Procedural"
    assert paths.research_dir == tmp_path / "mecha-brain/Research"
    assert paths.indices_dir == tmp_path / "mecha-brain/indices"
    assert paths.inbox_dir == tmp_path / "mecha-brain/_inbox"
    assert paths.meta_dir == tmp_path / "mecha-brain/_meta"
    assert paths.index_dir == tmp_path / "mecha-brain/_meta/index"
    assert paths.agents_file == tmp_path / "mecha-brain/AGENTS.md"
    assert paths.hot_file == tmp_path / "mecha-brain/hot.md"
    assert paths.index_file == tmp_path / "mecha-brain/index.md"
    assert paths.config_file == tmp_path / "mecha-brain/_meta/config.yaml"
    assert paths.schema_file == tmp_path / "mecha-brain/_meta/schema.md"


def test_authored_links_are_tracked_and_access_log_is_not() -> None:
    """links.jsonl is authored truth (_meta/); access.jsonl is derived (_meta/index/)."""
    paths = VaultPaths.for_root("vault")
    assert paths.links_file.parent == paths.meta_dir
    assert paths.access_file.parent == paths.index_dir


def test_for_root_does_not_touch_the_filesystem(tmp_path: Path) -> None:
    """`init` describes a vault before creating it."""
    paths = VaultPaths.for_root(tmp_path / "not-yet")
    assert not paths.mecha_brain.exists()
    assert paths.is_initialized is False


def test_episodic_is_partitioned_by_agent(tmp_vault: VaultPaths) -> None:
    assert tmp_vault.episodic_for("alpha") == tmp_vault.episodic_dir / "alpha"


def test_folder_for_accepts_enum_and_string(tmp_vault: VaultPaths) -> None:
    assert tmp_vault.folder_for(MemoryType.SEMANTIC) == tmp_vault.semantic_dir
    assert tmp_vault.folder_for("research") == tmp_vault.research_dir


def test_scope_index_shard(tmp_vault: VaultPaths) -> None:
    assert tmp_vault.scope_index("proj-a") == tmp_vault.indices_dir / "proj-a.md"


def test_memory_dirs_covers_every_type(tmp_vault: VaultPaths) -> None:
    assert set(tmp_vault.memory_dirs()) == set(MemoryType)


def test_contract_dirs_are_all_created_by_the_skeleton(tmp_vault: VaultPaths) -> None:
    for directory in tmp_vault.contract_dirs():
        assert directory.is_dir(), directory


# ── resolve / relative (R4.2) ───────────────────────────────────────
def test_resolve_joins_a_manifest_path_to_the_root(tmp_vault: VaultPaths) -> None:
    assert tmp_vault.resolve("mecha-brain/_inbox/") == tmp_vault.root / "mecha-brain/_inbox"


def test_resolve_refuses_an_absolute_path(tmp_vault: VaultPaths) -> None:
    with pytest.raises(ValueError, match="R4.2"):
        tmp_vault.resolve("/etc/passwd")


def test_relative_renders_vault_relative_posix(tmp_vault: VaultPaths) -> None:
    note = tmp_vault.semantic_dir / "2026-01-15_INS_x.md"
    assert tmp_vault.relative(note) == "mecha-brain/Semantic/2026-01-15_INS_x.md"


def test_relative_refuses_a_path_outside_the_vault(tmp_vault: VaultPaths, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the vault"):
        tmp_vault.relative(tmp_path.parent / "elsewhere.md")


def test_relative_and_resolve_round_trip(tmp_vault: VaultPaths) -> None:
    relative = "mecha-brain/Procedural/PROC_x.md"
    assert tmp_vault.relative(tmp_vault.resolve(relative)) == relative


# ── is_initialized / require_initialized ────────────────────────────
def test_initialized_vault_is_recognized(tmp_vault: VaultPaths) -> None:
    assert tmp_vault.is_initialized
    assert is_vault_root(tmp_vault.root)
    assert tmp_vault.require_initialized() is tmp_vault


def test_folder_without_manifest_is_not_a_vault(tmp_path: Path) -> None:
    (tmp_path / "mecha-brain" / "_meta").mkdir(parents=True)
    assert not is_vault_root(tmp_path)
    with pytest.raises(VaultNotFoundError) as excinfo:
        VaultPaths.for_root(tmp_path).require_initialized()
    assert excinfo.value.hint is not None and "init" in excinfo.value.hint


# ── R4.3 step 1: explicit argument ──────────────────────────────────
def test_explicit_argument_wins(tmp_vault: VaultPaths, make_vault: Callable[..., VaultPaths]) -> None:
    other = make_vault(name="other")
    found = discover_vault(other.root, env={VAULT_ENV_VAR: str(tmp_vault.root)})
    assert found.root == other.root


def test_explicit_argument_accepts_a_string(tmp_vault: VaultPaths) -> None:
    assert discover_vault(str(tmp_vault.root)).root == tmp_vault.root


def test_explicit_missing_path_raises_rather_than_falling_through(
    tmp_vault: VaultPaths, tmp_path: Path
) -> None:
    """A wrong --vault must never silently write to some other vault."""
    with pytest.raises(VaultNotFoundError) as excinfo:
        discover_vault(tmp_path / "ghost", env={VAULT_ENV_VAR: str(tmp_vault.root)})
    assert "--vault" in str(excinfo.value)
    assert "does not exist" in str(excinfo.value)


def test_explicit_uninitialized_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(VaultNotFoundError) as excinfo:
        discover_vault(tmp_path)
    assert "config.yaml" in str(excinfo.value)


def test_explicit_file_raises(tmp_vault: VaultPaths) -> None:
    with pytest.raises(VaultNotFoundError, match="not a directory"):
        discover_vault(tmp_vault.config_file)


def test_pointing_at_the_mecha_brain_folder_is_diagnosed(tmp_vault: VaultPaths) -> None:
    """The likeliest mistake: naming the folder instead of its parent."""
    with pytest.raises(VaultNotFoundError) as excinfo:
        discover_vault(tmp_vault.mecha_brain)
    assert excinfo.value.hint is not None
    assert str(tmp_vault.root) in excinfo.value.hint


def test_pointing_below_the_vault_root_names_the_root(tmp_vault: VaultPaths) -> None:
    deep = tmp_vault.semantic_dir
    with pytest.raises(VaultNotFoundError) as excinfo:
        discover_vault(deep)
    assert excinfo.value.hint is not None
    assert str(tmp_vault.root) in excinfo.value.hint


# ── R4.3 step 2: environment ────────────────────────────────────────
def test_env_var_is_used_when_no_argument(tmp_vault: VaultPaths) -> None:
    found = discover_vault(env={VAULT_ENV_VAR: str(tmp_vault.root)})
    assert found.root == tmp_vault.root


def test_env_var_outranks_the_upward_walk(
    tmp_vault: VaultPaths, make_vault: Callable[..., VaultPaths]
) -> None:
    other = make_vault(name="other")
    found = discover_vault(env={VAULT_ENV_VAR: str(other.root)}, start=tmp_vault.semantic_dir)
    assert found.root == other.root


def test_env_var_pointing_nowhere_raises(tmp_vault: VaultPaths, tmp_path: Path) -> None:
    with pytest.raises(VaultNotFoundError) as excinfo:
        discover_vault(env={VAULT_ENV_VAR: str(tmp_path / "ghost")}, start=tmp_vault.root)
    assert VAULT_ENV_VAR in str(excinfo.value)


def test_blank_env_var_is_treated_as_unset(tmp_vault: VaultPaths) -> None:
    found = discover_vault(env={VAULT_ENV_VAR: "   "}, start=tmp_vault.semantic_dir)
    assert found.root == tmp_vault.root


def test_env_var_is_read_from_the_process_by_default(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(VAULT_ENV_VAR, str(tmp_vault.root))
    assert discover_vault().root == tmp_vault.root


# ── R4.3 step 3: upward walk ────────────────────────────────────────
def test_walk_up_finds_the_vault_from_a_nested_folder(tmp_vault: VaultPaths) -> None:
    deep = tmp_vault.episodic_for("alpha")
    assert discover_vault(start=deep).root == tmp_vault.root


def test_walk_up_finds_the_vault_from_its_own_root(tmp_vault: VaultPaths) -> None:
    assert discover_vault(start=tmp_vault.root).root == tmp_vault.root


def test_walk_up_finds_the_vault_from_a_non_contract_subfolder(tmp_vault: VaultPaths) -> None:
    """The vault root is a whole PKM vault, not just mecha-brain/."""
    human = tmp_vault.root / "Notes" / "Deep"
    human.mkdir(parents=True)
    assert discover_vault(start=human).root == tmp_vault.root


def test_walk_up_stops_at_the_nearest_vault(
    tmp_vault: VaultPaths, make_vault: Callable[..., VaultPaths]
) -> None:
    inner_root = tmp_vault.root / "inner"
    inner = make_vault(name="vault/inner")
    assert inner.root == inner_root
    assert discover_vault(start=inner_root / "mecha-brain" / "Semantic").root == inner_root


def test_walk_up_uses_the_cwd_by_default(
    tmp_vault: VaultPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_vault.semantic_dir)
    assert discover_vault().root == tmp_vault.root


def test_no_vault_anywhere_teaches_init(tmp_path: Path) -> None:
    with pytest.raises(VaultNotFoundError) as excinfo:
        discover_vault(start=tmp_path)
    assert excinfo.value.rule == "R4.3"
    assert "no vault found" in str(excinfo.value)
    assert excinfo.value.hint is not None
    assert "init" in excinfo.value.hint
    assert VAULT_ENV_VAR in excinfo.value.hint


def test_find_vault_root_returns_none_instead_of_raising(tmp_path: Path) -> None:
    assert find_vault_root(tmp_path) is None


def test_find_vault_root_accepts_a_file(tmp_vault: VaultPaths) -> None:
    assert find_vault_root(tmp_vault.config_file) == tmp_vault.root
