"""Manifest parsing and §5 validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from mechabrain import __version__
from mechabrain.contract import MemoryType
from mechabrain.errors import KernelTooOldError, ManifestError
from mechabrain.manifest import Manifest, load_manifest


def parse(data: dict[str, Any]) -> Manifest:
    return Manifest.from_mapping(data)


def expect_error(data: dict[str, Any]) -> ManifestError:
    with pytest.raises(ManifestError) as excinfo:
        parse(data)
    return excinfo.value


# ── Defaults (§5) ───────────────────────────────────────────────────
def test_min_manifest_applies_spec_defaults(manifest_min: Manifest) -> None:
    assert manifest_min.agents == ()
    assert manifest_min.scopes.default == "global"
    assert manifest_min.scopes.known == ()
    assert manifest_min.naming.note_name == "{date}_{prefix}_{slug}.md"
    assert manifest_min.naming.dated_types == (
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.RESEARCH,
    )
    assert manifest_min.zones.proposals_dir == "mecha-brain/_inbox/"
    assert manifest_min.zones.research_enabled is True
    assert manifest_min.frontmatter.tag_namespaces.memory == "mem"
    assert manifest_min.retrieval.hybrid.vector_weight == 0.6
    assert manifest_min.retrieval.hybrid.bm25_weight == 0.4
    assert manifest_min.retrieval.link_expansion.default_hops == 1
    assert manifest_min.retrieval.link_expansion.max_hops == 2
    assert manifest_min.retrieval.contextual_retrieval is True
    assert manifest_min.retrieval.rerank is False
    assert manifest_min.maintenance.decay_days == 90
    assert manifest_min.maintenance.dedup_similarity == 0.92
    assert manifest_min.maintenance.commit_prefix == "chore(ai-memory):"
    assert manifest_min.maintenance.proc_stale_days == 180
    assert manifest_min.gate.reject_on == ()


def test_full_manifest_reads_every_key_rather_than_defaulting(manifest_full: Manifest) -> None:
    """Guards against a key being silently ignored: every value here is non-default."""
    assert manifest_full.agent_ids() == ("alpha", "beta")
    assert manifest_full.scopes.default == "proj-a"
    assert manifest_full.naming.note_name == "{date}-{prefix}-{slug}.md"
    assert manifest_full.naming.dated_types == (MemoryType.EPISODIC, MemoryType.SEMANTIC)
    assert manifest_full.naming.prefixes[MemoryType.PROCEDURAL] == "HOW"
    assert manifest_full.naming.proposal_name == "{date}_PROPOSAL_{slug}.md"
    assert manifest_full.zones.proposals_dir == "Inbox/"
    assert manifest_full.zones.read_only_index == ("Notes/", "Reference/")
    assert manifest_full.zones.research_enabled is False
    assert manifest_full.frontmatter.denylist_keys == ("publish", "internal-id")
    assert manifest_full.frontmatter.denylist_tags == ("automation/trigger",)
    assert manifest_full.frontmatter.required_extra_tags == ("source/ai",)
    assert manifest_full.frontmatter.tag_namespaces.agent == "author"
    assert manifest_full.retrieval.embedding.provider == "http"
    assert manifest_full.retrieval.embedding.model == "custom-model"
    assert manifest_full.retrieval.hybrid.vector_weight == 0.75
    assert manifest_full.retrieval.contextual_retrieval is False
    assert manifest_full.retrieval.rerank is True
    assert manifest_full.retrieval.link_expansion.max_hops == 3
    assert manifest_full.retrieval.store == "lancedb"
    assert manifest_full.maintenance.decay_days == 30
    assert manifest_full.maintenance.dedup_similarity == 0.85
    assert manifest_full.maintenance.proc_stale_days == 60
    assert manifest_full.gate.reject_on == ("confidence_unverified",)


# ── R5.1: strict keys ───────────────────────────────────────────────
def test_unknown_top_level_key_is_rejected(manifest_data_min: dict[str, Any]) -> None:
    manifest_data_min["retreival"] = {}
    error = expect_error(manifest_data_min)
    assert "unknown key" in error.message
    assert "retreival" in error.message
    assert error.rule == "R5.1"


def test_unknown_key_suggests_the_closest_spelling(manifest_data_ci: dict[str, Any]) -> None:
    """A typo must never fall through to a silent default (R5.1)."""
    manifest_data_ci["maintenance"]["decay_dayz"] = 10
    error = expect_error(manifest_data_ci)
    assert "maintenance.decay_dayz" in error.message
    assert error.hint is not None and "decay_days" in error.hint


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("scopes", "defualt"),
        ("naming", "note_nam"),
        ("zones", "proposals_directory"),
        ("frontmatter", "denylist_key"),
        ("retrieval", "reranker"),
        ("maintenance", "commit_prefixes"),
    ],
)
def test_unknown_key_is_rejected_at_every_nesting_level(
    manifest_data_ci: dict[str, Any], section: str, key: str
) -> None:
    manifest_data_ci[section][key] = "x"
    error = expect_error(manifest_data_ci)
    assert f"{section}.{key}" in error.message


def test_unknown_deep_key_reports_its_dotted_path(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["link_expansion"]["defualt_hops"] = 1
    error = expect_error(manifest_data_ci)
    assert "retrieval.link_expansion.defualt_hops" in error.message
    assert error.hint is not None and "default_hops" in error.hint


def test_unknown_agent_key_reports_the_list_index(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["agents"][1]["displayname"] = "Beta"
    error = expect_error(manifest_data_ci)
    assert "agents[1].displayname" in error.message


def test_unknown_prefix_type_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["prefixes"]["procedual"] = "PRC"
    error = expect_error(manifest_data_ci)
    assert "naming.prefixes.procedual" in error.message
    assert error.hint is not None and "procedural" in error.hint


def test_missing_handshake_section_is_rejected() -> None:
    error = expect_error({"scopes": {"default": "global"}})
    assert "mecha_brain" in error.message
    assert "missing" in error.message


def test_empty_manifest_is_rejected() -> None:
    error = expect_error({})
    assert "empty" in error.message
    assert error.hint is not None and "init" in error.hint


# ── R4.5: version handshake ─────────────────────────────────────────
def test_kernel_older_than_required_is_refused(manifest_data_min: dict[str, Any]) -> None:
    manifest_data_min["mecha_brain"]["kernel_min_version"] = "99.0.0"
    with pytest.raises(KernelTooOldError) as excinfo:
        parse(manifest_data_min)
    assert excinfo.value.rule == "R4.5"
    assert "99.0.0" in str(excinfo.value)
    assert __version__ in str(excinfo.value)


def test_kernel_equal_to_required_is_accepted(manifest_data_min: dict[str, Any]) -> None:
    manifest_data_min["mecha_brain"]["kernel_min_version"] = __version__
    assert parse(manifest_data_min).mecha_brain.kernel_min_version == __version__


def test_kernel_newer_than_required_is_accepted(manifest_data_min: dict[str, Any]) -> None:
    manifest_data_min["mecha_brain"]["kernel_min_version"] = "0.0.1"
    assert parse(manifest_data_min).mecha_brain.kernel_min_version == "0.0.1"


def test_version_comparison_is_numeric_not_lexicographic(
    manifest_data_min: dict[str, Any],
) -> None:
    """'0.10.0' > '0.1.0' numerically, though it sorts lower as a string."""
    manifest_data_min["mecha_brain"]["kernel_min_version"] = "0.10.0"
    with pytest.raises(KernelTooOldError):
        parse(manifest_data_min)


def test_foreign_spec_version_is_refused(manifest_data_min: dict[str, Any]) -> None:
    manifest_data_min["mecha_brain"]["spec_version"] = "0.2"
    error = expect_error(manifest_data_min)
    assert "spec_version" in error.message


def test_malformed_version_is_refused(manifest_data_min: dict[str, Any]) -> None:
    manifest_data_min["mecha_brain"]["kernel_min_version"] = "v0.1-beta"
    error = expect_error(manifest_data_min)
    assert "kernel_min_version" in error.message


# ── R4.2: no absolute paths ─────────────────────────────────────────
@pytest.mark.parametrize(
    "absolute",
    ["/srv/vault/inbox/", "C:\\vault\\inbox\\", "\\\\server\\share\\inbox", "~/vault/inbox/"],
)
def test_absolute_proposals_dir_is_rejected(
    manifest_data_ci: dict[str, Any], absolute: str
) -> None:
    manifest_data_ci["zones"]["proposals_dir"] = absolute
    error = expect_error(manifest_data_ci)
    assert "zones.proposals_dir" in error.message
    assert "relative" in error.message


def test_absolute_read_only_index_entry_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["zones"]["read_only_index"] = ["Notes/", "/etc/notes/"]
    error = expect_error(manifest_data_ci)
    assert "zones.read_only_index[1]" in error.message


def test_relative_paths_are_accepted(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["zones"]["proposals_dir"] = "Inbox/Proposals/"
    assert parse(manifest_data_ci).zones.proposals_dir == "Inbox/Proposals/"


# ── Types and ranges ────────────────────────────────────────────────
def test_weights_must_sum_to_one(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["hybrid"] = {"vector_weight": 0.6, "bm25_weight": 0.6}
    error = expect_error(manifest_data_ci)
    assert "sum to 1.0" in error.message


def test_weights_that_sum_to_one_with_float_error_are_accepted(
    manifest_data_ci: dict[str, Any],
) -> None:
    """0.7 + 0.3 is 0.9999999999999999 in binary floating point."""
    manifest_data_ci["retrieval"]["hybrid"] = {"vector_weight": 0.7, "bm25_weight": 0.3}
    assert parse(manifest_data_ci).retrieval.hybrid.vector_weight == 0.7


def test_weight_above_one_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["hybrid"] = {"vector_weight": 1.5, "bm25_weight": -0.5}
    error = expect_error(manifest_data_ci)
    assert "retrieval.hybrid.vector_weight" in error.message


def test_default_hops_above_max_hops_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["link_expansion"] = {"default_hops": 3, "max_hops": 2}
    error = expect_error(manifest_data_ci)
    assert "default_hops" in error.message and "max_hops" in error.message


def test_negative_hops_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["link_expansion"]["default_hops"] = -1
    error = expect_error(manifest_data_ci)
    assert "retrieval.link_expansion.default_hops" in error.message


def test_zero_hops_is_accepted(manifest_data_ci: dict[str, Any]) -> None:
    """default_hops: 0 is how a deployment turns link expansion off (§5)."""
    manifest_data_ci["retrieval"]["link_expansion"]["default_hops"] = 0
    assert parse(manifest_data_ci).retrieval.link_expansion.default_hops == 0


@pytest.mark.parametrize("value", [0, 0.0, 1.5, -0.2])
def test_dedup_similarity_outside_range_is_rejected(
    manifest_data_ci: dict[str, Any], value: float
) -> None:
    manifest_data_ci["maintenance"]["dedup_similarity"] = value
    error = expect_error(manifest_data_ci)
    assert "dedup_similarity" in error.message


def test_dedup_similarity_of_one_is_accepted(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["maintenance"]["dedup_similarity"] = 1
    assert parse(manifest_data_ci).maintenance.dedup_similarity == 1.0


def test_decay_days_must_be_positive(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["maintenance"]["decay_days"] = 0
    error = expect_error(manifest_data_ci)
    assert "decay_days" in error.message


# ── gate: (§8.2 opt-in strictness) ─────────────────────────────────
def test_reject_on_accepts_the_elevatable_check(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["gate"] = {"reject_on": ["confidence_unverified"]}
    manifest = parse(manifest_data_ci)
    assert manifest.gate.reject_on == ("confidence_unverified",)


def test_reject_on_refuses_a_judgement_check(manifest_data_ci: dict[str, Any]) -> None:
    """`reusable`/`atomic` must never be elevated: the hint explains why (§8.2)."""
    manifest_data_ci["gate"] = {"reject_on": ["atomic"]}
    error = expect_error(manifest_data_ci)
    assert "gate.reject_on[0]" in error.message
    assert "atomic" in error.message
    assert error.hint is not None and "confidence_unverified" in error.hint


def test_reject_on_suggests_the_intended_spelling(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["gate"] = {"reject_on": ["confidence_unverifed"]}
    error = expect_error(manifest_data_ci)
    assert error.hint is not None and "confidence_unverified" in error.hint


def test_unknown_gate_key_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["gate"] = {"strict_mode": True}
    error = expect_error(manifest_data_ci)
    assert "gate.strict_mode" in error.message


def test_proc_stale_days_zero_disables(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["maintenance"]["proc_stale_days"] = 0
    manifest = parse(manifest_data_ci)
    assert manifest.maintenance.proc_stale_days == 0


def test_proc_stale_days_rejects_a_negative(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["maintenance"]["proc_stale_days"] = -1
    error = expect_error(manifest_data_ci)
    assert "proc_stale_days" in error.message


def test_boolean_is_not_accepted_as_integer(manifest_data_ci: dict[str, Any]) -> None:
    """`bool` subclasses `int`; `decay_days: true` is a mistake, not 1."""
    manifest_data_ci["maintenance"]["decay_days"] = True
    error = expect_error(manifest_data_ci)
    assert "expected an integer" in error.message


def test_string_where_boolean_expected_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["rerank"] = "yes"
    error = expect_error(manifest_data_ci)
    assert "retrieval.rerank" in error.message
    assert "boolean" in error.message


def test_scalar_where_list_expected_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["frontmatter"]["denylist_tags"] = "forbidden/tag"
    error = expect_error(manifest_data_ci)
    assert "frontmatter.denylist_tags" in error.message


def test_agents_must_be_a_list(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["agents"] = {"alpha": {}}
    error = expect_error(manifest_data_ci)
    assert "agents" in error.message


def test_unknown_embedding_provider_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["embedding"]["provider"] = "sentence-transformer"
    error = expect_error(manifest_data_ci)
    assert "provider" in error.message
    assert error.hint is not None and "sentence-transformers" in error.hint


def test_unknown_store_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["retrieval"]["store"] = "chroma"
    error = expect_error(manifest_data_ci)
    assert "retrieval.store" in error.message


# ── Agents (R6.2, R6.6) ─────────────────────────────────────────────
@pytest.mark.parametrize("bad_id", ["Alpha", "my agent", "-alpha", "agent.one", "ágent"])
def test_agent_id_must_be_a_lowercase_slug(manifest_data_ci: dict[str, Any], bad_id: str) -> None:
    manifest_data_ci["agents"][0]["id"] = bad_id
    error = expect_error(manifest_data_ci)
    assert "agents[0].id" in error.message
    assert error.hint is not None


def test_duplicate_agent_ids_are_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["agents"][1]["id"] = "alpha"
    error = expect_error(manifest_data_ci)
    assert "duplicate agent id" in error.message


def test_duplicate_profiles_are_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["agents"][0]["profiles"] = ["tutor", "tutor"]
    error = expect_error(manifest_data_ci)
    assert "duplicate profile" in error.message


def test_agent_without_id_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    del manifest_data_ci["agents"][0]["id"]
    error = expect_error(manifest_data_ci)
    assert "agents[0].id" in error.message


def test_display_name_defaults_to_the_id(manifest_data_ci: dict[str, Any]) -> None:
    del manifest_data_ci["agents"][1]["display_name"]
    assert parse(manifest_data_ci).agent("beta").display_name == "beta"


def test_private_store_none_literal_means_absent(manifest_ci: Manifest) -> None:
    """§5 writes `private_store: none` for "there isn't one"."""
    assert manifest_ci.agent("alpha").private_store is None


def test_per_profile_private_store_is_accepted(manifest_full: Manifest) -> None:
    assert manifest_full.agent("beta").private_store == {"researcher": "per-profile store"}


def test_per_profile_private_store_rejects_undeclared_profile(
    manifest_data_ci: dict[str, Any],
) -> None:
    manifest_data_ci["agents"][0]["private_store"] = {"tutor": "ok", "ghost": "nope"}
    error = expect_error(manifest_data_ci)
    assert "ghost" in error.message


# ── Scopes (R6.5) ───────────────────────────────────────────────────
def test_default_scope_must_be_in_known(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["scopes"] = {"known": ["proj-a"], "default": "global"}
    error = expect_error(manifest_data_ci)
    assert "scopes.default" in error.message


def test_duplicate_known_scopes_are_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["scopes"]["known"] = ["proj-a", "proj-a"]
    error = expect_error(manifest_data_ci)
    assert "duplicate scope" in error.message


def test_is_known_scope_honours_the_allowlist(manifest_ci: Manifest) -> None:
    assert manifest_ci.is_known_scope("proj-a")
    assert manifest_ci.is_known_scope("global")
    assert not manifest_ci.is_known_scope("proj-z")
    assert not manifest_ci.is_known_scope("")


def test_empty_known_accepts_any_slug(manifest_min: Manifest) -> None:
    """§5: "vazio = qualquer slug aceito" -- but still a slug."""
    assert manifest_min.is_known_scope("anything-goes")
    assert not manifest_min.is_known_scope("Not A Slug")
    assert not manifest_min.is_known_scope("")


# ── Naming ──────────────────────────────────────────────────────────
def test_unknown_placeholder_in_note_name_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["note_name"] = "{date}_{prefix}_{titel}.md"
    error = expect_error(manifest_data_ci)
    assert "naming.note_name" in error.message
    assert "titel" in error.message


def test_note_name_without_slug_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["note_name"] = "{date}_{prefix}.md"
    error = expect_error(manifest_data_ci)
    assert "slug" in error.message


def test_note_name_must_end_with_md(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["note_name"] = "{date}_{prefix}_{slug}.txt"
    error = expect_error(manifest_data_ci)
    assert ".md" in error.message


def test_dated_types_without_date_placeholder_is_rejected(
    manifest_data_ci: dict[str, Any],
) -> None:
    manifest_data_ci["naming"]["note_name"] = "{prefix}_{slug}.md"
    error = expect_error(manifest_data_ci)
    assert "dated_types" in error.message


def test_unknown_dated_type_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["dated_types"] = ["episodic", "semantik"]
    error = expect_error(manifest_data_ci)
    assert "naming.dated_types[1]" in error.message
    assert error.hint is not None and "semantic" in error.hint


def test_prefix_with_path_separator_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["prefixes"]["semantic"] = "INS/X"
    error = expect_error(manifest_data_ci)
    assert "naming.prefixes.semantic" in error.message


def test_partial_prefixes_keep_spec_defaults(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["naming"]["prefixes"] = {"semantic": "FACT"}
    manifest = parse(manifest_data_ci)
    assert manifest.prefix_for(MemoryType.SEMANTIC) == "FACT"
    assert manifest.prefix_for(MemoryType.PROCEDURAL) == "PROC"


# ── Frontmatter ─────────────────────────────────────────────────────
def test_tag_namespaces_must_differ(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["frontmatter"]["tag_namespaces"] = {"memory": "x", "agent": "x"}
    error = expect_error(manifest_data_ci)
    assert "must differ" in error.message


def test_tag_namespace_with_slash_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["frontmatter"]["tag_namespaces"]["memory"] = "mem/x"
    error = expect_error(manifest_data_ci)
    assert "frontmatter.tag_namespaces.memory" in error.message


def test_tag_with_leading_hash_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["frontmatter"]["required_extra_tags"] = ["#source/ai"]
    error = expect_error(manifest_data_ci)
    assert "invalid tag" in error.message


def test_tag_both_required_and_denied_is_rejected(manifest_data_ci: dict[str, Any]) -> None:
    manifest_data_ci["frontmatter"]["required_extra_tags"] = ["forbidden/tag"]
    error = expect_error(manifest_data_ci)
    assert "both required and denied" in error.message


def test_tag_namespaces_generate_contract_tags(manifest_full: Manifest) -> None:
    namespaces = manifest_full.frontmatter.tag_namespaces
    assert namespaces.memory_tag(MemoryType.SEMANTIC) == "memory/semantic"
    assert namespaces.agent_tag("alpha") == "author/alpha"


# ── Query methods ───────────────────────────────────────────────────
def test_agent_ids_preserves_manifest_order(manifest_ci: Manifest) -> None:
    assert manifest_ci.agent_ids() == ("alpha", "beta")


def test_profiles_of_returns_declared_personas(manifest_ci: Manifest) -> None:
    assert manifest_ci.profiles_of("alpha") == ("tutor", "planner")
    assert manifest_ci.profiles_of("beta") == ()


def test_unknown_agent_raises_with_a_suggestion(manifest_ci: Manifest) -> None:
    with pytest.raises(ManifestError) as excinfo:
        manifest_ci.agent("alfa")
    assert excinfo.value.rule == "R6.2"
    assert excinfo.value.hint is not None and "alpha" in excinfo.value.hint


def test_is_known_agent_does_not_raise(manifest_ci: Manifest) -> None:
    assert manifest_ci.is_known_agent("alpha")
    assert not manifest_ci.is_known_agent("ghost")


def test_folder_and_prefix_lookups(manifest_ci: Manifest) -> None:
    assert manifest_ci.folder_for("semantic") == "Semantic"
    assert manifest_ci.folder_for(MemoryType.EPISODIC) == "Episodic"
    assert manifest_ci.prefix_for("procedural") == "PROC"
    assert manifest_ci.prefix_for(MemoryType.RESEARCH) == "RES"


def test_is_dated_reflects_the_manifest(manifest_ci: Manifest) -> None:
    assert manifest_ci.is_dated(MemoryType.SEMANTIC)
    assert not manifest_ci.is_dated(MemoryType.PROCEDURAL)


def test_is_enabled_only_gates_research(manifest_full: Manifest, manifest_ci: Manifest) -> None:
    assert manifest_ci.is_enabled(MemoryType.RESEARCH)
    assert not manifest_full.is_enabled(MemoryType.RESEARCH)
    assert manifest_full.is_enabled(MemoryType.SEMANTIC)


def test_unknown_memory_type_lookup_raises(manifest_ci: Manifest) -> None:
    with pytest.raises(ValueError, match="unknown memory type"):
        manifest_ci.folder_for("procedureal")


# ── Loading from disk ───────────────────────────────────────────────
def test_load_manifest_reads_the_vault_config(tmp_vault: Any) -> None:
    manifest = load_manifest(tmp_vault.config_file)
    assert manifest.retrieval.embedding.provider == "hash"
    assert manifest.retrieval.store == "numpy"
    assert manifest.source_path == tmp_vault.config_file


def test_load_manifest_on_missing_file_teaches_init(tmp_path: Path) -> None:
    with pytest.raises(ManifestError) as excinfo:
        load_manifest(tmp_path / "config.yaml")
    assert excinfo.value.hint is not None and "init" in excinfo.value.hint


def test_invalid_yaml_is_reported_as_a_manifest_error(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("mecha_brain: [unclosed\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid YAML"):
        load_manifest(config)


def test_yaml_round_trip_of_the_full_manifest(
    manifest_data_full: dict[str, Any], make_vault: Any
) -> None:
    paths = make_vault(manifest_data=copy.deepcopy(manifest_data_full), name="full")
    assert load_manifest(paths.config_file) == Manifest.from_mapping(
        manifest_data_full, source_path=paths.config_file
    )
