"""Tests for :mod:`mechabrain.generate`.

Weighted deliberately towards the managed block of ``AGENTS.md``: it is the one
place in the kernel that writes into a file a human also writes into, so the
cost of a bug there is somebody's lost prose, not a stack trace.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mechabrain import __version__
from mechabrain.contract import (
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    SPEC_VERSION,
)
from mechabrain.discovery import VaultPaths
from mechabrain.errors import MechabrainError
from mechabrain.generate import (
    HOT_SECTION_MAX_ENTRIES,
    SHARD_LINE_THRESHOLD,
    merge_managed_block,
    render_agents_md,
    render_default_config,
    render_hot,
    render_index,
    render_initial_hot,
    render_initial_index,
    render_managed_block,
    render_schema,
    write_agents_md,
    write_default_config,
    write_hot,
    write_index,
    write_schema,
)
from mechabrain.manifest import Manifest
from mechabrain.note import Note

BLOCK = f"{MANAGED_BLOCK_BEGIN}\nmanaged\n{MANAGED_BLOCK_END}"

#: Any absolute path in generated output would break the moment the vault moves
#: between machines (R4.2, §12.1).
ABSOLUTE_PATH_RE = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\")


def make_note(
    paths: VaultPaths,
    name: str,
    *,
    memory_type: str = "semantic",
    scope: str = "proj-a",
    agent: str = "alpha",
    status: str = "ativo",
    title: str | None = None,
    last_accessed: date | None = None,
    tags: list[str] | None = None,
) -> Note:
    """An in-memory note shaped like §6. Not written to disk: renderers are pure."""
    folder = {
        "semantic": paths.semantic_dir,
        "procedural": paths.procedural_dir,
        "research": paths.research_dir,
    }.get(memory_type, paths.episodic_for(agent))
    frontmatter: dict[str, Any] = {
        "title": title if title is not None else name.replace("-", " ").capitalize(),
        "tags": [f"mem/{memory_type}", f"agent/{agent}"] if tags is None else tags,
        "created": date(2026, 1, 10),
        "modified": date(2026, 1, 10),
        "agent": agent,
        "scope": scope,
        "source": "test-session",
        "confidence": "medium",
        "status": status,
    }
    if last_accessed is not None:
        frontmatter["last_accessed"] = last_accessed
    return Note(path=folder / f"{name}.md", frontmatter=frontmatter)


# ══════════════════════════════════════════════════════════════════════
# config.yaml (§5)
# ══════════════════════════════════════════════════════════════════════
class TestDefaultConfig:
    def test_parses_back_through_the_validator(self) -> None:
        # The one bug this module must not have: generating a config the
        # kernel's own strict parser rejects (R5.1).
        manifest = Manifest.from_yaml(render_default_config())
        assert manifest.mecha_brain.spec_version == SPEC_VERSION
        assert manifest.mecha_brain.kernel_min_version == __version__

    def test_ships_the_documented_defaults(self) -> None:
        manifest = Manifest.from_yaml(render_default_config())
        assert manifest.retrieval.embedding.provider == "sentence-transformers"
        assert manifest.retrieval.embedding.model == "BAAI/bge-m3"
        assert manifest.retrieval.store == "numpy"
        assert manifest.scopes.default == "global"
        assert manifest.maintenance.decay_days == 90
        assert manifest.retrieval.hybrid.vector_weight == pytest.approx(0.6)

    def test_registry_has_exactly_one_example_agent(self) -> None:
        manifest = Manifest.from_yaml(render_default_config())
        assert len(manifest.agents) == 1
        assert manifest.agents[0].profiles == ()
        assert manifest.agents[0].private_store is None

    @pytest.mark.parametrize(
        "section",
        ["mecha_brain", "agents", "scopes", "naming", "zones", "frontmatter",
         "retrieval", "maintenance"],
    )
    def test_every_section_of_the_spec_is_present(self, section: str) -> None:
        assert re.search(rf"^{section}:", render_default_config(), re.MULTILINE)

    @pytest.mark.parametrize(
        "key",
        ["spec_version", "kernel_min_version", "private_store", "known", "note_name",
         "dated_types", "prefixes", "proposal_name", "proposals_dir",
         "read_only_index", "research_enabled", "denylist_keys", "denylist_tags",
         "tag_namespaces", "required_extra_tags", "provider", "model",
         "vector_weight", "bm25_weight", "contextual_retrieval", "rerank",
         "link_expansion", "store", "decay_days", "dedup_similarity",
         "commit_prefix"],
    )
    def test_every_key_of_the_spec_is_present(self, key: str) -> None:
        # A key nobody can see is a key nobody sets: §5 lists them all, so the
        # generated manifest does too, commented.
        assert re.search(rf"^\s+{key}:", render_default_config(), re.MULTILINE)

    def test_is_commented(self) -> None:
        config = render_default_config()
        assert "# ── Agentes" in config
        assert "R4.2" in config  # relative paths only
        assert "R6.5" in config  # scope / cross-contamination
        assert "contaminação cruzada" in config.casefold()

    def test_carries_no_absolute_path(self) -> None:
        assert not ABSOLUTE_PATH_RE.search(render_default_config())

    def test_is_deterministic(self) -> None:
        assert render_default_config() == render_default_config()


# ══════════════════════════════════════════════════════════════════════
# _meta/schema.md (R6.4)
# ══════════════════════════════════════════════════════════════════════
class TestSchema:
    def test_renders_the_real_denylists(self, manifest_ci: Manifest) -> None:
        schema = render_schema(manifest_ci)
        assert "`forbidden-key`" in schema
        assert "`forbidden/tag`" in schema

    def test_renders_the_real_registry(self, manifest_ci: Manifest) -> None:
        schema = render_schema(manifest_ci)
        assert "`alpha`" in schema and "`beta`" in schema
        assert "`tutor`" in schema and "`planner`" in schema
        assert "`Episodic/alpha/`" in schema

    def test_renders_the_real_scopes(self, manifest_ci: Manifest) -> None:
        schema = render_schema(manifest_ci)
        assert "`proj-a`" in schema and "`proj-b`" in schema
        assert "R6.5" in schema

    def test_renders_the_real_tag_namespaces(self, manifest_full: Manifest) -> None:
        # manifest_full renames both namespaces; a hard-coded `mem/` would show.
        schema = render_schema(manifest_full)
        assert "`memory/semantic`" in schema
        assert "`author/alpha`" in schema
        assert "`mem/semantic`" not in schema

    def test_renders_the_real_naming(self, manifest_full: Manifest) -> None:
        schema = render_schema(manifest_full)
        assert "`{date}-{prefix}-{slug}.md`" in schema
        assert "`SEM`" in schema and "`HOW`" in schema

    def test_marks_a_disabled_type(self, manifest_full: Manifest) -> None:
        # manifest_full has research_enabled: false.
        assert "_(desabilitado)_" in render_schema(manifest_full)

    def test_says_when_a_list_is_empty_rather_than_leaving_a_hole(
        self, manifest_min: Manifest
    ) -> None:
        schema = render_schema(manifest_min)
        assert "_(nenhuma)_" in schema
        assert "_(registry vazio)_" in schema

    def test_carries_no_absolute_path(self, manifest_full: Manifest) -> None:
        assert not ABSOLUTE_PATH_RE.search(render_schema(manifest_full))

    def test_is_deterministic(self, manifest_ci: Manifest) -> None:
        assert render_schema(manifest_ci) == render_schema(manifest_ci)


# ══════════════════════════════════════════════════════════════════════
# AGENTS.md: content of the managed block (§10)
# ══════════════════════════════════════════════════════════════════════
class TestManagedBlockContent:
    def test_is_delimited_by_the_contract_markers(self, manifest_ci: Manifest) -> None:
        block = render_managed_block(manifest_ci)
        assert block.startswith(MANAGED_BLOCK_BEGIN)
        assert block.endswith(MANAGED_BLOCK_END)

    def test_emits_exactly_one_pair_of_markers(self, manifest_ci: Manifest) -> None:
        # The block explains its own markers in prose. Quoting them literally
        # would put two pairs in the file, and merge_managed_block would refuse
        # every sync from then on -- so the prose names them without the
        # comment syntax.
        block = render_managed_block(manifest_ci)
        assert block.count(MANAGED_BLOCK_BEGIN) == 1
        assert block.count(MANAGED_BLOCK_END) == 1

    def test_fresh_file_is_the_block_plus_a_final_newline(
        self, manifest_ci: Manifest
    ) -> None:
        assert render_agents_md(manifest_ci) == render_managed_block(manifest_ci) + "\n"

    def test_states_the_boundaries(self, manifest_ci: Manifest) -> None:
        block = render_managed_block(manifest_ci)
        assert "P4" in block
        assert "mecha-brain/_inbox/" in block  # zones.proposals_dir
        assert "memory_propose" in block
        assert "R8.1" in block  # hot/index/indices are the consolidator's

    def test_carries_the_routing_tree(self, manifest_ci: Manifest) -> None:
        block = render_managed_block(manifest_ci)
        assert "§8.1" in block
        for step in range(1, 9):
            assert re.search(rf"^{step}\. ", block, re.MULTILINE)
        assert "prefira o projeto" in block

    def test_carries_the_gate_checklist(self, manifest_ci: Manifest) -> None:
        block = render_managed_block(manifest_ci)
        assert "§8.2" in block
        for item in ("Reutilizável?", "Já existe?", "Atômico?", "Fonte declarada?",
                     "Escopado?", "Procedural: testado?", "Limpo?"):
            assert item in block

    def test_is_honest_about_what_the_kernel_cannot_enforce(
        self, manifest_ci: Manifest
    ) -> None:
        # Items 1 and 3 are judgement: the kernel has no LLM and must not
        # pretend to police them.
        block = render_managed_block(manifest_ci)
        assert block.count("**você** (warning)") == 2
        assert "não vai fingir que policia o que não consegue medir" in block

    def test_recommends_filtering_by_scope(self, manifest_ci: Manifest) -> None:
        block = render_managed_block(manifest_ci)
        assert "contexto, não verdade local" in block
        assert "`proj-a`" in block and "`proj-b`" in block

    def test_carries_the_registry_with_profiles(self, manifest_ci: Manifest) -> None:
        block = render_managed_block(manifest_ci)
        assert "`alpha`" in block and "`beta`" in block
        assert "`tutor`" in block and "`planner`" in block
        assert "R6.2" in block

    def test_carries_the_denylists(self, manifest_full: Manifest) -> None:
        block = render_managed_block(manifest_full)
        assert "`publish`" in block and "`internal-id`" in block
        assert "`automation/trigger`" in block
        assert "`source/ai`" in block  # required_extra_tags

    def test_maps_the_dotted_tool_names_to_the_mcp_ones(
        self, manifest_ci: Manifest
    ) -> None:
        block = render_managed_block(manifest_ci)
        for dotted, underscored in (
            ("memory.search", "memory_search"),
            ("memory.get", "memory_get"),
            ("memory.status", "memory_status"),
            ("memory.write", "memory_write"),
            ("memory.propose", "memory_propose"),
            ("memory.link", "memory_link"),
        ):
            assert dotted in block and underscored in block

    def test_reflects_manifest_values_rather_than_spec_defaults(
        self, manifest_full: Manifest
    ) -> None:
        block = render_managed_block(manifest_full)
        assert "`0.85`" in block  # dedup_similarity
        assert "`30`" in block  # decay_days
        assert "`chore(memory):`" in block
        assert "Inbox/" in block  # proposals_dir
        assert "`Notes/`" in block  # read_only_index

    def test_marks_research_as_disabled_when_the_manifest_says_so(
        self, manifest_full: Manifest, manifest_ci: Manifest
    ) -> None:
        assert "**Desabilitado neste deployment**" in render_managed_block(manifest_full)
        assert "**Desabilitado neste deployment**" not in render_managed_block(manifest_ci)

    def test_warns_loudly_about_an_empty_registry(self, manifest_min: Manifest) -> None:
        # With no agent registered every write is rejected (R6.2); saying so is
        # more useful than an empty table.
        assert "Registry vazio" in render_managed_block(manifest_min)

    def test_carries_no_absolute_path(self, manifest_full: Manifest) -> None:
        assert not ABSOLUTE_PATH_RE.search(render_managed_block(manifest_full))


# ══════════════════════════════════════════════════════════════════════
# AGENTS.md: splicing the managed block (§10)
# ══════════════════════════════════════════════════════════════════════
class TestMergeManagedBlock:
    def test_replaces_the_block_in_place(self) -> None:
        existing = f"{MANAGED_BLOCK_BEGIN}\nold content\n{MANAGED_BLOCK_END}"
        assert merge_managed_block(existing, BLOCK) == BLOCK
        assert "old content" not in merge_managed_block(existing, BLOCK)

    def test_text_before_the_block_survives(self) -> None:
        existing = f"# Meu título\n\nminha prosa\n\n{MANAGED_BLOCK_BEGIN}\nold\n{MANAGED_BLOCK_END}"
        merged = merge_managed_block(existing, BLOCK)
        assert merged == f"# Meu título\n\nminha prosa\n\n{BLOCK}"

    def test_text_after_the_block_survives(self) -> None:
        existing = f"{MANAGED_BLOCK_BEGIN}\nold\n{MANAGED_BLOCK_END}\n\n## Minha seção\n\nnotas\n"
        merged = merge_managed_block(existing, BLOCK)
        assert merged == f"{BLOCK}\n\n## Minha seção\n\nnotas\n"

    def test_text_on_both_sides_survives(self) -> None:
        existing = f"antes\n{MANAGED_BLOCK_BEGIN}\nold\n{MANAGED_BLOCK_END}\ndepois"
        assert merge_managed_block(existing, BLOCK) == f"antes\n{BLOCK}\ndepois"

    def test_is_idempotent(self) -> None:
        once = merge_managed_block(f"antes\n{BLOCK}\ndepois", BLOCK)
        assert merge_managed_block(once, BLOCK) == once

    def test_a_file_without_markers_keeps_everything_and_gains_the_block(self) -> None:
        merged = merge_managed_block("# Notas do humano\n\nmuito trabalho aqui\n", BLOCK)
        assert merged == f"# Notas do humano\n\nmuito trabalho aqui\n\n{BLOCK}\n"

    def test_an_empty_file_gets_only_the_block(self) -> None:
        assert merge_managed_block("", BLOCK) == f"{BLOCK}\n"
        assert merge_managed_block("\n\n", BLOCK) == f"{BLOCK}\n"

    def test_crlf_input_is_normalised_without_losing_text(self) -> None:
        existing = f"antes\r\n{MANAGED_BLOCK_BEGIN}\r\nold\r\n{MANAGED_BLOCK_END}\r\ndepois\r\n"
        assert merge_managed_block(existing, BLOCK) == f"antes\n{BLOCK}\ndepois\n"

    def test_preserves_a_block_that_is_not_at_the_top(self) -> None:
        existing = "\n".join(f"linha {i}" for i in range(50))
        existing += f"\n{MANAGED_BLOCK_BEGIN}\nold\n{MANAGED_BLOCK_END}\nfim\n"
        merged = merge_managed_block(existing, BLOCK)
        assert merged.startswith("linha 0\n")
        assert merged.endswith("\nfim\n")
        assert BLOCK in merged

    # ── Ambiguity: refuse rather than eat someone's text ────────────
    def test_duplicate_begin_marker_is_refused(self) -> None:
        existing = f"{MANAGED_BLOCK_BEGIN}\na\n{MANAGED_BLOCK_BEGIN}\nb\n{MANAGED_BLOCK_END}"
        with pytest.raises(MechabrainError) as excinfo:
            merge_managed_block(existing, BLOCK)
        assert "§10" in str(excinfo.value)
        assert excinfo.value.hint

    def test_duplicate_end_marker_is_refused(self) -> None:
        existing = f"{MANAGED_BLOCK_BEGIN}\na\n{MANAGED_BLOCK_END}\nb\n{MANAGED_BLOCK_END}"
        with pytest.raises(MechabrainError):
            merge_managed_block(existing, BLOCK)

    def test_a_whole_duplicated_block_is_refused(self) -> None:
        with pytest.raises(MechabrainError):
            merge_managed_block(f"{BLOCK}\n{BLOCK}", BLOCK)

    def test_a_lone_begin_marker_is_refused(self) -> None:
        with pytest.raises(MechabrainError) as excinfo:
            merge_managed_block(f"{MANAGED_BLOCK_BEGIN}\ntexto sem fim", BLOCK)
        assert MANAGED_BLOCK_END in str(excinfo.value)

    def test_a_lone_end_marker_is_refused(self) -> None:
        with pytest.raises(MechabrainError) as excinfo:
            merge_managed_block(f"texto\n{MANAGED_BLOCK_END}", BLOCK)
        assert MANAGED_BLOCK_BEGIN in str(excinfo.value)

    def test_markers_in_the_wrong_order_are_refused(self) -> None:
        existing = f"{MANAGED_BLOCK_END}\nmiolo\n{MANAGED_BLOCK_BEGIN}"
        with pytest.raises(MechabrainError) as excinfo:
            merge_managed_block(existing, BLOCK)
        assert "wrong order" in str(excinfo.value)

    def test_refusal_never_writes_a_truncated_file(self) -> None:
        existing = f"prosa preciosa\n{MANAGED_BLOCK_BEGIN}\nsem fim"
        with pytest.raises(MechabrainError):
            merge_managed_block(existing, BLOCK)


class TestRenderAgentsMd:
    def test_regeneration_is_byte_identical(self, manifest_ci: Manifest) -> None:
        once = render_agents_md(manifest_ci)
        assert render_agents_md(manifest_ci, once) == once
        assert render_agents_md(manifest_ci, render_agents_md(manifest_ci, once)) == once

    def test_human_sections_survive_regeneration(self, manifest_ci: Manifest) -> None:
        first = render_agents_md(manifest_ci)
        edited = f"# Cabeçalho meu\n\n{first}\n## Apêndice meu\n\nregras locais\n"
        regenerated = render_agents_md(manifest_ci, edited)
        assert regenerated.startswith("# Cabeçalho meu\n\n")
        assert regenerated.endswith("## Apêndice meu\n\nregras locais\n")

    def test_a_manifest_change_updates_the_block_and_only_the_block(
        self, manifest_ci: Manifest, manifest_data_ci: dict[str, Any]
    ) -> None:
        existing = f"prosa do humano\n\n{render_agents_md(manifest_ci)}\nposfácio\n"
        manifest_data_ci["agents"].append({"id": "gamma", "display_name": "Gamma"})
        regenerated = render_agents_md(Manifest.from_mapping(manifest_data_ci), existing)
        assert "`gamma`" in regenerated
        assert regenerated.startswith("prosa do humano\n\n")
        assert regenerated.endswith("\nposfácio\n")

    def test_edits_inside_the_block_are_discarded(self, manifest_ci: Manifest) -> None:
        tampered = render_agents_md(manifest_ci).replace(
            "## 1. Fronteiras (P4)", "## 1. Fronteiras (eu editei isto)"
        )
        assert render_agents_md(manifest_ci, tampered) == render_agents_md(manifest_ci)


# ══════════════════════════════════════════════════════════════════════
# index.md (§9.5)
# ══════════════════════════════════════════════════════════════════════
class TestRenderIndex:
    def test_lists_one_line_per_active_semantic_and_procedural_memory(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        master = render_index(sample_notes, manifest_ci).master
        assert "[[2026-01-15_INS_vector-search]]" in master
        assert "[[2026-01-15_INS_global-fact]]" in master
        assert "[[PROC_deploy-playbook]]" in master

    def test_excludes_episodic_and_research(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # A journal entry is not a claim about the world, and a report is not a
        # one-liner: neither belongs in the MOC.
        master = render_index(sample_notes, manifest_ci).master
        assert "MEM_session-one" not in master
        assert "RES_link-expansion" not in master

    def test_excludes_archived_memories(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        assert "INS_stale-fact" not in render_index(sample_notes, manifest_ci).master

    def test_groups_by_scope_with_global_last(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        master = render_index(sample_notes, manifest_ci).master
        assert master.index("## proj-a") < master.index("## global")

    def test_every_line_points_at_a_real_note(
        self, tmp_vault: VaultPaths, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        master = render_index(sample_notes, manifest_ci).master
        ids = {note.note_id for note in sample_notes}
        linked = set(re.findall(r"\[\[([^\]]+)\]\]", master))
        assert linked
        assert linked <= ids

    def test_falls_back_to_the_folder_when_tags_use_another_namespace(
        self, sample_notes: list[Note], manifest_full: Manifest
    ) -> None:
        # manifest_full's memory namespace is `memory`, so the notes' `mem/...`
        # tags say nothing: the type must come from the folder.
        master = render_index(sample_notes, manifest_full).master
        assert "[[2026-01-15_INS_vector-search]]" in master
        assert "[[PROC_deploy-playbook]]" in master

    def test_an_empty_vault_says_so(self, manifest_ci: Manifest) -> None:
        rendered = render_index((), manifest_ci)
        assert "_Nenhuma memória ativa ainda._" in rendered.master
        assert rendered.shards == {}
        assert not rendered.sharded

    def test_initial_index_is_the_empty_render(self, manifest_ci: Manifest) -> None:
        assert render_initial_index(manifest_ci) == render_index((), manifest_ci).master

    def test_is_deterministic(self, sample_notes: list[Note], manifest_ci: Manifest) -> None:
        # No timestamp: an unchanged vault must re-render byte-identically, or
        # every consolidate run commits a no-op diff.
        assert (
            render_index(sample_notes, manifest_ci).master
            == render_index(sample_notes, manifest_ci).master
        )

    def test_stays_flat_below_the_threshold(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        notes = [make_note(tmp_vault, f"INS_a{i}") for i in range(5)]
        rendered = render_index(notes, manifest_ci)
        assert not rendered.sharded
        assert rendered.shards == {}
        assert len(rendered.master.splitlines()) <= SHARD_LINE_THRESHOLD


class TestIndexSharding:
    @pytest.fixture
    def many_notes(self, tmp_vault: VaultPaths) -> list[Note]:
        """Enough memories in three scopes to blow the line budget."""
        return [
            make_note(tmp_vault, f"INS_{scope}-{i}", scope=scope)
            for scope in ("proj-a", "proj-b", "global")
            for i in range(100)
        ]

    def test_shards_by_scope_past_the_threshold(
        self, many_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        rendered = render_index(many_notes, manifest_ci)
        assert rendered.sharded
        assert set(rendered.shards) == {"proj-a", "proj-b"}

    def test_master_becomes_one_line_per_scope_plus_the_global_memories(
        self, many_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        master = render_index(many_notes, manifest_ci).master
        assert "[indices/proj-a.md](indices/proj-a.md)" in master
        assert "[indices/proj-b.md](indices/proj-b.md)" in master
        assert "100 memória(s) ativa(s)" in master
        # Global has no shard to live in: it stays listed in full.
        assert "[[INS_global-0]]" in master
        assert "[[INS_proj-a-0]]" not in master

    def test_shards_carry_their_scope_memories(
        self, many_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        shard = render_index(many_notes, manifest_ci).shards["proj-a"]
        assert "[[INS_proj-a-0]]" in shard
        assert "[[INS_proj-b-0]]" not in shard
        assert "[[index]]" in shard  # points back at the master

    def test_threshold_is_tunable(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        notes = [make_note(tmp_vault, f"INS_a{i}") for i in range(3)]
        assert not render_index(notes, manifest_ci).sharded
        assert render_index(notes, manifest_ci, shard_threshold=5).sharded


# ══════════════════════════════════════════════════════════════════════
# hot.md (§8.4, R8.2)
# ══════════════════════════════════════════════════════════════════════
class TestRenderHot:
    def test_one_section_per_active_scope(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # "O foco atual" is not one thing when several projects are live (R8.2).
        hot = render_hot(sample_notes, manifest_ci)
        assert "## proj-a" in hot
        assert "## global" in hot

    def test_excludes_archived_memories(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        hot = render_hot(sample_notes, manifest_ci)
        assert "INS_stale-fact" not in hot
        assert "## proj-b" not in hot

    def test_includes_episodic_and_research(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # hot.md is a cache of attention, not a MOC: what happened recently
        # counts.
        hot = render_hot(sample_notes, manifest_ci)
        assert "[[2026-01-15_MEM_session-one]]" in hot
        assert "[[2026-01-15_RES_link-expansion]]" in hot

    def test_every_line_points_at_a_real_note(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        hot = render_hot(sample_notes, manifest_ci)
        ids = {note.note_id for note in sample_notes}
        linked = set(re.findall(r"\[\[([^\]]+)\]\]", hot))
        assert linked <= ids | {"index"}

    def test_sections_are_capped(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        notes = [make_note(tmp_vault, f"INS_a{i}") for i in range(40)]
        hot = render_hot(notes, manifest_ci)
        listed = re.findall(r"^- \[\[", hot, re.MULTILINE)
        assert len(listed) == HOT_SECTION_MAX_ENTRIES
        assert f"mais {40 - HOT_SECTION_MAX_ENTRIES} em `proj-a`" in hot

    def test_cap_is_tunable(self, tmp_vault: VaultPaths, manifest_ci: Manifest) -> None:
        notes = [make_note(tmp_vault, f"INS_a{i}") for i in range(10)]
        hot = render_hot(notes, manifest_ci, max_entries=3)
        assert len(re.findall(r"^- \[\[", hot, re.MULTILINE)) == 3
        assert "mais 7" in hot

    def test_ranks_by_recency_within_a_section(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        old = make_note(tmp_vault, "INS_old", last_accessed=date(2026, 1, 1))
        fresh = make_note(tmp_vault, "INS_fresh", last_accessed=date(2026, 6, 1))
        hot = render_hot([old, fresh], manifest_ci)
        assert hot.index("[[INS_fresh]]") < hot.index("[[INS_old]]")

    def test_active_scopes_selects_and_orders_the_sections(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # The consolidator knows which scopes saw recent writes; the renderer
        # takes its word for it.
        hot = render_hot(sample_notes, manifest_ci, active_scopes=["global", "proj-a"])
        assert hot.index("## global") < hot.index("## proj-a")

        only_global = render_hot(sample_notes, manifest_ci, active_scopes=["global"])
        assert "## proj-a" not in only_global

    def test_an_active_scope_with_nothing_live_is_dropped(
        self, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # proj-b's only memory is archived: an empty section is noise.
        hot = render_hot(sample_notes, manifest_ci, active_scopes=["proj-b", "proj-a"])
        assert "## proj-b" not in hot
        assert "## proj-a" in hot

    def test_an_empty_vault_says_so(self, manifest_ci: Manifest) -> None:
        assert "Nenhum escopo ativo ainda" in render_hot((), manifest_ci)

    def test_initial_hot_is_the_empty_render(self, manifest_ci: Manifest) -> None:
        assert render_initial_hot(manifest_ci) == render_hot((), manifest_ci)

    def test_is_deterministic(self, sample_notes: list[Note], manifest_ci: Manifest) -> None:
        assert render_hot(sample_notes, manifest_ci) == render_hot(sample_notes, manifest_ci)


# ══════════════════════════════════════════════════════════════════════
# Writers
# ══════════════════════════════════════════════════════════════════════
class TestWriters:
    def test_write_default_config_never_clobbers_an_existing_manifest(
        self, tmp_path: Path
    ) -> None:
        # `init` is idempotent (§10) and the manifest is the one file a human
        # owns end to end.
        paths = VaultPaths.for_root(tmp_path / "fresh")
        write_default_config(paths)
        paths.config_file.write_text("mecha_brain: {spec_version: '0.1'}\n", encoding="utf-8")
        write_default_config(paths)
        assert paths.config_file.read_text(encoding="utf-8").startswith("mecha_brain:")

    def test_write_default_config_overwrites_on_demand(self, tmp_path: Path) -> None:
        paths = VaultPaths.for_root(tmp_path / "fresh")
        paths.config_file.parent.mkdir(parents=True)
        paths.config_file.write_text("stale\n", encoding="utf-8")
        write_default_config(paths, overwrite=True)
        assert Manifest.load(paths.config_file).retrieval.store == "numpy"

    def test_write_schema_always_regenerates(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        tmp_vault.schema_file.write_text("obsoleto\n", encoding="utf-8")
        write_schema(tmp_vault, manifest_ci)
        assert "obsoleto" not in tmp_vault.schema_file.read_text(encoding="utf-8")

    def test_write_agents_md_preserves_outside_text_across_a_sync(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest, manifest_data_ci: dict[str, Any]
    ) -> None:
        write_agents_md(tmp_vault, manifest_ci)
        with tmp_vault.agents_file.open("a", encoding="utf-8") as handle:
            handle.write("\n## Convenções desta vault\n\nnão me apague\n")

        manifest_data_ci["agents"].append({"id": "gamma", "display_name": "Gamma"})
        write_agents_md(tmp_vault, Manifest.from_mapping(manifest_data_ci))

        content = tmp_vault.agents_file.read_text(encoding="utf-8")
        assert "não me apague" in content
        assert "`gamma`" in content

    def test_write_agents_md_is_idempotent_on_disk(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        write_agents_md(tmp_vault, manifest_ci)
        first = tmp_vault.agents_file.read_text(encoding="utf-8")
        write_agents_md(tmp_vault, manifest_ci)
        assert tmp_vault.agents_file.read_text(encoding="utf-8") == first

    def test_write_agents_md_refuses_an_ambiguous_file(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        tmp_vault.agents_file.write_text(f"{MANAGED_BLOCK_BEGIN}\nsem fim\n", encoding="utf-8")
        with pytest.raises(MechabrainError):
            write_agents_md(tmp_vault, manifest_ci)

    def test_write_hot_writes_the_blackboard(
        self, tmp_vault: VaultPaths, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        write_hot(tmp_vault, sample_notes, manifest_ci)
        assert "## proj-a" in tmp_vault.hot_file.read_text(encoding="utf-8")

    def test_write_index_writes_master_and_shards(
        self, tmp_vault: VaultPaths, manifest_ci: Manifest
    ) -> None:
        notes = [
            make_note(tmp_vault, f"INS_{scope}-{i}", scope=scope)
            for scope in ("proj-a", "proj-b")
            for i in range(150)
        ]
        rendered = write_index(tmp_vault, notes, manifest_ci)
        assert rendered.sharded
        assert tmp_vault.scope_index("proj-a").is_file()
        assert "[[INS_proj-a-0]]" in tmp_vault.scope_index("proj-a").read_text(encoding="utf-8")
        assert "indices/proj-a.md" in tmp_vault.index_file.read_text(encoding="utf-8")

    def test_write_index_prunes_a_shard_that_no_longer_applies(
        self, tmp_vault: VaultPaths, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # A stale shard is worse than a missing one: it maps memories that moved.
        stale = tmp_vault.scope_index("proj-z")
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("# antigo\n", encoding="utf-8")
        write_index(tmp_vault, sample_notes, manifest_ci)
        assert not stale.exists()

    def test_write_index_unshards_when_the_vault_shrinks(
        self, tmp_vault: VaultPaths, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        many = [
            make_note(tmp_vault, f"INS_proj-a-{i}", scope="proj-a") for i in range(300)
        ]
        write_index(tmp_vault, many, manifest_ci)
        assert tmp_vault.scope_index("proj-a").is_file()

        write_index(tmp_vault, sample_notes, manifest_ci)
        assert not tmp_vault.scope_index("proj-a").exists()
        assert "[[2026-01-15_INS_vector-search]]" in tmp_vault.index_file.read_text(
            encoding="utf-8"
        )

    def test_generated_files_carry_no_absolute_path(
        self, tmp_vault: VaultPaths, sample_notes: list[Note], manifest_ci: Manifest
    ) -> None:
        # §12.1: the deployment must survive being moved between machines.
        write_default_config(tmp_vault, overwrite=True)
        write_schema(tmp_vault, manifest_ci)
        write_agents_md(tmp_vault, manifest_ci)
        write_hot(tmp_vault, sample_notes, manifest_ci)
        write_index(tmp_vault, sample_notes, manifest_ci)
        for path in (
            tmp_vault.config_file,
            tmp_vault.schema_file,
            tmp_vault.agents_file,
            tmp_vault.hot_file,
            tmp_vault.index_file,
        ):
            assert not ABSOLUTE_PATH_RE.search(path.read_text(encoding="utf-8")), path
