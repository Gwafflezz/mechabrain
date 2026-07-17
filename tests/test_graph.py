"""The authored link graph: extraction, resolution, persistence, expansion."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mechabrain.discovery import VaultPaths
from mechabrain.errors import ManifestError, NoteNotFound, SchemaViolation
from mechabrain.graph import (
    DEFAULT_RELATION,
    SUPERSEDES_RELATION,
    WIKILINK_RELATION,
    DropReason,
    EdgeOrigin,
    LinkEdge,
    LinkGraph,
    ViaChain,
    extract_wikilinks,
)
from mechabrain.manifest import Manifest
from mechabrain.note import Note


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def linked_notes(
    tmp_vault: VaultPaths, write_note: Callable[..., Note]
) -> Callable[..., None]:
    """Factory writing a bare semantic note that links wherever you say.

    ::

        link("a", body="See [[b]].")
        link("b", supersedes="[[a]]")
    """

    def _link(
        note_id: str,
        *,
        body: str = "",
        supersedes: Any = None,
        folder: Path | None = None,
    ) -> Note:
        frontmatter: dict[str, Any] = {"title": note_id, "scope": "proj-a"}
        if supersedes is not None:
            frontmatter["supersedes"] = supersedes
        target = (folder or tmp_vault.semantic_dir) / f"{note_id}.md"
        return write_note(target, frontmatter, body)

    return _link


def edge_pairs(graph: LinkGraph) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in graph.edges}


# ══════════════════════════════════════════════════════════════════════
# Wikilink extraction
# ══════════════════════════════════════════════════════════════════════
def test_extracts_plain_wikilink() -> None:
    assert extract_wikilinks("See [[target-note]] for context.") == ["target-note"]


def test_extracts_aliased_heading_and_embed_forms() -> None:
    body = (
        "[[target|an alias]] and [[other#Heading]] and [[third#^block-id]] "
        "and an embed ![[fourth]]."
    )
    assert extract_wikilinks(body) == ["target", "other", "third", "fourth"]


def test_intra_note_heading_link_is_not_an_edge() -> None:
    assert extract_wikilinks("Jump to [[#Evidence]].") == []


def test_wikilink_inside_fenced_block_is_documentation_not_an_edge() -> None:
    body = "Real [[a]].\n\n```markdown\nWrite [[b]] to link.\n```\n\nAlso [[c]]."
    assert extract_wikilinks(body) == ["a", "c"]


def test_wikilink_inside_tilde_fence_is_ignored() -> None:
    body = "~~~\n[[hidden]]\n~~~\n[[visible]]"
    assert extract_wikilinks(body) == ["visible"]


def test_wikilink_inside_inline_code_is_ignored() -> None:
    assert extract_wikilinks("The syntax is `[[note]]`, e.g. [[real-note]].") == [
        "real-note"
    ]


def test_duplicate_wikilinks_are_reported_once_as_one_edge(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a", body="[[b]] and again [[b]].")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert len(graph.edges) == 1


# ══════════════════════════════════════════════════════════════════════
# Building: the three authored sources
# ══════════════════════════════════════════════════════════════════════
def test_builds_edge_from_body_wikilink(
    tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert (
        "2026-01-15_RES_link-expansion",
        "2026-01-15_INS_vector-search",
    ) in edge_pairs(graph)


def test_builds_edge_from_supersedes_frontmatter(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("old")
    linked_notes("new", supersedes="[[old]]")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    (edge,) = graph.edges
    assert (edge.source, edge.target) == ("new", "old")
    assert edge.relation == SUPERSEDES_RELATION
    assert edge.origin is EdgeOrigin.SUPERSEDES


def test_supersedes_accepts_bare_id_and_list(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("old-one")
    linked_notes("old-two")
    linked_notes("bare", supersedes="old-one")
    linked_notes("merger", supersedes=["[[old-one]]", "[[old-two]]"])
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert ("bare", "old-one") in edge_pairs(graph)
    assert {("merger", "old-one"), ("merger", "old-two")} <= edge_pairs(graph)


def test_builds_edge_from_links_jsonl(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    tmp_vault.links_file.write_text(
        json.dumps(
            {
                "a": "a",
                "b": "b",
                "relation": "explains",
                "created": "2026-01-15T10:00:00+00:00",
                "agent": "alpha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    (edge,) = graph.edges
    assert (edge.source, edge.target, edge.relation) == ("a", "b", "explains")
    assert edge.origin is EdgeOrigin.AUTHORED
    assert edge.agent == "alpha"


def test_absent_links_file_is_an_empty_graph_not_an_error(
    tmp_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    assert not tmp_vault.links_file.exists()
    assert LinkGraph.build(tmp_vault, manifest_ci).edges == ()


def test_blank_lines_in_links_file_are_skipped(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    tmp_vault.links_file.write_text(
        '\n{"a": "a", "b": "b"}\n\n', encoding="utf-8"
    )
    assert len(LinkGraph.build(tmp_vault, manifest_ci).edges) == 1


def test_links_file_relation_defaults_when_unwritten(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    tmp_vault.links_file.write_text('{"a": "a", "b": "b"}\n', encoding="utf-8")
    (edge,) = LinkGraph.build(tmp_vault, manifest_ci).edges
    assert edge.relation == DEFAULT_RELATION


# ── links.jsonl is authored truth: it fails loud (R5.1) ─────────────
def test_malformed_links_line_names_the_line_number(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    tmp_vault.links_file.write_text(
        '{"a": "a", "b": "b"}\nnot json at all\n', encoding="utf-8"
    )
    with pytest.raises(SchemaViolation, match=r":2\b"):
        LinkGraph.build(tmp_vault, manifest_ci)


def test_links_line_with_unknown_key_is_rejected(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    tmp_vault.links_file.write_text(
        '{"a": "a", "b": "b", "reltion": "x"}\n', encoding="utf-8"
    )
    with pytest.raises(SchemaViolation, match="reltion"):
        LinkGraph.build(tmp_vault, manifest_ci)


def test_links_line_missing_an_endpoint_is_rejected(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    tmp_vault.links_file.write_text('{"a": "a"}\n', encoding="utf-8")
    with pytest.raises(SchemaViolation, match="'b'"):
        LinkGraph.build(tmp_vault, manifest_ci)


# ══════════════════════════════════════════════════════════════════════
# Which notes are nodes
# ══════════════════════════════════════════════════════════════════════
def test_nodes_are_the_notes_retrieval_can_return(
    tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    assert set(LinkGraph.build(tmp_vault, manifest_ci).note_ids) == {
        note.note_id for note in sample_notes
    }


def test_generated_moc_is_not_a_node_and_does_not_bridge_notes(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    """R8.1: index.md links are derived. A master MOC as a node would put every
    note two hops from every other, which is not an authored relation."""
    linked_notes("a")
    linked_notes("b")
    tmp_vault.index_file.write_text("- [[a]]\n- [[b]]\n", encoding="utf-8")
    tmp_vault.hot_file.write_text("- [[a]]\n- [[b]]\n", encoding="utf-8")
    (tmp_vault.indices_dir / "proj-a.md").write_text("- [[a]]\n- [[b]]\n", encoding="utf-8")

    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert graph.note_ids == ("a", "b")
    assert graph.edges == ()
    assert graph.expand(["a"], hops=2) == {"a": ViaChain(("a",), ())}


def test_proposals_are_not_nodes(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    (tmp_vault.inbox_dir / "2026-01-15_AI-PROPOSAL_x.md").write_text(
        "Change [[a]].\n", encoding="utf-8"
    )
    assert LinkGraph.build(tmp_vault, manifest_ci).note_ids == ("a",)


def test_read_only_index_folders_are_nodes(
    make_vault: Callable[..., VaultPaths],
    manifest_data_ci: dict[str, Any],
    write_note: Callable[..., Note],
) -> None:
    """A human note the deployment indexes as context is a legitimate link
    target: an agent citing it authored that edge."""
    manifest_data_ci["zones"]["read_only_index"] = ["Human/"]
    paths = make_vault(manifest_data_ci)
    manifest = Manifest.from_mapping(manifest_data_ci)
    write_note(paths.root / "Human" / "human-note.md", {"title": "Human"}, "Text.")
    write_note(
        paths.semantic_dir / "insight.md", {"title": "I"}, "Derived from [[human-note]]."
    )

    graph = LinkGraph.build(paths, manifest)
    assert set(graph.note_ids) == {"human-note", "insight"}
    assert ("insight", "human-note") in edge_pairs(graph)


def test_research_notes_are_not_nodes_when_the_zone_is_disabled(
    make_vault: Callable[..., VaultPaths],
    manifest_data_ci: dict[str, Any],
    write_note: Callable[..., Note],
) -> None:
    manifest_data_ci["zones"]["research_enabled"] = False
    paths = make_vault(manifest_data_ci)
    manifest = Manifest.from_mapping(manifest_data_ci)
    write_note(paths.research_dir / "report.md", {"title": "R"}, "")
    write_note(paths.semantic_dir / "kept.md", {"title": "K"}, "")
    assert LinkGraph.build(paths, manifest).note_ids == ("kept",)


def test_build_accepts_pre_scanned_notes(
    tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    """The indexer already walked the vault; the graph must not walk it again."""
    graph = LinkGraph.build(tmp_vault, manifest_ci, notes=sample_notes)
    assert set(graph.note_ids) == {note.note_id for note in sample_notes}
    assert len(graph.edges) == 1


# ══════════════════════════════════════════════════════════════════════
# Resolution
# ══════════════════════════════════════════════════════════════════════
def test_resolves_path_qualified_and_extended_targets(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("target")
    linked_notes("a", body="[[Semantic/target]] and [[target.md]] and [[TARGET]]")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert edge_pairs(graph) == {("a", "target")}
    assert graph.broken_links == ()


def test_broken_link_is_ignored_but_counted(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a", body="See [[nowhere]] and [[b]].")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)

    assert edge_pairs(graph) == {("a", "b")}
    (broken,) = graph.broken_links
    assert (broken.source, broken.raw_target) == ("a", "nowhere")
    assert broken.origin is EdgeOrigin.WIKILINK
    assert broken.reason is DropReason.BROKEN


def test_broken_supersedes_and_authored_edges_are_counted_too(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a", supersedes="[[deleted]]")
    tmp_vault.links_file.write_text('{"a": "a", "b": "vanished"}\n', encoding="utf-8")
    graph = LinkGraph.build(tmp_vault, manifest_ci)

    assert graph.edges == ()
    assert {d.origin for d in graph.broken_links} == {
        EdgeOrigin.SUPERSEDES,
        EdgeOrigin.AUTHORED,
    }


def test_self_link_is_ignored_but_counted(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a", body="This is [[a]], me.")
    graph = LinkGraph.build(tmp_vault, manifest_ci)

    assert graph.edges == ()
    assert graph.neighbors("a") == ()
    (self_link,) = graph.self_links
    assert self_link.reason is DropReason.SELF
    assert graph.expand(["a"], hops=2) == {"a": ViaChain(("a",), ())}


def test_ambiguous_basenames_are_reported_not_guessed_quietly(
    make_vault: Callable[..., VaultPaths],
    manifest_data_ci: dict[str, Any],
    write_note: Callable[..., Note],
) -> None:
    """Note ids are basenames, so a human folder tree collides. First in path
    order owns the id -- as it would in an editor -- and the clash is reported."""
    manifest_data_ci["zones"]["read_only_index"] = ["Human/"]
    paths = make_vault(manifest_data_ci)
    manifest = Manifest.from_mapping(manifest_data_ci)
    write_note(paths.root / "Human" / "A" / "README.md", {"title": "A"}, "")
    write_note(paths.root / "Human" / "B" / "README.md", {"title": "B"}, "")
    write_note(paths.semantic_dir / "insight.md", {"title": "I"}, "See [[README]].")

    graph = LinkGraph.build(paths, manifest)
    assert graph.ambiguous_ids == ("README",)
    assert graph.path_of("README") == paths.root / "Human" / "A" / "README.md"
    assert ("insight", "README") in edge_pairs(graph)


# ══════════════════════════════════════════════════════════════════════
# Neighbourhood: undirected, direction preserved
# ══════════════════════════════════════════════════════════════════════
def test_neighborhood_is_undirected_but_the_edge_keeps_its_direction(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("new", supersedes="[[old]]")
    linked_notes("old")
    graph = LinkGraph.build(tmp_vault, manifest_ci)

    assert graph.neighbor_ids("new") == ("old",)
    assert graph.neighbor_ids("old") == ("new",)
    (edge,) = graph.neighbors("old")
    assert (edge.source, edge.target) == ("new", "old")
    assert edge.other_end("old") == "new"


def test_neighbors_of_isolated_or_unknown_note_is_empty(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("lonely")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert graph.neighbors("lonely") == ()
    assert graph.neighbors("never-existed") == ()
    assert graph.has_note("lonely")
    assert not graph.has_note("never-existed")


def test_same_pair_keeps_one_edge_per_distinct_relation(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("new", body="Replaces [[old]].", supersedes="[[old]]")
    linked_notes("old")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert {edge.relation for edge in graph.edges} == {
        WIKILINK_RELATION,
        SUPERSEDES_RELATION,
    }
    assert graph.neighbor_ids("new") == ("old",)


def test_other_end_rejects_a_note_off_the_edge() -> None:
    edge = LinkEdge("a", "b", WIKILINK_RELATION, EdgeOrigin.WIKILINK)
    with pytest.raises(ValueError, match="not an endpoint"):
        edge.other_end("c")


def test_path_of_unknown_note_raises(
    tmp_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    with pytest.raises(NoteNotFound):
        LinkGraph.build(tmp_vault, manifest_ci).path_of("nope")


# ══════════════════════════════════════════════════════════════════════
# add_edge (memory_link, §7.2)
# ══════════════════════════════════════════════════════════════════════
def test_add_edge_persists_one_git_tracked_jsonl_line(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    edge = graph.add_edge("a", "b", "explains", "alpha")

    assert edge.origin is EdgeOrigin.AUTHORED
    # _meta/, never _meta/index/: an authored edge is truth, not derived state.
    assert tmp_vault.links_file.parent == tmp_vault.meta_dir
    (line,) = tmp_vault.links_file.read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {
        "a": "a",
        "b": "b",
        "relation": "explains",
        "created": edge.created,
        "agent": "alpha",
    }


def test_add_edge_is_visible_to_expansion_without_a_rebuild(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    graph.add_edge("a", "b", agent="alpha")

    assert graph.neighbor_ids("a") == ("b",)
    assert set(graph.expand(["a"], hops=1)) == {"a", "b"}
    assert LinkGraph.build(tmp_vault, manifest_ci).neighbor_ids("a") == ("b",)


def test_add_edge_appends_and_keeps_earlier_lines(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    linked_notes("c")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    graph.add_edge("a", "b", agent="alpha")
    graph.add_edge("b", "c", agent="beta")

    lines = tmp_vault.links_file.read_text(encoding="utf-8").splitlines()
    assert [(json.loads(line)["a"], json.loads(line)["b"]) for line in lines] == [
        ("a", "b"),
        ("b", "c"),
    ]


def test_add_edge_appends_to_a_file_with_no_trailing_newline(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    linked_notes("c")
    tmp_vault.links_file.write_text('{"a": "a", "b": "b"}', encoding="utf-8")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    graph.add_edge("b", "c", agent="alpha")

    assert len(tmp_vault.links_file.read_text(encoding="utf-8").splitlines()) == 2


def test_add_edge_is_idempotent(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    """An agent re-linking is normal; an append-only log that grows on every
    repeat is a log someone edits by hand."""
    linked_notes("a")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    first = graph.add_edge("a", "b", "explains", "alpha")
    again = graph.add_edge("a", "b", "explains", "beta")

    assert again == first
    assert len(tmp_vault.links_file.read_text(encoding="utf-8").splitlines()) == 1
    assert len(graph.edges) == 1


def test_add_edge_accepts_wikilink_and_path_forms(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    edge = graph.add_edge("[[a]]", "Semantic/b.md", agent="alpha")
    assert (edge.source, edge.target) == ("a", "b")


def test_add_edge_rejects_an_unknown_agent(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    with pytest.raises(ManifestError, match="unknown agent"):
        graph.add_edge("a", "b", agent="ghost")
    assert not tmp_vault.links_file.exists()


def test_add_edge_rejects_an_unknown_note(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    with pytest.raises(NoteNotFound, match="nowhere"):
        graph.add_edge("a", "nowhere", agent="alpha")


def test_add_edge_rejects_a_self_link(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    with pytest.raises(ValueError, match="itself"):
        graph.add_edge("a", "[[a]]", agent="alpha")


def test_add_edge_rejects_an_empty_relation(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    with pytest.raises(ValueError, match="relation"):
        graph.add_edge("a", "b", "   ", "alpha")


def test_add_edge_lock_lives_in_the_gitignored_runtime_layer(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a")
    linked_notes("b")
    LinkGraph.build(tmp_vault, manifest_ci).add_edge("a", "b", agent="alpha")
    assert list(tmp_vault.meta_dir.glob("*.lock")) == []
    assert list(tmp_vault.index_dir.glob("*.lock")) != []


# ══════════════════════════════════════════════════════════════════════
# Expansion (§7.1)
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def chain_graph(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> LinkGraph:
    """``a -> b -> c -> d``, plus an unreachable ``island``."""
    linked_notes("a", body="[[b]]")
    linked_notes("b", body="[[c]]")
    linked_notes("c", body="[[d]]")
    linked_notes("d")
    linked_notes("island")
    return LinkGraph.build(tmp_vault, manifest_ci)


def test_hops_zero_returns_the_seeds_untouched(chain_graph: LinkGraph) -> None:
    """`link_expansion.default_hops: 0` means off (§5)."""
    assert chain_graph.expand(["a"], hops=0) == {"a": ViaChain(("a",), ())}


def test_hops_one_reaches_the_direct_neighbourhood(chain_graph: LinkGraph) -> None:
    reached = chain_graph.expand(["b"], hops=1)
    assert set(reached) == {"a", "b", "c"}
    assert reached["c"].hops == 1
    assert reached["b"].hops == 0


def test_hops_two_reaches_two_edges_out_and_no_further(chain_graph: LinkGraph) -> None:
    reached = chain_graph.expand(["a"], hops=2)
    assert set(reached) == {"a", "b", "c"}
    assert reached["c"].path == ("a", "b", "c")
    assert reached["c"].hops == 2


def test_expansion_defaults_to_the_manifest_hop_count(
    chain_graph: LinkGraph, manifest_ci: Manifest
) -> None:
    assert manifest_ci.retrieval.link_expansion.default_hops == 1
    assert set(chain_graph.expand(["a"])) == set(chain_graph.expand(["a"], hops=1))


def test_expansion_above_the_manifest_ceiling_is_refused_not_clamped(
    chain_graph: LinkGraph, manifest_ci: Manifest
) -> None:
    """Clamping would tell the caller it walked five hops when it walked two."""
    ceiling = manifest_ci.retrieval.link_expansion.max_hops
    with pytest.raises(ValueError, match="max_hops"):
        chain_graph.expand(["a"], hops=ceiling + 1)


def test_negative_hops_is_refused(chain_graph: LinkGraph) -> None:
    with pytest.raises(ValueError, match="hops"):
        chain_graph.expand(["a"], hops=-1)


def test_via_chain_carries_the_provenance_path(chain_graph: LinkGraph) -> None:
    via = chain_graph.expand(["a"], hops=2)["c"]
    assert via.render() == "[[a]] → [[b]] → [[c]]"
    assert via.seed == "a"
    assert via.note_id == "c"
    assert via.relations == (WIKILINK_RELATION, WIKILINK_RELATION)


def test_expansion_ignores_seeds_that_are_not_notes(chain_graph: LinkGraph) -> None:
    """A stale index can hand over an id the vault no longer has."""
    assert set(chain_graph.expand(["a", "deleted-note"], hops=1)) == {"a", "b"}


def test_expansion_of_no_seeds_reaches_nothing(chain_graph: LinkGraph) -> None:
    assert chain_graph.expand([], hops=2) == {}


def test_expansion_does_not_leak_into_a_disconnected_component(
    chain_graph: LinkGraph,
) -> None:
    assert "island" not in chain_graph.expand(["a"], hops=2)


def test_multiple_seeds_keep_the_shortest_chain(chain_graph: LinkGraph) -> None:
    reached = chain_graph.expand(["a", "c"], hops=1)
    assert reached["b"].hops == 1
    assert reached["b"].seed in {"a", "c"}
    assert reached["d"].path == ("c", "d")


def test_a_seed_reached_from_another_seed_stays_a_seed(chain_graph: LinkGraph) -> None:
    reached = chain_graph.expand(["a", "b"], hops=1)
    assert reached["b"] == ViaChain(("b",), ())
    assert reached["b"].hops == 0


def test_expansion_is_deterministic_regardless_of_seed_order(
    chain_graph: LinkGraph,
) -> None:
    assert chain_graph.expand(["a", "c"], hops=2) == chain_graph.expand(
        ["c", "a"], hops=2
    )


def test_expansion_terminates_on_a_cycle(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a", body="[[b]]")
    linked_notes("b", body="[[c]]")
    linked_notes("c", body="[[a]]")
    graph = LinkGraph.build(tmp_vault, manifest_ci)

    reached = graph.expand(["a"], hops=2)
    assert set(reached) == {"a", "b", "c"}
    assert reached["a"].hops == 0
    assert reached["b"].hops == 1
    assert reached["c"].hops == 1  # via the c -> a edge, walked backwards


def test_expansion_terminates_on_a_two_note_cycle(
    tmp_vault: VaultPaths, manifest_ci: Manifest, linked_notes: Callable[..., Note]
) -> None:
    linked_notes("a", body="[[b]]")
    linked_notes("b", body="[[a]]")
    graph = LinkGraph.build(tmp_vault, manifest_ci)
    assert set(graph.expand(["a"], hops=2)) == {"a", "b"}
