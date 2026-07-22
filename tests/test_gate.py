"""The §8.2 write gate: what it enforces, what it only reports, and why.

The suite is organised around that split. Rejection tests pin the mechanical
checks; warning tests pin that judgement items never block; routing tests pin
that `Episodic/` and `Research/` skip the checklist.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from mechabrain.errors import GateRejected
from mechabrain.gate import (
    ATOMIC_BODY_CHARS,
    ATOMIC_MAX_SECTIONS,
    DEDUP_CANDIDATES,
    FULL_GATE_TYPES,
    GateResult,
    NearDuplicate,
    evaluate,
)
from mechabrain.manifest import Manifest


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════
class FakeSearch:
    """A `SearchFn` stub that returns a fixed hit list and records its calls.

    Deliberately returns `similarity` as a raw cosine, which is the contract the
    gate documents: a min-max normalised fusion score cannot be compared against
    `dedup_similarity`.
    """

    def __init__(self, hits: Sequence[Mapping[str, Any]] = ()) -> None:
        self.hits = list(hits)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        query: str,
        *,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        self.calls.append({"query": query, "k": k, "filters": dict(filters or {})})
        return list(self.hits)


def hit(wikilink: str, similarity: float, **extra: Any) -> dict[str, Any]:
    """One search hit carrying the provenance R7.1 demands."""
    return {
        "wikilink": wikilink,
        "id": wikilink.strip("[]"),
        "similarity": similarity,
        "title": extra.pop("title", "A neighbour"),
        "path": extra.pop("path", f"mecha-brain/Semantic/{wikilink.strip('[]')}.md"),
        **extra,
    }


@pytest.fixture
def no_hits() -> FakeSearch:
    """A search over an empty scope: nothing to be a duplicate of."""
    return FakeSearch()


@pytest.fixture
def meta() -> dict[str, Any]:
    """Frontmatter that passes every enforced check, for `manifest_ci`."""
    return {
        "title": "Brute force is fast enough at vault scale",
        "tags": ["mem/semantic", "agent/alpha"],
        "agent": "alpha",
        "profile": "tutor",
        "scope": "proj-a",
        "source": "session-2026-01-15",
        "confidence": "medium",
    }


@pytest.fixture
def body() -> str:
    return "Below ten thousand chunks, brute-force cosine beats ANN on latency."


@pytest.fixture
def gate(
    manifest_ci: Manifest, meta: dict[str, Any], body: str, no_hits: FakeSearch
) -> Callable[..., GateResult]:
    """Call the gate with the passing baseline, overriding one thing at a time.

    ::

        gate()                                  # approved
        gate(meta={**meta, "source": ""})       # one enforced check broken
        gate(type="procedural", search_fn=...)  # a different route
    """

    def _gate(
        type: str = "semantic",
        *,
        meta: Mapping[str, Any] = meta,
        body: str = body,
        manifest: Manifest = manifest_ci,
        search_fn: Any = no_hits,
    ) -> GateResult:
        return evaluate(type, dict(meta), body, manifest, search_fn=search_fn)

    return _gate


def checks(result: GateResult) -> set[str]:
    return {issue.check for issue in result.rejections}


def warned(result: GateResult) -> set[str]:
    return {issue.check for issue in result.warnings}


# ══════════════════════════════════════════════════════════════════════
# The happy path
# ══════════════════════════════════════════════════════════════════════
def test_a_clean_semantic_write_is_approved(gate: Callable[..., GateResult]) -> None:
    result = gate()
    assert result.approved
    assert result.rejections == ()
    assert result.near_duplicates == ()


def test_approval_is_not_the_absence_of_warnings(gate: Callable[..., GateResult]) -> None:
    """Judgement items ride along with a perfectly good write -- that is normal."""
    result = gate()
    assert result.approved
    assert "reusable" in warned(result)


def test_raise_if_rejected_is_silent_when_approved(gate: Callable[..., GateResult]) -> None:
    gate().raise_if_rejected()


# ══════════════════════════════════════════════════════════════════════
# Item 2 -- duplicates (enforced)
# ══════════════════════════════════════════════════════════════════════
def test_near_duplicate_above_threshold_is_rejected(
    gate: Callable[..., GateResult], manifest_ci: Manifest
) -> None:
    above = manifest_ci.maintenance.dedup_similarity + 0.05
    search = FakeSearch([hit("[[2026-01-15_INS_vector-search]]", above)])
    result = gate(search_fn=search)

    assert not result.approved
    assert checks(result) == {"duplicate"}
    assert [d.wikilink for d in result.near_duplicates] == ["[[2026-01-15_INS_vector-search]]"]


def test_similarity_at_the_threshold_is_not_a_duplicate(
    gate: Callable[..., GateResult], manifest_ci: Manifest
) -> None:
    """§8.2 says *above* `dedup_similarity`; the boundary itself passes."""
    search = FakeSearch([hit("[[x]]", manifest_ci.maintenance.dedup_similarity)])
    result = gate(search_fn=search)
    assert result.approved
    assert result.near_duplicates == ()


def test_threshold_comes_from_the_manifest_not_a_literal(
    gate: Callable[..., GateResult],
    manifest_data_ci: dict[str, Any],
) -> None:
    """A deployment that loosens `dedup_similarity` must actually loosen the gate (P6)."""
    data = copy.deepcopy(manifest_data_ci)
    data["maintenance"]["dedup_similarity"] = 0.5
    strict = Manifest.from_mapping(data)

    search = FakeSearch([hit("[[x]]", 0.6)])
    assert gate(search_fn=search).approved  # 0.6 < 0.92, the CI default
    assert not gate(search_fn=FakeSearch([hit("[[x]]", 0.6)]), manifest=strict).approved


def test_declared_supersedes_lets_a_duplicate_through(
    gate: Callable[..., GateResult], meta: dict[str, Any], manifest_ci: Manifest
) -> None:
    """The spec demands an explicit decision, not the absence of neighbours."""
    above = manifest_ci.maintenance.dedup_similarity + 0.05
    search = FakeSearch([hit("[[2026-01-15_INS_vector-search]]", above)])
    result = gate(meta={**meta, "supersedes": "[[2026-01-15_INS_vector-search]]"}, search_fn=search)

    assert result.approved
    assert len(result.near_duplicates) == 1, "still reported, just not blocking"


def test_declared_merge_lets_a_duplicate_through(
    gate: Callable[..., GateResult], meta: dict[str, Any], manifest_ci: Manifest
) -> None:
    above = manifest_ci.maintenance.dedup_similarity + 0.05
    search = FakeSearch([hit("[[x]]", above)])
    assert gate(meta={**meta, "merge": True}, search_fn=search).approved


def test_empty_supersedes_is_not_a_decision(
    gate: Callable[..., GateResult], meta: dict[str, Any], manifest_ci: Manifest
) -> None:
    above = manifest_ci.maintenance.dedup_similarity + 0.05
    search = FakeSearch([hit("[[x]]", above)])
    assert not gate(meta={**meta, "supersedes": "  "}, search_fn=search).approved


def test_dedup_search_is_filtered_to_the_candidates_scope_and_type(
    gate: Callable[..., GateResult], no_hits: FakeSearch
) -> None:
    """R6.5: a fact true in project A is not a duplicate of the same words in B."""
    gate(search_fn=no_hits)
    assert no_hits.calls[0]["filters"] == {"scope": "proj-a", "type": "semantic"}
    assert no_hits.calls[0]["k"] == DEDUP_CANDIDATES


def test_dedup_query_carries_title_and_body(
    gate: Callable[..., GateResult], no_hits: FakeSearch, body: str
) -> None:
    gate(search_fn=no_hits)
    query = no_hits.calls[0]["query"]
    assert "Brute force is fast enough" in query
    assert body in query


def test_missing_search_fn_is_a_wiring_error_not_an_approval(
    gate: Callable[..., GateResult]
) -> None:
    """R5.1: the gate must never approve a write whose dedup check never ran."""
    with pytest.raises(ValueError, match="search_fn"):
        gate(search_fn=None)


def test_a_broken_index_does_not_read_as_no_duplicates(
    gate: Callable[..., GateResult]
) -> None:
    def exploding(query: str, **kwargs: Any) -> Sequence[Any]:
        raise RuntimeError("index is corrupt")

    with pytest.raises(RuntimeError, match="corrupt"):
        gate(search_fn=exploding)


def test_dedup_is_skipped_when_the_scope_is_unusable(
    gate: Callable[..., GateResult], meta: dict[str, Any], no_hits: FakeSearch
) -> None:
    """Searching an invalid scope would silently search *every* scope."""
    result = gate(meta={**meta, "scope": ""}, search_fn=no_hits)
    assert not result.approved
    assert no_hits.calls == []


# ══════════════════════════════════════════════════════════════════════
# Items 4, 5, 6, 7 and the §6 registry rules (enforced)
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("value", ["", "   ", None])
def test_empty_source_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any], value: Any
) -> None:
    result = gate(meta={**meta, "source": value})
    assert checks(result) == {"source_missing"}


def test_missing_source_key_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    without = {k: v for k, v in meta.items() if k != "source"}
    assert checks(gate(meta=without)) == {"source_missing"}


def test_missing_scope_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    without = {k: v for k, v in meta.items() if k != "scope"}
    assert checks(gate(meta=without)) == {"scope_missing"}


def test_scope_outside_the_manifest_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    result = gate(meta={**meta, "scope": "proj-unknown"})
    assert checks(result) == {"scope_unknown"}
    assert "proj-a" in (result.rejections[0].hint or ""), "the hint lists the legal scopes"


def test_any_slug_is_a_scope_when_the_manifest_lists_none(
    gate: Callable[..., GateResult],
    meta: dict[str, Any],
    manifest_data_ci: dict[str, Any],
) -> None:
    """`scopes.known: []` means "any slug", per §5."""
    data = copy.deepcopy(manifest_data_ci)
    data["scopes"] = {"known": [], "default": "global"}
    open_scopes = Manifest.from_mapping(data)
    assert gate(meta={**meta, "scope": "anything-goes"}, manifest=open_scopes).approved


def test_an_open_scope_list_still_rejects_a_non_slug(
    gate: Callable[..., GateResult],
    meta: dict[str, Any],
    manifest_data_ci: dict[str, Any],
) -> None:
    """"Any slug" is not "any string" -- R6.5 still demands slug syntax."""
    data = copy.deepcopy(manifest_data_ci)
    data["scopes"] = {"known": [], "default": "global"}
    open_scopes = Manifest.from_mapping(data)
    result = gate(meta={**meta, "scope": "Not A Slug"}, manifest=open_scopes)
    assert checks(result) == {"scope_unknown"}


def test_unregistered_agent_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    result = gate(meta={**meta, "agent": "ghost"})
    assert checks(result) == {"agent_unknown"}
    assert "alpha" in (result.rejections[0].hint or "")


def test_missing_agent_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    without = {k: v for k, v in meta.items() if k != "agent"}
    assert "agent_missing" in checks(gate(meta=without))


def test_profile_outside_the_authors_registry_entry_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """R6.6 -- `profile:` is a persona of *that* runtime, not a free-text label."""
    assert checks(gate(meta={**meta, "profile": "researcher"})) == {"profile_unknown"}


def test_profile_is_optional(gate: Callable[..., GateResult], meta: dict[str, Any]) -> None:
    without = {k: v for k, v in meta.items() if k != "profile"}
    assert gate(meta=without).approved


def test_profile_of_an_unknown_agent_reports_the_agent_only(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """An unknown author has no profile list to check against; don't pile on."""
    result = gate(meta={**meta, "agent": "ghost", "profile": "tutor"})
    assert checks(result) == {"agent_unknown"}


def test_denied_frontmatter_key_is_rejected_citing_the_manifest_rule(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    result = gate(meta={**meta, "forbidden-key": "x"})
    assert checks(result) == {"denylist_keys"}
    assert "frontmatter.denylist_keys" in result.rejections[0].message
    assert result.rejections[0].rule == "R6.1"


def test_denied_tag_is_rejected_citing_the_manifest_rule(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    result = gate(meta={**meta, "tags": ["mem/semantic", "forbidden/tag"]})
    assert checks(result) == {"denylist_tags"}
    assert "frontmatter.denylist_tags" in result.rejections[0].message


def test_a_denied_tag_is_caught_through_its_hash_prefix(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """`#forbidden/tag` and `forbidden/tag` are the same tag."""
    assert checks(gate(meta={**meta, "tags": ["#forbidden/tag"]})) == {"denylist_tags"}


def test_confidence_outside_the_enum_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """§6 -- `confidence: high|medium|low` is a closed enum, not a free label."""
    result = gate(meta={**meta, "confidence": "certainly"})
    assert checks(result) == {"confidence_invalid"}
    assert "medium" in (result.rejections[0].hint or ""), "the hint lists the legal levels"


def test_confidence_is_optional(gate: Callable[..., GateResult], meta: dict[str, Any]) -> None:
    """Absent is fine: the writer stamps the honest `medium` default."""
    without = {k: v for k, v in meta.items() if k != "confidence"}
    assert gate(meta=without).approved


def test_an_episodic_note_faces_the_confidence_enum_too(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """§6 binds every note the kernel writes, whatever its type."""
    result = gate("episodic", meta={**meta, "confidence": "certainly"})
    assert checks(result) == {"confidence_invalid"}


def test_procedural_without_evidence_is_rejected(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """§8.2 item 6 -- a bad playbook propagates error to every agent."""
    assert checks(gate("procedural", meta=meta)) == {"evidence_missing"}


def test_procedural_with_evidence_is_approved(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    evidenced = {**meta, "evidence": "Ran `deploy --check` on 2026-01-14; exit 0."}
    assert gate("procedural", meta=evidenced).approved


def test_evidence_true_is_not_evidence(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """The point of item 6 is a citation, not a self-reported boolean."""
    assert checks(gate("procedural", meta={**meta, "evidence": True})) == {"evidence_missing"}


def test_semantic_does_not_require_evidence(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    assert gate("semantic", meta=meta).approved


def test_research_write_is_rejected_when_the_deployment_disables_it(
    gate: Callable[..., GateResult], meta: dict[str, Any], manifest_full: Manifest
) -> None:
    """`manifest_full` sets `zones.research_enabled: false` (§3)."""
    research_meta = {**meta, "profile": "planner", "scope": "proj-a"}
    result = gate("research", meta=research_meta, manifest=manifest_full)
    assert checks(result) == {"type_disabled"}


def test_every_broken_check_is_reported_at_once(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """One round-trip should tell the agent everything, not the first thing."""
    result = gate(
        "procedural",
        meta={**meta, "source": "", "agent": "ghost", "forbidden-key": 1},
    )
    assert checks(result) == {
        "agent_unknown",
        "denylist_keys",
        "source_missing",
        "evidence_missing",
    }


def test_an_unknown_type_is_a_wiring_error(gate: Callable[..., GateResult]) -> None:
    with pytest.raises(ValueError, match="unknown memory type"):
        gate("semantical")


# ══════════════════════════════════════════════════════════════════════
# Items 1, 3, 4b -- instructed, never enforced
# ══════════════════════════════════════════════════════════════════════
def test_reusability_is_returned_as_the_authors_call_every_time(
    gate: Callable[..., GateResult]
) -> None:
    """Item 1 has no textual signal, so the gate hands the obligation back."""
    result = gate()
    assert "reusable" in warned(result)
    assert result.approved, "a warning is not a veto"


def test_a_long_body_warns_but_never_blocks(
    gate: Callable[..., GateResult]
) -> None:
    result = gate(body="word " * (ATOMIC_BODY_CHARS // 2))
    assert "atomic" in warned(result)
    assert result.approved


def test_many_sibling_sections_warn_about_atomicity(
    gate: Callable[..., GateResult]
) -> None:
    sections = "".join(f"## Concept {i}\n\nText.\n\n" for i in range(ATOMIC_MAX_SECTIONS + 1))
    assert "atomic" in warned(gate(body=sections))


def test_a_structured_playbook_does_not_trip_the_atomicity_heuristic(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """Steps + Evidence is the shape §8.2 item 6 asks for; it is one procedure."""
    playbook = "## Steps\n\n1. Build.\n2. Ship.\n\n## Evidence\n\nRan on 2026-01-14.\n"
    result = gate("procedural", meta={**meta, "evidence": "ran it"}, body=playbook)
    assert "atomic" not in warned(result)


def test_subsections_do_not_count_as_separate_insights(
    gate: Callable[..., GateResult]
) -> None:
    nested = "# One idea\n\n## Detail a\n\n## Detail b\n\n## Detail c\n"
    assert "atomic" not in warned(gate(body=nested))


def test_a_comment_in_fenced_code_is_not_a_heading(
    gate: Callable[..., GateResult]
) -> None:
    code = "```bash\n# step one\n# step two\n# step three\n```\n"
    assert "atomic" not in warned(gate(body=code))


def test_high_confidence_without_evidence_warns(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    result = gate(meta={**meta, "confidence": "high"})
    assert "confidence_unverified" in warned(result)
    assert result.approved, "primacy of a source is a judgement, not a rejection"


def test_high_confidence_with_evidence_does_not_warn(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    verified = {**meta, "confidence": "high", "evidence": "Reproduced the benchmark."}
    assert "confidence_unverified" not in warned(gate(meta=verified))


def test_medium_confidence_does_not_warn(gate: Callable[..., GateResult]) -> None:
    assert "confidence_unverified" not in warned(gate())


# ══════════════════════════════════════════════════════════════════════
# gate.reject_on -- opt-in elevation of the one mechanical warning
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def strict_manifest(manifest_data_ci: dict[str, Any]) -> Manifest:
    """`manifest_ci` with `confidence_unverified` elevated to a rejection."""
    data = copy.deepcopy(manifest_data_ci)
    data["gate"] = {"reject_on": ["confidence_unverified"]}
    return Manifest.from_mapping(data)


def test_reject_on_elevates_unverified_high_confidence(
    gate: Callable[..., GateResult], meta: dict[str, Any], strict_manifest: Manifest
) -> None:
    result = gate(meta={**meta, "confidence": "high"}, manifest=strict_manifest)
    assert not result.approved
    assert "confidence_unverified" in checks(result)
    assert "confidence_unverified" not in warned(result), (
        "an elevated finding must not be reported twice"
    )


def test_elevation_spares_a_verified_high(
    gate: Callable[..., GateResult], meta: dict[str, Any], strict_manifest: Manifest
) -> None:
    verified = {**meta, "confidence": "high", "evidence": "Reproduced the benchmark."}
    result = gate(meta=verified, manifest=strict_manifest)
    assert result.approved
    assert "confidence_unverified" not in checks(result)


def test_elevation_leaves_judgement_warnings_alone(
    gate: Callable[..., GateResult], meta: dict[str, Any], strict_manifest: Manifest
) -> None:
    """Only the named check moves; `reusable`/`atomic` stay the author's call."""
    long_body = "One idea, told at length. " * 400
    result = gate(
        meta={**meta, "confidence": "high"}, body=long_body, manifest=strict_manifest
    )
    assert "confidence_unverified" in checks(result)
    assert "reusable" in warned(result)
    assert "atomic" in warned(result)


def test_default_manifest_never_elevates(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    """The baseline stays honest: with no `gate.reject_on`, warnings never block."""
    result = gate(meta={**meta, "confidence": "high"})
    assert result.approved
    assert "confidence_unverified" in warned(result)


# ══════════════════════════════════════════════════════════════════════
# Routing (§8.1) -- who faces the checklist
# ══════════════════════════════════════════════════════════════════════
def test_the_full_checklist_covers_exactly_semantic_and_procedural() -> None:
    """§8.2 is titled "for Semantic/ and Procedural/"."""
    assert {t.value for t in FULL_GATE_TYPES} == {"semantic", "procedural"}


def test_episodic_needs_no_source_and_no_search(meta: dict[str, Any], manifest_ci: Manifest) -> None:
    """§8.1 item 3 -- a diary is written directly: it is not truth, it is a record."""
    diary = {k: v for k, v in meta.items() if k != "source"}
    result = evaluate("episodic", diary, "Ran the pipeline; it worked.", manifest_ci)
    assert result.approved
    assert result.warnings == (), "no judgement items apply to a diary entry"


def test_episodic_is_never_deduped(meta: dict[str, Any], manifest_ci: Manifest) -> None:
    """The same thing happening twice is two events, not a duplicate."""
    search = FakeSearch([hit("[[2026-01-15_MEM_session-one]]", 0.99)])
    result = evaluate("episodic", meta, "Ran it again.", manifest_ci, search_fn=search)
    assert result.approved
    assert search.calls == [], "the gate must not even ask"


def test_episodic_still_obeys_the_denylist(meta: dict[str, Any], manifest_ci: Manifest) -> None:
    """R6.1 is a §6 schema rule: it binds every note, gated or not."""
    result = evaluate("episodic", {**meta, "forbidden-key": 1}, "Body.", manifest_ci)
    assert checks(result) == {"denylist_keys"}


def test_episodic_still_needs_a_known_agent_and_scope(
    meta: dict[str, Any], manifest_ci: Manifest
) -> None:
    result = evaluate("episodic", {**meta, "agent": "ghost", "scope": "nope"}, "B.", manifest_ci)
    assert checks(result) == {"agent_unknown", "scope_unknown"}


def test_research_is_not_asked_to_be_atomic(meta: dict[str, Any], manifest_ci: Manifest) -> None:
    """A report is long and multi-section by construction; §8.2 never claimed it."""
    report = "".join(f"## Section {i}\n\nText.\n\n" for i in range(6))
    result = evaluate("research", meta, report, manifest_ci)
    assert result.approved
    assert result.warnings == ()


# ══════════════════════════════════════════════════════════════════════
# Result plumbing
# ══════════════════════════════════════════════════════════════════════
def test_rejection_raises_with_the_near_duplicates_attached(
    gate: Callable[..., GateResult], manifest_ci: Manifest
) -> None:
    """§7.2 -- `memory_write` returns `{rejected, reason, near_duplicates[]}`."""
    above = manifest_ci.maintenance.dedup_similarity + 0.05
    search = FakeSearch([hit("[[2026-01-15_INS_vector-search]]", above)])
    result = gate(search_fn=search)

    with pytest.raises(GateRejected) as excinfo:
        result.raise_if_rejected()
    assert excinfo.value.near_duplicates == ["[[2026-01-15_INS_vector-search]]"]
    assert "§8.2 item 2" in str(excinfo.value)


def test_reason_lists_every_rejection(
    gate: Callable[..., GateResult], meta: dict[str, Any]
) -> None:
    reason = gate(meta={**meta, "source": "", "agent": "ghost"}).reason
    assert "ghost" in reason
    assert "source" in reason
    assert len(reason.splitlines()) > 2, "one line per rejection, plus hints"


def test_reason_is_empty_when_approved(gate: Callable[..., GateResult]) -> None:
    assert gate().reason == ""


def test_a_hit_object_works_as_well_as_a_mapping(
    gate: Callable[..., GateResult], manifest_ci: Manifest
) -> None:
    """`search_fn` is injected, so the gate must not assume the hit's class."""

    class Hit:
        wikilink = "[[x]]"
        id = "x"
        path = "mecha-brain/Semantic/x.md"
        title = "X"
        similarity = 0.99

    result = gate(search_fn=FakeSearch([Hit()]))  # type: ignore[list-item]
    assert not result.approved
    assert result.near_duplicates[0].wikilink == "[[x]]"


def test_similarity_wins_over_a_normalised_rank_score() -> None:
    """A fused `score` is min-max normalised: the top hit is 1.0 whatever it is."""
    candidate = NearDuplicate.from_hit({"wikilink": "[[x]]", "similarity": 0.30, "score": 1.0})
    assert candidate.similarity == pytest.approx(0.30)


def test_score_is_the_fallback_when_no_similarity_is_reported() -> None:
    candidate = NearDuplicate.from_hit({"wikilink": "[[x]]", "score": 0.42})
    assert candidate.similarity == pytest.approx(0.42)


def test_a_hit_without_any_similarity_is_a_broken_search() -> None:
    with pytest.raises(ValueError, match="no similarity"):
        NearDuplicate.from_hit({"wikilink": "[[x]]"})


def test_a_hit_without_provenance_is_a_broken_search() -> None:
    """R7.1 -- every hit carries a path and a wikilink."""
    with pytest.raises(ValueError, match="provenance"):
        NearDuplicate.from_hit({"similarity": 0.99})


def test_a_wikilink_is_derived_from_a_path_when_absent() -> None:
    candidate = NearDuplicate.from_hit(
        {"path": "mecha-brain/Semantic/2026-01-15_INS_x.md", "similarity": 0.99}
    )
    assert candidate.wikilink == "[[2026-01-15_INS_x]]"
    assert candidate.id == "2026-01-15_INS_x"


def test_the_gate_never_touches_the_callers_meta(
    manifest_ci: Manifest, meta: dict[str, Any], no_hits: FakeSearch
) -> None:
    before = copy.deepcopy(meta)
    evaluate("semantic", meta, "Body.", manifest_ci, search_fn=no_hits)
    assert meta == before
