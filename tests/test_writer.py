"""Write execution (§7.2): the gate decides, the writer executes.

The suite pins the seam between the two -- a gate rejection writes nothing, an
approval lands a §6 note at the right contractual path -- and the parts the gate
does not own: name resolution from the manifest template, collision avoidance,
evidence rendering, supersede archival (P8/R6.3), and the propose flow that never
touches its target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mechabrain.contract import STATUS_ACTIVE, STATUS_ARCHIVED
from mechabrain.discovery import VaultPaths
from mechabrain.errors import DenylistViolation, ManifestError
from mechabrain.manifest import Manifest
from mechabrain.note import Note
from mechabrain.writer import (
    EVIDENCE_HEADING,
    Indexer,
    ProposalResult,
    WriteResult,
    propose,
    slugify,
    write,
)

STAMP = date(2026, 7, 17)


# ══════════════════════════════════════════════════════════════════════
# Test doubles
# ══════════════════════════════════════════════════════════════════════
class FakeSearch:
    """A `SearchFn` returning a fixed hit list, recording every call (like test_gate)."""

    def __init__(self, hits: Sequence[Mapping[str, Any]] = ()) -> None:
        self.hits = list(hits)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, query: str, *, k: int = 8, filters: Mapping[str, Any] | None = None
    ) -> Sequence[Mapping[str, Any]]:
        self.calls.append({"query": query, "k": k, "filters": dict(filters or {})})
        return list(self.hits)


class RecordingIndexer:
    """An `Indexer` that remembers which notes it was asked to index."""

    def __init__(self) -> None:
        self.indexed: list[tuple[str, str]] = []

    def index_note(self, note: Note) -> None:
        self.indexed.append((note.note_id, str(note.get("status") or "")))


def hit(wikilink: str, similarity: float, **extra: Any) -> dict[str, Any]:
    note_id = wikilink.strip("[]")
    return {
        "wikilink": wikilink,
        "id": note_id,
        "similarity": similarity,
        "title": extra.pop("title", "A neighbour"),
        "path": extra.pop("path", f"mecha-brain/Semantic/{note_id}.md"),
        **extra,
    }


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def no_hits() -> FakeSearch:
    return FakeSearch()


@pytest.fixture
def semantic_meta() -> dict[str, Any]:
    """A frontmatter that clears the full §8.2 gate for a semantic write."""
    return {
        "title": "Brute-force cosine is fast below 10k chunks",
        "agent": "alpha",
        "profile": "tutor",
        "scope": "proj-a",
        "source": "test-session",
        "confidence": "medium",
    }


def do_write(
    memory_type: str,
    meta: Mapping[str, Any],
    manifest: Manifest,
    vault: VaultPaths,
    *,
    content: str = "Brute-force cosine is fast enough below ten thousand chunks.",
    search_fn: Any = None,
    indexer: Indexer | None = None,
) -> WriteResult:
    """`write` with the deterministic stamp and no lock churn between tests."""
    return write(
        memory_type,
        content,
        meta,
        manifest,
        vault,
        search_fn if search_fn is not None else FakeSearch(),
        indexer=indexer,
        today=STAMP,
    )


# ══════════════════════════════════════════════════════════════════════
# slugify
# ══════════════════════════════════════════════════════════════════════
class TestSlugify:
    def test_transliterates_accents(self) -> None:
        assert slugify("Configuração de Rede") == "configuracao-de-rede"

    def test_collapses_punctuation_runs(self) -> None:
        assert slugify("A/B  test — v2!!") == "a-b-test-v2"

    def test_falls_back_when_nothing_remains(self) -> None:
        assert slugify("—!!—") == "nota"
        assert slugify("") == "nota"


# ══════════════════════════════════════════════════════════════════════
# Gate integration: rejection writes nothing
# ══════════════════════════════════════════════════════════════════════
class TestGateRejection:
    def test_rejected_write_touches_no_disk(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        semantic_meta.pop("source")  # §8.2 item 4: missing source -> reject
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)

        assert result.rejected is True
        assert result.ok is False
        assert result.path is None
        assert "source" in result.reason
        assert list(tmp_vault.semantic_dir.glob("*.md")) == []

    def test_rejection_surfaces_near_duplicates(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        search = FakeSearch([hit("[[2026-01-15_INS_twin]]", 0.97)])
        result = do_write(
            "semantic", semantic_meta, manifest_ci, tmp_vault, search_fn=search
        )

        assert result.rejected is True
        assert [d.wikilink for d in result.near_duplicates] == ["[[2026-01-15_INS_twin]]"]
        assert list(tmp_vault.semantic_dir.glob("*.md")) == []

    def test_research_disabled_is_rejected_by_gate(self, tmp_vault: VaultPaths) -> None:
        # A manifest with Research turned off; the gate refuses the type (§3).
        disabled = Manifest.from_mapping(_ci_mapping(research_enabled=False))
        result = write(
            "research",
            "A long report body.",
            {"title": "Report", "agent": "alpha", "scope": "proj-a", "source": "s"},
            disabled,
            tmp_vault,
            FakeSearch(),
            today=STAMP,
        )
        assert result.rejected is True
        assert "research" in result.reason
        assert list(tmp_vault.research_dir.glob("*.md")) == []


# ══════════════════════════════════════════════════════════════════════
# Successful semantic write
# ══════════════════════════════════════════════════════════════════════
class TestSemanticWrite:
    def test_lands_dated_note_in_semantic_folder(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)

        assert result.ok is True
        assert result.path is not None
        assert result.path.parent == tmp_vault.semantic_dir
        assert result.path.name == "2026-07-17_INS_brute-force-cosine-is-fast-below-10k-chunks.md"
        assert result.note_id == result.path.stem
        assert result.wikilink == f"[[{result.path.stem}]]"
        assert result.path.is_file()

    def test_frontmatter_follows_section_6(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)
        note = Note.load(result.path)  # type: ignore[arg-type]
        fm = note.frontmatter

        assert fm["title"] == semantic_meta["title"]
        assert fm["tags"] == ["mem/semantic", "agent/alpha"]
        assert fm["created"] == STAMP
        assert fm["modified"] == STAMP
        assert fm["last_accessed"] == STAMP
        assert fm["agent"] == "alpha"
        assert fm["profile"] == "tutor"
        assert fm["scope"] == "proj-a"
        assert fm["source"] == "test-session"
        assert fm["confidence"] == "medium"
        assert fm["status"] == STATUS_ACTIVE
        assert "supersedes" not in fm
        # §6 key order is preserved on disk.
        assert list(fm)[:5] == ["title", "tags", "created", "modified", "agent"]

    def test_required_extra_and_author_tags_merge(
        self, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        manifest = Manifest.from_mapping(_ci_mapping(required_extra_tags=["source/ai"]))
        semantic_meta["tags"] = ["topic/retrieval", "source/ai"]
        result = do_write("semantic", semantic_meta, manifest, tmp_vault)
        fm = Note.load(result.path).frontmatter  # type: ignore[arg-type]

        # generated first, extras appended, deduplicated, order preserved.
        assert fm["tags"] == ["mem/semantic", "agent/alpha", "source/ai", "topic/retrieval"]

    def test_gate_only_keys_never_reach_frontmatter(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        semantic_meta["merge"] = True
        semantic_meta["evidence"] = "should not appear on a semantic note"
        semantic_meta["stray"] = "nope"
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)
        note = Note.load(result.path)  # type: ignore[arg-type]

        assert "merge" not in note.frontmatter
        assert "evidence" not in note.frontmatter
        assert "stray" not in note.frontmatter
        assert "evidence" not in note.body

    def test_body_is_the_content(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        result = do_write(
            "semantic", semantic_meta, manifest_ci, tmp_vault, content="The claim.\n"
        )
        # Bodies round-trip with the blank line the frontmatter fence leaves.
        assert Note.load(result.path).body.strip() == "The claim."  # type: ignore[arg-type]

    def test_confidence_defaults_to_medium(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        semantic_meta.pop("confidence")
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)
        assert Note.load(result.path).get("confidence") == "medium"  # type: ignore[arg-type]

    def test_warnings_pass_through_on_success(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)
        # §8.2 item 1 (reusability) always warns; it never blocks.
        assert any(w.check == "reusable" for w in result.warnings)
        assert result.ok is True


# ══════════════════════════════════════════════════════════════════════
# Naming: dated vs atemporal, collisions
# ══════════════════════════════════════════════════════════════════════
class TestNaming:
    def test_procedural_is_atemporal(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        meta = {
            "title": "Deploy playbook",
            "agent": "beta",
            "scope": "proj-a",
            "source": "run-2026-07-16",
            "evidence": "Executed on 2026-07-16.",
        }
        result = do_write(
            "procedural", meta, manifest_ci, tmp_vault, content="## Steps\n\n1. Build."
        )
        # No date, no leading separator -- the {date}_ prefix collapses cleanly.
        assert result.path.name == "PROC_deploy-playbook.md"  # type: ignore[union-attr]
        assert result.path.parent == tmp_vault.procedural_dir  # type: ignore[union-attr]

    def test_collision_never_overwrites(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        first = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)
        second = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)

        assert first.path != second.path
        assert second.path.name.endswith("-2.md")  # type: ignore[union-attr]
        assert first.path.is_file()  # type: ignore[union-attr]
        assert second.path.is_file()  # type: ignore[union-attr]

    def test_atemporal_collision_bumps_ordinal(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        meta = {
            "title": "Deploy playbook",
            "agent": "beta",
            "scope": "proj-a",
            "source": "run",
            "evidence": "ran",
        }
        first = do_write("procedural", meta, manifest_ci, tmp_vault, content="x")
        second = do_write("procedural", meta, manifest_ci, tmp_vault, content="y")
        assert first.path.name == "PROC_deploy-playbook.md"  # type: ignore[union-attr]
        assert second.path.name == "PROC_deploy-playbook-2.md"  # type: ignore[union-attr]


# ══════════════════════════════════════════════════════════════════════
# Episodic: direct write, append-only, own subfolder
# ══════════════════════════════════════════════════════════════════════
class TestEpisodic:
    def test_lands_in_agent_subfolder(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        meta = {"title": "Session one", "agent": "alpha", "scope": "proj-a", "source": "s"}
        result = do_write("episodic", meta, manifest_ci, tmp_vault, content="Ran it.")

        assert result.path.parent == tmp_vault.episodic_for("alpha")  # type: ignore[union-attr]
        assert result.path.name.startswith("2026-07-17_MEM_")  # type: ignore[union-attr]

    def test_no_source_still_writes(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        # Episodic is a diary, not truth: the gate does not demand a source here.
        meta = {"title": "Session two", "agent": "alpha", "scope": "proj-a"}
        result = do_write("episodic", meta, manifest_ci, tmp_vault, content="Happened.")
        assert result.ok is True
        assert Note.load(result.path).get("source") == ""  # type: ignore[arg-type]

    def test_episodic_write_takes_no_search(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        # No dedup for episodic, so search_fn may be absent.
        meta = {"title": "Session three", "agent": "beta", "scope": "proj-a"}
        result = write(
            "episodic", "Body.", meta, manifest_ci, tmp_vault, None, today=STAMP
        )
        assert result.ok is True


# ══════════════════════════════════════════════════════════════════════
# Procedural evidence rendering (§8.2 item 6)
# ══════════════════════════════════════════════════════════════════════
class TestEvidence:
    def _meta(self, evidence: Any) -> dict[str, Any]:
        return {
            "title": "Ship it",
            "agent": "beta",
            "scope": "proj-a",
            "source": "run",
            "evidence": evidence,
        }

    def test_string_evidence_renders_a_section(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        result = do_write(
            "procedural",
            self._meta("Ran `deploy.sh` on 2026-07-16, exit 0."),
            manifest_ci,
            tmp_vault,
            content="## Steps\n\n1. Build.\n2. Ship.",
        )
        body = Note.load(result.path).body.strip()  # type: ignore[arg-type]
        assert body.endswith(
            f"{EVIDENCE_HEADING}\n\nRan `deploy.sh` on 2026-07-16, exit 0."
        )
        assert body.startswith("## Steps")

    def test_list_evidence_renders_bullets(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        result = do_write(
            "procedural",
            self._meta(["ran on 2026-07-15", "ran again on 2026-07-16"]),
            manifest_ci,
            tmp_vault,
            content="Do the thing.",
        )
        body = Note.load(result.path).body  # type: ignore[arg-type]
        assert f"{EVIDENCE_HEADING}\n\n- ran on 2026-07-15\n- ran again on 2026-07-16" in body


# ══════════════════════════════════════════════════════════════════════
# Supersedes: archive, never delete (P8); episodic untouched (R6.3)
# ══════════════════════════════════════════════════════════════════════
class TestSupersedes:
    def _seed_semantic(self, vault: VaultPaths, note_id: str) -> Path:
        path = vault.semantic_dir / f"{note_id}.md"
        Note(
            path=path,
            frontmatter={
                "title": "Old fact",
                "tags": ["mem/semantic", "agent/alpha"],
                "scope": "proj-a",
                "status": STATUS_ACTIVE,
            },
            body="An older claim.",
        ).write()
        return path

    def test_superseded_semantic_is_archived_not_deleted(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        old = self._seed_semantic(tmp_vault, "2026-01-15_INS_old-fact")
        # A near-duplicate is present, but supersedes is the explicit decision.
        search = FakeSearch([hit("[[2026-01-15_INS_old-fact]]", 0.99)])
        semantic_meta["supersedes"] = "[[2026-01-15_INS_old-fact]]"

        result = do_write(
            "semantic", semantic_meta, manifest_ci, tmp_vault, search_fn=search
        )

        assert result.ok is True
        assert old.is_file()  # P8: archived, never deleted.
        assert Note.load(old).get("status") == STATUS_ARCHIVED
        assert result.superseded == ("[[2026-01-15_INS_old-fact]]",)
        # The new note records the supersede edge.
        assert Note.load(result.path).get("supersedes") == "[[2026-01-15_INS_old-fact]]"  # type: ignore[arg-type]

    def test_missing_target_is_reported_not_fatal(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        # Under eventual consistency (R7.6) the target may not exist locally.
        semantic_meta["supersedes"] = "[[2026-01-15_INS_absent]]"
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault)

        assert result.ok is True
        assert result.superseded == ()
        assert result.superseded_missing == ("[[2026-01-15_INS_absent]]",)

    def test_episodic_target_is_left_untouched(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        old_path = tmp_vault.episodic_for("alpha") / "2026-01-15_MEM_session.md"
        Note(
            path=old_path,
            frontmatter={"title": "Session", "scope": "proj-a", "status": STATUS_ACTIVE},
            body="What happened.",
        ).write()

        meta = {
            "title": "Session, corrected",
            "agent": "alpha",
            "scope": "proj-a",
            "source": "s",
            "supersedes": "[[2026-01-15_MEM_session]]",
        }
        result = do_write("episodic", meta, manifest_ci, tmp_vault, content="Corrected.")

        assert result.ok is True
        # R6.3: append-only -- the old episodic note keeps its active status.
        assert Note.load(old_path).get("status") == STATUS_ACTIVE
        assert result.superseded_episodic == ("[[2026-01-15_MEM_session]]",)
        assert result.superseded == ()


# ══════════════════════════════════════════════════════════════════════
# Indexing hook (§7.2 step 8)
# ══════════════════════════════════════════════════════════════════════
class TestIndexing:
    def test_new_note_is_indexed(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        indexer = RecordingIndexer()
        result = do_write(
            "semantic", semantic_meta, manifest_ci, tmp_vault, indexer=indexer
        )
        assert result.indexed is True
        assert (result.note_id, STATUS_ACTIVE) in indexer.indexed

    def test_archived_note_is_reindexed(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        indexer = RecordingIndexer()
        TestSupersedes()._seed_semantic(tmp_vault, "2026-01-15_INS_old-fact")
        semantic_meta["supersedes"] = "[[2026-01-15_INS_old-fact]]"
        do_write("semantic", semantic_meta, manifest_ci, tmp_vault, indexer=indexer)

        assert ("2026-01-15_INS_old-fact", STATUS_ARCHIVED) in indexer.indexed

    def test_absent_indexer_still_writes(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        result = do_write("semantic", semantic_meta, manifest_ci, tmp_vault, indexer=None)
        assert result.ok is True
        assert result.indexed is False


# ══════════════════════════════════════════════════════════════════════
# propose (§7.2): writes to the inbox, never touches the target
# ══════════════════════════════════════════════════════════════════════
class TestPropose:
    def _meta(self) -> dict[str, Any]:
        return {"agent": "alpha", "profile": "tutor", "scope": "proj-a", "source": "review"}

    def test_writes_into_proposals_dir_and_never_touches_target(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        target = tmp_vault.root / "Notes" / "Human Note.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original human content\n", encoding="utf-8")
        before = target.read_text(encoding="utf-8")

        result = propose(
            target,
            "Change section 2 to say X.",
            "It is currently inaccurate.",
            self._meta(),
            manifest_ci,
            tmp_vault,
            today=STAMP,
        )

        assert isinstance(result, ProposalResult)
        assert result.path.parent == tmp_vault.inbox_dir
        assert result.path.name == "2026-07-17_AI-PROPOSAL_human-note.md"
        # The target is untouched -- byte for byte.
        assert target.read_text(encoding="utf-8") == before

    def test_proposal_frontmatter_and_body(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        result = propose(
            "Notes/Foo.md",
            "Replace the intro.",
            "The intro is stale.",
            self._meta(),
            manifest_ci,
            tmp_vault,
            today=STAMP,
        )
        note = Note.load(result.path)

        assert note.get("agent") == "alpha"
        assert note.get("profile") == "tutor"
        assert note.get("target") == "Notes/Foo.md"
        assert note.get("status") == STATUS_ACTIVE
        assert "mem/semantic" not in note.tags  # a proposal is not a memory type
        assert "agent/alpha" in note.tags
        assert "The intro is stale." in note.body
        assert "Replace the intro." in note.body
        assert "[[Foo]]" in note.body

    def test_unknown_author_is_refused(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        meta = {"agent": "ghost", "scope": "proj-a"}
        with pytest.raises(ManifestError):
            propose("Notes/Foo.md", "c", "r", meta, manifest_ci, tmp_vault, today=STAMP)

    def test_proposal_collision_bumps(
        self, manifest_ci: Manifest, tmp_vault: VaultPaths
    ) -> None:
        first = propose("Notes/Foo.md", "c", "r", self._meta(), manifest_ci, tmp_vault, today=STAMP)
        second = propose("Notes/Foo.md", "c", "r", self._meta(), manifest_ci, tmp_vault, today=STAMP)
        assert first.path != second.path
        assert second.path.name.endswith("-2.md")


# ══════════════════════════════════════════════════════════════════════
# Denylist enforcement on assembled notes (R6.1)
# ══════════════════════════════════════════════════════════════════════
class TestDenylist:
    def test_write_rejects_denied_author_tag_via_gate(
        self, tmp_vault: VaultPaths, semantic_meta: dict[str, Any]
    ) -> None:
        manifest = Manifest.from_mapping(_ci_mapping(denylist_tags=["forbidden/tag"]))
        semantic_meta["tags"] = ["forbidden/tag"]
        result = do_write("semantic", semantic_meta, manifest, tmp_vault)
        assert result.rejected is True
        assert "forbidden/tag" in result.reason

    def test_propose_rejects_denied_key(
        self, tmp_vault: VaultPaths
    ) -> None:
        # A deployment that reserves `target` for its own automation collides
        # with the proposal's `target:` pointer -- the writer must fail loud.
        manifest = Manifest.from_mapping(_ci_mapping(denylist_keys=["target"]))
        with pytest.raises(DenylistViolation):
            propose(
                "Notes/Foo.md",
                "c",
                "r",
                {"agent": "alpha", "scope": "proj-a"},
                manifest,
                tmp_vault,
                today=STAMP,
            )


# ══════════════════════════════════════════════════════════════════════
# Helpers to build manifests for the negative cases
# ══════════════════════════════════════════════════════════════════════
def _ci_mapping(
    *,
    research_enabled: bool = True,
    required_extra_tags: Sequence[str] = (),
    denylist_tags: Sequence[str] = (),
    denylist_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """The CI manifest mapping, with a few knobs the negative tests turn."""
    return {
        "mecha_brain": {"spec_version": "0.1", "kernel_min_version": "0.1.0"},
        "agents": [
            {"id": "alpha", "display_name": "Alpha", "profiles": ["tutor", "planner"]},
            {"id": "beta", "display_name": "Beta"},
        ],
        "scopes": {"known": ["proj-a", "proj-b", "global"], "default": "global"},
        "naming": {
            "note_name": "{date}_{prefix}_{slug}.md",
            "dated_types": ["episodic", "semantic", "research"],
            "prefixes": {
                "episodic": "MEM",
                "semantic": "INS",
                "procedural": "PROC",
                "research": "RES",
            },
            "proposal_name": "{date}_AI-PROPOSAL_{slug}.md",
        },
        "zones": {
            "proposals_dir": "mecha-brain/_inbox/",
            "read_only_index": [],
            "research_enabled": research_enabled,
        },
        "frontmatter": {
            "denylist_keys": list(denylist_keys),
            "denylist_tags": list(denylist_tags),
            "tag_namespaces": {"memory": "mem", "agent": "agent"},
            "required_extra_tags": list(required_extra_tags),
        },
        "retrieval": {
            "embedding": {"provider": "hash", "model": "hash-256"},
            "hybrid": {"vector_weight": 0.6, "bm25_weight": 0.4},
            "contextual_retrieval": True,
            "rerank": False,
            "link_expansion": {"default_hops": 1, "max_hops": 2},
            "store": "numpy",
        },
        "maintenance": {
            "decay_days": 90,
            "dedup_similarity": 0.92,
            "commit_prefix": "chore(ai-memory):",
        },
    }
