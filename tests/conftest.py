"""Shared fixtures for the whole kernel test suite.

Every module's tests build on these -- prefer extending a fixture here over
hand-rolling a vault in your own test file, so that a change to the §3 skeleton
lands in one place.

Quick map:

======================  =====================================================
Fixture                 What you get
======================  =====================================================
``tmp_vault``           :class:`VaultPaths` of a disposable, initialized vault
                        (full §3 skeleton + a valid ``config.yaml``). Empty of
                        notes -- add ``sample_notes`` if you want content.
``sample_notes``        Six notes written into ``tmp_vault``, spanning all four
                        memory types, two scopes, two agents, and one archived.
``manifest_ci``         The parsed :class:`Manifest` that ``tmp_vault`` holds:
                        ``embedding.provider: hash`` + ``store: numpy``, so the
                        suite is deterministic and needs no model download.
``manifest_min``        Parsed manifest with only the required section set --
                        everything else is the §5 default. Use it to assert
                        defaults.
``manifest_full``       Parsed manifest with **every** §5 key set to a
                        non-default value. Use it to assert that a key is
                        actually read rather than defaulted.
``*_data`` variants     The raw ``dict`` behind each of the above. Mutate a copy
                        and re-parse to test one bad key at a time.
``make_vault``          Factory: ``make_vault(manifest_data=..., name=...)`` ->
                        ``VaultPaths``, for tests needing a second vault or a
                        custom manifest on disk.
``write_note``          Factory: ``write_note(path, frontmatter, body)`` -> Note.
``clean_env``           *autouse* -- unsets ``MECHABRAIN_VAULT`` so a developer's
                        real vault can never leak into a discovery test.
======================  =====================================================

Conventions the fixtures follow, mirroring the spec: agent ids are ``alpha`` and
``beta``, scopes are ``proj-a``, ``proj-b`` and ``global``. They are deliberately
meaningless -- the kernel knows no agent or vault by name (R4.1), and neither
should its tests.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from mechabrain.discovery import VAULT_ENV_VAR, VaultPaths
from mechabrain.errors import MechabrainError
from mechabrain.generate import write_agents_md, write_schema
from mechabrain.manifest import Manifest
from mechabrain.note import Note, write_atomic

# ══════════════════════════════════════════════════════════════════════
# Environment hygiene
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ``MECHABRAIN_VAULT`` for every test (autouse).

    R4.3 makes the env var outrank the upward walk, so a developer who exports
    it would otherwise see discovery tests pass against their real vault.
    """
    monkeypatch.delenv(VAULT_ENV_VAR, raising=False)


# ══════════════════════════════════════════════════════════════════════
# Manifest data
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def manifest_data_min() -> dict[str, Any]:
    """Smallest manifest §5 accepts: the contract handshake and nothing else."""
    return {
        "mecha_brain": {"spec_version": "0.1", "kernel_min_version": "0.1.0"},
    }


@pytest.fixture
def manifest_data_ci() -> dict[str, Any]:
    """The manifest ``tmp_vault`` installs: deterministic and offline.

    ``embedding.provider: hash`` and ``store: numpy`` are what make the suite
    runnable in CI with no model download and no heavy dependency. Never use
    this pair in a real deployment -- hash embeddings carry no semantics.
    """
    return {
        "mecha_brain": {"spec_version": "0.1", "kernel_min_version": "0.1.0"},
        "agents": [
            {
                "id": "alpha",
                "display_name": "Alpha",
                "profiles": ["tutor", "planner"],
                "private_store": "none",
            },
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
            "research_enabled": True,
        },
        "frontmatter": {
            "denylist_keys": ["forbidden-key"],
            "denylist_tags": ["forbidden/tag"],
            "tag_namespaces": {"memory": "mem", "agent": "agent"},
            "required_extra_tags": [],
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


@pytest.fixture
def manifest_data_full() -> dict[str, Any]:
    """Every §5 key set to a **non-default** value.

    The point is negative: if a test asserts a value here and the parser is
    silently defaulting the key, the assertion fails. Keep every value different
    from the spec default.
    """
    return {
        "mecha_brain": {"spec_version": "0.1", "kernel_min_version": "0.1.0"},
        "agents": [
            {
                "id": "alpha",
                "display_name": "Agent Alpha",
                "profiles": ["tutor", "planner"],
                "private_store": "an external behavioural store",
            },
            {
                "id": "beta",
                "display_name": "Agent Beta",
                "profiles": ["researcher"],
                "private_store": {"researcher": "per-profile store"},
            },
        ],
        "scopes": {"known": ["proj-a", "proj-b", "global"], "default": "proj-a"},
        "naming": {
            "note_name": "{date}-{prefix}-{slug}.md",
            "dated_types": ["episodic", "semantic"],
            "prefixes": {
                "episodic": "EP",
                "semantic": "SEM",
                "procedural": "HOW",
                "research": "REP",
            },
            "proposal_name": "{date}_PROPOSAL_{slug}.md",
        },
        "zones": {
            "proposals_dir": "Inbox/",
            "read_only_index": ["Notes/", "Reference/"],
            "research_enabled": False,
        },
        "frontmatter": {
            "denylist_keys": ["publish", "internal-id"],
            "denylist_tags": ["automation/trigger"],
            "tag_namespaces": {"memory": "memory", "agent": "author"},
            "required_extra_tags": ["source/ai"],
        },
        "gate": {"reject_on": ["confidence_unverified"]},
        "retrieval": {
            "embedding": {"provider": "http", "model": "custom-model"},
            "hybrid": {"vector_weight": 0.75, "bm25_weight": 0.25},
            "contextual_retrieval": False,
            # `rerank` is the one key that cannot be non-default here: `true` is
            # rejected until a reranker ships (see the dedicated manifest test).
            "rerank": False,
            "link_expansion": {"default_hops": 2, "max_hops": 3},
            "store": "lancedb",
        },
        "maintenance": {
            "decay_days": 30,
            "dedup_similarity": 0.85,
            "commit_prefix": "chore(memory):",
            "proc_stale_days": 60,
        },
    }


@pytest.fixture
def manifest_min(manifest_data_min: dict[str, Any]) -> Manifest:
    """Parsed :fixture:`manifest_data_min` -- everything at its §5 default."""
    return Manifest.from_mapping(manifest_data_min)


@pytest.fixture
def manifest_ci(manifest_data_ci: dict[str, Any]) -> Manifest:
    """Parsed :fixture:`manifest_data_ci` -- hash embeddings, numpy store."""
    return Manifest.from_mapping(manifest_data_ci)


@pytest.fixture
def manifest_full(manifest_data_full: dict[str, Any]) -> Manifest:
    """Parsed :fixture:`manifest_data_full` -- every key non-default."""
    return Manifest.from_mapping(manifest_data_full)


# ══════════════════════════════════════════════════════════════════════
# Vaults
# ══════════════════════════════════════════════════════════════════════
def init_skeleton(root: Path, manifest_data: Mapping[str, Any]) -> VaultPaths:
    """Create the §3 skeleton under ``root`` and write ``manifest_data``.

    The test-side equivalent of ``mechabrain init``: it deliberately does *not*
    call the CLI, so that these fixtures stay usable while `init` is being
    written and a bug there cannot silently break every other module's tests.
    Like the real ``init``, it also generates ``AGENTS.md``/``schema.md`` when
    the manifest parses -- ``check`` verifies the derived docs match the config
    (§10), so a fixture vault without them would read as drifted. A manifest
    that does *not* parse (a fixture testing a bad config) skips the docs, since
    the generators need a valid :class:`Manifest`.
    """
    paths = VaultPaths.for_root(root)
    for directory in paths.contract_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    for agent in manifest_data.get("agents") or []:
        paths.episodic_for(str(agent["id"])).mkdir(parents=True, exist_ok=True)
    write_atomic(
        paths.config_file,
        yaml.safe_dump(dict(manifest_data), sort_keys=False, allow_unicode=True),
    )
    try:
        manifest = Manifest.from_mapping(manifest_data)
    except MechabrainError:
        return paths
    write_schema(paths, manifest)
    write_agents_md(paths, manifest)
    return paths


@pytest.fixture
def make_vault(tmp_path: Path, manifest_data_ci: dict[str, Any]) -> Callable[..., VaultPaths]:
    """Factory for disposable vaults.

    ::

        paths = make_vault()                       # same config as tmp_vault
        other = make_vault(name="other")           # a second vault
        custom = make_vault(manifest_data={...})   # a specific manifest on disk

    Returns:
        ``(manifest_data=None, name="vault") -> VaultPaths``
    """

    def _make(
        manifest_data: Mapping[str, Any] | None = None,
        name: str = "vault",
    ) -> VaultPaths:
        data = copy.deepcopy(manifest_data_ci) if manifest_data is None else manifest_data
        return init_skeleton(tmp_path / name, data)

    return _make


@pytest.fixture
def tmp_vault(make_vault: Callable[..., VaultPaths]) -> VaultPaths:
    """A disposable, initialized vault: full §3 skeleton + :fixture:`manifest_ci`.

    Contains no notes. Its root is *not* the CWD, so an upward walk from it
    finds it while an upward walk from the CWD does not.
    """
    return make_vault()


# ══════════════════════════════════════════════════════════════════════
# Notes
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def write_note() -> Callable[..., Note]:
    """Factory writing one note to disk.

    ::

        note = write_note(tmp_vault.semantic_dir / "2026-01-15_INS_x.md",
                          {"title": "X", "scope": "proj-a"},
                          "body text")

    Returns:
        ``(path, frontmatter=None, body="") -> Note``
    """

    def _write(
        path: Path | str,
        frontmatter: Mapping[str, Any] | None = None,
        body: str = "",
    ) -> Note:
        note = Note(path=Path(path), frontmatter=dict(frontmatter or {}), body=body)
        note.write()
        return note

    return _write


def _frontmatter(
    title: str,
    memory_type: str,
    agent: str,
    scope: str,
    *,
    profile: str | None = None,
    status: str = "ativo",
    confidence: str = "medium",
) -> dict[str, Any]:
    """Frontmatter per §6, in the spec's key order."""
    frontmatter: dict[str, Any] = {
        "title": title,
        "tags": [f"mem/{memory_type}", f"agent/{agent}"],
        "created": date(2026, 1, 15),
        "modified": date(2026, 1, 15),
        "agent": agent,
    }
    if profile is not None:
        frontmatter["profile"] = profile
    frontmatter.update(
        {
            "scope": scope,
            "source": "test-session",
            "confidence": confidence,
            "last_accessed": date(2026, 1, 15),
            "status": status,
        }
    )
    return frontmatter


@pytest.fixture
def sample_notes(tmp_vault: VaultPaths, write_note: Callable[..., Note]) -> list[Note]:
    """Six notes written into :fixture:`tmp_vault`, returned in path order.

    Shaped to exercise the filters the retrieval and maintenance modules need
    (§7.1), so a test can assert that a filter excludes something real:

    ===================================  =========================================
    Note id                              Purpose
    ===================================  =========================================
    ``2026-01-15_INS_vector-search``     semantic, ``proj-a``, alpha, high conf.
    ``2026-01-15_INS_global-fact``       semantic, ``global``, beta -- cross-scope
                                         partner for R6.5 tests
    ``2026-01-15_INS_stale-fact``        semantic, ``proj-b``, ``status: arquivado``
                                         -- decay/`status` filter target (§9.3)
    ``2026-01-15_MEM_session-one``       episodic under ``Episodic/alpha/`` (R6.3)
    ``PROC_deploy-playbook``             procedural, atemporal, cites evidence
                                         (§8.2 item 6)
    ``2026-01-15_RES_link-expansion``    research, links to the semantic note --
                                         a seed for link-expansion tests (§7.1)
    ===================================  =========================================
    """
    notes = [
        write_note(
            tmp_vault.semantic_dir / "2026-01-15_INS_vector-search.md",
            _frontmatter("Vector search", "semantic", "alpha", "proj-a", profile="tutor", confidence="high"),
            "Brute-force cosine is fast enough below ten thousand chunks.",
        ),
        write_note(
            tmp_vault.semantic_dir / "2026-01-15_INS_global-fact.md",
            _frontmatter("A global fact", "semantic", "beta", "global"),
            "Markdown stays the source of truth; every index is derived.",
        ),
        write_note(
            tmp_vault.semantic_dir / "2026-01-15_INS_stale-fact.md",
            _frontmatter("A stale fact", "semantic", "alpha", "proj-b", status="arquivado"),
            "Nobody has read this in a long time.",
        ),
        write_note(
            tmp_vault.episodic_for("alpha") / "2026-01-15_MEM_session-one.md",
            _frontmatter("Session one", "episodic", "alpha", "proj-a", profile="planner"),
            "Ran the pipeline end to end and it worked.",
        ),
        write_note(
            tmp_vault.procedural_dir / "PROC_deploy-playbook.md",
            _frontmatter("Deploy playbook", "procedural", "beta", "proj-a", confidence="high"),
            "## Steps\n\n1. Build.\n2. Ship.\n\n## Evidence\n\nExecuted on 2026-01-14.",
        ),
        write_note(
            tmp_vault.research_dir / "2026-01-15_RES_link-expansion.md",
            _frontmatter("On link expansion", "research", "beta", "proj-a"),
            "Graph-lite beats extraction on an authored corpus. "
            "See [[2026-01-15_INS_vector-search]].",
        ),
    ]
    return sorted(notes, key=lambda note: str(note.path))
