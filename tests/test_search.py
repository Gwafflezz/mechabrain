"""Tests for memory.search (``mechabrain.search``, spec §7.1).

The suite runs on the deterministic pair the fixtures install -- ``provider:
hash`` + ``store: numpy`` -- so there is no model download and every score is
reproducible. Hash embeddings carry no semantics, which is exactly what makes
them a stable substrate for asserting *mechanics* (fusion, filters, expansion,
provenance) without depending on what a real model happens to think two
sentences mean.

The index is built the way a reindexer would: chunk each note, embed the
``embed_text``, upsert vectors with :func:`mechabrain.search.store_metadata`, and
mirror the same chunks into the BM25 index. That keeps the metadata contract in
one place -- :data:`~mechabrain.search.STORE_META_KEYS` -- shared by search and
whatever populates the index.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest

from mechabrain.access import AccessLog
from mechabrain.discovery import VaultPaths
from mechabrain.index.chunk import chunk_note
from mechabrain.index.embed import from_manifest as embed_from_manifest
from mechabrain.index.lexical import LEXICAL_DB_FILENAME, LexicalChunk, LexicalIndex
from mechabrain.index.store import from_manifest as store_from_manifest
from mechabrain.manifest import Manifest
from mechabrain.note import Note
from mechabrain.search import (
    ARCHIVED_PENALTY,
    HOP_DECAY,
    Hit,
    Retriever,
    search,
    store_metadata,
)

# Ids of the notes the ``sample_notes`` fixture writes.
VECTOR_SEARCH = "2026-01-15_INS_vector-search"
GLOBAL_FACT = "2026-01-15_INS_global-fact"
STALE_FACT = "2026-01-15_INS_stale-fact"
SESSION_ONE = "2026-01-15_MEM_session-one"
DEPLOY_PLAYBOOK = "PROC_deploy-playbook"
LINK_EXPANSION = "2026-01-15_RES_link-expansion"

# The §7.1 hit fields provenance and citation depend on (R7.1).
REQUIRED_FIELDS = (
    "id",
    "path",
    "wikilink",
    "title",
    "type",
    "agent",
    "confidence",
    "score",
    "excerpt",
    "created",
)


# ══════════════════════════════════════════════════════════════════════
# Indexing helper -- what a reindexer does, inline
# ══════════════════════════════════════════════════════════════════════
def _memory_type(note: Note) -> str:
    """The note's memory type, read off its ``mem/<type>`` tag (§6)."""
    for tag in note.tags:
        if tag.startswith("mem/"):
            return tag.split("/", 1)[1]
    raise AssertionError(f"note {note.note_id} carries no mem/<type> tag")


def index_notes(paths: VaultPaths, manifest: Manifest, notes: Sequence[Note]) -> None:
    """Populate the vector store and BM25 index over ``notes``.

    Mirrors a reindex: one embedding pass, vectors keyed by chunk id with
    :func:`store_metadata`, the same chunks mirrored into BM25 so a term in the
    context prefix is findable on both sides.
    """
    provider = embed_from_manifest(manifest)
    store = store_from_manifest(manifest, paths, autosave=False)
    lexical = LexicalIndex(paths.index_dir / LEXICAL_DB_FILENAME)
    contextual = manifest.retrieval.contextual_retrieval
    try:
        for note in notes:
            memory_type = _memory_type(note)
            chunks = chunk_note(note, contextual=contextual)
            if not chunks:
                continue
            vectors = provider.embed_texts([chunk.embed_text for chunk in chunks])
            store.upsert(
                [chunk.chunk_id for chunk in chunks],
                vectors,
                [store_metadata(note, memory_type) for _ in chunks],
            )
            lexical.upsert(
                LexicalChunk(
                    chunk_id=chunk.chunk_id,
                    note_id=note.note_id,
                    text=chunk.embed_text,
                    memory_type=memory_type,
                    agent=_opt(note.get("agent")),
                    profile=_opt(note.get("profile")),
                    scope=_opt(note.get("scope")),
                    status=_opt(note.get("status")) or "ativo",
                    confidence=_opt(note.get("confidence")),
                    tags=tuple(note.tags),
                )
                for chunk in chunks
            )
        store.flush()
    finally:
        lexical.close()


def _opt(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ids(hits: Sequence[Hit]) -> list[str]:
    return [hit.id for hit in hits]


# ══════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def retriever(
    tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> Iterator[Retriever]:
    """A retriever over the six ``sample_notes``, indexed and ready."""
    index_notes(tmp_vault, manifest_ci, sample_notes)
    with Retriever(tmp_vault, manifest_ci) as open_retriever:
        yield open_retriever


# ══════════════════════════════════════════════════════════════════════
# Provenance (R7.1)
# ══════════════════════════════════════════════════════════════════════
def test_every_hit_carries_provenance(retriever: Retriever) -> None:
    hits = retriever.search("brute force cosine chunks", k=8)
    assert hits, "expected at least one hit for a query drawn from the corpus"
    for hit in hits:
        assert hit.path, "R7.1: path must always be present"
        assert hit.wikilink == f"[[{hit.id}]]"
        assert not hit.path.startswith("/"), "R4.2: path is vault-relative, never absolute"


def test_hit_exposes_every_required_field(retriever: Retriever) -> None:
    hit = retriever.search("brute force cosine chunks", k=1)[0]
    wire = hit.as_dict()
    for field in REQUIRED_FIELDS:
        assert field in wire, f"R7.1 hit is missing {field!r}"
    assert wire["scope"] and wire["profile"] is not None or wire["profile"] is None
    assert hit.id == VECTOR_SEARCH
    assert hit.type == "semantic"
    assert hit.agent == "alpha"
    assert hit.scope == "proj-a"
    assert hit.created == "2026-01-15"


def test_excerpt_is_raw_text_not_the_context_prefix(retriever: Retriever) -> None:
    """The excerpt is the raw chunk, never the ``scope: ... | title: ...`` prefix (§7.1)."""
    hit = retriever.search("brute force cosine chunks", k=1)[0]
    assert "Brute-force cosine" in hit.excerpt
    assert "scope:" not in hit.excerpt
    assert "title:" not in hit.excerpt
    assert "mem/semantic" not in hit.excerpt


# ══════════════════════════════════════════════════════════════════════
# Filters (§7.1 item 4)
# ══════════════════════════════════════════════════════════════════════
def test_scope_filter_blocks_cross_contamination(retriever: Retriever) -> None:
    """R6.5: a fact true in one project is not returned as truth in another."""
    hits = retriever.search(
        "markdown source of truth derived index",
        k=8,
        filters={"scope": "proj-a"},
        expand_links=0,
    )
    assert GLOBAL_FACT not in ids(hits), "a global-scope note leaked past a proj-a filter"
    for hit in hits:
        assert hit.scope == "proj-a"


def test_scope_filter_admits_the_named_scope(retriever: Retriever) -> None:
    hits = retriever.search(
        "markdown source of truth derived index",
        k=8,
        filters={"scope": "global"},
        expand_links=0,
    )
    assert GLOBAL_FACT in ids(hits)


def test_archived_excluded_by_default(retriever: Retriever) -> None:
    hits = retriever.search("nobody has read this in a long time", k=8, expand_links=0)
    assert STALE_FACT not in ids(hits), "§9.3: archived notes are out of default results"


def test_archived_searchable_with_explicit_status(retriever: Retriever) -> None:
    hits = retriever.search(
        "nobody has read this in a long time",
        k=8,
        filters={"status": "arquivado"},
        expand_links=0,
    )
    assert STALE_FACT in ids(hits), "§9.3: archived stays searchable with an explicit filter"
    for hit in hits:
        assert hit.id == STALE_FACT


def test_type_filter_selects_one_memory_kind(retriever: Retriever) -> None:
    hits = retriever.search(
        "pipeline build ship deploy", k=8, filters={"type": "procedural"}, expand_links=0
    )
    assert ids(hits), "expected a procedural hit"
    assert all(hit.type == "procedural" for hit in hits)


def test_min_confidence_is_a_floor(retriever: Retriever) -> None:
    hits = retriever.search(
        "markdown source of truth derived index",
        k=8,
        filters={"min_confidence": "high"},
        expand_links=0,
    )
    assert ids(hits)
    assert all(hit.confidence == "high" for hit in hits)
    # The medium-confidence global fact would match the query but is below the floor.
    assert GLOBAL_FACT not in ids(hits)


def test_tags_filter_is_conjunctive(retriever: Retriever) -> None:
    hits = retriever.search(
        "brute force cosine markdown",
        k=8,
        filters={"tags": ["mem/semantic", "agent/alpha"]},
        expand_links=0,
    )
    assert ids(hits)
    for hit in hits:
        assert hit.type == "semantic"
        assert hit.agent == "alpha"
    # A note tagged agent/beta cannot satisfy an "all of" over agent/alpha.
    assert GLOBAL_FACT not in ids(hits)


def test_unknown_filter_is_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="unknown search filter"):
        retriever.search("anything", filters={"scpoe": "proj-a"})


def test_unknown_status_value_is_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="unknown status"):
        retriever.search("anything", filters={"status": "archived"})


# ══════════════════════════════════════════════════════════════════════
# Fusion (§7.1 item 3)
# ══════════════════════════════════════════════════════════════════════
def test_lexical_term_surfaces_its_note(retriever: Retriever) -> None:
    """A query that shares a term with one note's body finds it through fusion."""
    playbook = retriever.search("Ship", k=8, filters={"type": "procedural"}, expand_links=0)
    assert DEPLOY_PLAYBOOK in ids(playbook)


def test_scores_are_sorted_descending(retriever: Retriever) -> None:
    hits = retriever.search("markdown cosine pipeline deploy", k=8)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════
# Link expansion (§7.1)
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def link_vault(make_vault: Callable[..., VaultPaths], manifest_ci: Manifest) -> VaultPaths:
    """A two-note vault: a seed wikilinking a target that shares no query term.

    The target can be reached *only* through the authored link, so its presence
    at one hop and absence at zero is unambiguous.
    """
    paths = make_vault(name="links")
    seed = Note(
        path=paths.semantic_dir / "seed.md",
        frontmatter={
            "title": "Seed",
            "tags": ["mem/semantic", "agent/alpha"],
            "scope": "proj-a",
            "agent": "alpha",
            "confidence": "high",
            "status": "ativo",
        },
        body="Flabbergasted kumquat findings. See [[target]].",
    )
    target = Note(
        path=paths.semantic_dir / "target.md",
        frontmatter={
            "title": "Zzz",
            "tags": ["mem/semantic", "agent/beta"],
            "scope": "proj-b",
            "agent": "beta",
            "confidence": "medium",
            "status": "ativo",
        },
        body="Nnn ooo ppp qqq.",
    )
    seed.write()
    target.write()
    index_notes(paths, manifest_ci, [seed, target])
    return paths


def test_expansion_reaches_a_wikilink_only_note(
    link_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    with Retriever(link_vault, manifest_ci) as retriever:
        without = retriever.search("flabbergasted kumquat", k=8, expand_links=0)
        assert "target" not in ids(without), "target shares no term; it is not a direct hit"

        with_expansion = retriever.search("flabbergasted kumquat", k=8, expand_links=1)

    reached = {hit.id: hit for hit in with_expansion}
    assert "target" in reached, "one hop along the authored link should reach the target"
    target = reached["target"]
    assert target.hops == 1
    assert target.via == "[[seed]] → [[target]]"
    assert target.similarity == 0.0, "an expanded note has no direct cosine"


def test_expanded_score_decays_below_its_seed(
    link_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    with Retriever(link_vault, manifest_ci) as retriever:
        hits = {hit.id: hit for hit in retriever.search("flabbergasted kumquat", expand_links=1)}
    seed, target = hits["seed"], hits["target"]
    assert target.via is not None and seed.via is None
    assert target.score == pytest.approx(seed.score * HOP_DECAY, rel=1e-6)


def test_direct_hits_have_no_via(retriever: Retriever) -> None:
    hit = retriever.search("brute force cosine chunks", k=1)[0]
    assert hit.via is None
    assert hit.hops == 0


def test_default_hops_come_from_the_manifest(
    link_vault: VaultPaths, manifest_ci: Manifest
) -> None:
    """``expand_links=None`` uses ``link_expansion.default_hops`` (1 in manifest_ci)."""
    with Retriever(link_vault, manifest_ci) as retriever:
        hits = retriever.search("flabbergasted kumquat")
    assert "target" in ids(hits)


def test_expansion_does_not_resurrect_archived(
    make_vault: Callable[..., VaultPaths], manifest_ci: Manifest
) -> None:
    """A link to an archived note does not pull it back in (§9.3 / P8)."""
    paths = make_vault(name="archived-link")
    seed = Note(
        path=paths.semantic_dir / "live.md",
        frontmatter={
            "title": "Live",
            "tags": ["mem/semantic", "agent/alpha"],
            "scope": "proj-a",
            "agent": "alpha",
            "status": "ativo",
        },
        body="Flabbergasted kumquat. See [[dead]].",
    )
    archived = Note(
        path=paths.semantic_dir / "dead.md",
        frontmatter={
            "title": "Dead",
            "tags": ["mem/semantic", "agent/alpha"],
            "scope": "proj-a",
            "agent": "alpha",
            "status": "arquivado",
        },
        body="Nnn ooo ppp.",
    )
    seed.write()
    archived.write()
    index_notes(paths, manifest_ci, [seed, archived])
    with Retriever(paths, manifest_ci) as retriever:
        hits = retriever.search("flabbergasted kumquat", expand_links=1)
    assert "dead" not in ids(hits)


def test_expand_links_above_ceiling_is_an_error(retriever: Retriever) -> None:
    """A value above ``max_hops`` is refused, not clamped (R5.1)."""
    with pytest.raises(ValueError, match="max_hops"):
        retriever.search("anything", expand_links=99)


def test_negative_expand_links_is_an_error(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="hops"):
        retriever.search("anything", expand_links=-1)


# ══════════════════════════════════════════════════════════════════════
# Access tracking (R7.2)
# ══════════════════════════════════════════════════════════════════════
def test_search_records_access(retriever: Retriever, tmp_vault: VaultPaths) -> None:
    hits = retriever.search("brute force cosine chunks", k=3)
    accessed = AccessLog.for_vault(tmp_vault).aggregate()
    for hit in hits:
        assert hit.id in accessed, "R7.2: every returned note is logged for decay"


# ══════════════════════════════════════════════════════════════════════
# Edges
# ══════════════════════════════════════════════════════════════════════
def test_blank_query_returns_nothing(retriever: Retriever) -> None:
    assert retriever.search("   ", k=8) == []


def test_k_is_respected(retriever: Retriever) -> None:
    hits = retriever.search("markdown cosine pipeline deploy build ship", k=2)
    assert len(hits) <= 2


def test_k_below_one_is_rejected(retriever: Retriever) -> None:
    with pytest.raises(ValueError, match="k must be"):
        retriever.search("anything", k=0)


def test_module_level_search_matches_the_method(
    tmp_vault: VaultPaths, manifest_ci: Manifest, sample_notes: list[Note]
) -> None:
    index_notes(tmp_vault, manifest_ci, sample_notes)
    convenience = search(tmp_vault, manifest_ci, "brute force cosine chunks", k=3)
    with Retriever(tmp_vault, manifest_ci) as retriever:
        method = retriever.search("brute force cosine chunks", k=3)
    assert ids(convenience) == ids(method)


def test_store_metadata_matches_the_documented_contract(sample_notes: list[Note]) -> None:
    note = next(n for n in sample_notes if n.note_id == VECTOR_SEARCH)
    meta = store_metadata(note, "semantic")
    assert meta["note_id"] == VECTOR_SEARCH
    assert meta["type"] == "semantic"
    assert meta["agent"] == "alpha"
    assert meta["scope"] == "proj-a"
    assert meta["status"] == "ativo"
    assert meta["confidence"] == "high"
    assert "mem/semantic" in meta["tags"]


def test_archived_penalty_applied_when_explicitly_included(
    make_vault: Callable[..., VaultPaths], manifest_ci: Manifest
) -> None:
    """An explicitly-included archived note keeps only ARCHIVED_PENALTY of its weight."""
    paths = make_vault(name="penalty")
    # Identical title, tags and body -> identical embed_text -> identical vector
    # and BM25 scores, so the only thing separating the two fused scores is the
    # archived penalty. (Differing titles would let min-max break the tie.)
    body = "Flabbergasted kumquat findings here."
    common = {"title": "Same", "tags": ["mem/semantic", "agent/alpha"], "scope": "proj-a", "agent": "alpha"}
    live = Note(
        path=paths.semantic_dir / "live.md",
        frontmatter={**common, "status": "ativo"},
        body=body,
    )
    dead = Note(
        path=paths.semantic_dir / "dead.md",
        frontmatter={**common, "status": "arquivado"},
        body=body,
    )
    live.write()
    dead.write()
    index_notes(paths, manifest_ci, [live, dead])
    with Retriever(paths, manifest_ci) as retriever:
        hits = {
            hit.id: hit
            for hit in retriever.search(
                "flabbergasted kumquat findings",
                filters={"status": ["ativo", "arquivado"]},
                expand_links=0,
            )
        }
    assert set(hits) == {"live", "dead"}
    # Identical bodies -> identical fused score; the archived one is penalised.
    assert hits["dead"].score == pytest.approx(hits["live"].score * ARCHIVED_PENALTY, rel=1e-6)


# ══════════════════════════════════════════════════════════════════════
# Regression: the floor member of one ranking is a hit, not a cull (§7.1)
# ══════════════════════════════════════════════════════════════════════
def test_floor_hit_survives_in_a_small_corpus(
    tmp_vault: VaultPaths, manifest_ci: Manifest, write_note
) -> None:
    """Two matching notes in a two-note corpus must both come back.

    Min-max floors the weaker member of each half to 0; when the same note is
    the floor of both halves its fused score is exactly 0. The old cull
    (`score <= 0.0: continue`) silently dropped it -- in a small corpus that
    loses a valid match entirely.
    """
    from datetime import date

    def fm(title: str) -> dict:
        return {
            "title": title,
            "tags": ["mem/semantic", "agent/alpha"],
            "created": date(2026, 1, 15),
            "modified": date(2026, 1, 15),
            "agent": "alpha",
            "scope": "proj-a",
            "source": "test",
            "confidence": "medium",
            "last_accessed": date(2026, 1, 15),
            "status": "ativo",
        }

    strong = write_note(
        tmp_vault.semantic_dir / "2026-01-15_INS_zebraxq-a.md",
        fm("Zebraxq A"),
        "zebraxq protocol",
    )
    weak = write_note(
        tmp_vault.semantic_dir / "2026-01-15_INS_zebraxq-b.md",
        fm("Zebraxq B"),
        "zebraxq protocol with much longer additional unrelated commentary padding",
    )
    index_notes(tmp_vault, manifest_ci, [strong, weak])

    with Retriever(tmp_vault, manifest_ci) as retriever:
        hits = retriever.search("zebraxq protocol", k=8)

    ids = {hit.id for hit in hits}
    assert "2026-01-15_INS_zebraxq-a" in ids
    assert "2026-01-15_INS_zebraxq-b" in ids, "floor member was culled (regression)"
