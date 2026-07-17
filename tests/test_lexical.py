"""Tests for the BM25 lexical index (``mechabrain.index.lexical``).

The corpus is deliberately Portuguese and accented: the deployment this kernel
was written for writes pt-BR, and accent folding is the one tokenizer behaviour
that silently degrades recall rather than failing loudly.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from mechabrain.contract import MemoryType
from mechabrain.discovery import VaultPaths
from mechabrain.errors import MechabrainIndexError
from mechabrain.index.lexical import (
    FILTER_KEYS,
    LEXICAL_DB_FILENAME,
    SCHEMA_VERSION,
    LexicalChunk,
    LexicalIndex,
)


@pytest.fixture
def db_path(tmp_vault: VaultPaths) -> Path:
    """The index file where it really lives: derived state under `_meta/index/`."""
    return tmp_vault.index_dir / LEXICAL_DB_FILENAME


@pytest.fixture
def index(db_path: Path) -> Iterator[LexicalIndex]:
    with LexicalIndex(db_path) as open_index:
        yield open_index


@pytest.fixture
def make_chunk() -> Callable[..., LexicalChunk]:
    """Factory for a chunk with sane defaults; override only what a test is about."""

    def _make(chunk_id: str, text: str, **overrides: Any) -> LexicalChunk:
        fields: dict[str, Any] = {
            "chunk_id": chunk_id,
            "note_id": chunk_id.split("#")[0],
            "text": text,
        }
        fields.update(overrides)
        return LexicalChunk(**fields)

    return _make


def ids(hits: list[tuple[str, float]]) -> list[str]:
    return [chunk_id for chunk_id, _ in hits]


# ══════════════════════════════════════════════════════════════════════
# Lifecycle
# ══════════════════════════════════════════════════════════════════════
def test_opens_lazily_and_creates_the_file(db_path: Path) -> None:
    index = LexicalIndex(db_path)
    assert not db_path.exists(), "constructing the index must not touch the disk"
    assert index.count() == 0
    assert db_path.is_file()
    index.close()


def test_creates_missing_parent_directory(tmp_path: Path) -> None:
    """An empty index is a valid index: it bootstraps its own directory."""
    with LexicalIndex(tmp_path / "absent" / LEXICAL_DB_FILENAME) as index:
        assert index.count() == 0


def test_survives_close_and_reopen(db_path: Path, make_chunk: Callable[..., LexicalChunk]) -> None:
    with LexicalIndex(db_path) as index:
        index.upsert([make_chunk("n1#0", "manutenção do índice derivado")])

    with LexicalIndex(db_path) as reopened:
        assert reopened.count() == 1
        assert ids(reopened.search("manutenção")) == ["n1#0"]


def test_close_is_idempotent_and_reopens_on_use(db_path: Path) -> None:
    index = LexicalIndex(db_path)
    index.count()
    index.close()
    index.close()
    assert index.count() == 0


def test_rejects_a_database_from_another_schema_version(db_path: Path) -> None:
    """Derived state is never migrated: it is rebuilt (P1)."""
    LexicalIndex(db_path).close()
    connection = sqlite3.connect(db_path)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(MechabrainIndexError) as excinfo:
        LexicalIndex(db_path).count()
    assert "reindex" in str(excinfo.value)


# ══════════════════════════════════════════════════════════════════════
# Writing
# ══════════════════════════════════════════════════════════════════════
def test_upsert_counts_and_reports_note_ids(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    written = index.upsert(
        [
            make_chunk("n1#0", "primeiro trecho"),
            make_chunk("n1#1", "segundo trecho"),
            make_chunk("n2#0", "trecho de outra nota"),
        ]
    )
    assert written == 3
    assert index.count() == 3
    assert index.note_ids() == {"n1", "n2"}


def test_upsert_of_nothing_is_a_no_op(index: LexicalIndex) -> None:
    assert index.upsert([]) == 0
    assert index.count() == 0


def test_upsert_replaces_by_chunk_id_and_unindexes_the_old_terms(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    """The regression `INSERT OR REPLACE` would cause: stale terms left in FTS."""
    index.upsert([make_chunk("n1#0", "conteúdo antigo sobre girassóis")])
    index.upsert([make_chunk("n1#0", "conteúdo novo sobre bússolas")])

    assert index.count() == 1, "the same chunk_id must not duplicate"
    assert index.search("girassóis") == []
    assert ids(index.search("bússolas")) == ["n1#0"]


def test_upsert_replaces_tags_rather_than_accumulating(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert([make_chunk("n1#0", "texto", tags=("mem/semantic",))])
    index.upsert([make_chunk("n1#0", "texto", tags=("mem/research",))])

    assert index.search("texto", filters={"tags": ["mem/research"]}) != []
    assert index.search("texto", filters={"tags": ["mem/semantic"]}) == []


def test_upsert_accepts_mappings(index: LexicalIndex) -> None:
    index.upsert([{"chunk_id": "n1#0", "note_id": "n1", "text": "trecho avulso"}])
    assert ids(index.search("avulso")) == ["n1#0"]


def test_upsert_rejects_a_chunk_with_no_text(index: LexicalIndex) -> None:
    with pytest.raises(ValueError, match="text"):
        index.upsert([{"chunk_id": "n1#0", "note_id": "n1", "text": "   "}])


def test_upsert_rejects_a_foreign_type(index: LexicalIndex) -> None:
    with pytest.raises(TypeError, match="LexicalChunk"):
        index.upsert(["not a chunk"])


def test_a_failed_upsert_leaves_the_index_untouched(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    """One bad chunk must not half-write the batch."""
    index.upsert([make_chunk("n1#0", "estado anterior")])
    with pytest.raises(ValueError):
        index.upsert([make_chunk("n2#0", "novo"), {"note_id": "n3", "text": "sem id"}])

    assert index.count() == 1
    assert index.note_ids() == {"n1"}


# ══════════════════════════════════════════════════════════════════════
# Deleting
# ══════════════════════════════════════════════════════════════════════
def test_delete_removes_every_chunk_of_a_note_and_unindexes_it(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert(
        [
            make_chunk("n1#0", "alfa"),
            make_chunk("n1#1", "beta"),
            make_chunk("n2#0", "gama"),
        ]
    )
    assert index.delete(["n1"]) == 2
    assert index.note_ids() == {"n2"}
    assert index.search("alfa") == []
    assert ids(index.search("gama")) == ["n2#0"]


def test_delete_of_an_unknown_note_is_not_an_error(index: LexicalIndex) -> None:
    assert index.delete(["never-indexed"]) == 0


def test_delete_of_nothing_is_a_no_op(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert([make_chunk("n1#0", "alfa")])
    assert index.delete([]) == 0
    assert index.count() == 1


def test_clear_empties_the_index_but_keeps_it_usable(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert([make_chunk("n1#0", "alfa"), make_chunk("n2#0", "beta")])
    assert index.clear() == 2
    assert index.count() == 0
    assert index.note_ids() == set()
    assert index.search("alfa") == []

    index.upsert([make_chunk("n3#0", "gama")])
    assert ids(index.search("gama")) == ["n3#0"]


# ══════════════════════════════════════════════════════════════════════
# Portuguese
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    ("indexed", "query"),
    [
        ("manutenção", "manutencao"),
        ("manutencao", "manutenção"),
        ("índice", "indice"),
        ("Ávila", "avila"),
        ("coração", "CORAÇÃO"),
        ("consolidação", "CONSOLIDACAO"),
        ("ambiguïdade", "ambiguidade"),
    ],
)
def test_accents_fold_in_both_directions(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk], indexed: str, query: str
) -> None:
    index.upsert([make_chunk("n1#0", f"um texto com {indexed} no meio")])
    assert ids(index.search(query)) == ["n1#0"], f"{query!r} should find {indexed!r}"


def test_accented_multi_word_query(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert(
        [
            make_chunk("n1#0", "A consolidação agrega os logs de acesso e atualiza o frontmatter."),
            make_chunk("n2#0", "Uma nota qualquer sobre outro assunto."),
        ]
    )
    assert ids(index.search("consolidacao dos logs de acesso")) == ["n1#0"]


def test_hyphenated_and_punctuated_portuguese_is_tokenized(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert([make_chunk("n1#0", "sistemas não-lineares, avaliação (parcial): ok!")])
    assert ids(index.search("não-lineares")) == ["n1#0"]
    assert ids(index.search("lineares")) == ["n1#0"]


# ══════════════════════════════════════════════════════════════════════
# Query sanitisation
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "query",
    [
        'aspas "não" fechadas',
        '"',
        '""',
        "estrela *",
        "* alfa",
        "coluna: valor",
        "^inicio",
        "(parêntese",
        "parêntese)",
        "-negado",
        "alfa AND beta",
        "alfa OR beta",
        "alfa NOT beta",
        "alfa NEAR beta",
        "NEAR(alfa beta, 3)",
        "alfa*",
        "{coluna} : alfa",
        "índice AND (alfa OR NOT beta) NEAR/2",
        "'; DROP TABLE chunks; --",
    ],
)
def test_fts5_syntax_in_free_input_never_explodes(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk], query: str
) -> None:
    """Agents type prose, not FTS5. Every one of these is a term, not an operator."""
    index.upsert([make_chunk("n1#0", "alfa beta gama")])
    index.search(query)  # must not raise
    assert index.count() == 1, "a query must never mutate the index"


def test_operator_words_are_searched_as_terms(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    """`NOT` in a query is the word, not the operator -- so it cannot exclude."""
    index.upsert([make_chunk("n1#0", "alfa gama"), make_chunk("n2#0", "beta gama")])
    hits = ids(index.search("gama NOT beta"))

    # As an operator this would drop n2#0; as terms it ranks n2#0 first, since
    # n2#0 matches two of them ("gama", "beta") and n1#0 only one.
    assert sorted(hits) == ["n1#0", "n2#0"]
    assert hits[0] == "n2#0"


def test_a_query_with_no_searchable_term_returns_nothing(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert([make_chunk("n1#0", "alfa beta")])
    for empty in ["", "   ", "!!!", '"', "-*^:()", "___"]:
        assert index.search(empty) == [], f"{empty!r} holds no term"


def test_a_repeated_term_is_not_repeated_in_the_expression(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert([make_chunk("n1#0", "alfa beta")])
    assert ids(index.search("alfa alfa ALFA alfa")) == ["n1#0"]


def test_terms_are_or_ed_so_a_partial_match_still_ranks(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    """Under FTS5's implicit AND this would be empty -- useless to hybrid fusion."""
    index.upsert([make_chunk("n1#0", "o índice derivado é reconstruível")])
    hits = index.search("como reconstruir o índice depois de mover a vault")
    assert ids(hits) == ["n1#0"]


# ══════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════
def test_scores_are_positive_and_descending(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    """FTS5's bm25() is negative and ascending; the API inverts both."""
    index.upsert(
        [make_chunk(f"n{i}#0", f"documento {i} sobre índice comum") for i in range(10)]
        + [make_chunk("raro#0", "documento raro sobre bússolas e índice")]
    )
    hits = index.search("índice comum raro", k=11)

    scores = [score for _, score in hits]
    assert all(score > 0 for score in scores), f"scores must be positive, got {scores}"
    assert scores == sorted(scores, reverse=True)


def test_the_rarer_match_outranks_the_common_one(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert(
        [make_chunk(f"comum{i}#0", "documento sobre índice comum") for i in range(9)]
        + [make_chunk("raro#0", "documento sobre índice e bússolas")]
    )
    assert ids(index.search("índice bússolas"))[0] == "raro#0"


def test_ties_break_on_chunk_id_so_the_order_is_reproducible(
    index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]
) -> None:
    index.upsert(
        [
            make_chunk("n3#0", "texto idêntico"),
            make_chunk("n1#0", "texto idêntico"),
            make_chunk("n2#0", "texto idêntico"),
        ]
    )
    hits = index.search("idêntico")
    assert len({score for _, score in hits}) == 1, "the fixture must actually tie"
    assert ids(hits) == ["n1#0", "n2#0", "n3#0"]


def test_k_limits_the_hits(index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]) -> None:
    index.upsert([make_chunk(f"n{i}#0", "termo repetido") for i in range(10)])
    assert len(index.search("termo", k=3)) == 3
    assert len(index.search("termo", k=99)) == 10


def test_k_defaults_to_eight(index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]) -> None:
    index.upsert([make_chunk(f"n{i}#0", "termo repetido") for i in range(20)])
    assert len(index.search("termo")) == 8


@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_k_is_rejected(index: LexicalIndex, k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        index.search("alfa", k=k)


def test_searching_an_empty_index_is_fine(index: LexicalIndex) -> None:
    assert index.search("qualquer coisa") == []


# ══════════════════════════════════════════════════════════════════════
# Filters
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def filterable(index: LexicalIndex, make_chunk: Callable[..., LexicalChunk]) -> LexicalIndex:
    """Chunks spanning every filterable dimension, all matching the term "memória"."""
    index.upsert(
        [
            make_chunk(
                "sem-a#0",
                "memória semântica do projeto a",
                memory_type="semantic",
                agent="alpha",
                profile="tutor",
                scope="proj-a",
                status="ativo",
                confidence="high",
                tags=("mem/semantic", "agent/alpha"),
            ),
            make_chunk(
                "epi-b#0",
                "memória episódica do projeto b",
                memory_type="episodic",
                agent="beta",
                profile="planner",
                scope="proj-b",
                status="ativo",
                confidence="medium",
                tags=("mem/episodic", "agent/beta"),
            ),
            make_chunk(
                "arq-g#0",
                "memória arquivada global",
                memory_type="semantic",
                agent="beta",
                scope="global",
                status="arquivado",
                confidence="low",
                tags=("mem/semantic", "agent/beta"),
            ),
            make_chunk("sem-meta#0", "memória sem metadados"),
        ]
    )
    return index


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"type": "semantic"}, ["arq-g#0", "sem-a#0"]),
        ({"type": "episodic"}, ["epi-b#0"]),
        ({"agent": "alpha"}, ["sem-a#0"]),
        ({"agent": "beta"}, ["arq-g#0", "epi-b#0"]),
        ({"profile": "tutor"}, ["sem-a#0"]),
        ({"scope": "proj-a"}, ["sem-a#0"]),
        ({"scope": "global"}, ["arq-g#0"]),
        ({"status": "ativo"}, ["epi-b#0", "sem-a#0"]),
        ({"status": "arquivado"}, ["arq-g#0"]),
        ({"tags": ["mem/semantic"]}, ["arq-g#0", "sem-a#0"]),
        ({"tags": ["agent/beta"]}, ["arq-g#0", "epi-b#0"]),
        ({"min_confidence": "low"}, ["arq-g#0", "epi-b#0", "sem-a#0"]),
        ({"min_confidence": "medium"}, ["epi-b#0", "sem-a#0"]),
        ({"min_confidence": "high"}, ["sem-a#0"]),
        ({"scope": ["proj-a", "proj-b"]}, ["epi-b#0", "sem-a#0"]),
        ({"type": "semantic", "agent": "beta"}, ["arq-g#0"]),
        ({"type": "semantic", "status": "ativo"}, ["sem-a#0"]),
        ({"scope": "proj-a", "agent": "beta"}, []),
        ({}, ["arq-g#0", "epi-b#0", "sem-a#0", "sem-meta#0"]),
        (None, ["arq-g#0", "epi-b#0", "sem-a#0", "sem-meta#0"]),
    ],
)
def test_filters(
    filterable: LexicalIndex, filters: dict[str, Any] | None, expected: list[str]
) -> None:
    assert sorted(ids(filterable.search("memória", k=10, filters=filters))) == expected


def test_scope_isolation_holds_across_a_query_that_matches_everything(
    filterable: LexicalIndex,
) -> None:
    """R6.5: a fact true in proj-a must not surface as truth in proj-b."""
    assert ids(filterable.search("memória", k=10, filters={"scope": "proj-b"})) == ["epi-b#0"]


def test_tags_filter_is_conjunctive(filterable: LexicalIndex) -> None:
    """Several tags narrow: all must be present, not any."""
    assert ids(filterable.search("memória", filters={"tags": ["mem/semantic", "agent/alpha"]})) == [
        "sem-a#0"
    ]
    assert filterable.search("memória", filters={"tags": ["mem/semantic", "agent/nobody"]}) == []


def test_tags_filter_normalizes_the_leading_hash(filterable: LexicalIndex) -> None:
    """`#mem/semantic` and `mem/semantic` are the same tag (see normalize_tags)."""
    assert ids(filterable.search("memória", k=10, filters={"tags": "#mem/semantic"})) == ids(
        filterable.search("memória", k=10, filters={"tags": ["mem/semantic"]})
    )


def test_min_confidence_excludes_a_chunk_with_no_confidence(filterable: LexicalIndex) -> None:
    """An unranked chunk fails a floor -- the safe direction."""
    hits = ids(filterable.search("memória", k=10, filters={"min_confidence": "low"}))
    assert "sem-meta#0" not in hits


def test_a_memory_type_enum_works_as_a_filter_value(filterable: LexicalIndex) -> None:
    """MemoryType subclasses str, so it needs no special case."""
    assert sorted(ids(filterable.search("memória", k=10, filters={"type": MemoryType.SEMANTIC}))) == [
        "arq-g#0",
        "sem-a#0",
    ]


def test_an_unknown_filter_key_is_rejected(filterable: LexicalIndex) -> None:
    """R5.1: a dropped `scope` typo would silently cross a scope boundary."""
    with pytest.raises(ValueError, match="unknown search filter") as excinfo:
        filterable.search("memória", filters={"scopes": "proj-a"})
    assert "scope" in str(excinfo.value), "the error must name the valid filters"


def test_an_unknown_min_confidence_is_rejected(filterable: LexicalIndex) -> None:
    with pytest.raises(ValueError, match="unknown min_confidence") as excinfo:
        filterable.search("memória", filters={"min_confidence": "altíssima"})
    assert "high" in str(excinfo.value), "the error must list the valid levels"


def test_an_empty_scalar_filter_is_rejected(filterable: LexicalIndex) -> None:
    with pytest.raises(ValueError, match="omit it"):
        filterable.search("memória", filters={"scope": []})


def test_a_none_filter_value_is_ignored(filterable: LexicalIndex) -> None:
    """MCP clients send absent optionals as null rather than omitting the key."""
    assert len(filterable.search("memória", k=10, filters={"scope": None, "agent": None})) == 4


def test_filter_keys_are_exactly_the_spec_ones() -> None:
    assert FILTER_KEYS == {"type", "agent", "profile", "scope", "tags", "status", "min_confidence"}


# ══════════════════════════════════════════════════════════════════════
# LexicalChunk
# ══════════════════════════════════════════════════════════════════════
def test_from_mapping_accepts_the_chunkers_spellings() -> None:
    """The chunker emits `embed_text`; frontmatter-derived metadata says `type`."""
    chunk = LexicalChunk.from_mapping(
        {
            "chunk_id": "n1#0",
            "note_id": "n1",
            "embed_text": "escopo | título\n\ntexto",
            "type": "semantic",
            "tags": ["#mem/semantic", "agent/alpha"],
        }
    )
    assert chunk.text == "escopo | título\n\ntexto"
    assert chunk.memory_type == "semantic"
    assert chunk.tags == ("mem/semantic", "agent/alpha")


def test_from_mapping_prefers_text_over_embed_text() -> None:
    chunk = LexicalChunk.from_mapping(
        {"chunk_id": "n1#0", "note_id": "n1", "text": "explícito", "embed_text": "ignorado"}
    )
    assert chunk.text == "explícito"


def test_from_mapping_defaults_the_optional_metadata() -> None:
    chunk = LexicalChunk.from_mapping({"chunk_id": "n1#0", "note_id": "n1", "text": "texto"})
    assert chunk.memory_type is None
    assert chunk.scope is None
    assert chunk.tags == ()
    assert chunk.confidence_rank is None


@pytest.mark.parametrize("missing", ["chunk_id", "note_id", "text"])
def test_from_mapping_requires_identity_and_text(missing: str) -> None:
    data = {"chunk_id": "n1#0", "note_id": "n1", "text": "texto"}
    del data[missing]
    with pytest.raises(ValueError, match=missing):
        LexicalChunk.from_mapping(data)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [("low", 0), ("medium", 1), ("high", 2), (None, None), ("bogus", None)],
)
def test_confidence_rank(confidence: str | None, expected: int | None) -> None:
    """An unrecognised value ranks None rather than failing a whole reindex."""
    chunk = LexicalChunk(chunk_id="n1#0", note_id="n1", text="t", confidence=confidence)
    assert chunk.confidence_rank == expected
