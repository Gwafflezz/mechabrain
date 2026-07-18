"""BM25 lexical index over note chunks, on SQLite FTS5.

The lexical half of hybrid retrieval (§7.1). Built on the ``sqlite3`` module of
the standard library, so it costs the kernel zero dependencies: FTS5 ships
compiled into virtually every SQLite build Python links against.

The index is **derived state** (P1): it lives in ``_meta/index/``, is gitignored
and per-machine, and may be thrown away and rebuilt from the Markdown at any
time. Nothing here is a source of truth, so every failure mode is allowed to
resolve to "delete it and reindex".

Storage layout
--------------
``chunks``
    One row per chunk: the indexed text, the note it came from, and the metadata
    §7.1 filters on (``type``, ``agent``, ``profile``, ``scope``, ``status``,
    ``confidence``). The FTS5 *content table*.
``chunk_tags``
    ``tags`` is multi-valued, so it gets its own table rather than a delimited
    blob that ``LIKE`` would have to guess at.
``chunks_fts``
    An **external-content** FTS5 table (``content='chunks'``) -- terms are
    indexed but the text is not stored twice. Triggers on ``chunks`` keep it in
    sync, which is the pattern the SQLite documentation prescribes.

What gets indexed is :attr:`LexicalChunk.text`, which under Contextual Retrieval
is the chunker's ``embed_text``: the authored chunk behind a deterministic
prefix (scope, title, tags, heading path). The lexical and vector halves index
the same string, so a term in the prefix is findable on both sides.

Portuguese, and every other accented language
---------------------------------------------
The tokenizer is ``unicode61`` with ``remove_diacritics``, which folds accents
on both sides of the match: a query for ``manutencao`` finds ``manutenção`` and
vice versa. ``remove_diacritics 2`` (SQLite >= 3.27) covers codepoints that mode
1 misses; on older SQLite the index falls back to mode 1, which already folds
the whole Latin-1 range and so is exact for pt-BR.

No stemmer is applied. FTS5 ships none for Portuguese, and a wrong-language
stemmer is worse than none -- the vector half of the hybrid carries morphology.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from ..contract import CONFIDENCE_LEVELS
from ..errors import MechabrainIndexError
from ..note import normalize_tags

__all__ = [
    "LexicalIndex",
    "LexicalChunk",
    "LEXICAL_DB_FILENAME",
    "SCHEMA_VERSION",
    "FILTER_KEYS",
]

#: File name under ``_meta/index/``. The caller joins it to the discovered index
#: directory -- this module never builds a path of its own (R4.2).
LEXICAL_DB_FILENAME: Final[str] = "lexical.db"

#: Bumped whenever the schema below changes shape. A database written by a
#: different version is rejected rather than migrated: it is derived state, and
#: `reindex --full` is cheaper and safer than a migration path (P1).
SCHEMA_VERSION: Final[int] = 1

#: Accepted keys of the ``filters`` mapping (§7.1). Anything else is a caller
#: bug and is rejected loudly rather than silently ignored (R5.1) -- a typo'd
#: filter that is dropped would silently widen a scope boundary (R6.5).
FILTER_KEYS: Final[frozenset[str]] = frozenset(
    {"type", "agent", "profile", "scope", "tags", "status", "min_confidence"}
)

#: filter key -> column. ``tags`` and ``min_confidence`` are not plain equality
#: and are handled separately.
_SCALAR_FILTERS: Final[dict[str, str]] = {
    "type": "type",
    "agent": "agent",
    "profile": "profile",
    "scope": "scope",
    "status": "status",
}

#: One term = a run of letters/digits. Deliberately drops every character FTS5
#: would read as syntax, which is what makes free user input safe here.
_TERM_RE: Final[re.Pattern[str]] = re.compile(r"[^\W_]+", re.UNICODE)

#: Mode 2 folds codepoints mode 1 leaves alone; it needs SQLite 3.27 (2019).
_TOKENIZER: Final[str] = (
    "unicode61 remove_diacritics 2"
    if sqlite3.sqlite_version_info >= (3, 27, 0)
    else "unicode61 remove_diacritics 1"
)

_BUSY_TIMEOUT_MS: Final[int] = 30_000

_SCHEMA: Final[str] = f"""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    note_id         TEXT NOT NULL,
    text            TEXT NOT NULL,
    "type"          TEXT,
    agent           TEXT,
    profile         TEXT,
    scope           TEXT,
    status          TEXT,
    confidence      TEXT,
    confidence_rank INTEGER
);

CREATE INDEX IF NOT EXISTS chunks_by_note ON chunks(note_id);

CREATE TABLE IF NOT EXISTS chunk_tags (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    PRIMARY KEY (chunk_id, tag)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS chunk_tags_by_tag ON chunk_tags(tag);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='{_TOKENIZER}'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""


# ══════════════════════════════════════════════════════════════════════
# Input
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class LexicalChunk:
    """One indexable chunk: its text, its origin note, its filterable metadata.

    The input contract of :meth:`LexicalIndex.upsert`. It is deliberately flat
    and stringly-typed: this index filters on metadata but never interprets it,
    so it takes whatever the caller read off the note and does not re-validate
    the §6 schema its callers already own.

    It is a join of two things the caller holds and this module does not:
    a :class:`~mechabrain.index.chunk.Chunk` (text and identity) and the
    frontmatter of the :class:`~mechabrain.note.Note` it came from (metadata).
    Deriving the metadata is the caller's job on purpose -- ``type`` is not even
    a frontmatter key, it follows from the note's folder or its ``mem/`` tag, and
    that mapping is manifest-dependent (§5) rather than something a BM25 index
    should be guessing at.

    Attributes:
        chunk_id: Unique, stable id -- :attr:`Chunk.chunk_id` (``<note_id>#<n>``).
            Re-upserting the same id replaces the row.
        note_id: Basename of the source note, per
            :func:`mechabrain.note.note_id_for`. The unit :meth:`delete` works on.
        text: The text to index. Under Contextual Retrieval this is the chunker's
            ``embed_text`` (context prefix + authored chunk), so that the lexical
            and vector halves index the same string.
        memory_type: The ``type`` filter. Named ``memory_type`` here to avoid
            shadowing the builtin; ``from_mapping`` accepts either spelling.
        confidence: Frontmatter ``confidence``. Ranked via
            :data:`~mechabrain.contract.CONFIDENCE_LEVELS` for ``min_confidence``.
        tags: Tags, normalised by :func:`mechabrain.note.normalize_tags`.
    """

    chunk_id: str
    note_id: str
    text: str
    memory_type: str | None = None
    agent: str | None = None
    profile: str | None = None
    scope: str | None = None
    status: str | None = None
    confidence: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LexicalChunk":
        """Build a chunk from a mapping, e.g. a chunker's ``asdict`` plus metadata.

        Accepts the chunker's own spellings: ``embed_text`` for ``text`` and
        ``type`` for ``memory_type``. ``tags`` takes any shape
        :func:`mechabrain.note.normalize_tags` handles.

        Raises:
            ValueError: ``chunk_id``, ``note_id`` or the text is missing or blank.
        """
        text = data.get("text", data.get("embed_text"))
        memory_type = data.get("memory_type", data.get("type"))
        return cls(
            chunk_id=_require_str(data.get("chunk_id"), "chunk_id"),
            note_id=_require_str(data.get("note_id"), "note_id"),
            text=_require_str(text, "text"),
            memory_type=_optional_str(memory_type),
            agent=_optional_str(data.get("agent")),
            profile=_optional_str(data.get("profile")),
            scope=_optional_str(data.get("scope")),
            status=_optional_str(data.get("status")),
            confidence=_optional_str(data.get("confidence")),
            tags=tuple(normalize_tags(data.get("tags"))),
        )

    @property
    def confidence_rank(self) -> int | None:
        """Position in :data:`~mechabrain.contract.CONFIDENCE_LEVELS`, or ``None``.

        ``None`` for a missing or unrecognised value. Ranking here rather than
        rejecting keeps one malformed note from failing a whole reindex; the cost
        is that such a chunk never satisfies a ``min_confidence`` filter, which is
        the safe direction to fail.
        """
        if self.confidence is None:
            return None
        try:
            return CONFIDENCE_LEVELS.index(self.confidence)
        except ValueError:
            return None


def _require_str(value: Any, key: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"chunk is missing a non-empty {key!r}")
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


# ══════════════════════════════════════════════════════════════════════
# Query
# ══════════════════════════════════════════════════════════════════════
def _match_expression(query: str) -> str | None:
    """Compile free user text into an FTS5 MATCH expression, or ``None``.

    FTS5 MATCH takes a *query language*, not a string: bare input containing
    ``"``, ``*``, ``:``, ``^``, ``(``, ``-`` or ``AND``/``OR``/``NOT``/``NEAR``
    is either a syntax error or, worse, silently means something the user did not
    ask for. Queries here arrive verbatim from an LLM agent, so that input is
    arbitrary.

    The query is therefore not escaped, it is **rebuilt**: keep the alphanumeric
    runs, drop everything else, and emit each term as a quoted FTS5 string --
    quoted so a term that happens to spell ``OR`` stays a search term.

    Terms are joined with ``OR``, not FTS5's implicit ``AND``. A natural-language
    query ("como reconstruir o índice depois de mover a vault") shares few exact
    terms with any one chunk, and under ``AND`` a single absent word returns
    nothing at all -- an empty list is a far worse input to hybrid fusion than a
    ranked one, and BM25 already floats the chunks matching more, and rarer,
    terms to the top.

    Returns:
        The expression, or ``None`` if the query holds no searchable term.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for term in _TERM_RE.findall(query):
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        terms.append(f'"{term}"')
    if not terms:
        return None
    return " OR ".join(terms)


def _as_list(value: Any) -> list[str]:
    """Coerce a filter value to a list of strings.

    ``str`` is checked first: a bare string is one value, never a list of
    characters. :class:`~mechabrain.contract.MemoryType` subclasses ``str``, so
    enum members land here as their wire value with no special case.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def _scalar_clause(column: str, value: Any) -> tuple[str, list[Any]]:
    """Render an equality/IN clause for one scalar filter."""
    values = _as_list(value)
    if not values:
        raise ValueError(f"filter {column!r} is empty; omit it instead of passing an empty list")
    if len(values) == 1:
        return f'c."{column}" = ?', [values[0]]
    placeholders = ", ".join("?" * len(values))
    return f'c."{column}" IN ({placeholders})', list(values)


def _build_filters(filters: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
    """Turn a §7.1 ``filters`` mapping into SQL clauses and bound parameters.

    Raises:
        ValueError: unknown filter key, empty filter value, or a
            ``min_confidence`` outside :data:`CONFIDENCE_LEVELS`.
    """
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(
            f"unknown search filter(s): {', '.join(sorted(unknown))}; "
            f"valid filters are: {', '.join(sorted(FILTER_KEYS))}"
        )

    clauses: list[str] = []
    params: list[Any] = []

    for key, column in _SCALAR_FILTERS.items():
        if filters.get(key) is None:
            continue
        clause, values = _scalar_clause(column, filters[key])
        clauses.append(clause)
        params.extend(values)

    min_confidence = filters.get("min_confidence")
    if min_confidence is not None:
        level = str(min_confidence)
        if level not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"unknown min_confidence {level!r}; "
                f"valid levels are: {', '.join(CONFIDENCE_LEVELS)}"
            )
        # A NULL rank (missing or unrecognised confidence) fails this comparison
        # and is excluded, which is the conservative reading of a floor.
        clauses.append("c.confidence_rank >= ?")
        params.append(CONFIDENCE_LEVELS.index(level))

    tags = normalize_tags(filters.get("tags"))
    if tags:
        # Conjunctive: every listed tag must be present. A filter narrows.
        placeholders = ", ".join("?" * len(tags))
        clauses.append(
            f"(SELECT COUNT(DISTINCT t.tag) FROM chunk_tags t "
            f"WHERE t.chunk_id = c.chunk_id AND t.tag IN ({placeholders})) = ?"
        )
        params.extend(tags)
        params.append(len(tags))

    return clauses, params


# ══════════════════════════════════════════════════════════════════════
# Index
# ══════════════════════════════════════════════════════════════════════
class LexicalIndex:
    """BM25 index over chunks, backed by one SQLite file.

    ::

        with LexicalIndex(paths.index_dir / LEXICAL_DB_FILENAME) as index:
            index.upsert(chunks)
            hits = index.search("reconstruir o índice", k=8,
                                filters={"scope": "proj-a"})

    Opens lazily on first use and creates the file and its parent directory if
    absent -- an empty index is a valid index, being derived state.

    Concurrency: SQLite's own locking is the writer lock here, with WAL so
    readers never block on the writer and a busy timeout so a competing process
    waits rather than raising. This index therefore needs no
    :mod:`mechabrain.locking` file lock of its own; that fallback exists for
    derived state SQLite is not managing (R7.4).

    Not thread-safe: one instance holds one connection, so give each thread its
    own instance.

    Args:
        path: The database file. Always passed in -- the caller derives it from
            :class:`~mechabrain.discovery.VaultPaths`, never this module (R4.2).
    """

    __slots__ = ("path", "_connection", "_in_transaction")

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._in_transaction = False

    # ── Lifecycle ───────────────────────────────────────────────────
    @property
    def connection(self) -> sqlite3.Connection:
        """The open connection, opening and creating the schema on first touch."""
        if self._connection is None:
            self._connection = self._open()
        return self._connection

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self.path, isolation_level=None)
        except sqlite3.Error as exc:
            raise MechabrainIndexError(
                f"cannot open the lexical index at {self.path}: {exc}",
                hint="check the directory is writable, or run `mechabrain reindex --full`",
            ) from exc

        try:
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA foreign_keys = ON")
            self._check_schema_version(connection)
            connection.executescript(_SCHEMA)
        except sqlite3.OperationalError as exc:
            connection.close()
            raise self._schema_error(exc) from exc
        except BaseException:
            connection.close()
            raise
        return connection

    def _check_schema_version(self, connection: sqlite3.Connection) -> None:
        """Stamp ``user_version``, or refuse a database this kernel did not write.

        The caller owns ``connection`` and closes it if this raises.
        """
        found = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if found == 0:
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if found != SCHEMA_VERSION:
            raise MechabrainIndexError(
                f"the lexical index at {self.path} has schema version {found}, "
                f"but this kernel writes version {SCHEMA_VERSION}",
                hint="the index is derived state: delete it and run `mechabrain reindex --full`",
            )

    def _schema_error(self, exc: sqlite3.OperationalError) -> MechabrainIndexError:
        if "fts5" in str(exc).lower():
            return MechabrainIndexError(
                f"this Python's SQLite has no FTS5 module, so the BM25 index at "
                f"{self.path} cannot be created: {exc}",
                hint=(
                    "FTS5 is a compile-time SQLite option (SQLITE_ENABLE_FTS5). "
                    "Install a Python built against an FTS5-enabled SQLite."
                ),
            )
        return MechabrainIndexError(
            f"cannot create the lexical index schema at {self.path}: {exc}",
            hint="the index is derived state: delete it and run `mechabrain reindex --full`",
        )

    def close(self) -> None:
        """Close the connection. Idempotent; the instance reopens on next use.

        Closing inside an open :meth:`transaction` rolls it back -- that is the
        crash-consistent reading: an unfinished batch never half-lands.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._in_transaction = False

    def __enter__(self) -> "LexicalIndex":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self._connection is not None else "closed"
        return f"LexicalIndex({str(self.path)!r}, {state})"

    # ── Writing ─────────────────────────────────────────────────────
    def upsert(self, chunks: Iterable[LexicalChunk | Mapping[str, Any]]) -> int:
        """Insert or replace ``chunks`` by ``chunk_id``, in one transaction.

        Replace is an explicit DELETE followed by an INSERT rather than
        ``INSERT OR REPLACE``: REPLACE only fires delete triggers when recursive
        triggers are on, so it would leave the FTS index holding the superseded
        terms forever.

        Args:
            chunks: :class:`LexicalChunk` instances, or mappings that
                :meth:`LexicalChunk.from_mapping` accepts.

        Returns:
            The number of chunks written.

        Raises:
            ValueError: a chunk is missing ``chunk_id``, ``note_id`` or text.
            TypeError: a chunk is neither a :class:`LexicalChunk` nor a mapping.
            MechabrainIndexError: the write failed.
        """
        rows = [_coerce(chunk) for chunk in chunks]
        if not rows:
            return 0

        with self._transaction() as connection:
            connection.executemany(
                "DELETE FROM chunks WHERE chunk_id = ?",
                [(row.chunk_id,) for row in rows],
            )
            connection.executemany(
                'INSERT INTO chunks (chunk_id, note_id, text, "type", agent, profile, '
                "scope, status, confidence, confidence_rank) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        row.chunk_id,
                        row.note_id,
                        row.text,
                        row.memory_type,
                        row.agent,
                        row.profile,
                        row.scope,
                        row.status,
                        row.confidence,
                        row.confidence_rank,
                    )
                    for row in rows
                ],
            )
            connection.executemany(
                "INSERT INTO chunk_tags (chunk_id, tag) VALUES (?, ?)",
                [(row.chunk_id, tag) for row in rows for tag in dict.fromkeys(row.tags)],
            )
        return len(rows)

    def delete(self, note_ids: Iterable[str]) -> int:
        """Drop every chunk belonging to ``note_ids``.

        The unit a caller actually has: a note was edited or removed, and all of
        its chunks are stale. Unknown ids are not an error -- deleting what is
        already absent is the requested end state.

        Returns:
            The number of chunks removed.
        """
        ids = list(dict.fromkeys(note_ids))
        if not ids:
            return 0
        placeholders = ", ".join("?" * len(ids))
        with self._transaction() as connection:
            cursor = connection.execute(
                f"DELETE FROM chunks WHERE note_id IN ({placeholders})", ids
            )
            return int(cursor.rowcount)

    def clear(self) -> int:
        """Empty the index, keeping the schema.

        Deletes through ``chunks`` so the triggers unindex each row, rather than
        FTS5's ``delete-all``: the two tables must not disagree afterwards, and
        it is one statement either way.

        Returns:
            The number of chunks removed.
        """
        with self._transaction() as connection:
            cursor = connection.execute("DELETE FROM chunks")
            return int(cursor.rowcount)

    # ── Reading ─────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank chunks against ``query`` by BM25.

        Scores are ``-bm25()``: FTS5 returns BM25 negated, so its smallest value
        is the best match. Negating restores "larger is better" and is monotone,
        which is all the §7.1 fusion needs before it min-max normalises this list
        against the vector list. The result is also strictly positive -- FTS5
        floors a term's IDF at 1e-6 rather than letting it go negative for a term
        appearing in most of the corpus, so ``bm25()`` never returns >= 0.

        Args:
            query: Free text. Rebuilt into an FTS5 expression -- see
                :func:`_match_expression`. No syntax is honoured: a user typing
                ``NOT`` searches for the word "not".
            k: Maximum hits.
            filters: §7.1 filters -- ``type``, ``agent``, ``profile``, ``scope``,
                ``tags``, ``status``, ``min_confidence``. Scalars take one value
                or a list (OR within the key); ``tags`` requires *all* listed
                tags; ``min_confidence`` is a floor on
                :data:`~mechabrain.contract.CONFIDENCE_LEVELS`. Keys are AND-ed.

        Returns:
            ``(chunk_id, score)`` best first, ties broken by ``chunk_id`` so the
            order is total and reproducible. Empty if the query holds no
            searchable term.

        Raises:
            ValueError: ``k`` < 1, or ``filters`` is invalid.
            MechabrainIndexError: the query failed against the index.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        match = _match_expression(query)
        if match is None:
            return []

        clauses, params = _build_filters(filters or {})
        where = " AND ".join(["chunks_fts MATCH ?", *clauses])
        sql = (
            "SELECT c.chunk_id, -bm25(chunks_fts) AS score "
            "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            f"WHERE {where} "
            "ORDER BY score DESC, c.chunk_id ASC LIMIT ?"
        )
        rows = self._query(sql, [match, *params, k])
        return [(str(chunk_id), float(score)) for chunk_id, score in rows]

    def note_ids(self) -> set[str]:
        """Every note id with at least one chunk indexed.

        What an incremental reindex diffs the vault against, to find notes whose
        file is gone and whose chunks must be dropped.
        """
        return {str(row[0]) for row in self._query("SELECT DISTINCT note_id FROM chunks", [])}

    def count(self) -> int:
        """Number of chunks indexed. Feeds ``memory_status`` (§7.1)."""
        return int(self._query("SELECT COUNT(*) FROM chunks", [])[0][0])

    def _query(self, sql: str, params: Sequence[Any]) -> list[tuple[Any, ...]]:
        try:
            return self.connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise MechabrainIndexError(
                f"lexical index query failed against {self.path}: {exc}",
                hint="the index is derived state: run `mechabrain reindex --full`",
            ) from exc

    # ── Transactions ────────────────────────────────────────────────
    def transaction(self) -> "_Transaction":
        """One write transaction spanning any number of operations.

        ::

            with index.transaction():
                index.clear()
                index.upsert(chunks)

        Nothing is committed until the block exits cleanly; any exception rolls
        the whole batch back, leaving the previous contents intact. This is what
        makes a rebuild atomic: a process that dies between ``clear()`` and the
        last ``upsert()`` never destroys the index it meant to replace (SQLite
        rolls an uncommitted transaction back on the next open).

        Reentrant: ``upsert``/``delete``/``clear`` each open their own
        transaction when called bare, and join the enclosing one when called
        inside this block -- only the outermost block commits or rolls back.
        """
        return _Transaction(self)

    def _transaction(self) -> "_Transaction":
        return _Transaction(self)


class _Transaction:
    """One explicit write transaction, rolled back on any error.

    The connection runs in autocommit (``isolation_level=None``) so that DDL and
    PRAGMAs behave; writes therefore state their own BEGIN/COMMIT rather than
    relying on sqlite3's implicit-transaction heuristics. Nested instances join
    the outermost transaction rather than issuing a BEGIN SQLite would reject.
    """

    __slots__ = ("_index", "_connection", "_nested")

    def __init__(self, index: LexicalIndex) -> None:
        self._index = index
        self._connection: sqlite3.Connection | None = None
        self._nested = False

    def __enter__(self) -> sqlite3.Connection:
        connection = self._index.connection
        if self._index._in_transaction:
            self._nested = True
        else:
            connection.execute("BEGIN IMMEDIATE")
            self._index._in_transaction = True
        self._connection = connection
        return connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        connection = self._connection
        if connection is None or self._nested:
            return False
        self._index._in_transaction = False
        try:
            if exc_type is None:
                connection.execute("COMMIT")
            else:
                connection.execute("ROLLBACK")
        except sqlite3.Error as error:
            if exc_type is None:
                raise MechabrainIndexError(
                    f"lexical index write failed against {self._index.path}: {error}",
                    hint="the index is derived state: run `mechabrain reindex --full`",
                ) from error
        return False


def _coerce(chunk: LexicalChunk | Mapping[str, Any]) -> LexicalChunk:
    if isinstance(chunk, LexicalChunk):
        return chunk
    if isinstance(chunk, Mapping):
        return LexicalChunk.from_mapping(chunk)
    raise TypeError(f"expected a LexicalChunk or a mapping, got {type(chunk).__name__}")
