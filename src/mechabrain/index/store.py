"""Pluggable vector store for the derived index (`retrieval.store`, §5/§7.1).

Everything here belongs to the **runtime** layer (§4): state lives under
``mecha-brain/_meta/index/``, is gitignored, per-machine, and is always
rebuildable from the Markdown (P1). That location is enforced, not merely
documented -- see :func:`require_index_dir`.

Three implementations behind one :class:`VectorStore` protocol:

===============  =========================================================
``numpy``        Default. Brute-force cosine over an in-memory ``float32``
                 matrix persisted as ``vectors.npy`` + ``vectors.jsonl``.
                 No dependency beyond numpy, which is core.
``lancedb``      Optional extra ``mechabrain[lancedb]``.
``sqlite-vec``   Optional extra ``mechabrain[sqlite-vec]``.
===============  =========================================================

**Why brute force is the default.** At personal-vault scale (~1e3 notes, ~1e4
chunks) a dot product against a ``(1e4, 1024)`` matrix is a few milliseconds --
one BLAS call. An ANN index buys nothing measurable there and costs a heavy
dependency plus a second consistency problem. The optional stores exist for the
day a deployment outgrows that, and the protocol is the seam.

**Scores.** ``search`` returns raw cosine similarity in ``[-1, 1]``, highest
first, never a distance. Fusion with BM25 min-max normalises each list before
applying the manifest weights, so the store must not normalise or clamp on its
own -- that would destroy the spread the fusion needs.

**Filters.** The filter language here is deliberately generic: the store knows
nothing of §6 frontmatter (R4.1). It matches ``{key: value}`` and
``{key: [v1, v2]}`` (OR) against stored metadata, with list-valued metadata
matching on intersection so ``tags`` works. Anything *ordered* -- notably the
``min_confidence`` of §7.1 -- is the retrieval layer's job to lower into this
language (``min_confidence: medium`` -> ``confidence: ["medium", "high"]`` via
``contract.CONFIDENCE_LEVELS``) before it gets here.

**Locking.** No store takes a lock. Writes are atomic per file, but a
read-modify-write cycle is a *decision* and its transaction boundary belongs to
the caller, which is the only party that knows where the cycle starts and ends.
Callers hold ``locking.file_lock(paths.index_dir / INDEX_LOCK_FILE)`` around the
whole cycle (R7.4). A store that grabbed the lock internally would also deadlock
any caller already holding it -- ``FileLock`` is not reentrant across instances.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..contract import CONFIG_FILE, INDEX_DIR, META_DIR
from ..discovery import VaultPaths
from ..errors import MechabrainIndexError
from ..manifest import Manifest

__all__ = [
    "VectorStore",
    "NumpyStore",
    "LanceDBStore",
    "SqliteVecStore",
    "from_manifest",
    "open_store",
    "matches_filters",
    "require_index_dir",
    "INDEX_LOCK_FILE",
    "VECTORS_FILE",
    "METAS_FILE",
    "LANCEDB_DIR",
    "LANCEDB_TABLE",
    "SQLITE_VEC_FILE",
]

#: Lock every writer of the derived index takes (R7.4 fallback). Named here so
#: the store, the reindexer and the daemon agree on one file without importing
#: each other.
INDEX_LOCK_FILE: Final[str] = "index.lock"

#: `numpy` store state, both under `_meta/index/`.
VECTORS_FILE: Final[str] = "vectors.npy"
METAS_FILE: Final[str] = "vectors.jsonl"

#: `lancedb` store state.
LANCEDB_DIR: Final[str] = "lancedb"
LANCEDB_TABLE: Final[str] = "chunks"

#: `sqlite-vec` store state.
SQLITE_VEC_FILE: Final[str] = "vectors.sqlite3"

#: Growth factor when a store that can only post-filter has to over-fetch.
_OVERFETCH: Final[int] = 8

_REBUILD_HINT: Final[str] = "run `mechabrain reindex --full`; the index is derived and always rebuildable"

Meta = Mapping[str, Any]
Filters = Mapping[str, Any]
#: One search result: the id given at upsert, and cosine similarity.
Hit = tuple[str, float]


# ══════════════════════════════════════════════════════════════════════
# Protocol
# ══════════════════════════════════════════════════════════════════════
@runtime_checkable
class VectorStore(Protocol):
    """What the retrieval layer may assume of any vector backend.

    Implementations are free about *where inside* ``_meta/index/`` they keep
    state and in what format, and about whether they filter before or after the
    nearest-neighbour scan -- but not about the semantics below.
    """

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Any,
        metas: Sequence[Meta],
    ) -> int:
        """Insert or replace ``len(ids)`` rows and return how many were written.

        Replacement is by ``id``: re-upserting an id overwrites its vector and
        its metadata wholesale (never merges). Ids repeated within one batch
        collapse to the last occurrence.

        Args:
            ids: Non-empty, unique-by-last-occurrence row keys.
            vectors: Array-like of shape ``(len(ids), dim)``. Coerced to
                ``float32`` and L2-normalised; magnitude is never significant.
            metas: One JSON-serialisable mapping per id. ``date``/``datetime``
                values are coerced to ISO strings, because filters arrive over
                the MCP wire as JSON and must compare equal to what is stored.
        """

    def delete(self, ids: Sequence[str]) -> int:
        """Remove rows by id and return how many existed. Unknown ids are not an error."""

    def search(self, vector: Any, k: int = 8, filters: Filters | None = None) -> list[Hit]:
        """Return up to ``k`` ``(id, cosine)`` pairs, best first.

        ``filters`` constrains *before* ranking: ``k`` counts matching rows, so
        a filter can never be starved by a non-matching neighbour. See
        :func:`matches_filters` for the language.
        """

    def clear(self) -> None:
        """Drop every row. The store stays usable and may be re-dimensioned."""

    def count(self) -> int:
        """Number of rows currently stored."""

    def flush(self) -> None:
        """Make pending writes durable. A no-op for stores that commit eagerly.

        Beyond the five methods the design called for, because the default store
        holds one file for the whole matrix: rewriting it per row would turn a
        rebuild into O(n) full-matrix writes. Bulk callers construct with
        ``autosave=False`` and flush once.
        """


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════
def require_index_dir(directory: Path | str) -> Path:
    """Return ``directory``, or refuse it if it is not a ``_meta/index/``.

    The §4 layer table is normative: derived index state is per-machine and
    gitignored, and the vault's ``.gitignore`` only covers
    ``mecha-brain/_meta/index/``. A store writing anywhere else would get
    committed and synced to other machines -- silently, and wrongly, since a
    vector index is not portable across embedding models.

    Checked by path *shape*, never against a hardcoded location (R4.1): the
    folder must be named ``index`` inside a folder named ``_meta``. Existence is
    not required -- ``mechabrain init`` creates it, and stores create it on
    first write.
    """
    path = Path(directory)
    if path.name != INDEX_DIR or path.parent.name != META_DIR:
        raise MechabrainIndexError(
            f"a vector store may only persist inside {META_DIR}/{INDEX_DIR}/, got {path}",
            rule="§4",
            hint="pass VaultPaths.index_dir; it is the only gitignored, per-machine location",
        )
    return path


def matches_filters(meta: Meta, filters: Filters | None) -> bool:
    """Whether ``meta`` satisfies every constraint in ``filters``.

    The language, in full:

    * ``{key: value}`` -- ``meta[key]`` equals ``value``, or *contains* it when
      ``meta[key]`` is a list (so ``{"tags": "mem/semantic"}`` works);
    * ``{key: [a, b]}`` -- any of them match (OR). Against a list-valued
      ``meta[key]`` this is set intersection;
    * ``{key: None}`` -- ignored, so a caller may pass an optional filter
      through without stripping it first;
    * a key absent from ``meta`` -- no match. Never a wildcard.

    Constraints AND together. Values compare after ``str()``, since a filter
    crossing MCP is JSON and a stored ``date`` was coerced to ISO on upsert.

    Exported because the BM25 half of the hybrid must apply exactly these
    semantics to exactly this metadata, and two implementations would drift.

    Raises:
        ValueError: a filter value is an empty sequence -- unsatisfiable by
            construction, so it is a caller bug rather than "match nothing".
    """
    if not filters:
        return True
    for key, wanted in filters.items():
        if wanted is None:
            continue
        if isinstance(wanted, (list, tuple, set, frozenset)):
            if not wanted:
                raise ValueError(
                    f"filter {key!r} is an empty sequence, which nothing can satisfy; "
                    f"omit the key to leave {key!r} unconstrained"
                )
            candidates = {_scalar(item) for item in wanted}
        else:
            candidates = {_scalar(wanted)}
        if key not in meta:
            return False
        stored = meta[key]
        if isinstance(stored, (list, tuple, set, frozenset)):
            if not {_scalar(item) for item in stored} & candidates:
                return False
        elif _scalar(stored) not in candidates:
            return False
    return True


def _scalar(value: Any) -> str:
    """Render a filter/metadata leaf for comparison. See :func:`matches_filters`."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _json_default(value: Any) -> Any:
    """Coerce what the note layer legitimately produces; refuse the rest loudly."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        raise TypeError(
            f"metadata holds a Path ({value!r}); store vault-relative strings "
            f"instead -- an absolute path must never reach the index (R4.2)"
        )
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON-serialisable: {value!r}")


def _dump_meta(meta: Meta, item_id: str) -> dict[str, Any]:
    """Round-trip ``meta`` through JSON so what is stored is what is compared."""
    try:
        return json.loads(json.dumps(dict(meta), default=_json_default, ensure_ascii=False))
    except TypeError as exc:
        raise MechabrainIndexError(
            f"metadata for {item_id!r} is not JSON-serialisable: {exc}",
            rule="P1",
            hint="index metadata is a projection of the note's frontmatter; keep it to "
            "strings, numbers, booleans, dates and lists of those",
        ) from exc


def _as_matrix(vectors: Any, expected_rows: int) -> NDArray[np.float32]:
    """Coerce ``vectors`` to an L2-normalised ``(expected_rows, dim)`` float32 matrix.

    Normalising at write time makes cosine a plain dot product at read time --
    the whole reason brute force is fast enough to be the default. A zero vector
    keeps its zeros and scores 0.0 against everything, rather than yielding NaN.
    """
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MechabrainIndexError(
            f"vectors are not a numeric array: {exc}",
            rule="P1",
            hint="pass a 2-D array-like of shape (len(ids), dim)",
        ) from exc
    if matrix.ndim != 2:
        raise MechabrainIndexError(
            f"vectors must be 2-D of shape (len(ids), dim), got shape {matrix.shape}",
            rule="P1",
            hint="a single vector still needs an outer dimension: vector.reshape(1, -1)",
        )
    if matrix.shape[0] != expected_rows:
        raise MechabrainIndexError(
            f"got {expected_rows} ids but {matrix.shape[0]} vectors",
            rule="P1",
        )
    if matrix.shape[1] == 0:
        raise MechabrainIndexError("vectors have dimension 0", rule="P1")
    if not np.isfinite(matrix).all():
        raise MechabrainIndexError(
            "vectors contain NaN or infinity",
            rule="R5.1",
            hint="the embedding provider returned a broken vector; do not index it",
        )
    return _l2_normalize(matrix)


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _as_query(vector: Any) -> NDArray[np.float32]:
    """Coerce a query to a 1-D L2-normalised float32 vector."""
    try:
        query = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise MechabrainIndexError(f"query vector is not numeric: {exc}", rule="P1") from exc
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1:
        raise MechabrainIndexError(
            f"query vector must be 1-D, got shape {query.shape}",
            rule="P1",
        )
    if not np.isfinite(query).all():
        raise MechabrainIndexError("query vector contains NaN or infinity", rule="R5.1")
    return _l2_normalize(query.reshape(1, -1))[0]


def _dim_mismatch(stored: int, given: int) -> MechabrainIndexError:
    return MechabrainIndexError(
        f"vector dimension {given} does not match the {stored} of the existing index",
        rule="R7.4",
        hint=f"`retrieval.embedding.model` changed, or two providers are mixed. {_REBUILD_HINT}",
    )


def _clean_ids(ids: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    for position, raw in enumerate(ids):
        if not isinstance(raw, str) or not raw.strip():
            raise MechabrainIndexError(
                f"id at position {position} is not a non-empty string: {raw!r}",
                rule="P1",
            )
        cleaned.append(raw)
    return cleaned


def _prepare_batch(
    ids: Sequence[str],
    vectors: Any,
    metas: Sequence[Meta],
) -> tuple[list[str], NDArray[np.float32], list[dict[str, Any]]]:
    """Validate a batch and collapse ids repeated inside it to their last row."""
    id_list = _clean_ids(ids)
    meta_list = list(metas)
    if len(meta_list) != len(id_list):
        raise MechabrainIndexError(
            f"got {len(id_list)} ids but {len(meta_list)} metadata mappings",
            rule="P1",
        )
    matrix = _as_matrix(vectors, len(id_list))

    last_row: dict[str, int] = {}
    for row, item_id in enumerate(id_list):
        last_row[item_id] = row
    keep = sorted(last_row.values())
    if len(keep) != len(id_list):
        id_list = [id_list[row] for row in keep]
        meta_list = [meta_list[row] for row in keep]
        matrix = matrix[keep]

    dumped = [_dump_meta(meta, item_id) for item_id, meta in zip(id_list, meta_list)]
    return id_list, matrix, dumped


def _rank(
    scores: NDArray[np.float32],
    ids: Sequence[str],
    k: int,
) -> list[int]:
    """Indices of the ``k`` best scores, ties broken by id so results are stable.

    Without the tie-break, equal scores would come out in insertion order, which
    depends on the order notes happened to be indexed in -- reproducible on one
    machine, not across two. `lexsort` puts the last key first in precedence.
    """
    if scores.size == 0:
        return []
    order = np.lexsort((np.asarray(ids, dtype=object), -scores))
    return [int(index) for index in order[:k]]


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Atomic byte write (R7.5): temp file beside the target, fsync, ``os.replace``.

    :func:`mechabrain.note.write_atomic` is the text twin; ``.npy`` is binary and
    a text codec would corrupt it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _require_module(module_name: str, extra: str) -> Any:
    """Import an optional backend, or say exactly how to get it."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MechabrainIndexError(
            f"vector store {extra!r} needs the optional dependency {module_name!r}, "
            f"which is not installed",
            rule="R5.1",
            hint=(
                f"install it (`uv pip install 'mechabrain[{extra}]'`), or set "
                f"`retrieval.store: numpy` in {META_DIR}/{CONFIG_FILE} -- the default "
                f"store needs no extra and is faster below ~1e4 chunks"
            ),
        ) from exc


# ══════════════════════════════════════════════════════════════════════
# numpy -- the default
# ══════════════════════════════════════════════════════════════════════
class NumpyStore:
    """Brute-force cosine over a ``float32`` matrix. The default (§5).

    State is two files in ``_meta/index/``: ``vectors.npy`` (the L2-normalised
    matrix, row-aligned) and ``vectors.jsonl`` (one ``{"id", "meta"}`` object per
    row, same order). Both are rewritten whole on every flush -- at ~1e4 chunks
    that is a ~40 MB write, cheap next to the embedding pass that produced it,
    and it keeps the format trivially inspectable.

    The two files are written in sequence, so a crash between them leaves them
    inconsistent. That is detected on load (row count vs id count) and raises
    with the rebuild hint rather than serving a wrong id for a vector: the index
    is derived, so a torn write costs a reindex, never data (P1).

    Args:
        index_dir: A ``_meta/index/``. See :func:`require_index_dir`.
        autosave: Persist after each mutation. Set ``False`` for a bulk rebuild
            and call :meth:`flush` once at the end.
    """

    def __init__(self, index_dir: Path | str, *, autosave: bool = True) -> None:
        self.index_dir: Final[Path] = require_index_dir(index_dir)
        self.vectors_path: Final[Path] = self.index_dir / VECTORS_FILE
        self.metas_path: Final[Path] = self.index_dir / METAS_FILE
        self.autosave = autosave
        self._matrix: NDArray[np.float32] = np.zeros((0, 0), dtype=np.float32)
        self._ids: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._positions: dict[str, int] = {}
        self._loaded = False

    def __repr__(self) -> str:
        return f"NumpyStore({str(self.index_dir)!r}, rows={self.count()})"

    # ── Reads ───────────────────────────────────────────────────────
    def count(self) -> int:
        self._load()
        return len(self._ids)

    def search(self, vector: Any, k: int = 8, filters: Filters | None = None) -> list[Hit]:
        self._load()
        if k <= 0 or not self._ids:
            return []
        query = _as_query(vector)
        if query.shape[0] != self._matrix.shape[1]:
            raise _dim_mismatch(self._matrix.shape[1], query.shape[0])

        rows = self._filtered_rows(filters)
        if not rows:
            return []
        scores = self._matrix[rows] @ query
        candidate_ids = [self._ids[row] for row in rows]
        return [
            (candidate_ids[index], float(scores[index]))
            for index in _rank(scores, candidate_ids, k)
        ]

    def _filtered_rows(self, filters: Filters | None) -> list[int]:
        """Row indices surviving ``filters`` -- computed before ranking, per §7.1."""
        if not filters:
            return list(range(len(self._ids)))
        return [row for row, meta in enumerate(self._metas) if matches_filters(meta, filters)]

    # ── Writes ──────────────────────────────────────────────────────
    def upsert(self, ids: Sequence[str], vectors: Any, metas: Sequence[Meta]) -> int:
        id_list, matrix, meta_list = _prepare_batch(ids, vectors, metas)
        if not id_list:
            return 0
        self._load()
        self._adopt_dim(matrix.shape[1])

        existing = [(self._positions[i], row) for row, i in enumerate(id_list) if i in self._positions]
        fresh = [row for row, i in enumerate(id_list) if i not in self._positions]

        if existing:
            positions = np.fromiter((p for p, _ in existing), dtype=np.intp, count=len(existing))
            rows = np.fromiter((r for _, r in existing), dtype=np.intp, count=len(existing))
            self._matrix[positions] = matrix[rows]
            for position, row in existing:
                self._metas[position] = meta_list[row]

        if fresh:
            self._matrix = np.vstack([self._matrix, matrix[fresh]])
            for row in fresh:
                self._positions[id_list[row]] = len(self._ids)
                self._ids.append(id_list[row])
                self._metas.append(meta_list[row])

        self._save_if_autosaving()
        return len(id_list)

    def delete(self, ids: Sequence[str]) -> int:
        self._load()
        doomed = {item_id for item_id in ids if item_id in self._positions}
        if not doomed:
            return 0
        keep = [row for row, item_id in enumerate(self._ids) if item_id not in doomed]
        self._matrix = self._matrix[keep]
        self._ids = [self._ids[row] for row in keep]
        self._metas = [self._metas[row] for row in keep]
        self._reindex_positions()
        self._save_if_autosaving()
        return len(doomed)

    def clear(self) -> None:
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._ids = []
        self._metas = []
        self._positions = {}
        self._loaded = True
        self._save_if_autosaving()

    def flush(self) -> None:
        """Write both files atomically (R7.5). Safe to call when nothing changed."""
        self._load()
        _write_bytes_atomic(self.vectors_path, self._encode_matrix())
        lines = "".join(
            json.dumps({"id": item_id, "meta": meta}, ensure_ascii=False) + "\n"
            for item_id, meta in zip(self._ids, self._metas)
        )
        _write_bytes_atomic(self.metas_path, lines.encode("utf-8"))

    def _encode_matrix(self) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, self._matrix, allow_pickle=False)
        return buffer.getvalue()

    def _save_if_autosaving(self) -> None:
        if self.autosave:
            self.flush()

    # ── State ───────────────────────────────────────────────────────
    def _adopt_dim(self, dim: int) -> None:
        """Fix the store's dimension on first write; refuse a later change."""
        if self._matrix.shape[1] == dim:
            return
        if len(self._ids) == 0:
            self._matrix = np.zeros((0, dim), dtype=np.float32)
            return
        raise _dim_mismatch(self._matrix.shape[1], dim)

    def _reindex_positions(self) -> None:
        self._positions = {item_id: row for row, item_id in enumerate(self._ids)}

    def _load(self) -> None:
        """Read both files once, or start empty. Idempotent."""
        if self._loaded:
            return
        self._loaded = True
        if not (self.vectors_path.is_file() and self.metas_path.is_file()):
            return
        self._matrix = self._read_matrix()
        self._ids, self._metas = self._read_metas()
        if self._matrix.shape[0] != len(self._ids):
            raise self._corrupt(
                f"{VECTORS_FILE} holds {self._matrix.shape[0]} rows but "
                f"{METAS_FILE} holds {len(self._ids)}"
            )
        self._reindex_positions()
        if len(self._positions) != len(self._ids):
            raise self._corrupt(f"{METAS_FILE} repeats an id")

    def _read_matrix(self) -> NDArray[np.float32]:
        try:
            matrix = np.load(self.vectors_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise self._corrupt(f"{VECTORS_FILE} is unreadable: {exc}") from exc
        if matrix.ndim != 2:
            raise self._corrupt(f"{VECTORS_FILE} is not a 2-D matrix (shape {matrix.shape})")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _read_metas(self) -> tuple[list[str], list[dict[str, Any]]]:
        ids: list[str] = []
        metas: list[dict[str, Any]] = []
        try:
            text = self.metas_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise self._corrupt(f"{METAS_FILE} is unreadable: {exc}") from exc
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                ids.append(str(record["id"]))
                metas.append(dict(record["meta"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise self._corrupt(f"{METAS_FILE} line {number} is malformed: {exc}") from exc
        return ids, metas

    def _corrupt(self, problem: str) -> MechabrainIndexError:
        return MechabrainIndexError(
            f"the index in {self.index_dir} is corrupt: {problem}",
            rule="R7.4",
            hint=_REBUILD_HINT,
        )


# ══════════════════════════════════════════════════════════════════════
# lancedb -- optional
# ══════════════════════════════════════════════════════════════════════
class LanceDBStore:
    """LanceDB backend (extra ``mechabrain[lancedb]``), rooted in ``_meta/index/lancedb/``.

    Metadata rides along as one JSON column and filters are applied **after** the
    ANN scan, with geometric over-fetch until ``k`` matches are found or the
    table is exhausted -- so results are correct, at worst a full scan. Pushing
    filters down into LanceDB's SQL would mean promoting every filterable
    frontmatter key to a typed column, i.e. teaching the kernel the §6 field
    names it is forbidden to know (R4.1), or inferring a schema from whatever
    the first batch happened to carry. Neither is worth it while the default
    store filters exactly and needs no schema at all.
    """

    def __init__(self, index_dir: Path | str, *, table_name: str = LANCEDB_TABLE) -> None:
        self.index_dir: Final[Path] = require_index_dir(index_dir)
        self.db_path: Final[Path] = self.index_dir / LANCEDB_DIR
        self.table_name = table_name
        self._db: Any = None

    def __repr__(self) -> str:
        return f"LanceDBStore({str(self.index_dir)!r}, table={self.table_name!r})"

    def _connect(self) -> Any:
        if self._db is None:
            lancedb = _require_module("lancedb", "lancedb")
            self.db_path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _table(self) -> Any:
        """The table, or ``None`` before the first upsert creates it.

        Asked for by opening it rather than by listing the database: the listing
        call has been renamed and re-paginated across LanceDB releases, while
        ``open_table`` raising on a missing table has been stable throughout.
        """
        db = self._connect()
        try:
            return db.open_table(self.table_name)
        except (FileNotFoundError, ValueError):
            return None

    def upsert(self, ids: Sequence[str], vectors: Any, metas: Sequence[Meta]) -> int:
        id_list, matrix, meta_list = _prepare_batch(ids, vectors, metas)
        if not id_list:
            return 0
        rows = [
            {"id": item_id, "vector": vector.tolist(), "meta": json.dumps(meta, ensure_ascii=False)}
            for item_id, vector, meta in zip(id_list, matrix, meta_list)
        ]
        table = self._table()
        if table is None:
            self._connect().create_table(self.table_name, data=rows)
            return len(id_list)
        self._check_dim(table, matrix.shape[1])
        self._delete_where(table, id_list)
        table.add(rows)
        return len(id_list)

    def delete(self, ids: Sequence[str]) -> int:
        table = self._table()
        if table is None:
            return 0
        present = [item_id for item_id in dict.fromkeys(ids) if self._exists(table, item_id)]
        if present:
            self._delete_where(table, present)
        return len(present)

    def search(self, vector: Any, k: int = 8, filters: Filters | None = None) -> list[Hit]:
        table = self._table()
        if k <= 0 or table is None:
            return []
        query = _as_query(vector)
        self._check_dim(table, query.shape[0])
        total = table.count_rows()
        if total == 0:
            return []

        limit = k if not filters else min(k * _OVERFETCH, total)
        while True:
            hits = [
                (str(row["id"]), 1.0 - float(row["_distance"]))
                for row in table.search(query).metric("cosine").limit(limit).to_list()
                if matches_filters(json.loads(row["meta"]), filters)
            ]
            if len(hits) >= k or limit >= total:
                return hits[:k]
            limit = min(limit * _OVERFETCH, total)

    def clear(self) -> None:
        if self._table() is None:
            return
        self._connect().drop_table(self.table_name)

    def count(self) -> int:
        table = self._table()
        return 0 if table is None else int(table.count_rows())

    def flush(self) -> None:
        """No-op: LanceDB commits each operation."""

    def _check_dim(self, table: Any, dim: int) -> None:
        field = table.schema.field("vector")
        stored = getattr(field.type, "list_size", None)
        if stored is not None and int(stored) != dim:
            raise _dim_mismatch(int(stored), dim)

    def _exists(self, table: Any, item_id: str) -> bool:
        return bool(table.search().where(f"id = {_sql_quote(item_id)}").limit(1).to_list())

    def _delete_where(self, table: Any, ids: Sequence[str]) -> None:
        table.delete(f"id IN ({', '.join(_sql_quote(item_id) for item_id in ids)})")


def _sql_quote(value: str) -> str:
    """Single-quote a literal for LanceDB's SQL filter, doubling embedded quotes."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


# ══════════════════════════════════════════════════════════════════════
# sqlite-vec -- optional
# ══════════════════════════════════════════════════════════════════════
class SqliteVecStore:
    """sqlite-vec backend (extra ``mechabrain[sqlite-vec]``), one file in ``_meta/index/``.

    Rows live in two tables sharing a rowid: a ``vec0`` virtual table for the
    vectors and an ordinary table for id + JSON metadata. Filters are
    post-applied with over-fetch, for the reason given on :class:`LanceDBStore`.

    ``vec0`` is queried with its default L2 metric and the distance is converted
    back to cosine exactly -- for L2-normalised vectors ``d^2 = 2 - 2cos``, so
    ``cos = 1 - d^2/2``. That avoids depending on the ``distance_metric=cosine``
    declaration, which only exists in newer sqlite-vec builds.
    """

    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir: Final[Path] = require_index_dir(index_dir)
        self.db_path: Final[Path] = self.index_dir / SQLITE_VEC_FILE
        self._conn: sqlite3.Connection | None = None

    def __repr__(self) -> str:
        return f"SqliteVecStore({str(self.index_dir)!r})"

    def _connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        sqlite_vec = _require_module("sqlite_vec", "sqlite-vec")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError) as exc:
            conn.close()
            raise MechabrainIndexError(
                f"this Python's sqlite3 cannot load the sqlite-vec extension: {exc}",
                rule="R5.1",
                hint=f"set `retrieval.store: numpy` in {META_DIR}/{CONFIG_FILE}, or use a "
                f"Python built with SQLite extension support",
            ) from exc
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rows_meta ("
            " rowid INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, meta TEXT NOT NULL)"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.commit()
        self._conn = conn
        return conn

    def upsert(self, ids: Sequence[str], vectors: Any, metas: Sequence[Meta]) -> int:
        id_list, matrix, meta_list = _prepare_batch(ids, vectors, metas)
        if not id_list:
            return 0
        conn = self._connection()
        self._ensure_vectors_table(conn, matrix.shape[1])
        with conn:
            self._delete_ids(conn, id_list)
            for item_id, vector, meta in zip(id_list, matrix, meta_list):
                cursor = conn.execute(
                    "INSERT INTO rows_meta (id, meta) VALUES (?, ?)",
                    (item_id, json.dumps(meta, ensure_ascii=False)),
                )
                conn.execute(
                    "INSERT INTO vectors (rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, self._serialize(vector)),
                )
        return len(id_list)

    def delete(self, ids: Sequence[str]) -> int:
        conn = self._connection()
        with conn:
            return self._delete_ids(conn, list(dict.fromkeys(ids)))

    def search(self, vector: Any, k: int = 8, filters: Filters | None = None) -> list[Hit]:
        conn = self._connection()
        dim = self._stored_dim(conn)
        if k <= 0 or dim is None:
            return []
        query = _as_query(vector)
        if query.shape[0] != dim:
            raise _dim_mismatch(dim, query.shape[0])
        total = self.count()
        if total == 0:
            return []

        limit = k if not filters else min(k * _OVERFETCH, total)
        blob = self._serialize(query)
        while True:
            rows = conn.execute(
                "SELECT m.id, m.meta, v.distance FROM vectors v "
                "JOIN rows_meta m ON m.rowid = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
                (blob, limit),
            ).fetchall()
            hits = [
                (str(item_id), 1.0 - (float(distance) ** 2) / 2.0)
                for item_id, meta, distance in rows
                if matches_filters(json.loads(meta), filters)
            ]
            if len(hits) >= k or limit >= total:
                return hits[:k]
            limit = min(limit * _OVERFETCH, total)

    def clear(self) -> None:
        conn = self._connection()
        with conn:
            conn.execute("DROP TABLE IF EXISTS vectors")
            conn.execute("DELETE FROM rows_meta")
            conn.execute("DELETE FROM settings")

    def count(self) -> int:
        conn = self._connection()
        return int(conn.execute("SELECT count(*) FROM rows_meta").fetchone()[0])

    def flush(self) -> None:
        """No-op: every mutation commits its own transaction."""

    def close(self) -> None:
        """Close the connection. Not part of the protocol; for a caller that owns the store."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _serialize(self, vector: NDArray[np.float32]) -> bytes:
        return np.ascontiguousarray(vector, dtype=np.float32).tobytes()

    def _delete_ids(self, conn: sqlite3.Connection, ids: Sequence[str]) -> int:
        if not ids or self._stored_dim(conn) is None:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        rowids = [
            row[0]
            for row in conn.execute(
                f"SELECT rowid FROM rows_meta WHERE id IN ({placeholders})", tuple(ids)
            ).fetchall()
        ]
        for rowid in rowids:
            conn.execute("DELETE FROM vectors WHERE rowid = ?", (rowid,))
            conn.execute("DELETE FROM rows_meta WHERE rowid = ?", (rowid,))
        return len(rowids)

    def _ensure_vectors_table(self, conn: sqlite3.Connection, dim: int) -> None:
        stored = self._stored_dim(conn)
        if stored is None:
            with conn:
                conn.execute(f"CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{dim}])")
                conn.execute("INSERT INTO settings (key, value) VALUES ('dim', ?)", (str(dim),))
            return
        if stored != dim:
            raise _dim_mismatch(stored, dim)

    def _stored_dim(self, conn: sqlite3.Connection) -> int | None:
        row = conn.execute("SELECT value FROM settings WHERE key = 'dim'").fetchone()
        return None if row is None else int(row[0])


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════
_STORES: Final[dict[str, type]] = {
    "numpy": NumpyStore,
    "lancedb": LanceDBStore,
    "sqlite-vec": SqliteVecStore,
}


def open_store(name: str, index_dir: Path | str, **kwargs: Any) -> VectorStore:
    """Build the store called ``name`` over ``index_dir``.

    Raises:
        MechabrainIndexError: unknown name, or the backend's extra is missing.
            The manifest validator already rejects unknown names, so this fires
            only for a caller bypassing it.
    """
    try:
        factory = _STORES[name]
    except KeyError:
        raise MechabrainIndexError(
            f"unknown vector store {name!r}",
            rule="R5.1",
            hint=f"available stores: {', '.join(sorted(_STORES))}",
        ) from None
    return factory(index_dir, **kwargs)


def from_manifest(manifest: Manifest, paths: VaultPaths, **kwargs: Any) -> VectorStore:
    """The store this deployment asked for (`retrieval.store`), over its index dir.

    The one constructor the rest of the kernel should call: it is what keeps the
    backend a manifest decision instead of an import decision, and it is what
    guarantees no store can be pointed outside ``_meta/index/``.
    """
    return open_store(manifest.retrieval.store, paths.index_dir, **kwargs)
