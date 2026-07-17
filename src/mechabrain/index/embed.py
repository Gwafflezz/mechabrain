"""The embedding layer: text -> unit vectors.

One :class:`EmbeddingProvider` protocol, three implementations, one factory
driven by the manifest (`retrieval.embedding`, §5). The kernel knows the three
backend *names*; which one a deployment uses, and with which model, is data
(P6, R4.1).

===========================  ==========================================
`retrieval.embedding.provider`  Backend
===========================  ==========================================
``sentence-transformers``    Local model (default, `BAAI/bge-m3`). Needs
                             the ``embed`` extra.
``http``                     Any OpenAI-`/v1/embeddings`-compatible
                             endpoint. Endpoint and key come from the
                             environment, never from the manifest.
``hash``                     Deterministic, offline, **non-semantic**.
                             For tests and CI only.
===========================  ==========================================

**Every vector this module returns is L2-normalised**, so cosine similarity is a
dot product and the vector store never has to normalise again. Callers may rely
on that: ``vectors @ query`` *is* cosine similarity.

A text that carries no features (empty, whitespace, punctuation only) embeds to
the **zero vector** rather than to NaN. Its similarity to everything is 0, which
is the honest answer for a chunk with no content.

Typical use::

    from mechabrain.index.embed import from_manifest

    provider = from_manifest(manifest)
    vectors = provider.embed_texts([chunk.text for chunk in chunks])
    query = provider.embed_texts([question])[0]
    scores = vectors @ query

Loaded models are cached per process, keyed by model name: `serve` is a daemon
(R7.4) and must not reload a multi-hundred-megabyte model per request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from ..errors import MechabrainError
from ..manifest import EMBEDDING_PROVIDERS, Manifest

__all__ = [
    "EmbeddingProvider",
    "EmbeddingError",
    "HashProvider",
    "SentenceTransformersProvider",
    "HttpProvider",
    "from_manifest",
    "l2_normalize",
    "clear_model_cache",
    "EMBED_DTYPE",
    "DEFAULT_HASH_DIM",
    "ENDPOINT_ENV_VAR",
    "API_KEY_ENV_VAR",
    "TIMEOUT_ENV_VAR",
    "DEFAULT_HTTP_TIMEOUT",
    "HTTP_BATCH_SIZE",
]

#: Vectors are float32 throughout: half the memory of float64 for a cosine
#: score whose meaningful precision is two decimals.
EMBED_DTYPE: Final[type[np.float32]] = np.float32

#: Width of a :class:`HashProvider` vector when the model name names none.
DEFAULT_HASH_DIM: Final[int] = 256

# ── Environment (runtime layer, §4) ─────────────────────────────────
#: Full URL of the OpenAI-compatible embeddings endpoint, e.g.
#: ``http://127.0.0.1:8080/v1/embeddings``. Environment, not manifest: it is a
#: per-machine address like the daemon's port, and a synced manifest carrying
#: one would break on the next machine.
ENDPOINT_ENV_VAR: Final[str] = "MECHABRAIN_EMBED_ENDPOINT"

#: API key for that endpoint, sent as ``Authorization: Bearer``. Never in the
#: manifest: the manifest is committed to the vault's git (§4).
API_KEY_ENV_VAR: Final[str] = "MECHABRAIN_EMBED_API_KEY"

#: Optional override of :data:`DEFAULT_HTTP_TIMEOUT`, in seconds.
TIMEOUT_ENV_VAR: Final[str] = "MECHABRAIN_EMBED_TIMEOUT"

#: Long enough for a cold remote model, short enough that a wedged endpoint
#: surfaces as an error instead of hanging a session.
DEFAULT_HTTP_TIMEOUT: Final[float] = 60.0

#: Texts per HTTP request. Bounds the payload; a full reindex of a personal
#: vault is thousands of chunks and no endpoint wants them in one body.
HTTP_BATCH_SIZE: Final[int] = 64

# ── Hash features ───────────────────────────────────────────────────
#: Word n-gram widths the hash provider extracts (unigrams + bigrams).
HASH_WORD_NGRAMS: Final[tuple[int, ...]] = (1, 2)

#: Character n-gram width, over whitespace-normalised text. Gives the hash
#: provider signal on text the word tokenizer cannot split (CJK) and on
#: near-duplicates that differ inside a word.
HASH_CHAR_NGRAM: Final[int] = 3

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)
#: `hash` or `hash-<dim>`, and nothing else -- see :meth:`HashProvider._dim_from_model`.
_HASH_MODEL_RE: Final[re.Pattern[str]] = re.compile(r"^hash(?:[-_](\d+))?$")

# Loading a model is slow and memory-hungry; the daemon does it once. The lock
# is held across the load so two threads racing on a cold cache produce one
# model, not two downloads of the same weights.
_MODEL_CACHE: Final[dict[str, Any]] = {}
_HTTP_DIM_CACHE: Final[dict[tuple[str, str], int]] = {}
_CACHE_LOCK: Final[threading.Lock] = threading.Lock()


class EmbeddingError(MechabrainError):
    """The embedding backend is unavailable, misconfigured or misbehaving.

    Covers a missing optional dependency, an unset endpoint, a transport
    failure and a response the kernel cannot read. Never raised for a text the
    model merely finds meaningless -- that embeds to the zero vector.
    """


# ══════════════════════════════════════════════════════════════════════
# Protocol
# ══════════════════════════════════════════════════════════════════════
@runtime_checkable
class EmbeddingProvider(Protocol):
    """What the index needs of an embedding backend.

    Implementations are pluggable (§5 `retrieval.embedding.provider`); the
    index, the retriever and the dedup gate depend on this protocol and never
    on a concrete backend.
    """

    @property
    def name(self) -> str:
        """Stable identity of backend **and** model, e.g. ``http:custom-model``.

        Stamped into the index so a rebuild is forced when a deployment swaps
        model: vectors from two models share no space, and silently mixing them
        would return plausible nonsense.
        """
        ...

    @property
    def dim(self) -> int:
        """Width of the vectors :meth:`embed_texts` returns."""
        ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an L2-normalised ``(len(texts), dim)`` float32 array.

        Row ``i`` is the embedding of ``texts[i]``: order is contractual, and a
        provider that batches must restore it. An empty ``texts`` returns an
        empty ``(0, dim)`` array rather than raising.

        Raises:
            EmbeddingError: the backend is unavailable or returned something
                unreadable.
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row of ``matrix`` to unit length, in float32.

    A zero row stays zero instead of becoming NaN: an empty chunk has no
    direction, and NaN would poison every downstream score silently. Cosine
    similarity against a zero row is 0, which is what "no content" should mean.

    Args:
        matrix: A 2-D ``(n, dim)`` array.

    Returns:
        A new ``(n, dim)`` float32 array; rows have norm 1 or 0.

    Raises:
        ValueError: ``matrix`` is not 2-D.
    """
    array = np.asarray(matrix, dtype=EMBED_DTYPE)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D (n, dim) array, got {array.ndim}-D with shape {array.shape}"
        )
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # Divide by 1 where the norm is 0: 0/1 == 0 keeps the zero row zero.
    return np.ascontiguousarray(array / np.where(norms == 0.0, 1.0, norms), dtype=EMBED_DTYPE)


def _empty(dim: int) -> np.ndarray:
    return np.zeros((0, dim), dtype=EMBED_DTYPE)


def _check_texts(texts: Sequence[str], provider: str) -> list[str]:
    """Reject a bare string early -- it would embed as one row per character."""
    if isinstance(texts, str):
        raise TypeError(
            f"{provider}.embed_texts takes a sequence of texts, got a single str; "
            f"pass [text] to embed one"
        )
    out: list[str] = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"{provider}.embed_texts: texts[{i}] is {type(text).__name__}, expected str"
            )
        out.append(text)
    return out


def clear_model_cache() -> None:
    """Drop every cached model and probed dimension.

    For tests and for a `reindex` that follows a manifest change; a running
    daemon has no reason to call it.
    """
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _HTTP_DIM_CACHE.clear()


# ══════════════════════════════════════════════════════════════════════
# hash -- deterministic, offline, NOT semantic
# ══════════════════════════════════════════════════════════════════════
class HashProvider:
    """Deterministic feature hashing. **For tests and CI only.**

    This provider has **no semantics**. It hashes word n-grams
    (:data:`HASH_WORD_NGRAMS`) and character n-grams (:data:`HASH_CHAR_NGRAM`)
    into a fixed number of buckets with the signed hashing trick, then
    normalises. Texts sharing wording land near each other; texts meaning the
    same thing in different words do not. Backing a real deployment with it
    would make `memory_search` a bad keyword search and the dedup gate (§8.2)
    blind to paraphrase.

    It exists because the suite must be deterministic and offline: no model
    download, no network, identical vectors on every machine and every Python
    process. It uses BLAKE2b rather than :func:`hash`, whose string hashing is
    randomised per process and would produce a different index on every boot.

    Args:
        model: `retrieval.embedding.model`. Either ``hash`` (width
            :data:`DEFAULT_HASH_DIM`) or ``hash-<dim>``, e.g. ``hash-256``.

    Raises:
        EmbeddingError: the model name is not one of those two forms.
    """

    __slots__ = ("_model", "_dim")

    def __init__(self, model: str = "hash") -> None:
        self._model = model
        self._dim = self._dim_from_model(model)

    @staticmethod
    def _dim_from_model(model: str) -> int:
        """Width named by ``model``, which must be ``hash`` or ``hash-<dim>``.

        Deliberately strict. Reading a width out of any trailing integer would
        turn `provider: hash` left with the default `model: BAAI/bge-m3` into a
        silent 3-dimensional index -- a mistake that only surfaces as bad
        retrieval, months later (R5.1).
        """
        match = _HASH_MODEL_RE.match(model)
        if match is None:
            raise EmbeddingError(
                f"the 'hash' embedding provider does not know the model {model!r}",
                rule="R5.1",
                hint=(
                    f"set retrieval.embedding.model to 'hash' (width "
                    f"{DEFAULT_HASH_DIM}) or 'hash-<dim>' such as 'hash-256'; "
                    f"hash embeddings carry no semantics and are for tests only"
                ),
            )
        if match.group(1) is None:
            return DEFAULT_HASH_DIM
        dim = int(match.group(1))
        if dim <= 0:
            raise EmbeddingError(
                f"hash embedding model {model!r} asks for {dim} dimensions",
                rule="R5.1",
                hint=f"use a positive width, e.g. 'hash-{DEFAULT_HASH_DIM}'",
            )
        return dim

    @property
    def name(self) -> str:
        return f"hash:{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts, "HashProvider")
        if not items:
            return _empty(self._dim)
        matrix = np.zeros((len(items), self._dim), dtype=EMBED_DTYPE)
        for row, text in enumerate(items):
            for feature in _hash_features(text):
                bucket, sign = _bucket_and_sign(feature, self._dim)
                matrix[row, bucket] += sign
        return l2_normalize(matrix)


def _hash_features(text: str) -> Iterator[str]:
    """Yield the word and character n-grams of ``text``, lowercased.

    Namespaced (``w:`` / ``c:``) so a word and an identical character n-gram
    are distinct features rather than colliding into one inflated bucket.
    """
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    for width in HASH_WORD_NGRAMS:
        for i in range(len(tokens) - width + 1):
            yield "w:" + " ".join(tokens[i : i + width])
    packed = " ".join(tokens)
    for i in range(len(packed) - HASH_CHAR_NGRAM + 1):
        yield "c:" + packed[i : i + HASH_CHAR_NGRAM]


def _bucket_and_sign(feature: str, dim: int) -> tuple[int, float]:
    """Map ``feature`` to a bucket and a sign (the signed hashing trick).

    The sign comes from the top bit and the bucket from the low bits of the same
    digest, so collisions cancel on average instead of always adding -- the
    standard fix for the bias of unsigned feature hashing.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if value >> 63 else -1.0


# ══════════════════════════════════════════════════════════════════════
# sentence-transformers -- the default
# ══════════════════════════════════════════════════════════════════════
class SentenceTransformersProvider:
    """A local `sentence-transformers` model (§5 default: `BAAI/bge-m3`).

    The import is lazy and the model loads on first use, so importing the
    kernel, reading a manifest or running `check` costs nothing and works with
    the extra uninstalled. The load is cached per process (R7.4: the daemon
    serves many requests).

    Args:
        model: Model name or local path, from `retrieval.embedding.model`.
    """

    __slots__ = ("_model_name",)

    def __init__(self, model: str) -> None:
        self._model_name = model

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    @property
    def dim(self) -> int:
        """Width reported by the model. Loads it on first access.

        Raises:
            EmbeddingError: the extra is missing, the model will not load, or
                it reports no dimension.
        """
        model = self._load()
        dim = model.get_sentence_embedding_dimension()
        if not isinstance(dim, int) or dim <= 0:
            raise EmbeddingError(
                f"model {self._model_name!r} reports an unusable embedding "
                f"dimension: {dim!r}",
                hint="pick a sentence-transformers model that exposes its dimension",
            )
        return dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts, "SentenceTransformersProvider")
        model = self._load()
        if not items:
            return _empty(self.dim)
        try:
            raw = model.encode(items, convert_to_numpy=True, normalize_embeddings=False)
        except Exception as exc:  # backend-specific; surface it as ours
            raise EmbeddingError(
                f"model {self._model_name!r} failed to encode {len(items)} text(s): {exc}",
                hint="check the model name and that its weights are present",
            ) from exc
        matrix = np.asarray(raw, dtype=EMBED_DTYPE)
        if matrix.ndim != 2 or matrix.shape[0] != len(items):
            raise EmbeddingError(
                f"model {self._model_name!r} returned shape {matrix.shape} for "
                f"{len(items)} text(s), expected ({len(items)}, dim)"
            )
        return l2_normalize(matrix)

    def _load(self) -> Any:
        """Return the cached model, loading it under the cache lock if cold."""
        with _CACHE_LOCK:
            cached = _MODEL_CACHE.get(self.name)
            if cached is not None:
                return cached
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "the 'sentence-transformers' embedding provider needs the "
                    "'embed' extra, which is not installed",
                    hint=(
                        "pip install mechabrain[embed]  "
                        "(or set retrieval.embedding.provider to 'http' in config.yaml)"
                    ),
                ) from exc
            try:
                model = SentenceTransformer(self._model_name)
            except Exception as exc:  # network, disk, bad name -- all fatal here
                raise EmbeddingError(
                    f"cannot load sentence-transformers model {self._model_name!r}: {exc}",
                    hint=(
                        "check retrieval.embedding.model in config.yaml; the first "
                        "load downloads the weights and needs network access"
                    ),
                ) from exc
            _MODEL_CACHE[self.name] = model
            return model


# ══════════════════════════════════════════════════════════════════════
# http -- any OpenAI-compatible endpoint
# ══════════════════════════════════════════════════════════════════════
class HttpProvider:
    """An OpenAI-`/v1/embeddings`-compatible endpoint (local server or API).

    Endpoint and key are **runtime**, never manifest (§4): the manifest is
    synced with the vault's git, so it holds no per-machine address and no
    secret. They come from :data:`ENDPOINT_ENV_VAR` and :data:`API_KEY_ENV_VAR`.

    The endpoint's dimension is not declared anywhere, so the first
    :attr:`dim` access probes it with a one-token request and caches the answer
    per (endpoint, model).

    Args:
        model: `retrieval.embedding.model`, sent as the request's ``model``.
        endpoint: Full URL. Defaults to :data:`ENDPOINT_ENV_VAR`.
        api_key: Bearer token. Defaults to :data:`API_KEY_ENV_VAR`; omitted from
            the request when unset, since local servers usually want no auth.
        timeout: Seconds per request. Defaults to :data:`TIMEOUT_ENV_VAR`, then
            to :data:`DEFAULT_HTTP_TIMEOUT`.

    Raises:
        EmbeddingError: no endpoint configured, or the timeout is unreadable.
    """

    __slots__ = ("_model", "_endpoint", "_api_key", "_timeout")

    def __init__(
        self,
        model: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved = endpoint if endpoint is not None else os.environ.get(ENDPOINT_ENV_VAR, "")
        if not resolved.strip():
            raise EmbeddingError(
                f"the 'http' embedding provider needs an endpoint, and "
                f"${ENDPOINT_ENV_VAR} is unset",
                hint=(
                    f"export {ENDPOINT_ENV_VAR}=http://127.0.0.1:8080/v1/embeddings "
                    f"(the full URL of the embeddings endpoint). It is runtime "
                    f"config, not manifest: it differs per machine"
                ),
            )
        self._model = model
        self._endpoint = resolved.strip()
        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV_VAR)
        self._timeout = _resolve_timeout(timeout)

    @property
    def name(self) -> str:
        """Backend and model -- deliberately not the endpoint.

        The index is rebuilt when the *model* changes; the same model reached at
        a different local address is the same vector space.
        """
        return f"http:{self._model}"

    @property
    def dim(self) -> int:
        """Width reported by the endpoint, probed once per (endpoint, model).

        Raises:
            EmbeddingError: the endpoint is unreachable or unreadable.
        """
        key = (self._endpoint, self._model)
        with _CACHE_LOCK:
            cached = _HTTP_DIM_CACHE.get(key)
        if cached is not None:
            return cached
        probe = self._request(["dimension probe"])
        dim = probe.shape[1]
        with _CACHE_LOCK:
            _HTTP_DIM_CACHE[key] = dim
        return dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        items = _check_texts(texts, "HttpProvider")
        if not items:
            return _empty(self.dim)
        batches = [
            self._request(items[start : start + HTTP_BATCH_SIZE])
            for start in range(0, len(items), HTTP_BATCH_SIZE)
        ]
        widths = {batch.shape[1] for batch in batches}
        if len(widths) > 1:
            raise EmbeddingError(
                f"endpoint {self._endpoint} returned vectors of differing widths "
                f"({sorted(widths)}) within one call",
                hint="the endpoint is not serving a single model consistently",
            )
        return l2_normalize(np.vstack(batches))

    def _request(self, batch: Sequence[str]) -> np.ndarray:
        """POST one batch and return its raw ``(len(batch), dim)`` matrix."""
        payload = {"model": self._model, "input": list(batch), "encoding_format": "float"}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = _http_post_json(self._endpoint, payload, headers, self._timeout)
        return self._parse(response, len(batch))

    def _parse(self, response: Any, expected: int) -> np.ndarray:
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            raise EmbeddingError(
                f"endpoint {self._endpoint} returned no 'data' list",
                hint="it must speak the OpenAI /v1/embeddings response shape",
            )
        data: list[Any] = response["data"]
        if len(data) != expected:
            raise EmbeddingError(
                f"endpoint {self._endpoint} returned {len(data)} embedding(s) for "
                f"{expected} input(s)"
            )
        # The OpenAI contract permits any order and carries `index` to fix it;
        # trusting arrival order would silently mislabel every chunk.
        rows: list[Any] = [None] * expected
        for position, item in enumerate(data):
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingError(
                    f"endpoint {self._endpoint}: data[{position}] carries no 'embedding'"
                )
            index = item.get("index", position)
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < expected:
                raise EmbeddingError(
                    f"endpoint {self._endpoint}: data[{position}] has out-of-range "
                    f"index {index!r} for a batch of {expected}"
                )
            if rows[index] is not None:
                raise EmbeddingError(
                    f"endpoint {self._endpoint}: two embeddings claim index {index}"
                )
            rows[index] = item["embedding"]
        try:
            matrix = np.asarray(rows, dtype=EMBED_DTYPE)
        except (ValueError, TypeError) as exc:
            raise EmbeddingError(
                f"endpoint {self._endpoint} returned embeddings that are not a "
                f"rectangular array of numbers: {exc}",
                hint="if it is returning base64, ask it for encoding_format=float",
            ) from exc
        if matrix.ndim != 2 or matrix.shape[0] != expected or matrix.shape[1] == 0:
            raise EmbeddingError(
                f"endpoint {self._endpoint} returned shape {matrix.shape}, "
                f"expected ({expected}, dim)"
            )
        return matrix


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is not None:
        return timeout
    raw = os.environ.get(TIMEOUT_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_HTTP_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        raise EmbeddingError(
            f"${TIMEOUT_ENV_VAR} is not a number: {raw!r}",
            hint=f"unset it to use the default of {DEFAULT_HTTP_TIMEOUT:g}s",
        ) from None
    if value <= 0:
        raise EmbeddingError(f"${TIMEOUT_ENV_VAR} must be > 0, got {value:g}")
    return value


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Any:
    """POST ``payload`` as JSON and return the decoded response.

    Stands on :mod:`urllib` so the kernel's core stays at three dependencies.
    The single seam between :class:`HttpProvider` and the network -- tests
    replace this function rather than a socket.

    Raises:
        EmbeddingError: transport failure, HTTP error status, or a body that is
            not JSON. There is no retry: a failing endpoint is reported, not
            papered over (R5.1).
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise EmbeddingError(
            f"embedding endpoint {url} returned HTTP {exc.code}"
            + (f": {detail[:500]}" if detail else ""),
            hint=f"check the endpoint URL and ${API_KEY_ENV_VAR}",
        ) from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(
            f"cannot reach embedding endpoint {url}: {exc.reason}",
            hint=f"check that the server is up and ${ENDPOINT_ENV_VAR} is right",
        ) from exc
    except OSError as exc:
        raise EmbeddingError(f"embedding request to {url} failed: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise EmbeddingError(
            f"embedding endpoint {url} returned a body that is not JSON: {exc}",
            hint="it must speak the OpenAI /v1/embeddings protocol",
        ) from exc


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════
def from_manifest(manifest: Manifest) -> EmbeddingProvider:
    """Build the provider `retrieval.embedding` asks for (§5).

    Construction is cheap and offline for every backend: no model loads and no
    request is made until the first :meth:`~EmbeddingProvider.embed_texts` or
    :attr:`~EmbeddingProvider.dim`. `http` is the one exception -- it validates
    its endpoint here, because a missing endpoint is a configuration error and
    should surface at boot, not mid-reindex.

    Raises:
        EmbeddingError: the provider name is unknown to this kernel, or the
            chosen backend cannot be configured.
    """
    spec = manifest.retrieval.embedding
    if spec.provider == "hash":
        return HashProvider(spec.model)
    if spec.provider == "sentence-transformers":
        return SentenceTransformersProvider(spec.model)
    if spec.provider == "http":
        return HttpProvider(spec.model)
    # Unreachable through load_manifest, which validates the name (§5). Reached
    # only by a hand-built Manifest, and still worth a real message.
    raise EmbeddingError(
        f"unknown embedding provider {spec.provider!r}",
        rule="R5.1",
        hint=(
            f"retrieval.embedding.provider must be one of: "
            f"{', '.join(sorted(EMBEDDING_PROVIDERS))}"
        ),
    )
