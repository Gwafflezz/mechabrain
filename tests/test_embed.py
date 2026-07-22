"""The embedding layer: the protocol, its three backends, and the manifest factory."""

from __future__ import annotations

import copy
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from mechabrain.index.embed import (
    API_KEY_ENV_VAR,
    DEFAULT_HASH_DIM,
    DEFAULT_HTTP_TIMEOUT,
    EMBED_DTYPE,
    ENDPOINT_ENV_VAR,
    HTTP_BATCH_SIZE,
    TIMEOUT_ENV_VAR,
    EmbeddingError,
    EmbeddingProvider,
    HashProvider,
    HttpProvider,
    SentenceTransformersProvider,
    clear_model_cache,
    from_manifest,
)
from mechabrain.manifest import Manifest

ENDPOINT = "http://127.0.0.1:9/v1/embeddings"


@pytest.fixture(autouse=True)
def isolate_embedding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop cached models and every embedding env var (autouse).

    The cache is process-global and the env vars are the runtime layer: a
    developer's own endpoint must never make a test pass, or fail.
    """
    for var in (ENDPOINT_ENV_VAR, API_KEY_ENV_VAR, TIMEOUT_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    clear_model_cache()
    yield
    clear_model_cache()


# ══════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════
class FakeSentenceTransformer:
    """Stand-in for the real model: counts loads, embeds by character code.

    Lets the suite exercise the provider and its cache without the `embed`
    extra, which is deliberately not a core dependency.
    """

    loads = 0
    dimension = 4

    def __init__(self, model_name: str) -> None:
        type(self).loads += 1
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return type(self).dimension

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        # Unnormalised on purpose: the provider owns normalisation.
        return np.array(
            [[float(len(text))] * type(self).dimension for text in texts],
            dtype=np.float64,
        )


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    """Install a fake `sentence_transformers` module for the duration of a test."""
    FakeSentenceTransformer.loads = 0
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeSentenceTransformer


@pytest.fixture
def no_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the import fails, whether or not the extra is installed."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


def openai_response(vectors: list[list[float]], *, shuffle: bool = False) -> dict[str, Any]:
    """An OpenAI `/v1/embeddings` response body carrying ``vectors``."""
    data = [
        {"object": "embedding", "index": i, "embedding": vector}
        for i, vector in enumerate(vectors)
    ]
    if shuffle:
        data.reverse()
    return {"object": "list", "data": data, "model": "fake"}


class FakeTransport:
    """Records every POST and replays canned responses."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> Any:
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
        )
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    """Factory replacing the module's single network seam."""

    def _install(*responses: Any) -> FakeTransport:
        fake = FakeTransport(*responses)
        monkeypatch.setattr("mechabrain.index.embed._http_post_json", fake)
        return fake

    return _install


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two rows this module produced -- a plain dot product."""
    return float(a @ b)


# ══════════════════════════════════════════════════════════════════════
# The protocol
# ══════════════════════════════════════════════════════════════════════
def test_every_backend_satisfies_the_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENDPOINT_ENV_VAR, ENDPOINT)
    for provider in (
        HashProvider("hash-8"),
        SentenceTransformersProvider("some-model"),
        HttpProvider("some-model"),
    ):
        assert isinstance(provider, EmbeddingProvider)


# ══════════════════════════════════════════════════════════════════════
# HashProvider
# ══════════════════════════════════════════════════════════════════════
def test_hash_returns_one_normalised_float32_row_per_text() -> None:
    provider = HashProvider("hash-64")
    vectors = provider.embed_texts(["alpha beta", "gamma delta", "epsilon"])

    assert vectors.shape == (3, 64)
    assert vectors.dtype == EMBED_DTYPE
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_hash_cosine_is_a_dot_product() -> None:
    provider = HashProvider("hash-64")
    vectors = provider.embed_texts(["the vector store is brute force"])
    assert cosine(vectors[0], vectors[0]) == pytest.approx(1.0, abs=1e-6)


def test_hash_is_deterministic_across_instances() -> None:
    text = "markdown is the source of truth"
    assert np.array_equal(
        HashProvider("hash-128").embed_texts([text]),
        HashProvider("hash-128").embed_texts([text]),
    )


def test_hash_is_stable_across_processes() -> None:
    """The whole point of BLAKE2b over `hash()`: PYTHONHASHSEED cannot move it.

    A per-process vector would mean an index that silently stops matching its
    own corpus after a daemon restart.
    """
    import os
    import subprocess

    code = (
        "from mechabrain.index.embed import HashProvider;"
        "print(HashProvider('hash-32').embed_texts(['stable text'])[0][:4].tolist())"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1


def test_hash_scores_shared_wording_above_unrelated_text() -> None:
    """Lexical, not semantic -- but enough signal to drive a test corpus."""
    provider = HashProvider("hash-256")
    near, far, base = provider.embed_texts(
        [
            "brute force cosine is fast enough below ten thousand chunks",
            "the consolidator writes the blackboard once per cycle",
            "brute force cosine is fast enough below ten thousand vectors",
        ]
    )
    assert cosine(base, near) > cosine(base, far)


def test_hash_embeds_empty_text_as_the_zero_vector() -> None:
    vectors = HashProvider("hash-32").embed_texts(["", "   ", "real text"])

    assert not np.isnan(vectors).any()
    assert np.allclose(vectors[0], 0.0)
    assert np.allclose(vectors[1], 0.0)
    assert np.linalg.norm(vectors[2]) == pytest.approx(1.0, abs=1e-6)


def test_empty_input_returns_an_empty_matrix_of_the_right_width() -> None:
    vectors = HashProvider("hash-32").embed_texts([])
    assert vectors.shape == (0, 32)
    assert vectors.dtype == EMBED_DTYPE


def test_hash_dim_comes_from_the_model_name() -> None:
    assert HashProvider("hash-16").dim == 16
    assert HashProvider("hash").dim == DEFAULT_HASH_DIM


def test_hash_name_carries_backend_and_model() -> None:
    assert HashProvider("hash-256").name == "hash:hash-256"


def test_hash_rejects_a_model_name_it_does_not_understand() -> None:
    """`provider: hash` with the default model must fail, not infer dim 3 from 'bge-m3'."""
    with pytest.raises(EmbeddingError, match="does not know the model 'BAAI/bge-m3'"):
        HashProvider("BAAI/bge-m3")


def test_hash_rejects_a_non_positive_width() -> None:
    with pytest.raises(EmbeddingError, match="0 dimensions"):
        HashProvider("hash-0")


def test_a_bare_string_is_rejected_rather_than_embedded_per_character() -> None:
    with pytest.raises(TypeError, match=r"pass \[text\]"):
        HashProvider("hash-32").embed_texts("not a list")  # type: ignore[arg-type]


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"texts\[1\] is int"):
        HashProvider("hash-32").embed_texts(["fine", 7])  # type: ignore[list-item]


# ══════════════════════════════════════════════════════════════════════
# SentenceTransformersProvider
# ══════════════════════════════════════════════════════════════════════
def test_missing_extra_names_the_install_command(no_sentence_transformers: None) -> None:
    provider = SentenceTransformersProvider("BAAI/bge-m3")
    with pytest.raises(EmbeddingError, match=r"pip install mechabrain\[embed\]"):
        provider.embed_texts(["text"])


def test_constructing_the_default_provider_does_not_import_the_extra(
    no_sentence_transformers: None,
) -> None:
    """`check`, `init` and manifest parsing must work with the extra uninstalled."""
    assert SentenceTransformersProvider("BAAI/bge-m3").name == "sentence-transformers:BAAI/bge-m3"


def test_sentence_transformers_output_is_normalised(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    vectors = SentenceTransformersProvider("fake-model").embed_texts(["ab", "abcd"])

    assert vectors.shape == (2, 4)
    assert vectors.dtype == EMBED_DTYPE
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_the_model_loads_once_per_process(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    """R7.4: the daemon serves many requests and must not reload the weights."""
    SentenceTransformersProvider("fake-model").embed_texts(["one"])
    SentenceTransformersProvider("fake-model").embed_texts(["two"])
    assert fake_sentence_transformers.loads == 1


def test_a_different_model_is_a_different_cache_entry(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    SentenceTransformersProvider("model-a").embed_texts(["x"])
    SentenceTransformersProvider("model-b").embed_texts(["x"])
    assert fake_sentence_transformers.loads == 2


def test_sentence_transformers_dim_comes_from_the_model(
    fake_sentence_transformers: type[FakeSentenceTransformer],
) -> None:
    assert SentenceTransformersProvider("fake-model").dim == 4


def test_a_failing_encode_surfaces_as_an_embedding_error(
    fake_sentence_transformers: type[FakeSentenceTransformer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self: Any, texts: list[str], **kwargs: Any) -> np.ndarray:
        raise RuntimeError("CUDA is on fire")

    monkeypatch.setattr(FakeSentenceTransformer, "encode", boom)
    with pytest.raises(EmbeddingError, match="failed to encode 1 text"):
        SentenceTransformersProvider("fake-model").embed_texts(["text"])


# ══════════════════════════════════════════════════════════════════════
# HttpProvider
# ══════════════════════════════════════════════════════════════════════
def test_http_without_an_endpoint_names_the_env_var() -> None:
    with pytest.raises(EmbeddingError, match=ENDPOINT_ENV_VAR):
        HttpProvider("some-model")


def test_http_reads_the_endpoint_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, transport: Any
) -> None:
    monkeypatch.setenv(ENDPOINT_ENV_VAR, ENDPOINT)
    fake = transport(openai_response([[3.0, 4.0]]))

    HttpProvider("some-model").embed_texts(["text"])

    assert fake.calls[0]["url"] == ENDPOINT
    assert fake.calls[0]["timeout"] == DEFAULT_HTTP_TIMEOUT


def test_http_normalises_what_the_endpoint_returns(transport: Any) -> None:
    transport(openai_response([[3.0, 4.0]]))
    vectors = HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["text"])

    assert vectors.dtype == EMBED_DTYPE
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_http_sends_model_and_input(transport: Any) -> None:
    fake = transport(openai_response([[1.0, 0.0], [0.0, 1.0]]))
    HttpProvider("custom-model", endpoint=ENDPOINT).embed_texts(["a", "b"])

    assert fake.calls[0]["payload"]["model"] == "custom-model"
    assert fake.calls[0]["payload"]["input"] == ["a", "b"]


def test_the_api_key_comes_from_the_environment_and_never_from_the_manifest(
    monkeypatch: pytest.MonkeyPatch, transport: Any
) -> None:
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret-token")
    fake = transport(openai_response([[1.0, 0.0]]))

    HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["text"])

    assert fake.calls[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_no_auth_header_without_a_key(transport: Any) -> None:
    fake = transport(openai_response([[1.0, 0.0]]))
    HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["text"])
    assert "Authorization" not in fake.calls[0]["headers"]


def test_http_restores_the_order_the_caller_asked_for(transport: Any) -> None:
    """The OpenAI contract permits any order; arrival order would mislabel chunks."""
    transport(openai_response([[1.0, 0.0], [0.0, 1.0]], shuffle=True))
    vectors = HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["first", "second"])

    assert np.allclose(vectors[0], [1.0, 0.0])
    assert np.allclose(vectors[1], [0.0, 1.0])


def test_http_batches_long_inputs_and_keeps_them_in_order(transport: Any) -> None:
    count = HTTP_BATCH_SIZE + 3
    responses = [
        openai_response([[float(i), 0.0] for i in range(HTTP_BATCH_SIZE)]),
        openai_response([[float(i), 0.0] for i in range(HTTP_BATCH_SIZE, count)]),
    ]
    fake = transport(*responses)

    vectors = HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(
        [f"text {i}" for i in range(count)]
    )

    assert len(fake.calls) == 2
    assert len(fake.calls[0]["payload"]["input"]) == HTTP_BATCH_SIZE
    assert len(fake.calls[1]["payload"]["input"]) == 3
    assert vectors.shape == (count, 2)
    # Row 0 is the only zero-length input vector; every other row is [1, 0].
    assert np.allclose(vectors[1:, 0], 1.0)


def test_http_dim_is_probed_once_and_cached(transport: Any) -> None:
    fake = transport(openai_response([[1.0, 0.0, 0.0]]))
    provider = HttpProvider("some-model", endpoint=ENDPOINT)

    assert provider.dim == 3
    assert provider.dim == 3
    assert len(fake.calls) == 1


def test_http_rejects_a_response_with_the_wrong_count(transport: Any) -> None:
    transport(openai_response([[1.0, 0.0]]))
    with pytest.raises(EmbeddingError, match=r"returned 1 embedding\(s\) for 2 input\(s\)"):
        HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["a", "b"])


def test_http_rejects_a_body_that_is_not_an_embeddings_response(transport: Any) -> None:
    transport({"error": "model not found"})
    with pytest.raises(EmbeddingError, match="no 'data' list"):
        HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["a"])


def test_http_rejects_ragged_vectors(transport: Any) -> None:
    transport(openai_response([[1.0, 0.0], [1.0]]))
    with pytest.raises(EmbeddingError, match="rectangular array of numbers"):
        HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["a", "b"])


def test_http_rejects_a_base64_embedding_with_an_actionable_hint(transport: Any) -> None:
    transport({"data": [{"index": 0, "embedding": "eyJhIjogMX0="}]})
    with pytest.raises(EmbeddingError, match="encoding_format=float"):
        HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["a"])


def test_http_rejects_an_out_of_range_index(transport: Any) -> None:
    transport({"data": [{"index": 7, "embedding": [1.0, 0.0]}]})
    with pytest.raises(EmbeddingError, match="out-of-range index 7"):
        HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["a"])


def test_an_unreadable_timeout_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TIMEOUT_ENV_VAR, "soon")
    with pytest.raises(EmbeddingError, match="is not a number"):
        HttpProvider("some-model", endpoint=ENDPOINT)


def test_the_timeout_can_be_overridden_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, transport: Any
) -> None:
    monkeypatch.setenv(TIMEOUT_ENV_VAR, "2.5")
    fake = transport(openai_response([[1.0, 0.0]]))

    HttpProvider("some-model", endpoint=ENDPOINT).embed_texts(["a"])

    assert fake.calls[0]["timeout"] == 2.5


def test_http_name_ignores_the_endpoint() -> None:
    """The index is rebuilt when the model changes; a new local address is the same space."""
    a = HttpProvider("custom-model", endpoint=ENDPOINT)
    b = HttpProvider("custom-model", endpoint="http://elsewhere:1234/v1/embeddings")
    assert a.name == b.name == "http:custom-model"


# ══════════════════════════════════════════════════════════════════════
# from_manifest
# ══════════════════════════════════════════════════════════════════════
def test_the_ci_manifest_builds_the_hash_provider(manifest_ci: Manifest) -> None:
    provider = from_manifest(manifest_ci)

    assert isinstance(provider, HashProvider)
    assert provider.name == "hash:hash-256"
    assert provider.dim == 256


def test_the_default_manifest_builds_the_sentence_transformers_provider(
    manifest_min: Manifest, no_sentence_transformers: None
) -> None:
    """§5 default; building it must not need the extra."""
    provider = from_manifest(manifest_min)

    assert isinstance(provider, SentenceTransformersProvider)
    assert provider.name == "sentence-transformers:BAAI/bge-m3"


def test_the_factory_reads_the_provider_and_model_from_the_manifest(
    manifest_full: Manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENDPOINT_ENV_VAR, ENDPOINT)
    provider = from_manifest(manifest_full)

    assert isinstance(provider, HttpProvider)
    assert provider.name == "http:custom-model"


def test_an_http_deployment_without_an_endpoint_fails_at_build_time(
    manifest_full: Manifest,
) -> None:
    """A missing endpoint is configuration, and must surface at boot, not mid-reindex."""
    with pytest.raises(EmbeddingError, match=ENDPOINT_ENV_VAR):
        from_manifest(manifest_full)


def test_an_unknown_provider_lists_the_known_ones(manifest_data_ci: dict[str, Any]) -> None:
    """Unreachable via load_manifest, which validates the name -- still a real message."""
    data = copy.deepcopy(manifest_data_ci)
    manifest = Manifest.from_mapping(data)
    broken = object.__new__(type(manifest.retrieval.embedding))
    object.__setattr__(broken, "provider", "word2vec")
    object.__setattr__(broken, "model", "x")
    object.__setattr__(manifest.retrieval, "embedding", broken)

    with pytest.raises(EmbeddingError, match="sentence-transformers"):
        from_manifest(manifest)
