"""The derived index: embeddings, chunks, vector store, BM25, search.

Everything under this package is the **runtime** layer (§4): it lives in
``mecha-brain/_meta/index/``, is gitignored, per-machine, and rebuildable from
the Markdown at any time (P1). Nothing here is ever the source of truth, so a
corrupt or missing index is always answerable with ``mechabrain reindex --full``
rather than with recovery.

Submodules are imported directly, keeping the package import cheap::

    from mechabrain.index.embed import from_manifest

    provider = from_manifest(manifest)
    vectors = provider.embed_texts(["a chunk of text"])
"""

from __future__ import annotations

__all__: list[str] = []
