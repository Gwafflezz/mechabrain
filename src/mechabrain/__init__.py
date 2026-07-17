"""Mecha-Brain: drop-in agentic memory for Markdown vaults.

A memory service and MCP server over a contractual ``mecha-brain/`` folder
installed in any Markdown vault. Conceptual base: CoALA (Sumers et al.,
arXiv:2309.02427).

Three layers, never mixed (P2):

* **kernel** -- this package: code, installed outside the vault;
* **deployment** -- ``mecha-brain/``: memories + manifest, synced with the vault's git;
* **runtime** -- ``_meta/index/``: derived index, per-machine, gitignored.

The kernel holds no vault name, no agent name and no path (R4.1). Everything
deployment-specific comes from the manifest -- see :mod:`mechabrain.manifest`.
The kernel never calls an LLM: it implements what is mechanically verifiable
and reports the rest for an agent to judge.

Submodules are imported directly. This package exposes only the version and the
error base, keeping the import cheap and free of cycles::

    from mechabrain.discovery import discover_vault
    from mechabrain.manifest import load_manifest
    from mechabrain.note import Note
"""

from __future__ import annotations

__version__ = "0.1.0"

from .errors import MechabrainError

__all__ = ["__version__", "MechabrainError"]
