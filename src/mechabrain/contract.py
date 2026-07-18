"""The portable invariant (spec §3).

Everything in this module is the *contract* the kernel knows by name: folder
names, file names, memory types and the markers of the managed block. None of
it is configurable. Anything a deployment may change lives in the manifest
(``_meta/config.yaml``) instead -- see :mod:`mechabrain.manifest`.

Rule of thumb (R4.1): if a name appears here, it is part of the spec and is
identical in every vault on earth. If it varies per vault, it does not belong
here.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

__all__ = [
    "SPEC_VERSION",
    "MemoryType",
    "ROOT_DIR",
    "SEMANTIC_DIR",
    "EPISODIC_DIR",
    "PROCEDURAL_DIR",
    "RESEARCH_DIR",
    "INDICES_DIR",
    "INBOX_DIR",
    "META_DIR",
    "INDEX_DIR",
    "AGENTS_FILE",
    "HOT_FILE",
    "INDEX_FILE",
    "CONFIG_FILE",
    "SCHEMA_FILE",
    "LINKS_FILE",
    "ACCESS_FILE",
    "ACTIONS_FILE",
    "MANAGED_BLOCK_BEGIN",
    "MANAGED_BLOCK_END",
    "TYPE_FOLDERS",
    "GITIGNORE_INDEX_ENTRY",
    "MARKDOWN_SUFFIX",
    "FRONTMATTER_FENCE",
    "STATUS_ACTIVE",
    "STATUS_ARCHIVED",
    "STATUS_DEPRECATED",
    "CONFIDENCE_LEVELS",
    "folder_for_type",
    "type_for_folder",
]

#: Version of the folder/frontmatter contract this kernel implements (§5).
SPEC_VERSION: Final[str] = "0.1"


class MemoryType(str, Enum):
    """The four CoALA-derived memory types the kernel routes writes into (§3, §8.1).

    Inherits from :class:`str` so members compare equal to their wire value and
    serialise straight to YAML/JSON without a custom encoder.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    RESEARCH = "research"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: str) -> "MemoryType":
        """Coerce ``value`` to a member, raising ``ValueError`` with the valid set.

        Prefer this over ``MemoryType(value)`` at boundaries: the stock enum
        error does not tell the caller what the alternatives are (R5.1).
        """
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"unknown memory type {value!r}; valid types are: {valid}"
            ) from None


# ── Folder names (§3) ───────────────────────────────────────────────
#: The contractual folder installed at the host vault root.
ROOT_DIR: Final[str] = "mecha-brain"
SEMANTIC_DIR: Final[str] = "Semantic"
EPISODIC_DIR: Final[str] = "Episodic"
PROCEDURAL_DIR: Final[str] = "Procedural"
RESEARCH_DIR: Final[str] = "Research"
INDICES_DIR: Final[str] = "indices"
INBOX_DIR: Final[str] = "_inbox"
META_DIR: Final[str] = "_meta"
#: Derived vector/BM25 state. Runtime layer: gitignored, per-machine (§4).
INDEX_DIR: Final[str] = "index"

# ── File names (§3) ─────────────────────────────────────────────────
AGENTS_FILE: Final[str] = "AGENTS.md"
HOT_FILE: Final[str] = "hot.md"
INDEX_FILE: Final[str] = "index.md"
CONFIG_FILE: Final[str] = "config.yaml"
SCHEMA_FILE: Final[str] = "schema.md"
#: Authored edges from ``memory_link`` (§7.2). Git-tracked: source of truth,
#: not derived state -- hence ``_meta/`` and not ``_meta/index/``.
LINKS_FILE: Final[str] = "links.jsonl"
#: Access log feeding decay (R7.2). Derived + gitignored: lives under index/.
ACCESS_FILE: Final[str] = "access.jsonl"
#: Kernel action log for observability (v0.2.1). Derived + gitignored: index/.
ACTIONS_FILE: Final[str] = "actions.jsonl"

# ── Managed block markers (§10) ─────────────────────────────────────
# The kernel regenerates only what sits between these markers in AGENTS.md;
# anything a human writes outside them survives `mechabrain sync`.
MANAGED_BLOCK_BEGIN: Final[str] = "<!-- mechabrain:begin -->"
MANAGED_BLOCK_END: Final[str] = "<!-- mechabrain:end -->"

# ── Misc contract literals ──────────────────────────────────────────
MARKDOWN_SUFFIX: Final[str] = ".md"
FRONTMATTER_FENCE: Final[str] = "---"
#: Line `mechabrain init` adds to the host vault's .gitignore (§10).
GITIGNORE_INDEX_ENTRY: Final[str] = f"{ROOT_DIR}/{META_DIR}/{INDEX_DIR}/"

#: Frontmatter `status:` values (§6). `deprecated` is procedural-only (§9.4).
STATUS_ACTIVE: Final[str] = "ativo"
STATUS_ARCHIVED: Final[str] = "arquivado"
STATUS_DEPRECATED: Final[str] = "deprecado"

#: Frontmatter `confidence:` values (§6), ordered weakest to strongest so the
#: index is usable as a rank for `min_confidence` filtering (§7.1).
CONFIDENCE_LEVELS: Final[tuple[str, ...]] = ("low", "medium", "high")

#: type -> folder, relative to `mecha-brain/` (§3).
#: Episodic notes go one level deeper, under `Episodic/<agent-id>/` (R6.3);
#: that subfolder is a deployment value and comes from the manifest registry.
TYPE_FOLDERS: Final[dict[MemoryType, str]] = {
    MemoryType.EPISODIC: EPISODIC_DIR,
    MemoryType.SEMANTIC: SEMANTIC_DIR,
    MemoryType.PROCEDURAL: PROCEDURAL_DIR,
    MemoryType.RESEARCH: RESEARCH_DIR,
}

_FOLDER_TYPES: Final[dict[str, MemoryType]] = {
    folder: mtype for mtype, folder in TYPE_FOLDERS.items()
}


def folder_for_type(memory_type: MemoryType | str) -> str:
    """Return the contractual folder name for ``memory_type``.

    The result is relative to ``mecha-brain/``, never an absolute path (R4.2).
    """
    return TYPE_FOLDERS[MemoryType.parse(str(memory_type))]


def type_for_folder(folder: str) -> MemoryType:
    """Inverse of :func:`folder_for_type`; raises ``KeyError`` for a non-memory folder."""
    return _FOLDER_TYPES[folder]
