"""Vault discovery by convention (R4.3).

The kernel is installed outside the vault and must find it without a single
absolute path in its own source (R4.1/R4.2). The resolution order is normative
and implemented here exactly:

1. explicit argument (``--vault``);
2. environment variable ``MECHABRAIN_VAULT``;
3. walk up from the CWD until ``mecha-brain/_meta/config.yaml`` appears --
   the way git finds ``.git``.

The **vault root** is defined as the parent of ``mecha-brain/``. Every derived
path hangs off :class:`VaultPaths`; no other module should join contract folder
names by hand.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .contract import (
    ACCESS_FILE,
    AGENTS_FILE,
    CONFIG_FILE,
    EPISODIC_DIR,
    HOT_FILE,
    INBOX_DIR,
    INDEX_DIR,
    INDEX_FILE,
    INDICES_DIR,
    LINKS_FILE,
    META_DIR,
    ROOT_DIR,
    SCHEMA_FILE,
    MemoryType,
    folder_for_type,
)
from .errors import VaultNotFoundError

__all__ = [
    "VaultPaths",
    "discover_vault",
    "find_vault_root",
    "is_vault_root",
    "VAULT_ENV_VAR",
]

#: Environment variable holding the vault root (step 2 of R4.3).
VAULT_ENV_VAR: Final[str] = "MECHABRAIN_VAULT"

_INIT_HINT: Final[str] = (
    "run `mechabrain init <vault>` to create it, pass --vault <path>, "
    f"or set {VAULT_ENV_VAR}"
)


@dataclass(frozen=True, slots=True)
class VaultPaths:
    """Every path the contract (§3) defines, derived from one vault root.

    Build with :meth:`for_root` or get one from :func:`discover_vault`. Paths
    are absolute *in memory only* -- they are computed at runtime from the
    discovered root and never persisted (R4.2).

    Attributes:
        root: The vault root -- parent of ``mecha-brain/``.
        mecha_brain: The contractual folder itself.
        meta_dir: ``mecha-brain/_meta/`` -- manifest, schema, authored links.
        index_dir: ``mecha-brain/_meta/index/`` -- derived, gitignored, per-machine.
        config_file: ``_meta/config.yaml`` -- the manifest.
        links_file: ``_meta/links.jsonl`` -- authored edges; git-tracked.
        access_file: ``_meta/index/access.jsonl`` -- access log; gitignored (R7.2).
    """

    root: Path
    mecha_brain: Path
    semantic_dir: Path
    episodic_dir: Path
    procedural_dir: Path
    research_dir: Path
    indices_dir: Path
    inbox_dir: Path
    meta_dir: Path
    index_dir: Path
    agents_file: Path
    hot_file: Path
    index_file: Path
    config_file: Path
    schema_file: Path
    links_file: Path
    access_file: Path

    @classmethod
    def for_root(cls, root: Path | str) -> "VaultPaths":
        """Derive every contract path from ``root`` (the parent of ``mecha-brain/``).

        Does not touch the filesystem: use it to describe a vault that does not
        exist yet, as ``mechabrain init`` does.
        """
        root_path = Path(root).expanduser()
        mecha_brain = root_path / ROOT_DIR
        meta_dir = mecha_brain / META_DIR
        index_dir = meta_dir / INDEX_DIR
        return cls(
            root=root_path,
            mecha_brain=mecha_brain,
            semantic_dir=mecha_brain / folder_for_type(MemoryType.SEMANTIC),
            episodic_dir=mecha_brain / EPISODIC_DIR,
            procedural_dir=mecha_brain / folder_for_type(MemoryType.PROCEDURAL),
            research_dir=mecha_brain / folder_for_type(MemoryType.RESEARCH),
            indices_dir=mecha_brain / INDICES_DIR,
            inbox_dir=mecha_brain / INBOX_DIR,
            meta_dir=meta_dir,
            index_dir=index_dir,
            agents_file=mecha_brain / AGENTS_FILE,
            hot_file=mecha_brain / HOT_FILE,
            index_file=mecha_brain / INDEX_FILE,
            config_file=meta_dir / CONFIG_FILE,
            schema_file=meta_dir / SCHEMA_FILE,
            links_file=meta_dir / LINKS_FILE,
            access_file=index_dir / ACCESS_FILE,
        )

    # ── Derived paths ───────────────────────────────────────────────
    def folder_for(self, memory_type: MemoryType | str) -> Path:
        """Absolute folder for ``memory_type``.

        For ``episodic`` this is the parent ``Episodic/``; notes go one level
        deeper, per agent -- see :meth:`episodic_for` (R6.3).
        """
        return self.mecha_brain / folder_for_type(memory_type)

    def episodic_for(self, agent_id: str) -> Path:
        """``Episodic/<agent_id>/`` -- the append-only journal of one runtime.

        Episodic is partitioned by *agent* (runtime), never by profile: profiles
        of one runtime share a process, so only the agent boundary is
        enforceable (R6.6).
        """
        return self.episodic_dir / agent_id

    def scope_index(self, scope: str) -> Path:
        """``indices/<scope>.md`` -- the per-scope shard of the master MOC (§9.5)."""
        return self.indices_dir / f"{scope}.md"

    def resolve(self, relative_path: str | Path) -> Path:
        """Resolve a manifest path (always vault-root-relative, R4.2) to absolute.

        Raises:
            ValueError: ``relative_path`` is absolute -- a caller passing one has
                a manifest bug the validator should have caught.
        """
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError(
                f"R4.2: manifest paths are relative to the vault root, got absolute "
                f"{relative_path!r}"
            )
        return self.root / candidate

    def relative(self, path: Path | str) -> str:
        """Render ``path`` as a POSIX path relative to the vault root.

        The form that is safe to write into a note or a report: absolute paths
        never enter the deployment layer (R4.2).

        Raises:
            ValueError: ``path`` lies outside the vault.
        """
        target = Path(path)
        base = self.root
        try:
            return target.relative_to(base).as_posix()
        except ValueError:
            try:
                return target.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                raise ValueError(f"{path} is outside the vault at {self.root}") from None

    # ── State ───────────────────────────────────────────────────────
    @property
    def is_initialized(self) -> bool:
        """Whether a manifest exists -- the marker that makes a folder a vault."""
        return self.config_file.is_file()

    def memory_dirs(self) -> dict[MemoryType, Path]:
        """The four memory folders, keyed by type (§3)."""
        return {mtype: self.folder_for(mtype) for mtype in MemoryType}

    def contract_dirs(self) -> tuple[Path, ...]:
        """Every directory the §3 skeleton requires, parents before children."""
        return (
            self.mecha_brain,
            self.semantic_dir,
            self.episodic_dir,
            self.procedural_dir,
            self.research_dir,
            self.indices_dir,
            self.inbox_dir,
            self.meta_dir,
            self.index_dir,
        )

    def require_initialized(self) -> "VaultPaths":
        """Return self, or raise if no manifest is present.

        Raises:
            VaultNotFoundError: the folder is not an initialized deployment.
        """
        if not self.is_initialized:
            raise VaultNotFoundError(
                f"{self.root} has no {ROOT_DIR}/{META_DIR}/{CONFIG_FILE}",
                hint=_INIT_HINT,
            )
        return self


def is_vault_root(candidate: Path | str) -> bool:
    """Whether ``candidate`` is a vault root, i.e. holds ``mecha-brain/_meta/config.yaml``."""
    return VaultPaths.for_root(candidate).is_initialized


def find_vault_root(start: Path | str | None = None) -> Path | None:
    """Walk up from ``start`` (default CWD) to the first vault root, or ``None``.

    Step 3 of R4.3. Checks ``start`` itself before its ancestors, so running
    from the vault root works, and stops at the filesystem root.
    """
    current = Path(start).expanduser() if start is not None else Path.cwd()
    try:
        current = current.resolve()
    except OSError:
        return None
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if is_vault_root(directory):
            return directory
    return None


def discover_vault(
    explicit: Path | str | None = None,
    *,
    start: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> VaultPaths:
    """Locate the vault by convention and return its paths (R4.3).

    Order: ``explicit`` > ``$MECHABRAIN_VAULT`` > upward walk from ``start``.
    An earlier source that is set but wrong is an error -- it never falls
    through to the next one, because silently searching elsewhere after an
    explicit ``--vault`` would write memory to a vault the user did not name.

    Args:
        explicit: Value of ``--vault``, if the user passed one.
        start: Directory the upward walk begins at. Defaults to the CWD.
        env: Environment to read. Defaults to ``os.environ``; inject a dict in
            tests instead of mutating the process environment.

    Returns:
        The paths of an initialized vault.

    Raises:
        VaultNotFoundError: no vault at the named location, or none found by
            walking up. The message teaches the fix.
    """
    environ = os.environ if env is None else env

    if explicit is not None:
        return _require_vault(explicit, source="--vault")

    from_env = environ.get(VAULT_ENV_VAR)
    if from_env and from_env.strip():
        return _require_vault(from_env.strip(), source=f"${VAULT_ENV_VAR}")

    found = find_vault_root(start)
    if found is not None:
        return VaultPaths.for_root(found)

    origin = Path(start).expanduser() if start is not None else Path.cwd()
    raise VaultNotFoundError(
        f"no vault found: walked up from {origin} without finding "
        f"{ROOT_DIR}/{META_DIR}/{CONFIG_FILE}",
        hint=_INIT_HINT,
    )


def _require_vault(candidate: Path | str, *, source: str) -> VaultPaths:
    """Accept ``candidate`` as a vault root, or explain precisely why it is not."""
    root = Path(candidate).expanduser()
    paths = VaultPaths.for_root(root)
    if paths.is_initialized:
        return paths

    if not root.exists():
        raise VaultNotFoundError(
            f"{source} points at {root}, which does not exist",
            hint=_INIT_HINT,
        )
    if not root.is_dir():
        raise VaultNotFoundError(
            f"{source} points at {root}, which is not a directory",
            hint="the vault root is the folder that contains mecha-brain/",
        )
    # A common miss: pointing at mecha-brain/ instead of at its parent.
    if root.name == ROOT_DIR and (root / META_DIR / CONFIG_FILE).is_file():
        raise VaultNotFoundError(
            f"{source} points at the {ROOT_DIR}/ folder itself ({root})",
            hint=f"point it at the vault root instead: {root.parent}",
        )
    inner = find_vault_root(root)
    if inner is not None and inner != root:
        raise VaultNotFoundError(
            f"{source} points at {root}, which has no {ROOT_DIR}/{META_DIR}/{CONFIG_FILE}",
            hint=f"the nearest vault root above it is {inner}",
        )
    raise VaultNotFoundError(
        f"{source} points at {root}, which has no {ROOT_DIR}/{META_DIR}/{CONFIG_FILE}",
        hint=_INIT_HINT,
    )
