"""The manifest: `mecha-brain/_meta/config.yaml` (spec §5).

The single home of everything deployment-specific. The dataclasses here mirror
§5 one-to-one; the kernel reads vault names, agent ids, denylists and weights
from an instance of :class:`Manifest` and never from a literal in the source
(R4.1, P6).

Validation is **strict** (R5.1): an unknown key at any depth is an error naming
the dotted path and, when a near miss exists, the intended spelling. There are
no silent defaults for a typo -- a mistyped `dedup_similarity` that quietly
kept the default would corrupt memory for months before anyone noticed.

Typical use::

    from mechabrain.discovery import discover_vault
    from mechabrain.manifest import load_manifest

    paths = discover_vault()
    manifest = load_manifest(paths.config_file)
"""

from __future__ import annotations

import difflib
import math
import ntpath
import re
import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

import yaml

from . import __version__
from .contract import SPEC_VERSION, MemoryType, folder_for_type
from .errors import KernelTooOldError, ManifestError

__all__ = [
    "Manifest",
    "MechaBrainMeta",
    "AgentSpec",
    "ScopesSpec",
    "NamingSpec",
    "ZonesSpec",
    "TagNamespaces",
    "FrontmatterSpec",
    "EmbeddingSpec",
    "HybridSpec",
    "LinkExpansionSpec",
    "RetrievalSpec",
    "MaintenanceSpec",
    "load_manifest",
    "EMBEDDING_PROVIDERS",
    "VECTOR_STORES",
    "DEFAULT_PREFIXES",
    "GLOBAL_SCOPE",
]

#: Scope every memory falls back to when it belongs to no project (R6.5).
GLOBAL_SCOPE: Final[str] = "global"

#: Embedding backends the kernel ships. `hash` is deterministic and offline --
#: it exists for tests/CI and must never back a real deployment.
EMBEDDING_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"sentence-transformers", "http", "hash"}
)

#: Vector stores the kernel ships. `numpy` is the default: at personal-vault
#: scale (~1e4 chunks) brute-force cosine is <10ms and ANN buys nothing.
VECTOR_STORES: Final[frozenset[str]] = frozenset({"numpy", "lancedb", "sqlite-vec"})

DEFAULT_PREFIXES: Final[dict[MemoryType, str]] = {
    MemoryType.EPISODIC: "MEM",
    MemoryType.SEMANTIC: "INS",
    MemoryType.PROCEDURAL: "PROC",
    MemoryType.RESEARCH: "RES",
}

_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"^[^\s#][^\s]*$")
_VERSION_RE: Final[re.Pattern[str]] = re.compile(r"^\d+(\.\d+)*$")


# ══════════════════════════════════════════════════════════════════════
# Strict readers
# ══════════════════════════════════════════════════════════════════════
def _fail(path: str, problem: str, hint: str | None = None) -> "ManifestError":
    where = path or "<root>"
    return ManifestError(f"{where}: {problem}", hint=hint)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _check_unknown_keys(data: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    """Reject any key outside ``allowed``, suggesting the closest legal spelling (R5.1)."""
    allowed_list = list(allowed)
    for key in data:
        if key in allowed_list:
            continue
        near = difflib.get_close_matches(str(key), allowed_list, n=1, cutoff=0.6)
        hint = (
            f"did you mean {_join(path, near[0])!r}?"
            if near
            else f"valid keys here: {', '.join(sorted(allowed_list))}"
        )
        raise _fail(_join(path, str(key)), "unknown key", hint)


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _fail(path, f"expected a mapping, got {_type_name(value)}")
    for key in value:
        if not isinstance(key, str):
            raise _fail(path, f"keys must be strings, got {_type_name(key)} ({key!r})")
    return dict(value)


def _section(data: Mapping[str, Any], key: str, path: str) -> dict[str, Any]:
    return _as_mapping(data.get(key), _join(path, key))


def _get_str(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: str | None = None,
    *,
    allow_empty: bool = False,
) -> str:
    here = _join(path, key)
    if key not in data or data[key] is None:
        if default is None:
            raise _fail(here, "required key is missing")
        return default
    value = data[key]
    if not isinstance(value, str):
        raise _fail(
            here,
            f"expected a string, got {_type_name(value)} ({value!r})",
            "quote the value if YAML is coercing it",
        )
    if not allow_empty and not value.strip():
        raise _fail(here, "must not be empty")
    return value


def _get_bool(data: Mapping[str, Any], key: str, path: str, default: bool) -> bool:
    here = _join(path, key)
    if key not in data or data[key] is None:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise _fail(
            here,
            f"expected a boolean, got {_type_name(value)} ({value!r})",
            "use true or false",
        )
    return value


def _get_int(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    here = _join(path, key)
    if key not in data or data[key] is None:
        value: Any = default
    else:
        value = data[key]
    # bool is an int subclass; `hops: true` is a mistake, not the number 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(here, f"expected an integer, got {_type_name(value)} ({value!r})")
    if minimum is not None and value < minimum:
        raise _fail(here, f"must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise _fail(here, f"must be <= {maximum}, got {value}")
    return value


def _get_float(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
) -> float:
    here = _join(path, key)
    value: Any = default if key not in data or data[key] is None else data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(here, f"expected a number, got {_type_name(value)} ({value!r})")
    value = float(value)
    if minimum is not None:
        if exclusive_min and value <= minimum:
            raise _fail(here, f"must be > {minimum}, got {value}")
        if not exclusive_min and value < minimum:
            raise _fail(here, f"must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise _fail(here, f"must be <= {maximum}, got {value}")
    return value


def _get_str_list(
    data: Mapping[str, Any],
    key: str,
    path: str,
    default: Sequence[str] = (),
) -> list[str]:
    here = _join(path, key)
    if key not in data or data[key] is None:
        return list(default)
    value = data[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _fail(
            here,
            f"expected a list of strings, got {_type_name(value)} ({value!r})",
            "write a YAML list, e.g. [a, b]",
        )
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise _fail(f"{here}[{i}]", f"expected a string, got {_type_name(item)}")
        if not item.strip():
            raise _fail(f"{here}[{i}]", "must not be empty")
        out.append(item)
    return out


def _require_unique(values: Sequence[str], path: str, what: str) -> None:
    seen: set[str] = set()
    for i, value in enumerate(values):
        if value in seen:
            raise _fail(f"{path}[{i}]", f"duplicate {what}: {value!r}")
        seen.add(value)


def _require_slug(value: str, path: str, what: str) -> str:
    if not _SLUG_RE.match(value):
        raise _fail(
            path,
            f"{what} must be a lowercase slug (a-z, 0-9, '-', '_'), got {value!r}",
            f"try {_slugify(value)!r}",
        )
    return value


def _slugify(value: str) -> str:
    out = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower()).strip("-")
    return out or "agent"


def _require_relative_path(value: str, path: str) -> str:
    """Reject absolute paths anywhere in the manifest (R4.2).

    Covers POSIX roots, Windows drives/UNC and `~` expansion: a deployment is
    synced across machines and operating systems, so any of these breaks the
    moment the vault moves.
    """
    if value.startswith("~"):
        raise _fail(
            path,
            f"path must be relative to the vault root, got a home-expanded path {value!r}",
            "the vault root is the parent of mecha-brain/ -- write the path relative to it",
        )
    if Path(value).is_absolute() or ntpath.isabs(value) or value.startswith("\\\\"):
        raise _fail(
            path,
            f"path must be relative to the vault root, got absolute {value!r}",
            "R4.2: a deployment holds no absolute path -- it must survive being "
            "moved between machines and operating systems",
        )
    return value


def _parse_version(value: str, path: str) -> tuple[int, ...]:
    if not _VERSION_RE.match(value):
        raise _fail(
            path,
            f"expected a dotted numeric version like '0.1.0', got {value!r}",
        )
    return tuple(int(part) for part in value.split("."))


def _pad(version: tuple[int, ...], length: int = 3) -> tuple[int, ...]:
    return version + (0,) * (length - len(version))


def _check_template(
    template: str,
    path: str,
    allowed_fields: frozenset[str],
    required_fields: frozenset[str],
) -> set[str]:
    """Validate a `{placeholder}` naming template, returning the fields it uses."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise _fail(path, f"malformed template {template!r}: {exc}") from exc
    used = {name for _, name, _, _ in parsed if name}
    for name in sorted(used):
        if name not in allowed_fields:
            near = difflib.get_close_matches(name, sorted(allowed_fields), n=1, cutoff=0.6)
            hint = (
                f"did you mean {{{near[0]}}}?"
                if near
                else f"available placeholders: {', '.join('{' + f + '}' for f in sorted(allowed_fields))}"
            )
            raise _fail(path, f"unknown placeholder {{{name}}} in {template!r}", hint)
    missing = required_fields - used
    if missing:
        raise _fail(
            path,
            f"template {template!r} is missing required placeholder(s): "
            f"{', '.join('{' + f + '}' for f in sorted(missing))}",
        )
    if not template.endswith(".md"):
        raise _fail(path, f"template {template!r} must end with '.md'")
    return used


# ══════════════════════════════════════════════════════════════════════
# Sections (§5)
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MechaBrainMeta:
    """`mecha_brain:` -- the contract handshake."""

    spec_version: str = SPEC_VERSION
    kernel_min_version: str = "0.1.0"

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "MechaBrainMeta":
        _check_unknown_keys(data, ("spec_version", "kernel_min_version"), path)
        spec_version = _get_str(data, "spec_version", path)
        kernel_min_version = _get_str(data, "kernel_min_version", path)

        if _pad(_parse_version(spec_version, _join(path, "spec_version")), 2) != _pad(
            _parse_version(SPEC_VERSION, "<kernel>"), 2
        ):
            raise _fail(
                _join(path, "spec_version"),
                f"deployment declares spec {spec_version!r}, this kernel implements {SPEC_VERSION!r}",
                "upgrade the kernel, or migrate the deployment to this spec version",
            )

        required = _parse_version(kernel_min_version, _join(path, "kernel_min_version"))
        if _pad(required) > _pad(_parse_version(__version__, "<kernel>")):
            raise KernelTooOldError(
                f"deployment requires kernel >= {kernel_min_version}, "
                f"this kernel is {__version__}",
                hint="upgrade the kernel: uv tool upgrade mechabrain",
            )
        return cls(spec_version=spec_version, kernel_min_version=kernel_min_version)


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """One entry of the `agents:` registry.

    `id` is the **runtime** -- who executes, which `Episodic/` subfolder is
    writable, who is accountable. It is the only boundary the kernel can
    enforce. `profiles` are personas of that same runtime: provenance metadata
    and a search filter, never a write boundary (R6.6).

    `private_store` is informational only (§8.3): the kernel never manages it.
    It may be a description, `None`, or a mapping of profile -> description.
    """

    id: str
    display_name: str
    profiles: tuple[str, ...] = ()
    private_store: str | dict[str, str] | None = None

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "AgentSpec":
        _check_unknown_keys(
            data, ("id", "display_name", "profiles", "private_store"), path
        )
        agent_id = _require_slug(_get_str(data, "id", path), _join(path, "id"), "agent id")
        display_name = _get_str(data, "display_name", path, default=agent_id)

        profiles = _get_str_list(data, "profiles", path)
        _require_unique(profiles, _join(path, "profiles"), "profile")
        for i, profile in enumerate(profiles):
            _require_slug(profile, f"{_join(path, 'profiles')}[{i}]", "profile")

        private_store = cls._parse_private_store(
            data.get("private_store"), _join(path, "private_store"), profiles
        )
        return cls(
            id=agent_id,
            display_name=display_name,
            profiles=tuple(profiles),
            private_store=private_store,
        )

    @staticmethod
    def _parse_private_store(
        value: Any, path: str, profiles: Sequence[str]
    ) -> str | dict[str, str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            # `private_store: none` is the spec's way of writing "there is none".
            return None if value.strip().lower() in {"none", ""} else value
        if isinstance(value, Mapping):
            per_profile = _as_mapping(value, path)
            for key, description in per_profile.items():
                if key not in profiles:
                    near = difflib.get_close_matches(key, list(profiles), n=1, cutoff=0.6)
                    hint = (
                        f"did you mean {near[0]!r}?"
                        if near
                        else f"declared profiles: {', '.join(profiles) or '(none)'}"
                    )
                    raise _fail(
                        _join(path, key),
                        f"per-profile private_store names an undeclared profile {key!r}",
                        hint,
                    )
                if not isinstance(description, str):
                    raise _fail(
                        _join(path, key),
                        f"expected a string, got {_type_name(description)}",
                    )
            return {str(k): str(v) for k, v in per_profile.items()}
        raise _fail(
            path,
            f"expected a string, null, or a profile -> description mapping, "
            f"got {_type_name(value)}",
        )


@dataclass(frozen=True, slots=True)
class ScopesSpec:
    """`scopes:` -- the cross-contamination boundary (R6.5).

    An empty `known` means any slug is accepted.
    """

    known: tuple[str, ...] = ()
    default: str = GLOBAL_SCOPE

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "ScopesSpec":
        _check_unknown_keys(data, ("known", "default"), path)
        known = _get_str_list(data, "known", path)
        _require_unique(known, _join(path, "known"), "scope")
        for i, scope in enumerate(known):
            _require_slug(scope, f"{_join(path, 'known')}[{i}]", "scope")

        default = _require_slug(
            _get_str(data, "default", path, default=GLOBAL_SCOPE),
            _join(path, "default"),
            "scope",
        )
        if known and default not in known:
            near = difflib.get_close_matches(default, known, n=1, cutoff=0.6)
            raise _fail(
                _join(path, "default"),
                f"default scope {default!r} is not in scopes.known",
                f"did you mean {near[0]!r}?"
                if near
                else f"add {default!r} to scopes.known, or pick one of: {', '.join(known)}",
            )
        return cls(known=tuple(known), default=default)


@dataclass(frozen=True, slots=True)
class NamingSpec:
    """`naming:` -- filename templates and per-type prefixes."""

    note_name: str = "{date}_{prefix}_{slug}.md"
    dated_types: tuple[MemoryType, ...] = (
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.RESEARCH,
    )
    prefixes: dict[MemoryType, str] = field(default_factory=lambda: dict(DEFAULT_PREFIXES))
    proposal_name: str = "{date}_AI-PROPOSAL_{slug}.md"

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "NamingSpec":
        _check_unknown_keys(
            data, ("note_name", "dated_types", "prefixes", "proposal_name"), path
        )
        note_name = _get_str(data, "note_name", path, default="{date}_{prefix}_{slug}.md")
        note_fields = _check_template(
            note_name,
            _join(path, "note_name"),
            allowed_fields=frozenset({"date", "prefix", "slug"}),
            required_fields=frozenset({"slug"}),
        )

        raw_dated = _get_str_list(
            data, "dated_types", path, default=("episodic", "semantic", "research")
        )
        _require_unique(raw_dated, _join(path, "dated_types"), "type")
        dated_types: list[MemoryType] = []
        for i, name in enumerate(raw_dated):
            here = f"{_join(path, 'dated_types')}[{i}]"
            try:
                dated_types.append(MemoryType.parse(name))
            except ValueError as exc:
                near = difflib.get_close_matches(
                    name, [m.value for m in MemoryType], n=1, cutoff=0.6
                )
                raise _fail(
                    here, str(exc), f"did you mean {near[0]!r}?" if near else None
                ) from None
        if dated_types and "date" not in note_fields:
            raise _fail(
                _join(path, "dated_types"),
                f"types are declared dated ({', '.join(t.value for t in dated_types)}) "
                f"but naming.note_name {note_name!r} has no {{date}} placeholder",
                "add {date} to naming.note_name, or empty naming.dated_types",
            )

        prefixes = cls._parse_prefixes(_section(data, "prefixes", path), _join(path, "prefixes"))
        if "prefix" not in note_fields:
            # Prefixes exist to disambiguate filenames; without {prefix} they are inert.
            raise _fail(
                _join(path, "note_name"),
                f"template {note_name!r} has no {{prefix}} placeholder, so naming.prefixes "
                f"would never be used",
                "add {prefix} to naming.note_name",
            )

        proposal_name = _get_str(
            data, "proposal_name", path, default="{date}_AI-PROPOSAL_{slug}.md"
        )
        _check_template(
            proposal_name,
            _join(path, "proposal_name"),
            allowed_fields=frozenset({"date", "slug"}),
            required_fields=frozenset({"slug"}),
        )
        return cls(
            note_name=note_name,
            dated_types=tuple(dated_types),
            prefixes=prefixes,
            proposal_name=proposal_name,
        )

    @staticmethod
    def _parse_prefixes(data: Mapping[str, Any], path: str) -> dict[MemoryType, str]:
        _check_unknown_keys(data, [m.value for m in MemoryType], path)
        prefixes = dict(DEFAULT_PREFIXES)
        for memory_type in MemoryType:
            if memory_type.value not in data:
                continue
            here = _join(path, memory_type.value)
            value = _get_str(data, memory_type.value, path)
            if any(char in value for char in ' \t/\\'):
                raise _fail(
                    here,
                    f"prefix {value!r} must not contain whitespace or path separators",
                )
            prefixes[memory_type] = value
        return prefixes


@dataclass(frozen=True, slots=True)
class ZonesSpec:
    """`zones:` -- write boundaries. All paths are relative to the vault root (R4.2)."""

    proposals_dir: str = "mecha-brain/_inbox/"
    read_only_index: tuple[str, ...] = ()
    research_enabled: bool = True

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "ZonesSpec":
        _check_unknown_keys(
            data, ("proposals_dir", "read_only_index", "research_enabled"), path
        )
        proposals_dir = _require_relative_path(
            _get_str(data, "proposals_dir", path, default="mecha-brain/_inbox/"),
            _join(path, "proposals_dir"),
        )
        read_only_index = _get_str_list(data, "read_only_index", path)
        _require_unique(read_only_index, _join(path, "read_only_index"), "path")
        for i, folder in enumerate(read_only_index):
            _require_relative_path(folder, f"{_join(path, 'read_only_index')}[{i}]")
        return cls(
            proposals_dir=proposals_dir,
            read_only_index=tuple(read_only_index),
            research_enabled=_get_bool(data, "research_enabled", path, default=True),
        )


@dataclass(frozen=True, slots=True)
class TagNamespaces:
    """`frontmatter.tag_namespaces:` -- generates `mem/<type>` and `agent/<id>` tags."""

    memory: str = "mem"
    agent: str = "agent"

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "TagNamespaces":
        _check_unknown_keys(data, ("memory", "agent"), path)
        memory = _get_str(data, "memory", path, default="mem")
        agent = _get_str(data, "agent", path, default="agent")
        for name, value in (("memory", memory), ("agent", agent)):
            here = _join(path, name)
            if any(char in value for char in " \t#/"):
                raise _fail(
                    here,
                    f"tag namespace {value!r} must not contain whitespace, '#' or '/'",
                    "it is the prefix the kernel joins with '/', e.g. 'mem' -> 'mem/semantic'",
                )
        if memory == agent:
            raise _fail(
                path,
                f"memory and agent tag namespaces must differ (both are {memory!r})",
            )
        return cls(memory=memory, agent=agent)

    def memory_tag(self, memory_type: MemoryType | str) -> str:
        """Tag marking a note's memory type, e.g. ``mem/semantic``."""
        return f"{self.memory}/{MemoryType.parse(str(memory_type)).value}"

    def agent_tag(self, agent_id: str) -> str:
        """Tag marking a note's author runtime, e.g. ``agent/researcher``."""
        return f"{self.agent}/{agent_id}"


@dataclass(frozen=True, slots=True)
class FrontmatterSpec:
    """`frontmatter:` -- what agent notes must and must not carry (R6.1)."""

    denylist_keys: tuple[str, ...] = ()
    denylist_tags: tuple[str, ...] = ()
    tag_namespaces: TagNamespaces = field(default_factory=TagNamespaces)
    required_extra_tags: tuple[str, ...] = ()

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "FrontmatterSpec":
        _check_unknown_keys(
            data,
            ("denylist_keys", "denylist_tags", "tag_namespaces", "required_extra_tags"),
            path,
        )
        denylist_keys = _get_str_list(data, "denylist_keys", path)
        _require_unique(denylist_keys, _join(path, "denylist_keys"), "key")

        denylist_tags = _get_str_list(data, "denylist_tags", path)
        _require_unique(denylist_tags, _join(path, "denylist_tags"), "tag")

        required_extra_tags = _get_str_list(data, "required_extra_tags", path)
        _require_unique(required_extra_tags, _join(path, "required_extra_tags"), "tag")

        for name, tags in (
            ("denylist_tags", denylist_tags),
            ("required_extra_tags", required_extra_tags),
        ):
            for i, tag in enumerate(tags):
                here = f"{_join(path, name)}[{i}]"
                if not _TAG_RE.match(tag):
                    raise _fail(
                        here,
                        f"invalid tag {tag!r}",
                        "write tags without the leading '#' and without spaces",
                    )

        conflict = set(denylist_tags) & set(required_extra_tags)
        if conflict:
            raise _fail(
                path,
                f"tag(s) both required and denied: {', '.join(sorted(conflict))}",
            )
        return cls(
            denylist_keys=tuple(denylist_keys),
            denylist_tags=tuple(denylist_tags),
            tag_namespaces=TagNamespaces._parse(
                _section(data, "tag_namespaces", path), _join(path, "tag_namespaces")
            ),
            required_extra_tags=tuple(required_extra_tags),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """`retrieval.embedding:`."""

    provider: str = "sentence-transformers"
    model: str = "BAAI/bge-m3"

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "EmbeddingSpec":
        _check_unknown_keys(data, ("provider", "model"), path)
        provider = _get_str(data, "provider", path, default="sentence-transformers")
        if provider not in EMBEDDING_PROVIDERS:
            near = difflib.get_close_matches(provider, sorted(EMBEDDING_PROVIDERS), n=1)
            raise _fail(
                _join(path, "provider"),
                f"unknown embedding provider {provider!r}",
                f"did you mean {near[0]!r}?"
                if near
                else f"available providers: {', '.join(sorted(EMBEDDING_PROVIDERS))}",
            )
        return cls(provider=provider, model=_get_str(data, "model", path, default="BAAI/bge-m3"))


@dataclass(frozen=True, slots=True)
class HybridSpec:
    """`retrieval.hybrid:` -- fusion weights, which must sum to ~1."""

    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "HybridSpec":
        _check_unknown_keys(data, ("vector_weight", "bm25_weight"), path)
        vector_weight = _get_float(data, "vector_weight", path, 0.6, minimum=0.0, maximum=1.0)
        bm25_weight = _get_float(data, "bm25_weight", path, 0.4, minimum=0.0, maximum=1.0)
        total = vector_weight + bm25_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise _fail(
                path,
                f"vector_weight + bm25_weight must sum to 1.0, got {total:g} "
                f"({vector_weight:g} + {bm25_weight:g})",
                "the weights are a convex blend of two min-max normalised score lists",
            )
        return cls(vector_weight=vector_weight, bm25_weight=bm25_weight)


@dataclass(frozen=True, slots=True)
class LinkExpansionSpec:
    """`retrieval.link_expansion:` -- graph-lite over the *authored* graph (§7.1)."""

    default_hops: int = 1
    max_hops: int = 2

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "LinkExpansionSpec":
        _check_unknown_keys(data, ("default_hops", "max_hops"), path)
        max_hops = _get_int(data, "max_hops", path, 2, minimum=0)
        default_hops = _get_int(data, "default_hops", path, 1, minimum=0)
        if default_hops > max_hops:
            raise _fail(
                _join(path, "default_hops"),
                f"default_hops ({default_hops}) must be <= max_hops ({max_hops})",
            )
        return cls(default_hops=default_hops, max_hops=max_hops)


@dataclass(frozen=True, slots=True)
class RetrievalSpec:
    """`retrieval:`."""

    embedding: EmbeddingSpec = field(default_factory=EmbeddingSpec)
    hybrid: HybridSpec = field(default_factory=HybridSpec)
    contextual_retrieval: bool = True
    rerank: bool = False
    link_expansion: LinkExpansionSpec = field(default_factory=LinkExpansionSpec)
    store: str = "numpy"

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "RetrievalSpec":
        _check_unknown_keys(
            data,
            (
                "embedding",
                "hybrid",
                "contextual_retrieval",
                "rerank",
                "link_expansion",
                "store",
            ),
            path,
        )
        store = _get_str(data, "store", path, default="numpy")
        if store not in VECTOR_STORES:
            near = difflib.get_close_matches(store, sorted(VECTOR_STORES), n=1)
            raise _fail(
                _join(path, "store"),
                f"unknown vector store {store!r}",
                f"did you mean {near[0]!r}?"
                if near
                else f"available stores: {', '.join(sorted(VECTOR_STORES))}",
            )
        return cls(
            embedding=EmbeddingSpec._parse(
                _section(data, "embedding", path), _join(path, "embedding")
            ),
            hybrid=HybridSpec._parse(_section(data, "hybrid", path), _join(path, "hybrid")),
            contextual_retrieval=_get_bool(data, "contextual_retrieval", path, default=True),
            rerank=_get_bool(data, "rerank", path, default=False),
            link_expansion=LinkExpansionSpec._parse(
                _section(data, "link_expansion", path), _join(path, "link_expansion")
            ),
            store=store,
        )


@dataclass(frozen=True, slots=True)
class MaintenanceSpec:
    """`maintenance:` -- consolidation knobs (§9)."""

    decay_days: int = 90
    dedup_similarity: float = 0.92
    commit_prefix: str = "chore(ai-memory):"

    @classmethod
    def _parse(cls, data: Mapping[str, Any], path: str) -> "MaintenanceSpec":
        _check_unknown_keys(data, ("decay_days", "dedup_similarity", "commit_prefix"), path)
        return cls(
            decay_days=_get_int(data, "decay_days", path, 90, minimum=1),
            dedup_similarity=_get_float(
                data,
                "dedup_similarity",
                path,
                0.92,
                minimum=0.0,
                maximum=1.0,
                exclusive_min=True,
            ),
            commit_prefix=_get_str(data, "commit_prefix", path, default="chore(ai-memory):"),
        )


# ══════════════════════════════════════════════════════════════════════
# Manifest
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Manifest:
    """A parsed, validated `_meta/config.yaml` (§5).

    Construct through :func:`load_manifest`, :meth:`from_yaml` or
    :meth:`from_mapping` -- never by hand from untrusted data, since ``__init__``
    does not validate.
    """

    mecha_brain: MechaBrainMeta = field(default_factory=MechaBrainMeta)
    agents: tuple[AgentSpec, ...] = ()
    scopes: ScopesSpec = field(default_factory=ScopesSpec)
    naming: NamingSpec = field(default_factory=NamingSpec)
    zones: ZonesSpec = field(default_factory=ZonesSpec)
    frontmatter: FrontmatterSpec = field(default_factory=FrontmatterSpec)
    retrieval: RetrievalSpec = field(default_factory=RetrievalSpec)
    maintenance: MaintenanceSpec = field(default_factory=MaintenanceSpec)
    #: Where this manifest was read from, when it came from disk.
    source_path: Path | None = None

    _TOP_LEVEL_KEYS: ClassVar[tuple[str, ...]] = (
        "mecha_brain",
        "agents",
        "scopes",
        "naming",
        "zones",
        "frontmatter",
        "retrieval",
        "maintenance",
    )

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, source_path: Path | None = None
    ) -> "Manifest":
        """Validate ``data`` against §5 and build a manifest.

        Raises:
            ManifestError: unknown key, wrong type, out-of-range value, absolute
                path, or broken cross-reference. The message names the dotted
                key path (R5.1).
            KernelTooOldError: `kernel_min_version` exceeds this kernel (R4.5).
        """
        root = _as_mapping(data, "")
        if not root:
            raise _fail(
                "",
                "manifest is empty",
                "run `mechabrain init <vault>` to generate a default config.yaml",
            )
        _check_unknown_keys(root, cls._TOP_LEVEL_KEYS, "")

        if "mecha_brain" not in root:
            raise _fail(
                "mecha_brain",
                "required section is missing",
                'add:\n    mecha_brain:\n      spec_version: "0.1"\n      kernel_min_version: "0.1.0"',
            )

        manifest = cls(
            mecha_brain=MechaBrainMeta._parse(
                _section(root, "mecha_brain", ""), "mecha_brain"
            ),
            agents=cls._parse_agents(root.get("agents"), "agents"),
            scopes=ScopesSpec._parse(_section(root, "scopes", ""), "scopes"),
            naming=NamingSpec._parse(_section(root, "naming", ""), "naming"),
            zones=ZonesSpec._parse(_section(root, "zones", ""), "zones"),
            frontmatter=FrontmatterSpec._parse(
                _section(root, "frontmatter", ""), "frontmatter"
            ),
            retrieval=RetrievalSpec._parse(_section(root, "retrieval", ""), "retrieval"),
            maintenance=MaintenanceSpec._parse(
                _section(root, "maintenance", ""), "maintenance"
            ),
            source_path=source_path,
        )
        return manifest

    @classmethod
    def from_yaml(cls, text: str, *, source_path: Path | None = None) -> "Manifest":
        """Parse YAML ``text`` into a validated manifest."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            where = f" in {source_path}" if source_path else ""
            raise ManifestError(
                f"config.yaml is not valid YAML{where}: {exc}",
                hint="check indentation and quoting",
            ) from exc
        return cls.from_mapping(data if data is not None else {}, source_path=source_path)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Read and validate the manifest at ``path``."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ManifestError(
                f"no manifest at {path}",
                rule="R4.3",
                hint="run `mechabrain init <vault>` to create one",
            ) from exc
        except OSError as exc:
            raise ManifestError(f"cannot read manifest at {path}: {exc}") from exc
        return cls.from_yaml(text, source_path=path)

    @staticmethod
    def _parse_agents(value: Any, path: str) -> tuple[AgentSpec, ...]:
        if value is None:
            return ()
        if isinstance(value, Mapping) or not isinstance(value, Sequence) or isinstance(value, str):
            raise _fail(
                path,
                f"expected a list of agent entries, got {_type_name(value)}",
                "write a YAML list:\n    agents:\n      - id: example\n        display_name: Example",
            )
        agents = [
            AgentSpec._parse(_as_mapping(item, f"{path}[{i}]"), f"{path}[{i}]")
            for i, item in enumerate(value)
        ]
        _require_unique([a.id for a in agents], path, "agent id")
        return tuple(agents)

    # ── Queries ─────────────────────────────────────────────────────
    def agent_ids(self) -> tuple[str, ...]:
        """Every registered runtime id, in manifest order (R6.2)."""
        return tuple(agent.id for agent in self.agents)

    def agent(self, agent_id: str) -> AgentSpec:
        """The registry entry for ``agent_id``.

        Raises:
            ManifestError: the id is not registered; the message suggests the
                closest registered id (R6.2).
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        known = self.agent_ids()
        near = difflib.get_close_matches(agent_id, list(known), n=1, cutoff=0.6)
        raise ManifestError(
            f"unknown agent {agent_id!r}",
            rule="R6.2",
            hint=f"did you mean {near[0]!r}?"
            if near
            else f"registered agents: {', '.join(known) or '(none -- add one to agents: in config.yaml)'}",
        )

    def is_known_agent(self, agent_id: str) -> bool:
        """Whether ``agent_id`` is in the registry (non-raising form of :meth:`agent`)."""
        return agent_id in self.agent_ids()

    def profiles_of(self, agent_id: str) -> tuple[str, ...]:
        """Personas declared for ``agent_id``; empty tuple if it declares none (R6.6).

        Raises:
            ManifestError: the agent is not registered.
        """
        return self.agent(agent_id).profiles

    def is_known_scope(self, scope: str) -> bool:
        """Whether ``scope`` is writable under R6.5.

        An empty `scopes.known` accepts any non-empty slug -- that is the spec's
        "vazio = qualquer slug aceito", not a bypass of slug syntax.
        """
        if not scope:
            return False
        if not self.scopes.known:
            return bool(_SLUG_RE.match(scope))
        return scope in self.scopes.known

    def folder_for(self, memory_type: MemoryType | str) -> str:
        """Contractual folder for ``memory_type``, relative to ``mecha-brain/`` (§3).

        Episodic notes live one level deeper, under ``Episodic/<agent-id>/``;
        use :meth:`mechabrain.discovery.VaultPaths.episodic_for` for the full path.
        """
        return folder_for_type(memory_type)

    def prefix_for(self, memory_type: MemoryType | str) -> str:
        """Filename prefix for ``memory_type``, e.g. ``INS`` (`naming.prefixes`)."""
        return self.naming.prefixes[MemoryType.parse(str(memory_type))]

    def is_dated(self, memory_type: MemoryType | str) -> bool:
        """Whether filenames of ``memory_type`` carry `{date}` (`naming.dated_types`)."""
        return MemoryType.parse(str(memory_type)) in self.naming.dated_types

    def is_enabled(self, memory_type: MemoryType | str) -> bool:
        """Whether ``memory_type`` may be written.

        Only `research` is switchable, via `zones.research_enabled` (§3).
        """
        parsed = MemoryType.parse(str(memory_type))
        if parsed is MemoryType.RESEARCH:
            return self.zones.research_enabled
        return True


def load_manifest(path: Path) -> Manifest:
    """Read and validate the manifest at ``path``.

    Module-level alias of :meth:`Manifest.load`; prefer it at call sites for
    readability.
    """
    return Manifest.load(path)
