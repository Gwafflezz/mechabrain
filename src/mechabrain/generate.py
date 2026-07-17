"""Artifacts derived from the manifest: `config.yaml`, `schema.md`, `AGENTS.md`,
`index.md`, `indices/<scope>.md`, `hot.md` (§9, §10, R6.4, R8.1).

Everything here is one half of the same idea: **the manifest is the only source
of deployment truth, so every document that states a boundary is generated from
it**. Documentation of fronteiras maintained by hand drifts from the config it
describes; a generated document cannot.

Two audiences, two languages. The kernel's code and errors are English; the text
these templates produce is read by the humans and agents of a deployment and is
written in **pt-BR**, the language of the spec this implements.

What each renderer owns:

===========================  ==========================================
:func:`render_default_config` `_meta/config.yaml` for a fresh `init` --
                              every §5 key, with the spec's comments.
:func:`render_schema`         `_meta/schema.md` (R6.4) -- §6 rendered with
                              this manifest's real denylists, agents,
                              scopes and tag namespaces.
:func:`render_agents_md`      `AGENTS.md`, splicing a **managed block**
                              into whatever a human wrote around it (§10).
:func:`render_index`          `index.md` + `indices/<scope>.md` shards (§9.5).
:func:`render_hot`            `hot.md`, one section per active scope (R8.2).
===========================  ==========================================

Templates live in ``mechabrain/templates/`` and are filled with
:class:`string.Template` -- no jinja2, which is not a dependency and would buy
nothing here: these documents have no loops a f-string cannot express.

**Nothing generated carries a timestamp.** Regenerating an unchanged deployment
must produce byte-identical output, otherwise every `consolidate` run would
commit a no-op diff to `index.md` and the vault's history would fill with noise.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Final

from . import __version__
from .contract import (
    INDICES_DIR,
    MANAGED_BLOCK_BEGIN,
    MANAGED_BLOCK_END,
    MARKDOWN_SUFFIX,
    SPEC_VERSION,
    STATUS_ACTIVE,
    CONFIDENCE_LEVELS,
    STATUS_ARCHIVED,
    STATUS_DEPRECATED,
    MemoryType,
    type_for_folder,
)
from .discovery import VaultPaths
from .errors import MechabrainError
from .manifest import GLOBAL_SCOPE, Manifest
from .note import Note, write_atomic

__all__ = [
    "render_default_config",
    "render_schema",
    "render_managed_block",
    "merge_managed_block",
    "render_agents_md",
    "render_index",
    "render_hot",
    "render_initial_index",
    "render_initial_hot",
    "IndexRender",
    "write_default_config",
    "write_schema",
    "write_agents_md",
    "write_index",
    "write_hot",
    "SHARD_LINE_THRESHOLD",
    "HOT_SECTION_MAX_ENTRIES",
    "INDEX_TYPES",
]

#: `index.md` shards by scope once it grows past this many lines (§9.5). The
#: spec says "~200": the number is a readability budget, not a hard invariant.
SHARD_LINE_THRESHOLD: Final[int] = 200

#: Ceiling of memories listed per scope section of `hot.md` (R8.2, "~15").
#: hot.md is a cache of attention: a section nobody can read at a glance has
#: stopped being one.
HOT_SECTION_MAX_ENTRIES: Final[int] = 15

#: Types `index.md` maps. Episodic is a journal, not a claim about the world,
#: and Research is long-form: neither belongs in a one-line-per-memory MOC (§9.5).
INDEX_TYPES: Final[tuple[MemoryType, ...]] = (MemoryType.SEMANTIC, MemoryType.PROCEDURAL)

#: Statuses that keep a memory out of the generated maps (§9.3, §9.4). They stay
#: searchable behind an explicit filter -- consolidation archives, never deletes (P8).
_HIDDEN_STATUSES: Final[frozenset[str]] = frozenset({STATUS_ARCHIVED, STATUS_DEPRECATED})

_TEMPLATE_DIR: Final[str] = "templates"
_NONE: Final[str] = "_(nenhuma)_"
_EMPTY_INDEX: Final[str] = "_Nenhuma memória ativa ainda._"
_EMPTY_HOT: Final[str] = (
    "_Nenhum escopo ativo ainda. A primeira consolidação depois da primeira "
    "escrita preenche este arquivo._"
)


# ══════════════════════════════════════════════════════════════════════
# Templates
# ══════════════════════════════════════════════════════════════════════
def _render_template(name: str, /, **fields: object) -> str:
    """Fill the packaged template ``name`` with ``fields``.

    Uses ``substitute``, never ``safe_substitute``: a placeholder the caller
    forgot must fail loudly here rather than ship a literal ``$scope`` into a
    user's vault (R5.1).
    """
    raw = files(__package__).joinpath(_TEMPLATE_DIR, name).read_text(encoding="utf-8")
    try:
        return string.Template(raw).substitute(fields)
    except KeyError as exc:
        raise MechabrainError(
            f"template {name!r} uses placeholder ${exc.args[0]}, which the "
            f"renderer does not provide",
            hint="templates and mechabrain.generate change together",
        ) from None
    except ValueError as exc:
        raise MechabrainError(
            f"template {name!r} is malformed: {exc}",
            hint="a literal '$' in a template must be written '$$'",
        ) from exc


# ══════════════════════════════════════════════════════════════════════
# Small formatters
# ══════════════════════════════════════════════════════════════════════
def _inline_code_list(values: Iterable[str], *, empty: str = _NONE) -> str:
    """``a, b`` as inline code, or an italic "none" -- never an empty cell."""
    rendered = ", ".join(f"`{value}`" for value in values)
    return rendered or empty


def _cell(value: str | None) -> str:
    """A table cell that is never blank and never breaks the row."""
    text = (value or "").strip().replace("|", "\\|")
    return text or "—"


def _private_store_cell(agent_private_store: str | dict[str, str] | None) -> str:
    """Render `agents[].private_store`, which §8.3 allows to be per profile."""
    if agent_private_store is None:
        return "—"
    if isinstance(agent_private_store, Mapping):
        return "; ".join(
            f"`{profile}`: {description}"
            for profile, description in agent_private_store.items()
        )
    return str(agent_private_store)


def _agents_table(manifest: Manifest) -> str:
    """The registry as a Markdown table, or an honest warning when it is empty."""
    if not manifest.agents:
        return (
            "> [!danger] Registry vazio\n"
            "> Nenhum agente declarado em `agents:` do manifest — e o kernel recusa\n"
            "> autor desconhecido (R6.2), então **toda escrita será rejeitada**.\n"
            "> Declare ao menos um agente e rode `mechabrain sync`."
        )
    header = (
        "| `agent` (runtime) | Nome | `profile` (personas) | Episodic | Store privado |\n"
        "|---|---|---|---|---|"
    )
    rows = [
        "| `{id}` | {name} | {profiles} | `Episodic/{id}/` | {store} |".format(
            id=agent.id,
            name=_cell(agent.display_name),
            profiles=_inline_code_list(agent.profiles, empty="—"),
            store=_cell(_private_store_cell(agent.private_store)),
        )
        for agent in manifest.agents
    ]
    return "\n".join([header, *rows])


def _scopes_inline(manifest: Manifest) -> str:
    """The legal `scope:` values, spelling out the open case."""
    if not manifest.scopes.known:
        return "qualquer slug (`scopes.known` está vazio)"
    return _inline_code_list(manifest.scopes.known)


def _scopes_paragraph(manifest: Manifest) -> str:
    if not manifest.scopes.known:
        return (
            "`scopes.known` está vazio: **qualquer slug minúsculo é aceito** como "
            f"`scope:`. O default é `{manifest.scopes.default}`."
        )
    return (
        f"Escopos válidos: {_inline_code_list(manifest.scopes.known)}. "
        f"O default é `{manifest.scopes.default}`. Um slug fora desta lista é recusado."
    )


def _naming_rows(manifest: Manifest) -> str:
    rows = []
    for memory_type in MemoryType:
        enabled = "" if manifest.is_enabled(memory_type) else " _(desabilitado)_"
        rows.append(
            f"| `{memory_type.value}`{enabled} | `{manifest.folder_for(memory_type)}/` "
            f"| `{manifest.prefix_for(memory_type)}` |"
        )
    return "\n".join(rows)


def _tags_example(manifest: Manifest) -> str:
    """The `tags:` line §6 shows, with this deployment's real namespaces."""
    namespaces = manifest.frontmatter.tag_namespaces
    tags = [
        namespaces.memory_tag(MemoryType.SEMANTIC),
        f"{namespaces.agent}/<id>",
        *manifest.frontmatter.required_extra_tags,
    ]
    return ", ".join(tags)


# ══════════════════════════════════════════════════════════════════════
# config.yaml (§5)
# ══════════════════════════════════════════════════════════════════════
def render_default_config() -> str:
    """The default `_meta/config.yaml` for a fresh deployment (§5).

    Every §5 key is present with the spec's own comments, because a manifest is
    read far more often than it is written and a key nobody can see is a key
    nobody sets. The two departures from the §5 listing are deliberate defaults
    of this kernel:

    * ``retrieval.store: numpy`` -- brute-force cosine, no heavy dependency. At
      personal-vault scale (~1e4 chunks) it answers in under 10ms and ANN buys
      nothing; ``lancedb``/``sqlite-vec`` stay available behind an extra.
    * ``kernel_min_version`` is this kernel's version, so R4.5 has something
      truthful to compare against.

    The result parses back through :meth:`Manifest.from_yaml` -- generating a
    config the validator would reject is the one bug this module must not have.
    """
    namespaces = Manifest().frontmatter.tag_namespaces
    return _render_template(
        "config.yaml.tmpl",
        spec_version=SPEC_VERSION,
        kernel_min_version=__version__,
        memory_ns=namespaces.memory,
        agent_ns=namespaces.agent,
    )


# ══════════════════════════════════════════════════════════════════════
# _meta/schema.md (R6.4)
# ══════════════════════════════════════════════════════════════════════
def render_schema(manifest: Manifest) -> str:
    """`_meta/schema.md`: §6 rendered against ``manifest`` (R6.4).

    The human- and agent-readable face of the frontmatter contract, always
    consistent with the config because it is derived from it -- never edited by
    hand.
    """
    namespaces = manifest.frontmatter.tag_namespaces
    return _render_template(
        "schema.md.tmpl",
        tags_example=_tags_example(manifest),
        confidence_values="|".join(CONFIDENCE_LEVELS),
        status_values="|".join((STATUS_ACTIVE, STATUS_ARCHIVED, STATUS_DEPRECATED)),
        agent_ids_inline=_inline_code_list(
            manifest.agent_ids(), empty="_(registry vazio)_"
        ),
        scopes_inline=_scopes_inline(manifest),
        scope_default=manifest.scopes.default,
        memory_ns=namespaces.memory,
        agent_ns=namespaces.agent,
        memory_tags_inline=_inline_code_list(
            namespaces.memory_tag(mtype) for mtype in MemoryType
        ),
        agent_tags_inline=_inline_code_list(
            (namespaces.agent_tag(agent_id) for agent_id in manifest.agent_ids()),
            empty=f"_(registry vazio — nenhuma tag `{namespaces.agent}/` possível)_",
        ),
        required_extra_tags_inline=_inline_code_list(
            manifest.frontmatter.required_extra_tags
        ),
        denylist_keys_inline=_inline_code_list(manifest.frontmatter.denylist_keys),
        denylist_tags_inline=_inline_code_list(manifest.frontmatter.denylist_tags),
        agents_table=_agents_table(manifest),
        scopes_paragraph=_scopes_paragraph(manifest),
        note_name=manifest.naming.note_name,
        proposal_name=manifest.naming.proposal_name,
        dated_types_inline=_inline_code_list(
            (mtype.value for mtype in manifest.naming.dated_types),
            empty="_(nenhum — nenhum nome carrega data)_",
        ),
        naming_rows=_naming_rows(manifest),
        decay_days=manifest.maintenance.decay_days,
        dedup_similarity=manifest.maintenance.dedup_similarity,
    )


# ══════════════════════════════════════════════════════════════════════
# AGENTS.md and its managed block (§10)
# ══════════════════════════════════════════════════════════════════════
def render_managed_block(manifest: Manifest) -> str:
    """The managed block of `AGENTS.md`, markers included, no trailing newline.

    Holds exactly what drifts: boundaries (P4), the §8.1 routing tree, the §8.2
    gate checklist, the agent/profile registry, denylists, paths and the
    normative advice to filter by scope. It is honest about the gate: items the
    kernel cannot check without an LLM are marked as the agent's judgement, not
    dressed up as enforcement.
    """
    namespaces = manifest.frontmatter.tag_namespaces
    research_enabled = manifest.zones.research_enabled
    block = _render_template(
        "agents.md.tmpl",
        begin_marker=MANAGED_BLOCK_BEGIN,
        end_marker=MANAGED_BLOCK_END,
        research_cell="escreve" if research_enabled else "**desabilitado**",
        research_routing=(
            "Gate completo."
            if research_enabled
            else "**Desabilitado neste deployment** (`zones.research_enabled: false`): "
            "vá para o passo 6."
        ),
        proposals_dir=manifest.zones.proposals_dir,
        read_only_index_inline=_inline_code_list(
            manifest.zones.read_only_index,
            empty="_(nenhuma — só `mecha-brain/` é indexado)_",
        ),
        scopes_inline=_scopes_inline(manifest),
        scope_default=manifest.scopes.default,
        default_hops=manifest.retrieval.link_expansion.default_hops,
        max_hops=manifest.retrieval.link_expansion.max_hops,
        dedup_similarity=manifest.maintenance.dedup_similarity,
        agents_table=_agents_table(manifest),
        memory_ns=namespaces.memory,
        agent_ns=namespaces.agent,
        required_extra_tags_inline=_inline_code_list(
            manifest.frontmatter.required_extra_tags
        ),
        denylist_keys_inline=_inline_code_list(manifest.frontmatter.denylist_keys),
        denylist_tags_inline=_inline_code_list(manifest.frontmatter.denylist_tags),
        note_name=manifest.naming.note_name,
        proposal_name=manifest.naming.proposal_name,
        decay_days=manifest.maintenance.decay_days,
        commit_prefix=manifest.maintenance.commit_prefix,
    )
    return block.strip("\n")


def _marker_offsets(text: str, marker: str) -> list[int]:
    offsets: list[int] = []
    start = text.find(marker)
    while start != -1:
        offsets.append(start)
        start = text.find(marker, start + len(marker))
    return offsets


def _ambiguous(problem: str, hint: str) -> MechabrainError:
    return MechabrainError(
        f"AGENTS.md has {problem}, so the managed block cannot be replaced "
        f"without risking hand-written text",
        rule="§10",
        hint=hint,
    )


def merge_managed_block(existing: str, block: str) -> str:
    """Splice ``block`` into ``existing``, preserving everything around it (§10).

    This is the whole reason the block exists: the kernel owns what it generates
    and **nothing else**. Text before the begin marker and after the end marker
    is copied through byte for byte, so a human's own sections survive every
    ``mechabrain sync``.

    A file with no markers keeps all of its content and gets the block appended
    -- never overwritten. Ambiguity (a lone marker, markers out of order, a
    marker written twice) raises instead of guessing: guessing wrong here means
    deleting something a human wrote.
    """
    text = existing.replace("\r\n", "\n").replace("\r", "\n")
    begins = _marker_offsets(text, MANAGED_BLOCK_BEGIN)
    ends = _marker_offsets(text, MANAGED_BLOCK_END)

    if not begins and not ends:
        head = text.rstrip("\n")
        return f"{head}\n\n{block}\n" if head else f"{block}\n"

    if len(begins) > 1 or len(ends) > 1:
        raise _ambiguous(
            f"{len(begins)} '{MANAGED_BLOCK_BEGIN}' and {len(ends)} "
            f"'{MANAGED_BLOCK_END}' markers",
            "keep exactly one pair; delete the extra markers, or the whole "
            "block to have it regenerated",
        )
    if not begins:
        raise _ambiguous(
            f"an unmatched '{MANAGED_BLOCK_END}' marker",
            f"add the missing '{MANAGED_BLOCK_BEGIN}', or delete the stray "
            f"end marker to have the block appended fresh",
        )
    if not ends:
        raise _ambiguous(
            f"an unmatched '{MANAGED_BLOCK_BEGIN}' marker",
            f"add the missing '{MANAGED_BLOCK_END}', or delete the stray "
            f"begin marker to have the block appended fresh",
        )
    if ends[0] < begins[0]:
        raise _ambiguous(
            "its markers in the wrong order (end before begin)",
            f"the block runs '{MANAGED_BLOCK_BEGIN}' first, "
            f"'{MANAGED_BLOCK_END}' last",
        )

    return text[: begins[0]] + block + text[ends[0] + len(MANAGED_BLOCK_END) :]


def render_agents_md(manifest: Manifest, existing: str | None = None) -> str:
    """`AGENTS.md` for ``manifest``, merged into ``existing`` when there is one.

    Args:
        manifest: The deployment's manifest.
        existing: Current file content, if the file is already there. ``None``
            generates the file from scratch (the block and nothing else).

    Returns:
        The full file content. Idempotent: feeding the result back as
        ``existing`` returns it unchanged.
    """
    block = render_managed_block(manifest)
    if existing is None:
        return f"{block}\n"
    return merge_managed_block(existing, block)


# ══════════════════════════════════════════════════════════════════════
# Reading notes for the maps
# ══════════════════════════════════════════════════════════════════════
def _as_date(value: Any) -> date | None:
    """Coerce a frontmatter date, which YAML may hand over already parsed."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _type_of(note: Note, manifest: Manifest) -> MemoryType | None:
    """The memory type of ``note``: from its tags, else from its folder.

    Tags first because they are the manifest's own vocabulary
    (``<memory_ns>/<type>``) and travel with the note if it is moved; the folder
    is the fallback for a note whose tags a human trimmed.
    """
    namespaces = manifest.frontmatter.tag_namespaces
    for memory_type in MemoryType:
        if note.has_tag(namespaces.memory_tag(memory_type)):
            return memory_type
    if note.path is None:
        return None
    for folder in list(note.path.parents)[:2]:
        try:
            return type_for_folder(folder.name)
        except KeyError:
            continue
    return None


def _status_of(note: Note) -> str:
    """`status:`, defaulting to active -- a note without one is not archived."""
    value = note.get("status")
    return str(value).strip() if value not in (None, "") else STATUS_ACTIVE


def _scope_of(note: Note, manifest: Manifest) -> str:
    value = note.get("scope")
    scope = str(value).strip() if value not in (None, "") else ""
    return scope or manifest.scopes.default


@dataclass(frozen=True, slots=True)
class _Entry:
    """One memory, reduced to what a map line shows."""

    note_id: str
    title: str
    scope: str
    memory_type: MemoryType
    prefix: str
    agent: str
    confidence: str
    recency: date

    @property
    def sort_key(self) -> tuple[str, str]:
        return (self.title.casefold(), self.note_id)

    @property
    def recency_key(self) -> tuple[date, str, str]:
        return (self.recency, *self.sort_key)

    def index_line(self) -> str:
        """``- [[id]] — Title · INS · agent · high`` (P7: the link is the point)."""
        parts = [part for part in (self.prefix, self.agent, self.confidence) if part]
        return f"- [[{self.note_id}]] — {self.title} · {' · '.join(parts)}"

    def hot_line(self) -> str:
        """Same shape as :meth:`index_line`, dated instead of scored."""
        stamp = self.recency.isoformat() if self.recency != date.min else ""
        parts = [part for part in (self.prefix, self.agent, stamp) if part]
        return f"- [[{self.note_id}]] — {self.title} · {' · '.join(parts)}"


def _entries(
    memories: Iterable[Note],
    manifest: Manifest,
    *,
    types: Sequence[MemoryType] | None = None,
) -> list[_Entry]:
    """Reduce notes to :class:`_Entry`, dropping what the maps must not show.

    Skipped: archived and deprecated notes (§9.3/§9.4), notes of an unwanted
    type, and notes with no path (nothing to link to -- a map line that cannot
    be opened is worse than no line).
    """
    wanted = set(types) if types is not None else set(MemoryType)
    entries: list[_Entry] = []
    for note in memories:
        if note.path is None or _status_of(note) in _HIDDEN_STATUSES:
            continue
        memory_type = _type_of(note, manifest)
        if memory_type is None or memory_type not in wanted:
            continue
        recency = (
            _as_date(note.get("last_accessed"))
            or _as_date(note.get("modified"))
            or _as_date(note.get("created"))
            or date.min
        )
        entries.append(
            _Entry(
                note_id=note.note_id,
                title=note.title,
                scope=_scope_of(note, manifest),
                memory_type=memory_type,
                prefix=manifest.prefix_for(memory_type),
                agent=str(note.get("agent") or "").strip(),
                confidence=str(note.get("confidence") or "").strip(),
                recency=recency,
            )
        )
    return entries


def _group_by_scope(entries: Iterable[_Entry]) -> dict[str, list[_Entry]]:
    grouped: dict[str, list[_Entry]] = {}
    for entry in entries:
        grouped.setdefault(entry.scope, []).append(entry)
    return grouped


def _scope_order(scopes: Iterable[str], manifest: Manifest) -> list[str]:
    """Manifest order first, then alphabetical, with the global scope last.

    `scopes.known` is the order the deployment chose to think in; global goes
    last because it is the fallback, not a project.
    """
    known = list(manifest.scopes.known)

    def rank(scope: str) -> tuple[int, int, str]:
        if scope == GLOBAL_SCOPE:
            return (2, 0, scope)
        if scope in known:
            return (0, known.index(scope), scope)
        return (1, 0, scope)

    return sorted(set(scopes), key=rank)


# ══════════════════════════════════════════════════════════════════════
# index.md + indices/<scope>.md (§9.5)
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class IndexRender:
    """Result of :func:`render_index`.

    Attributes:
        master: Content of `index.md`.
        shards: ``scope -> content`` of `indices/<scope>.md`. Empty while the
            index is small enough to live in one file.
        sharded: Whether the master was reduced to a master-of-scopes (§9.5).
            The global scope is never sharded out: it stays listed in the master,
            since a memory that belongs to no project has no shard to live in.
    """

    master: str
    shards: dict[str, str] = field(default_factory=dict)
    sharded: bool = False


def _index_sections(grouped: Mapping[str, list[_Entry]], manifest: Manifest) -> str:
    blocks = []
    for scope in _scope_order(grouped, manifest):
        entries = sorted(grouped[scope], key=lambda entry: entry.sort_key)
        lines = [entry.index_line() for entry in entries]
        blocks.append(f"## {scope}\n\n" + "\n".join(lines))
    return "\n\n".join(blocks) if blocks else _EMPTY_INDEX


def _index_document(body: str, manifest: Manifest, threshold: int) -> str:
    return _render_template(
        "index.md.tmpl",
        body=body,
        threshold=threshold,
        global_scope=GLOBAL_SCOPE,
    )


def render_index(
    memories: Iterable[Note],
    manifest: Manifest,
    *,
    shard_threshold: int = SHARD_LINE_THRESHOLD,
) -> IndexRender:
    """Render `index.md` and, past ``shard_threshold`` lines, its scope shards (§9.5).

    One line per **active** `semantic`/`procedural` memory, grouped by scope.
    Archived and deprecated notes are left out: they stay searchable behind an
    explicit filter, but the map is of what is alive (§9.3).

    Sharding is decided on the rendered result, not guessed from a count: the
    flat document is produced first and only replaced by a master-of-scopes if
    it actually exceeds the budget. Past the threshold, each non-global scope
    moves to ``indices/<scope>.md`` and the master keeps one line per scope plus
    the `global` memories in full.

    Args:
        memories: Notes to map. Anything the kernel can read works -- this
            renderer filters, so a caller may hand it the whole vault.
        manifest: Source of scope order, prefixes and tag namespaces.
        shard_threshold: Line budget for `index.md` (§9.5's "~200 linhas").

    Returns:
        An :class:`IndexRender`. Deterministic and timestamp-free: an unchanged
        vault re-renders byte-identically.
    """
    grouped = _group_by_scope(_entries(memories, manifest, types=INDEX_TYPES))
    flat = _index_document(_index_sections(grouped, manifest), manifest, shard_threshold)
    if len(flat.splitlines()) <= shard_threshold:
        return IndexRender(master=flat)

    shard_scopes = [scope for scope in _scope_order(grouped, manifest) if scope != GLOBAL_SCOPE]
    blocks: list[str] = []
    if shard_scopes:
        rows = [
            f"- **`{scope}`** — {len(grouped[scope])} memória(s) ativa(s) → "
            f"[{INDICES_DIR}/{scope}{MARKDOWN_SUFFIX}]({INDICES_DIR}/{scope}{MARKDOWN_SUFFIX})"
            for scope in shard_scopes
        ]
        blocks.append("## Escopos\n\n" + "\n".join(rows))
    if GLOBAL_SCOPE in grouped:
        blocks.append(
            _index_sections({GLOBAL_SCOPE: grouped[GLOBAL_SCOPE]}, manifest)
        )

    shards = {
        scope: _render_template(
            "index_shard.md.tmpl",
            scope=scope,
            threshold=shard_threshold,
            body=_index_sections({scope: grouped[scope]}, manifest),
        )
        for scope in shard_scopes
    }
    return IndexRender(
        master=_index_document("\n\n".join(blocks), manifest, shard_threshold),
        shards=shards,
        sharded=True,
    )


def render_initial_index(manifest: Manifest) -> str:
    """`index.md` for a vault with no memories yet -- what `init` writes (§10)."""
    return render_index((), manifest).master


# ══════════════════════════════════════════════════════════════════════
# hot.md (§8.4, R8.2)
# ══════════════════════════════════════════════════════════════════════
def render_hot(
    memories: Iterable[Note],
    manifest: Manifest,
    *,
    active_scopes: Sequence[str] | None = None,
    max_entries: int = HOT_SECTION_MAX_ENTRIES,
) -> str:
    """Render `hot.md`: one section per active scope, newest first (R8.2).

    "O foco atual" is not one thing when several projects are live, so there is
    no single list -- there is a section per scope, each capped, each pointing at
    real notes. Ranking is by recency (`last_accessed`, else `modified`, else
    `created`), which is the only "attention" signal the kernel can measure
    without an LLM.

    Args:
        memories: Candidate notes. Archived and deprecated ones are dropped.
        manifest: Source of scope order, prefixes and tag namespaces.
        active_scopes: Scopes the consolidator judged active (recent write or
            access), in the order to render. ``None`` falls back to every scope
            that has a live memory, most recently touched first -- correct for a
            first run, but the consolidator knows better and should say so.
        max_entries: Ceiling per section (R8.2's "~15 linhas").

    Returns:
        The file content. Timestamp-free, so an unchanged vault re-renders
        byte-identically and consolidation commits nothing.
    """
    grouped = _group_by_scope(_entries(memories, manifest))
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry.recency_key, reverse=True)

    if active_scopes is None:
        # Most recently touched scope first, ties broken alphabetically -- which
        # `reverse=True` would break backwards.
        ordered = sorted(
            grouped,
            key=lambda scope: ((date.max - grouped[scope][0].recency).days, scope),
        )
    else:
        ordered = [scope for scope in active_scopes if grouped.get(scope)]

    blocks: list[str] = []
    for scope in ordered:
        entries = grouped[scope]
        lines = [entry.hot_line() for entry in entries[:max_entries]]
        if len(entries) > max_entries:
            lines.append(
                f"- _… mais {len(entries) - max_entries} em `{scope}` — "
                f"veja [[index]] ou busque com `filters: {{scope: {scope}}}`_"
            )
        blocks.append(f"## {scope}\n\n" + "\n".join(lines))

    return _render_template(
        "hot.md.tmpl",
        body="\n\n".join(blocks) if blocks else _EMPTY_HOT,
        max_entries=max_entries,
    )


def render_initial_hot(manifest: Manifest) -> str:
    """`hot.md` for a vault with no memories yet -- what `init` writes (§10)."""
    return render_hot((), manifest)


# ══════════════════════════════════════════════════════════════════════
# Writers
# ══════════════════════════════════════════════════════════════════════
def write_default_config(paths: VaultPaths, *, overwrite: bool = False) -> Path:
    """Write `_meta/config.yaml`, keeping an existing one (§10: `init` is idempotent).

    The manifest is the one file in the deployment a human owns end to end.
    Rerunning `init` must never clobber it, so this returns the existing path
    untouched unless ``overwrite`` says otherwise.
    """
    if paths.config_file.exists() and not overwrite:
        return paths.config_file
    return write_atomic(paths.config_file, render_default_config())


def write_schema(paths: VaultPaths, manifest: Manifest) -> Path:
    """Write `_meta/schema.md`, always regenerating it (R6.4: it is derived)."""
    return write_atomic(paths.schema_file, render_schema(manifest))


def write_agents_md(paths: VaultPaths, manifest: Manifest) -> Path:
    """Regenerate the managed block of `AGENTS.md` in place (§10).

    Reads whatever is there, replaces only the block, writes atomically (R7.5).
    Free-form sections outside the markers survive.

    Raises:
        MechabrainError: the file's markers are ambiguous. See
            :func:`merge_managed_block`.
    """
    existing = (
        paths.agents_file.read_text(encoding="utf-8")
        if paths.agents_file.is_file()
        else None
    )
    return write_atomic(paths.agents_file, render_agents_md(manifest, existing))


def write_hot(
    paths: VaultPaths,
    memories: Iterable[Note],
    manifest: Manifest,
    *,
    active_scopes: Sequence[str] | None = None,
    max_entries: int = HOT_SECTION_MAX_ENTRIES,
) -> Path:
    """Write `hot.md` (R8.1: only the consolidator calls this)."""
    return write_atomic(
        paths.hot_file,
        render_hot(memories, manifest, active_scopes=active_scopes, max_entries=max_entries),
    )


def write_index(
    paths: VaultPaths,
    memories: Iterable[Note],
    manifest: Manifest,
    *,
    shard_threshold: int = SHARD_LINE_THRESHOLD,
) -> IndexRender:
    """Write `index.md` and its shards, pruning the shards that no longer apply.

    `indices/` is generated on demand and owned entirely by the consolidator
    (§3, R8.1), so a shard for a scope that no longer has active memories -- or
    every shard, when the index shrinks back under the threshold -- is deleted.
    A stale shard is worse than a missing one: it is a map of memories that have
    moved.

    Returns:
        The :class:`IndexRender` that was written.
    """
    rendered = render_index(memories, manifest, shard_threshold=shard_threshold)
    write_atomic(paths.index_file, rendered.master)

    if rendered.shards:
        paths.indices_dir.mkdir(parents=True, exist_ok=True)
    for scope, content in rendered.shards.items():
        write_atomic(paths.scope_index(scope), content)

    if paths.indices_dir.is_dir():
        for stale in sorted(paths.indices_dir.glob(f"*{MARKDOWN_SUFFIX}")):
            if stale.is_file() and stale.stem not in rendered.shards:
                stale.unlink()
    return rendered
