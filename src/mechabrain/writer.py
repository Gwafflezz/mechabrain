"""Write execution: the learning action of the decision cycle (spec §7.2).

The gate (:mod:`mechabrain.gate`) *evaluates*; this module *executes*. Once
:func:`mechabrain.gate.evaluate` approves a candidate, :func:`write` resolves its
filename from the manifest's `naming` template, assembles the §6 frontmatter,
writes the `.md` atomically to its contractual folder (R7.5), archives whatever
it supersedes (P8 -- archive, never delete), and reindexes it incrementally.

Division of labour, kept honest
===============================

* **Policy is the gate's.** :func:`write` never re-decides what
  :func:`~mechabrain.gate.evaluate` decided. A disabled `research` type, an
  unknown author, a blind near-duplicate, a missing `source:` -- all of these
  are the gate's rejections, surfaced verbatim in :class:`WriteResult`. This
  module adds no second opinion, so there is one place to read the §8.2 rules.
* **Retrieval and indexing are injected.** The dedup search the gate needs, and
  the incremental reindex step 8 asks for, both live in modules that sit *above*
  this one in the import graph. So, exactly as the gate takes a
  :class:`~mechabrain.gate.SearchFn`, :func:`write` takes an optional
  :class:`Indexer`. With no indexer wired in, the note is still written and
  archived correctly; only the derived index lags until the next reindex --
  which is safe, because the index is always rebuildable from the Markdown (P1).

Two entry points
================

:func:`write`
    The §7.2 ``memory.write``: the governed path into ``mecha-brain/``. Episodic
    goes to ``Episodic/<agent>/`` and is append-only -- a colliding path never
    overwrites, it spawns a fresh name (R6.3).

:func:`propose`
    The §7.2 ``memory.propose``: the *only* way an agent may affect a note
    outside ``mecha-brain/``. It writes a proposal into ``zones.proposals_dir``
    and **never touches the target file** -- a human applies the change.

Both refuse to overwrite: a resolved path that already exists is bumped to a
free name, so no write this module makes can destroy an existing note.
"""

from __future__ import annotations

import string
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Protocol

from .contract import (
    MARKDOWN_SUFFIX,
    STATUS_ACTIVE,
    STATUS_ARCHIVED,
    MemoryType,
)
from .discovery import VaultPaths
from .errors import DenylistViolation
from .gate import GateIssue, NearDuplicate, SearchFn, evaluate
from .graph import extract_wikilinks
from .index.store import INDEX_LOCK_FILE
from .locking import file_lock
from .manifest import Manifest
from .note import (
    Note,
    iter_notes,
    normalize_tags,
    note_id_for,
    wikilink_for,
)

__all__ = [
    "Indexer",
    "WriteResult",
    "ProposalResult",
    "write",
    "propose",
    "slugify",
    "EVIDENCE_HEADING",
    "SLUG_FALLBACK",
]

#: Heading of the evidence section :func:`write` renders into a procedural note
#: body from ``meta.evidence`` (§8.2 item 6). pt-BR: it is vault content read by
#: the deployment's humans and agents, not kernel diagnostics (rule: docs pt-BR).
EVIDENCE_HEADING: Final[str] = "## Evidência"

#: Slug used when a title reduces to nothing sluggable (e.g. all punctuation or
#: emoji). A note still needs a filename; a stable fallback beats an empty one.
SLUG_FALLBACK: Final[str] = "nota"

#: Characters a naming template uses to join placeholders. Stripped around an
#: omitted ``{date}`` so an atemporal type's name has no dangling separator.
_NAME_SEPARATORS: Final[str] = "-_. "

_LOCK_PURPOSE: Final[str] = "memory_write"


# ══════════════════════════════════════════════════════════════════════
# Injected indexing
# ══════════════════════════════════════════════════════════════════════
class Indexer(Protocol):
    """The incremental reindexer :func:`write` calls after a note lands (§7.2 step 8).

    Injected rather than imported, for the reason the gate injects its search:
    the real indexer builds on the note and manifest layers this module also
    builds on, and reindexing a single note is a superset of what a write needs.
    `memory_write` binds the daemon's indexer; a test binds a stub; ``None``
    skips the step, leaving only the derived index stale (P1: rebuildable).

    Contract: ``index_note`` takes a placed :class:`~mechabrain.note.Note` (its
    ``path`` set) and makes the index reflect that note's current bytes --
    whether it was created, or edited to `status: arquivado` by a supersede.
    """

    def index_note(self, note: Note) -> None: ...


# ══════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class WriteResult:
    """The outcome of :func:`write` -- the §7.2 ``{path, id}`` or ``{rejected, ...}``.

    Attributes:
        rejected: The gate refused the write; nothing was written. When true,
            ``path``/``note_id``/``wikilink`` are empty and ``reason`` +
            ``near_duplicates`` explain why.
        path: Where the note was written, or ``None`` when rejected.
        note_id: Stable id (basename without ``.md``), or ``""`` when rejected.
        wikilink: ``[[id]]`` for citing the new note (P7), or ``""``.
        reason: Every gate rejection, one per line; ``""`` on success.
        near_duplicates: Same-scope neighbours above `dedup_similarity` the gate
            found. Populated on a duplicate rejection (§8.2 item 2), and also on
            a *successful* write that declared `supersedes`/`merge` over them.
        warnings: The gate's instructed items (reusability, atomicity, …). May
            be non-empty on a successful write -- they never block.
        superseded: Wikilinks of notes this write archived (`status: arquivado`),
            per P8.
        superseded_missing: Supersede targets that resolved to no local note --
            not an error under eventual consistency (R7.6): the target may live
            on another machine and reconcile at the next consolidation.
        superseded_episodic: Supersede targets that resolve into ``Episodic/``
            and were therefore left untouched -- episodic is append-only (R6.3),
            so a correction links to the old entry without editing it.
        indexed: Whether an :class:`Indexer` was called for the new note.
    """

    rejected: bool
    path: Path | None = None
    note_id: str = ""
    wikilink: str = ""
    reason: str = ""
    near_duplicates: tuple[NearDuplicate, ...] = ()
    warnings: tuple[GateIssue, ...] = ()
    superseded: tuple[str, ...] = ()
    superseded_missing: tuple[str, ...] = ()
    superseded_episodic: tuple[str, ...] = ()
    indexed: bool = False

    @property
    def ok(self) -> bool:
        """Whether the note was written. The inverse of :attr:`rejected`."""
        return not self.rejected


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """The outcome of :func:`propose` -- a proposal note in ``zones.proposals_dir``.

    Attributes:
        path: Where the proposal was written.
        note_id: Stable id of the proposal note.
        wikilink: ``[[id]]`` of the proposal note.
        target: Vault-relative reference to the note the proposal is *about*.
            The proposal records it; the target file itself is never opened.
    """

    path: Path
    note_id: str
    wikilink: str
    target: str


# ══════════════════════════════════════════════════════════════════════
# write (§7.2 memory.write)
# ══════════════════════════════════════════════════════════════════════
def write(
    type: MemoryType | str,  # noqa: A002 -- the §7.2 `memory_write(type, ...)` parameter name
    content: str,
    meta: Mapping[str, Any],
    manifest: Manifest,
    vault: VaultPaths,
    search_fn: SearchFn | None = None,
    *,
    indexer: Indexer | None = None,
    today: date | None = None,
    lock: bool = True,
) -> WriteResult:
    """Execute a governed write into ``mecha-brain/`` (§7.2).

    Runs the §8.2 gate first and writes nothing if it rejects. On approval:
    resolves the filename from `naming` (dated unless the type is atemporal),
    builds the §6 frontmatter, renders `meta.evidence` into the body for a
    procedural note, writes atomically to the contractual folder (R7.5) without
    ever overwriting, archives what it supersedes (P8), and reindexes.

    Args:
        type: Target memory type. The `type` argument of `memory_write`.
        content: The note body, without frontmatter.
        meta: The frontmatter to carry (§6: `title`, `agent`, `scope`, `source`,
            `confidence`, optional `profile`/`supersedes`, optional `tags`) plus
            the gate-only keys `evidence` (§8.2 item 6) and `merge` (§8.2 item 2).
            Gate-only keys are consumed here, never written to frontmatter.
        manifest: Parsed deployment config -- the source of naming, tags,
            folders and thresholds (P6).
        vault: Discovered vault paths (R4.3). Supplies every write location.
        search_fn: Retrieval for the gate's dedup check. Required for
            `semantic`/`procedural`; unused for `episodic`/`research`.
        indexer: Incremental reindexer for the new and archived notes. ``None``
            leaves the derived index stale until the next reindex (P1).
        today: Date stamped into filenames and `created`/`modified`/
            `last_accessed`. Defaults to :meth:`datetime.date.today`; inject a
            fixed date in tests.
        lock: Hold the index write-lock for the whole read-modify-write cycle
            (R7.4 fallback). Pass ``False`` only when the caller already holds
            it -- ``FileLock`` deadlocks against a second instance of itself on
            one path in one process (see :mod:`mechabrain.locking`).

    Returns:
        A :class:`WriteResult`. ``rejected`` distinguishes the two §7.2 shapes.

    Raises:
        ValueError: `type` is not a memory type, or `search_fn` is missing for a
            gated type -- both are the gate's wiring errors, raised there.
        Exception: whatever ``search_fn`` or ``indexer`` raises propagates.
    """
    memory_type = MemoryType.parse(str(type))
    stamp = today if today is not None else date.today()

    if not lock:
        return _write_locked(
            memory_type, content, meta, manifest, vault, search_fn, indexer, stamp
        )
    with file_lock(vault.index_dir / INDEX_LOCK_FILE, purpose=_LOCK_PURPOSE):
        return _write_locked(
            memory_type, content, meta, manifest, vault, search_fn, indexer, stamp
        )


def _write_locked(
    memory_type: MemoryType,
    content: str,
    meta: Mapping[str, Any],
    manifest: Manifest,
    vault: VaultPaths,
    search_fn: SearchFn | None,
    indexer: Indexer | None,
    stamp: date,
) -> WriteResult:
    gate = evaluate(memory_type, meta, content, manifest, search_fn=search_fn)
    if not gate.approved:
        return WriteResult(
            rejected=True,
            reason=gate.reason,
            near_duplicates=gate.near_duplicates,
            warnings=gate.warnings,
        )

    agent = _text(meta.get("agent"))
    scope = _text(meta.get("scope"))
    supersedes_ids = _supersedes_ids(meta.get("supersedes"))

    note = Note(
        frontmatter=_frontmatter(memory_type, meta, manifest, agent, scope, supersedes_ids, stamp),
        body=_body(memory_type, content, meta.get("evidence")),
    )
    directory = (
        vault.episodic_for(agent)
        if memory_type is MemoryType.EPISODIC
        else vault.folder_for(memory_type)
    )
    target = _unique_path(directory, _filename(memory_type, note.title, manifest, stamp))
    note.write(target)

    archived, missing, episodic = _apply_supersedes(
        supersedes_ids, note.note_id, vault, manifest, indexer
    )

    indexed = indexer is not None
    if indexer is not None:
        indexer.index_note(note)

    return WriteResult(
        rejected=False,
        path=note.path,
        note_id=note.note_id,
        wikilink=note.wikilink,
        near_duplicates=gate.near_duplicates,
        warnings=gate.warnings,
        superseded=tuple(wikilink_for(note_id) for note_id in archived),
        superseded_missing=tuple(wikilink_for(note_id) for note_id in missing),
        superseded_episodic=tuple(wikilink_for(note_id) for note_id in episodic),
        indexed=indexed,
    )


# ══════════════════════════════════════════════════════════════════════
# propose (§7.2 memory.propose)
# ══════════════════════════════════════════════════════════════════════
def propose(
    target_path: Path | str,
    proposed_change: str,
    rationale: str,
    meta: Mapping[str, Any],
    manifest: Manifest,
    vault: VaultPaths,
    *,
    today: date | None = None,
) -> ProposalResult:
    """Record a proposed change to a note outside ``mecha-brain/`` (§7.2).

    The only sanctioned way for an agent to affect a note it may not write
    directly (P4). Writes a proposal note into ``zones.proposals_dir`` carrying
    the rationale, the proposed change and a link back to the target -- and
    **never opens or edits the target file**. A human reviews and applies it.

    Args:
        target_path: The note the proposal is about. Referenced only; never read.
        proposed_change: The suggested edit, diff or new text.
        rationale: Why the change is proposed (P7: provenance is mandatory).
        meta: Authorship: `agent` (required, must be registered -- R6.2),
            optional `profile`, `scope`, `source`, `confidence`.
        manifest: Supplies `zones.proposals_dir`, `naming.proposal_name`, tag
            namespaces and denylists (P6).
        vault: Discovered vault paths.
        today: Date stamp; defaults to today.

    Returns:
        A :class:`ProposalResult` naming the proposal note that was written.

    Raises:
        ManifestError: `agent` is missing or not in the registry (R6.2).
        DenylistViolation: an assembled key or tag is on a denylist (R6.1).
    """
    stamp = today if today is not None else date.today()

    agent = _text(meta.get("agent"))
    manifest.agent(agent)  # R6.2: fail loud on an unknown or missing author.
    profile = _text(meta.get("profile"))
    if profile and profile not in manifest.profiles_of(agent):
        raise DenylistViolation(
            f"agent {agent!r} declares no profile {profile!r}",
            rule="R6.6",
            hint=f"profiles of {agent!r}: {', '.join(manifest.profiles_of(agent)) or '(none)'}",
        )

    target_ref = _target_ref(vault, target_path)
    target_id = note_id_for(target_path)
    frontmatter = _proposal_frontmatter(meta, manifest, agent, profile, target_ref, target_id, stamp)
    _reject_denylisted(frontmatter, manifest)

    note = Note(
        frontmatter=frontmatter,
        body=_proposal_body(target_id, target_ref, rationale, proposed_change),
    )
    directory = vault.resolve(manifest.zones.proposals_dir)
    filename = _fill_name_template(
        manifest.naming.proposal_name,
        {"date": stamp.isoformat(), "slug": slugify(target_id)},
    )
    target = _unique_path(directory, filename)
    note.write(target)

    return ProposalResult(
        path=note.path,
        note_id=note.note_id,
        wikilink=note.wikilink,
        target=target_ref,
    )


# ══════════════════════════════════════════════════════════════════════
# Frontmatter (§6)
# ══════════════════════════════════════════════════════════════════════
def _frontmatter(
    memory_type: MemoryType,
    meta: Mapping[str, Any],
    manifest: Manifest,
    agent: str,
    scope: str,
    supersedes_ids: Sequence[str],
    stamp: date,
) -> dict[str, Any]:
    """Assemble the §6 frontmatter in the spec's key order.

    Only §6 keys are emitted: the gate-only `evidence`/`merge` and any other
    stray `meta` keys never reach disk, so a write cannot smuggle a field past
    the gate's denylist check by carrying it through here.
    """
    frontmatter: dict[str, Any] = {
        "title": _title(meta),
        "tags": _tags(memory_type, meta, manifest, agent),
        "created": stamp,
        "modified": stamp,
        "agent": agent,
    }
    profile = _text(meta.get("profile"))
    if profile:
        frontmatter["profile"] = profile
    frontmatter["scope"] = scope
    frontmatter["source"] = _text(meta.get("source"))
    frontmatter["confidence"] = _confidence(meta)
    frontmatter["last_accessed"] = stamp
    if memory_type is MemoryType.PROCEDURAL:
        # §8.2 item 6 guaranteed `meta.evidence` before this point, and evidence
        # attests a successful run as of this write -- so the write date is an
        # honest `last_tested:`. The §9.4 stale-procedural report reads it; a
        # human (or a superseding write) refreshes it after a re-test.
        frontmatter["last_tested"] = stamp
    if supersedes_ids:
        frontmatter["supersedes"] = _supersedes_value(supersedes_ids)
    frontmatter["status"] = STATUS_ACTIVE
    return frontmatter


def _tags(
    memory_type: MemoryType, meta: Mapping[str, Any], manifest: Manifest, agent: str
) -> list[str]:
    """The §6 tags: `mem/<type>`, `agent/<id>`, the required extras, then the
    author's own -- deduplicated, order preserved."""
    namespaces = manifest.frontmatter.tag_namespaces
    generated = [
        namespaces.memory_tag(memory_type),
        namespaces.agent_tag(agent),
        *manifest.frontmatter.required_extra_tags,
    ]
    return _dedupe([*generated, *normalize_tags(meta.get("tags"))])


def _proposal_frontmatter(
    meta: Mapping[str, Any],
    manifest: Manifest,
    agent: str,
    profile: str,
    target_ref: str,
    target_id: str,
    stamp: date,
) -> dict[str, Any]:
    """Frontmatter for a proposal note.

    A proposal is not one of the four memory types, so it carries no `mem/<type>`
    tag; it carries the author tag and the deployment's required extras, plus a
    `target:` pointer for a reviewer to open.
    """
    namespaces = manifest.frontmatter.tag_namespaces
    frontmatter: dict[str, Any] = {
        "title": f"Proposta de mudança — {target_id}",
        "tags": _dedupe([namespaces.agent_tag(agent), *manifest.frontmatter.required_extra_tags]),
        "created": stamp,
        "modified": stamp,
        "agent": agent,
    }
    if profile:
        frontmatter["profile"] = profile
    frontmatter["scope"] = _text(meta.get("scope")) or manifest.scopes.default
    frontmatter["source"] = _text(meta.get("source"))
    frontmatter["confidence"] = _confidence(meta)
    frontmatter["last_accessed"] = stamp
    frontmatter["target"] = target_ref
    frontmatter["status"] = STATUS_ACTIVE
    return frontmatter


def _reject_denylisted(frontmatter: Mapping[str, Any], manifest: Manifest) -> None:
    """R6.1 for a note the kernel assembles itself: refuse a denied key or tag."""
    spec = manifest.frontmatter
    denied_keys = sorted(set(frontmatter) & set(spec.denylist_keys))
    if denied_keys:
        raise DenylistViolation(
            f"frontmatter key(s) {', '.join(repr(k) for k in denied_keys)} "
            f"are forbidden by frontmatter.denylist_keys",
            rule="R6.1",
        )
    denied_tags = sorted(set(normalize_tags(frontmatter.get("tags"))) & set(spec.denylist_tags))
    if denied_tags:
        raise DenylistViolation(
            f"tag(s) {', '.join(repr(t) for t in denied_tags)} "
            f"are forbidden by frontmatter.denylist_tags",
            rule="R6.1",
        )


# ══════════════════════════════════════════════════════════════════════
# Body
# ══════════════════════════════════════════════════════════════════════
def _body(memory_type: MemoryType, content: str, evidence: Any) -> str:
    """The note body, with the evidence section appended for a procedural note.

    §8.2 item 6 makes `meta.evidence` mandatory for procedural and asks it be
    cited in the body; the gate guarantees it is present, so rendering it here
    turns "evidence cited in the body" into a mechanical fact.
    """
    body = content.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if memory_type is not MemoryType.PROCEDURAL:
        return body
    rendered = _render_evidence(evidence)
    if not rendered:
        return body
    section = f"{EVIDENCE_HEADING}\n\n{rendered}"
    return f"{body}\n\n{section}" if body else section


def _render_evidence(value: Any) -> str:
    """Render `meta.evidence` as Markdown: a bullet list, or a paragraph."""
    if isinstance(value, (list, tuple, set)):
        items = [_text(item) for item in value if _text(item)]
        return "\n".join(f"- {item}" for item in items)
    return _text(value)


def _proposal_body(target_id: str, target_ref: str, rationale: str, proposed_change: str) -> str:
    """The proposal note body: a callout naming the target, then rationale and change."""
    return (
        f"> [!note] Proposta de mudança gerada por agente\n"
        f"> Alvo: [[{target_id}]] (`{target_ref}`)\n\n"
        f"## Justificativa\n\n"
        f"{rationale.strip()}\n\n"
        f"## Mudança proposta\n\n"
        f"{proposed_change.strip()}"
    )


# ══════════════════════════════════════════════════════════════════════
# Naming
# ══════════════════════════════════════════════════════════════════════
def _filename(memory_type: MemoryType, title: str, manifest: Manifest, stamp: date) -> str:
    """Resolve a note filename from `naming.note_name`.

    `naming.dated_types` decides whether the name carries `{date}`; a type left
    out of it (procedural by default) is atemporal, and its `{date}` placeholder
    -- and the separator that joined it -- drop out entirely.
    """
    values = {
        "prefix": manifest.prefix_for(memory_type),
        "slug": slugify(title),
    }
    if manifest.is_dated(memory_type):
        values["date"] = stamp.isoformat()
    return _fill_name_template(manifest.naming.note_name, values)


def _fill_name_template(template: str, values: Mapping[str, str]) -> str:
    """Fill a `{placeholder}` naming template, collapsing an omitted field.

    A placeholder absent from ``values`` -- an atemporal type's ``{date}`` -- is
    removed together with the single separator that attached it, so
    ``{date}_{prefix}_{slug}.md`` renders ``PROC_x.md``, never ``_PROC_x.md``.
    The template grammar is already validated by the manifest, so no field here
    is unexpected.
    """
    parts: list[str] = []
    strip_next_leading_sep = False
    for literal, field_name, _spec, _conv in string.Formatter().parse(template):
        if literal:
            if strip_next_leading_sep:
                literal = literal.lstrip(_NAME_SEPARATORS)
            parts.append(literal)
        strip_next_leading_sep = False
        if field_name is None:
            continue
        value = values.get(field_name)
        if value is not None:
            parts.append(value)
            continue
        # Omitted field: drop the separator that joined it -- trailing on what we
        # already emitted, else leading on the next literal.
        if parts and parts[-1] and parts[-1].rstrip(_NAME_SEPARATORS) != parts[-1]:
            parts[-1] = parts[-1].rstrip(_NAME_SEPARATORS)
        else:
            strip_next_leading_sep = True
    return "".join(parts)


def slugify(text: str) -> str:
    """Reduce ``text`` to a filesystem- and wikilink-safe slug.

    Transliterates accents (``Configuração`` -> ``configuracao``) so a
    pt-BR title yields an ASCII filename, lowercases, and collapses every run of
    non-alphanumerics to a single ``-``. Falls back to :data:`SLUG_FALLBACK`
    when nothing sluggable remains.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    out: list[str] = []
    prev_dash = False
    for char in ascii_text:
        if char.isalnum():
            out.append(char)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug or SLUG_FALLBACK


def _unique_path(directory: Path, filename: str) -> Path:
    """A path in ``directory`` for ``filename`` that does not exist yet.

    Never overwrites (R6.3 for episodic, and prudence everywhere else): a taken
    name is bumped ``name-2.md``, ``name-3.md`` … until a free one is found.
    """
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    if filename.endswith(MARKDOWN_SUFFIX):
        stem, suffix = filename[: -len(MARKDOWN_SUFFIX)], MARKDOWN_SUFFIX
    else:
        stem, suffix = filename, ""
    ordinal = 2
    while True:
        candidate = directory / f"{stem}-{ordinal}{suffix}"
        if not candidate.exists():
            return candidate
        ordinal += 1


# ══════════════════════════════════════════════════════════════════════
# Supersedes (§6, P8)
# ══════════════════════════════════════════════════════════════════════
def _apply_supersedes(
    supersedes_ids: Sequence[str],
    new_note_id: str,
    vault: VaultPaths,
    manifest: Manifest,
    indexer: Indexer | None,
) -> tuple[list[str], list[str], list[str]]:
    """Archive every note this write supersedes (P8: archive, never delete).

    Returns ``(archived, missing, episodic)`` ids. A target under ``Episodic/``
    is left untouched -- episodic is append-only (R6.3), so the new note links
    to it without editing it. A target with no local note is *missing*, not an
    error: under eventual consistency it may live on another machine (R7.6).
    """
    if not supersedes_ids:
        return [], [], []

    by_id = _memory_note_index(vault, manifest)
    archived: list[str] = []
    missing: list[str] = []
    episodic: list[str] = []
    for note_id in supersedes_ids:
        if note_id == new_note_id:
            continue
        path = by_id.get(note_id)
        if path is None:
            missing.append(note_id)
        elif _is_episodic(vault, path):
            episodic.append(note_id)
        else:
            _archive(path, indexer)
            archived.append(note_id)
    return archived, missing, episodic


def _memory_note_index(vault: VaultPaths, manifest: Manifest) -> dict[str, Path]:
    """Map note id -> path across the enabled memory folders, first path winning."""
    by_id: dict[str, Path] = {}
    for memory_type in MemoryType:
        if not manifest.is_enabled(memory_type):
            continue
        for note in iter_notes(vault.folder_for(memory_type)):
            if note.path is not None:
                by_id.setdefault(note.note_id, note.path)
    return by_id


def _archive(path: Path, indexer: Indexer | None) -> None:
    """Flip a superseded note to `status: arquivado` in place, then reindex it."""
    note = Note.load(path)
    if _text(note.get("status")) == STATUS_ARCHIVED:
        return
    note.frontmatter["status"] = STATUS_ARCHIVED
    note.write()
    if indexer is not None:
        indexer.index_note(note)


def _is_episodic(vault: VaultPaths, path: Path) -> bool:
    try:
        path.relative_to(vault.episodic_dir)
    except ValueError:
        return False
    return True


def _supersedes_ids(value: Any) -> list[str]:
    """Note ids named by a `supersedes:` value (§6): wikilink, bare id, or a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return _dedupe(sid for item in value for sid in _supersedes_ids(item))
    text = str(value).strip()
    if not text:
        return []
    if "[[" in text:
        return _dedupe(note_id_for(target) for target in extract_wikilinks(text))
    return [note_id_for(text)]


def _supersedes_value(supersedes_ids: Sequence[str]) -> str | list[str]:
    """The `supersedes:` frontmatter value: a single wikilink, or a list of them."""
    links = [wikilink_for(note_id) for note_id in supersedes_ids]
    return links[0] if len(links) == 1 else links


# ══════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════
def _title(meta: Mapping[str, Any]) -> str:
    """`title:` for the frontmatter, defaulting to a readable stand-in.

    The gate does not require a title, so a write may arrive without one; a note
    still needs a human-facing title, so fall back rather than emit a blank.
    """
    return _text(meta.get("title")) or "Sem título"


def _confidence(meta: Mapping[str, Any]) -> str:
    """`confidence:`, defaulting to `medium` -- the honest middle when unstated.

    `high` is the level §8.2 item 4 guards; defaulting to it would launder an
    unverified claim, and defaulting to `low` would understate a real one.
    """
    return _text(meta.get("confidence")) or "medium"


def _target_ref(vault: VaultPaths, target_path: Path | str) -> str:
    """A vault-relative POSIX reference to ``target_path`` (R4.2: never absolute)."""
    path = Path(target_path)
    try:
        return vault.relative(path) if path.is_absolute() else path.as_posix()
    except ValueError:
        return path.as_posix()


def _text(value: Any) -> str:
    """A frontmatter value as trimmed text; ``""`` for None/blank."""
    return "" if value is None else str(value).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication, dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
