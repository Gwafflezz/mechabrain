"""The write gate: the evaluation stage of the decision cycle (spec §8.2).

`memory_write` runs every candidate note through :func:`evaluate` before it
touches the disk. In CoALA terms this is *evaluation* before a *learning
action*: memory without curation becomes noise (P5).

Honesty about what a gate can enforce
=====================================

The §8.2 checklist mixes two kinds of item, and this module keeps them apart on
purpose. **The kernel never calls an LLM**, so it implements what is mechanically
verifiable and *reports* the rest -- it does not pretend to enforce judgement by
asking the agent to tick a boolean it will always tick.

**Enforced** -- these reject the write (:attr:`GateResult.rejections`):

=================================  ===========================================
Check                              Why it is mechanical
=================================  ===========================================
§8.2 item 2 -- already exists?      a similarity score against same-scope notes
                                    is a number compared to a threshold
§8.2 item 4 -- source declared?     `source:` is either non-empty or it is not
§8.2 item 5 -- scoped?              `scope:` is either in the manifest or not
§8.2 item 6 -- procedural tested?   `meta.evidence` is either present or not
§8.2 item 7 -- clean?               set intersection with the denylists
R6.2 -- known author                membership in the manifest registry
R6.6 -- known profile               membership in the author's profile list
§3 -- type enabled                  `zones.research_enabled`
=================================  ===========================================

**Instructed** -- these only warn (:attr:`GateResult.warnings`), because code
cannot decide them:

* §8.2 item 1 -- *reusable beyond this session?* There is no signal in the text
  that separates a durable insight from a session note. This is the author's
  call; the gate restates the obligation and returns it. `AGENTS.md` carries the
  same rule in imperative form -- that is where item 1 actually lives.
* §8.2 item 3 -- *atomic?* Heuristics only (see :func:`_atomicity_warning`): a
  very long body, or several sibling top-level sections, *suggest* more than one
  insight. A long note may still be one idea, and a short one may hide three.
* §8.2 item 4b -- `confidence: high` demands verification or a primary source.
  Whether a source is primary is a judgement; the gate can only notice that no
  evidence was declared alongside the claim.

Enforcing a judgement item would mean trusting a self-report, which is not
enforcement -- it is a boolean the agent sets to `true` to get past the gate.

Routing (§8.1)
==============

Only `Semantic/` and `Procedural/` face the full checklist -- that is the
literal scope of §8.2 (:data:`FULL_GATE_TYPES`). `Episodic/` is a diary, not
truth: §8.1 item 3 routes it to a direct write, so it is never deduped, never
asked for evidence and never warned about atomicity. `Research/` is likewise
outside §8.2, and a report is non-atomic by construction.

Both still face the §6 *schema* invariants -- denylists (R6.1), a known author
(R6.2), a known profile (R6.6) and a valid scope (R6.5). Those are not gate
items: they hold for every note the kernel writes, whatever its type. A note
with an unknown `agent:` is a broken registry reference, not a curation call.

Dependency direction
====================

The dedup check needs retrieval, but this module must not import it: `search`
imports the note/manifest layer that `memory_write` also builds on, and the gate
sits between them. So the search is **injected** -- see :class:`SearchFn`.

Typical use::

    result = evaluate("semantic", meta, body, manifest, search_fn=search)
    result.raise_if_rejected()
    for warning in result.warnings:
        log(warning.message)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from .contract import MemoryType
from .errors import GateRejected
from .manifest import Manifest
from .note import normalize_tags, note_id_for

__all__ = [
    "GateIssue",
    "GateResult",
    "NearDuplicate",
    "SearchFn",
    "evaluate",
    "FULL_GATE_TYPES",
    "DEDUP_CANDIDATES",
    "DEDUP_QUERY_CHARS",
    "ATOMIC_BODY_CHARS",
    "ATOMIC_MAX_SECTIONS",
]

#: Types the §8.2 checklist governs. Everything else takes the §6 schema checks
#: only -- see the module docstring on routing.
FULL_GATE_TYPES: Final[frozenset[MemoryType]] = frozenset(
    {MemoryType.SEMANTIC, MemoryType.PROCEDURAL}
)

#: How many same-scope neighbours the internal dedup search asks for. Small: the
#: gate needs the *nearest* notes, not a reading list -- anything past the first
#: handful is far below `dedup_similarity` by construction.
DEDUP_CANDIDATES: Final[int] = 5

#: Prefix of title+body used as the dedup query. Embedding models truncate long
#: inputs anyway, and a note's opening states its claim (§8.2 item 3).
DEDUP_QUERY_CHARS: Final[int] = 2000

#: Body length above which the atomicity heuristic fires (§8.2 item 3).
ATOMIC_BODY_CHARS: Final[int] = 4000

#: Sibling top-level sections tolerated before the atomicity heuristic fires.
#: Two is deliberate: a playbook legitimately has "Steps" and "Evidence"
#: (§8.2 item 6), while three sibling sections in one note usually means three
#: notes.
ATOMIC_MAX_SECTIONS: Final[int] = 2

#: Hit fields read as a similarity, in order of preference. `similarity` must be
#: a raw cosine in [0, 1] against the query; `score` is the fallback for a
#: search that reports only its fused rank score. See :class:`SearchFn`.
SIMILARITY_FIELDS: Final[tuple[str, ...]] = ("similarity", "score")

_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})[ \t]+\S", re.MULTILINE)
_CODE_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


class SearchFn(Protocol):
    """The retrieval callable the gate is given (§8.2 item 2, "internal search").

    Injected rather than imported: see the module docstring. `memory_write`
    binds the real `memory_search` to it; a test binds a stub.

    Contract:

    * called as ``search_fn(query, k=..., filters={"scope": ..., "type": ...})``;
    * returns a sequence of hits, each a mapping or an object, carrying at least
      ``wikilink`` and a similarity field (:data:`SIMILARITY_FIELDS`);
    * hits already honour `filters` -- the gate does not re-filter, so a
      `search_fn` that ignores `scope` would silently dedup across scopes, which
      R6.5 forbids.

    **The similarity must be calibrated against `maintenance.dedup_similarity`.**
    A min-max normalised fusion score is not: it maps the best hit to 1.0
    whatever it is, so every write into a non-empty scope would look like a
    duplicate. A search that fuses and normalises must therefore expose the raw
    cosine as ``similarity`` alongside its ranked ``score``.
    """

    def __call__(
        self,
        query: str,
        *,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Any]: ...


# ══════════════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GateIssue:
    """One gate finding: a rejection or a warning.

    Attributes:
        check: Stable machine id, e.g. ``"duplicate"``. Callers and tests match
            on this; `message` is prose and may be reworded.
        rule: Spec citation, e.g. ``"§8.2 item 2"`` or ``"R6.1"`` -- the clause
            to grep for.
        message: What is wrong, or (for a warning) what the author must decide.
        hint: The next action. Never a restatement of `message`.
    """

    check: str
    rule: str
    message: str
    hint: str | None = None

    def __str__(self) -> str:
        text = f"[{self.rule}] {self.message}"
        return f"{text}\n  hint: {self.hint}" if self.hint else text


@dataclass(frozen=True, slots=True)
class NearDuplicate:
    """A same-scope note close enough to the candidate to demand a decision.

    Returned so the author can pick one of the three §8.2 item 2 outcomes:
    declare `supersedes`, merge, or drop the write.
    """

    id: str
    wikilink: str
    similarity: float
    title: str = ""
    path: str = ""

    @classmethod
    def from_hit(cls, hit: Any) -> "NearDuplicate":
        """Build from one `search_fn` hit; see :class:`SearchFn` for the shape."""
        wikilink = _hit_field(hit, "wikilink")
        path = _hit_field(hit, "path")
        note_id = _hit_field(hit, "id")
        if note_id is None and path is not None:
            note_id = note_id_for(str(path))
        if wikilink is None and note_id is not None:
            wikilink = f"[[{note_id}]]"
        if wikilink is None:
            raise ValueError(
                "search hit carries neither 'wikilink', 'id' nor 'path'; "
                "provenance is mandatory (R7.1)"
            )
        return cls(
            id=str(note_id) if note_id is not None else note_id_for(str(wikilink).strip("[]")),
            wikilink=str(wikilink),
            similarity=_similarity_of(hit),
            title=str(_hit_field(hit, "title") or ""),
            path=str(path) if path is not None else "",
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    """The verdict of :func:`evaluate`.

    Attributes:
        approved: True iff nothing mechanically checkable failed. Warnings never
            affect it -- they are the author's items, not the kernel's.
        rejections: Enforced failures. Empty iff `approved`.
        warnings: Instructed items the author owns. May be non-empty on an
            approved write; that is the normal case.
        near_duplicates: Same-scope neighbours above `dedup_similarity`.
            Populated whenever the dedup search found any -- including when the
            author already declared `supersedes`/`merge` and so was *not*
            rejected, since the caller may still want to report them.
    """

    approved: bool
    rejections: tuple[GateIssue, ...] = ()
    warnings: tuple[GateIssue, ...] = ()
    near_duplicates: tuple[NearDuplicate, ...] = ()

    @property
    def reason(self) -> str:
        """Every rejection, one per line. ``""`` when approved."""
        return "\n".join(str(issue) for issue in self.rejections)

    def raise_if_rejected(self) -> None:
        """Raise :class:`~mechabrain.errors.GateRejected` unless approved.

        The exception carries the near-duplicate wikilinks so an agent can act
        on them without a second search (§7.2).
        """
        if self.approved:
            return
        first = self.rejections[0]
        raise GateRejected(
            self.reason,
            rule=first.rule,
            hint=first.hint,
            near_duplicates=[duplicate.wikilink for duplicate in self.near_duplicates],
        )


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
def evaluate(
    type: MemoryType | str,  # noqa: A002 -- the §7.2 `memory_write(type, ...)` parameter name
    meta: Mapping[str, Any],
    body: str,
    manifest: Manifest,
    search_fn: SearchFn | None = None,
) -> GateResult:
    """Run the §8.2 gate over a candidate note.

    Never writes, never reads the vault, and never calls an LLM: given the same
    inputs and the same `search_fn` it returns the same verdict.

    Which checks run depends on `type` -- see the module docstring on routing.
    `Semantic/` and `Procedural/` take the full checklist; `Episodic/` and
    `Research/` take only the §6 schema invariants.

    Args:
        type: Target memory type. The `type` argument of `memory_write` (§7.2).
        meta: The frontmatter the note would carry (§6), plus the gate-only keys
            `evidence` (§8.2 item 6) and `merge` (§8.2 item 2).
        body: The note body, without frontmatter.
        manifest: Parsed deployment config -- the source of every threshold,
            denylist and registry consulted here (P6).
        search_fn: Retrieval for the dedup check. Required for
            :data:`FULL_GATE_TYPES`; unused otherwise.

    Returns:
        A :class:`GateResult`. Rejections and warnings are independent: an
        approved write commonly carries warnings.

    Raises:
        ValueError: `type` is not a memory type, or `search_fn` is missing for a
            gated type. Both are wiring bugs in the caller, not author errors --
            the gate refuses to approve a write whose dedup check never ran
            (R5.1: no silent skip).
        Exception: whatever `search_fn` raises propagates. A broken index must
            not read as "no duplicates found".
    """
    memory_type = MemoryType.parse(str(type))
    rejections: list[GateIssue] = []
    warnings: list[GateIssue] = []
    near_duplicates: tuple[NearDuplicate, ...] = ()

    _check_enabled(memory_type, manifest, rejections)
    scope_ok = _check_scope(meta, manifest, rejections)
    if _check_agent(meta, manifest, rejections):
        _check_profile(meta, manifest, rejections)
    _check_denylists(meta, manifest, rejections)

    if memory_type in FULL_GATE_TYPES:
        _check_source(meta, rejections)
        if memory_type is MemoryType.PROCEDURAL:
            _check_evidence(meta, rejections)
        if scope_ok:
            near_duplicates = _check_duplicates(
                memory_type, meta, body, manifest, search_fn, rejections
            )
        warnings.extend(_instructed_warnings(meta, body))

    return GateResult(
        approved=not rejections,
        rejections=tuple(rejections),
        warnings=tuple(warnings),
        near_duplicates=near_duplicates,
    )


# ══════════════════════════════════════════════════════════════════════
# Enforced checks
# ══════════════════════════════════════════════════════════════════════
def _check_enabled(
    memory_type: MemoryType, manifest: Manifest, rejections: list[GateIssue]
) -> None:
    """The deployment may switch a memory type off (§3, `zones.research_enabled`)."""
    if manifest.is_enabled(memory_type):
        return
    rejections.append(
        GateIssue(
            check="type_disabled",
            rule="§3",
            message=f"memory type {memory_type.value!r} is disabled in this deployment",
            hint=f"set zones.research_enabled: true in {_config_name(manifest)} to enable it",
        )
    )


def _check_scope(
    meta: Mapping[str, Any], manifest: Manifest, rejections: list[GateIssue]
) -> bool:
    """§8.2 item 5 / R6.5 -- `scope:` present and known.

    Returns whether the scope is usable, since a dedup search filtered on an
    invalid scope would either fail or, worse, silently search every scope.
    """
    scope = _text(meta.get("scope"))
    if not scope:
        rejections.append(
            GateIssue(
                check="scope_missing",
                rule="R6.5",
                message="scope: is required on every memory",
                hint=f"declare the project slug, or {manifest.scopes.default!r} "
                f"when the memory belongs to no project",
            )
        )
        return False
    if not manifest.is_known_scope(scope):
        known = ", ".join(manifest.scopes.known)
        rejections.append(
            GateIssue(
                check="scope_unknown",
                rule="R6.5",
                message=f"scope {scope!r} is not in scopes.known",
                hint=f"use one of: {known}"
                if known
                else "scope must be a lowercase slug (a-z, 0-9, '-', '_')",
            )
        )
        return False
    return True


def _check_agent(
    meta: Mapping[str, Any], manifest: Manifest, rejections: list[GateIssue]
) -> bool:
    """R6.2 -- the author must be in the registry. Returns whether it is."""
    agent = _text(meta.get("agent"))
    if not agent:
        rejections.append(
            GateIssue(
                check="agent_missing",
                rule="R6.2",
                message="agent: is required -- every memory names its author (P7)",
                hint=f"registered agents: {_registry(manifest)}",
            )
        )
        return False
    if not manifest.is_known_agent(agent):
        rejections.append(
            GateIssue(
                check="agent_unknown",
                rule="R6.2",
                message=f"unknown agent {agent!r}",
                hint=f"registered agents: {_registry(manifest)}",
            )
        )
        return False
    return True


def _check_profile(
    meta: Mapping[str, Any], manifest: Manifest, rejections: list[GateIssue]
) -> None:
    """R6.6 -- `profile:`, when present, belongs to the author's registry entry."""
    profile = _text(meta.get("profile"))
    if not profile:
        return
    agent = _text(meta.get("agent"))
    declared = manifest.profiles_of(agent)
    if profile in declared:
        return
    rejections.append(
        GateIssue(
            check="profile_unknown",
            rule="R6.6",
            message=f"agent {agent!r} declares no profile {profile!r}",
            hint=f"profiles of {agent!r}: {', '.join(declared) or '(none)'}",
        )
    )


def _check_denylists(
    meta: Mapping[str, Any], manifest: Manifest, rejections: list[GateIssue]
) -> None:
    """§8.2 item 7 / R6.1 -- no denied key, no denied tag. The error cites the rule."""
    spec = manifest.frontmatter
    config = _config_name(manifest)

    denied_keys = sorted(set(meta) & set(spec.denylist_keys))
    if denied_keys:
        rejections.append(
            GateIssue(
                check="denylist_keys",
                rule="R6.1",
                message=f"frontmatter key(s) {', '.join(repr(k) for k in denied_keys)} "
                f"are forbidden by frontmatter.denylist_keys",
                hint=f"drop the key(s); {config} reserves them for the host vault",
            )
        )

    denied_tags = sorted(set(normalize_tags(meta.get("tags"))) & set(spec.denylist_tags))
    if denied_tags:
        rejections.append(
            GateIssue(
                check="denylist_tags",
                rule="R6.1",
                message=f"tag(s) {', '.join(repr(t) for t in denied_tags)} "
                f"are forbidden by frontmatter.denylist_tags",
                hint=f"drop the tag(s); {config} reserves them for the host vault",
            )
        )


def _check_source(meta: Mapping[str, Any], rejections: list[GateIssue]) -> None:
    """§8.2 item 4 -- `source:` non-empty. Provenance is what makes memory citable (P7)."""
    if _text(meta.get("source")):
        return
    rejections.append(
        GateIssue(
            check="source_missing",
            rule="§8.2 item 4",
            message="source: is empty -- a memory with no origin cannot be audited",
            hint="name the session, run or bridge this came from",
        )
    )


def _check_evidence(meta: Mapping[str, Any], rejections: list[GateIssue]) -> None:
    """§8.2 item 6 -- a procedural write must carry evidence it ran.

    The spec asks that the procedure "have been executed successfully at least
    once, with the evidence cited in the body". Whether a run succeeded is not
    observable from here; requiring `meta.evidence` and rendering it into the
    body is the mechanical half of that -- it makes the citation checkable, and
    forces the author to produce one rather than assert a bare `true`. A
    playbook is the riskiest thing to write: it propagates error to every agent.
    """
    if _non_empty(meta.get("evidence")):
        return
    rejections.append(
        GateIssue(
            check="evidence_missing",
            rule="§8.2 item 6",
            message="procedural memory requires meta.evidence: the procedure must "
            "have run successfully at least once",
            hint="cite the run -- command, date, outcome. It is rendered into the "
            "note body as the evidence section",
        )
    )


def _check_duplicates(
    memory_type: MemoryType,
    meta: Mapping[str, Any],
    body: str,
    manifest: Manifest,
    search_fn: SearchFn | None,
    rejections: list[GateIssue],
) -> tuple[NearDuplicate, ...]:
    """§8.2 item 2 -- refuse a blind near-duplicate write.

    The search is filtered to the candidate's own scope *and* type: R6.5 makes
    cross-scope similarity meaningful rather than redundant (the same sentence
    about two projects is two facts), and the remedy §8.2 offers -- `supersedes`
    -- only links notes of one kind. A `PROC` resembling an `INS` is a related
    note, not a duplicate of it.

    An author who declared `supersedes` or `merge` has made the explicit decision
    the spec demands, so the neighbours are returned but do not reject.
    """
    if search_fn is None:
        raise ValueError(
            f"the {memory_type.value} write gate requires search_fn: §8.2 item 2 makes "
            f"the internal dedup search mandatory, and the gate must not approve a "
            f"write whose duplicate check never ran"
        )

    threshold = manifest.maintenance.dedup_similarity
    scope = _text(meta.get("scope"))
    hits = search_fn(
        _dedup_query(meta, body),
        k=DEDUP_CANDIDATES,
        filters={"scope": scope, "type": memory_type.value},
    )
    near = tuple(
        candidate
        for candidate in (NearDuplicate.from_hit(hit) for hit in hits)
        if candidate.similarity > threshold
    )
    if not near or _has_decision(meta):
        return near

    listed = ", ".join(f"{d.wikilink} ({d.similarity:.2f})" for d in near)
    rejections.append(
        GateIssue(
            check="duplicate",
            rule="§8.2 item 2",
            message=f"{len(near)} note(s) in scope {scope!r} are above "
            f"maintenance.dedup_similarity ({threshold:g}): {listed}",
            hint="decide explicitly: set supersedes: to replace one, set merge: true "
            "after folding the detail in, or drop this write",
        )
    )
    return near


# ══════════════════════════════════════════════════════════════════════
# Instructed items -- warnings only
# ══════════════════════════════════════════════════════════════════════
def _instructed_warnings(meta: Mapping[str, Any], body: str) -> list[GateIssue]:
    """The §8.2 items the kernel reports instead of enforcing.

    Item 1 is unconditional: there is no textual signal for "reusable beyond this
    session", so the only faithful move is to hand the obligation back with the
    write. Items 3 and 4b fire on a heuristic and stay quiet otherwise.
    """
    warnings = [
        GateIssue(
            check="reusable",
            rule="§8.2 item 1",
            message="reusability is not machine-checkable: confirm this memory is "
            "worth more than the session that produced it",
            hint="if it only describes what happened, it is episodic (§8.1 item 3)",
        )
    ]
    atomicity = _atomicity_warning(body)
    if atomicity is not None:
        warnings.append(atomicity)
    confidence = _confidence_warning(meta)
    if confidence is not None:
        warnings.append(confidence)
    return warnings


def _atomicity_warning(body: str) -> GateIssue | None:
    """§8.2 item 3 -- one insight per note (Zettelkasten).

    Two honest, weak signals: length, and sibling top-level sections. Both are
    suggestive only -- a long note may hold one idea and a short one may hold
    three -- which is exactly why this warns instead of rejecting.
    """
    length = len(body)
    sections = _top_level_sections(body)
    reasons: list[str] = []
    if length > ATOMIC_BODY_CHARS:
        reasons.append(f"body is {length} chars (over {ATOMIC_BODY_CHARS})")
    if sections > ATOMIC_MAX_SECTIONS:
        reasons.append(f"{sections} sibling top-level sections")
    if not reasons:
        return None
    return GateIssue(
        check="atomic",
        rule="§8.2 item 3",
        message=f"this may hold more than one insight ({'; '.join(reasons)}) -- "
        f"the kernel cannot tell, so it is your call",
        hint="if it is several insights, split it into one note each and link them",
    )


def _confidence_warning(meta: Mapping[str, Any]) -> GateIssue | None:
    """§8.2 item 4 -- `confidence: high` only with verification or a primary source.

    Whether `source:` is *primary* is a judgement. What the gate can see is a
    high-confidence claim with no evidence declared next to it.
    """
    if _text(meta.get("confidence")).lower() != "high":
        return None
    if _non_empty(meta.get("evidence")):
        return None
    return GateIssue(
        check="confidence_unverified",
        rule="§8.2 item 4",
        message="confidence: high declares no verification -- it is warranted only "
        "by a check you ran or a primary source",
        hint="add meta.evidence, or lower confidence to medium",
    )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _text(value: Any) -> str:
    """A frontmatter value as trimmed text; ``""`` for None/blank."""
    return "" if value is None else str(value).strip()


def _non_empty(value: Any) -> bool:
    """Whether a meta value carries content, for scalars and lists alike."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set)):
        return any(_non_empty(item) for item in value)
    return True


def _has_decision(meta: Mapping[str, Any]) -> bool:
    """Whether the author made the §8.2 item 2 call: `supersedes` or `merge`."""
    return _non_empty(meta.get("supersedes")) or bool(meta.get("merge"))


def _dedup_query(meta: Mapping[str, Any], body: str) -> str:
    """The text the dedup search runs on: the claim, title first."""
    title = _text(meta.get("title"))
    return f"{title}\n\n{body.strip()}".strip()[:DEDUP_QUERY_CHARS]


def _top_level_sections(body: str) -> int:
    """Count sibling headings at the body's shallowest heading level.

    Subsections do not count -- a playbook nesting steps under one heading is one
    procedure. Fenced code is stripped first, so a shell comment is not a
    heading.
    """
    prose = _CODE_FENCE_RE.sub("", body)
    levels = [len(match.group(1)) for match in _HEADING_RE.finditer(prose)]
    if not levels:
        return 0
    shallowest = min(levels)
    return sum(1 for level in levels if level == shallowest)


def _hit_field(hit: Any, name: str) -> Any:
    """Read `name` off a hit, which may be a mapping or an object."""
    if isinstance(hit, Mapping):
        return hit.get(name)
    return getattr(hit, name, None)


def _similarity_of(hit: Any) -> float:
    """The hit's similarity. See :class:`SearchFn` on why `similarity` wins."""
    for name in SIMILARITY_FIELDS:
        value = _hit_field(hit, name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"search hit field {name!r} is not a number: {value!r}"
            ) from None
    raise ValueError(
        f"search hit carries no similarity; expected one of: "
        f"{', '.join(SIMILARITY_FIELDS)}"
    )


def _registry(manifest: Manifest) -> str:
    ids = manifest.agent_ids()
    return ", ".join(ids) if ids else "(none -- add one to agents: in config.yaml)"


def _config_name(manifest: Manifest) -> str:
    return manifest.source_path.name if manifest.source_path is not None else "config.yaml"
