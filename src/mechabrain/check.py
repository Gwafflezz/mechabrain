"""Deployment lint: ``mechabrain check`` (spec §10).

`check` is the CI-facing conscience of a deployment. It answers one question --
*is this ``mecha-brain/`` folder in a state the kernel can serve without
surprises?* -- and answers it **without an LLM and without writing anything**.
Every finding cites the spec rule it enforces and says how to fix it, so the
output is a to-do list, not a verdict.

What it verifies
================

===================================  ==========================================
Check                                Rule
===================================  ==========================================
manifest parses and this kernel is    R5.1 / R4.5 (and R4.2: the manifest
new enough to serve it                loader rejects absolute paths, so a
                                      manifest that parses holds none)
every note under ``mecha-brain/``      R6.1
carries no denied key or tag
every note's ``agent:`` is registered  R6.2
and its ``profile:`` is declared       R6.6
every note has a present, valid        R6.5
``scope:``
an episodic note sits under the        R6.3 / §3
``Episodic/<agent>/`` of its author
the §3 skeleton is intact              §3
``Episodic/<agent>/`` exists for        R6.2 / §3
every registered agent
``_meta/index/`` is gitignored and      §10
``_meta/links.jsonl`` is **not**
notes stranded outside the contract    §3
tree
===================================  ==========================================

Errors versus warnings
======================

The report keeps the same honest split :mod:`mechabrain.gate` does. An **error**
is a mechanically-certain violation that would make the kernel misbehave -- a
denied key, an unknown author, a committed index. Errors set
:attr:`CheckReport.ok` to ``False`` and drive a non-zero
:attr:`CheckReport.exit_code`, which is what a CI job trips on.

A **warning** is an advisory the kernel cannot turn into a hard failure without
overreaching: a stray note that a human may have parked on purpose, or a
gitignore rule the kernel could not verify because the vault is not a git
repository. Warnings are reported but never fail the build -- pretending
otherwise would make ``check`` unusable on the vaults where those states are
legitimate.

Reuse, not reimplementation
===========================

The decision logic lives where it already lived: :meth:`Manifest.is_known_agent`,
:meth:`Manifest.is_known_scope`, :meth:`Manifest.profiles_of` and the
``frontmatter.denylist_*`` tuples are the authorities on §6 conformance, exactly
as the write gate uses them. This module maps their answers onto
:class:`Problem` records; it does not re-derive them. The one thing it adds is
the git-ignore probe, which is a lint concern the gate never has.

Typical use::

    from mechabrain.discovery import discover_vault
    from mechabrain.check import check

    report = check(discover_vault())
    print(report.render())
    raise SystemExit(report.exit_code)
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from .contract import MARKDOWN_SUFFIX, MemoryType
from .discovery import VaultPaths
from .errors import MechabrainError, NoteNotFound, SchemaViolation
from .manifest import Manifest, load_manifest
from .note import Note

__all__ = [
    "Severity",
    "Problem",
    "CheckReport",
    "GitIgnoreProbe",
    "check",
]

#: Filename the git-ignore probe tests inside ``_meta/index/``. It need not
#: exist: ``git check-ignore`` matches a pathname against the ignore rules
#: without touching the filesystem, so a representative path under the directory
#: proves the whole directory is covered whatever spelling the rule uses.
_INDEX_PROBE_NAME: Final[str] = "vectors.probe"


class Severity(str, Enum):
    """How a finding weighs on the build.

    ``ERROR`` fails the check (``ok`` False, non-zero exit); ``WARNING`` is
    advisory and never does. The split mirrors the gate's rejections/warnings:
    the kernel enforces only what it can prove and reports the rest.
    """

    ERROR = "error"
    WARNING = "warning"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Problem:
    """One finding of :func:`check`.

    Attributes:
        check: Stable machine id, e.g. ``"denylist_keys"``. Tests and tooling
            match on this; ``message`` is prose and may be reworded.
        rule: Spec citation, e.g. ``"R6.1"`` or ``"§10"`` -- the clause to grep.
        message: What is wrong, in English (kernel messages are English).
        severity: Whether this fails the build. See :class:`Severity`.
        hint: The next action to take. Never a restatement of ``message``.
        path: Vault-root-relative path of the offending file, when the finding
            is about one. ``None`` for deployment-wide findings.
    """

    check: str
    rule: str
    message: str
    severity: Severity = Severity.ERROR
    hint: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        head = f"{self.severity.value.upper()} [{self.rule}]"
        where = f" {self.path}:" if self.path else ""
        text = f"{head}{where} {self.message}"
        return f"{text}\n  hint: {self.hint}" if self.hint else text


@dataclass(frozen=True, slots=True)
class CheckReport:
    """The verdict of :func:`check`.

    Attributes:
        problems: Every finding, errors and warnings alike, in discovery order.

    ``ok`` is ``True`` iff no finding is an error: warnings alone do not fail a
    deployment. ``exit_code`` turns that into the 0/1 a CI job reads.
    """

    problems: tuple[Problem, ...] = ()

    @property
    def errors(self) -> tuple[Problem, ...]:
        """Findings that fail the build."""
        return tuple(p for p in self.problems if p.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Problem, ...]:
        """Advisory findings that do not fail the build."""
        return tuple(p for p in self.problems if p.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """Whether the deployment passes: no error-severity finding."""
        return not self.errors

    @property
    def exit_code(self) -> int:
        """``0`` when ``ok``, ``1`` otherwise -- the code ``mechabrain check`` returns."""
        return 0 if self.ok else 1

    def render(self) -> str:
        """A human-readable report: one block per finding, errors first.

        Deterministic and side-effect-free -- the CLI prints it; nothing here
        writes to the vault (§10: ``check`` only reads).
        """
        if not self.problems:
            return "check: OK — no problems found."
        lines = [str(problem) for problem in (*self.errors, *self.warnings)]
        summary = f"check: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([*lines, "", summary])


# ══════════════════════════════════════════════════════════════════════
# Git-ignore probe
# ══════════════════════════════════════════════════════════════════════
#: Answers "does the vault's git ignore this path?" -- ``True`` ignored,
#: ``False`` tracked/untracked-but-not-ignored, ``None`` when it cannot be told
#: (no git, or the vault is not a work tree). Injected so tests need no real repo.
GitIgnoreProbe = Callable[[Path], "bool | None"]


def _git_ignore_probe(root: Path) -> GitIgnoreProbe:
    """A :data:`GitIgnoreProbe` backed by ``git check-ignore`` run in ``root``.

    Shelling out is deliberate: ``check-ignore`` honours the full rule stack a
    hand-rolled parser would miss -- nested ``.gitignore`` files, negations,
    ``.git/info/exclude`` and the user's global excludes -- and matches a
    pathname whether or not it exists on disk. It never mutates the repository.
    """

    def probe(path: Path) -> bool | None:
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", "--", str(path)],
                cwd=root,
                capture_output=True,
            )
        except (FileNotFoundError, OSError):
            return None  # git is not installed on this machine.
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        return None  # 128: not a git repository, or another git-side error.

    return probe


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
def check(
    vault: VaultPaths | Path | str,
    *,
    git_ignored: GitIgnoreProbe | None = None,
) -> CheckReport:
    """Lint the deployment at ``vault`` and return its problems (§10).

    Reads only: no note is written, no index is touched, no LLM is called. Given
    the same vault it returns the same report.

    Args:
        vault: An initialized vault. A :class:`VaultPaths` is used as-is; a path
            or string is taken as the vault root (the parent of
            ``mecha-brain/``). Discovery (R4.3) is the caller's job -- pass the
            result of :func:`mechabrain.discovery.discover_vault`.
        git_ignored: Override for the git-ignore probe, for tests that must not
            depend on a real repository. Defaults to :func:`_git_ignore_probe`
            bound to the vault root.

    Returns:
        A :class:`CheckReport`. If the manifest itself fails to load, the report
        carries that single error and nothing else -- every other check needs a
        valid manifest, so continuing would only produce noise.
    """
    paths = vault if isinstance(vault, VaultPaths) else VaultPaths.for_root(vault)

    try:
        manifest = load_manifest(paths.config_file)
    except MechabrainError as exc:
        return CheckReport((_manifest_problem(exc),))

    problems: list[Problem] = []
    _check_contract_tree(paths, manifest, problems)
    _check_episodic_subfolders(paths, manifest, problems)
    _check_notes(paths, manifest, problems)
    _check_orphans(paths, problems)
    _check_gitignore(paths, git_ignored or _git_ignore_probe(paths.root), problems)
    return CheckReport(tuple(problems))


# ══════════════════════════════════════════════════════════════════════
# Manifest (R5.1 / R4.5 / R4.2)
# ══════════════════════════════════════════════════════════════════════
def _manifest_problem(exc: MechabrainError) -> Problem:
    """Turn a manifest load failure into the report's single stopping error.

    The exception already names its rule -- R5.1 for a bad key, R4.5 for a
    kernel too old, R4.3 for an absent manifest, and (via the loader's
    absolute-path guard) R4.2 -- so this preserves it rather than inventing one.
    """
    return Problem(
        check="manifest_invalid",
        rule=exc.rule or "R5.1",
        message=exc.message,
        severity=Severity.ERROR,
        hint=exc.hint or "run `mechabrain init <vault>` to regenerate a valid config.yaml",
    )


# ══════════════════════════════════════════════════════════════════════
# Contract tree (§3)
# ══════════════════════════════════════════════════════════════════════
def _check_contract_tree(paths: VaultPaths, manifest: Manifest, out: list[Problem]) -> None:
    """The §3 memory folders must exist.

    ``indices/``, ``_inbox/`` and ``_meta/index/`` are deliberately not required:
    the first two are generated on demand (§3) and the third is derived, per
    machine and gitignored (§4) -- a fresh clone legitimately has none of them.
    ``Research/`` is required only where ``zones.research_enabled`` keeps it.
    """
    required: list[tuple[str, Path]] = [
        (paths.relative(paths.semantic_dir), paths.semantic_dir),
        (paths.relative(paths.episodic_dir), paths.episodic_dir),
        (paths.relative(paths.procedural_dir), paths.procedural_dir),
    ]
    if manifest.zones.research_enabled:
        required.append((paths.relative(paths.research_dir), paths.research_dir))

    for name, directory in required:
        if not directory.is_dir():
            out.append(
                Problem(
                    check="contract_dir_missing",
                    rule="§3",
                    message=f"contract folder {name}/ is missing",
                    hint="run `mechabrain init <vault>` (idempotent) to restore the skeleton",
                    path=name,
                )
            )


def _check_episodic_subfolders(
    paths: VaultPaths, manifest: Manifest, out: list[Problem]
) -> None:
    """Every registered agent needs its ``Episodic/<agent>/`` (R6.2, §3).

    Episodic is append-only and partitioned by runtime (R6.3/R6.6); a write for
    an agent whose folder is absent would fail, so `mechabrain sync` creates one
    per new agent (§10). A missing folder means the registry and the tree have
    drifted apart.
    """
    for agent_id in manifest.agent_ids():
        folder = paths.episodic_for(agent_id)
        if not folder.is_dir():
            out.append(
                Problem(
                    check="episodic_subfolder_missing",
                    rule="R6.2",
                    message=f"registered agent {agent_id!r} has no {paths.relative(folder)}/ folder",
                    hint="run `mechabrain sync` to create Episodic/ subfolders for new agents",
                    path=paths.relative(folder),
                )
            )


# ══════════════════════════════════════════════════════════════════════
# Notes (R6.1 / R6.2 / R6.5 / R6.6 / R6.3)
# ══════════════════════════════════════════════════════════════════════
def _check_notes(paths: VaultPaths, manifest: Manifest, out: list[Problem]) -> None:
    """Validate every memory note against the §6 schema invariants.

    Scans the four memory folders only: ``read_only_index`` is human territory
    the kernel does not author, and ``_inbox/`` holds proposals, not memories,
    so neither faces the frontmatter contract. A note with malformed frontmatter
    is reported, not raised over -- one broken note must not blind the lint to
    the rest.
    """
    for memory_type in MemoryType:
        folder = paths.folder_for(memory_type)
        for note_path in _iter_note_paths(folder):
            try:
                note = Note.load(note_path)
            except SchemaViolation as exc:
                out.append(
                    Problem(
                        check="frontmatter_malformed",
                        rule="§6",
                        message=f"frontmatter cannot be parsed: {exc.message}",
                        hint=exc.hint,
                        path=paths.relative(note_path),
                    )
                )
                continue
            except NoteNotFound:
                continue  # Vanished between glob and load; nothing to check.
            _check_note_schema(note, note_path, memory_type, paths, manifest, out)


def _check_note_schema(
    note: Note,
    note_path: Path,
    memory_type: MemoryType,
    paths: VaultPaths,
    manifest: Manifest,
    out: list[Problem],
) -> None:
    """Apply the §6 checks to one loaded note. See the module docstring on reuse."""
    rel = paths.relative(note_path)
    _check_denylists(note, manifest, rel, out)
    agent = _check_agent(note, manifest, rel, out)
    _check_profile(note, agent, manifest, rel, out)
    _check_scope(note, manifest, rel, out)
    if memory_type is MemoryType.EPISODIC:
        _check_episodic_placement(note, note_path, agent, paths, rel, out)


def _check_denylists(
    note: Note, manifest: Manifest, rel: str, out: list[Problem]
) -> None:
    """R6.1 -- no key from ``denylist_keys``, no tag from ``denylist_tags``."""
    spec = manifest.frontmatter
    denied_keys = sorted(set(note.frontmatter) & set(spec.denylist_keys))
    if denied_keys:
        out.append(
            Problem(
                check="denylist_keys",
                rule="R6.1",
                message=f"frontmatter key(s) {_quote(denied_keys)} are forbidden by "
                f"frontmatter.denylist_keys",
                hint="drop the key(s); the manifest reserves them for the host vault",
                path=rel,
            )
        )
    denied_tags = sorted(set(note.tags) & set(spec.denylist_tags))
    if denied_tags:
        out.append(
            Problem(
                check="denylist_tags",
                rule="R6.1",
                message=f"tag(s) {_quote(denied_tags)} are forbidden by frontmatter.denylist_tags",
                hint="drop the tag(s); the manifest reserves them for the host vault",
                path=rel,
            )
        )


def _check_agent(
    note: Note, manifest: Manifest, rel: str, out: list[Problem]
) -> str:
    """R6.2 -- the author must be registered. Returns the agent as written (may be blank)."""
    agent = _text(note.get("agent"))
    if not agent:
        out.append(
            Problem(
                check="agent_missing",
                rule="R6.2",
                message="agent: is missing — every memory names its author (P7)",
                hint=f"set agent: to one of: {_registry(manifest)}",
                path=rel,
            )
        )
    elif not manifest.is_known_agent(agent):
        out.append(
            Problem(
                check="agent_unknown",
                rule="R6.2",
                message=f"agent {agent!r} is not in the registry",
                hint=f"registered agents: {_registry(manifest)}; add it to agents: and "
                f"run `mechabrain sync`, or fix the note",
                path=rel,
            )
        )
    return agent


def _check_profile(
    note: Note, agent: str, manifest: Manifest, rel: str, out: list[Problem]
) -> None:
    """R6.6 -- ``profile:``, when present, must be declared by a *known* author.

    Skipped when the agent is unknown: :meth:`Manifest.profiles_of` would raise
    on it, and the unknown-agent error already covers the note.
    """
    profile = _text(note.get("profile"))
    if not profile or not agent or not manifest.is_known_agent(agent):
        return
    declared = manifest.profiles_of(agent)
    if profile not in declared:
        out.append(
            Problem(
                check="profile_unknown",
                rule="R6.6",
                message=f"agent {agent!r} declares no profile {profile!r}",
                hint=f"profiles of {agent!r}: {', '.join(declared) or '(none)'}",
                path=rel,
            )
        )


def _check_scope(
    note: Note, manifest: Manifest, rel: str, out: list[Problem]
) -> None:
    """R6.5 -- ``scope:`` present and, when ``scopes.known`` is set, one of it."""
    scope = _text(note.get("scope"))
    if not scope:
        out.append(
            Problem(
                check="scope_missing",
                rule="R6.5",
                message="scope: is missing — a memory with no scope can contaminate "
                "another project",
                hint=f"declare the project slug, or {manifest.scopes.default!r} when it "
                f"belongs to no project",
                path=rel,
            )
        )
    elif not manifest.is_known_scope(scope):
        known = ", ".join(manifest.scopes.known)
        out.append(
            Problem(
                check="scope_unknown",
                rule="R6.5",
                message=f"scope {scope!r} is not in scopes.known",
                hint=f"use one of: {known}"
                if known
                else "scope must be a lowercase slug (a-z, 0-9, '-', '_')",
                path=rel,
            )
        )


def _check_episodic_placement(
    note: Note,
    note_path: Path,
    agent: str,
    paths: VaultPaths,
    rel: str,
    out: list[Problem],
) -> None:
    """R6.3/§3 -- an episodic note lives under the ``Episodic/<agent>/`` of its author.

    Episodic is partitioned by runtime; a note filed under the wrong agent's
    journal misattributes who did what (P7). The subfolder is the first path
    component under ``Episodic/``.
    """
    try:
        parts = note_path.relative_to(paths.episodic_dir).parts
    except ValueError:
        return
    if len(parts) < 2:
        # A note directly in Episodic/, not in any agent subfolder.
        out.append(
            Problem(
                check="episodic_unpartitioned",
                rule="R6.3",
                message="episodic note is not under an Episodic/<agent>/ subfolder",
                hint="move it into Episodic/<agent>/ — episodic memory is per-agent (R6.6)",
                path=rel,
            )
        )
        return
    folder_agent = parts[0]
    if agent and folder_agent != agent:
        out.append(
            Problem(
                check="episodic_agent_mismatch",
                rule="R6.3",
                message=f"episodic note under Episodic/{folder_agent}/ declares "
                f"agent: {agent!r}",
                hint=f"episodic memory is per-agent — move the note to "
                f"Episodic/{agent}/, or fix its agent:",
                path=rel,
            )
        )


# ══════════════════════════════════════════════════════════════════════
# Orphans (§3)
# ══════════════════════════════════════════════════════════════════════
def _check_orphans(paths: VaultPaths, out: list[Problem]) -> None:
    """Notes stranded in ``mecha-brain/`` outside any contract location (§3).

    A ``.md`` file the kernel neither generates nor indexes is invisible to
    search and to the maps: a memory parked there is lost. Reported as a warning,
    not an error -- a human may have deliberately left a scratch note under
    ``mecha-brain/``, and failing CI over it would overreach.
    """
    allowed_trees = (
        paths.semantic_dir,
        paths.episodic_dir,
        paths.procedural_dir,
        paths.research_dir,
        paths.indices_dir,
        paths.inbox_dir,
        paths.meta_dir,
    )
    generated_files = {paths.agents_file, paths.hot_file, paths.index_file}
    for note_path in _iter_note_paths(paths.mecha_brain):
        if note_path in generated_files:
            continue
        if any(_is_within(note_path, tree) for tree in allowed_trees):
            continue
        out.append(
            Problem(
                check="orphan_note",
                rule="§3",
                message="note sits outside every contract folder — the kernel will "
                "not index or find it",
                severity=Severity.WARNING,
                hint="move it into Semantic/, Episodic/<agent>/, Procedural/ or "
                "Research/, or out of mecha-brain/ entirely",
                path=paths.relative(note_path),
            )
        )


# ══════════════════════════════════════════════════════════════════════
# Git-ignore (§10)
# ══════════════════════════════════════════════════════════════════════
def _check_gitignore(
    paths: VaultPaths, git_ignored: GitIgnoreProbe, out: list[Problem]
) -> None:
    """``_meta/index/`` must be ignored and ``_meta/links.jsonl`` must not (§10).

    Two opposite invariants. The derived index is per-machine runtime state:
    committing it bloats the vault's history and lets two machines fight over one
    LanceDB/SQLite file across a sync (R7.4/R7.6). ``links.jsonl`` is the
    opposite -- authored edges, a source of truth (§7.2) -- so a rule that
    swallowed it would silently drop what agents recorded.

    When the probe cannot tell (no git, or the vault is not a work tree) a single
    warning is emitted rather than two misleading errors: the kernel does not
    require the vault to be a git repository, it only checks the rules if one is.
    """
    index_probe = paths.index_dir / _INDEX_PROBE_NAME
    index_ignored = git_ignored(index_probe)
    links_ignored = git_ignored(paths.links_file)

    if index_ignored is None or links_ignored is None:
        out.append(
            Problem(
                check="git_unavailable",
                rule="§10",
                message="cannot verify gitignore rules — the vault is not a git "
                "repository, or git is unavailable",
                severity=Severity.WARNING,
                hint="init the vault as a git repo so the derived index stays out of "
                "history and authored links stay in",
            )
        )
        return

    if not index_ignored:
        out.append(
            Problem(
                check="index_not_ignored",
                rule="§10",
                message=f"{paths.relative(paths.index_dir)}/ is not gitignored — the "
                f"derived, per-machine index would be committed",
                hint=f"add `{paths.relative(paths.index_dir)}/` to the vault's .gitignore "
                f"(this is what `mechabrain init` does)",
                path=paths.relative(paths.index_dir),
            )
        )
    if links_ignored:
        out.append(
            Problem(
                check="links_ignored",
                rule="§10",
                message=f"{paths.relative(paths.links_file)} is gitignored — authored "
                f"links are a source of truth and must be committed (§7.2)",
                hint="remove the rule that ignores links.jsonl, or add a negation "
                "(e.g. `!mecha-brain/_meta/links.jsonl`)",
                path=paths.relative(paths.links_file),
            )
        )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _iter_note_paths(folder: Path) -> Iterator[Path]:
    """Yield every ``.md`` file under ``folder``, sorted, skipping dotfiles.

    Mirrors :func:`mechabrain.note.iter_notes` on which files count -- dotfiles
    and the temporary files of atomic writes are skipped -- but yields *paths*,
    so the caller can catch a malformed note per file instead of the whole walk
    raising.
    """
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("**/*")):
        if not path.is_file() or path.suffix != MARKDOWN_SUFFIX:
            continue
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        yield path


def _is_within(path: Path, folder: Path) -> bool:
    """Whether ``path`` is inside ``folder`` (or is it)."""
    return path == folder or folder in path.parents


def _text(value: object) -> str:
    """A frontmatter value as trimmed text; ``""`` for None/blank."""
    return "" if value is None else str(value).strip()


def _quote(values: list[str]) -> str:
    return ", ".join(repr(value) for value in values)


def _registry(manifest: Manifest) -> str:
    ids = manifest.agent_ids()
    return ", ".join(ids) if ids else "(none — add one to agents: in config.yaml)"
