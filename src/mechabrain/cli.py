"""The command-line interface: ``mechabrain`` (spec §10).

Built on :mod:`argparse` from the stdlib -- no third-party CLI framework, which
would be a dependency the core does not otherwise need. Every subcommand of the
§10 table is here and does exactly what that table says:

===============================  ==========================================
``mechabrain init <vault>``       Create the §3 skeleton, the default manifest,
                                  the ``.gitignore`` entry, ``AGENTS.md`` and
                                  ``schema.md``; print the integration snippet.
                                  **Idempotent** -- a re-run destroys nothing.
``mechabrain sync``               Regenerate the manifest-derived artifacts
                                  (``AGENTS.md`` block, ``schema.md``, new
                                  ``Episodic/<agent>/`` folders) after a config
                                  edit.
``mechabrain serve``              Bring up the MCP memory daemon (R7.4).
``mechabrain reindex [--full]``   Rebuild the derived index.
``mechabrain consolidate``        Run the §9 maintenance pipeline.
``mechabrain check``              Lint the deployment; exit non-zero on an error.
===============================  ==========================================

Vault discovery, everywhere
===========================

Every subcommand accepts ``--vault`` and honours the R4.3 order -- explicit
argument, then ``$MECHABRAIN_VAULT``, then an upward walk from the CWD -- through
:func:`mechabrain.discovery.discover_vault`. ``init`` is the one exception that
may target a vault that does not exist yet, so it resolves a root directly (its
positional argument, then ``--vault``, then the env var, then the CWD) instead of
discovering an already-initialized one.

Two languages, one boundary
===========================

The CLI's own operational output is English, like every kernel message (R5.1).
The one block it prints in pt-BR is the integration snippet meant to be pasted
into the host vault's ``CLAUDE.md`` -- that is deployment content, read by the
humans and agents of the vault, exactly like ``AGENTS.md`` and ``schema.md``.

The daemon ships separately
===========================

``serve`` needs the MCP server, which is a distinct kernel component. This module
resolves the vault, validates the manifest and the runtime port, then delegates
to :func:`_run_daemon` -- a seam a test overrides and the daemon component fills
via ``mechabrain.server.serve``. Runtime is never deployment: the port comes from
``--port`` or ``$MECHABRAIN_PORT``, never from the manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

from . import __version__
from .check import check as run_check
from .consolidate import consolidate as run_consolidate
from .contract import GITIGNORE_INDEX_ENTRY
from .discovery import VAULT_ENV_VAR, VaultPaths, discover_vault
from .errors import MechabrainError
from .generate import (
    render_initial_hot,
    render_initial_index,
    write_agents_md,
    write_default_config,
    write_mecha_scribe_skill,
    write_schema,
)
from .index.indexer import Indexer, IndexReport
from .manifest import Manifest, load_manifest
from .note import write_atomic

__all__ = ["main", "build_parser"]

#: Loopback by default: a memory daemon is a local single-writer (R7.4), not a
#: network service. A deployment that wants otherwise passes ``--host``.
DEFAULT_HOST: Final[str] = "127.0.0.1"

#: Fallback port when neither ``--port`` nor ``$MECHABRAIN_PORT`` is set. The
#: port is *runtime* state -- per machine, never in the manifest (§4).
DEFAULT_PORT: Final[int] = 8765

#: Environment variable the port falls back to, below ``--port`` and above the
#: default. Named like the vault env var so the two runtime knobs pair up.
PORT_ENV_VAR: Final[str] = "MECHABRAIN_PORT"

#: The host vault's ignore file, at its root -- where the derived-index rule goes.
_GITIGNORE_FILE: Final[str] = ".gitignore"

#: Fenced markers around the pasteable snippet, so an operator can see where to
#: cut. They are visual only -- unlike ``AGENTS.md``'s managed block, the kernel
#: never reads them back.
_SNIPPET_RULE: Final[str] = "-" * 60


# ══════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════
def _out(message: str = "") -> None:
    """Print operator-facing output to stdout."""
    print(message)


def _err(message: str) -> None:
    """Print a diagnostic or progress line to stderr, off the machine-readable path."""
    print(message, file=sys.stderr)


def _print_json(obj: Any) -> None:
    """Emit ``obj`` as indented UTF-8 JSON on stdout."""
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ══════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``mechabrain`` argument parser.

    Split out so a test can introspect it without dispatching. Every subparser
    binds its handler through ``set_defaults(handler=...)``, so :func:`main` never
    branches on the command name.
    """
    parser = argparse.ArgumentParser(
        prog="mechabrain",
        description="Drop-in agentic memory for Markdown vaults (CoALA-based).",
    )
    parser.add_argument(
        "--version", action="version", version=f"mechabrain {__version__}"
    )

    # Shared by every subcommand: the R4.3 explicit vault override.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--vault",
        metavar="PATH",
        help="vault root (parent of mecha-brain/); overrides $MECHABRAIN_VAULT "
        "and the upward search (R4.3)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser(
        "init",
        parents=[common],
        help="create the mecha-brain/ skeleton and manifest in a vault (idempotent)",
        description="Create the §3 skeleton, default config.yaml, .gitignore "
        "entry, AGENTS.md and schema.md. Re-running never destroys anything.",
    )
    p_init.add_argument(
        "vault_path",
        nargs="?",
        metavar="vault",
        help="vault root to initialize; defaults to --vault, $MECHABRAIN_VAULT, "
        "then the current directory",
    )
    p_init.add_argument("--json", action="store_true", help="print a JSON summary")
    p_init.set_defaults(handler=_cmd_init)

    p_sync = subparsers.add_parser(
        "sync",
        parents=[common],
        help="regenerate manifest-derived artifacts after editing config.yaml",
        description="Regenerate the AGENTS.md managed block and schema.md, and "
        "create Episodic/ subfolders for newly registered agents.",
    )
    p_sync.add_argument("--json", action="store_true", help="print a JSON summary")
    p_sync.set_defaults(handler=_cmd_sync)

    p_serve = subparsers.add_parser(
        "serve",
        parents=[common],
        help="run the MCP memory daemon",
        description="Bring up the MCP server over the discovered vault (R7.4). "
        "The port is runtime, not deployment: --port or $MECHABRAIN_PORT.",
    )
    p_serve.add_argument(
        "--host", default=None, help=f"bind address (default {DEFAULT_HOST})"
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"bind port (default: ${PORT_ENV_VAR} or {DEFAULT_PORT})",
    )
    p_serve.add_argument(
        "--stdio",
        action="store_true",
        help="serve over stdio for a single client instead of the local daemon; "
        "risks multiple writers -- see the warning it prints (R7.4)",
    )
    p_serve.set_defaults(handler=_cmd_serve)

    p_reindex = subparsers.add_parser(
        "reindex",
        parents=[common],
        help="rebuild the derived index",
        description="Bring the derived index up to date. --full clears and "
        "rebuilds from scratch; without it the pass is incremental.",
    )
    p_reindex.add_argument(
        "--full", action="store_true", help="clear and rebuild the whole index"
    )
    p_reindex.add_argument("--json", action="store_true", help="print a JSON report")
    p_reindex.set_defaults(handler=_cmd_reindex)

    p_consolidate = subparsers.add_parser(
        "consolidate",
        parents=[common],
        help="run the §9 maintenance pipeline",
        description="Flush accesses, decay, deprecate procedurals, rebuild the "
        "index and surfaces, and commit -- reporting merge candidates for an "
        "agent to act on.",
    )
    p_consolidate.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the report without writing anything",
    )
    p_consolidate.add_argument(
        "--no-commit",
        dest="commit",
        action="store_false",
        help="run every step but make no git commit",
    )
    p_consolidate.add_argument("--json", action="store_true", help="print a JSON report")
    p_consolidate.set_defaults(handler=_cmd_consolidate)

    p_check = subparsers.add_parser(
        "check",
        parents=[common],
        help="lint the deployment; exit non-zero on an error",
        description="Verify the manifest, the frontmatter contract, the §3 "
        "skeleton and the .gitignore rules. Warnings do not fail the build.",
    )
    p_check.add_argument("--json", action="store_true", help="print a JSON report")
    p_check.set_defaults(handler=_cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``mechabrain`` console script.

    Parses ``argv`` (defaulting to ``sys.argv[1:]``), dispatches to the selected
    subcommand's handler and returns its exit code. Any deliberate kernel error
    (:class:`~mechabrain.errors.MechabrainError`) is printed with its rule and
    hint and turned into exit code 1; everything else propagates, because an
    unexpected exception is a bug the operator should see in full (R5.1).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except MechabrainError as exc:
        _err(f"mechabrain: {exc}")
        return 1


# ══════════════════════════════════════════════════════════════════════
# init
# ══════════════════════════════════════════════════════════════════════
def _cmd_init(args: argparse.Namespace) -> int:
    """Create or repair the deployment, then print how to wire agents to it (§10).

    Idempotent by construction: directories are created with ``exist_ok``; the
    manifest is kept if present (:func:`write_default_config`); ``schema.md`` is
    regenerated deterministically; ``AGENTS.md`` only has its managed block
    replaced, preserving a human's free-form sections; ``index.md``/``hot.md`` are
    written only when absent, so a consolidator's populated maps are never clobbered.
    """
    paths = VaultPaths.for_root(_init_root(args))

    # Research/ is the one switchable folder (§3, zones.research_enabled), so it
    # waits for the manifest: a re-run must not resurrect a folder the
    # deployment disabled. Everything else is unconditional contract.
    for directory in paths.contract_dirs():
        if directory != paths.research_dir:
            directory.mkdir(parents=True, exist_ok=True)

    config_kept = paths.config_file.is_file()
    write_default_config(paths)
    manifest = load_manifest(paths.config_file)
    if manifest.zones.research_enabled:
        paths.research_dir.mkdir(parents=True, exist_ok=True)

    write_schema(paths, manifest)
    write_agents_md(paths, manifest)
    skill_file = write_mecha_scribe_skill(paths)

    index_created = _write_if_absent(paths.index_file, render_initial_index(manifest))
    hot_created = _write_if_absent(paths.hot_file, render_initial_hot(manifest))

    for agent_id in manifest.agent_ids():
        paths.episodic_for(agent_id).mkdir(parents=True, exist_ok=True)

    gitignore_added = _ensure_gitignore(paths)

    if args.json:
        _print_json(
            {
                "vault": str(paths.root),
                "config": paths.relative(paths.config_file),
                "config_kept": config_kept,
                "agents": list(manifest.agent_ids()),
                "index_created": index_created,
                "hot_created": hot_created,
                "gitignore_added": gitignore_added,
                "scribe_skill": paths.relative(skill_file),
            }
        )
    else:
        _print_init_human(paths, manifest, config_kept, gitignore_added)
    return 0


def _init_root(args: argparse.Namespace) -> Path:
    """Resolve where ``init`` should act, tolerating a not-yet-initialized vault.

    Precedence: the positional ``vault`` argument, then ``--vault``, then
    ``$MECHABRAIN_VAULT``, then the current directory. Unlike the other
    subcommands, ``init`` does not discover an *existing* vault -- creating one is
    the whole point -- so it never walks up the tree.
    """
    if args.vault_path is not None:
        return Path(args.vault_path).expanduser()
    if args.vault is not None:
        return Path(args.vault).expanduser()
    from_env = os.environ.get(VAULT_ENV_VAR)
    if from_env and from_env.strip():
        return Path(from_env.strip()).expanduser()
    return Path.cwd()


def _write_if_absent(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` only if nothing is there yet. Returns whether it wrote.

    The consolidator owns ``index.md``/``hot.md`` at runtime (R8.1); ``init``
    seeds them once so the files exist, but a re-run must not overwrite what a
    consolidation pass has since populated.
    """
    if path.exists():
        return False
    write_atomic(path, text)
    return True


def _ensure_gitignore(paths: VaultPaths) -> bool:
    """Add the derived-index rule to the vault's ``.gitignore`` without duplicating it.

    Creates the file if it is absent. The derived index is per-machine runtime
    state and must never be committed (§4, §10). Returns whether a line was added,
    so a re-run reports "already present" rather than churning the file.
    """
    gitignore = paths.root / _GITIGNORE_FILE
    entry = GITIGNORE_INDEX_ENTRY
    if gitignore.is_file():
        existing = gitignore.read_text(encoding="utf-8")
        present = {line.strip() for line in existing.splitlines()}
        if entry in present or entry.rstrip("/") in present:
            return False
        separator = "" if not existing or existing.endswith("\n") else "\n"
        new_text = f"{existing}{separator}{entry}\n"
    else:
        new_text = f"{entry}\n"
    write_atomic(gitignore, new_text)
    return True


def _print_init_human(
    paths: VaultPaths, manifest: Manifest, config_kept: bool, gitignore_added: bool
) -> None:
    """The human summary of an ``init`` run, plus the integration snippet."""
    agents = ", ".join(manifest.agent_ids()) or "(none registered -- add some to config.yaml)"
    _out(f"mechabrain: initialized vault at {paths.root}")
    _out(f"  manifest:  {paths.relative(paths.config_file)}" + (" (kept existing)" if config_kept else " (written)"))
    _out("  generated: AGENTS.md, _meta/schema.md, index.md, hot.md")
    _out("  skill:     .claude/skills/mecha-scribe/SKILL.md (escriba -- documentar na vault)")
    _out(f"  agents:    {agents}")
    _out(
        "  gitignore: added " + repr(GITIGNORE_INDEX_ENTRY)
        if gitignore_added
        else f"  gitignore: {GITIGNORE_INDEX_ENTRY!r} already present"
    )
    _out("")
    _out("Next steps:")
    _out(f"  1. Review the manifest:   {paths.relative(paths.config_file)}")
    _out(f"  2. Start the daemon:      mechabrain serve --vault {paths.root}")
    _out(
        "  3. Register it with your MCP client at its local SSE endpoint:\n"
        f"       http://{DEFAULT_HOST}:<port>/sse   "
        f"(port: --port, ${PORT_ENV_VAR}, or {DEFAULT_PORT})"
    )
    _out("  4. Paste the block below into the vault's CLAUDE.md / agent instructions:")
    _out("")
    _out(_SNIPPET_RULE)
    _out(_integration_snippet(manifest))
    _out(_SNIPPET_RULE)


def _integration_snippet(manifest: Manifest) -> str:
    """The pt-BR block an operator pastes into the host vault's agent instructions.

    Deployment content, not kernel output: it points agents at ``AGENTS.md`` (the
    real contract) and names the MCP tools, in the language of that contract. It
    stays deliberately short -- the authority lives in ``AGENTS.md``, which the
    manifest regenerates; duplicating its detail here would only drift.
    """
    scopes = manifest.scopes.known
    scope_hint = (
        ", ".join(scopes) if scopes else f"qualquer slug — default {manifest.scopes.default}"
    )
    return (
        "## Memória agentica (Mecha-Brain)\n"
        "\n"
        "Esta vault tem memória agentica compartilhada, exposta por MCP. Antes de\n"
        "agir, busque memória relevante; ao aprender algo reutilizável e citável,\n"
        "grave-o.\n"
        "\n"
        "- Contrato completo e sempre atual: `mecha-brain/AGENTS.md` — **leia antes de escrever**.\n"
        "- Ferramentas MCP: `memory_search`, `memory_get`, `memory_status`,\n"
        "  `memory_write`, `memory_propose`, `memory_link`.\n"
        "- Todo resultado carrega `path`/`wikilink`: **cite a fonte** (P7).\n"
        f"- Filtre `memory_search` pelo escopo do trabalho corrente ({scope_hint}); um\n"
        "  hit de outro escopo é contexto, não verdade local.\n"
    )


# ══════════════════════════════════════════════════════════════════════
# sync
# ══════════════════════════════════════════════════════════════════════
def _cmd_sync(args: argparse.Namespace) -> int:
    """Regenerate the manifest-derived artifacts after a config edit (§10).

    The narrow counterpart of ``init``: it rewrites the ``AGENTS.md`` managed
    block and ``schema.md`` from the current manifest and creates an
    ``Episodic/<agent>/`` for every newly registered agent. It never touches the
    manifest, the notes or the consolidator-owned maps.
    """
    paths = _discover(args)
    manifest = load_manifest(paths.config_file)

    write_agents_md(paths, manifest)
    write_schema(paths, manifest)
    write_mecha_scribe_skill(paths)

    created: list[str] = []
    for agent_id in manifest.agent_ids():
        folder = paths.episodic_for(agent_id)
        if not folder.exists():
            created.append(agent_id)
        folder.mkdir(parents=True, exist_ok=True)

    if args.json:
        _print_json(
            {
                "vault": str(paths.root),
                "agents": list(manifest.agent_ids()),
                "episodic_created": created,
            }
        )
    else:
        _out(f"mechabrain: synced {paths.root}")
        _out("  regenerated: AGENTS.md (managed block), _meta/schema.md")
        if created:
            _out(f"  new Episodic/ subfolders: {', '.join(created)}")
        else:
            _out("  Episodic/ subfolders: already current")
    return 0


# ══════════════════════════════════════════════════════════════════════
# serve
# ══════════════════════════════════════════════════════════════════════
def _cmd_serve(args: argparse.Namespace) -> int:
    """Bring up the MCP daemon over the discovered vault (R7.4).

    Fails fast on a bad vault or an unparseable port *before* handing off to the
    server, so an operator is not left with a half-bound daemon. The handoff goes
    through :func:`_run_daemon`, which a test overrides and the daemon component
    fills.
    """
    paths = _discover(args)
    manifest = load_manifest(paths.config_file)
    host = args.host or DEFAULT_HOST
    port = _resolve_port(args.port)

    if args.stdio:
        _err(
            "mechabrain serve: --stdio runs one kernel process per client; "
            "concurrent sessions become multiple writers to one index and can "
            "corrupt it (R7.4) -- prefer the default local daemon."
        )
        _err(f"mechabrain serve: starting stdio server for {paths.root}")
    else:
        _err(
            f"mechabrain serve: starting daemon for {paths.root} on "
            f"http://{host}:{port}/sse"
        )

    result = _run_daemon(paths, manifest, host=host, port=port, stdio=args.stdio, emit=_err)
    return int(result) if result is not None else 0


def _resolve_port(explicit: int | None) -> int:
    """Resolve the runtime port: ``--port``, then ``$MECHABRAIN_PORT``, then the default.

    Raises:
        MechabrainError: ``$MECHABRAIN_PORT`` is set but not an integer -- a
            typo the operator must see, not a silent fallback to the default (R5.1).
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(PORT_ENV_VAR)
    if raw and raw.strip():
        try:
            return int(raw.strip())
        except ValueError:
            raise MechabrainError(
                f"{PORT_ENV_VAR}={raw!r} is not an integer port",
                rule="R7.4",
                hint=f"set {PORT_ENV_VAR} to a port number, or pass --port",
            ) from None
    return DEFAULT_PORT


def _run_daemon(
    paths: VaultPaths,
    manifest: Manifest,
    *,
    host: str,
    port: int,
    stdio: bool,
    emit: Any,
) -> int | None:
    """Delegate to the MCP server component, or explain that it is not installed.

    ``serve`` is the one subcommand that needs the ``mcp`` package at runtime,
    so this imports the daemon lazily and calls
    :func:`mechabrain.mcp_server.serve`. A missing ``mcp`` dependency fails
    with an actionable message rather than a raw ``ImportError`` -- every other
    subcommand works without it.
    """
    del manifest, emit  # the daemon loads the manifest itself and logs its own progress
    try:
        from .mcp_server import serve as serve_entry  # noqa: PLC0415 -- needs `mcp` at runtime
    except ImportError as exc:
        raise MechabrainError(
            "the MCP daemon cannot start: the `mcp` package is not importable",
            rule="R7.4",
            hint="reinstall mechabrain (the `mcp` dependency is required for serve); "
            "init, sync, reindex, consolidate and check do not require it",
        ) from exc
    serve_entry(
        vault=paths.root,
        host=host,
        port=port,
        transport="stdio" if stdio else "sse",
    )
    return 0


# ══════════════════════════════════════════════════════════════════════
# reindex
# ══════════════════════════════════════════════════════════════════════
def _cmd_reindex(args: argparse.Namespace) -> int:
    """Rebuild the derived index, incrementally or in full (§10).

    Progress lines go to stderr so ``--json`` keeps stdout to the one report
    object. A ``--full=False`` pass may still be forced to a full rebuild by the
    indexer's fingerprint check; the report says so in ``reason``.
    """
    paths = _discover(args)
    manifest = load_manifest(paths.config_file)

    progress = None if args.json else _err
    indexer = Indexer(paths, manifest, progress=progress)
    report = indexer.reindex(full=args.full)

    if args.json:
        _print_json(dataclasses.asdict(report))
    else:
        _out(_render_index_report(report))
    return 0


def _render_index_report(report: IndexReport) -> str:
    """A compact human summary of an :class:`IndexReport`."""
    lines = [f"mechabrain reindex: {'full rebuild' if report.full else 'incremental'}"]
    if report.reason:
        lines.append(f"  forced full: {report.reason}")
    lines.append(
        f"  notes: {report.notes_indexed} indexed, {report.notes_unchanged} "
        f"unchanged, {report.notes_removed} removed ({report.notes_total} total)"
    )
    lines.append(
        f"  chunks: +{report.chunks_written} / -{report.chunks_deleted} "
        f"({report.chunks_total} total)"
    )
    if report.read_only_indexed:
        lines.append(f"  read-only context indexed: {report.read_only_indexed}")
    if report.ambiguous_ids:
        lines.append(
            f"  ambiguous ids (only the first path was indexed): "
            f"{', '.join(report.ambiguous_ids)}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# consolidate
# ══════════════════════════════════════════════════════════════════════
def _cmd_consolidate(args: argparse.Namespace) -> int:
    """Run the §9 maintenance pipeline over the discovered vault (§10)."""
    paths = _discover(args)
    manifest = load_manifest(paths.config_file)

    report = run_consolidate(paths, manifest, dry_run=args.dry_run, commit=args.commit)

    if args.json:
        _print_json(report.as_dict())
    else:
        _out(_render_consolidation_report(report))
    return 0


def _render_consolidation_report(report: Any) -> str:
    """A human summary of a :class:`~mechabrain.consolidate.ConsolidationReport`."""
    counts = report.counts
    lines = ["mechabrain consolidate" + (" (dry run)" if report.dry_run else "") + ":"]
    lines.append(
        f"  scanned {counts.get('notes_scanned', 0)} memory note(s), "
        f"{counts.get('readonly_scanned', 0)} read-only"
    )
    lines.append(f"  access stamps applied: {counts.get('accesses_applied', 0)}")
    lines.append(f"  archived (decay): {counts.get('decayed', 0)}")
    lines.append(f"  deprecated procedurals: {counts.get('deprecated', 0)}")
    lines.append(f"  merge candidates (same scope): {counts.get('merge_candidates', 0)}")
    lines.append(
        f"  cross-scope similar (never merged): {counts.get('cross_scope_similar', 0)}"
    )
    lines.append(
        f"  stale procedurals (retest suggested): {counts.get('stale_procedurals', 0)}"
    )
    lines.append(f"  docs citing dead memories: {counts.get('docs_citing_dead', 0)}")
    lines.append(f"  chunks indexed: {counts.get('chunks_indexed', 0)}")
    if report.committed:
        lines.append(f"  committed: {report.commit}")
    elif not report.dry_run:
        lines.append("  committed: nothing to commit (or the vault is not a git repo)")
    if report.merge_candidates:
        lines.append("  review these merge candidates (an agent decides supersedes/merge):")
        for pair in report.merge_candidates:
            lines.append(
                f"    - [[{pair.a}]] ~ [[{pair.b}]]  "
                f"({pair.similarity:.2f}, {pair.memory_type}, scope {pair.scope_a})"
            )
    if report.stale_procedurals:
        lines.append("  retest these stale procedurals (an agent runs and refreshes them):")
        for stale in report.stale_procedurals:
            tested = (
                f"last tested {stale.last_tested.isoformat()}"
                if stale.last_tested
                else "never tested since creation"
            )
            lines.append(
                f"    - [[{stale.note_id}]]  ({tested}, {stale.days_stale} days, "
                f"scope {stale.scope})"
            )
    if report.docs_citing_dead:
        lines.append("  update these docs -- they cite dead memories (propose the edit):")
        for cite in report.docs_citing_dead:
            arrow = f" -> [[{cite.successor}]]" if cite.successor else ""
            lines.append(f"    - [[{cite.doc}]] cites [[{cite.cited}]] ({cite.status}{arrow})")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# check
# ══════════════════════════════════════════════════════════════════════
def _cmd_check(args: argparse.Namespace) -> int:
    """Lint the deployment and return the report's exit code (§10).

    Discovery only checks that a manifest *file* is present; ``check`` itself
    loads and validates it, so a malformed manifest surfaces as a reported
    problem with a non-zero exit rather than an exception here.
    """
    paths = _discover(args)
    report = run_check(paths)

    if args.json:
        _print_json(
            {
                "ok": report.ok,
                "exit_code": report.exit_code,
                "errors": len(report.errors),
                "warnings": len(report.warnings),
                "problems": [
                    {
                        "check": problem.check,
                        "rule": problem.rule,
                        "severity": str(problem.severity),
                        "message": problem.message,
                        "hint": problem.hint,
                        "path": problem.path,
                    }
                    for problem in report.problems
                ],
            }
        )
    else:
        _out(report.render())
    return report.exit_code


# ══════════════════════════════════════════════════════════════════════
# Discovery
# ══════════════════════════════════════════════════════════════════════
def _discover(args: argparse.Namespace) -> VaultPaths:
    """Resolve the vault for a subcommand that requires an initialized one (R4.3).

    Passes ``--vault`` through to :func:`~mechabrain.discovery.discover_vault`,
    which then honours ``$MECHABRAIN_VAULT`` and the upward walk in order and
    raises an actionable :class:`~mechabrain.errors.VaultNotFoundError` on a miss.
    """
    return discover_vault(explicit=args.vault)


if __name__ == "__main__":
    raise SystemExit(main())
