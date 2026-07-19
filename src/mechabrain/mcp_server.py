"""The MCP server: the kernel's contract to agents (spec §7).

`mechabrain serve` runs this as a **local daemon** and every agent's MCP client
points at it. That single-daemon shape is the whole concurrency model (R7.4):
one writer per machine, so two agents cannot corrupt the derived index by
writing it at once. The transport therefore defaults to HTTP/SSE bound to
``127.0.0.1`` with a runtime port; the per-session stdio mode -- one kernel
process per client, i.e. many writers -- is available only behind an explicit
flag and with a loud warning.

The six tools, and the notation
===============================

The spec writes the tools with a dotted namespace (``memory.search``); MCP tool
names cannot carry a ``.``, so each maps to the underscored name an MCP client
actually calls:

======================  ==============  ============================================
Spec (§7)               MCP tool        What it does
======================  ==============  ============================================
``memory.search``       memory_search   hybrid retrieval + link expansion (§7.1)
``memory.get``          memory_get      one note, frontmatter + body
``memory.status``       memory_status   index health, counts, last consolidation
``memory.write``        memory_write    the §8.2 write gate, then a governed write
``memory.propose``      memory_propose  a proposal for a note outside the sandbox
``memory.link``         memory_link     an authored edge that feeds expansion
======================  ==============  ============================================

Where the policy lives
======================

A tool *description* is a prompt the client agent reads before it calls. That is
where the §8.1 routing tree, the §8.2 gate checklist and the normative
recommendation to filter by the current ``scope`` are stated in imperative form
-- see :data:`_SEARCH_DESCRIPTION` and friends. The kernel does not merely
document the policy here; the description is how the policy becomes behaviour on
the client side, and the gate (:mod:`mechabrain.gate`) is how the mechanical half
of it is enforced on this side.

No LLM, and honest errors
=========================

This module calls no model of its own: it wires the deterministic kernel modules
(:mod:`~mechabrain.search`, :mod:`~mechabrain.writer`, :mod:`~mechabrain.graph`,
:mod:`~mechabrain.index.indexer`) behind the transport and translates their
results into JSON an agent can act on. A gate rejection is a *normal* result --
``{"rejected": true, "reason": ..., "near_duplicates": [...]}`` -- not an
exception. Every other kernel error (a bad filter, an unknown agent, a missing
note) is caught at the tool boundary and returned as a structured
``{"error", "rule", "hint"}`` object, so an agent gets an actionable message
rather than a stack trace.

State and consistency
=====================

:class:`MemoryService` holds the vault, the manifest and the long-lived read
objects a daemon must not rebuild per request (R7.4): the authored
:class:`~mechabrain.graph.LinkGraph` and the :class:`~mechabrain.search.Retriever`
built over it. A write changes the corpus, so it reindexes and then drops those
snapshots; the next read rebuilds them against the new bytes. Every tool call is
serialised on one lock -- the daemon is the single writer, and this keeps its own
concurrent requests from racing the snapshot lifecycle.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

from mcp.server.fastmcp import FastMCP

from .access import AccessKind, AccessLog
from .actions import ActionLog
from .consolidate import CONSOLIDATION_REPORT_FILE
from .contract import STATUS_ACTIVE, MemoryType
from .discovery import VaultPaths, discover_vault
from .errors import MechabrainError, NoteNotFound
from .gate import GateIssue, NearDuplicate
from .graph import DEFAULT_RELATION, LinkGraph
from .index.indexer import Indexer
from .index.store import INDEX_LOCK_FILE
from .index.store import from_manifest as store_from_manifest
from .locking import FileLock
from .manifest import Manifest, load_manifest
from .note import Note, iter_notes, note_id_for, wikilink_for
from .search import Retriever
from .writer import propose as writer_propose
from .writer import write as writer_write

__all__ = [
    "MemoryService",
    "build_server",
    "serve",
    "resolve_host",
    "resolve_port",
    "SERVER_NAME",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "HOST_ENV_VAR",
    "PORT_ENV_VAR",
    "TRANSPORTS",
]

logger = logging.getLogger("mechabrain.mcp_server")

#: Name the daemon advertises to MCP clients. A kernel identifier (not a
#: deployment name), identical in every vault (R4.1).
SERVER_NAME: Final[str] = "mechabrain"

#: Loopback by default (R7.4): the daemon serves the agents on this one machine,
#: never the network. A deployment that wants otherwise overrides it at runtime.
DEFAULT_HOST: Final[str] = "127.0.0.1"

#: Default TCP port. Runtime config, deliberately distinctive to avoid the common
#: ``8000``/``8080`` dev-server clash. Never the manifest's: the port is a
#: per-machine address like the embedding endpoint (§4), so it comes from a flag
#: or the environment, never from the synced deployment.
DEFAULT_PORT: Final[int] = 8765

#: Environment override for the bind host. Runtime layer, never the manifest.
HOST_ENV_VAR: Final[str] = "MECHABRAIN_HOST"

#: Environment override for the bind port. Runtime layer, never the manifest.
PORT_ENV_VAR: Final[str] = "MECHABRAIN_PORT"

#: Transports :func:`serve` accepts. ``sse``/``streamable-http`` are the daemon
#: shape R7.4 mandates; ``stdio`` is the per-session escape hatch, gated behind a
#: multi-writer warning.
TRANSPORTS: Final[tuple[str, ...]] = ("sse", "streamable-http", "stdio")

#: Hops walked by the gate's internal dedup search. Zero: dedup compares the
#: candidate against notes that already exist, and a link-expanded neighbour is
#: context, not a duplicate (§8.2 item 2).
_DEDUP_HOPS: Final[int] = 0

#: Chunk count past which `memory_status` warns about the numpy store. Brute
#: force is the right default at personal-vault scale, but its cost is linear:
#: at ~50k chunks (1024-dim float32 ≈ 200MB matrix) a search is tens of
#: milliseconds and growing -- time to consider `lancedb`/`sqlite-vec`.
NUMPY_CHUNKS_WARNING: Final[int] = 50_000


# ══════════════════════════════════════════════════════════════════════
# Service
# ══════════════════════════════════════════════════════════════════════
class MemoryService:
    """The kernel behind the six MCP tools, over one discovered vault (§7).

    Construct it once per daemon. It owns the read snapshots a daemon must keep
    warm (R7.4) -- the authored :class:`~mechabrain.graph.LinkGraph` and a
    :class:`~mechabrain.search.Retriever` over it -- and the incremental
    :class:`~mechabrain.index.indexer.Indexer`. Each write indexes just the note
    it wrote (and any it archived) via ``index_note`` and drops the snapshots so
    the next read sees the new bytes -- never a whole-corpus re-embed.

    Every method is safe to call from :func:`build_server`'s tools directly; the
    tools add only the JSON-error translation. Methods raise
    :class:`~mechabrain.errors.MechabrainError` (or ``ValueError``) on a bad
    request and return plain, JSON-serialisable ``dict``\\ s otherwise. A gate
    rejection is a returned ``dict``, not a raise -- it is an expected outcome.

    Not safe to share across threads without the internal lock this class already
    takes on every public method: the retriever holds one SQLite connection, and
    the snapshot lifecycle is a read-modify-write of shared state.

    Args:
        paths: The discovered vault (:func:`mechabrain.discovery.discover_vault`).
        manifest: Its parsed manifest -- the single source of every threshold,
            registry and weight consulted here (P6).
    """

    __slots__ = ("paths", "manifest", "_indexer", "_index_lock", "_lock", "_graph", "_retriever")

    def __init__(self, paths: VaultPaths, manifest: Manifest) -> None:
        self.paths = paths
        self.manifest = manifest
        # One FileLock instance shared with the indexer so a write can hold it
        # and the surgical index_note nested inside re-enters reentrantly instead
        # of deadlocking (FileLock is reentrant within an instance) -- the same
        # pattern consolidate uses for its cycle.
        self._index_lock = FileLock(paths.index_dir / INDEX_LOCK_FILE, purpose="memory_write")
        self._indexer = Indexer(paths, manifest, lock=self._index_lock)
        self._lock = threading.RLock()
        self._graph: LinkGraph | None = None
        self._retriever: Retriever | None = None

    # ── Lifecycle ────────────────────────────────────────────────────
    def close(self) -> None:
        """Release the retriever's index connection. Idempotent."""
        with self._lock:
            self._drop_snapshots()

    def __enter__(self) -> "MemoryService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── memory.search (§7.1) ─────────────────────────────────────────
    def memory_search(
        self,
        query: str,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
        expand_links: int | None = None,
    ) -> dict[str, Any]:
        """Hybrid retrieval with link expansion (§7.1).

        Delegates to :meth:`mechabrain.search.Retriever.search`, which fuses the
        vector and BM25 halves with the manifest weights, applies the ``filters``,
        walks the authored graph ``expand_links`` hops from the top hits, and
        records the touch to the access log (R7.2). Every hit carries its
        provenance -- ``path`` and ``wikilink`` (R7.1) -- and an expanded hit
        carries its ``via`` chain.

        Returns:
            ``{"hits": [<hit>, ...], "count": n}`` where each hit is the §7.1 wire
            shape (see :meth:`mechabrain.search.Hit.as_dict`).

        Raises:
            ValueError: ``k`` < 1, an unknown/malformed filter, or ``expand_links``
                out of the manifest's range.
            MechabrainError: the index or embedding backend is unavailable.
        """
        with self._lock:
            retriever = self._get_retriever()
            hits = retriever.search(query, k=k, filters=filters, expand_links=expand_links)
        return {"hits": [hit.as_dict() for hit in hits], "count": len(hits)}

    # ── memory.get (§7.1) ────────────────────────────────────────────
    def memory_get(self, id_or_wikilink: str) -> dict[str, Any]:
        """Return one note in full -- frontmatter and body (§7.1).

        Accepts a bare id or a ``[[wikilink]]``; both reduce to the same note id.
        Resolves against the authored graph, whose nodes are exactly the notes
        retrieval can return, and records the read to the access log (R7.2).

        Returns:
            ``{"id", "path", "wikilink", "title", "type", "frontmatter", "body",
            "tags"}``. ``path`` is vault-relative (R4.2); date values in
            ``frontmatter`` are rendered as ISO strings.

        Raises:
            NoteNotFound: no note carries that id.
        """
        note_id = note_id_for(_strip_wikilink(id_or_wikilink))
        with self._lock:
            graph = self._get_graph()
            if not graph.has_note(note_id):
                raise NoteNotFound(
                    f"no note {note_id!r} in the vault",
                    hint="check the id, or run memory_search to find it",
                )
            note = Note.load(graph.path_of(note_id))
            AccessLog.for_vault(self.paths).record(note_id, AccessKind.GET)
        path = note.path
        return {
            "id": note_id,
            "path": self.paths.relative(path) if path is not None else "",
            "wikilink": note.wikilink,
            "title": note.title,
            "type": self._type_of(note),
            "frontmatter": _jsonable(note.frontmatter),
            "body": note.body,
            "tags": note.tags,
        }

    # ── memory.status (§7.1) ─────────────────────────────────────────
    def memory_status(self) -> dict[str, Any]:
        """Index health, counts by type/scope/status, last consolidation (§7.1).

        A read-only snapshot for an agent to sanity-check the store it is talking
        to. Counts come from scanning the enabled memory folders (the Markdown is
        the source of truth, P1); the chunk count comes from the vector store; the
        broken-link count is an authored-graph health signal; the last
        consolidation timestamp is read from the maintenance report (§9), if one
        has been written.

        Returns:
            A nested ``dict`` of counts and health fields, all JSON-serialisable.
        """
        with self._lock:
            total, by_type, by_scope, by_status = self._scan_counts()
            graph = self._get_graph()
            graph_health = {
                "nodes": len(graph.note_ids),
                "edges": len(graph.edges),
                "broken_links": len(graph.broken_links),
                "ambiguous_ids": list(graph.ambiguous_ids),
            }
            index_health = self._index_health()
        embedding = self.manifest.retrieval.embedding
        return {
            "vault_root": str(self.paths.root),
            "notes": total,
            "by_type": by_type,
            "by_scope": by_scope,
            "by_status": by_status,
            "index": {
                "store": self.manifest.retrieval.store,
                "embedding_provider": embedding.provider,
                "embedding_model": embedding.model,
                **index_health,
            },
            "graph": graph_health,
            "last_consolidation": self._last_consolidation(),
        }

    # ── memory.write (§7.2) ──────────────────────────────────────────
    def memory_write(
        self,
        type: str,  # noqa: A002 -- the §7.2 `memory_write(type, ...)` parameter name
        content: str,
        meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run the §8.2 write gate and, if it approves, write the note (§7.2).

        Routes through :func:`mechabrain.writer.write`, which evaluates the gate
        first (the dedup search is this service's own retriever, scoped and typed
        as §8.2 item 2 requires) and writes nothing if it rejects. On approval it
        resolves the filename from the manifest, writes the note atomically
        (R7.5), archives whatever it supersedes (P8), and indexes the new note
        (and any it archived) incrementally via ``index_note`` -- immediately
        searchable, with no re-embed of the unchanged corpus (§7.2 step 8).

        The gate, the write and the incremental index all run under one held
        index lock: this service and its indexer share a single ``FileLock``
        instance, so the ``index_note`` nested in the write re-enters it
        reentrantly. The snapshots are dropped after, so no concurrent read sees
        a half-state.

        Args:
            type: Target memory type: ``episodic``, ``semantic``, ``procedural``
                or ``research``.
            content: The note body, without frontmatter.
            meta: The §6 frontmatter to carry (``title``, ``agent``, ``scope``,
                ``source``, ``confidence``, optional ``profile``/``supersedes``/
                ``tags``) plus the gate-only ``evidence`` (§8.2 item 6) and
                ``merge`` (§8.2 item 2) keys.

        Returns:
            On success ``{"rejected": false, "path", "id", "wikilink", "warnings",
            "superseded", "superseded_missing", "superseded_episodic",
            "near_duplicates"}``; on a gate rejection ``{"rejected": true,
            "reason", "near_duplicates", "warnings"}``.

        Raises:
            ValueError: ``type`` is not a memory type.
            MechabrainError: the embedding backend or index is unavailable.
        """
        with self._lock, self._index_lock:
            # Surgical, not whole-vault: we hold the shared index lock and pass
            # the indexer to the writer, so the write's incremental hook indexes
            # exactly the new note (and any note a supersede archived) via
            # index_note -- no re-embed of the unchanged corpus (§7.2 step 8).
            # lock=False because we already hold the one lock instance the
            # nested index_note re-enters.
            result = writer_write(
                type,
                content,
                meta,
                self.manifest,
                self.paths,
                search_fn=self._dedup_search_fn(),
                indexer=self._indexer,
                lock=False,
            )
            if result.rejected:
                # Observability (v0.2.1): a rejection exists nowhere else --
                # the note was never written -- so the action log is its record.
                ActionLog.for_vault(self.paths).record(
                    "write_rejected",
                    type=str(type),
                    agent=str(meta.get("agent") or ""),
                    scope=str(meta.get("scope") or ""),
                    title=str(meta.get("title") or ""),
                    reason=result.reason.splitlines()[0] if result.reason else "",
                    near_duplicates=len(result.near_duplicates),
                )
                return {
                    "rejected": True,
                    "reason": result.reason,
                    "near_duplicates": [_near_duplicate(nd) for nd in result.near_duplicates],
                    "warnings": [_issue(issue) for issue in result.warnings],
                }
            self._drop_snapshots()
            ActionLog.for_vault(self.paths).record(
                "write_accepted",
                type=str(type),
                id=result.note_id,
                agent=str(meta.get("agent") or ""),
                scope=str(meta.get("scope") or ""),
                superseded=list(result.superseded),
                warnings=[issue.check for issue in result.warnings],
            )
            return {
                "rejected": False,
                "path": self.paths.relative(result.path) if result.path is not None else "",
                "id": result.note_id,
                "wikilink": result.wikilink,
                "warnings": [_issue(issue) for issue in result.warnings],
                "near_duplicates": [_near_duplicate(nd) for nd in result.near_duplicates],
                "superseded": list(result.superseded),
                "superseded_missing": list(result.superseded_missing),
                "superseded_episodic": list(result.superseded_episodic),
            }

    # ── memory.propose (§7.2) ────────────────────────────────────────
    def memory_propose(
        self,
        target_path: str,
        proposed_change: str,
        rationale: str,
        meta: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record a proposed change to a note outside ``mecha-brain/`` (§7.2).

        The only sanctioned way for an agent to affect a note it may not write
        directly (P4, §8.1 item 7). Delegates to
        :func:`mechabrain.writer.propose`, which writes a proposal into
        ``zones.proposals_dir`` and **never touches the target file** -- a human
        reviews and applies it. The proposal lives in ``_inbox/`` and is not
        indexed, so no reindex follows.

        Args:
            target_path: The note the proposal is about, referenced but never read.
            proposed_change: The suggested edit, diff or new text.
            rationale: Why the change is proposed (P7: provenance is mandatory).
            meta: Authorship: ``agent`` (required, must be registered -- R6.2),
                optional ``profile``/``scope``/``source``/``confidence``.

        Returns:
            ``{"path", "id", "wikilink", "target"}`` naming the proposal note.

        Raises:
            MechabrainError: ``agent`` is missing/unregistered (R6.2), the profile
                is undeclared (R6.6), or an assembled key/tag is denied (R6.1).
        """
        with self._lock:
            result = writer_propose(
                target_path, proposed_change, rationale, meta, self.manifest, self.paths
            )
            ActionLog.for_vault(self.paths).record(
                "proposal",
                id=result.note_id,
                target=result.target,
                agent=str(meta.get("agent") or ""),
            )
        return {
            "path": self.paths.relative(result.path) if result.path is not None else "",
            "id": result.note_id,
            "wikilink": result.wikilink,
            "target": result.target,
        }

    # ── memory.link (§7.2) ───────────────────────────────────────────
    def memory_link(
        self,
        a: str,
        b: str,
        relation: str = DEFAULT_RELATION,
        agent: str | None = None,
    ) -> dict[str, Any]:
        """Record an authored edge between two notes (§7.2).

        Appends the edge to the git-tracked ``_meta/links.jsonl`` (a source of
        truth, not derived state) and drops the read snapshots so the next
        :meth:`memory_search` walks it. This is the graph the link-expansion of
        §7.1 consults: the better agents link, the better the multi-hop recall --
        the graph improves by curation, never by LLM extraction.

        Args:
            a: Source note -- id, wikilink or vault path; all resolve the same way.
            b: Target note.
            relation: What the author means by the edge (default ``related``).
            agent: The runtime recording it, validated against the registry (R6.2)
                when given. Provide the caller's own id for provenance (P7);
                ``None`` records an edge with no author.

        Returns:
            ``{"a", "b", "relation", "wikilink_a", "wikilink_b", "created",
            "agent"}`` describing the recorded edge.

        Raises:
            NoteNotFound: either end resolves to no note.
            ValueError: the two ends are the same note, or ``relation`` is empty.
            MechabrainError: ``agent`` is not registered (R6.2).
        """
        with self._lock:
            graph = self._get_graph()
            edge = graph.add_edge(a, b, relation, agent)
            self._drop_snapshots()
            ActionLog.for_vault(self.paths).record(
                "link", a=edge.source, b=edge.target, relation=edge.relation, agent=agent
            )
        return {
            "a": edge.source,
            "b": edge.target,
            "relation": edge.relation,
            "wikilink_a": wikilink_for(edge.source),
            "wikilink_b": wikilink_for(edge.target),
            "created": edge.created,
            "agent": edge.agent,
        }

    # ── Snapshot lifecycle ───────────────────────────────────────────
    def _get_graph(self) -> LinkGraph:
        """The authored graph, built once and kept until a write invalidates it."""
        if self._graph is None:
            self._graph = LinkGraph.build(self.paths, self.manifest)
        return self._graph

    def _get_retriever(self) -> Retriever:
        """The retriever over the current graph, built lazily and kept warm (R7.4)."""
        if self._retriever is None:
            self._retriever = Retriever(self.paths, self.manifest, graph=self._get_graph())
        return self._retriever

    def _drop_snapshots(self) -> None:
        """Close and forget the retriever and graph; the next read rebuilds them."""
        if self._retriever is not None:
            self._retriever.close()
            self._retriever = None
        self._graph = None

    def _dedup_search_fn(self):  # type: ignore[no-untyped-def]
        """The gate's internal search (§8.2 item 2), as :class:`mechabrain.gate.SearchFn`.

        Lazy: the gate calls it only for the gated types, so an episodic or
        research write never builds the retriever. Runs with expansion off -- a
        near-duplicate is a note that already exists, not one an authored link
        reaches -- and yields the §7.1 hit shape, whose ``similarity`` is the raw
        cosine the gate calibrates against ``maintenance.dedup_similarity``.
        """

        def search_fn(
            query: str,
            *,
            k: int = 8,
            filters: Mapping[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            hits = self._get_retriever().search(query, k=k, filters=filters, expand_links=_DEDUP_HOPS)
            return [hit.as_dict() for hit in hits]

        return search_fn

    # ── memory.status helpers ────────────────────────────────────────
    def _scan_counts(self) -> tuple[int, dict[str, int], dict[str, int], dict[str, int]]:
        """Tally the enabled memory folders by type, scope and status."""
        by_type: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total = 0
        for memory_type in MemoryType:
            if not self.manifest.is_enabled(memory_type):
                continue
            for note in iter_notes(self.paths.folder_for(memory_type)):
                total += 1
                by_type[memory_type.value] = by_type.get(memory_type.value, 0) + 1
                scope = self._scope_of(note)
                by_scope[scope] = by_scope.get(scope, 0) + 1
                status = _status_of(note)
                by_status[status] = by_status.get(status, 0) + 1
        return total, by_type, by_scope, by_status

    def _index_health(self) -> dict[str, Any]:
        """Chunk count and lexical-index presence, tolerant of an unbuilt index."""
        chunks: int | None
        try:
            chunks = store_from_manifest(self.manifest, self.paths).count()
        except MechabrainError as exc:
            logger.warning("cannot read the vector store for memory_status: %s", exc)
            chunks = None
        health: dict[str, Any] = {
            "chunks": chunks,
            "built": chunks is not None and chunks > 0,
        }
        if (
            self.manifest.retrieval.store == "numpy"
            and chunks is not None
            and chunks > NUMPY_CHUNKS_WARNING
        ):
            health["warning"] = (
                f"the numpy store holds {chunks} chunks (> {NUMPY_CHUNKS_WARNING}): "
                f"brute-force search cost grows linearly with the corpus -- "
                f"consider retrieval.store: lancedb or sqlite-vec (each behind "
                f"its extra)"
            )
        return health

    def _last_consolidation(self) -> str | None:
        """The ``generated`` timestamp of the latest maintenance report (§9), or ``None``."""
        report_path = self.paths.index_dir / CONSOLIDATION_REPORT_FILE
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        generated = data.get("generated") if isinstance(data, dict) else None
        return generated if isinstance(generated, str) else None

    def _type_of(self, note: Note) -> str | None:
        """The note's memory type from the folder it lives in, or ``None``."""
        path = note.path
        if path is None:
            return None
        for memory_type in MemoryType:
            folder = self.paths.folder_for(memory_type)
            if path == folder or path.is_relative_to(folder):
                return memory_type.value
        return None

    def _scope_of(self, note: Note) -> str:
        """Frontmatter ``scope:``, defaulting to the manifest default when absent."""
        value = note.get("scope")
        scope = str(value).strip() if value not in (None, "") else ""
        return scope or self.manifest.scopes.default


# ══════════════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════════════
def build_server(
    service: MemoryService,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    name: str = SERVER_NAME,
) -> FastMCP:
    """Wire ``service`` behind a :class:`~mcp.server.fastmcp.FastMCP` (§7).

    Registers the six tools with the descriptions that carry the §8.1 routing and
    §8.2 gate policy to the client agent. ``host``/``port`` are fixed on the
    server here because FastMCP reads them at :meth:`~mcp.server.fastmcp.FastMCP.run`;
    the transport is chosen there. Every tool translates a kernel error into a
    structured ``{"error", "rule", "hint"}`` object -- an agent never receives a
    stack trace.

    Args:
        service: The :class:`MemoryService` over the target vault.
        host: Bind address; loopback by default (R7.4).
        port: Bind port; runtime config (§4).
        name: The server name advertised to clients.

    Returns:
        A configured server, ready for :meth:`~mcp.server.fastmcp.FastMCP.run`.
    """
    server: FastMCP = FastMCP(name=name, instructions=_INSTRUCTIONS, host=host, port=port)

    @server.tool(name="memory_search", description=_SEARCH_DESCRIPTION)
    def memory_search(  # type: ignore[no-untyped-def]
        query: str,
        k: int = 8,
        filters: dict[str, Any] | None = None,
        expand_links: int | None = None,
    ) -> dict[str, Any]:
        return _guard(
            lambda: service.memory_search(query, k=k, filters=filters, expand_links=expand_links)
        )

    @server.tool(name="memory_get", description=_GET_DESCRIPTION)
    def memory_get(id_or_wikilink: str) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        return _guard(lambda: service.memory_get(id_or_wikilink))

    @server.tool(name="memory_status", description=_STATUS_DESCRIPTION)
    def memory_status() -> dict[str, Any]:  # type: ignore[no-untyped-def]
        return _guard(service.memory_status)

    @server.tool(name="memory_write", description=_WRITE_DESCRIPTION)
    def memory_write(  # type: ignore[no-untyped-def]
        type: str,  # noqa: A002 -- the §7.2 tool parameter name
        content: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        return _guard(lambda: service.memory_write(type, content, meta))

    @server.tool(name="memory_propose", description=_PROPOSE_DESCRIPTION)
    def memory_propose(  # type: ignore[no-untyped-def]
        target_path: str,
        proposed_change: str,
        rationale: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        return _guard(
            lambda: service.memory_propose(target_path, proposed_change, rationale, meta)
        )

    @server.tool(name="memory_link", description=_LINK_DESCRIPTION)
    def memory_link(  # type: ignore[no-untyped-def]
        a: str,
        b: str,
        relation: str = DEFAULT_RELATION,
        agent: str | None = None,
    ) -> dict[str, Any]:
        return _guard(lambda: service.memory_link(a, b, relation=relation, agent=agent))

    return server


def serve(
    *,
    vault: Path | str | None = None,
    host: str | None = None,
    port: int | None = None,
    transport: str = "sse",
    env: Mapping[str, str] | None = None,
) -> None:
    """Discover the vault and run the MCP daemon (R7.4). Blocks until stopped.

    The `mechabrain serve` entry point. Defaults to HTTP/SSE on ``127.0.0.1`` with
    the runtime port, which is the single-writer-per-machine shape R7.4 mandates.
    ``transport="stdio"`` is the per-session escape hatch and is accompanied by a
    loud warning: several stdio clients are several kernel processes writing one
    index, which corrupts it.

    Args:
        vault: Explicit vault root (``--vault``); otherwise discovered (R4.3).
        host: Bind address; defaults to ``$MECHABRAIN_HOST`` then
            :data:`DEFAULT_HOST`.
        port: Bind port; defaults to ``$MECHABRAIN_PORT`` then
            :data:`DEFAULT_PORT`. Runtime only -- never the manifest (§4).
        transport: One of :data:`TRANSPORTS`.
        env: Environment to read for discovery and the host/port defaults;
            defaults to the process environment.

    Raises:
        VaultNotFoundError: no vault resolved (R4.3).
        ValueError: ``transport`` is not one of :data:`TRANSPORTS`, or the port
            override is not an integer.
    """
    if transport not in TRANSPORTS:
        raise ValueError(
            f"unknown transport {transport!r}; valid transports are: {', '.join(TRANSPORTS)}"
        )
    paths = discover_vault(vault, env=env).require_initialized()
    manifest = load_manifest(paths.config_file)
    _warn_on_stale_docs(paths, manifest)
    resolved_host = resolve_host(host, env)
    resolved_port = resolve_port(port, env)

    if transport == "stdio":
        logger.warning(
            "starting in stdio mode: one kernel process per client. R7.4 requires "
            "a SINGLE writer per machine -- running several stdio clients at once "
            "means several writers on one index, which corrupts it. Prefer the "
            "default HTTP/SSE daemon and point every MCP client at it."
        )

    service = MemoryService(paths, manifest)
    server = build_server(service, host=resolved_host, port=resolved_port)
    try:
        if transport == "stdio":
            server.run(transport="stdio")
        else:
            logger.info("mechabrain serving on %s://%s:%s", transport, resolved_host, resolved_port)
            server.run(transport=transport)  # type: ignore[arg-type]
    finally:
        service.close()


def _warn_on_stale_docs(paths: VaultPaths, manifest: Manifest) -> None:
    """Log a boot-time warning when `AGENTS.md`/`schema.md` lag the manifest (§10).

    The daemon serves the manifest's rules whatever the docs say; what drifts is
    what the *agents read*. A warning -- never a refusal: a stale doc must not
    take the memory offline, and `mechabrain check` reports the same finding for
    CI to catch.
    """
    from .generate import derived_docs_status  # noqa: PLC0415 -- serve-only concern

    status = derived_docs_status(paths, manifest)
    if status.any_stale:
        logger.warning(
            "%s do not match config.yaml -- agents are reading stale rules; "
            "run `mechabrain sync`",
            " and ".join(status.stale_names),
        )
    if status.agents_ambiguous:
        logger.warning(
            "AGENTS.md has ambiguous managed-block markers -- `mechabrain sync` "
            "cannot regenerate it; keep exactly one begin/end pair"
        )


def resolve_host(host: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The bind host: explicit argument, then ``$MECHABRAIN_HOST``, then the default."""
    if host is not None and host.strip():
        return host.strip()
    environ = _environ(env)
    from_env = environ.get(HOST_ENV_VAR)
    if from_env and from_env.strip():
        return from_env.strip()
    return DEFAULT_HOST


def resolve_port(port: int | None = None, env: Mapping[str, str] | None = None) -> int:
    """The bind port: explicit argument, then ``$MECHABRAIN_PORT``, then the default.

    Raises:
        ValueError: the environment override is set but not an integer, or the
            resolved port is outside ``1..65535``. A silent fallback would hide a
            typo behind an unexpected port (R5.1).
    """
    resolved = port
    if resolved is None:
        raw = _environ(env).get(PORT_ENV_VAR)
        if raw and raw.strip():
            try:
                resolved = int(raw.strip())
            except ValueError:
                raise ValueError(
                    f"${PORT_ENV_VAR} is not an integer: {raw!r}"
                ) from None
    if resolved is None:
        return DEFAULT_PORT
    if not 1 <= resolved <= 65535:
        raise ValueError(f"port must be in 1..65535, got {resolved}")
    return resolved


def _environ(env: Mapping[str, str] | None) -> Mapping[str, str]:
    if env is not None:
        return env
    import os

    return os.environ


# ══════════════════════════════════════════════════════════════════════
# Error translation
# ══════════════════════════════════════════════════════════════════════
def _guard(call):  # type: ignore[no-untyped-def]
    """Run ``call``; turn a kernel error into a structured response, not a raise.

    A gate rejection already comes back as a normal ``dict`` from the service, so
    only genuine errors reach here -- a bad filter, an unknown agent, a missing
    note. The agent gets ``{"error", "rule", "hint"}`` it can act on instead of an
    MCP transport error carrying a stack trace. A non-kernel exception is a real
    bug and is left to propagate.
    """
    try:
        return call()
    except MechabrainError as exc:
        return {"error": exc.message, "rule": exc.rule, "hint": exc.hint}
    except ValueError as exc:
        return {"error": str(exc), "rule": None, "hint": None}


def _issue(issue: GateIssue) -> dict[str, Any]:
    """A gate warning/rejection as JSON (§8.2)."""
    return {
        "check": issue.check,
        "rule": issue.rule,
        "message": issue.message,
        "hint": issue.hint,
    }


def _near_duplicate(duplicate: NearDuplicate) -> dict[str, Any]:
    """A same-scope near-duplicate as JSON (§8.2 item 2)."""
    return {
        "id": duplicate.id,
        "wikilink": duplicate.wikilink,
        "similarity": round(duplicate.similarity, 4),
        "title": duplicate.title,
        "path": duplicate.path,
    }


# ══════════════════════════════════════════════════════════════════════
# Small helpers
# ══════════════════════════════════════════════════════════════════════
def _strip_wikilink(text: str) -> str:
    """Reduce a wikilink to its target id; leave a bare id or path untouched.

    Handles every form the authored graph itself understands: ``[[id]]``,
    ``[[id|alias]]`` and ``[[id#heading]]`` (and their combination -- the
    target is what precedes ``#`` and ``|``).
    """
    stripped = text.strip()
    if stripped.startswith("[[") and stripped.endswith("]]"):
        inner = stripped[2:-2]
        return inner.split("|", 1)[0].split("#", 1)[0].strip()
    return stripped


def _status_of(note: Note) -> str:
    """Frontmatter ``status:``, defaulting to active when absent (§6)."""
    value = note.get("status")
    return str(value).strip() if value not in (None, "") else STATUS_ACTIVE


def _jsonable(value: Any) -> Any:
    """Render a frontmatter value as JSON-native, coercing dates to ISO strings."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


# ══════════════════════════════════════════════════════════════════════
# Tool descriptions -- the prompt each client agent reads before it calls
# ══════════════════════════════════════════════════════════════════════
_INSTRUCTIONS: Final[str] = (
    "Shared agentic memory over a Markdown vault. Read with memory_search and "
    "memory_get; write only through memory_write (governed by the write gate) and "
    "memory_propose. Every memory names its author and its scope. Cite what you "
    "retrieve by its wikilink -- provenance is mandatory. Filter memory_search by "
    "the scope of the work in hand: a hit from another scope is context, not local "
    "truth."
)

_SEARCH_DESCRIPTION: Final[str] = (
    "Hybrid search over the shared memory (vector + BM25, fused with the manifest "
    "weights), with Contextual Retrieval and authored-graph link expansion.\n\n"
    "Arguments:\n"
    "- query: natural-language query.\n"
    "- k: max hits (default 8).\n"
    "- filters: any of {type, agent, profile, scope, tags, status, min_confidence}. "
    "STRONG RECOMMENDATION: pass scope=<the scope of your current task>. A hit from "
    "another scope is context, not local truth (R6.5). Omitting status hides "
    "archived and deprecated notes; pass status explicitly to include them.\n"
    "- expand_links: hops to walk the authored graph (wikilinks, supersedes, "
    "memory_link edges) out from the top hits. Defaults to the manifest.\n\n"
    "Each hit carries its provenance -- path and wikilink -- so you can cite the "
    "source in your answer. A hit reached by expansion carries a 'via' chain "
    "([[seed]] -> [[reached]]); treat it as related context, weaker than a direct "
    "hit. Returns {hits: [...], count}."
)

_GET_DESCRIPTION: Final[str] = (
    "Fetch one memory in full -- frontmatter and body -- by its id or [[wikilink]]. "
    "Use it to read a note that memory_search surfaced before you rely on or cite "
    "it. Returns {id, path, wikilink, title, type, frontmatter, body, tags}, or a "
    "structured error if no note carries that id."
)

_STATUS_DESCRIPTION: Final[str] = (
    "Report the health of the memory store: total notes and counts by type, scope "
    "and status; the vector index (store backend, embedding, chunk count); the "
    "authored graph (nodes, edges, broken links); and the timestamp of the last "
    "consolidation run. Read-only. Use it to sanity-check the store before a "
    "session, or to confirm a write landed."
)

_WRITE_DESCRIPTION: Final[str] = (
    "Write a memory into the shared store, through the write gate. First decide "
    "WHERE it goes (routing, stop at the first match):\n"
    "1. Config, secret or machine state -> do NOT write it here.\n"
    "2. An agent's own behaviour/user-model -> its private store, not here.\n"
    "3. A record of what happened this session -> type='episodic' (a diary; written "
    "directly, no gate).\n"
    "4. A tested, reusable procedure/how-to -> type='procedural' (full gate).\n"
    "5. A long research report -> type='research' (if enabled).\n"
    "6. A reusable, citable fact/insight -> type='semantic' (full gate).\n"
    "7. A change to a human note outside the sandbox -> use memory_propose, never "
    "this.\n\n"
    "Always declare meta.scope: the project slug the memory belongs to, or 'global'. "
    "When unsure between a project and global, choose the project -- promotion to "
    "global is a consolidation decision, not a write decision.\n\n"
    "The gate (semantic and procedural) checks, mechanically: not a near-duplicate "
    "in the same scope (if it is, you must decide: set meta.supersedes to replace "
    "one, set meta.merge=true after folding the detail in, or drop the write); "
    "meta.source is declared; confidence:high only with verification; scope is valid; "
    "procedural requires meta.evidence (a run that succeeded); no denied keys/tags. "
    "It also returns warnings you own: is this reusable beyond the session, and is it "
    "one atomic insight (split it if not).\n\n"
    "meta keys: title, agent (required, must be registered), scope (required), "
    "source, confidence (low|medium|high), optional profile, tags, supersedes, and "
    "the gate-only evidence and merge.\n\n"
    "Returns {rejected:false, path, id, wikilink, warnings, superseded, ...} on "
    "success, or {rejected:true, reason, near_duplicates, warnings} when the gate "
    "refuses -- act on near_duplicates and re-submit with a decision."
)

_PROPOSE_DESCRIPTION: Final[str] = (
    "Propose a change to a note OUTSIDE the memory sandbox (a human note). This is "
    "the ONLY sanctioned way to affect such a note -- never edit it directly. Writes "
    "a proposal note (rationale + suggested change + a link back to the target) into "
    "the inbox for a human to review and apply; the target file is never touched.\n\n"
    "Arguments: target_path (the note it is about), proposed_change (the edit/diff/"
    "new text), rationale (why), and meta with agent (required, must be registered) "
    "and optional profile/scope/source/confidence. Returns {path, id, wikilink, "
    "target}."
)

_LINK_DESCRIPTION: Final[str] = (
    "Record an authored relation between two memories. This edge is a source of "
    "truth and directly feeds memory_search's link expansion: the better you link "
    "related notes, the better the multi-hop recall -- the graph improves by your "
    "curation, not by extraction. Prefer a wikilink in the note body when you author "
    "the note; use this for a relation between two existing notes.\n\n"
    "Arguments: a and b (note id, wikilink or path), relation (what you mean, e.g. "
    "'related', 'supports', 'refines'; default 'related'), and optional agent (your "
    "own registered id, for provenance). Returns the recorded edge."
)
