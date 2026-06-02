"""`memory` domain: operates ADK's runtime MEMORY service (P2b).

Like `sessions` (and unlike the P1 domains that *write* code into ``agent.py``), this domain
**instantiates a real ADK memory service** and calls it asynchronously. The concrete service
(``InMemoryMemoryService`` / ``VertexAiRagMemoryService`` / ``VertexAiMemoryBankService``) is
chosen by the backend persisted in ``<app_dir>/.adk_toolkit/runtime.json`` and provided by the
singleton factory :mod:`adk_toolkit_mcp.runtime` (the ``in_memory`` instance is shared across
tool calls, so the memory state survives within the process).

A FastMCP sub-server mounted under ``namespace="memory"`` → tools exposed as ``memory_<name>``.
Functions with **BARE** names (``service_set``, ``add_session``, ``search``).

ADK reminder (cf. ``docs/adk-api-notes/memory-artifacts.md``):
- ``add_session_to_memory(session)`` ingests a session (the events CARRYING text);
- ``search_memory(*, app_name, user_id, query) -> SearchMemoryResponse`` returns
  ``MemoryEntry`` objects (``content``/``author``/``timestamp``); we serialize them into simple
  dicts.
- ``InMemoryMemoryService`` does a KEYWORD recall (not semantic): only events with textual
  ``content.parts`` are indexed.

Each tool returns the ``{ok, data, error}`` envelope; invalid inputs, corrupt config and a
session not found return ``err(...)`` (never an exception that propagates).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..runtime import (
    MEMORY_KINDS,
    MemoryBackend,
    RuntimeConfig,
    get_memory_service,
    get_session_service,
    load_runtime_config,
    save_runtime_config,
)
from ..workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints only
    from google.adk.memory import BaseMemoryService
    from google.adk.memory.memory_entry import MemoryEntry

memory_server: FastMCP = FastMCP("memory")


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _config_for(path: str, app_name: str) -> RuntimeConfig | dict[str, Any]:
    """Load the app's runtime config, or return an ``err(...)`` if corrupt."""
    ws = _app_ws(path, app_name)
    try:
        return load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _memory_service_for(path: str, app_name: str) -> BaseMemoryService | dict[str, Any]:
    """Return the (cached) memory service configured for the app, or an ``err(...)``.

    ``err`` if the config is corrupt, if no memory backend has been chosen
    (``memory_service_set`` not called), or if the backend is invalid (missing required field /
    missing ``gcp`` extra).
    """
    config = _config_for(path, app_name)
    if isinstance(config, dict):
        return config
    if config.memory is None:
        return err("No memory service configured for this app. Call memory_service_set first.")
    try:
        return get_memory_service(config.memory)
    except ValueError as exc:
        return err(str(exc))


def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize a ``MemoryEntry`` into a simple dict (concatenated text + author + timestamp).

    ``content`` is flattened via ``model_dump(exclude_none=True)`` (form
    ``{"parts": [{"text": …}], "role": …}``); ``text`` aggregates the textual parts for direct
    access on the caller side.
    """
    content = entry.content
    parts = list(content.parts or []) if content is not None else []
    text = "".join(p.text for p in parts if p.text)
    return {
        "author": entry.author,
        "timestamp": entry.timestamp,
        "text": text,
        "content": content.model_dump(exclude_none=True) if content is not None else None,
    }


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@memory_server.tool(tags={"memory"})
def service_set(
    path: str,
    app_name: str,
    kind: str,
    project: str | None = None,
    location: str | None = None,
    rag_corpus: str | None = None,
    agent_engine_id: str | None = None,
) -> dict[str, Any]:
    """Choose and persist the app's memory service backend (``runtime.json``).

    ``kind`` ∈ {``in_memory``, ``vertex_rag``, ``vertex_memory_bank``}.
    - ``vertex_rag`` requires ``rag_corpus`` (full RAG corpus name); ``gcp`` extra.
    - ``vertex_memory_bank`` requires ``project``, ``location`` and ``agent_engine_id``; ``gcp``
      extra.

    Does NOT instantiate the service (shape validation only); preserves the session and artifacts
    backends already written. Returns the persisted memory config.
    """
    if kind not in MEMORY_KINDS:
        return err(f"Invalid kind: {kind!r}. Expected one of: {', '.join(sorted(MEMORY_KINDS))}.")
    if kind == "vertex_rag" and not (rag_corpus and rag_corpus.strip()):
        return err("kind='vertex_rag' requires 'rag_corpus' (full RAG corpus name).")
    if kind == "vertex_memory_bank" and not (
        (project and project.strip())
        and (location and location.strip())
        and (agent_engine_id and agent_engine_id.strip())
    ):
        return err(
            "kind='vertex_memory_bank' requires 'project', 'location' and 'agent_engine_id'."
        )

    ws = _app_ws(path, app_name)
    backend = MemoryBackend(
        kind=kind,  # type: ignore[arg-type]  # validated above against MEMORY_KINDS
        project=project,
        location=location,
        rag_corpus=rag_corpus,
        agent_engine_id=agent_engine_id,
    )
    # Preserve the session/artifacts backends already persisted.
    try:
        existing = load_runtime_config(ws, app_name)
    except ValueError:
        existing = RuntimeConfig()
    config = RuntimeConfig(session=existing.session, memory=backend, artifacts=existing.artifacts)
    changed = save_runtime_config(ws, config)

    return ok(
        {
            "app_name": app_name,
            "kind": backend.kind,
            "project": backend.project,
            "location": backend.location,
            "rag_corpus": backend.rag_corpus,
            "agent_engine_id": backend.agent_engine_id,
            "config_path": str(ws.path(".adk_toolkit/runtime.json")),
            "changed": changed,
        }
    )


@memory_server.tool(tags={"memory"})
async def add_session(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Ingest an existing session into memory (``add_session_to_memory``).

    Loads the session via the configured SESSIONS service (same ``runtime.json``), then adds it to
    the MEMORY service. Only the text-carrying events will be recallable by ``search`` (ADK
    semantics). Returns the session id and its event count.
    """
    if not session_id.strip():
        return err("session_id is empty.")

    config = _config_for(path, app_name)
    if isinstance(config, dict):
        return config
    if config.memory is None:
        return err("No memory service configured for this app. Call memory_service_set first.")

    try:
        session_service = get_session_service(config.session)
        memory_service = get_memory_service(config.memory)
    except ValueError as exc:
        return err(str(exc))

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if session is None:
        return err(f"Session not found: {session_id!r} (app={app_name}, user={user_id}).")

    await memory_service.add_session_to_memory(session)
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session.id,
            "event_count": len(session.events),
        }
    )


@memory_server.tool(tags={"memory"})
async def search(path: str, app_name: str, user_id: str, query: str) -> dict[str, Any]:
    """Search memory and return the matching memories (serialized).

    Calls ``search_memory(app_name=, user_id=, query=)`` and flattens the
    ``SearchMemoryResponse`` into a list of ``{author, timestamp, text, content}`` dicts.
    ``InMemoryMemoryService`` does a keyword recall (a word from the query must appear in an
    ingested event's text).
    """
    if not query.strip():
        return err("query is empty.")

    service = _memory_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    response = await service.search_memory(app_name=app_name, user_id=user_id, query=query)
    memories = [_entry_to_dict(entry) for entry in response.memories]
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "query": query,
            "count": len(memories),
            "memories": memories,
        }
    )
