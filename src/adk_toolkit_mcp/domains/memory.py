"""Domaine `memory` : opère le service de MÉMOIRE runtime d'ADK (P2b).

Comme `sessions` (et contrairement aux domaines P1 qui *écrivent* du code dans ``agent.py``),
ce domaine **instancie un vrai service de mémoire ADK** et l'appelle de façon asynchrone. Le
service concret (``InMemoryMemoryService`` / ``VertexAiRagMemoryService`` /
``VertexAiMemoryBankService``) est choisi par le backend persisté dans
``<app_dir>/.adk_toolkit/runtime.json`` et fourni par la fabrique singleton
:mod:`adk_toolkit_mcp.runtime` (l'instance ``in_memory`` est partagée entre appels d'outils,
donc l'état mémoire survit dans le process).

Sous-serveur FastMCP monté sous ``namespace="memory"`` → outils exposés ``memory_<nom>``.
Fonctions à noms **BARE** (``service_set``, ``add_session``, ``search``).

Rappel ADK (cf. ``docs/adk-api-notes/memory-artifacts.md``) :
- ``add_session_to_memory(session)`` ingère une session (les événements PORTANT du texte) ;
- ``search_memory(*, app_name, user_id, query) -> SearchMemoryResponse`` renvoie des
  ``MemoryEntry`` (``content``/``author``/``timestamp``) ; on les sérialise en dicts simples.
- ``InMemoryMemoryService`` fait un rappel par MOTS-CLÉS (pas sémantique) : seuls les
  événements avec ``content.parts`` textuels sont indexés.

Chaque outil renvoie l'enveloppe ``{ok, data, error}`` ; entrées invalides, config corrompue
et session introuvable renvoient ``err(...)`` (jamais d'exception qui remonte).
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

if TYPE_CHECKING:  # pragma: no cover - hints seulement
    from google.adk.memory import BaseMemoryService
    from google.adk.memory.memory_entry import MemoryEntry

memory_server: FastMCP = FastMCP("memory")


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _config_for(path: str, app_name: str) -> RuntimeConfig | dict[str, Any]:
    """Charge la config runtime de l'app, ou renvoie un ``err(...)`` si corrompue."""
    ws = _app_ws(path, app_name)
    try:
        return load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _memory_service_for(path: str, app_name: str) -> BaseMemoryService | dict[str, Any]:
    """Renvoie le service de mémoire (caché) configuré pour l'app, ou un ``err(...)``.

    ``err`` si la config est corrompue, si aucun backend mémoire n'a été choisi
    (``memory_service_set`` non appelé), ou si le backend est invalide (champ requis manquant /
    extra ``gcp`` absent).
    """
    config = _config_for(path, app_name)
    if isinstance(config, dict):
        return config
    if config.memory is None:
        return err(
            "Aucun service de mémoire configuré pour cette app. Appelle d'abord memory_service_set."
        )
    try:
        return get_memory_service(config.memory)
    except ValueError as exc:
        return err(str(exc))


def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
    """Sérialise un ``MemoryEntry`` en dict simple (texte concaténé + author + timestamp).

    ``content`` est aplati via ``model_dump(exclude_none=True)`` (forme
    ``{"parts": [{"text": …}], "role": …}``) ; ``text`` agrège les parts textuelles pour un
    accès direct côté appelant.
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
# Outils MCP
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
    """Choisit et persiste le backend du service de mémoire de l'app (``runtime.json``).

    ``kind`` ∈ {``in_memory``, ``vertex_rag``, ``vertex_memory_bank``}.
    - ``vertex_rag`` exige ``rag_corpus`` (nom de corpus RAG complet) ; extra ``gcp``.
    - ``vertex_memory_bank`` exige ``project``, ``location`` et ``agent_engine_id`` ; extra ``gcp``.

    N'instancie PAS le service (validation de forme seulement) ; préserve les backends session
    et artifacts déjà écrits. Renvoie la config mémoire persistée.
    """
    if kind not in MEMORY_KINDS:
        return err(
            f"kind invalide : {kind!r}. Attendu l'un de : {', '.join(sorted(MEMORY_KINDS))}."
        )
    if kind == "vertex_rag" and not (rag_corpus and rag_corpus.strip()):
        return err("kind='vertex_rag' nécessite 'rag_corpus' (nom de corpus RAG complet).")
    if kind == "vertex_memory_bank" and not (
        (project and project.strip())
        and (location and location.strip())
        and (agent_engine_id and agent_engine_id.strip())
    ):
        return err(
            "kind='vertex_memory_bank' nécessite 'project', 'location' et 'agent_engine_id'."
        )

    ws = _app_ws(path, app_name)
    backend = MemoryBackend(
        kind=kind,  # type: ignore[arg-type]  # validé ci-dessus contre MEMORY_KINDS
        project=project,
        location=location,
        rag_corpus=rag_corpus,
        agent_engine_id=agent_engine_id,
    )
    # Préserve les backends session/artifacts déjà persistés.
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
    """Ingère une session existante dans la mémoire (``add_session_to_memory``).

    Charge la session via le service de SESSIONS configuré (même ``runtime.json``), puis
    l'ajoute au service de MÉMOIRE. Seuls les événements porteurs de texte seront rappelables
    par ``search`` (sémantique ADK). Renvoie l'id de session et son nombre d'événements.
    """
    if not session_id.strip():
        return err("session_id est vide.")

    config = _config_for(path, app_name)
    if isinstance(config, dict):
        return config
    if config.memory is None:
        return err(
            "Aucun service de mémoire configuré pour cette app. Appelle d'abord memory_service_set."
        )

    try:
        session_service = get_session_service(config.session)
        memory_service = get_memory_service(config.memory)
    except ValueError as exc:
        return err(str(exc))

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if session is None:
        return err(f"Session introuvable : {session_id!r} (app={app_name}, user={user_id}).")

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
    """Cherche dans la mémoire et renvoie les souvenirs correspondants (sérialisés).

    Appelle ``search_memory(app_name=, user_id=, query=)`` et aplatit la
    ``SearchMemoryResponse`` en une liste de dicts ``{author, timestamp, text, content}``.
    ``InMemoryMemoryService`` fait un rappel par mots-clés (un mot de la requête doit figurer
    dans le texte d'un événement ingéré).
    """
    if not query.strip():
        return err("query est vide.")

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
