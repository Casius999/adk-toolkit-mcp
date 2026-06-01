"""Domaine `sessions` : opère le service de SESSIONS runtime d'ADK (P2a).

Contrairement aux domaines P1 (qui *écrivent* du code dans ``agent.py``), ce domaine
**instancie un vrai service de session ADK** et l'appelle de façon asynchrone. Le service
concret (``InMemorySessionService`` / ``DatabaseSessionService`` / ``VertexAiSessionService``)
est choisi par le backend persisté dans ``<app_dir>/.adk_toolkit/runtime.json`` et fourni par
la fabrique singleton :mod:`adk_toolkit_mcp.runtime` (l'instance ``in_memory`` est partagée
entre appels d'outils, donc l'état survit dans le process).

Sous-serveur FastMCP monté sous ``namespace="sessions"`` → outils exposés ``sessions_<nom>``.
Fonctions à noms **BARE** (``create``, ``get``, ``delete``, …). ``list`` et ``set`` sont des
builtins Python : les fonctions s'appellent ``list_sessions_tool`` / ``state_set`` mais sont
enregistrées sous les noms d'outils bare ``list`` / ``state_set``.

Mécanisme d'ÉTAT (cf. ``docs/adk-api-notes/sessions.md``) : ``session.state`` est en lecture
seule entre événements ; on mute via ``append_event(Event(actions=EventActions(state_delta=…)))``.
Les scopes app/user/temp sont préfixés via ``State.APP_PREFIX`` (``app:``) / ``USER_PREFIX``
(``user:``) / ``TEMP_PREFIX`` (``temp:``). ATTENTION : l'état ``temp:`` n'est PAS persisté par
``get_session`` (sémantique ADK) ; ``state_set`` renvoie donc l'état lu sur l'objet qu'il
vient de muter (où ``temp`` est visible), tandis qu'un ``state_get`` ultérieur sur ``temp`` ne
le retrouvera pas.

Chaque outil renvoie l'enveloppe ``{ok, data, error}`` ; les entrées invalides et les sessions
introuvables renvoient ``err(...)`` (jamais d'exception qui remonte).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

from fastmcp import FastMCP

from ..envelope import err, ok
from ..runtime import (
    SESSION_KINDS,
    RuntimeConfig,
    SessionBackend,
    get_session_service,
    load_runtime_config,
    save_runtime_config,
)
from ..workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints seulement
    from google.adk.sessions import BaseSessionService, Session

sessions_server: FastMCP = FastMCP("sessions")

#: Scopes d'état exposés et leur correspondance vers le préfixe de clé ADK.
Scope = Literal["session", "app", "user", "temp"]
_SCOPES: frozenset[str] = frozenset({"session", "app", "user", "temp"})


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _redact_db_url(url: str) -> str:
    """Masque les credentials dans une URL de base de données pour les logs/réponses MCP.

    Parse l'URL avec ``urllib.parse.urlsplit`` ; si un ``userinfo`` (user[:pass]@) est présent,
    le remplace par ``***``. Le schéma, l'hôte, le port et le chemin (nom de la base) sont
    conservés tels quels. Les URLs sans credentials (ex. SQLite) sont renvoyées intactes.

    Exemples ::

        >>> _redact_db_url("postgresql+asyncpg://user:s3cret@host:5432/db")
        'postgresql+asyncpg://***@host:5432/db'
        >>> _redact_db_url("sqlite+aiosqlite:///path/to.db")
        'sqlite+aiosqlite:///path/to.db'
    """
    parsed = urlsplit(url)
    if not parsed.username:
        # Pas de credentials → URL inchangée (SQLite, URLs relatives, etc.)
        return url
    # Reconstruit netloc en remplaçant userinfo par ***
    host_part = parsed.hostname or ""
    if parsed.port:
        host_part = f"{host_part}:{parsed.port}"
    redacted_netloc = f"***@{host_part}"
    redacted = urlunsplit(
        (parsed.scheme, redacted_netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return redacted


def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _scope_prefix(scope: str) -> str:
    """Renvoie le préfixe de clé pour un scope (chaîne vide pour ``session``).

    Importe ``State`` paresseusement pour garder le préfixe ancré sur la VRAIE constante
    ADK (``State.APP_PREFIX`` etc.) plutôt que sur un littéral codé en dur.
    """
    from google.adk.sessions import State

    return {
        "session": "",
        "app": State.APP_PREFIX,
        "user": State.USER_PREFIX,
        "temp": State.TEMP_PREFIX,
    }[scope]


def _service_for(path: str, app_name: str) -> BaseSessionService | dict[str, Any]:
    """Charge le backend persisté et renvoie le service (caché) ou un ``err(...)``.

    Convertit une config corrompue (``ValueError``) ou un backend invalide en ``err``.
    """
    ws = _app_ws(path, app_name)
    try:
        config = load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))
    try:
        return get_session_service(config.session)
    except ValueError as exc:
        return err(str(exc))


def _session_payload(session: Session) -> dict[str, Any]:
    """Sérialise un ``Session`` en payload d'enveloppe (id, compteur d'événements, état)."""
    return {
        "session_id": session.id,
        "app_name": session.app_name,
        "user_id": session.user_id,
        "event_count": len(session.events),
        "state": dict(session.state),
    }


async def _append_state_delta(
    service: BaseSessionService,
    session: Session,
    state_delta: dict[str, Any],
    author: str,
) -> Session:
    """Ajoute un événement portant ``state_delta`` et renvoie l'objet session muté.

    L'objet ``session`` passé est mis à jour en place par ADK : on le renvoie tel quel afin
    que l'appelant lise l'état post-delta (utile pour ``temp`` qui ne survit pas à un refetch).
    """
    from google.adk.events import Event, EventActions

    event = Event(author=author, actions=EventActions(state_delta=state_delta))
    await service.append_event(session, event)
    return session


# --------------------------------------------------------------------------- #
# Outils MCP
# --------------------------------------------------------------------------- #
@sessions_server.tool
def service_set(
    path: str,
    app_name: str,
    kind: str,
    db_url: str | None = None,
    project: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Choisit et persiste le backend du service de sessions de l'app (``runtime.json``).

    ``kind`` ∈ {``in_memory``, ``database``, ``vertex``}.
    - ``database`` exige ``db_url`` (pilote async requis pour SQLite :
      ``sqlite+aiosqlite:///chemin.db`` ; un ``sqlite:///`` simple échouera côté ADK).
    - ``vertex`` exige ``project`` et ``location``.

    N'instancie PAS le service (validation de forme seulement) ; renvoie la config persistée.
    """
    if kind not in SESSION_KINDS:
        return err(
            f"kind invalide : {kind!r}. Attendu l'un de : {', '.join(sorted(SESSION_KINDS))}."
        )
    if kind == "database" and not (db_url and db_url.strip()):
        return err("kind='database' nécessite 'db_url' (ex. 'sqlite+aiosqlite:///s.db').")
    if kind == "vertex" and not ((project and project.strip()) and (location and location.strip())):
        return err("kind='vertex' nécessite 'project' et 'location'.")

    ws = _app_ws(path, app_name)
    backend = SessionBackend(
        kind=kind,  # type: ignore[arg-type]  # validé ci-dessus contre SESSION_KINDS
        db_url=db_url,
        project=project,
        location=location,
    )
    # Préserve les emplacements memory/artifacts déjà écrits (compat P2b).
    try:
        existing = load_runtime_config(ws, app_name)
    except ValueError:
        existing = RuntimeConfig()
    config = RuntimeConfig(session=backend, memory=existing.memory, artifacts=existing.artifacts)
    changed = save_runtime_config(ws, config)

    return ok(
        {
            "app_name": app_name,
            "kind": backend.kind,
            "db_url": _redact_db_url(backend.db_url) if backend.db_url else backend.db_url,
            "project": backend.project,
            "location": backend.location,
            "config_path": str(ws.path(".adk_toolkit/runtime.json")),
            "changed": changed,
        }
    )


@sessions_server.tool
async def create(
    path: str,
    app_name: str,
    user_id: str,
    state: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Crée une session via le service configuré. Renvoie l'id et l'état initial.

    ``state`` : état initial optionnel (les préfixes ``app:``/``user:`` y sont respectés par
    ADK). ``session_id`` : id explicite optionnel (sinon généré).
    """
    if not user_id.strip():
        return err("user_id est vide.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )
    return ok(_session_payload(session))


@sessions_server.tool
async def get(path: str, app_name: str, user_id: str, session_id: str) -> dict[str, Any]:
    """Renvoie une session : id, compteur d'événements, état complet (dict)."""
    if not session_id.strip():
        return err("session_id est vide.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session introuvable : {session_id!r} (app={app_name}, user={user_id}).")
    return ok(_session_payload(session))


@sessions_server.tool(name="list")
async def list_sessions_tool(path: str, app_name: str, user_id: str) -> dict[str, Any]:
    """Liste les ids de session pour ``(app_name, user_id)``.

    Nommée ``list_sessions_tool`` en Python (``list`` est un builtin) mais enregistrée sous le
    nom d'outil bare ``list`` → exposée ``sessions_list`` côté client.
    """
    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    response = await service.list_sessions(app_name=app_name, user_id=user_id)
    session_ids = [s.id for s in response.sessions]
    return ok({"app_name": app_name, "user_id": user_id, "session_ids": session_ids})


@sessions_server.tool
async def delete(path: str, app_name: str, user_id: str, session_id: str) -> dict[str, Any]:
    """Supprime une session. Renvoie l'id supprimé (idempotent côté service)."""
    if not session_id.strip():
        return err("session_id est vide.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    await service.delete_session(app_name=app_name, user_id=user_id, session_id=session_id)
    return ok({"deleted": session_id, "app_name": app_name, "user_id": user_id})


@sessions_server.tool
async def state_set(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    key: str,
    value: Any,
    scope: str = "session",
) -> dict[str, Any]:
    """Définit une clé d'état dans le scope donné et PERSISTE via ``append_event``.

    ``scope`` ∈ {``session``, ``app``, ``user``, ``temp``} → la clé est préfixée par
    ``""``/``app:``/``user:``/``temp:`` (constantes ``State.*_PREFIX``). L'écriture passe par
    ``append_event(EventActions(state_delta={<clé préfixée>: value}))`` (mécanisme ADK réel).

    Renvoie l'état résultant lu sur la session **mutée** (donc une valeur ``temp`` y figure,
    même si un ``state_get`` ultérieur ne la retrouvera pas — l'état ``temp`` n'est pas
    persisté par ADK).
    """
    if scope not in _SCOPES:
        return err(f"scope invalide : {scope!r}. Attendu l'un de : {', '.join(sorted(_SCOPES))}.")
    if not key.strip():
        return err("key est vide.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session introuvable : {session_id!r} (app={app_name}, user={user_id}).")

    prefixed_key = _scope_prefix(scope) + key
    session = await _append_state_delta(service, session, {prefixed_key: value}, author="user")

    return ok(
        {
            "session_id": session.id,
            "scope": scope,
            "key": key,
            "stored_key": prefixed_key,
            "event_count": len(session.events),
            "state": dict(session.state),
        }
    )


@sessions_server.tool
async def state_get(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    key: str,
    scope: str = "session",
) -> dict[str, Any]:
    """Lit une clé d'état (préfixée selon ``scope``) depuis ``session.state``.

    ``found`` indique si la clé préfixée est présente ; ``value`` vaut ``None`` si absente.
    Rappel : une clé ``temp`` posée lors d'un appel précédent ne sera PAS retrouvée ici
    (l'état ``temp`` n'est pas persisté entre invocations).
    """
    if scope not in _SCOPES:
        return err(f"scope invalide : {scope!r}. Attendu l'un de : {', '.join(sorted(_SCOPES))}.")
    if not key.strip():
        return err("key est vide.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session introuvable : {session_id!r} (app={app_name}, user={user_id}).")

    prefixed_key = _scope_prefix(scope) + key
    state = dict(session.state)
    found = prefixed_key in state
    return ok(
        {
            "session_id": session.id,
            "scope": scope,
            "key": key,
            "stored_key": prefixed_key,
            "found": found,
            "value": state.get(prefixed_key),
        }
    )


@sessions_server.tool
async def append_event(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    author: str,
    text: str | None = None,
    state_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ajoute un vrai ``Event`` à la session et renvoie le nouveau compteur d'événements.

    Construit ``Event(author=..., content=<texte optionnel>, actions=EventActions(
    state_delta=<delta optionnel>))``. ``state_delta`` est appliqué TEL QUEL (les clés doivent
    déjà être préfixées si l'on cible app/user/temp — utiliser ``state_set`` pour le mapping
    automatique des scopes).
    """
    if not author.strip():
        return err("author est vide.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session introuvable : {session_id!r} (app={app_name}, user={user_id}).")

    from google.adk.events import Event, EventActions

    content = None
    if text is not None:
        from google.genai import types

        content = types.Content(role=author, parts=[types.Part(text=text)])

    event = Event(
        author=author,
        content=content,
        actions=EventActions(state_delta=state_delta or {}),
    )
    await service.append_event(session, event)

    return ok(
        {
            "session_id": session.id,
            "event_count": len(session.events),
            "state": dict(session.state),
        }
    )
