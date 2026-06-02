"""`sessions` domain: operates ADK's runtime SESSIONS service (P2a).

Unlike the P1 domains (which *write* code into ``agent.py``), this domain **instantiates a real
ADK session service** and calls it asynchronously. The concrete service
(``InMemorySessionService`` / ``DatabaseSessionService`` / ``VertexAiSessionService``) is chosen
by the backend persisted in ``<app_dir>/.adk_toolkit/runtime.json`` and provided by the singleton
factory :mod:`adk_toolkit_mcp.runtime` (the ``in_memory`` instance is shared across tool calls,
so the state survives within the process).

A FastMCP sub-server mounted under ``namespace="sessions"`` → tools exposed as ``sessions_<name>``.
Functions with **BARE** names (``create``, ``get``, ``delete``, …). ``list`` and ``set`` are
Python builtins: the functions are called ``list_sessions_tool`` / ``state_set`` but are
registered under the bare tool names ``list`` / ``state_set``.

STATE mechanism (cf. ``docs/adk-api-notes/sessions.md``): ``session.state`` is read-only between
events; we mutate via ``append_event(Event(actions=EventActions(state_delta=…)))``. The
app/user/temp scopes are prefixed via ``State.APP_PREFIX`` (``app:``) / ``USER_PREFIX``
(``user:``) / ``TEMP_PREFIX`` (``temp:``). WARNING: ``temp:`` state is NOT persisted by
``get_session`` (ADK semantics); ``state_set`` therefore returns the state read on the object it
just mutated (where ``temp`` is visible), whereas a later ``state_get`` on ``temp`` will not find
it.

Each tool returns the ``{ok, data, error}`` envelope; invalid inputs and sessions not found
return ``err(...)`` (never an exception that propagates).
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

if TYPE_CHECKING:  # pragma: no cover - hints only
    from google.adk.sessions import BaseSessionService, Session

sessions_server: FastMCP = FastMCP("sessions")

#: Exposed state scopes and their mapping to the ADK key prefix.
Scope = Literal["session", "app", "user", "temp"]
_SCOPES: frozenset[str] = frozenset({"session", "app", "user", "temp"})


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _redact_db_url(url: str) -> str:
    """Mask the credentials in a database URL for the MCP logs/responses.

    Parses the URL with ``urllib.parse.urlsplit``; if a ``userinfo`` (user[:pass]@) is present,
    replaces it with ``***``. The scheme, host, port and path (database name) are kept as-is.
    URLs without credentials (e.g. SQLite) are returned intact.

    Examples ::

        >>> _redact_db_url("postgresql+asyncpg://user:s3cret@host:5432/db")
        'postgresql+asyncpg://***@host:5432/db'
        >>> _redact_db_url("sqlite+aiosqlite:///path/to.db")
        'sqlite+aiosqlite:///path/to.db'
    """
    parsed = urlsplit(url)
    if not parsed.username:
        # No credentials → URL unchanged (SQLite, relative URLs, etc.)
        return url
    # Rebuild netloc, replacing userinfo with ***
    host_part = parsed.hostname or ""
    if parsed.port:
        host_part = f"{host_part}:{parsed.port}"
    redacted_netloc = f"***@{host_part}"
    redacted = urlunsplit(
        (parsed.scheme, redacted_netloc, parsed.path, parsed.query, parsed.fragment)
    )
    return redacted


def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _scope_prefix(scope: str) -> str:
    """Return the key prefix for a scope (empty string for ``session``).

    Imports ``State`` lazily to keep the prefix anchored on the REAL ADK constant
    (``State.APP_PREFIX`` etc.) rather than a hardcoded literal.
    """
    from google.adk.sessions import State

    return {
        "session": "",
        "app": State.APP_PREFIX,
        "user": State.USER_PREFIX,
        "temp": State.TEMP_PREFIX,
    }[scope]


def _service_for(path: str, app_name: str) -> BaseSessionService | dict[str, Any]:
    """Load the persisted backend and return the (cached) service or an ``err(...)``.

    Converts a corrupt config (``ValueError``) or an invalid backend into ``err``.
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
    """Serialize a ``Session`` into an envelope payload (id, event count, state)."""
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
    """Append an event carrying ``state_delta`` and return the mutated session object.

    The passed ``session`` object is updated in place by ADK: we return it as-is so the caller
    reads the post-delta state (useful for ``temp`` which does not survive a refetch).
    """
    from google.adk.events import Event, EventActions

    event = Event(author=author, actions=EventActions(state_delta=state_delta))
    await service.append_event(session, event)
    return session


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@sessions_server.tool(tags={"sessions"})
def service_set(
    path: str,
    app_name: str,
    kind: str,
    db_url: str | None = None,
    project: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Choose and persist the app's session service backend (``runtime.json``).

    ``kind`` ∈ {``in_memory``, ``database``, ``vertex``}.
    - ``database`` requires ``db_url`` (async driver required for SQLite:
      ``sqlite+aiosqlite:///path.db``; a plain ``sqlite:///`` will fail on the ADK side).
    - ``vertex`` requires ``project`` and ``location``.

    Does NOT instantiate the service (shape validation only); returns the persisted config.
    """
    if kind not in SESSION_KINDS:
        return err(f"Invalid kind: {kind!r}. Expected one of: {', '.join(sorted(SESSION_KINDS))}.")
    if kind == "database" and not (db_url and db_url.strip()):
        return err("kind='database' requires 'db_url' (e.g. 'sqlite+aiosqlite:///s.db').")
    if kind == "vertex" and not ((project and project.strip()) and (location and location.strip())):
        return err("kind='vertex' requires 'project' and 'location'.")

    ws = _app_ws(path, app_name)
    backend = SessionBackend(
        kind=kind,  # type: ignore[arg-type]  # validated above against SESSION_KINDS
        db_url=db_url,
        project=project,
        location=location,
    )
    # Preserve the memory/artifacts slots already written (P2b compatibility).
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


@sessions_server.tool(tags={"sessions"})
async def create(
    path: str,
    app_name: str,
    user_id: str,
    state: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create a session via the configured service. Returns the id and the initial state.

    ``state``: optional initial state (the ``app:``/``user:`` prefixes are honored by ADK there).
    ``session_id``: optional explicit id (otherwise generated).
    """
    if not user_id.strip():
        return err("user_id is empty.")

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


@sessions_server.tool(tags={"sessions"})
async def get(path: str, app_name: str, user_id: str, session_id: str) -> dict[str, Any]:
    """Return a session: id, event count, full state (dict)."""
    if not session_id.strip():
        return err("session_id is empty.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session not found: {session_id!r} (app={app_name}, user={user_id}).")
    return ok(_session_payload(session))


@sessions_server.tool(tags={"sessions"}, name="list")
async def list_sessions_tool(path: str, app_name: str, user_id: str) -> dict[str, Any]:
    """List the session ids for ``(app_name, user_id)``.

    Named ``list_sessions_tool`` in Python (``list`` is a builtin) but registered under the bare
    tool name ``list`` → exposed as ``sessions_list`` on the client side.
    """
    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    response = await service.list_sessions(app_name=app_name, user_id=user_id)
    session_ids = [s.id for s in response.sessions]
    return ok({"app_name": app_name, "user_id": user_id, "session_ids": session_ids})


@sessions_server.tool(tags={"sessions"})
async def delete(path: str, app_name: str, user_id: str, session_id: str) -> dict[str, Any]:
    """Delete a session. Returns the deleted id (idempotent on the service side)."""
    if not session_id.strip():
        return err("session_id is empty.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    await service.delete_session(app_name=app_name, user_id=user_id, session_id=session_id)
    return ok({"deleted": session_id, "app_name": app_name, "user_id": user_id})


@sessions_server.tool(tags={"sessions"})
async def state_set(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    key: str,
    value: Any,
    scope: str = "session",
) -> dict[str, Any]:
    """Set a state key in the given scope and PERSIST it via ``append_event``.

    ``scope`` ∈ {``session``, ``app``, ``user``, ``temp``} → the key is prefixed by
    ``""``/``app:``/``user:``/``temp:`` (the ``State.*_PREFIX`` constants). The write goes through
    ``append_event(EventActions(state_delta={<prefixed key>: value}))`` (the real ADK mechanism).

    Returns the resulting state read on the **mutated** session (so a ``temp`` value appears
    there, even if a later ``state_get`` will not find it — ``temp`` state is not persisted by
    ADK).
    """
    if scope not in _SCOPES:
        return err(f"Invalid scope: {scope!r}. Expected one of: {', '.join(sorted(_SCOPES))}.")
    if not key.strip():
        return err("key is empty.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session not found: {session_id!r} (app={app_name}, user={user_id}).")

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


@sessions_server.tool(tags={"sessions"})
async def state_get(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    key: str,
    scope: str = "session",
) -> dict[str, Any]:
    """Read a state key (prefixed per ``scope``) from ``session.state``.

    ``found`` indicates whether the prefixed key is present; ``value`` is ``None`` if absent.
    Reminder: a ``temp`` key set during a previous call will NOT be found here (``temp`` state is
    not persisted across invocations).
    """
    if scope not in _SCOPES:
        return err(f"Invalid scope: {scope!r}. Expected one of: {', '.join(sorted(_SCOPES))}.")
    if not key.strip():
        return err("key is empty.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session not found: {session_id!r} (app={app_name}, user={user_id}).")

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


@sessions_server.tool(tags={"sessions"})
async def append_event(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    author: str,
    text: str | None = None,
    state_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a real ``Event`` to the session and return the new event count.

    Builds ``Event(author=..., content=<optional text>, actions=EventActions(
    state_delta=<optional delta>))``. ``state_delta`` is applied AS-IS (the keys must already be
    prefixed if targeting app/user/temp — use ``state_set`` for automatic scope mapping).
    """
    if not author.strip():
        return err("author is empty.")

    service = _service_for(path, app_name)
    if isinstance(service, dict):
        return service

    session = await service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    if session is None:
        return err(f"Session not found: {session_id!r} (app={app_name}, user={user_id}).")

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
