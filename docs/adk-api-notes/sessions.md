# ADK API notes — `sessions` (P2 runtime services)

Captured 2026-06-01 by introspection. `google-adk` **2.1.0**, Python 3.12.

These notes back the shared `runtime.py` factory and the `sessions` domain sub-server.
P2 domains differ from P1: they instantiate **real ADK service objects** and call them
(all async), rather than authoring `agent.py` source.

## Imports

```python
from google.adk.sessions import (
    BaseSessionService,
    InMemorySessionService,
    DatabaseSessionService,   # requires SQLAlchemy (the `db` extra) — see below
    Session,
    State,
)
from google.adk.events import Event, EventActions
```

`VertexAiSessionService` also exists (`from google.adk.sessions import VertexAiSessionService`)
and is reserved for a future `vertex` backend; not exercised here.

## Session service API is fully ASYNC

Every operation on a session service is a coroutine (`inspect.iscoroutinefunction`
returns `True` for all of them). Confirmed signatures (keyword-only except
`append_event`):

```text
async create_session(*, app_name: str, user_id: str,
                      state: Optional[dict[str, Any]] = None,
                      session_id: Optional[str] = None) -> Session
async get_session(*, app_name: str, user_id: str, session_id: str,
                  config: Optional[GetSessionConfig] = None) -> Optional[Session]
async list_sessions(*, app_name: str, user_id: Optional[str] = None) -> ListSessionsResponse
async delete_session(*, app_name: str, user_id: str, session_id: str) -> None
async append_event(session: Session, event: Event) -> Event
```

Notes:
- `get_session` returns `None` (not an exception) for a missing session → tools must
  treat `None` as a clean `err(...)`.
- `list_sessions` returns a `ListSessionsResponse` whose `.sessions` is a list of
  `Session` objects; `user_id` is technically optional but the `sessions` domain always
  passes it.
- `append_event` is the ONLY positional-args method: `append_event(session, event)`.

## Session object attributes

A `Session` exposes: `id`, `app_name`, `user_id`, `state` (a dict-like mapping),
`events` (a list of `Event`), and `last_update_time`. Event count = `len(session.events)`.

## STATE-MUTATION MECHANISM (important)

`session.state` is **read-only between events**. The supported way to mutate/persist
state is to append an `Event` carrying an `EventActions(state_delta={...})`:

```python
ev = Event(author="user", actions=EventActions(state_delta={"key": "value"}))
await service.append_event(session, ev)
# session.state now reflects the delta (and it is persisted by the service)
```

`Event` and `EventActions` are Pydantic models. Although `inspect.signature` shows
camelCase aliases (`stateDelta`, …), they are constructed with **snake_case** field
names (`state_delta=`, `author=`, `actions=`) thanks to `populate_by_name`. `Event.id`
and `Event.timestamp` auto-populate (default factories) — no need to pass them.

## State scopes and prefix constants

Scope → key prefix is governed by `State.*_PREFIX` (confirmed exact strings):

| Scope     | Constant            | Prefix value |
|-----------|---------------------|--------------|
| session   | (none)              | `""`         |
| app       | `State.APP_PREFIX`  | `app:`       |
| user      | `State.USER_PREFIX` | `user:`      |
| temp      | `State.TEMP_PREFIX` | `temp:`      |

To set app/user/temp state, prefix the key (e.g. `app:my_key`) inside the `state_delta`.

### ⚠️ `temp:` state is NOT persisted (by design)

Confirmed behavior of `InMemorySessionService` (and the same applies to
`DatabaseSessionService`):

- `app:`, `user:`, and **session-scoped** keys survive a fresh `get_session(...)`.
- `temp:` keys are visible on the **same in-memory `Session` object** immediately after
  `append_event` (the live `session.state` reflects the just-applied delta), but they
  are **discarded** on the next `get_session` — temp state lives only for the current
  invocation/turn.

Design consequence for the `sessions` domain:
- `state_set` reads back the resulting state from the **session object it just mutated**,
  so the return payload correctly shows the value for ALL scopes including `temp`.
- A subsequent independent `state_get` (which does a fresh `get_session`) for `temp`
  scope will NOT find the value. This is ADK semantics, not a bug. Tests assert
  temp via the `state_set` return; session/app/user round-trip through `state_get`.

## `DatabaseSessionService` requires an ASYNC SQLAlchemy driver

`DatabaseSessionService(db_url: str, **kwargs)`.

- It needs the **`sqlalchemy`** package, which is NOT in `google-adk` core deps. ADK
  raises `ImportError("install google-adk[db]")` without it. We add `sqlalchemy>=2.0`
  to both the `dev` optional-deps (so CI persistence tests run) and a user-facing `db`
  extra in `pyproject.toml`.
- ADK builds the engine with **`create_async_engine(db_url)`** on the raw URL and does
  NOT rewrite the scheme. A plain `sqlite:///path` therefore FAILS with
  `InvalidRequestError: The asyncio extension requires an async driver ... 'pysqlite'
  is not async`. Callers MUST use an async driver URL:
  **`sqlite+aiosqlite:///<abs-path>`**.
- `aiosqlite` is already a **direct dependency of `google-adk`** (always installed), so
  no extra package is needed for SQLite beyond `sqlalchemy`. The `db` extra is therefore
  just `sqlalchemy>=2.0`.
- An in-memory SQLite (`sqlite+aiosqlite:///:memory:`) gets a `StaticPool` automatically
  but does not persist across separate service instances; use a file URL for persistence.

### Proven cross-instance persistence (the functional acceptance test)

Two SEPARATE `DatabaseSessionService` instances over the same SQLite file:
instance A `create_session` + `append_event(state_delta=...)`, instance B
`get_session` → state read back intact (`{'foo':'init','sk':'sv','app:ak':'av',
'user:uk':'uv'}`, event count 1). This proves real persistence through the DB service.

## Runtime singleton requirement

`InMemorySessionService` holds all state in process memory, so two tool calls that share
the same in-memory backend MUST receive the **same instance** or state is lost between
calls. `runtime.py` caches service instances by a stable key (kind + url/project/location).
