# 04 — Sessions & state (the `sessions` domain)

Operate the real ADK **session service** (runtime, async). Maps to the `sessions_*` tools. Unlike the
authoring domains, these instantiate a real ADK service (chosen by `runtime.json`) and call it.

## Backends — `sessions_service_set`
```
sessions_service_set(path, app_name, kind, db_url=None, project=None, location=None)
```
- `kind` ∈ {`in_memory`, `database`, `vertex`}. Validates **shape only** (does not instantiate);
  persists to `runtime.json`, preserving any memory/artifacts backends already set.
- **`in_memory`** (default) — state lives in the server process; a process-singleton cache means
  state survives across tool calls. Lost on restart. Great for dev/test.
- **`database`** — requires `db_url`. **Must be an async-driver URL.** For SQLite use
  `sqlite+aiosqlite:///<abs-path>` — a plain `sqlite:///` **fails** (pysqlite is sync). Needs the `db`
  extra (SQLAlchemy). `aiosqlite` ships with google-adk. Proven to persist across separate instances.
- **`vertex`** — requires `project` + `location` (Vertex-managed sessions).

> `db_url` credentials are **redacted** in tool responses (e.g. `postgresql+asyncpg://***@host/db`).

## State mutation mechanism (the key concept)

`session.state` is **read-only between events**. You mutate it by appending an `Event` carrying an
`EventActions(state_delta={...})`. The toolkit does this for you in `sessions_state_set`; you rarely
construct events by hand. The real ADK mechanism:
```python
ev = Event(author="user", actions=EventActions(state_delta={prefixed_key: value}))
await service.append_event(session, ev)   # state now reflects the delta and is persisted
```

## State scopes & prefixes (`app:` / `user:` / `temp:`)

`sessions_state_set(... scope=...)` prefixes the key per the real `State.*_PREFIX` constants:

| `scope` | Prefix | Survives a fresh `get_session`? | Use for |
|---|---|---|---|
| `session` (default) | `""` | ✅ yes | per-conversation state |
| `app` | `app:` | ✅ yes | app-wide config shared across users |
| `user` | `user:` | ✅ yes | per-user state across that user's sessions |
| `temp` | `temp:` | ❌ **NO** | scratch values for the current turn only |

> **⚠️ `temp:` is NOT persisted (by design).** `sessions_state_set(scope="temp")` returns the value
> in its response (it reads back the just-mutated session object), but a **later**
> `sessions_state_get(scope="temp")` won't find it — `temp` lives only for the current invocation.
> This is ADK semantics, not a bug. Use `temp` for ephemeral scratch; use `session`/`user`/`app` to persist.

## The `sessions` domain tools (all async, return the envelope)

| Tool | Key args | Notes |
|---|---|---|
| `sessions_service_set` | `kind, db_url?, project?, location?` | Choose/persist the backend (above). |
| `sessions_create` | `user_id, state=None, session_id=None` | Create a session; returns `{session_id, app_name, user_id, event_count, state}`. |
| `sessions_get` | `user_id, session_id` | Returns the session payload. Missing → clean `err`. |
| `sessions_list` | `user_id` | Returns `session_ids` for `(app, user)`. |
| `sessions_delete` | `user_id, session_id` | Delete (idempotent). |
| `sessions_state_set` | `user_id, session_id, key, value, scope="session"` | Set state via append_event with the scope prefix. Returns the mutated state. |
| `sessions_state_get` | `user_id, session_id, key, scope="session"` | Read a (prefixed) key; `{found, value}`. Remember `temp` won't be found later. |
| `sessions_append_event` | `user_id, session_id, author, text=None, state_delta=None` | Append a raw `Event`. `state_delta` is applied **as-is** (keys must already be prefixed — use `state_state_set` for auto-prefixing). |

## Typical flow

1. `sessions_service_set(path, app_name, kind="in_memory")` (or `database`/`vertex`).
2. `sessions_create(path, app_name, user_id="u1")` → get a `session_id`.
3. `sessions_state_set(... key="lang", value="fr", scope="user")` — persists per-user.
4. `run_agent(... user_id="u1", session_id=...)` (the `run` domain) — runs against this session.
5. `sessions_get` / `sessions_state_get` to inspect.

## Notes

- All session-service methods are async and keyword-only except `append_event(session, event)`
  (positional). The toolkit handles this; you just call the MCP tools.
- A session-service backend is separate from memory/artifacts backends — set each independently
  (`memory_service_set`, `artifacts_service_set`). See `05-memory-artifacts.md`.
- `run_*` tools default to `in_memory` sessions even without `sessions_service_set` (a missing
  `runtime.json` yields the default config), so you can run an agent before configuring sessions.
