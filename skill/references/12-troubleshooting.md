# 12 — Troubleshooting (known pitfalls + fixes)

The accumulated, verified gotchas for google-adk **2.1.0** as exposed by adk-toolkit-mcp. Each is a
real fact confirmed by introspection, not a guess. Scan this when something behaves unexpectedly.

## Deprecations (still functional, but warned)

- **Workflow agents are deprecated.** `SequentialAgent`/`ParallelAgent`/`LoopAgent` emit a
  `DeprecationWarning` ("use Workflow instead") in 2.1.0 but **work**. The toolkit keeps emitting them
  (the Workflow successor isn't stable). Fix: ignore the warning, or restructure as LLM agents +
  sub_agents where possible. **Don't** assume they're broken.
- **`Runner(plugins=[...])` is deprecated.** It raises under `-W error::DeprecationWarning`. The toolkit
  wires plugins via the **`App`** path (`Runner(app=App(plugins=[...]))`) — zero warnings. If you
  hand-write a Runner, use `App`, not the `plugins=` kwarg.
- **`langchain_tool`/`crewai_tool` import shims** warn-deprecate (they re-export from
  `google.adk.integrations.*`). Harmless for authoring (generated code runs in your venv).

## Missing / wrong names (the task's guesses that were WRONG)

- **`request_input` does NOT exist** in 2.1.0. For human input use `get_user_choice` (builtin) or a
  `long_running` function tool. Don't reference `request_input`.
- **`RemoteA2aAgent` is NOT in `google.adk.agents`.** The only working import is
  `from google.adk.agents.remote_a2a_agent import RemoteA2aAgent` — which is exactly what `a2a_consume`
  generates. Don't import it from the package root.
- **CLI flags that drift** (verified for 2.1.0):
  - `deploy agent_engine` has **no `--app_name`**; `app_name` → `--display_name`. `--staging_bucket` is
    **deprecated/no-op** (never emitted).
  - cloud_run/gke/agent_engine Cloud Trace flag is **`--trace_to_cloud`**, NOT `--enable_cloud_trace`.
  - gke cluster flag is **`--cluster_name`**, NOT `--cluster`.
  - `adk run` takes the message as a **positional QUERY**, NOT an `--input` flag (which doesn't exist).
  - `web`/`api_server` have **no `--app_name`** — point the positional `AGENTS_DIR`.
  - The toolkit validates every emitted flag against the installed ADK's `--help` and returns a clean
    `err` listing unknowns on drift. If a deploy errors with "unknown flag", you're on a different ADK
    version — re-check the flags.

## Runtime services

- **DatabaseSessionService needs an async driver URL.** Use `sqlite+aiosqlite:///<abs-path>`, NOT plain
  `sqlite:///` (pysqlite is sync → `InvalidRequestError`). Needs the **`db`** extra (SQLAlchemy);
  `aiosqlite` ships with google-adk. In-memory SQLite (`:memory:`) doesn't persist across instances.
- **`temp:` state is not persisted.** `sessions_state_set(scope="temp")` shows the value in its own
  return, but a later `sessions_state_get(scope="temp")` finds nothing. Use `session`/`user`/`app` to
  persist. (See `04-sessions-state.md`.)
- **In-memory memory recall is keyword-based, not semantic.** A query word must literally appear in an
  ingested event's text. Only events with **text content** are indexed (bare `state_delta` events
  aren't). For semantic recall use a Vertex backend. (See `05-memory-artifacts.md`.)
- **Vertex/GCS backends need the `gcp` extra + credentials.** The missing-dependency `ImportError` is
  raised inside the service constructor; the toolkit converts it to an actionable `err`
  (install `adk-toolkit-mcp[gcp]`). Same pattern for `db`, `eval`, `a2a`, `community`.

## Generated code & the sidecar

- **Don't hand-edit `agent.py`.** It is regenerated wholesale from the sidecar
  (`.adk_toolkit/agents.json`) on every authoring tool call — your edits will be lost. To change the
  app, call a tool. The same applies to `plugins.py` (managed by `safety_add_plugin`).
- **Create children before parents.** Workflow agents and `agents_compose` require named sub_agents to
  already exist. A cycle, a self-reference, or assigning one agent two parents → a clean `err` (the
  single-parent rule is enforced by ADK at import time).
- **Only LlmAgents carry tools / models / callbacks.** `tools_add_*`, `models_*`, and `safety_add_callback`
  reject non-LLM agents.
- **`OpenAPIToolset` and all toolsets go directly into `tools=[...]`** — never call `.get_tools()`.

## Backends & creds (Vertex vs AI Studio)

- **AI Studio** (`backend="ai_studio"`): `GOOGLE_GENAI_USE_VERTEXAI=FALSE` + `GOOGLE_API_KEY`. Simplest.
- **Vertex** (`backend="vertex"`): `GOOGLE_GENAI_USE_VERTEXAI=TRUE` + `GOOGLE_CLOUD_PROJECT` +
  `GOOGLE_CLOUD_LOCATION` + `gcloud` auth. Required for Vertex memory/RAG and Agent Engine deploys.
- **Never hardcode keys.** LiteLlm keys flow via `os.getenv("<ENV>")`; put values in `.env`
  (`project_set_env`). Tools never return secret values; `.env` keys and DB URLs are redacted.

## `-W error::DeprecationWarning` (CI strictness)

The toolkit's own code stays warning-clean. Where ADK internals emit a `DeprecationWarning` it can't
avoid (e.g. eval internally building `Runner(plugins=...)`), the toolkit wraps **only that call** in a
narrow `warnings` filter — your code stays strict. A plain function in `tools=[...]` emits a benign
`UserWarning` (`JSON_SCHEMA_FOR_FUNC_DECL`), not a `DeprecationWarning`, so it passes.

## "It returned ok=False" — that's not a crash

`{ok: False, error: "..."}` is a **clean, actionable failure** with a message telling you what to fix
(missing extra, missing arg, missing agent, etc.). The toolkit never raises an exception out of a tool
and never hangs (network paths like `run_live` short-circuit before connecting). Read the `error` string.

## Quick symptom → cause table

| Symptom | Likely cause | Fix |
|---|---|---|
| "unknown flag" on deploy | ADK version drift | Re-check flags in `08-deploy.md`; the tool lists the unknowns |
| DB session errors with "async driver" | plain `sqlite://` URL | use `sqlite+aiosqlite:///<abs-path>` + `db` extra |
| memory search returns nothing | keyword miss or no text events | use words that appear; ingest text events; or Vertex backend |
| `temp` state vanished | `temp:` not persisted by design | use `session`/`user`/`app` scope |
| ImportError about `vertexai`/`google.cloud` | missing `gcp` extra | install `adk-toolkit-mcp[gcp]` |
| eval errors about `rouge_score`/`pandas` | missing `eval` extra | install `adk-toolkit-mcp[eval]` |
| A2A surface errors at runtime | missing `a2a` extra | install `adk-toolkit-mcp[a2a]` |
| edits to `agent.py` disappeared | regenerated from sidecar | call a tool instead of editing |
| `run_live` returns an err immediately | no creds / non-live model | set `GOOGLE_API_KEY` + use a live Gemini model (experimental) |
