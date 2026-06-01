# adk-toolkit-mcp — Build Progress

> Durable progress tracker for the multi-phase build (subagent-driven). Resume from here.
> Spec: `../docs/superpowers/specs/2026-05-31-adk-toolkit-mcp-design.md` (in the parent "Claude code" dir, NOT in this repo).
> Plan: `../docs/superpowers/plans/2026-05-31-adk-toolkit-mcp-00-foundation.md`.
> **Hard constraint:** 100% standalone. No link to the sibling `sin/` project. Local-only git, no remote until P6 (user confirms push).

## Conventions (authoritative: `docs/adk-api-notes/conventions.md`)
- Each domain = `FastMCP("<domain>")` sub-server exporting `<domain>_server`; tool functions **bare-named**; mounted via `mcp.mount(<domain>_server, namespace="<domain>")` (NOT deprecated `prefix=`). Exposed name = `<domain>_<bare>`.
- Every tool returns `{ok,data,error}` (`adk_toolkit_mcp.envelope`). Validate inputs; `err(...)` never raises/swallows.
- Code-first authoring: sidecar `.adk_toolkit/agents.json` is source of truth; `agent.py` fully regenerated. Generated code must pass `ast.parse`, `ruff format --check`, AND `ruff check --select I` (isort).
- Optional deps imported lazily / codegen-only; never hardcode secrets (use `os.getenv`).

## Status

| Phase | Domains | State |
|---|---|---|
| **P0 Foundation** | repo, packaging, envelope, deps, workspace, versions, resources, server, CI, Docker | ✅ DONE |
| **P1 Author** | project, agents, tools (3a+3b), models | ✅ DONE |
| **P2 State** | sessions ✅ (P2a), memory, artifacts (P2b ⏳) | 🟡 IN PROGRESS |
| **P3 Runtime/Eval** | run, eval | ⬜ |
| **P4 Ops** | deploy, a2a, observability, safety, mcp_bridge, dev | ⬜ |
| **P5 Skill** | `adk-toolkit` skill (SKILL.md + 14 refs) | ⬜ |
| **P6 Finish** | Code Mode, prompts, docs, repo publish (confirm GitHub) | ⬜ |

## Exposed tools so far (39)
- `project_*`: create, inspect, set_env, add_extra, agent_config
- `agents_*`: create_llm, create_sequential, create_parallel, create_loop, create_custom, compose, as_tool, set_root, list, get
- `tools_*`: add_function, add_long_running, add_builtin, add_agent_tool, add_openapi, add_bigquery, add_spanner, add_mcp_toolset, add_apihub, add_langchain, add_crewai, set_auth, list
- `models_*`: set, configure_litellm, generate_config
- `sessions_*`: service_set, create, get, list, delete, state_set, state_get, append_event
- Resources: `adk://version`, `adk://models`

## P2a runtime/sessions facts (see `docs/adk-api-notes/sessions.md`)
- Shared `runtime.py`: `SessionBackend`(kind in_memory/database/vertex) + `RuntimeConfig`
  (reserved memory/artifacts slots for P2b) persisted at `.adk_toolkit/runtime.json`.
  `get_session_service(backend)` is a process-singleton cache → same in_memory backend
  returns the SAME instance (state survives across tool calls); database keyed by db_url.
- ADK session services are ALL async: `create_session`/`get_session`/`list_sessions`/
  `delete_session` (keyword-only) and `append_event(session, event)` (positional).
- STATE MUTATION: `session.state` is read-only between events; mutate via
  `append_event(Event(actions=EventActions(state_delta={prefixed_key: value})))`. Event/
  EventActions construct with snake_case fields; id/timestamp auto-populate.
- State prefixes (real `State.*_PREFIX`): app=`app:`, user=`user:`, temp=`temp:`, session=``.
  ⚠️ `temp:` is NOT persisted across `get_session` (ADK design): `state_set` reads back the
  mutated session (temp visible in its return); a later `state_get` on temp finds nothing.
- `DatabaseSessionService` needs SQLAlchemy (NOT in core — added `db` extra + `dev`) AND an
  async driver URL: use `sqlite+aiosqlite:///path` (plain `sqlite:///` fails: pysqlite is
  sync). aiosqlite is already a google-adk dep. Cross-instance SQLite persistence proven.

## Key ADK 2.1.0 facts learned (see `docs/adk-api-notes/`)
- `google-adk` 2.1.0, `fastmcp` 3.3.1, Python 3.12 local (CI matrix 3.11/3.12).
- `FastMCP.mount(prefix=)` is DEPRECATED → use `namespace=`.
- Agent types import from `google.adk.agents`; `Agent is LlmAgent`. **SequentialAgent/ParallelAgent/LoopAgent are DEPRECATED in 2.1.0** ("use Workflow") but still functional — kept + backlogged (Workflow Runtime API UNCERTAIN).
- Plain function in `tools=[...]` resolves to `FunctionTool` via `canonical_tools()`. Builtins are pre-instantiated objects. `request_input` does NOT exist in 2.1.0.
- `OpenAPIToolset` goes directly in `tools=[...]`. `McpToolset` from `google.adk.tools.mcp_tool` (+ Stdio/SSE/StreamableHTTP connection params, `mcp.StdioServerParameters`).
- `AuthScheme` is a `Union` (not constructible); render `auth_credential=AuthCredential(...)` only. bigquery/spanner use `credentials_config`, not auth kwargs.
- `LiteLlm` from `google.adk.models.lite_llm`. `GenerateContentConfig`/`SafetySetting`/`HarmCategory`/`HarmBlockThreshold` from `google.genai.types`.
- langchain/crewai re-exported under `google.adk.integrations.*` (the `google.adk.tools.*` paths warn-deprecate).

## Test/quality state
318 passed, 1 skipped (litellm probe), coverage ~96%. ruff + mypy clean; full suite green
under `-W error::DeprecationWarning`. `runtime.py` 79 stmts (<500 lines), `sessions.py` <800.
Added `sqlalchemy>=2.0` to `dev` + new user-facing `db` extra (uv.lock gained sqlalchemy +
greenlet only). Functional DB persistence proven (SQLite file, cross-call read-back).

## Resume instructions
Next: P2b — `memory` and `artifacts` domains. Extend `RuntimeConfig` (memory/artifacts slots
are already reserved as opaque dicts) with `MemoryBackend`/`ArtifactBackend` dataclasses +
`get_memory_service`/`get_artifact_service` singletons in `runtime.py` (mirror the session
pattern). Introspect `google.adk.memory` (InMemoryMemoryService, VertexAiRagMemoryService?)
and `google.adk.artifacts` (InMemoryArtifactService, GcsArtifactService) — confirm async API
and required extras before building. Reuse `_service_for`-style helpers and the `{ok,data,error}`
envelope. Then P3 (run/eval) wires Runner over these services ("hybrid execute"). Update this
file after each phase.
