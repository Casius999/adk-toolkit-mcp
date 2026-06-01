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
| **P2 State** | sessions (P2a), memory + artifacts (P2b) | ✅ DONE |
| **P3 Runtime/Eval** | run (P3a) ✅ · eval (P3b) ⬜ **(next)** | 🟡 IN PROGRESS |
| **P4 Ops** | deploy, a2a, observability, safety, mcp_bridge, dev | ⬜ |
| **P5 Skill** | `adk-toolkit` skill (SKILL.md + 14 refs) | ⬜ |
| **P6 Finish** | Code Mode, prompts, docs, repo publish (confirm GitHub) | ⬜ |

## Exposed tools so far (53)
- `project_*`: create, inspect, set_env, add_extra, agent_config
- `agents_*`: create_llm, create_sequential, create_parallel, create_loop, create_custom, compose, as_tool, set_root, list, get
- `tools_*`: add_function, add_long_running, add_builtin, add_agent_tool, add_openapi, add_bigquery, add_spanner, add_mcp_toolset, add_apihub, add_langchain, add_crewai, set_auth, list
- `models_*`: set, configure_litellm, generate_config
- `sessions_*`: service_set, create, get, list, delete, state_set, state_get, append_event
- `memory_*`: service_set, add_session, search
- `artifacts_*`: service_set, save, load, list, delete, versions
- `run_*`: agent, stream, live, config_build, inspect_events
- Resources: `adk://version`, `adk://models`

## P2a runtime/sessions facts (see `docs/adk-api-notes/sessions.md`)
- Shared `runtime.py`: `SessionBackend`(kind in_memory/database/vertex) + `RuntimeConfig`
  (now also `MemoryBackend`/`ArtifactBackend`, see P2b) persisted at `.adk_toolkit/runtime.json`.
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

## P2b runtime/memory/artifacts facts (see `docs/adk-api-notes/memory-artifacts.md`)
- `runtime.py` extended (now 443 lines, still <500): `MemoryBackend`(kind
  in_memory/vertex_rag/vertex_memory_bank; project/location/rag_corpus/agent_engine_id) and
  `ArtifactBackend`(kind in_memory/gcs; bucket) REPLACE the old reserved opaque dicts in
  `RuntimeConfig`. `get_memory_service`/`get_artifact_service` mirror `get_session_service`
  (lazy import, process-singleton cache). `runtime.json` stays backward-compatible with a
  P2a-only file (memory/artifacts serialize to `null` when unset; null/unknown-kind load
  cleanly). `reset_service_cache()` now clears ALL THREE caches.
- `google.adk.memory` + `google.adk.artifacts` use lazy `__getattr__`: concrete classes
  (`InMemory*`, `VertexAi*`, `Gcs*`) are importable by name but NOT in `dir(module)`.
- Memory API async: `add_session_to_memory(session)` (positional) returns None;
  `search_memory(*, app_name, user_id, query) -> SearchMemoryResponse{memories: [MemoryEntry
  {content, author, timestamp, custom_metadata, id}]}`. `InMemoryMemoryService` recall is
  KEYWORD-based (not semantic): only events with `content.parts` text are indexed; a query
  word must literally (case-insensitively) appear. FUNCTIONAL test proves a "Paris" hit + a
  no-match → 0 + a state-only event not recalled.
- Artifact API async + keyword-only: `save_artifact(...) -> int` (0-based version),
  `load_artifact(..., version=None) -> Optional[Part]` (`.text` for text, else
  `.inline_data.{data,mime_type}`; None ⇒ absent), `list_artifact_keys`, `delete_artifact`,
  `list_versions`. `Part.from_text(text=)` / `Part.from_bytes(data=, mime_type=)`. `user:`
  filename prefix = user-scoped. FUNCTIONAL test proves v0→v1, latest+specific round-trip,
  list/versions/delete, user:-prefix, and byte-for-byte base64 binary round-trip.
- Extras: Vertex memory (rag + memory_bank) → `gcp`; GCS artifacts → `gcp`. The gcp ImportError
  is raised INSIDE the service `__init__` (lazy `import vertexai` / `google.cloud.storage`),
  not at class import — `runtime.py` wraps import+construction together and converts to an
  actionable `ValueError` (uv add 'adk-toolkit-mcp[gcp]'). NO `uv.lock`/`pyproject` change
  (in_memory memory/artifacts + genai `types` are all core google-adk).

## P3a runtime/run facts (see `docs/adk-api-notes/runtime-run.md`)
- New `run_core.py` (221 lines, <300) factors the execution core for OFFLINE testing:
  `build_runner(app_name, root_agent, runtime_config)` wires `google.adk.runners.Runner`
  (keyword-only; `session_service` required; memory/artifact services passed only when a
  backend is configured — NOT `InMemoryRunner`, which would bypass `runtime.json` + the
  singleton cache); `collect_events(runner, *, user_id, session_id, new_message_text,
  run_config=None, progress=None)` ensures the session exists (creates if `get_session` is
  None — `Runner.auto_create_session` defaults False), runs `run_async`, collects `Event`s,
  and **awaits a `progress` callback per event** (SSE); `serialize_event` → `{author, text,
  function_calls:[{name,args}], function_responses:[{name,response}], state_delta,
  transfer_to_agent, is_final, partial}`; `build_run_config(streaming_mode, max_llm_calls,
  response_modalities)` validates the mode by NAME against the real `StreamingMode` enum
  (NONE/SSE/BIDI; values None/'sse'/'bidi'); `max_llm_calls=None` keeps ADK default 500.
- `Runner.run_async(*, user_id, session_id, new_message=None, run_config=None, ...)` is an
  **async generator of `Event`**. `new_message` = `types.Content(role="user",
  parts=[types.Part.from_text(text=msg)])`. `Event` accessors confirmed: `get_function_calls()`
  (`.name`/`.args`), `get_function_responses()` (`.name`/`.response`), `.author`, `.content`,
  `.actions.state_delta`/`.transfer_to_agent`, `.is_final_response()`, `.partial`.
- **FakeLlm offline proof:** `BaseLlm.generate_content_async(self, llm_request, stream=False)`
  is an **async generator** (NOT a coroutine) yielding `LlmResponse`. A `FakeLlm(BaseLlm)`
  (pydantic; scripting state as a field) overrides it: final-text case yields one
  `LlmResponse(content=Content(role="model", parts=[Part.from_text(...)]))`; tool-call case
  yields a `Part.from_function_call(name=, args=)` first, then final text. Wiring an
  `LlmAgent(name=, model=FakeLlm(...), tools=[py_fn])` through `build_runner`+`collect_events`
  PROVED the full loop offline (no key): function_call event → function_response event (ADK
  auto-ran the tool) → final-text event. A plain py fn in `tools=[...]` emits a benign
  `UserWarning` (`JSON_SCHEMA_FOR_FUNC_DECL`) — NOT a `DeprecationWarning`, so it passes
  `-W error::DeprecationWarning`.
- `import_root_agent(path, app_name)` loads `<path>/<app_name>/agent.py`'s `root_agent` with a
  UNIQUE module name per call. CRITICAL Windows gotcha: `spec.loader.exec_module` caches
  bytecode by (path, mtime); two writes within one mtime tick serve a STALE version even with
  a fresh module name. Fix: read the source and `compile()`+`exec()` it into the module dict
  (defeats the source/bytecode cache). Errors wrapped in `RootAgentImportError` → tool `err`.
  Reload-after-edit proven.
- `run_live` (BIDI) uses `BaseLlm.connect` (Gemini Live websocket), NOT
  `generate_content_async`; base `connect` raises `NotImplementedError`, only `Gemini`
  overrides it. CANNOT run in CI. Degrades cleanly: detect (a) creds (`GOOGLE_API_KEY`/
  `GEMINI_API_KEY` OR `GOOGLE_GENAI_USE_VERTEXAI=TRUE`+`GOOGLE_CLOUD_PROJECT`) and (b) model
  live-capability (`type(model).connect is not BaseLlm.connect`); returns an actionable `err`
  BEFORE opening any connection (never hangs). Marked experimental.
- `fastmcp.Context` is auto-injected by FastMCP even with a `ctx: Context | None = None`
  annotation, and is NOT exposed as a client input. `run_stream` awaits
  `ctx.report_progress(i, message=...)` + `ctx.info(...)` per event (proven via a Client
  `progress_handler`). `run.py` is the first domain to use `Context`.

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
439 passed, 1 skipped (litellm probe), coverage 96.96%. ruff + mypy clean; full suite green
under `-W error::DeprecationWarning`. P3a files: `run_core.py` 221 lines (100% cov),
`domains/run.py` 354 lines (100% cov); `tests/unit/test_run_core.py` + `test_run.py` +
shared `tests/unit/fake_llm.py` (FakeLlm/ScriptedLlm fixture). NO `uv.lock`/`pyproject` change
in P3a (Runner/RunConfig/BaseLlm/Content are all core google-adk). FUNCTIONAL result: a
FakeLlm-backed `LlmAgent` runs a FULL agent loop OFFLINE (no key) — final text proven, and
tool-call loop proven (function_call → ADK-executed function_response → final text), both via
the core helpers AND via the mounted `run_agent` tool (file-imported agent). `run_stream`
progress relayed to a `fastmcp.Client`; `run_live` returns an actionable `err` without a
key/live-model (no hang).

## Resume instructions
Next: **P3b — `eval` domain** (the remaining half of P3). Introspect `google.adk.evaluation`
(needs the `eval` extra — confirm what's installed; e.g. `AgentEvaluator`, `EvalCase`,
`EvalSet`, response/trajectory evaluators, and the `adk eval` data formats `*.evalset.json` /
`*.test.json`) BEFORE building. Reuse `run_core.build_runner`/`collect_events` to generate
trajectories where useful, the `runtime.py` factories, and the `{ok,data,error}` envelope;
mirror the async tool style. Mount under `namespace="eval"` (exposed `eval_*`). The eval extra
may be heavy/absent in CI — degrade with an actionable `err` (like the `gcp`/`db` extras) and
keep any heavy import lazy. Record API facts in `docs/adk-api-notes/runtime-eval.md` and commit
it. Update this file after the phase.
