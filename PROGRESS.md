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
| **P3 Runtime/Eval** | run (P3a) ✅ · eval (P3b) ✅ | ✅ DONE |
| **P4 Ops** | deploy (P4a) ✅ · dev (P4a) ✅ · a2a (P4b) ✅ · mcp_bridge (P4b) ✅ · safety (P4c) ✅ · observability (P4c) ✅ | ✅ **DONE** |
| **P5 Skill** | `adk-toolkit` skill (SKILL.md + 14 refs) + install + test | ✅ **DONE** |
| **P6a Code Mode + prompts** | domain tags · opt-in FastMCP Code Mode · 5 workflow prompts | ✅ **DONE** |
| **P6 Finish (rest)** | docs (`ARCHITECTURE.md`/`TOOL_CATALOG.md`/`CONTRIBUTING.md`), repo publish (confirm GitHub) | ⬜ **(next)** |

## Exposed tools so far (81)
- `project_*`: create, inspect, set_env, add_extra, agent_config
- `agents_*`: create_llm, create_sequential, create_parallel, create_loop, create_custom, compose, as_tool, set_root, list, get
- `tools_*`: add_function, add_long_running, add_builtin, add_agent_tool, add_openapi, add_bigquery, add_spanner, add_mcp_toolset, add_apihub, add_langchain, add_crewai, set_auth, list
- `models_*`: set, configure_litellm, generate_config
- `sessions_*`: service_set, create, get, list, delete, state_set, state_get, append_event
- `memory_*`: service_set, add_session, search
- `artifacts_*`: service_set, save, load, list, delete, versions
- `run_*`: agent, stream, live, config_build, inspect_events
- `eval_*`: create_set, set_criteria, run, report
- `deploy_*`: preflight, agent_engine, cloud_run, gke, containerize, status
- `dev_*`: web, api_server, run, stop, status, logs
- `a2a_*`: consume, expose, agent_card
- `mcp_bridge_*`: expose_adk_tools, convert_builtin
- `safety_*`: add_callback, add_plugin, settings
- `observability_*`: enable_otel, cloud_trace, third_party, trace_view
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

## P3b eval facts (see `docs/adk-api-notes/eval.md`)
- `domains/eval.py` (≈400 lines): `eval_create_set`, `eval_set_criteria`, `eval_run`,
  `eval_report`. Files live under `<app_dir>/eval/` (`<name>.evalset.json`, `test_config.json`,
  `reports/<id>.json`). All `google.adk.evaluation` imports are LAZY (extra may be absent →
  actionable `err`). Operates on a project `(path, app_name)`.
- **Schema:** `*.evalset.json` is a serialized **`EvalSet`** (`google.adk.evaluation.eval_set`):
  `EvalSet(*eval_set_id, name?, eval_cases: list[EvalCase])`; `EvalCase(*eval_id,
  conversation: list[Invocation])`; `Invocation(*user_content: Content, final_response?: Content,
  intermediate_data?: IntermediateData)`; `IntermediateData(tool_uses: list[FunctionCall],
  ...)`. `create_set` builds these from the real models and serialises
  `model_dump_json(indent=2, exclude_none=True)` — the test asserts
  `EvalSet.model_validate_json(file)` round-trips (schema conformance PROVEN, not guessed). The
  older `*.test.json` format (`query`/`reference`/`expected_tool_use`) is still auto-detected by
  ADK (`_load_eval_set_from_file` tries `EvalSet` first, falls back on `ValidationError`); the
  toolkit emits the NEW schema only.
- **`test_config.json`** = a serialized `EvalConfig` `{"criteria": {"tool_trajectory_avg_score":
  float, "response_match_score": float}}` (flat floats; auto-wrapped to `BaseCriterion`). ADK
  reads it from the SAME folder as the eval file (`find_config_for_test_file`). `set_criteria`
  writes the flat form; thresholds validated to `[0,1]`.
- **`AgentEvaluator.evaluate` / `evaluate_eval_set` are `async` and ASSERT-based** (return None
  on pass; raise `AssertionError` with per-metric detail on fail). `evaluate(path_or_dir)` walks
  a dir for `*.test.json` OR takes a single file path directly. `criteria` (flat dict) is
  DEPRECATED (a `logger.warning`, NOT a `DeprecationWarning`) in favour of `eval_config=`.
  `_get_agent_for_eval(module_name)` uses `importlib.import_module` on a **DOTTED module path**
  (not a file): needs a member `agent` OR a name ending `.agent`, then `root_agent`. `eval_run`
  inserts `path` on `sys.path`, evicts `sys.modules[<app>...]` (pick up edits), and imports
  **`<app_name>.agent`** (a scaffolded app is a package: `__init__.py`+`agent.py`).
- **Metrics OFFLINE:** `tool_trajectory_avg_score` (`TrajectoryEvaluator`, PURE structural
  compare of `tool_uses`, no model/no rouge) and `response_match_score`
  (`final_response_match_v1.RougeEvaluator`, **ROUGE-1**, needs `rouge_score` but **no model**).
  LLM-judge metrics (`response_evaluation_score`, `*_v1/_v2`, safety, hallucinations) need a
  judge model + creds → NOT offline; the toolkit's offline path uses only the first two.
- **Rich report:** to capture per-metric SCORES (the assert API only yields pass/fail), `eval_run`
  reuses ADK's own internals `AgentEvaluator._get_agent_for_eval` + `_get_eval_results_by_eval_id`
  (the core of `evaluate`) → `dict[eval_id -> list[EvalCaseResult]]`; each
  `EvalCaseResult.final_eval_status` (`EvalStatus.PASSED/FAILED/NOT_EVALUATED`) +
  `overall_eval_metric_results[*].(metric_name, score, threshold, eval_status)`. Verdict = all
  cases PASSED. Report persisted to `eval/reports/<ts>-<eval_set_id>.json`; `eval_report` reads
  it by `(path, app_name, report_id)` — a TOOL not a `adk://eval/{id}` resource (a report has
  THREE addressing coords; a FastMCP 3.3.1 template carries only one opaque id).
- **FUNCTIONAL offline result (load-bearing, no API key):** a `FakeLlm`-backed agent whose answer
  == the case's `expected_response` PASSES `response_match_score` (ROUGE score 1.0); a
  `ScriptedLlm`+`add_numbers` agent PASSES `tool_trajectory_avg_score`=1.0 AND
  `response_match_score`=1.0; a deliberately wrong expected answer correctly FAILS (`ok=True,
  passed=False`) — the pipeline genuinely evaluates, no faked pass. An eval *failure* is a NORMAL
  result (`ok=True, passed=False`); real errors (missing eval set, import, model creds, LLM-judge,
  extra absent) → clean `err` (no hang).
- **`-W error::DeprecationWarning` gotcha:** ADK's eval internally builds `Runner(plugins=...)`,
  emitting a `DeprecationWarning` from `google.adk.runners`. Under `-W error` that warning is
  RAISED inside ADK and ABORTS the eval inference (caught there, recorded as "Inference failed").
  No public API avoids the internal call → `eval_run` wraps the ADK call in
  `warnings.catch_warnings()` + a NARROW `filterwarnings("ignore", message="The `plugins`
  argument is deprecated.*", category=DeprecationWarning, module="google.adk.runners")`. Scoped to
  that block only; OUR code stays strict. CLI `-W` would otherwise override an ini `filterwarnings`
  (so the suppression MUST be in-code, not pyproject).
- **Extra `eval` REQUIRED for offline metrics** (`rouge-score`/`pandas`/`tabulate`/`nltk`/
  `scikit-learn`/`jinja2`/`gepa`/`google-cloud-aiplatform[evaluation]`). Added
  `adk-toolkit-mcp[eval]` to the `dev` extra so CI installs it. `uv.lock` already had these
  resolved (user-facing `eval`/`all` locked in P0) → only a 2-line `uv.lock` metadata edge
  (`dev → eval`), no new package versions. Heavy imports stay lazy; `ModuleNotFoundError` for an
  eval dep → actionable `err`. NB: installing `eval` pulls `vertexai`/`google.cloud.storage`
  (via aiplatform) + `litellm`, so 2 gcp-absent tests now SKIP and the litellm probe runs (was
  1 skip → now 2). No regression.

## P4a deploy+dev facts (see `docs/adk-api-notes/deploy-dev.md`)
- New shared `adk_cli.py` (~280 lines, 93% cov): `adk_executable()` (prefers venv
  `Scripts/adk.exe`, else PATH `adk`, else `[sys.executable,"-m","google.adk.cli"]` — VERIFIED
  real module: `google.adk.cli.__main__` exists, `python -m google.adk.cli --version`=2.1.0);
  `run_adk(args,cwd,timeout)` → `{argv,rc,stdout,stderr}` (argv list, **never shell=True**);
  `available_flags(subcommand)` parses `--flag` tokens from real `--help` (CACHED per subcommand)
  so the toolkit can't emit a flag this ADK lacks; a **process registry** (`make_key`,
  `start_process`/`process_status`/`process_logs`/`stop_process`/`stop_all_processes`) using
  `Popen` + a log file. Windows stop = `CREATE_NEW_PROCESS_GROUP` at launch + `taskkill /F /T`
  to kill the TREE. PROVEN with a trivial `python -c "time.sleep(30)"`: starts (running), writes
  log, `stop` genuinely terminates (status not-running after). No orphans left.
- **EXACT 2.1.0 flags** (drifted from the task's guesses — captured the real truth):
  - `deploy agent_engine AGENT`: `--project/--region/--display_name/--requirements_file`.
    Has **NO `--app_name`**; **`--staging_bucket` is DEPRECATED (no-op)**. Toolkit maps the
    `app_name` param → `--display_name`, and only NOTES `staging_bucket` (never emits it).
  - `deploy cloud_run AGENT`: `--project/--region/--service_name/--app_name/--with_ui` +
    **`--trace_to_cloud`** (NOT `--enable_cloud_trace` — toolkit's `enable_cloud_trace` maps to it).
  - `deploy gke AGENT`: `--project/--region/`**`--cluster_name`** (NOT `--cluster`)`/--service_name/
    --app_name/--service_type[ClusterIP|LoadBalancer]`. Toolkit's `cluster` param → `--cluster_name`.
  - `web`/`api_server [AGENTS_DIR]`: positional AGENTS_DIR (dir-of-agents OR single agent folder);
    real `--host`(127.0.0.1)/`--port`(INTEGER, no default). NO `--app_name` (point AGENTS_DIR at the
    folder). api_server also has `--auto_create_session`/`--with_ui`.
  - `run AGENT [QUERY]`: the message is a **positional QUERY** (NOT `--input` — that flag does NOT
    exist); no QUERY ⇒ interactive (would block). `--jsonl`/`--timeout 30s`/`--in_memory` exist.
- **`deploy.py`** (94% cov): builds the EXACT argv (positional AGENT=`<path>/<app_name>` LAST),
  validates required args, and **validates every emitted flag against `available_flags`** → `err`
  listing unknowns if drift. `execute=False` (default) returns `{argv,plan,notes,executed:False}`
  and NEVER calls `run_adk`; `execute=True` runs the real deploy (GCP — NOT exercised in CI).
  `containerize` writes an idempotent `Dockerfile` serving `adk api_server` on `$PORT`. `preflight`
  (gcloud/adk/kubectl on PATH) + `status` (shell to gcloud/kubectl with a 20s timeout, else
  `available:False` guidance) are best-effort and never hang. Real cloud deploy command
  construction proven by exact-token asserts; flag-validity proven against real `--help`.
- **`dev.py`** (95% cov): `web`/`api_server` start a MANAGED bg process via the registry (they
  block serving, so NOT `run_adk`); return `{key,pid,port,url,...}`. `run` is one-shot
  `adk run AGENT "<message>"` via `run_adk` with a bounded timeout (no message ⇒ guidance, since
  interactive blocks); `stop`/`status`/`logs` drive a started process by key. **FUNCTIONAL proof
  (gated `ADK_TOOLKIT_TEST_API_SERVER=1`, ran locally in ~3.4s):** a real `adk api_server` BOOTED
  on an ephemeral port and answered HTTP `GET /docs`, then `stop` terminated it. Ungated, the
  registry lifecycle (start real `adk api_server` → status running → logs → stop → not-running)
  is always tested, plus argv construction + flag-validity.

## P4b a2a+mcp_bridge facts (see `docs/adk-api-notes/a2a-mcp-bridge.md`)
- **Introspection done after a transient `uv pip install "google-adk[a2a]"`** (added only
  `a2a-sdk==0.3.26`), then `uv sync --extra dev` restored the env. `uv.lock`/`pyproject` blob
  hashes UNCHANGED (verified vs baseline `fc5334a7`/`e792cbef`); the `a2a` extra was already
  declared in pyproject from P0 — only INSTALLED transiently, never re-locked.
- **`adk_to_mcp_tool_type(tool: BaseTool) -> mcp.types.Tool`** — `mcp` is **CORE** (fastmcp dep),
  so `mcp_bridge` is fully CI-testable with **NO extra**. Returns an `mcp.types.Tool` (fields:
  name/title/description/inputSchema(dict)/outputSchema/icons/annotations/meta/execution).
  FUNCTIONAL: `google_search` → `{name:'google_search', description:'google_search',
  inputSchema:{}}` (builtin, no params → EMPTY schema); a `FunctionTool(add_numbers)` → real
  JSON-Schema `{properties:{a,b}, required:[a,b], type:'object', title:'add_numbersParams'}`.
- **`expose_adk_tools` robust path**: `agent.tools` holds RAW entries (a plain `def` is a bare
  `function`, NOT a FunctionTool). Use **`await agent.canonical_tools(ctx=None)`** (async →
  `list[BaseTool]`): wraps functions into FunctionTool, normalises to BaseTool — every element then
  converts. Import the project agent via `run_core.import_root_agent`, locate it with
  **`BaseAgent.find_agent(name)`** (recursive; returns self for the root name, None if absent),
  then convert. Workflow agents (Sequential/Parallel/Loop) lack `canonical_tools` → guarded `err`.
- **`convert_builtin(kind)`**: only `CORE_BUILTINS` (no-arg). Some "core" builtins are BaseTool
  instances (google_search/url_context/load_memory/get_user_choice → convert directly); a few are
  bare functions (`exit_loop`/`transfer_to_agent` → wrapped in FunctionTool). `vertex_ai_search`
  (ARG_BUILTIN) is rejected with guidance to use `expose_adk_tools`.
- **a2a needs the `a2a` extra for ALL three surfaces** (each module imports `a2a.*` at top):
  - **`to_a2a(agent, *, host='localhost', port=8000, protocol='http', agent_card=None, ...) ->
    Starlette`** (`@a2a_experimental` → `UserWarning`, NOT Deprecation). ⚠️ Real sig adds
    `host`/`protocol` vs the task's guess. **Routes are registered LAZILY in the Starlette
    lifespan (on startup)** via `A2AStarletteApplication.add_routes_to_app` — `app.routes` is empty
    until uvicorn starts the app (only a LIVE probe sees the route). Well-known card route =
    `a2a.utils.constants.AGENT_CARD_WELL_KNOWN_PATH` = **`/.well-known/agent-card.json`**. A `str`
    `agent_card=` is a FILE PATH (not a URL).
  - **⚠️ `RemoteA2aAgent` is NOT in `google.adk.agents`** in 2.1.0 (not in `__all__`, no lazy
    getattr). The ONLY import is **`from google.adk.agents.remote_a2a_agent import
    RemoteA2aAgent`**. Sig `RemoteA2aAgent(name: str, agent_card: AgentCard|str, *, ...)` —
    `agent_card` is the 2nd positional; a `BaseAgent` subclass → composes as a `sub_agent`.
  - **`AgentCardBuilder(*, agent: BaseAgent, rpc_url=None, ...)`** with **async** `build() ->
    a2a.types.AgentCard`. `to_a2a` uses it internally (`rpc_url=f"{protocol}://{host}:{port}/"`).
- **project_model `remote_a2a`** (new AgentType): renders `<name> = RemoteA2aAgent(name=...,
  agent_card="<url>")` + the submodule import; `_needed_agent_imports` EXCLUDES it (different
  module → `_REMOTE_A2A_IMPORT` appended separately, merged/sorted by `_merge_tool_imports` →
  isort-clean). Topological order treats it like any agent reference (no children, can be a
  sub_agent). `validate_spec` requires a non-empty `agent_card`. ast.parse + ruff format + isort
  all clean (proven).
- **a2a domain** (`domains/a2a.py`, 78% cov — gated execute/build blocks need the extra):
  `consume` adds a `remote_a2a` proxy (codegen-only, no extra) + regenerates; `expose` writes
  `a2a_app.py` (`a2a_app = to_a2a(root_agent, port=PORT)` + `from agent import root_agent`),
  `execute=False` returns file + `uvicorn a2a_app:a2a_app` (cwd = app dir resolves the sibling
  `agent` import), `execute=True` gates on `find_spec('a2a')` + starts a managed uvicorn via the
  adk_cli registry; `agent_card` builds the AgentCard (gated, actionable `err` when absent). The
  real-type subprocess probe, live uvicorn boot (`ADK_TOOLKIT_TEST_A2A=1`), and real AgentCard
  build are GATED on `find_spec('a2a')` and SKIP without the extra (3 of the 6 suite skips).
- **mcp_bridge domain** (`domains/mcp_bridge.py`, 92% cov): FULLY functional in CI (no extra).
  In-process construction of a deprecated `SequentialAgent` (in two tests) emits a real
  `DeprecationWarning` → wrapped in a scoped `warnings.catch_warnings()` ignore filter (mirrors the
  eval domain pattern; OUR code stays strict under `-W error::DeprecationWarning`).

## Key ADK 2.1.0 facts learned (see `docs/adk-api-notes/`)
- `google-adk` 2.1.0, `fastmcp` 3.3.1, Python 3.12 local (CI matrix 3.11/3.12).
- `FastMCP.mount(prefix=)` is DEPRECATED → use `namespace=`.
- Agent types import from `google.adk.agents`; `Agent is LlmAgent`. **SequentialAgent/ParallelAgent/LoopAgent are DEPRECATED in 2.1.0** ("use Workflow") but still functional — kept + backlogged (Workflow Runtime API UNCERTAIN).
- Plain function in `tools=[...]` resolves to `FunctionTool` via `canonical_tools()`. Builtins are pre-instantiated objects. `request_input` does NOT exist in 2.1.0.
- `OpenAPIToolset` goes directly in `tools=[...]`. `McpToolset` from `google.adk.tools.mcp_tool` (+ Stdio/SSE/StreamableHTTP connection params, `mcp.StdioServerParameters`).
- `AuthScheme` is a `Union` (not constructible); render `auth_credential=AuthCredential(...)` only. bigquery/spanner use `credentials_config`, not auth kwargs.
- `LiteLlm` from `google.adk.models.lite_llm`. `GenerateContentConfig`/`SafetySetting`/`HarmCategory`/`HarmBlockThreshold` from `google.genai.types`.
- langchain/crewai re-exported under `google.adk.integrations.*` (the `google.adk.tools.*` paths warn-deprecate).

## P4c safety+observability facts (see `docs/adk-api-notes/safety-observability.md`)
- **Agent callbacks (`LlmAgent` kwargs, positional callables)**: `before_model_callback(callback_
  context, llm_request) -> LlmResponse | None` (non-None short-circuits the LLM — PROVEN offline),
  `before_tool_callback(tool, args, tool_context) -> dict | None` (non-None short-circuits the
  tool), `before_agent_callback(callback_context) -> Content | None`; also after_*/on_*_error. Each
  kwarg accepts a single callable OR list OR None. `project_model` gains `CallbackSpec` (hook +
  policy + params) on `AgentSpec.callbacks`; renders a REAL generated guardrail `def` per hook
  attached via the real kwarg. 3 concrete policies: `block_keywords`/`max_input_chars` (before_model)
  + `block_tool` (before_tool). Shared `_user_text`/`_refuse` helpers emitted once (the `_refuse`
  helper keeps the guardrail body a single-arg call → ruff-stable even for long refusals; E501 only
  fires on a long refusal *with spaces*, same as a long instruction — generated code is held to
  format+isort, the established bar). ast.parse + ruff format + isort clean (proven).
- **Plugins via `App` (NOT deprecated `Runner(plugins=)`)**: `Runner(plugins=[...])` emits a real
  `DeprecationWarning` in 2.1.0; the clean path is `Runner(app=App(name=, root_agent=, plugins=[...]),
  session_service=...)` — ZERO warnings (verified). `build_runner(..., plugins=None)`: no plugins →
  unchanged `Runner(app_name=, agent=)`; with plugins → the App path. `runtime.json` gains a
  `plugins` manifest (`PluginSpec{var,name,kind}`), emitted ONLY when non-empty (byte-identical to
  P2a/P2b otherwise). `import_project_plugins` loads `plugins.py` instances by var name (fresh
  compile/exec, no stale `sys.modules`). PROVEN offline: a `BasePlugin.on_event_callback` plugin
  records events through `build_runner`+App.
- **`BasePlugin` hooks are keyword-only async** (`on_event_callback(self,*,invocation_context,event)`,
  `before_tool_callback(self,*,tool,tool_args,tool_context)`). `safety_add_plugin` generates real
  subclasses: `logging` (on_event → module-level `<var>_events` list + `logging`) / `tool_denylist`
  (before_tool → blocks denylisted tools). Both PROVEN offline (events recorded / tool blocked).
- **`safety_settings`**: `gemini_safety` routes through the EXISTING `GenerateContentConfigSpec` +
  models-domain `types.SafetySetting` rendering (merges with an existing GCC; NO duplication).
  `max_llm_calls` → new `AgentSpec.max_llm_calls` field, serialized to the sidecar but NOT rendered
  into `agent.py` (it's a `RunConfig` setting). PROVEN: SafetySetting lands in GCC; max_llm_calls
  absent from `agent.py`.
- **Observability = standard OpenTelemetry**. ADK's `google.adk.telemetry.tracer` is an OTel
  `ProxyTracer` on the GLOBAL provider → a user enables a custom exporter by installing a
  `TracerProvider` as the global provider. `enable_otel` generates `otel_setup.py` (ast-valid,
  ruff/isort-clean): console exporter (base OTel SDK — CORE, `setup_otel()` actually runs + installs
  the global provider, PROVEN) or OTLP (lazy import; `opentelemetry-exporter-otlp` is a SEPARATE
  package, NOT installed — codegen-only with an install hint). `cloud_trace(target)` returns the REAL
  `--trace_to_cloud` flag (+ `--otel_to_cloud`), both confirmed on deploy cloud_run/agent_engine/gke
  + web/api_server (NOT `adk run`), and references the deploy/dev tool that applies it (no flag emitted
  — no duplication). `third_party(provider)` (phoenix/arize/weave/signoz/otlp) emits OTLP env vars +
  a setup snippet (secrets via env, never hardcoded). `trace_view` DELEGATES to `dev.web` (the ADK
  Web UI hosts the trace view) via the same `adk_cli` process registry. **NO `uv.lock`/`pyproject`
  change** (OTel SDK is a core google-adk dep; OTLP codegen-only).

## Test/quality state
**641 passed, 6 skipped** (after P5; +4 skill tests + prior additions on top of P4c's 633). Coverage
gate green (suite exits 0; the skill test lives in `tests/`, source coverage unchanged ~95%). ruff +
mypy clean (34 source files); full
suite green under `-W error::DeprecationWarning` (28 warnings, all benign `UserWarning` from
experimental ADK features — zero `DeprecationWarning`). **`uv.lock`/`pyproject` UNCHANGED by P5**
(docs + stdlib-only test). P5 facts below; P4c snapshot:
suite green under `-W error::DeprecationWarning` (28 warnings, all benign `UserWarning` from
experimental ADK features — zero `DeprecationWarning`). **`uv.lock`/`pyproject` UNCHANGED by P4c**
(no new deps — OTel SDK already a google-adk dep; OTLP/plugins codegen-only). P4c files:
`domains/safety.py` (141 stmts, 87% cov), `domains/safety_plugins.py` (75, 91%),
`domains/observability.py` (64, 98%), `domains/observability_setup.py` (29, 97%), `project_model`
callbacks (specs/_codegen/render/sidecar — specs 100%, render 98%, _codegen 98%), `run_core` plugins
(100% cov), `runtime` PluginSpec (91%); tests `test_safety.py` (20, incl. 5 functional offline),
`test_observability.py` (19), extended `test_project_model.py` (+13 callbacks), `test_run_core.py`
(+6 plugins), `test_runtime.py` (+3 manifest). FUNCTIONAL safety proof (no key): a `block_keywords`
before_model guardrail returns the canned refusal AND the instrumented FakeLlm's call-list stays
EMPTY (short-circuit proven end-to-end); a `block_tool` denylist short-circuits the tool; a generated
`logging` plugin records events + a `tool_denylist` plugin blocks a tool through `build_runner`(App).
The 6 skips are unchanged (gcp×2 + real-api_server boot + 3 a2a-gated). No orphaned processes/ports.

## P5 skill facts (the `adk-toolkit` companion skill)
- **Deliverable:** `skill/SKILL.md` + `skill/references/00..13-*.md` (14 refs), ALSO installed to
  `C:\Users\bojac\.claude\skills\adk-toolkit\` (SKILL.md + references/) — the only out-of-repo write.
  Skill confirmed discoverable (appears in the harness `available_skills` list after install).
- **SKILL.md** = a SHORT routing index (mental model + a task→reference→tool-prefix table + the golden
  workflow). Frontmatter `name: adk-toolkit` + a pushy, trigger-rich `description` (build/scaffold ADK
  agent, google-adk, LlmAgent, sub_agents, AgentTool, tools, LiteLlm, state prefixes, Runner/RunConfig,
  evalset, deploy Agent Engine/Cloud Run/GKE, A2A/RemoteA2aAgent, guardrail/callback/plugin, OTel, etc.).
  Progressive disclosure: detail lives in the references, not the body.
- **References (dense, accurate to the api-notes + real tool names):** 00-mental-model (ADK + sidecar +
  `project_*`), 01-agent-types (decision tree sub_agents vs AgentTool vs RemoteA2aAgent; Loop/Parallel
  deprecated-but-functional; `agents_*`), 02-tools (all kinds + extras + auth; `tools_*`), 03-models
  (Gemini vs LiteLlm/LM-Studio loopback; GCC+safety; `models_*`), 04-sessions-state (backends, prefixes,
  temp-not-persisted, async; `sessions_*`), 05-memory-artifacts (keyword recall vs versioned Parts;
  `memory_*`/`artifacts_*`), 06-runtime (Runner/RunConfig/streaming; `run_*`), 07-eval (offline metrics
  vs LLM-judge; `eval_*`), 08-deploy (REAL 2.1.0 flags + dev servers; `deploy_*`/`dev_*`), 09-a2a
  (expose/consume + MCP bridge; `a2a_*`/`mcp_bridge_*`), 10-observability (OTel/Cloud Trace/3rd-party;
  `observability_*`), 11-safety (callbacks vs plugins via App; `safety_*`), 12-troubleshooting (every
  known pitfall: deprecations, `request_input` gone, RemoteA2aAgent import path, DB async URL, drifted
  CLI flags, missing extras, regen-don't-edit), 13-tool-catalog (THE bridge: all 81 tools by domain).
- **Tool-name cross-check (load-bearing):** built the server, listed tools (**81**), grepped every
  `<domain>_<name>` token across the skill — ALL 81 real tools are mentioned; the only non-matching
  tokens were legitimate ADK API terms / param names, NOT claimed tools (`a2a_app` the generated var,
  `a2a_experimental` the decorator, `eval_set_file`/`run_config`/`run_async` params, `project_model` the
  internal module explicitly flagged as not-exposed). Per-domain counts in 13-tool-catalog match reality
  exactly (5/10/13/3/8/3/6/5/4/6/6/2/3/3/4 = 81).
- **Test:** `tests/unit/test_skill.py` (4 tests, no new deps) — minimal frontmatter parse (handles the
  `>-` block scalar): asserts SKILL.md exists, `name == "adk-toolkit"` + non-empty description, every
  `references/*.md` cited by SKILL.md exists, and the 14 canonical refs are present.
- **No `pyproject`/`uv.lock` change** (docs + a stdlib-only test). README already pointed at `skill/`.

## P6a Code Mode + prompts facts (see `docs/adk-api-notes/fastmcp-codemode.md`)
- **Real Code Mode EXISTS in fastmcp 3.3.1** as a catalog transform:
  `fastmcp.experimental.transforms.code_mode.CodeMode` (a `CatalogTransform`/`Transform` subclass,
  NOT re-exported at the top level). Applied via `mcp.add_transform(CodeMode(...))` AFTER all mounts;
  it REPLACES the whole catalog with discovery + `execute` meta-tools. Proven: 81 tools → 4
  (`search`/`get_schema`/`tags`/`execute`) via in-memory `Client.list_tools()`. `@FastMCP.tool` and
  `@FastMCP.prompt` both accept `tags: set[str]`.
- **HONEST caveat (documented in api-notes + README):** the `execute` meta-tool's default sandbox
  `MontySandboxProvider` lazily `import`s `pydantic-monty` (the `fastmcp[code-mode]` extra), which is
  NOT installed here → calling `execute` raises a clear `ImportError`. The DISCOVERY tools
  (`search`/`get_schema`/`tags`/`list_tools`) work WITHOUT monty (verified via Client). The
  token-efficiency win (cheap catalog) is fully functional without the extra; we wire real Code Mode
  and gate it, and do NOT add `pydantic-monty` to the locked deps. NOT faked.
- **TASK 1 — domain tags:** all 81 `@<domain>_server.tool` decorators carry `tags={"<domain>"}`
  (mechanical inject: bare `@x.tool` → `@x.tool(tags={"d"})`; `@x.tool(name="list")` →
  `@x.tool(tags={"d"}, name="list")`). Exposed tool NAMES unchanged (tags are metadata); surfaced to
  MCP clients via `_meta.fastmcp.tags`. Server-side `mcp.list_tools()` `Tool.tags` carries the tag;
  the client's `mcp.types.Tool` does NOT expose `.tags` (use `_meta`). The 5 prompts carry
  `tags={"workflow"}`.
- **TASK 2 — opt-in Code Mode:** `build_server(code_mode: bool = False)`. DEFAULT = direct tools
  (all 81 by name → read-through tests unchanged). `code_mode=True` (or env `ADK_TOOLKIT_CODE_MODE` ∈
  {1,true,yes,on}, parsed by `code_mode_enabled()`) applies `CodeMode(discovery_tools=[Search(),
  GetSchemas(), GetTags()])` — `GetTags` added BECAUSE we tag by domain (browse 15 domains →
  `search(tags=[...])` → `get_schema` → `execute`). `main()` reads the env flag. Token-surface
  reduction proven: 81 → 4 (≥90% fewer top-level tools).
- **TASK 3 — 5 workflow prompts** (`prompts.py`, real `@mcp.prompt`, each returns a templated string):
  `scaffold_multi_agent(goal)` (project_create → agents_create_llm → agents_create_sequential/
  parallel/loop → agents_compose/set_root → models_set/configure_litellm → run_agent),
  `add_guardrail(agent, concern)` (callback per-agent via `safety_add_callback` vs global plugin via
  `safety_add_plugin` decision + `safety_settings`), `write_evalset(agent)` (eval_create_set →
  eval_set_criteria → eval_run → eval_report, offline-metric guidance), `deploy_checklist(target)`
  (deploy_preflight → deploy_containerize → deploy_agent_engine/cloud_run/gke with REAL 2.1.0 flags +
  creds reminders → deploy_status), `debug_agent(symptom)` (run_agent/run_stream →
  run_inspect_events; agents_list/get, tools_list; known pitfalls). **Cross-check (load-bearing):**
  37 distinct `<domain>_*` tokens cited across the prompts, ALL real tools (the only false-positive
  was the param `eval_set_file` — reworded out so the regex stays clean, mirroring the P5 approach).
- **Read via the REAL client API:** `Client.list_prompts()` (→ `mcp.types.Prompt` with `.name`/
  `.description`/`.arguments`) and `Client.get_prompt(name, {args})` (→ `result.messages[0].content`
  is a `TextContent` with `.text`). Confirmed by introspection before writing tests.
- **Tests:** `tests/unit/test_server.py` (rewritten: 81-count + sample names + no double-prefix; every
  tool's domain tag via server-side `.tags`; tag surfaced via client `_meta`; code_mode surface
  collapse + `tags` discovery reachable; `code_mode_enabled` env parsing parametrized) and NEW
  `tests/unit/test_prompts.py` (5 registered + declared args; render non-empty/actionable; arg
  interpolation; cross-check every cited token is a real tool; key pivot tools cited; workflow tag).
- **Quality:** 669 passed, 6 skipped (was 641/6 after P5; +28 P6a tests), coverage 95.41% (gate 80%),
  under `-W error::DeprecationWarning` (28 benign UserWarning, zero DeprecationWarning). ruff + mypy
  clean (34 src files / 64 formatted). **`uv.lock`/`pyproject` UNCHANGED** (Code Mode + tags + prompts
  are stdlib/fastmcp-only; `pydantic-monty` deliberately NOT added). `prompts.py` 100% cov, `server.py`
  96%. The 6 skips are unchanged (gcp×2 + real-api_server + 3 a2a-gated).

## (historical) P4b test/quality snapshot
**572 passed, 6 skipped, coverage 95.67%** (after P4b). ruff + mypy clean (30 source files); full
suite green under `-W error::DeprecationWarning` (28 warnings, all benign `UserWarning` from
experimental ADK features — zero `DeprecationWarning`). **`uv.lock`/`pyproject` UNCHANGED by P4b**
(a2a extra installed transiently for introspection only; blob hashes verified vs baseline). P4b
files: `domains/a2a.py` (104 stmts, 78% cov — execute/build blocks gated on the extra),
`domains/mcp_bridge.py` (52 stmts, 92% cov), `project_model` `remote_a2a` (specs/render/sidecar);
tests `test_a2a.py` (22 tests, 3 gated skips), `test_mcp_bridge.py` (12 tests, fully CI), extended
`test_project_model.py` (+7 remote_a2a tests). The 6 skips = 3 prior (gcp×2 + real-api_server boot)
+ 3 new a2a-gated (real RemoteA2aAgent type / live uvicorn boot / real AgentCard build). FUNCTIONAL
mcp_bridge proof (no extra): `adk_to_mcp_tool_type(google_search)` → MCP Tool
`{name:'google_search', description:'google_search', inputSchema:{}}`; a FunctionTool → a real
JSON-Schema. P4a files: `adk_cli.py` (~280 lines, 93% cov),
`domains/deploy.py` (155 stmts, 94% cov), `domains/dev.py` (86 stmts, 95% cov); test files
`test_adk_cli.py` (21 tests), `test_deploy.py` (33), `test_dev.py` (15, 1 gated skip). The 3rd
skip is the gated real-`adk api_server`-boot test (`ADK_TOOLKIT_TEST_API_SERVER=1`; PASSES locally
in ~3.4s). No orphaned processes/ports after the suite (verified). No pyproject/uv.lock change
(adk_cli shells to the CLI; no new deps). P3b files below unchanged.

P3b files: `domains/eval.py` ≈400 lines (98% cov);
`tests/unit/test_eval.py` (33 tests) reusing `tests/unit/fake_llm.py`. `pyproject` `dev` extra
now references `adk-toolkit-mcp[eval]`; `uv.lock` 2-line metadata edge only (eval packages were
already locked). The 2 skips are gcp-absent conditional tests (now skipped because the eval extra
pulls `vertexai`/`google.cloud.storage` via aiplatform); the litellm probe now RUNS (litellm is
an eval-extra transitive dep) — was 1 skip in P3a, now 2, no regression. FUNCTIONAL result: a
FakeLlm/ScriptedLlm agent PASSES `tool_trajectory_avg_score`=1.0 + `response_match_score` (ROUGE)
OFFLINE (no key) via a REAL `AgentEvaluator`; a wrong expected answer correctly FAILS
(`ok=True, passed=False`); per-metric scores captured into a persisted report; missing-creds /
LLM-judge / eval-extra-absent return a clean `err` (no hang). P3a still green (run_core/run 100%).

## Resume instructions
**P6a COMPLETE** (domain tags on all 81 tools; opt-in FastMCP Code Mode in `build_server(code_mode=)` +
`ADK_TOOLKIT_CODE_MODE` env; 5 workflow prompts in `prompts.py` with a load-bearing tool-name
cross-check; `docs/adk-api-notes/fastmcp-codemode.md`; rewritten `test_server.py` + new `test_prompts.py`).
**P5 COMPLETE** (the `adk-toolkit` skill: SKILL.md + 14 refs, installed to `~/.claude/skills/adk-toolkit/`,
4-test stdlib-only `test_skill.py`, all 81 tools cross-checked). P4 also COMPLETE (deploy+dev P4a ✅ ·
a2a+mcp_bridge P4b ✅ · safety+observability P4c ✅).
Next: **P6 — Finish (rest)** (docs `ARCHITECTURE.md` / `TOOL_CATALOG.md` / `CONTRIBUTING.md`, then repo
publish — **confirm GitHub push with the user** before adding any remote; local-only until then).
- P6a notes: Code Mode is REAL in fastmcp 3.3.1 (`fastmcp.experimental.transforms.code_mode.CodeMode`,
  applied via `mcp.add_transform(...)` AFTER mounts). DEFAULT stays direct-tools (81 by name) — do NOT
  flip the default. The `execute` sandbox needs the optional `fastmcp[code-mode]`/`pydantic-monty` extra
  (NOT installed, NOT locked); discovery tools work without it. If P6 ADDS/renames a tool: keep its
  `tags={"<domain>"}` decorator, and if it's cited in a prompt re-run the `test_prompts.py` cross-check.
  Prompts are read via `Client.list_prompts()`/`get_prompt(name, {args})` (`messages[0].content.text`).
- Established patterns to keep: lazy optional imports, `{ok,data,error}` envelope, bare tool names
  mounted via `namespace=`, sidecar/Workspace for generated files, actionable `err` for an absent
  extra (`gcp`/`db`/`eval`/`a2a`), generated code held to ast.parse + ruff format + isort.
- P5 notes: the skill is the user-facing surface for the 81 tools — if P6 ADDS/renames a tool, update
  BOTH `skill/references/13-tool-catalog.md` (+ the relevant domain ref) AND re-run the grep cross-check,
  AND re-copy the skill to `~/.claude/skills/adk-toolkit/` (the test only checks structure, not tool
  names — keep the catalog honest manually). The skill description in SKILL.md is the trigger surface.
- P4c notes: agent guardrails render as real functions attached via the real `LlmAgent` callback
  kwargs; plugins wire via `Runner(app=App(plugins=[...]))` (NOT the deprecated `plugins=` kwarg);
  `safety_settings` reuses the models GCC/SafetySetting rendering; observability wraps standard
  OTel (console core, OTLP codegen-only) and `cloud_trace`/`trace_view` reference deploy/dev (no
  duplication). Reuse `adk_cli` (process registry) for any new CLI-backed tool.
- Keep the suite green under `-W error::DeprecationWarning`, coverage ≥80%, no orphaned processes.
