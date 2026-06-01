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
| **P4 Ops** | deploy (P4a) ✅ · dev (P4a) ✅ · a2a, observability, safety, mcp_bridge | ⬜ **(P4a done; a2/mcp_bridge/safety/observability next)** |
| **P5 Skill** | `adk-toolkit` skill (SKILL.md + 14 refs) | ⬜ |
| **P6 Finish** | Code Mode, prompts, docs, repo publish (confirm GitHub) | ⬜ |

## Exposed tools so far (69)
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
537 passed, 3 skipped, coverage 96.28%. ruff + mypy clean; full suite green under
`-W error::DeprecationWarning` (27 warnings, all benign `UserWarning` from experimental ADK
features — zero `DeprecationWarning`). P4a files: `adk_cli.py` (~280 lines, 93% cov),
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
P4a (deploy + dev) ✅ DONE. Next: **P4 remaining — a2a, mcp_bridge, safety, observability**.
Follow the established pattern: lazy optional imports, `{ok,data,error}` envelope, bare tool
names mounted via `namespace=`, sidecar/Workspace for any generated files, actionable `err` for
an absent extra (mirror `gcp`/`db`/`eval`). Introspect each ADK surface BEFORE building and record
facts in `docs/adk-api-notes/<domain>.md`; commit per-domain.
- `a2a` (Agent-to-Agent) likely needs the `a2a` extra — introspect `google.adk` for the A2A
  server/card surface (e.g. `to_a2a`/agent card export) AND note `adk deploy cloud_run`/`gke`
  already expose an `--a2a` endpoint flag (the dev/deploy domains could surface that too).
- `observability` may wrap OpenTelemetry (already a core google-adk dep) — note the real CLI flags
  `--trace_to_cloud`/`--otel_to_cloud` already discovered on web/api_server/deploy.
- `safety` overlaps with `models` SafetySetting (P1) — check for reuse; ADK eval also has safety
  LLM-judge metrics (needs creds — keep offline-friendly).
- `mcp_bridge` wraps `McpToolset` (P1 `tools_add_mcp_toolset` already exists — avoid duplication;
  expose the *serving* side if distinct: ADK can also EXPOSE tools as an MCP server).
Reuse `adk_cli` (process registry + `run_adk` + `available_flags`) for any new CLI-backed tool.
Keep the suite green under `-W error::DeprecationWarning`, coverage ≥80%, no orphaned processes.
After P4: P5 (skill) then P6 (Code Mode, prompts, docs, repo publish — confirm GitHub push).
