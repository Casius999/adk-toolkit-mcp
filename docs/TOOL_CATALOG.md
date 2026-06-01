# adk-toolkit-mcp — Tool Catalog

All **81 tools** across 14 domains. Every tool returns `{ok, data, error}`. Exposed name =
`<domain>_<bare>`. Tool parameters shown beside each entry; `path` and `app_name` are
required by most tools (the project directory and agent package name).

**Count:** 5 + 10 + 13 + 3 + 8 + 3 + 6 + 5 + 4 + 6 + 6 + 3 + 2 + 3 + 4 = **81 tools**

---

## Contents

- [project (5)](#project) — scaffold and inspect
- [agents (10)](#agents) — compose the agent graph
- [tools (13)](#tools) — attach tools to LlmAgents
- [models (3)](#models) — model and generation config
- [sessions (8)](#sessions) — session state service
- [memory (3)](#memory) — long-term memory service
- [artifacts (6)](#artifacts) — versioned blob service
- [run (5)](#run) — execute the agent loop
- [eval (4)](#eval) — evaluate against an evalset
- [deploy (6)](#deploy) — build adk deploy commands
- [dev (6)](#dev) — managed local CLI servers
- [a2a (3)](#a2a) — Agent-to-Agent
- [mcp_bridge (2)](#mcp_bridge) — expose ADK tools as MCP
- [safety (3)](#safety) — guardrails and safety settings
- [observability (4)](#observability) — OpenTelemetry tracing
- [Resources and prompts](#resources-and-prompts)

---

## project

Scaffold a new ADK application and inspect/configure it. All project tools operate on the
on-disk layout: `<path>/<app_name>/` (`__init__.py`, `agent.py`, `.env`, `.adk_toolkit/`).

| Tool | Purpose | Key parameters |
|---|---|---|
| `project_create` | Scaffold `__init__.py`, `agent.py`, `.env` with a minimal `LlmAgent` | `path`, `app_name`, `model="gemini-2.5-flash"`, `backend="ai_studio"` |
| `project_inspect` | Inspect app structure: root agent present, Python files, `.env` keys | `path`, `app_name` |
| `project_set_env` | Set/merge `.env` key-value pairs (values redacted in the response) | `path`, `app_name`, `values: dict[str, str]` |
| `project_add_extra` | Append a `google-adk[extra]` dependency hint to the project | `path`, `app_name`, `extra` (e.g. `"gcp"`, `"eval"`, `"a2a"`) |
| `project_agent_config` | Write or read a no-code `root_agent.yaml` agent config | `path`, `app_name`, `yaml_content=None` |

---

## agents

Compose the agent graph. All tools write to the sidecar (`.adk_toolkit/agents.json`) and
regenerate `agent.py`. Generated code passes `ast.parse` + ruff format + isort.

| Tool | Purpose | Key parameters |
|---|---|---|
| `agents_create_llm` | Add or update an `LlmAgent` | `path`, `app_name`, `name`, `model="gemini-2.5-flash"`, `instruction=""`, `description=""`, `output_key=None` |
| `agents_create_sequential` | Add a `SequentialAgent` pipeline (runs sub-agents in order) | `path`, `app_name`, `name`, `sub_agents: list[str]`, `description=""` |
| `agents_create_parallel` | Add a `ParallelAgent` fan-out (runs sub-agents concurrently) | `path`, `app_name`, `name`, `sub_agents: list[str]`, `description=""` |
| `agents_create_loop` | Add a `LoopAgent` that repeats sub-agents until N iterations | `path`, `app_name`, `name`, `sub_agents: list[str]`, `max_iterations=3`, `description=""` |
| `agents_create_custom` | Add a custom `BaseAgent` stub with `_run_async_impl` | `path`, `app_name`, `name`, `description=""` |
| `agents_compose` | Replace an existing agent's `sub_agents` list | `path`, `app_name`, `name`, `sub_agents: list[str]` |
| `agents_set_root` | Designate which agent is the app's `root_agent` | `path`, `app_name`, `name` |
| `agents_as_tool` | Return the `AgentTool(agent=<name>)` source snippet (no file change) | `path`, `app_name`, `agent_name` |
| `agents_list` | List all agents with name, type, and root flag | `path`, `app_name` |
| `agents_get` | Get one agent's full spec (model, instruction, tools, callbacks, …) | `path`, `app_name`, `name` |

> **Note on workflow agents:** `SequentialAgent`, `ParallelAgent`, and `LoopAgent` emit a
> `DeprecationWarning` in ADK 2.1.0 ("use Workflow instead"). They are still functional and
> retained here. The toolkit's tests never construct them in-process; the generated `agent.py`
> is run in a subprocess.

---

## tools

Attach tools to an existing `LlmAgent`. All take `path`, `app_name`, `agent_name` as the
first three parameters. Only `LlmAgent`s carry tools.

| Tool | Purpose | Key parameters |
|---|---|---|
| `tools_add_function` | Add a Python function tool (generates a `def` in `agent.py`) | `func_name`, `params: list[{name, type, default?}]`, `docstring`, `returns="dict"`, `body="return {}"` |
| `tools_add_long_running` | Add a `LongRunningFunctionTool` (same signature as `add_function`) | same as `add_function` |
| `tools_add_builtin` | Add a core ADK builtin (`google_search`, `url_context`, `load_memory`, etc.) | `kind` (builtin name), `args=None` (only for `vertex_ai_search`: `data_store_id` or `search_engine_id`) |
| `tools_add_agent_tool` | Wrap an existing agent as a tool via `AgentTool(agent=<name>)` | `target_agent` (name of an agent in the same project) |
| `tools_add_openapi` | Add an `OpenAPIToolset` from a JSON/YAML spec string | `spec: str`, `name=None` |
| `tools_add_bigquery` | Add a `BigQueryToolset` (needs `bigquery` extra) | `name=None`, `args=None` |
| `tools_add_spanner` | Add a `SpannerToolset` (needs `spanner` extra) | `name=None`, `args=None` |
| `tools_add_mcp_toolset` | Add an `McpToolset` (consume another MCP server from the agent) | `transport` (`stdio`/`sse`/`http`), `command=None`, `args=None`, `url=None`, `headers=None`, `tool_filter=None`, `name=None` |
| `tools_add_apihub` | Add an `APIHubToolset` from an API Hub resource name | `apihub_resource_name`, `name=None` |
| `tools_add_langchain` | Add a `LangchainTool` wrapper around a LangChain tool | `import_line`, `tool_expr`, `name=None` |
| `tools_add_crewai` | Add a `CrewaiTool` wrapper around a CrewAI tool | `import_line`, `tool_expr`, `name`, `description` |
| `tools_set_auth` | Attach auth credential to a toolset (`openapi`/`apihub`/`mcp_toolset` only) | `tool_name`, `scheme` (`apikey`/`oauth2`/`service_account`/`bearer`), `credential: dict` |
| `tools_list` | List all tools attached to an agent | `agent_name` |

---

## models

Configure the model and generation parameters for an `LlmAgent`.

| Tool | Purpose | Key parameters |
|---|---|---|
| `models_set` | Set a native Gemini model string (clears any LiteLlm config) | `path`, `app_name`, `agent_name`, `model` (e.g. `"gemini-2.5-flash"`) |
| `models_configure_litellm` | Use a non-Gemini provider via `LiteLlm` (`litellm` extra required in the generated app) | `path`, `app_name`, `agent_name`, `provider` (e.g. `"openai"`, `"anthropic"`, `"ollama"`), `model`, `api_base=""`, `api_key_env=""` |
| `models_generate_config` | Set `GenerateContentConfig` fields and/or Gemini safety settings on an agent | `path`, `app_name`, `agent_name`, `temperature=None`, `max_output_tokens=None`, `top_p=None`, `top_k=None`, `safety_settings=None`, `response_modalities=None` |

---

## sessions

Manage ADK session state. All calls are async. The session backend is configured once via
`sessions_service_set` and persisted to `runtime.json`.

| Tool | Purpose | Key parameters |
|---|---|---|
| `sessions_service_set` | Choose and persist the session backend | `path`, `app_name`, `kind` (`in_memory`/`database`/`vertex`), `db_url=None`, `project=None`, `location=None` |
| `sessions_create` | Create a new session | `path`, `app_name`, `user_id`, `state=None`, `session_id=None` |
| `sessions_get` | Get a session (id, event count, state snapshot) | `path`, `app_name`, `user_id`, `session_id` |
| `sessions_list` | List session ids for a user | `path`, `app_name`, `user_id` |
| `sessions_delete` | Delete a session | `path`, `app_name`, `user_id`, `session_id` |
| `sessions_state_set` | Mutate a state key (persisted via `append_event(state_delta)`) | `path`, `app_name`, `user_id`, `session_id`, `key`, `value`, `scope="session"` (`session`/`app`/`user`/`temp`) |
| `sessions_state_get` | Read a state key | `path`, `app_name`, `user_id`, `session_id`, `key`, `scope="session"` |
| `sessions_append_event` | Append a raw `Event` (text turn and/or state delta) | `path`, `app_name`, `user_id`, `session_id`, `author`, `text=None`, `state_delta=None` |

> **State prefix rules:** `session:` (bare key) / `app:` / `user:` / `temp:`. `temp:` keys are
> visible immediately after `state_set` but are NOT persisted across a subsequent `get_session`.
> Database backend requires `sqlite+aiosqlite:///<path>` (async driver).

---

## memory

Long-term memory service. All calls are async.

| Tool | Purpose | Key parameters |
|---|---|---|
| `memory_service_set` | Choose and persist the memory backend | `path`, `app_name`, `kind` (`in_memory`/`vertex_rag`/`vertex_memory_bank`), `project=None`, `location=None`, `rag_corpus=None`, `agent_engine_id=None` |
| `memory_add_session` | Ingest a completed session into memory | `path`, `app_name`, `user_id`, `session_id` |
| `memory_search` | Search memory (keyword-based for `in_memory`; semantic for Vertex backends) | `path`, `app_name`, `user_id`, `query` |

> `in_memory` recall is keyword-based (not semantic): a query word must literally appear in an
> event's text. Only events with `content.parts` text are indexed.

---

## artifacts

Versioned blob storage. All calls are async.

| Tool | Purpose | Key parameters |
|---|---|---|
| `artifacts_service_set` | Choose and persist the artifact backend | `path`, `app_name`, `kind` (`in_memory`/`gcs`), `bucket=None` |
| `artifacts_save` | Save a text or binary artifact; returns version int (0-based) | `path`, `app_name`, `user_id`, `session_id`, `filename`, `text=None`, `bytes_b64=None`, `mime_type="text/plain"` (exactly one of `text`/`bytes_b64`) |
| `artifacts_load` | Load an artifact (latest or specific version) | `path`, `app_name`, `user_id`, `session_id`, `filename`, `version=None` |
| `artifacts_list` | List artifact filenames for a session | `path`, `app_name`, `user_id`, `session_id` |
| `artifacts_delete` | Delete all versions of an artifact | `path`, `app_name`, `user_id`, `session_id`, `filename` |
| `artifacts_versions` | List all version numbers for an artifact | `path`, `app_name`, `user_id`, `session_id`, `filename` |

> `user:`-prefixed filenames (e.g. `user:profile.json`) are user-scoped (shared across
> sessions for that user). GCS backend needs the `gcp` extra.

---

## run

Execute a real ADK agent loop. The domain imports the project's `root_agent` from `agent.py`
and wires a `Runner` using the configured runtime services.

| Tool | Purpose | Key parameters |
|---|---|---|
| `run_agent` | Run the root agent on a message; returns all serialized events | `path`, `app_name`, `user_id`, `session_id`, `message`, `max_llm_calls=None`, `streaming_mode="NONE"` |
| `run_stream` | Run with SSE progress reporting (reports one progress notification per event) | `path`, `app_name`, `user_id`, `session_id`, `message`, `max_llm_calls=None` |
| `run_live` | BIDI live session (experimental; requires live-capable Gemini + creds) | `path`, `app_name`, `user_id`, `session_id`, `message`, `max_llm_calls=None` |
| `run_config_build` | Validate and describe a `RunConfig` without running the agent | `path`, `app_name`, `streaming_mode="NONE"`, `max_llm_calls=None`, `response_modalities=None` |
| `run_inspect_events` | Summarize a list of serialized events (pure; no agent invoked) | `path`, `app_name`, `events: list[dict]` |

> `run_live` degrades gracefully: it detects missing credentials or a non-live-capable model
> and returns `err(...)` before opening any connection.

---

## eval

Evaluate the project's agent against an evalset. All calls are async. Eval files live in
`<app_dir>/eval/`.

| Tool | Purpose | Key parameters |
|---|---|---|
| `eval_create_set` | Write a schema-conformant `.evalset.json` (real `EvalSet` pydantic models) | `path`, `app_name`, `name`, `cases: list[{query, expected_response, expected_tool_use?}]` |
| `eval_set_criteria` | Write offline metric thresholds to `test_config.json` | `path`, `app_name`, `tool_trajectory_avg_score=1.0`, `response_match_score=0.8` |
| `eval_run` | Import agent, run evaluation, persist a report; returns verdict + per-metric scores | `path`, `app_name`, `eval_set_file`, `config_file=None`, `num_runs=1`, `agent_name=None` |
| `eval_report` | Read a stored evaluation report by id | `path`, `app_name`, `report_id` |

> Offline metrics (no API key): `tool_trajectory_avg_score` (structural trajectory compare) and
> `response_match_score` (ROUGE-1). LLM-judge metrics need a model and credentials. The `eval`
> extra is required. An eval failure (`agent does not meet thresholds`) returns
> `ok=True, data={passed: False}` — a normal result, not a tool error.
>
> Internally `eval_run` uses `AgentEvaluator._get_eval_results_by_eval_id` +
> `final_eval_status` to capture per-metric scores, rather than the assert-based public API.

---

## deploy

Build and optionally execute `adk deploy` commands against GCP. `execute=False` (default)
returns the planned argv and notes without running anything; `execute=True` calls the real CLI.

| Tool | Purpose | Key parameters |
|---|---|---|
| `deploy_preflight` | Check that `gcloud`/`adk`/`kubectl` are on PATH | `path`, `app_name`, `target="cloud_run"` |
| `deploy_agent_engine` | Deploy to Vertex AI Agent Engine | `path`, `app_name`, `project`, `region`, `display_name=None`, `requirements_file=None`, `execute=False` |
| `deploy_cloud_run` | Deploy to Cloud Run | `path`, `app_name`, `project`, `region`, `service_name=None`, `with_ui=False`, `enable_cloud_trace=False`, `execute=False` |
| `deploy_gke` | Deploy to GKE | `path`, `app_name`, `project`, `region`, `cluster`, `service_name=None`, `execute=False` |
| `deploy_containerize` | Write an idempotent `Dockerfile` that serves `adk api_server` | `path`, `app_name` |
| `deploy_status` | Best-effort deployment status via `gcloud`/`kubectl` (20 s timeout) | `path`, `app_name`, `target`, `project=None`, `region=None`, `service_name=None`, `cluster=None` |

> Flags are validated against the real `adk <subcommand> --help` output (`available_flags`)
> to prevent drift. ADK 2.1.0 real flag names: `--display_name` (not `--app_name`) for
> `agent_engine`; `--trace_to_cloud` (not `--enable_cloud_trace`) for `cloud_run`/`gke`;
> `--cluster_name` (not `--cluster`) for `gke`.

---

## dev

Managed local development servers using the `adk_cli` process registry.

| Tool | Purpose | Key parameters |
|---|---|---|
| `dev_web` | Start `adk web` (Dev UI + Eval/Trace tabs) as a managed background process | `path`, `app_name=None`, `port=8000`, `host="127.0.0.1"` |
| `dev_api_server` | Start `adk api_server` (FastAPI, no UI) as a managed background process | `path`, `app_name=None`, `port=8000`, `host="127.0.0.1"` |
| `dev_run` | One-shot `adk run AGENT "<message>"` (non-interactive, bounded timeout) | `path`, `app_name`, `message=None` |
| `dev_stop` | Stop a managed server by its process key | `key` |
| `dev_status` | Check the status of a managed server | `key` |
| `dev_logs` | Tail log output of a managed server | `key`, `tail=50` |

> The `message` positional arg to `adk run` must be provided for non-interactive use; without
> it the CLI enters interactive mode (which would block). `dev_run` returns guidance when
> `message` is omitted.

---

## a2a

Agent-to-Agent (A2A) integration. Requires the `a2a` extra for `a2a_expose` and
`a2a_agent_card`. `a2a_consume` is codegen-only (no extra needed to generate, only to run).

| Tool | Purpose | Key parameters |
|---|---|---|
| `a2a_consume` | Add a `RemoteA2aAgent` proxy to the sidecar (consume a remote A2A agent) | `path`, `app_name`, `name`, `agent_card_url` |
| `a2a_expose` | Write `a2a_app.py` and optionally serve via `uvicorn` (needs `a2a` extra) | `path`, `app_name`, `port=8001`, `execute=False` |
| `a2a_agent_card` | Build and return the `AgentCard` for the project's root agent (needs `a2a` extra) | `path`, `app_name`, `port=8001` |

> `RemoteA2aAgent` must be imported from `google.adk.agents.remote_a2a_agent` (not from
> `google.adk.agents` — it is not in `__all__` in 2.1.0). The generated `a2a_app.py` uses
> `uvicorn a2a_app:a2a_app` as the serve command; routes are registered lazily on startup.

---

## mcp_bridge

Expose ADK tools as MCP tool schemas. No extra required (`mcp` is a core `fastmcp` dep).

| Tool | Purpose | Key parameters |
|---|---|---|
| `mcp_bridge_convert_builtin` | Convert a core ADK builtin to an `mcp.types.Tool` schema | `kind` (core builtin name; not `vertex_ai_search` which needs an arg) |
| `mcp_bridge_expose_adk_tools` | Import the project agent and convert all its tools to MCP schemas | `path`, `app_name`, `agent_name` |

> Uses `agent.canonical_tools(ctx=None)` (async) to ensure plain functions are wrapped into
> `FunctionTool` before conversion. Workflow agents (`SequentialAgent` etc.) lack
> `canonical_tools` and are rejected with an actionable error.

---

## safety

Guardrails and safety configuration.

| Tool | Purpose | Key parameters |
|---|---|---|
| `safety_add_callback` | Attach a generated guardrail function to an agent via its callback kwarg | `path`, `app_name`, `agent_name`, `hook` (`before_model`/`after_model`/`before_tool`/`after_tool`/`before_agent`/`after_agent`), `policy: {kind, …}` (e.g. `{kind: "block_keywords", keywords: [...]}`) |
| `safety_add_plugin` | Generate a `BasePlugin` subclass in `plugins.py` for a global guardrail | `path`, `app_name`, `name`, `kind` (`logging`/`tool_denylist`), `config=None` |
| `safety_settings` | Set Gemini safety thresholds and/or max LLM call budget on an agent | `path`, `app_name`, `agent_name`, `max_llm_calls=None`, `gemini_safety: list[{category, threshold}]` |

> Plugins are wired via `Runner(app=App(name, root_agent, plugins=[...]))` (the non-deprecated
> path). Callback guardrails are generated functions attached via the real `LlmAgent` callback
> kwargs (e.g. `before_model_callback=_guard_before_model`). Three concrete policies:
> `block_keywords` and `max_input_chars` (before_model) and `block_tool` (before_tool).

---

## observability

OpenTelemetry tracing setup.

| Tool | Purpose | Key parameters |
|---|---|---|
| `observability_enable_otel` | Generate `otel_setup.py` with a console or OTLP span exporter | `path`, `app_name`, `exporter="console"`, `endpoint=None` |
| `observability_cloud_trace` | Return the `--trace_to_cloud` CLI flag and the deploy/dev tool to which it applies | `path`, `app_name`, `target` (`cloud_run`/`agent_engine`/`gke`/`web`/`api_server`) |
| `observability_third_party` | Return OTLP env vars and a setup snippet for a third-party backend | `path`, `app_name`, `provider` (`phoenix`/`arize`/`weave`/`signoz`/`otlp`), `endpoint=None`, `headers=None` |
| `observability_trace_view` | Open the ADK Web UI trace view (delegates to `dev_web`) | `path`, `app_name=None`, `port=8000` |

> ADK uses standard OpenTelemetry: install a `TracerProvider` as the global provider before
> the agent runs and ADK spans flow to your exporter automatically. The OTLP exporter
> (`opentelemetry-exporter-otlp`) is NOT installed by default; the generated `otel_setup.py`
> documents the install step.

---

## Resources and prompts

### Resources (not tools)

| URI | Returns |
|---|---|
| `adk://version` | Pinned `google-adk`, `fastmcp`, Python version strings. |
| `adk://models` | Common Gemini model strings. |

### Prompts (5)

Accessed via `get_prompt(name, {args})`. Each returns a templated string that cites the exact
tool call sequence for a common task.

| Prompt | Arg(s) | Covers |
|---|---|---|
| `scaffold_multi_agent` | `goal` | `project_create` → `agents_create_llm` → `agents_create_sequential`/`parallel`/`loop` → `agents_compose`/`agents_set_root` → `models_set`/`models_configure_litellm` → `run_agent` |
| `add_guardrail` | `agent`, `concern` | `safety_add_callback` (per-agent) vs `safety_add_plugin` (global) + `safety_settings` |
| `write_evalset` | `agent` | `eval_create_set` → `eval_set_criteria` → `eval_run` → `eval_report` (offline metrics) |
| `deploy_checklist` | `target` | `deploy_preflight` → `deploy_containerize` → `deploy_agent_engine`/`deploy_cloud_run`/`deploy_gke` with real 2.1.0 flags → `deploy_status` |
| `debug_agent` | `symptom` | `run_agent`/`run_stream` → `run_inspect_events`; `agents_list`/`agents_get`, `tools_list`; known pitfalls |
