# 13 — Tool catalog: "I want to do X" → exact MCP tool

The authoritative bridge from intent to the exact adk-toolkit-mcp tool. **All 81 tools across 17
domains**, grouped by domain, with exact exposed names and key arguments. Every tool returns
`{ok, data, error}`. When unsure which tool implements a step, this is the file to open.

> Exposed name = `<domain>_<bare>`. Args shown are the tool parameters (besides the always-present
> `path, app_name` for app-scoped tools). Defaults shown where they matter.

## Contents
- [project (5)](#project) · [agents (10)](#agents) · [tools (13)](#tools) · [models (3)](#models)
- [sessions (8)](#sessions) · [memory (3)](#memory) · [artifacts (6)](#artifacts)
- [run (5)](#run) · [eval (4)](#eval) · [deploy (6)](#deploy) · [dev (6)](#dev)
- [a2a (3)](#a2a) · [mcp_bridge (2)](#mcp_bridge) · [safety (3)](#safety) · [observability (4)](#observability)
- [Resources](#resources) · [Count check](#count)

## <a name="project"></a>project — scaffold & inspect (5)
| I want to… | Tool | Key args |
|---|---|---|
| Scaffold a new ADK app (`__init__.py` + `agent.py` + `.env`) | `project_create` | `path, app_name, model="gemini-2.5-flash", backend="ai_studio"` |
| Inspect an app (root_agent?, .py files, .env keys) | `project_inspect` | `path` |
| Set/merge `.env` values (idempotent, redacted) | `project_set_env` | `path, values: dict` |
| Add a `google-adk[extra]` dependency | `project_add_extra` | `path, extra` (gcp/bigquery/spanner/a2a/eval/mcp/community/litellm) |
| Write/inspect the no-code Agent Config `root_agent.yaml` | `project_agent_config` | `path, yaml_content=None` |

## <a name="agents"></a>agents — compose the graph (10)
| I want to… | Tool | Key args |
|---|---|---|
| Add/update an LlmAgent | `agents_create_llm` | `name, model="gemini-2.5-flash", instruction="", description="", output_key=None` |
| Add a Sequential pipeline | `agents_create_sequential` | `name, sub_agents: list[str], description=""` |
| Add a Parallel fan-out | `agents_create_parallel` | `name, sub_agents: list[str], description=""` |
| Add a Loop (repeat until N) | `agents_create_loop` | `name, sub_agents: list[str], max_iterations=3, description=""` |
| Add a custom BaseAgent stub | `agents_create_custom` | `name, description=""` |
| Replace an agent's sub_agents | `agents_compose` | `name, sub_agents: list[str]` |
| Set the app's root agent | `agents_set_root` | `name` |
| Get the `AgentTool(...)` source snippet (no file change) | `agents_as_tool` | `agent_name` |
| List agents (name, type, root) | `agents_list` | `(path, app_name)` |
| Get one agent's full spec | `agents_get` | `name` |

## <a name="tools"></a>tools — attach tools to an LlmAgent (13)
All take `(path, app_name, agent_name, …)`. Only LlmAgents carry tools.
| I want to… | Tool | Key args |
|---|---|---|
| Add a Python function tool | `tools_add_function` | `func_name, params: list[{name,type,default}], docstring, returns="dict", body="return {}"` |
| Add a long-running function tool | `tools_add_long_running` | same as add_function |
| Add a builtin (google_search, url_context, …) | `tools_add_builtin` | `kind, args=None` (args only for `vertex_ai_search` → `data_store_id`/`search_engine_id`) |
| Wrap another agent as a tool | `tools_add_agent_tool` | `target_agent` |
| Add an OpenAPI toolset (from a spec string) | `tools_add_openapi` | `spec, name=None` |
| Add a BigQuery toolset | `tools_add_bigquery` | `name=None, args=None` (source exprs) |
| Add a Spanner toolset | `tools_add_spanner` | `name=None, args=None` (source exprs) |
| Add an MCP toolset (consume another MCP server) | `tools_add_mcp_toolset` | `transport(stdio/sse/http), command=None, args=None, url=None, headers=None, tool_filter=None, name=None` |
| Add an API Hub toolset | `tools_add_apihub` | `apihub_resource_name, name=None` |
| Add a LangChain tool | `tools_add_langchain` | `import_line, tool_expr, name=None` |
| Add a CrewAI tool | `tools_add_crewai` | `import_line, tool_expr, name, description` |
| Attach auth to a toolset (openapi/apihub/mcp only) | `tools_set_auth` | `tool_name, scheme(apikey/oauth2/service_account/bearer), credential: dict` |
| List an agent's attached tools | `tools_list` | `agent_name` |

## <a name="models"></a>models — model & generation config (3)
All take `(path, app_name, agent_name, …)`; target an existing LlmAgent.
| I want to… | Tool | Key args |
|---|---|---|
| Set a native Gemini model (string) | `models_set` | `model` (e.g. "gemini-2.5-flash"); clears any LiteLlm |
| Use a non-Gemini provider via LiteLlm | `models_configure_litellm` | `provider, model, api_base="", api_key_env=""` |
| Set generation config + Gemini safety | `models_generate_config` | `temperature, max_output_tokens, top_p, top_k, safety_settings, response_modalities` (all optional; all None clears) |

## <a name="sessions"></a>sessions — runtime state service (8)
All async; take `(path, app_name, …)`.
| I want to… | Tool | Key args |
|---|---|---|
| Choose/persist the session backend | `sessions_service_set` | `kind(in_memory/database/vertex), db_url=None, project=None, location=None` |
| Create a session | `sessions_create` | `user_id, state=None, session_id=None` |
| Get a session (id, event_count, state) | `sessions_get` | `user_id, session_id` |
| List a user's session ids | `sessions_list` | `user_id` |
| Delete a session | `sessions_delete` | `user_id, session_id` |
| Set a state key (scoped, persisted) | `sessions_state_set` | `user_id, session_id, key, value, scope="session"` (session/app/user/temp) |
| Read a state key | `sessions_state_get` | `user_id, session_id, key, scope="session"` |
| Append a raw Event (text and/or state_delta) | `sessions_append_event` | `user_id, session_id, author, text=None, state_delta=None` |

## <a name="memory"></a>memory — long-term recall service (3)
All take `(path, app_name, …)`.
| I want to… | Tool | Key args |
|---|---|---|
| Choose/persist the memory backend | `memory_service_set` | `kind(in_memory/vertex_rag/vertex_memory_bank), project, location, rag_corpus, agent_engine_id` |
| Ingest a session into memory (async) | `memory_add_session` | `user_id, session_id` |
| Search memory (async; keyword for in_memory) | `memory_search` | `user_id, query` |

## <a name="artifacts"></a>artifacts — versioned blobs service (6)
All take `(path, app_name, …)`; save/load/etc. are async.
| I want to… | Tool | Key args |
|---|---|---|
| Choose/persist the artifact backend | `artifacts_service_set` | `kind(in_memory/gcs), bucket=None` |
| Save a text or binary artifact (→ version int) | `artifacts_save` | `user_id, session_id, filename, text=None, bytes_b64=None, mime_type="text/plain"` (exactly one of text/bytes_b64) |
| Load an artifact (latest or a version) | `artifacts_load` | `user_id, session_id, filename, version=None` |
| List artifact filenames | `artifacts_list` | `user_id, session_id` |
| Delete an artifact (all versions) | `artifacts_delete` | `user_id, session_id, filename` |
| List an artifact's versions | `artifacts_versions` | `user_id, session_id, filename` |

## <a name="run"></a>run — execute the agent loop (5)
| I want to… | Tool | Key args |
|---|---|---|
| Run the root agent on a message (async) | `run_agent` | `user_id, session_id, message, max_llm_calls=None, streaming_mode="NONE"` |
| Run with SSE progress reporting (async) | `run_stream` | `user_id, session_id, message, max_llm_calls=None` |
| Run Live/BIDI (experimental; needs creds) | `run_live` | `user_id, session_id, message, max_llm_calls=None` |
| Validate/describe a RunConfig (no run) | `run_config_build` | `streaming_mode="NONE", max_llm_calls=None, response_modalities=None` |
| Summarize a serialized event list (pure) | `run_inspect_events` | `events: list[dict]` |

## <a name="eval"></a>eval — evaluate the agent (4)
All take `(path, app_name, …)`; async.
| I want to… | Tool | Key args |
|---|---|---|
| Write a schema-conformant evalset | `eval_create_set` | `name, cases: list[{query, expected_response, expected_tool_use?}]` |
| Write offline criteria/thresholds | `eval_set_criteria` | `tool_trajectory_avg_score=1.0, response_match_score=0.8` |
| Run the evaluation + persist a report | `eval_run` | `eval_set_file, config_file=None, num_runs=1, agent_name=None` |
| Re-read a stored report | `eval_report` | `report_id` |

## <a name="deploy"></a>deploy — build/run `adk deploy …` (6)
`execute=False` (default) returns the plan; `execute=True` runs the real GCP deploy.
| I want to… | Tool | Key args |
|---|---|---|
| Preflight checks (gcloud/adk/kubectl) | `deploy_preflight` | `target="cloud_run"` |
| Deploy to Vertex AI Agent Engine | `deploy_agent_engine` | `project, region, staging_bucket=None, display_name=None, requirements_file=None, execute=False` |
| Deploy to Cloud Run | `deploy_cloud_run` | `project, region, service_name=None, with_ui=False, enable_cloud_trace=False, execute=False` |
| Deploy to GKE | `deploy_gke` | `project, region, cluster, service_name=None, execute=False` |
| Write a Dockerfile (serves `adk api_server`) | `deploy_containerize` | `(path, app_name)` |
| Best-effort deployment status | `deploy_status` | `target, project=None, region=None, service_name=None, cluster=None` |

## <a name="dev"></a>dev — managed local CLI servers (6)
| I want to… | Tool | Key args |
|---|---|---|
| Start `adk web` (dev UI + Eval/Trace tabs) | `dev_web` | `app_name=None, port=8000, host="127.0.0.1"` |
| Start `adk api_server` (FastAPI, no UI) | `dev_api_server` | `app_name=None, port=8000, host="127.0.0.1"` |
| One-shot `adk run AGENT "<message>"` | `dev_run` | `app_name, message=None` |
| Stop a managed server | `dev_stop` | `key` |
| Status of a managed server | `dev_status` | `key` |
| Tail a managed server's logs | `dev_logs` | `key, tail=50` |

## <a name="a2a"></a>a2a — Agent-to-Agent (3)
| I want to… | Tool | Key args |
|---|---|---|
| Consume a remote A2A agent (RemoteA2aAgent proxy) | `a2a_consume` | `name, agent_card_url` |
| Expose my agent over A2A (write a2a_app.py; optionally serve) | `a2a_expose` | `port=8001, execute=False` |
| Build/inspect the AgentCard (async; needs a2a extra) | `a2a_agent_card` | `port=8001` |

## <a name="mcp_bridge"></a>mcp_bridge — expose ADK tools as MCP (2)
| I want to… | Tool | Key args |
|---|---|---|
| Convert a core builtin to an MCP tool schema | `mcp_bridge_convert_builtin` | `kind` (core builtins only; not `vertex_ai_search`) |
| Convert an agent's tools to MCP tool schemas (async) | `mcp_bridge_expose_adk_tools` | `(path, app_name, agent_name)` |

## <a name="safety"></a>safety — guardrails (3)
| I want to… | Tool | Key args |
|---|---|---|
| Attach a per-agent guardrail callback | `safety_add_callback` | `agent_name, hook(before_model/after_model/before_tool/after_tool/before_agent/after_agent), policy: {kind: block_keywords/max_input_chars/block_tool, …}` |
| Add a global plugin (logging / tool denylist) | `safety_add_plugin` | `name, kind(logging/tool_denylist), config=None` |
| Set Gemini safety thresholds + LLM call budget | `safety_settings` | `agent_name, max_llm_calls=None, gemini_safety: list[{category, threshold}]` |

## <a name="observability"></a>observability — tracing (4)
| I want to… | Tool | Key args |
|---|---|---|
| Generate `otel_setup.py` (console/OTLP exporter) | `observability_enable_otel` | `exporter="console", endpoint=None` |
| Find the Cloud Trace flag + the tool that applies it | `observability_cloud_trace` | `target` (cloud_run/agent_engine/gke/web/api_server) |
| Emit OTLP config for a third-party backend | `observability_third_party` | `provider(phoenix/arize/weave/signoz/otlp), endpoint=None, headers=None` |
| Open the `adk web` Trace UI (delegates to dev_web; async) | `observability_trace_view` | `app_name=None, port=8000` |

## <a name="resources"></a>Resources (not tools)
| Resource URI | Returns |
|---|---|
| `adk://version` | The pinned google-adk / fastmcp / Python versions. |
| `adk://models` | Common Gemini model strings. |

## <a name="count"></a>Count check
5 (project) + 10 (agents) + 13 (tools) + 3 (models) + 8 (sessions) + 3 (memory) + 6 (artifacts) +
5 (run) + 4 (eval) + 6 (deploy) + 6 (dev) + 3 (a2a) + 2 (mcp_bridge) + 3 (safety) + 4 (observability)
= **81 tools** across 17 domains. (15 domains expose tools; `project_model`/`runtime` are internal
support modules, not exposed.)
