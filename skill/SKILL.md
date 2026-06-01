---
name: adk-toolkit
description: >-
  Complete mastery of Google ADK (Agent Development Kit, google-adk 2.1.0) driven through the
  adk-toolkit-mcp MCP server. Use this skill WHENEVER the user wants to build, scaffold, wire,
  run, evaluate, deploy, or debug a Google ADK agent — even if they don't name ADK explicitly.
  Triggers include: "build/scaffold a Google ADK agent", "google-adk", "agent development kit",
  "LlmAgent", "SequentialAgent / ParallelAgent / LoopAgent", "multi-agent / sub_agents", "AgentTool",
  "FunctionTool / long-running tool / OpenAPI / BigQuery / Spanner / MCP toolset / APIHub / LangChain /
  CrewAI tool", "LiteLlm / Anthropic / OpenAI / Ollama / LM Studio model", "session state / memory /
  artifacts", "app: / user: / temp: state prefix", "Runner / run_async / RunConfig / streaming",
  "evalset / AgentEvaluator / tool_trajectory / response_match", "deploy to Agent Engine / Cloud Run /
  GKE / containerize", "A2A / RemoteA2aAgent / to_a2a / AgentCard", "guardrail / callback / plugin /
  safety settings / max_llm_calls", "OpenTelemetry / Cloud Trace / Phoenix / Arize", "adk web / adk
  api_server", or any mention of the adk-toolkit-mcp server. This skill teaches the ADK craft AND maps
  every task to the exact MCP tool to call, so you never forget a step or hit a known pitfall.
---

# adk-toolkit — Google ADK mastery via adk-toolkit-mcp

This skill confers complete, accurate mastery of **Google ADK 2.1.0** as exposed by the
**adk-toolkit-mcp** MCP server (81 tools across 17 domains). The server is a **code-first sidecar**:
you author agents by calling MCP tools that maintain a sidecar (`.adk_toolkit/agents.json` +
`runtime.json`) and **regenerate `agent.py` wholesale** from it. Never hand-edit generated `agent.py`
— call a tool and it is re-rendered. Generated code is held to `ast.parse` + `ruff format` + isort.

## ADK mental model (read this first)

- **Code-first.** An ADK app is a Python package folder `<app_name>/` with `__init__.py` + `agent.py`
  exposing a top-level **`root_agent`**. The toolkit owns `agent.py` (regenerated from the sidecar).
- **Build → run → eval → deploy.** Scaffold a project, add agents/tools/models, wire runtime services
  (sessions/memory/artifacts), run the agent loop via a `Runner`, evaluate against an evalset, then
  deploy (Agent Engine / Cloud Run / GKE / container).
- **Runtime services are real and async**, chosen per-app and persisted in `runtime.json`
  (in_memory by default). The toolkit instantiates the real ADK service objects and calls them.
- **Optional deps are codegen-only or lazy.** The toolkit emits code that imports an extra
  (`bigquery`/`spanner`/`gcp`/`a2a`/`eval`/`community`) only in the *generated* app — your own venv
  needs the extra to RUN, not to author. A missing extra yields an actionable error, never a crash.
- **Every tool returns `{ok, data, error}`.** `ok=False` means a clean, actionable failure (never an
  exception, never a hang). An eval that fails its thresholds is `ok=True, passed=False` (normal).

## Routing index — task → reference + tool prefix

Load the reference file for the dimension you are working on. Detail lives in references (progressive
disclosure); this body is just the map. **`references/13-tool-catalog.md` is the authoritative
"I want to do X → call this exact tool" bridge covering all 81 tools — open it whenever unsure.**

| Dimension / task | Reference | Tool prefix |
|---|---|---|
| What ADK is, lifecycle, sidecar model, when to use ADK | `references/00-mental-model.md` | — |
| Scaffold a project; `.env`; extras; backend (AI Studio vs Vertex); no-code Agent Config | `references/00-mental-model.md` | `project_*` |
| Choose an agent type; sub_agents vs AgentTool vs RemoteA2aAgent; Loop/Parallel; set root | `references/01-agent-types.md` | `agents_*` |
| Attach tools (function, long-running, builtin, agent-tool, OpenAPI, BigQuery, Spanner, MCP, APIHub, LangChain, CrewAI) + auth | `references/02-tools.md` | `tools_*` |
| Pick a model: Gemini string vs LiteLlm (Anthropic/OpenAI/Ollama/LM Studio); generate_content_config + safety | `references/03-models.md` | `models_*` |
| Session service backend; state + `app:`/`user:`/`temp:` prefixes; append_event | `references/04-sessions-state.md` | `sessions_*` |
| Memory (ingest + keyword search) vs Artifacts (versioned Parts); when each | `references/05-memory-artifacts.md` | `memory_*`, `artifacts_*` |
| Run the agent loop; Runner; RunConfig (streaming NONE/SSE/BIDI, max_llm_calls); inspect events | `references/06-runtime.md` | `run_*` |
| Build an evalset; criteria/thresholds; offline metrics vs LLM-judge; run + read report | `references/07-eval.md` | `eval_*`; also `dev_*` for `adk web` eval UI |
| Deploy: target choice (Agent Engine/Cloud Run/GKE/container); real 2.1.0 flags; preflight; dev server | `references/08-deploy.md` | `deploy_*`, `dev_*` |
| A2A: expose (`to_a2a`, AgentCard) vs consume (`RemoteA2aAgent`); MCP bridge | `references/09-a2a.md` | `a2a_*`, `mcp_bridge_*` |
| Observability: OTel (console/OTLP), Cloud Trace, third-party (Phoenix/Arize/Weave/SigNoz), trace view | `references/10-observability.md` | `observability_*` |
| Guardrails: callbacks (per-agent, short-circuit) vs plugins (global); safety settings + max_llm_calls | `references/11-safety.md` | `safety_*` |
| Known pitfalls + fixes (deprecations, missing flags/imports/extras, DB async URL, regen) | `references/12-troubleshooting.md` | — |
| **Complete task → exact MCP tool(s) map (all 81 tools, by domain)** | `references/13-tool-catalog.md` | **all** |

## The 17 domains at a glance

`project` (scaffold/inspect) · `agents` (compose) · `tools` (attach tools) · `models` (model/config) ·
`sessions` · `memory` · `artifacts` (runtime state) · `run` (execute) · `eval` (evaluate) ·
`deploy` · `dev` (CLI servers) · `a2a` · `mcp_bridge` (interop) · `safety` · `observability` (ops).

Exposed tool names are always `<domain>_<bare>` (e.g. `agents_create_llm`, `tools_add_mcp_toolset`,
`run_agent`, `eval_run`, `deploy_cloud_run`). The MCP server is `adk-toolkit` (run via
`uv run adk-toolkit-mcp`). Two resources: `adk://version`, `adk://models`.

## Golden workflow (the path that never forgets a step)

1. `project_create(path, app_name, model, backend)` — scaffold `<app_name>/` with a `root_agent`.
2. `agents_create_llm / create_sequential / …` then `agents_set_root` — build the agent graph.
3. `tools_add_*` — attach tools to LlmAgents (only LlmAgents carry tools).
4. `models_set / configure_litellm / generate_config` — pick model + sampling/safety.
5. `sessions_service_set` (+ `memory_service_set` / `artifacts_service_set` if needed) — wire runtime.
6. `run_agent` (or `run_stream`) — execute and inspect events. Offline-testable via a fake LLM.
7. `eval_create_set` → `eval_set_criteria` → `eval_run` → `eval_report` — measure quality.
8. `deploy_preflight` → `deploy_cloud_run` (or `agent_engine`/`gke`/`containerize`) — ship it.

Add `safety_*` (guardrails), `a2a_*` (expose/consume), `observability_*` (tracing) as the task needs.
When in doubt about which tool implements a step, consult `references/13-tool-catalog.md`.
