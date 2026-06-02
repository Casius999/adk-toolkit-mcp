# Changelog

All notable changes to **adk-toolkit-mcp** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Nothing yet.

## [0.1.0] - 2026-06-01

Initial public release: an exhaustive MCP server wrapping the entire **Google ADK**
(`google-adk` 2.1.0) surface, plus a companion Claude Code skill.

### Added

- **81 MCP tools across 15 tool-exposing FastMCP sub-servers** (2 additional internal
  support modules: `project_model`, `runtime`), covering the full agent lifecycle:
  - `project` (5) — scaffold an app, inspect, `.env`, extras, Agent Config YAML.
  - `agents` (10) — `LlmAgent`, Sequential / Parallel / Loop pipelines, custom, compose, root.
  - `tools` (13) — function, long-running, builtin, AgentTool, OpenAPI, BigQuery, Spanner,
    MCP toolset, APIHub, LangChain, CrewAI, auth, list.
  - `models` (3) — Gemini model, LiteLlm (any OpenAI-compatible provider),
    `GenerateContentConfig` + safety settings.
  - `sessions` (8) — session backend, create / get / list / delete, state set/get, append event.
  - `memory` (3) — memory backend, ingest session, search (keyword or Vertex RAG).
  - `artifacts` (6) — artifact backend, save/load (versioned), list, delete, list versions.
  - `run` (5) — execute agent (sync / SSE / live), build `RunConfig`, inspect events.
  - `eval` (4) — create evalset, set offline criteria, run evaluation, read report.
  - `deploy` (6) — Agent Engine, Cloud Run, GKE, Dockerfile, preflight, status.
  - `dev` (6) — `adk web`, `adk api_server`, one-shot run, stop / status / logs.
  - `a2a` (3) — consume a remote A2A agent, expose over A2A, build an `AgentCard`.
  - `mcp_bridge` (2) — convert ADK tools to MCP schemas.
  - `safety` (3) — per-agent callback guardrails, global plugins, Gemini safety + LLM-call budget.
  - `observability` (4) — OpenTelemetry setup, Cloud Trace flag, third-party OTLP, trace view.
- **Code-first sidecar authoring.** `.adk_toolkit/agents.json` is the source of truth;
  `agent.py` is regenerated wholesale on every change. Generated code is held to
  `ast.parse` + `ruff format` + isort.
- **Uniform `{ok, data, error}` envelope** on every tool; an actionable `err(...)` for any
  missing optional dependency (`gcp` / `db` / `eval` / `a2a` / `bigquery` / `spanner` /
  `mcp` / `community`), never a raised exception or a hang.
- **2 MCP resources**: `adk://version`, `adk://models`.
- **5 workflow prompts**: `scaffold_multi_agent`, `add_guardrail`, `write_evalset`,
  `deploy_checklist`, `debug_agent`.
- **Opt-in Code Mode** (`ADK_TOOLKIT_CODE_MODE=1` or `build_server(code_mode=True)`):
  collapses the 81-tool catalog to a 4-tool discovery surface
  (`search` / `get_schema` / `tags` / `execute`) using FastMCP 3.3.1's real `CodeMode`
  transform. All tools are tagged by domain.
- **Companion `adk-toolkit` skill** (`skill/`): a routing index plus 14 reference files
  covering every domain, ADK gotchas, and troubleshooting.
- **Documentation**: `docs/ARCHITECTURE.md`, `docs/TOOL_CATALOG.md`, `docs/CONTRIBUTING.md`,
  and per-domain ADK introspection notes under `docs/adk-api-notes/`.
- **Runnable examples** under `examples/` (`01_hello_agent.py`, `02_multi_agent.py`,
  `03_eval.py`).
- **MCP-registry-ready** `server.json` manifest.

### Tested

- **669 unit tests + 1 gated live end-to-end test**, ~95% coverage, green under
  `-W error::DeprecationWarning`.
- **Proven live E2E**: a real Kimi K2.6 model (via NVIDIA NIM, OpenAI-compatible through
  LiteLLM) driven through the mounted MCP server end to end (`project_create` →
  `agents_create_llm` → `agents_set_root` → `models_configure_litellm` → `run_agent`)
  returns a real response. The test is CI-safe: it discovers the key from a gitignored
  `.env` and skips cleanly when absent.
- Offline agent-loop, eval, safety, and plugin behaviour is proven with a fake LLM
  (no API key required).

[Unreleased]: https://github.com/Casius999/adk-toolkit-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Casius999/adk-toolkit-mcp/releases/tag/v0.1.0
