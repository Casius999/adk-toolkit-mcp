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
| **P2 State** | sessions, memory, artifacts | ⏳ NEXT |
| **P3 Runtime/Eval** | run, eval | ⬜ |
| **P4 Ops** | deploy, a2a, observability, safety, mcp_bridge, dev | ⬜ |
| **P5 Skill** | `adk-toolkit` skill (SKILL.md + 14 refs) | ⬜ |
| **P6 Finish** | Code Mode, prompts, docs, repo publish (confirm GitHub) | ⬜ |

## Exposed tools so far (31)
- `project_*`: create, inspect, set_env, add_extra, agent_config
- `agents_*`: create_llm, create_sequential, create_parallel, create_loop, create_custom, compose, as_tool, set_root, list, get
- `tools_*`: add_function, add_long_running, add_builtin, add_agent_tool, add_openapi, add_bigquery, add_spanner, add_mcp_toolset, add_apihub, add_langchain, add_crewai, set_auth, list
- `models_*`: set, configure_litellm, generate_config
- Resources: `adk://version`, `adk://models`

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
267 passed, 1 skipped (litellm probe), coverage ~97%. ruff + mypy clean. `project_model/` is a package (specs/sidecar/render/_codegen), all files <800 lines.

## Resume instructions
Next: P2. Build a shared `src/adk_toolkit_mcp/runtime.py` (ADK service factory + process-singleton cache keyed by backend config; persists backend config in `.adk_toolkit/runtime.json`), then the `sessions`/`memory`/`artifacts` domains (async tools that instantiate the configured ADK service and operate on it). Functional acceptance: DatabaseSessionService on a SQLite tmp file round-trips state across separate tool calls; state prefixes `app:`/`user:`/`temp:` respected. Then combined review. Update this file after each phase.
