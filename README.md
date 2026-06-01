# adk-toolkit-mcp

Exhaustive MCP server wrapping **Google ADK** (`google-adk` 2.1.0): 15 sub-servers, **81 tools**
across 14 exposed domains, covering the full agent development lifecycle — scaffold, compose,
run, evaluate, deploy, observe. Built code-first: the sidecar `.adk_toolkit/agents.json` is the
source of truth; `agent.py` is fully regenerated on every change.

> **Standalone project.** No link to any sibling project. All dependencies are declared in
> `pyproject.toml`; nothing is borrowed from local paths.

---

## Contents

- [Install](#install)
- [Run](#run)
- [MCP client config](#mcp-client-config-claude-code)
- [Code Mode (opt-in)](#code-mode-opt-in)
- [Companion skill](#companion-skill)
- [Quickstart](#quickstart)
- [Domains](#domains)
- [Workflow prompts](#workflow-prompts)
- [Docs](#docs)

---

## Install

```bash
uv venv && uv sync --extra dev
```

Optional extras (install any subset; `all` installs everything):

| Extra | What it enables |
|---|---|
| `litellm` | Non-Gemini models via LiteLlm (OpenAI, Anthropic, Ollama, …) |
| `gcp` | Vertex AI session/memory/artifact backends |
| `bigquery` | BigQuery toolsets |
| `spanner` | Spanner toolsets |
| `a2a` | Agent-to-Agent (A2A) SDK — expose/consume A2A agents |
| `eval` | Offline evaluation metrics (ROUGE + trajectory; `google-adk[eval]`) |
| `mcp` | MCP toolset support in generated agents |
| `community` | Community integrations (`google-adk[community]`) |
| `db` | DatabaseSessionService via SQLAlchemy |
| `all` | All of the above |

```bash
uv sync --extra all
```

---

## Run

```bash
uv run adk-toolkit-mcp
```

Runs on stdio (the MCP standard transport). The server exposes all 81 tools immediately on
startup; no configuration file is required.

---

## MCP client config (Claude Code)

Add to your `claude_desktop_config.json` (or `settings.json` `mcpServers` block):

```json
{
  "mcpServers": {
    "adk-toolkit": {
      "command": "uv",
      "args": ["run", "adk-toolkit-mcp"],
      "cwd": "/absolute/path/to/adk-toolkit-mcp"
    }
  }
}
```

---

## Code Mode (opt-in)

By default all 81 tools are exposed by name (`project_create`, `run_agent`, …). For
token-heavy contexts you can collapse the catalog to a 4-tool discovery surface using FastMCP's
real `CodeMode` transform (fastmcp 3.3.1):

```bash
ADK_TOOLKIT_CODE_MODE=1 uv run adk-toolkit-mcp
```

Or in code:

```python
from adk_toolkit_mcp.server import build_server
mcp = build_server(code_mode=True)
```

The surface becomes `search` / `get_schema` / `tags` / `execute`. All 81 tools are tagged by
domain (`agents`, `deploy`, …), so a client can do `tags()` → `search(tags=["run"])` →
`get_schema(tools=["run_agent"])` → `execute(...)`.

**Note on `execute`:** the discovery tools (`search`/`get_schema`/`tags`) work with no extra
dependencies. The `execute` sandbox (`MontySandboxProvider`) requires `pydantic-monty`
(`fastmcp[code-mode]`), which is **not installed by default**; calling `execute` without it
raises a clear `ImportError`. Install if you need server-side code execution:

```bash
uv pip install 'fastmcp[code-mode]'
```

See `docs/adk-api-notes/fastmcp-codemode.md` for details.

---

## Companion skill

The `skill/` directory contains an `adk-toolkit` Claude Code skill — a routing index with 14
reference files covering every domain, ADK gotchas, and troubleshooting. Install it:

```bash
cp -r skill/. ~/.claude/skills/adk-toolkit/
```

After install it appears in the harness `available-skills` list and can be invoked as
`/adk-toolkit` in Claude Code sessions.

---

## Quickstart

### 1. Scaffold an app

```
project_create(path="/my/projects", app_name="greeter", model="gemini-2.5-flash", backend="ai_studio")
project_set_env(path="/my/projects", values={"GOOGLE_API_KEY": "your-key"})
```

### 2. Add an agent with a tool

```
agents_create_llm(path="/my/projects", app_name="greeter", name="root_agent",
                  model="gemini-2.5-flash", instruction="Greet the user.")
tools_add_function(path="/my/projects", app_name="greeter", agent_name="root_agent",
                   func_name="get_greeting", params=[{"name": "name", "type": "str"}],
                   docstring="Return a greeting.", body='return {"greeting": f"Hello, {name}!"}')
agents_set_root(path="/my/projects", app_name="greeter", name="root_agent")
```

### 3. Run locally

```
dev_web(path="/my/projects", app_name="greeter", port=8000)
```

Or execute a single turn:

```
run_agent(path="/my/projects", app_name="greeter", user_id="u1", session_id="s1",
          message="Greet Alice")
```

---

## Domains

| Domain | Tools | What it covers |
|---|---|---|
| `project` | 5 | Scaffold app, inspect, `.env`, extras, agent config YAML |
| `agents` | 10 | LlmAgent, Sequential/Parallel/Loop pipeline, custom, compose, root |
| `tools` | 13 | Function, long-running, builtins, AgentTool, OpenAPI, BigQuery, Spanner, MCP, APIHub, LangChain, CrewAI, auth, list |
| `models` | 3 | Gemini model, LiteLlm (any provider), GenerateContentConfig + safety |
| `sessions` | 8 | Session backend, create/get/list/delete, state set/get, append event |
| `memory` | 3 | Memory backend, ingest session, search (keyword or Vertex RAG) |
| `artifacts` | 6 | Artifact backend, save/load (versioned), list, delete, list versions |
| `run` | 5 | Execute agent (sync/SSE/live), build RunConfig, inspect events |
| `eval` | 4 | Create evalset, set offline criteria, run evaluation, read report |
| `deploy` | 6 | Agent Engine, Cloud Run, GKE, Dockerfile, preflight, status |
| `dev` | 6 | `adk web`, `adk api_server`, one-shot run, stop/status/logs |
| `a2a` | 3 | Consume remote A2A agent, expose over A2A, build AgentCard |
| `mcp_bridge` | 2 | Convert ADK tools to MCP schemas |
| `safety` | 3 | Per-agent callback guardrails, global plugins, Gemini safety + LLM call budget |
| `observability` | 4 | OTel setup, Cloud Trace flag, third-party OTLP, trace view |

---

## Workflow prompts

Five MCP prompts (`get_prompt`) scaffold common multi-step tasks by citing the exact tool call
sequence:

| Prompt | Arg(s) | Covers |
|---|---|---|
| `scaffold_multi_agent` | `goal` | `project_create` → agent graph → `models_set` → `run_agent` |
| `add_guardrail` | `agent`, `concern` | per-agent `safety_add_callback` vs global `safety_add_plugin` + `safety_settings` |
| `write_evalset` | `agent` | `eval_create_set` → `eval_set_criteria` → `eval_run` → `eval_report` |
| `deploy_checklist` | `target` | `deploy_preflight` → `deploy_containerize` → `deploy_agent_engine`/`cloud_run`/`gke` → `deploy_status` |
| `debug_agent` | `symptom` | `run_inspect_events`, `run_stream`, `agents_get`/`list`, `tools_list` + known pitfalls |

---

## Docs

| File | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Root server + sub-server mount pattern, code-first sidecar model, `project_model`, `runtime`, `run_core`, `adk_cli`, envelope, Code Mode |
| `docs/TOOL_CATALOG.md` | All 81 tools grouped by domain, with purpose and key parameters |
| `docs/CONTRIBUTING.md` | Dev setup, conventions, how to add a new domain |
| `docs/adk-api-notes/` | Per-domain ADK API introspection notes (ground truth for the implementation) |
