# adk-toolkit-mcp

[![CI](https://github.com/__OWNER__/adk-toolkit-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/__OWNER__/adk-toolkit-mcp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-purple.svg)](https://modelcontextprotocol.io/)
[![tests](https://img.shields.io/badge/tests-670%20passing-brightgreen.svg)](#testing--quality)
[![coverage](https://img.shields.io/badge/coverage-~95%25-brightgreen.svg)](#testing--quality)

> The most exhaustive **MCP server for Google ADK** (Agent Development Kit, `google-adk` 2.x):
> **81 tools across 15 domains** to scaffold, compose, run, evaluate, deploy, and observe
> ADK agents — driven by any MCP client (Claude Code, etc.), **code-first** and deploy-ready.

`adk-toolkit-mcp` turns the entire Google ADK surface into Model Context Protocol tools. An
agent (or you) can build a complete, runnable, deployable ADK project end to end without
leaving the MCP client — and the output is **real ADK Python you own**, not a black box.

### Highlights

- **Full ADK surface, 81 tools / 15 domains** — agents, tools, models, sessions, memory,
  artifacts, runtime, evaluation, deployment, A2A, observability, safety.
- **Code-first.** A sidecar `.adk_toolkit/agents.json` is the source of truth; `agent.py` is
  regenerated wholesale and is always `ast.parse` + `ruff format` + isort clean — commit it,
  deploy it, read it.
- **Verified live, end-to-end.** A gated integration test drives a *real* OpenAI-compatible
  model through the mounted server (`project_create → agents_create_llm →
  models_configure_litellm → run_agent`) and asserts a real response flows back. See
  [Verified end-to-end](#verified-end-to-end).
- **Token-efficient.** Opt-in [Code Mode](#code-mode-opt-in) collapses the 81-tool catalog to a
  4-tool discovery surface using FastMCP 3.3.1's real `CodeMode` transform.
- **Provider-agnostic models.** Gemini natively, plus Anthropic / OpenAI / Ollama / LM Studio /
  NVIDIA NIM / any OpenAI-compatible endpoint via LiteLLM.
- **Companion skill.** An `adk-toolkit` Claude skill (14 reference files) teaches the ADK craft
  and maps every task to the exact tool.
- **670 tests, ~95% coverage**, `ruff` + `mypy` clean, CI on Python 3.11 & 3.12.

> **Standalone project.** No link to any sibling project; all dependencies are declared in
> `pyproject.toml`.

---

## Contents

- [Install](#install) · [Run](#run) · [MCP client config](#mcp-client-config)
- [Verified end-to-end](#verified-end-to-end) · [Quickstart](#quickstart)
- [Code Mode](#code-mode-opt-in) · [Companion skill](#companion-skill)
- [Domains](#domains) · [Workflow prompts](#workflow-prompts)
- [Testing & quality](#testing--quality) · [Docs](#docs) · [License](#license)

---

## Install

**Run without cloning** (once published) via `uvx`:

```bash
uvx --from git+https://github.com/__OWNER__/adk-toolkit-mcp adk-toolkit-mcp
```

**From a clone (recommended for development):**

```bash
git clone https://github.com/__OWNER__/adk-toolkit-mcp
cd adk-toolkit-mcp
uv venv && uv sync --extra dev
```

A PyPI release (`pip install adk-toolkit-mcp`) is planned.

### Optional extras

Install any subset; `all` installs everything. Tools whose backend isn't installed return an
actionable `error` telling you which extra to add — they never crash the server.

| Extra | Enables |
|---|---|
| `litellm` | Non-Gemini models via LiteLlm (OpenAI, Anthropic, Ollama, LM Studio, NVIDIA NIM, …) |
| `gcp` | Vertex AI session/memory/artifact backends |
| `bigquery` / `spanner` | BigQuery / Spanner toolsets |
| `a2a` | Agent-to-Agent (A2A) — expose/consume A2A agents |
| `eval` | Offline evaluation metrics (ROUGE + tool trajectory) |
| `mcp` | MCP toolset support in generated agents |
| `community` | Community integrations (LangChain / CrewAI tools) |
| `db` | `DatabaseSessionService` via SQLAlchemy |
| `all` | All of the above |

```bash
uv sync --extra all
```

---

## Run

```bash
uv run adk-toolkit-mcp        # stdio transport; all 81 tools available on startup
```

## MCP client config

Add to your MCP client (e.g. Claude Code `settings.json` → `mcpServers`):

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

Or, without a clone:

```json
{
  "mcpServers": {
    "adk-toolkit": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/__OWNER__/adk-toolkit-mcp", "adk-toolkit-mcp"]
    }
  }
}
```

This server is also **MCP-registry-ready** — see [`server.json`](server.json).

---

## Verified end-to-end

The toolkit's job is the *plumbing*: scaffold an ADK project, wire a model, and run the agent
through ADK's real `Runner`, returning the model's response over MCP. This is proven by a
gated live integration test ([`tests/integration/test_e2e_kimi.py`](tests/integration/test_e2e_kimi.py))
that drives a **real OpenAI-compatible model** end to end through the *mounted* server:

```
project_create → agents_create_llm → agents_set_root → models_configure_litellm → run_agent
```

It asserts a real, non-empty model response flows back through the full MCP stack (it does not
police the third-party model's wording). It's CI-safe: it **skips** unless you provide a key
and opt in, so a normal test run never makes a paid network call.

```bash
# Put a key in a gitignored .env, e.g.:  NVIDIA_API_KEY=nvapi-...
ADK_TOOLKIT_TEST_LIVE=1 uv run pytest tests/integration/test_e2e_kimi.py -s
```

Point the same flow at **any** OpenAI-compatible endpoint — NVIDIA NIM, OpenAI, Anthropic,
Ollama, or LM Studio — via `models_configure_litellm(provider="openai", model=..., api_base=...,
api_key_env=...)`. The API key is read from the environment at run time and is **never** written
into the generated code (`api_key=os.getenv("...")`).

See [`examples/`](examples/) for runnable scripts (two are fully offline, no key required).

---

## Quickstart

```text
# 1. Scaffold
project_create(path="/proj", app_name="greeter", model="gemini-2.5-flash", backend="ai_studio")

# 2. Author an agent + a tool, set it as root
agents_create_llm(path="/proj", app_name="greeter", name="assistant",
                  instruction="Greet the user warmly.")
tools_add_function(path="/proj", app_name="greeter", agent_name="assistant",
                   func_name="get_greeting", params=[{"name": "name", "type": "str"}],
                   docstring="Return a greeting.", body='return {"greeting": f"Hello, {name}!"}')
agents_set_root(path="/proj", app_name="greeter", name="assistant")

# 3. Run one turn (Gemini needs GOOGLE_API_KEY; or wire any provider via models_configure_litellm)
run_agent(path="/proj", app_name="greeter", user_id="u1", session_id="s1", message="Greet Alice")
```

---

## Code Mode (opt-in)

By default all 81 tools are exposed by name. For token-heavy contexts, collapse the catalog to a
4-tool discovery surface (`search` / `get_schema` / `tags` / `execute`) using FastMCP 3.3.1's
real `CodeMode` transform:

```bash
ADK_TOOLKIT_CODE_MODE=1 uv run adk-toolkit-mcp     # or build_server(code_mode=True)
```

All tools are tagged by domain, so a client does `tags()` → `search(tags=["run"])` →
`get_schema(...)` → `execute(...)`. The discovery tools need no extra deps; the `execute` sandbox
requires `uv pip install 'fastmcp[code-mode]'` (documented in
[`docs/adk-api-notes/fastmcp-codemode.md`](docs/adk-api-notes/fastmcp-codemode.md)).

## Companion skill

[`skill/`](skill/) contains the `adk-toolkit` Claude skill — a routing index + 14 reference files
covering every domain, ADK gotchas, and a task→tool catalog. Install it:

```bash
cp -r skill/. ~/.claude/skills/adk-toolkit/
```

---

## Domains

| Domain | Tools | Covers |
|---|---|---|
| `project` | 5 | Scaffold app, inspect, `.env`, extras, agent config YAML |
| `agents` | 10 | LlmAgent, Sequential/Parallel/Loop pipeline, custom, compose, as-tool, root |
| `tools` | 13 | Function, long-running, builtins, AgentTool, OpenAPI, BigQuery, Spanner, MCP toolset, APIHub, LangChain, CrewAI, auth |
| `models` | 3 | Gemini model, LiteLlm (any provider), `GenerateContentConfig` + safety |
| `sessions` | 8 | Session backend, CRUD, state set/get (`app:`/`user:`/`temp:`), append event |
| `memory` | 3 | Memory backend, ingest session, search (keyword or Vertex RAG) |
| `artifacts` | 6 | Artifact backend, save/load (versioned), list, delete, versions |
| `run` | 5 | Execute agent (sync/SSE/live), build `RunConfig`, inspect events |
| `eval` | 4 | Create evalset, set offline criteria, run evaluation, read report |
| `deploy` | 6 | Agent Engine, Cloud Run, GKE, Dockerfile, preflight, status |
| `dev` | 6 | `adk web`, `adk api_server`, one-shot run, stop/status/logs |
| `a2a` | 3 | Consume remote A2A agent, expose over A2A, build AgentCard |
| `mcp_bridge` | 2 | Convert ADK tools to MCP schemas |
| `safety` | 3 | Per-agent callback guardrails, global plugins, Gemini safety + LLM-call budget |
| `observability` | 4 | OpenTelemetry setup, Cloud Trace flag, third-party OTLP, trace view |

Plus 2 resources (`adk://version`, `adk://models`) and 5 [workflow prompts](#workflow-prompts).
Full reference: [`docs/TOOL_CATALOG.md`](docs/TOOL_CATALOG.md).

## Workflow prompts

| Prompt | Args | Sequences |
|---|---|---|
| `scaffold_multi_agent` | `goal` | `project_create` → agent graph → models → `run_agent` |
| `add_guardrail` | `agent`, `concern` | `safety_add_callback` vs `safety_add_plugin` + `safety_settings` |
| `write_evalset` | `agent` | `eval_create_set` → `eval_set_criteria` → `eval_run` → `eval_report` |
| `deploy_checklist` | `target` | `deploy_preflight` → containerize → `deploy_*` → `deploy_status` |
| `debug_agent` | `symptom` | `run_inspect_events`, `run_stream`, `agents_get`/`list` + pitfalls |

---

## Testing & quality

- **670 tests** (`uv run pytest`) — unit + 1 gated live E2E — **~95% line coverage**.
- Green under `-W error::DeprecationWarning` (no deprecations slip through).
- `ruff` (lint + format) and `mypy` clean; the package ships `py.typed`.
- **CI** runs the full gate on Python **3.11 and 3.12** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- Generated `agent.py` is validated (`ast.parse` + `ruff format` + isort) before it lands.
- `uv build` produces a clean wheel + sdist.

```bash
uv run ruff check . && uv run mypy src && uv run pytest --cov
```

---

## Docs

| File | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Root server + sub-server mounts, code-first sidecar, `project_model`, `runtime`, `run_core`, `adk_cli`, envelope, Code Mode |
| [`docs/TOOL_CATALOG.md`](docs/TOOL_CATALOG.md) | All 81 tools by domain, with purpose and key parameters |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Dev setup, conventions, how to add a domain |
| [`docs/adk-api-notes/`](docs/adk-api-notes/) | Per-domain ADK API introspection notes (implementation ground truth) |
| [`CHANGELOG.md`](CHANGELOG.md) · [`SECURITY.md`](SECURITY.md) · [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Release notes · security policy · code of conduct |

---

## License

[Apache-2.0](LICENSE).

Built on [Google ADK](https://google.github.io/adk-docs/) and
[FastMCP](https://gofastmcp.com/). Not affiliated with Google.
