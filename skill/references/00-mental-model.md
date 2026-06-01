# 00 — ADK mental model + the project domain

What Google ADK is, when to reach for it, the build→run→eval→deploy lifecycle, and how the
adk-toolkit-mcp **code-first sidecar** model works. Maps to the `project_*` tools.

## What ADK is

The **Agent Development Kit** (`google-adk`, this skill targets **2.1.0**) is Google's **code-first**
framework for building LLM agents. An ADK app is a Python package, not a YAML config or a hosted
no-code flow (though a no-code "Agent Config" path exists — see below). The core objects:

- **`root_agent`** — a module-level variable in `<app_name>/agent.py` that is the entry point. Every
  ADK app exposes exactly one `root_agent` (an `LlmAgent`, a workflow agent, a custom `BaseAgent`, or
  a `RemoteA2aAgent`).
- **`Runner`** — the execution engine. It wires an agent to **services** (session/memory/artifact) and
  drives the async agent loop (`run_async` yields `Event`s).
- **Services** — pluggable backends for conversation **state** (sessions), long-term **memory**, and
  binary **artifacts**. In-memory by default; production backends are Vertex / GCS / a database.

### When to use ADK

Use ADK when you need a **structured, deployable** agent: multi-agent orchestration, tool calling,
session state, evaluation, and first-class deployment to Vertex AI Agent Engine / Cloud Run / GKE.
For a one-off prompt with no tools/state/deploy story, plain genai is simpler. ADK shines when the
agent is a real application with a build→run→eval→deploy lifecycle.

## The lifecycle (and the golden workflow)

```
project_create  →  agents_create_* + set_root  →  tools_add_*  →  models_*
        →  sessions/memory/artifacts service_set  →  run_agent
        →  eval_create_set → set_criteria → run → report  →  deploy_*
```

Each arrow is one or more MCP tool calls. The lifecycle is the spine of every reference file; the
exact tool for each step is in `13-tool-catalog.md`.

## The code-first sidecar model (how this toolkit authors code)

This is the single most important thing to understand about the toolkit:

- **Sidecar = source of truth.** Authoring tools (`agents_*`, `tools_*`, `models_*`, `safety_*`,
  `a2a_consume`) do **not** edit `agent.py` directly. They load a JSON **sidecar**
  `<path>/<app_name>/.adk_toolkit/agents.json`, apply an **immutable** mutation, save it, then
  **regenerate `agent.py` (+ `__init__.py`) wholesale** from the sidecar.
- **Never hand-edit `agent.py`.** Your edits would be clobbered on the next tool call. To change the
  app, call a tool. The generated file always passes `ast.parse`, `ruff format --check`, and
  `ruff check --select I` (isort) — it is clean, deterministic, topologically ordered code.
- **Runtime config sidecar.** Service backends (sessions/memory/artifacts) + a plugins manifest live
  in a second sidecar `<path>/<app_name>/.adk_toolkit/runtime.json`. The `*_service_set` tools write
  it; `run_*`/`sessions_*`/`memory_*`/`artifacts_*` read it. In-memory services are process-singletons
  (state survives across tool calls in one server process).
- **Idempotent file writes.** Re-running the same tool with the same inputs reports `changed: false`
  and rewrites nothing. Most authoring tools return `{app_name, agents, root, sidecar, regenerated,
  changed}`.

## The `project` domain — scaffold & inspect

| Tool | Purpose |
|---|---|
| `project_create(path, app_name, model="gemini-2.5-flash", backend="ai_studio")` | Scaffold `<path>/<app_name>/` with `__init__.py`, `agent.py` (a `root_agent = LlmAgent(...)`), and a backend-appropriate `.env`. Mirrors `adk create`. |
| `project_inspect(path)` | Report `has_root_agent`, `py_files`, and `.env` **key names** (values never returned). |
| `project_set_env(path, values)` | Merge `values` into `.env` (idempotent; never overwrites the rest). Returns redacted keys. |
| `project_add_extra(path, extra)` | Add a `google-adk[<extra>]` dependency to `pyproject.toml` (else `requirements.txt`). |
| `project_agent_config(path, yaml_content=None)` | The no-code **Agent Config** path: write/inspect `root_agent.yaml`. |

### Backend: AI Studio vs Vertex (decision)

- **`backend="ai_studio"`** → `.env` has `GOOGLE_GENAI_USE_VERTEXAI=FALSE` + `GOOGLE_API_KEY=`. Use a
  Google AI Studio API key. Simplest for local/dev.
- **`backend="vertex"`** → `.env` has `GOOGLE_GENAI_USE_VERTEXAI=TRUE` + `GOOGLE_CLOUD_PROJECT=` +
  `GOOGLE_CLOUD_LOCATION=`. Use Google Cloud credentials. Required for Vertex-only features (Vertex
  memory/RAG, Agent Engine). Fill the blanks with `project_set_env`.

> The toolkit writes `FALSE`/`TRUE` (readable) where raw `adk create` writes `0`/`1`; ADK accepts both.
> Secrets are **never** written to generated code or returned by tools — `.env` values are redacted.

### `app_name` rule

`app_name` is a **Python identifier** (it is both the folder name and the package/module name): letters,
digits, underscore; cannot start with a digit. Invalid names yield a clean `err`.

### Known extras (for `project_add_extra`)

`gcp`, `bigquery`, `spanner`, `a2a`, `eval`, `mcp`, `community`, `litellm`. Each unlocks a family of
tools/backends — see the relevant reference. Unknown extras are rejected with the known list.

## Next steps

- Choosing the agent graph → `01-agent-types.md`.
- The complete tool map → `13-tool-catalog.md`.
