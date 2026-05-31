# ADK API notes — `project` domain

Captured by introspection on 2026-06-01. google-adk **2.1.0**, fastmcp **3.3.1**, Python 3.12.
These are observed facts (run against the installed packages), not guesses.

## `adk create` — real scaffolder

### CLI

```
Usage: adk create [OPTIONS] APP_NAME

  Creates a new app in the current folder with prepopulated agent template.
  APP_NAME: required, the folder of the agent source code.

Options:
  --model TEXT    Optional. The model used for the root agent.
  --api_key TEXT  Optional. The API Key needed to access the model (Google AI API Key).
  --project TEXT  Optional. The Google Cloud Project for using VertexAI as backend.
  --region TEXT   Optional. The Google Cloud Region for using VertexAI as backend.
  --help          Show this message and exit.
```

There is **no** `--backend` flag. Backend is implied:
- supply `--api_key` -> AI Studio backend (`GOOGLE_GENAI_USE_VERTEXAI=0`)
- supply `--project` + `--region` -> Vertex backend (`GOOGLE_GENAI_USE_VERTEXAI=1`)

Non-interactive when `--model` plus the backend-selecting flag(s) are passed
(`--api_key`, or `--project`/`--region`). Otherwise it prompts.

### Generated layout (exact)

`adk create demo_app --model gemini-2.5-flash --api_key TESTKEY123` creates `demo_app/`
containing exactly three files:

`__init__.py`
```python
from . import agent
```

`agent.py`
```python
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
```

`.env` (AI Studio / `--api_key`)
```
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=TESTKEY123
```

`.env` (Vertex / `--project` + `--region`)
```
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=my-gcp-proj
GOOGLE_CLOUD_LOCATION=us-central1
```

No trailing newline on the real `.env`. `agent.py`/`__init__.py` end with one newline.

### DELIBERATE DIVERGENCE in our `project_create`

The task spec mandates the boolean be spelled `FALSE` / `TRUE` (more readable, matches
older ADK docs). The real scaffolder emits `0` / `1`. Both are accepted by ADK at runtime
(`GOOGLE_GENAI_USE_VERTEXAI` is parsed truthily). We follow the task spec (`FALSE`/`TRUE`)
and keep `backend` as an explicit `Literal["ai_studio","vertex"]` arg instead of inferring
from which credential flags are present. Structure (`__init__.py` + `agent.py` with a
top-level `root_agent =` + `.env`) is identical to the real output.

We import the canonical name `LlmAgent` from `google.adk.agents` (the real template uses the
`Agent` alias from `google.adk.agents.llm_agent`; `Agent is LlmAgent` -> True). We keep the
`description` line for fidelity.

## `google.adk.agents.LlmAgent`

- Canonical import: `from google.adk.agents import LlmAgent`
- `google.adk.agents.Agent is LlmAgent` -> `True` (alias).
- Pydantic model; relevant fields present: `name`, `model`, `instruction`, `description`.
- Minimal construct works: `LlmAgent(name='root_agent', model='gemini-2.5-flash', instruction='...')`.

We only reference `LlmAgent` inside the *generated source string*, so the toolkit process
itself does not need to import google-adk to scaffold. (The generated app imports it at its
own runtime.)

## FastMCP mounting (fastmcp 3.3.1)

```
FastMCP.mount(self, server, namespace: str | None = None, as_proxy: bool | None = None,
              tool_names: dict[str,str] | None = None, prefix: str | None = None) -> None
```

- `prefix=` is **DEPRECATED**: passing it emits
  `DeprecationWarning: The 'prefix' parameter is deprecated, use 'namespace' instead`.
- Use `namespace="project"`. (We diverge from the literal `prefix=` in the task brief to
  avoid the deprecation warning and keep ruff/mypy/pytest clean — the brief explicitly
  allows adapting to the real signature.)
- Naming rule: a mounted tool `project_create` under namespace `project` is exposed as
  **`project_project_create`** (namespace + `_` + tool name).

## `Client.call_tool` return shape

```
Client.call_tool(name, arguments=None, *, ...) -> CallToolResult | ToolTask
```

- Returns a `CallToolResult`.
- `result.data` -> the deserialized return value. For our tools that is the envelope dict
  `{"ok": ..., "data": ..., "error": ...}`.
- `result.structured_content` -> same dict; `result.content` -> raw content blocks.
- Read-through assertion used in tests: `result.data["ok"] is True`.

## Tool naming convention (refactored 2026-06-01)

Tool functions in `domains/project.py` are named with **bare names** (no domain prefix):
`create`, `inspect`, `set_env`, `add_extra`, `agent_config`.

Mounted with `namespace="project"`, FastMCP exposes them as:
`project_create`, `project_inspect`, `project_set_env`, `project_add_extra`, `project_agent_config`.

This is the project-wide convention — see `docs/adk-api-notes/conventions.md`.
