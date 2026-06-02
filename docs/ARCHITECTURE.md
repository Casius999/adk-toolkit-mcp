# adk-toolkit-mcp — Architecture

This document describes the internal structure of the server: how the FastMCP sub-server tree
is composed, the code-first sidecar model, the support packages, and the key design
invariants.

---

## Directory tree

```
adk-toolkit-mcp/
├── src/adk_toolkit_mcp/
│   ├── server.py           # Root FastMCP server + build_server() + main()
│   ├── envelope.py         # {ok, data, error} helpers: ok(...) / err(...)
│   ├── workspace.py        # Workspace(path) — thin helper for sidecar file I/O
│   ├── versions.py         # google-adk / fastmcp / Python version strings
│   ├── resources.py        # register_resources() — adk://version, adk://models
│   ├── prompts.py          # register_prompts() — 5 workflow prompts
│   ├── deps.py             # Lazy optional-dep helpers
│   ├── adk_cli.py          # adk_executable(), run_adk(), process registry
│   ├── runtime.py          # RuntimeConfig, service factories, singleton cache
│   ├── run_core.py         # build_runner() (agent= or node=), collect_events(), import_root_agent()
│   ├── domains/            # 17 FastMCP sub-servers (one per file)
│   │   ├── project.py      # project_server
│   │   ├── agents.py       # agents_server (incl. set_planner)
│   │   ├── tools.py        # tools_server
│   │   ├── models.py       # models_server
│   │   ├── sessions.py     # sessions_server
│   │   ├── memory.py       # memory_server
│   │   ├── artifacts.py    # artifacts_server
│   │   ├── workflow.py     # workflow_server (graph-orchestration engine)
│   │   ├── skills.py       # skills_server (Agent Skill Registry)
│   │   ├── run.py          # run_server
│   │   ├── eval.py         # eval_server
│   │   ├── deploy.py       # deploy_server
│   │   ├── dev.py          # dev_server
│   │   ├── a2a.py          # a2a_server
│   │   ├── mcp_bridge.py   # mcp_bridge_server
│   │   ├── safety.py       # safety_server
│   │   ├── observability.py        # observability_server
│   │   └── observability_setup.py  # OTel codegen helper (not a sub-server)
│   │   └── safety_plugins.py       # Plugin codegen helper (not a sub-server)
│   └── project_model/      # Code-first sidecar + codegen engine
│       ├── specs.py        # Pydantic spec models (AgentSpec, ToolSpec, WorkflowSpec, …)
│       ├── sidecar.py      # Read/write .adk_toolkit/agents.json
│       ├── render.py       # render_agent_module() → agent.py source (agents, workflows, planners, skills)
│       ├── _codegen.py     # Low-level AST/source builder helpers
│       └── _workflow_codegen.py  # Workflow graph → source helpers (nodes/edges/root)
├── docs/
│   ├── ARCHITECTURE.md     # This file
│   ├── TOOL_CATALOG.md     # All 94 tools
│   ├── CONTRIBUTING.md     # Dev setup + conventions
│   └── adk-api-notes/      # Per-domain ADK introspection notes (ground truth)
├── skill/                  # adk-toolkit Claude Code companion skill
│   ├── SKILL.md            # Routing index (install to ~/.claude/skills/adk-toolkit/)
│   └── references/         # 14 reference files (00-13)
└── tests/
    └── unit/               # pytest test suite (755 passed, 6 skipped)
```

---

## Root server and sub-server mount pattern

`build_server()` in `server.py` constructs the root `FastMCP` instance and mounts 17
domain sub-servers onto it:

```python
mcp = FastMCP("adk-toolkit-mcp")
mcp.mount(project_server,      namespace="project")
mcp.mount(agents_server,       namespace="agents")
mcp.mount(tools_server,        namespace="tools")
# … 12 more domains …
```

Each domain file (`domains/<domain>.py`) declares its own `FastMCP("<domain>")` instance
(named `<domain>_server`) and registers tools on it with bare function names:

```python
# domains/project.py
project_server: FastMCP = FastMCP("project")

@project_server.tool(tags={"project"})
def create(...) -> dict[str, Any]: ...
```

FastMCP concatenates `namespace + "_" + bare_name`, so `create` mounted under
`namespace="project"` is exposed to clients as `project_create`. The full convention is
documented in `docs/adk-api-notes/conventions.md`.

**Important:** `prefix=` (the older FastMCP mount parameter) is deprecated in fastmcp 3.3.1
and emits a `DeprecationWarning`. The project uses `namespace=` exclusively.

### All 94 tools have a domain tag

Every `@<domain>_server.tool` decorator carries `tags={"<domain>"}`. This enables Code Mode
discovery (`search(tags=["run"])`) and is also visible to MCP clients via
`tool._meta.fastmcp.tags`.

### Resources and prompts

`register_resources(mcp)` adds two read-only resources:

- `adk://version` — pinned `google-adk`, `fastmcp`, Python version strings.
- `adk://models` — common Gemini model strings.

`register_prompts(mcp)` adds five workflow prompts (see `prompts.py`):
`scaffold_multi_agent`, `add_guardrail`, `write_evalset`, `deploy_checklist`, `debug_agent`.
Each carries `tags={"workflow"}`.

---

## Code-first sidecar model

The author domains (`project`, `agents`, `tools`, `models`, `safety`, `workflow`, `skills`)
never import `google-adk` at the tool level (the `skills` domain imports the read-only ADK skill
*loaders* inside its own tool bodies to read skill fields, but never bakes ADK into the generated
app). Instead they manipulate a **JSON sidecar** and regenerate `agent.py` on every mutation:

```
.adk_toolkit/
└── agents.json   ← source of truth: agents + workflows (read/written by project_model.sidecar)
agent.py          ← fully regenerated by project_model.render
__init__.py       ← static (from agent import root_agent → for eval/run)
.env              ← env vars (project_set_env)
runtime.json      ← session/memory/artifact backend config (runtime.py)
plugins.py        ← generated plugin instances (safety_add_plugin)
otel_setup.py     ← generated OTel bootstrap (observability_enable_otel)
a2a_app.py        ← generated A2A server (a2a_expose)
skills/<name>/    ← on-disk SKILL.md skill folders (skills_create; loaded at the agent's runtime)
```

`agents.json` holds the declarative graph: a list of `AgentSpec` objects **and** a list of
`WorkflowSpec` objects (pydantic, defined in `project_model/specs.py`), plus which one is the
root and the root kind (`agent` vs `workflow`). A tool call such as `agents_create_llm(...)`,
`agents_set_planner(...)`, `workflow_add_node(...)`, or `skills_attach(...)` deserializes the
sidecar, adds or updates the relevant spec, serializes it back, then calls
`render.render_agent_module(...)` to regenerate `agent.py`.

### Codegen quality bar

Generated `agent.py` must pass three checks (asserted in tests):

1. `ast.parse(source)` — syntactically valid Python.
2. `ruff format --check` — ruff-formatted (no style diffs).
3. `ruff check --select I` — isort-clean imports.

### `project_model` package

| Module | Role |
|---|---|
| `specs.py` | `AgentSpec` (incl. a `PlannerSpec` for `agents_set_planner`), `ToolSpec` (incl. the `skill_toolset` kind), `CallbackSpec`, `GenerateContentConfigSpec`, `WorkflowSpec`/`NodeSpec`/`EdgeSpec`, … — all pydantic models representing the declarative description of the agent graph and workflow graphs. |
| `sidecar.py` | `load_specs(ws)` / `save_specs(ws, specs)` — JSON round-trip for `agents.json` (agents + workflows + root). Validates on load; raises `ValueError` on schema violation. |
| `render.py` | `render_agent_module(...)` → source string. Topological sort (child before parent), import merge (isort-clean), tool helper blocks before agents; also renders planners (`planner=…`), `SkillToolset` blocks, and `Workflow(...)` graphs, and emits the workflow-vs-agent root. |
| `_codegen.py` | Low-level source builder: `_Call`, `_Import`, `_FuncDef`, `_merge_tool_imports`, etc. Pure string construction; no `ast` module at render time. |
| `_workflow_codegen.py` | Workflow-specific source helpers: render `Workflow(name=..., edges=[...])`, agent / `@node` function / `JoinNode` nodes, and unconditional vs conditional (`route` / route-dict) edges. |

---

## Runtime service factory (`runtime.py`)

`runtime.py` centralises three things:

1. **`RuntimeConfig`** — `SessionBackend` + `MemoryBackend` + `ArtifactBackend` pydantic
   dataclasses persisted to `.adk_toolkit/runtime.json` alongside the sidecar. A missing file
   defaults to in-memory backends for all three.

2. **Singleton cache** — `get_session_service(backend)`, `get_memory_service(backend)`,
   `get_artifact_service(backend)` each maintain a process-level `dict` cache keyed by
   `(kind, url/project/location/…)`. This ensures that two tool calls sharing the same
   `in_memory` backend receive the **same service instance** — critical because
   `InMemorySessionService` (and its memory/artifact counterparts) hold state in process
   memory.

3. **Lazy optional deps** — `DatabaseSessionService` (needs `sqlalchemy`), Vertex backends
   (need `gcp`), and GCS artifacts (need `gcp`) are imported inside the factory function
   only when requested. A missing extra raises an `ImportError` that is converted to a
   `ValueError` with an actionable install hint.

`reset_service_cache()` clears all three caches — used in tests for isolation.

---

## `run_core.py` — execution core

`run_core.py` factors the ADK agent execution loop so it can be tested fully offline with a
`FakeLlm`. The domain `run.py` is a thin MCP wrapper on top of these helpers.

Key functions:

| Function | Purpose |
|---|---|
| `build_runner(app_name, root_agent, runtime_config, plugins=None)` | Constructs `google.adk.runners.Runner` wired with the toolkit's session/memory/artifact services. Accepts **either** a `BaseAgent` root (passed via `agent=`) **or** a `BaseNode` root such as a `Workflow` graph (passed via `node=`, detected by `is_workflow_node_root`); the agent path is unchanged and backward compatible. When `plugins` is non-empty uses `Runner(app=App(name, root_agent, plugins=[...]))` to avoid the deprecated `Runner(plugins=)` kwarg (`App.root_agent` also accepts a `BaseNode`). |
| `collect_events(runner, *, user_id, session_id, new_message_text, run_config=None, progress=None)` | Ensures the session exists (creates if absent), runs `runner.run_async(...)` as an async generator, collects `Event` objects, and awaits `progress(event)` per event for SSE. |
| `serialize_event(event)` | Flattens an ADK `Event` to `{author, text, function_calls, function_responses, state_delta, transfer_to_agent, is_final, partial}`. |
| `import_root_agent(path, app_name)` | Loads `<path>/<app_name>/agent.py`'s `root_agent` via `importlib` with a unique module name per call (defeats the `sys.modules` / bytecode cache on re-import after edits). Raises `RootAgentImportError` on failure. |
| `build_run_config(streaming_mode, max_llm_calls, response_modalities)` | Validates `streaming_mode` against the real `StreamingMode` enum by name (`NONE`/`SSE`/`BIDI`) and constructs `RunConfig`. |

---

## `adk_cli.py` — CLI wrapper and process registry

`adk_cli.py` provides two surfaces:

**CLI invocation:**

- `adk_executable()` — resolves the ADK binary: venv `Scripts/adk.exe` first, then PATH `adk`,
  then `[sys.executable, "-m", "google.adk.cli"]`.
- `run_adk(args, cwd, timeout)` — runs `<adk> <args>` via `subprocess.run` (argv list, never
  `shell=True`) and returns `{argv, rc, stdout, stderr}`.
- `available_flags(subcommand)` — parses `adk <subcommand> --help` and returns the set of
  valid `--flag` tokens. Cached per subcommand. The deploy/dev domains validate every emitted
  flag against this set to prevent drift between the toolkit and the installed ADK version.

**Process registry** (for long-running servers):

- `start_process(key, argv, cwd, log_file)` — launches via `Popen`; on Windows uses
  `CREATE_NEW_PROCESS_GROUP` so `stop_process` can kill the process tree.
- `process_status(key)`, `process_logs(key, tail)`, `stop_process(key)`, `stop_all_processes()`
- Used by `dev_web`, `dev_api_server`, `a2a_expose(execute=True)`, and `observability_trace_view`.

---

## `{ok, data, error}` envelope

Every tool returns the uniform envelope defined in `envelope.py`:

```python
{"ok": True,  "data": <payload>, "error": None}   # success
{"ok": False, "data": None,      "error": "<msg>"} # failure
```

Use the helpers:

```python
from adk_toolkit_mcp.envelope import ok, err
return ok({"key": "value"})
return err("Something went wrong — hint about how to fix it.")
```

`err(...)` never raises and never swallows; it always returns a dict. An eval failure
(`eval_run` where the agent does not meet thresholds) returns `ok=True, data={passed: False}` —
a normal result, not an error.

---

## Code Mode

`build_server(code_mode=False)` is the default entry point. When `code_mode=True` (or
`ADK_TOOLKIT_CODE_MODE=1` in the environment), `server.py` calls:

```python
from fastmcp.experimental.transforms.code_mode import CodeMode, Search, GetSchemas, GetTags
mcp.add_transform(CodeMode(discovery_tools=[Search(), GetSchemas(), GetTags()]))
```

This is applied **after all mounts** and collapses the 94-tool catalog to 4 meta-tools:
`search`, `get_schema`, `tags`, `execute`. Code Mode is an **experimental** FastMCP feature
(available since FastMCP 3.1+; latest stable 3.3.1). The `execute` sandbox requires the optional
`fastmcp[code-mode]` / `pydantic-monty` package (not installed by default); discovery tools
work without it.

---

## Lazy optional dependencies

No optional dependency is imported at module load time. Patterns used:

- **Author domains** (`project`, `agents`, `tools`, `models`, `workflow`): never import
  `google-adk` at all — they only manipulate the sidecar and generate source strings. (`skills`
  is also an author domain but imports the read-only ADK skill *loaders* inside its tool bodies to
  read on-disk skill fields; it still never bakes ADK into the generated app.)
- **Runtime domains** (`sessions`, `memory`, `artifacts`, `run`, `eval`): import ADK inside the
  tool body. A `ModuleNotFoundError` for an absent extra is caught and returned as `err(...)`.
- **`TYPE_CHECKING` guards**: used throughout for type hints that reference heavy types
  (`BaseAgent`, `Runner`, `Event`, etc.) without importing them at runtime.

---

## Key invariants

- Tool names: `<domain>_<bare>` (single prefix). Never `<domain>_<domain>_<bare>`.
- Every tool returns the `{ok, data, error}` envelope.
- Generated `agent.py`: must pass `ast.parse` + `ruff format --check` + `ruff check --select I`.
- Tests run fully offline (no Google API key) using `FakeLlm`/`ScriptedLlm` fixtures.
- Suite green under `-W error::DeprecationWarning` (28 benign `UserWarning` from ADK experimental
  features; zero `DeprecationWarning` from the toolkit itself).
- Coverage ≥ 80% (current: ~95%).
