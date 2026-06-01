# a2a + mcp_bridge — ADK 2.1.0 API notes (P4 part b)

Captured 2026-06-01 by introspection. `google-adk` **2.1.0**, `fastmcp` 3.3.1, Python 3.12.
The `a2a` surface was introspected after a **transient** `uv pip install "google-adk[a2a]"`
(installed only `a2a-sdk==0.3.26`); the env was then restored via `uv sync --extra dev` and
`git status` confirmed **NO `uv.lock`/`pyproject` change** (the `a2a` extra was already declared
in `pyproject` from P0 — we only INSTALLED it imperatively to introspect, never re-locked).

## Which surfaces need which extra

| Surface | Import path | Needs `a2a` extra? | Notes |
|---|---|---|---|
| `to_a2a` | `google.adk.a2a.utils.agent_to_a2a` | **YES** | module does `from a2a.server.apps import …` at top → `ModuleNotFoundError` without the extra |
| `RemoteA2aAgent` | `google.adk.agents.remote_a2a_agent` | **YES** | module imports `a2a.*` at top |
| `AgentCardBuilder` | `google.adk.a2a.utils.agent_card_builder` | **YES** | builds `a2a.types.AgentCard` |
| `adk_to_mcp_tool_type` | `google.adk.tools.mcp_tool.conversion_utils` | **NO** | `mcp` is CORE (fastmcp depends on it) → fully CI-testable |

## a2a — confirmed signatures (each REQUIRES the `a2a` extra)

### `to_a2a` → `Starlette`
```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a  # needs a2a extra
to_a2a(
    agent: BaseAgent,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: AgentCard | str | None = None,   # str = PATH to a JSON file (not a URL!)
    push_config_store=None, task_store=None, runner=None,
    lifespan=None, agent_executor_factory=None,
) -> starlette.applications.Starlette
```
- Returns a **Starlette** app. **Decorated `@a2a_experimental`** (emits an experimental
  `UserWarning`, NOT a `DeprecationWarning`).
- ⚠️ The task's guessed signature was `to_a2a(agent, *, port=...)`. The REAL signature also has
  `host`/`protocol`. Default **host is `localhost`**, default **port 8000**.
- **Routes are registered LAZILY in the Starlette lifespan (on startup)**, not at construction:
  `setup_a2a()` (an async lifespan step) calls `AgentCardBuilder(...).build()` then
  `A2AStarletteApplication(agent_card=…, http_handler=…).add_routes_to_app(app)`. So
  `app.routes` is EMPTY until the app has started (uvicorn lifespan). A live HTTP probe (server
  actually running) is the only way to see the well-known route.
- The well-known agent-card route is `a2a.utils.constants.AGENT_CARD_WELL_KNOWN_PATH`
  = **`/.well-known/agent-card.json`** (confirmed by resolving the constant). Extended card =
  `/agent/authenticatedExtendedCard`.
- Serve command (no `execute`): **`uvicorn <module>:a2a_app --host localhost --port <PORT>`**
  (the ADK docstring's own example uses `uvicorn module:app`).

### `RemoteA2aAgent` (a `BaseAgent` subclass)
```python
# ⚠️ NOT in google.adk.agents in 2.1.0 — only this submodule path works:
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent  # needs a2a extra
RemoteA2aAgent(
    name: str,
    agent_card: AgentCard | str,        # SECOND POSITIONAL; str = a URL or a file path
    *,
    description: str = "",
    httpx_client=None, timeout: float = 600.0,
    genai_part_converter=…, a2a_part_converter=…, a2a_client_factory=None,
    a2a_request_meta_provider=None, full_history_when_stateless=False,
    config=None, use_legacy=True, **kwargs,
) -> None
```
- **KEY DEVIATION FROM THE TASK:** the task said `from google.adk.agents import RemoteA2aAgent`.
  That FAILS in 2.1.0 — `RemoteA2aAgent` is **NOT** in `google.adk.agents.__all__` and there is
  **no lazy `__getattr__` export** for it. The ONLY import is
  `from google.adk.agents.remote_a2a_agent import RemoteA2aAgent`. The renderer MUST emit that
  submodule import.
- `agent_card` is the **second positional** arg (`RemoteA2aAgent(name=…, agent_card=…)`).
  Constructed cleanly with just `name` + a URL string; `isinstance(r, BaseAgent)` is True →
  composes as a `sub_agents` member of another agent.

### `AgentCardBuilder`
```python
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder  # needs a2a extra
AgentCardBuilder(
    *,
    agent: BaseAgent,
    rpc_url: str | None = None,
    capabilities=None, doc_url=None, provider=None, agent_version=None, security_schemes=None,
)
async def build(self) -> a2a.types.AgentCard   # ASYNC
```
- `build()` is **async** → `AgentCard` (`a2a.types.AgentCard`, a pydantic model). For the
  default `rpc_url="http://localhost:8001"` and an `LlmAgent(name="root_agent",
  description="…")`, `card.model_dump(exclude_none=True)` keys are: `capabilities`,
  `defaultInputModes`, `defaultOutputModes`, `description`, `name`, `preferredTransport`,
  `protocolVersion`, `skills`, `supportsAuthenticatedExtendedCard`, `url`, `version`. `card.name`
  = the agent's name; `card.url` = the rpc_url.
- `to_a2a` itself uses this builder internally (`rpc_url=f"{protocol}://{host}:{port}/"`), so the
  `agent_card` tool can build the same card the served app would expose. Building the card needs
  the agent imported (so `import_root_agent` + the a2a extra) → gate gracefully when absent.

## mcp_bridge — confirmed (NO extra; `mcp` is core)

### `adk_to_mcp_tool_type(tool: BaseTool) -> mcp.types.Tool`
```python
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type  # mcp is CORE
adk_to_mcp_tool_type(tool: BaseTool) -> mcp.types.Tool
```
- Returns an **`mcp.types.Tool`** (pydantic). Fields: `name`, `title`, `description`,
  `inputSchema` (a **dict** JSON-Schema), `outputSchema`, `icons`, `annotations`, `meta`,
  `execution`. The toolkit surfaces `{name, description, inputSchema}`.
- **FUNCTIONAL result (load-bearing, no extra):** `adk_to_mcp_tool_type(google_search)` →
  `Tool(name='google_search', description='google_search', inputSchema={})` (a builtin with no
  declared params → **empty** inputSchema `{}`). A `FunctionTool(add_numbers)` over
  `def add_numbers(a: int, b: int) -> int: '''Add two integers and return the sum.'''` →
  `name='add_numbers'`, `description='Add two integers and return the sum.'`, and a real schema:
  ```json
  {"properties": {"a": {"title": "A", "type": "integer"},
                  "b": {"title": "B", "type": "integer"}},
   "required": ["a", "b"], "title": "add_numbersParams", "type": "object"}
  ```

### Extracting an agent's tools — the ROBUST path
- `LlmAgent.tools` holds the RAW entries: a builtin shows as e.g. `GoogleSearchTool`, but a plain
  `def` is stored as a bare **`function`** (NOT yet a `FunctionTool`) → `adk_to_mcp_tool_type`
  would reject it.
- ✅ Use **`await agent.canonical_tools(ctx=None)`** (an **async** method →
  `list[BaseTool]`): it wraps plain functions into `FunctionTool` and normalises EVERYTHING to
  `BaseTool`, so every element converts cleanly. Verified: `[google_search, add_numbers]` →
  canonical `[GoogleSearchTool, FunctionTool]`, all `isinstance BaseTool`, both convert.
- `mcp_bridge.expose_adk_tools` therefore takes path **(a)**: `run_core.import_root_agent(path,
  app_name)` to get the project's real agent, find the named agent (walk `root_agent` +
  sub-agents), `await agent.canonical_tools()`, and convert each. This reuses the proven
  import machinery (unique module name + compile/exec to defeat the Windows mtime bytecode cache)
  and exercises the user's ACTUAL tool specs (no re-derivation from the sidecar).

## Implications for the toolkit

- **project_model `remote_a2a`**: render
  `<name> = RemoteA2aAgent(name="<name>", agent_card="<url>")` with the import
  `from google.adk.agents.remote_a2a_agent import RemoteA2aAgent`. No children; can be a
  `sub_agents` member (topological dep edge handled like any agent reference). Generated code is
  codegen-only — the toolkit NEVER imports `RemoteA2aAgent` itself, so no extra is needed to
  GENERATE it (only to RUN the generated app).
- **a2a.expose** writes `a2a_app.py` (`from google.adk.a2a.utils.agent_to_a2a import to_a2a` +
  `a2a_app = to_a2a(root_agent, port=PORT)`) — codegen-only (no import at toolkit runtime).
  `execute=True` gates on `find_spec("a2a")` and starts a managed `uvicorn a2a_app:a2a_app`
  process via the `adk_cli` registry; `execute=False` (default) returns the file path + the
  serve command.
- **a2a.consume / a2a.agent_card**: the FUNCTIONAL probes (real `RemoteA2aAgent` type / real
  `AgentCard` build) are **gated on `find_spec("a2a")`** and SKIP when the extra is absent;
  `agent_card` returns a clean actionable `err` ("install adk-toolkit-mcp[a2a]") when gated off.
- **mcp_bridge**: fully CI-testable (mcp is core). No extra, no gating.
