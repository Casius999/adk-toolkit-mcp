# ADK API notes — `tools` domain (pass 3a, dependency-free tools)

Captured by introspection on 2026-06-01. google-adk **2.1.0**, fastmcp **3.3.1**, Python 3.12.
These are observed facts (run against the installed packages), not guesses.

## Canonical imports (package root)

Everything the `tools` domain renders imports cleanly from the package root in one line:

```python
from google.adk.tools import (
    google_search, url_context, load_memory, preload_memory, load_artifacts,
    get_user_choice, exit_loop, transfer_to_agent, enterprise_web_search,
    google_maps_grounding, VertexAiSearchTool, FunctionTool, LongRunningFunctionTool, AgentTool,
)
```

`OpenAPIToolset` is **not** at the package root; its confirmed import path is:

```python
from google.adk.tools.openapi_tool import OpenAPIToolset
```

`dir(google.adk.tools)` (public) on 2.1.0:
`APIHubToolset, AgentTool, ApiRegistry, AuthToolArguments, BaseTool, DiscoveryEngineSearchTool,
ExampleTool, FunctionTool, LongRunningFunctionTool, MCPToolset, McpToolset, SearchResultMode,
TransferToAgentTool, VertexAiSearchTool, computer_use, enterprise_web_search, exit_loop,
get_user_choice, google_maps_grounding, google_search, load_artifacts, load_memory,
preload_memory, set_model_response_tool, transfer_to_agent, url_context`.

## How a plain function appears in `tools=[...]` (CRITICAL)

A bare Python function passed in `tools=[my_fn]` is **NOT** wrapped eagerly at construction time.

```python
def my_fn(x: str) -> dict: ...
a = LlmAgent(name="probe", model="gemini-2.5-flash", instruction="hi", tools=[my_fn])
[type(x).__name__ for x in a.tools]            # -> ['function']   (raw field, unchanged)
[type(x).__name__ for x in await a.canonical_tools()]  # -> ['FunctionTool']  (lazy wrap)
```

Implications for the functional probe:

- Immediately after `LlmAgent(...)`, the **raw** `.tools` field holds the original objects
  (`function` for a plain function). It does **not** become a `FunctionTool` yet.
- The wrapping into `FunctionTool` happens lazily inside the async `canonical_tools()`
  (a coroutine; needs `await`). The probe therefore asserts on **both**:
  raw `.tools` length/identity *and* the awaited `canonical_tools()` types.

So in generated code we emit a plain `def <name>(...)` and put the **bare name** in `tools=[...]`
(ADK auto-wraps via `canonical_tools`). No `FunctionTool(...)` wrapper is rendered for `function`.

## Constructor signatures (confirmed)

```python
FunctionTool.__init__(self, func: Callable[..., Any], *, require_confirmation=False)
LongRunningFunctionTool.__init__(self, func: Callable)          # func is positional-or-keyword
AgentTool.__init__(self, agent: BaseAgent, skip_summarization=False, *,
                   include_plugins=True, propagate_grounding_metadata=False)
VertexAiSearchTool.__init__(self, *, data_store_id=None, data_store_specs=None,
                   search_engine_id=None, filter=None, max_results=None,
                   bypass_multi_tools_limit=False)             # data_store_id / search_engine_id are kw-only
OpenAPIToolset.__init__(self, *, spec_dict=None, spec_str=None, spec_str_type='json',
                   auth_scheme=None, auth_credential=None, credential_key=None,
                   tool_filter=None, tool_name_prefix=None, ssl_verify=None,
                   header_provider=None, preserve_property_names=False)   # all kw-only
```

- `LongRunningFunctionTool(func=<name>)` — type stays `LongRunningFunctionTool` in both the raw
  `.tools` field and `canonical_tools()`.
- `AgentTool(agent=<var>)` — type stays `AgentTool` in both raw and canonical.
- `VertexAiSearchTool(data_store_id="...")` instantiates fine with just `data_store_id` (or
  `search_engine_id`). Both are keyword-only.

## Builtins: pre-instantiated tool *objects* vs plain functions (CRITICAL)

The "function-style" builtins are **already-instantiated tool instances** exported at module level —
**not** classes and **not** (mostly) plain callables. They drop straight into `tools=[name]`:

| Builtin name        | Runtime type (instance)      | callable? | Goes in `tools=[...]` as | Needs arg? |
|---------------------|------------------------------|-----------|--------------------------|------------|
| `google_search`     | `GoogleSearchTool`           | no        | `google_search`          | core, none |
| `url_context`       | `UrlContextTool`             | no        | `url_context`            | core, none |
| `load_memory`       | `LoadMemoryTool`             | no        | `load_memory`            | core, none |
| `preload_memory`    | `PreloadMemoryTool`          | no        | `preload_memory`         | core, none |
| `load_artifacts`    | `LoadArtifactsTool`          | no        | `load_artifacts`         | core, none |
| `get_user_choice`   | `LongRunningFunctionTool`    | no        | `get_user_choice`        | core, none |
| `exit_loop`         | `function`                   | yes       | `exit_loop`              | core, none |
| `transfer_to_agent` | `function`                   | yes       | `transfer_to_agent`      | core, none |
| `enterprise_web_search` | `EnterpriseWebSearchTool` | no       | `enterprise_web_search`  | core, none |
| `google_maps_grounding` | `GoogleMapsGroundingTool` | no       | `google_maps_grounding`  | core, none |
| `vertex_ai_search`  | `VertexAiSearchTool` (class) | n/a       | `VertexAiSearchTool(data_store_id="...")` | **needs arg** |

- All of the above except `vertex_ai_search` are **core** (no extra argument): emit the bare name,
  import it from `google.adk.tools`.
- `vertex_ai_search` is the only one that needs an arg: emit `VertexAiSearchTool(data_store_id="...")`
  (or `search_engine_id="..."`), importing the `VertexAiSearchTool` **class** from `google.adk.tools`.
- `request_input` is **NOT present** in google-adk 2.1.0 (`hasattr(google.adk.tools, "request_input")`
  is `False`). It is therefore omitted from the supported builtin set.

Because these are tool *instances*, they need **no wrapping**; they appear unchanged in the raw
`.tools` field (e.g. `GoogleSearchTool`) and unchanged after `canonical_tools()`.

## `OpenAPIToolset` goes DIRECTLY into `tools=[...]` (CRITICAL — resolves the open question)

`OpenAPIToolset` is a `BaseToolset`. An `LlmAgent` accepts a toolset **directly** in its tools list;
**`.get_tools()` is NOT required**:

```python
ts = OpenAPIToolset(spec_str=SPEC, spec_str_type="json")
a = LlmAgent(name="probe", model="gemini-2.5-flash", instruction="hi", tools=[ts])  # accepted
[type(x).__name__ for x in a.tools]                     # -> ['OpenAPIToolset'] (raw, the toolset itself)
[type(x).__name__ for x in await a.canonical_tools()]   # -> ['RestApiTool', ...] (expanded lazily)
```

So generated code emits a top-level construction `<id> = OpenAPIToolset(spec_str=<...>, spec_str_type="json")`
and places the **bare `<id>`** in `tools=[...]`. We do **not** call `.get_tools()` (it is an async
coroutine anyway; ADK does the expansion through `canonical_tools`). `spec_str_type` defaults to `'json'`;
we render it explicitly for clarity/stability.

## Rendering decisions for `project_model.render_tool_ref` (3a)

| kind          | imports                              | helper (top-level block)                                   | `ref` in `tools=[...]`            |
|---------------|--------------------------------------|------------------------------------------------------------|-----------------------------------|
| `function`    | (none)                               | `def <name>(<typed params>) -> <ret>: """doc""" <body>`    | `<name>`                          |
| `long_running`| `LongRunningFunctionTool`            | same `def <name>(...)`                                     | `LongRunningFunctionTool(func=<name>)` |
| `builtin` (core) | the builtin name                  | (none)                                                     | `<name>` (e.g. `google_search`)   |
| `builtin` `vertex_ai_search` | `VertexAiSearchTool`    | (none)                                                     | `VertexAiSearchTool(data_store_id="...")` |
| `agent_tool`  | `AgentTool`                          | (none — target is an existing agent var)                   | `AgentTool(agent=<target_var>)`   |
| `openapi`     | `OpenAPIToolset` (submodule path)    | `<id> = OpenAPIToolset(spec_str=<...>, spec_str_type="json")` | `<id>`                         |

- Tool helper blocks (function defs, toolset constructions) are emitted **before** the agents that
  reference them, with PEP 8 / ruff-format spacing (2 blank lines around top-level `def`s).
- `agent_tool` enforces: target must be an agent in the model; topo-order ensures the target agent is
  defined before the agent that wraps it; the target is **not** also added as a `sub_agent`
  (respects ADK's single-parent rule — an `AgentTool`-wrapped agent is a tool, not a child).
