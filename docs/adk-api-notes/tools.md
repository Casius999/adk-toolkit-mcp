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

---

# Pass 3b — optional-dependency toolsets + auth

Captured by introspection on 2026-06-01, google-adk **2.1.0**. **Runtime-confirmed** means the
class/signature was obtained from the *installed* package (after a temporary `uv pip install`);
**docs-only/source** means read from the package source file (import chain blocked by a missing or
conflicting transitive dep) but still authoritative for the constructor shape.

## How extras were probed (and the env restored)

The base `google-adk` install already vendors **mcp** (the `mcp` extra's deps: `McpToolset`,
connection params, and `mcp.StdioServerParameters` all import cleanly) and the **auth** classes.
The cloud/community toolsets need extra wheels. Probed by temporary imperative installs that do
**not** touch `pyproject.toml`/`uv.lock`:

- `google-cloud-bigquery` + `google-cloud-dataplex` → `BigQueryToolset` runtime-confirmed.
- `google-cloud-spanner` → `SpannerToolset` runtime-confirmed.
- `langchain-core` → `LangchainTool` runtime-confirmed.
- `crewai-tools` → install **downgraded opentelemetry** and broke ADK's own import chain
  (`ImportError: cannot import name 'GEN_AI_INPUT_MESSAGES'`). `CrewaiTool` signature therefore
  taken from **source** (`google/adk/integrations/crewai/crewai_tool.py`), not a live import.

Afterwards `uv sync --extra dev` restored the venv to the locked state (extras removed, ADK 2.1.0
imports cleanly, full suite green). `git hash-object uv.lock` is unchanged
(`ab272f4e7269ff00f5baa5df4ec82fbc72a7aa3e`) before and after.

## Confirmed import paths

```python
# MCP toolset + connection params (RUNTIME-CONFIRMED — base install, no extra needed here)
from google.adk.tools.mcp_tool import (
    McpToolset, StdioConnectionParams, StreamableHTTPConnectionParams, SseConnectionParams,
)
from mcp import StdioServerParameters

# BigQuery (RUNTIME-CONFIRMED with google-cloud-bigquery + google-cloud-dataplex)
from google.adk.tools.bigquery import BigQueryToolset            # + BigQueryCredentialsConfig

# Spanner (RUNTIME-CONFIRMED with google-cloud-spanner)
from google.adk.tools.spanner import SpannerToolset              # + SpannerCredentialsConfig

# API Hub (RUNTIME-CONFIRMED — base install; also re-exported at package root)
from google.adk.tools.apihub_tool import APIHubToolset

# Langchain / CrewAI (the task-specified paths; both RE-EXPORT from google.adk.integrations.*
# and emit a DeprecationWarning — see note below)
from google.adk.tools.langchain_tool import LangchainTool        # RUNTIME-CONFIRMED (langchain-core)
from google.adk.tools.crewai_tool import CrewaiTool              # source-confirmed (crewai import broke)

# Auth (RUNTIME-CONFIRMED — base install)
from google.adk.auth import AuthScheme, AuthCredential, AuthCredentialTypes  # AuthConfig also present
```

> **DeprecationWarning (langchain/crewai):** `google.adk.tools.langchain_tool` and
> `google.adk.tools.crewai_tool` are thin shims that `warnings.warn(..., DeprecationWarning)` and
> re-export from `google.adk.integrations.langchain` / `google.adk.integrations.crewai`. We emit the
> **task-specified** `google.adk.tools.*` paths (they still work and are what users expect from the
> task), but a future pass may switch to `google.adk.integrations.*` to avoid the warning. This is
> harmless for the toolkit: the generated `agent.py` is never imported by our tests (CI lacks the
> extras), and end users run it in their own venv with `-W` of their choosing.

## Confirmed constructor signatures

```python
# RUNTIME-CONFIRMED
McpToolset.__init__(self, *, connection_params, tool_filter=None, tool_name_prefix=None,
                    errlog=<stderr>, auth_scheme=None, auth_credential=None,
                    require_confirmation=False, header_provider=None, ...)   # all kw-only

StdioConnectionParams(server_params: StdioServerParameters, timeout: float = ...)   # pydantic
StreamableHTTPConnectionParams(url: str, headers: dict[str, Any] | None = None, timeout=..., ...)
SseConnectionParams(url: str, headers: dict[str, Any] | None = None, timeout=..., ...)
StdioServerParameters(command: str, args: list[str] = [], env=None, cwd=None, ...)   # from `mcp`

BigQueryToolset.__init__(self, *, tool_filter=None, credentials_config=None,
                         bigquery_tool_config=None)                          # all kw-only
SpannerToolset.__init__(self, *, tool_filter=None, credentials_config=None,
                        spanner_tool_settings=None)                          # all kw-only

APIHubToolset.__init__(self, *, apihub_resource_name: str, access_token=None,
                       service_account_json=None, name='', description='',
                       lazy_load_spec=False, auth_scheme=None, auth_credential=None,
                       apihub_client=None, tool_filter=None)                 # all kw-only

LangchainTool.__init__(self, tool, name: Optional[str] = None,
                       description: Optional[str] = None)                    # tool positional

# SOURCE-CONFIRMED (live import blocked by opentelemetry conflict from crewai-tools)
CrewaiTool.__init__(self, tool, *, name: str, description: str = '')         # name kw-only REQUIRED
```

Key signature facts that shaped the renderer:

- **`McpToolset` and `APIHubToolset` natively accept `auth_scheme=` / `auth_credential=` kwargs.**
  `BigQueryToolset` / `SpannerToolset` do **not** take auth kwargs directly — they take a
  `credentials_config` object. So "attach auth" only renders `auth_scheme=`/`auth_credential=` on
  the toolset kinds that accept them (mcp/apihub/openapi); for bigquery/spanner an `auth` sub-spec
  is rejected by validation (use their credentials args instead).
- All four GCP/MCP toolsets are `BaseToolset`s → they go **directly** into `tools=[...]` (same as
  `OpenAPIToolset` in 3a; no `.get_tools()`).
- `LangchainTool` / `CrewaiTool` subclass `FunctionTool` → they go directly into `tools=[...]` too.
  `CrewaiTool` **requires** a `name` (keyword-only); `LangchainTool`'s name/description are optional.

## Auth class shapes (RUNTIME-CONFIRMED, base install)

```python
AuthScheme = Union[APIKey, HTTPBase, OAuth2, OpenIdConnect, HTTPBearer, OpenIdConnectWithConfig,
                   CustomAuthScheme]   # a typing.Union, NOT a constructible class

class AuthCredentialTypes(Enum):       # the `auth_type` discriminator
    API_KEY='apiKey'; HTTP='http'; OAUTH2='oauth2'; OPEN_ID_CONNECT='openIdConnect';
    SERVICE_ACCOUNT='serviceAccount'

AuthCredential(auth_type: AuthCredentialTypes, *, resource_ref=None, api_key=None,
               http: HttpAuth|None=None, service_account: ServiceAccount|None=None,
               oauth2: OAuth2Auth|None=None)
HttpAuth(scheme: str, credentials: HttpCredentials, additional_headers=None)
HttpCredentials(username=None, password=None, token=None)     # `token` carries a bearer token
OAuth2Auth(client_id=None, client_secret=None, access_token=None, refresh_token=None, ...)
ServiceAccount(service_account_credential=None, scopes=None, use_default_credential=None, ...)
```

### Auth rendering decisions (`scheme` ∈ {apikey, oauth2, service_account, bearer})

`set_auth(... scheme, credential)` attaches an `auth` sub-spec to a toolset `ToolSpec`. We render an
**`AuthCredential(...)`** expression as `auth_credential=` (the part with secrets) and import the
needed names from `google.adk.auth`. We deliberately do **not** synthesize an `auth_scheme=`
(the ADK `AuthScheme` is a `Union` of FastAPI OpenAPI models — there is no single stable
constructor, and most toolsets already infer the scheme from the credential / spec). The four
schemes map to:

| `scheme`          | `auth_type`                          | rendered `AuthCredential` kwargs (from the `credential` dict)             | extra imports |
|-------------------|--------------------------------------|---------------------------------------------------------------------------|---------------|
| `apikey`          | `AuthCredentialTypes.API_KEY`        | `api_key="<credential['api_key']>"`                                       | `AuthCredential, AuthCredentialTypes` |
| `bearer`          | `AuthCredentialTypes.HTTP`           | `http=HttpAuth(scheme="bearer", credentials=HttpCredentials(token="<token>"))` | `+ HttpAuth, HttpCredentials` (from `google.adk.auth.auth_credential`) |
| `oauth2`          | `AuthCredentialTypes.OAUTH2`         | `oauth2=OAuth2Auth(client_id=..., client_secret=..., [access_token=...])` | `+ OAuth2Auth` |
| `service_account` | `AuthCredentialTypes.SERVICE_ACCOUNT`| `service_account=ServiceAccount(use_default_credential=True \| scopes=[...])` | `+ ServiceAccount` |

- `HttpAuth`/`HttpCredentials`/`OAuth2Auth`/`ServiceAccount` live in
  `google.adk.auth.auth_credential` (confirmed); `AuthCredential`/`AuthCredentialTypes` re-export at
  `google.adk.auth`.
- Only toolset kinds that accept the kwarg carry auth: **`openapi`, `apihub`, `mcp_toolset`**.
  `bigquery`/`spanner` reject an `auth` sub-spec at validation (they use `credentials_config`).

## Rendering decisions for `render_tool_ref` (3b additions)

| kind          | imports                                                        | helper (top-level block)                                                            | `ref` in `tools=[...]` |
|---------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------|------------------------|
| `bigquery`    | `from google.adk.tools.bigquery import BigQueryToolset`        | `<id> = BigQueryToolset(<args>)`                                                     | `<id>`                 |
| `spanner`     | `from google.adk.tools.spanner import SpannerToolset`          | `<id> = SpannerToolset(<args>)`                                                      | `<id>`                 |
| `mcp_toolset` | `McpToolset` + connection-params class + (stdio) `StdioServerParameters` | `<id> = McpToolset(connection_params=..., tool_filter=[...][, auth_*])`  | `<id>`                 |
| `apihub`      | `from google.adk.tools.apihub_tool import APIHubToolset`       | `<id> = APIHubToolset(apihub_resource_name="...", [auth_*])`                         | `<id>`                 |
| `langchain`   | the user `import_line` (verbatim) + `LangchainTool`            | (none)                                                                               | `LangchainTool(tool=<expr>)` |
| `crewai`      | the user `import_line` (verbatim) + `CrewaiTool`               | (none)                                                                               | `CrewaiTool(tool=<expr>, name="...", description="...")` |

- `mcp_toolset` transports: `stdio` → `StdioConnectionParams(server_params=StdioServerParameters(command="...", args=[...]))`;
  `sse` → `SseConnectionParams(url="...", headers={...})`; `http` → `StreamableHTTPConnectionParams(url="...", headers={...})`.
- `langchain`/`crewai` take a **user-provided `import_line`** (e.g. `from langchain_community.tools import WikipediaQueryRun`)
  rendered verbatim before the agents, plus a `tool_expr` (e.g. `WikipediaQueryRun(...)`) — the
  toolkit cannot know which third-party tool the user wants, so it accepts the construction expression.
