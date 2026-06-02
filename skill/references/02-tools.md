# 02 — Tools & auth (the `tools` domain)

Attach tools to **LlmAgents** (only LlmAgents carry tools). Maps to the `tools_*` tools. Each tool
operates on `(path, app_name, agent_name, …)`, attaches/replaces a tool spec (append-unique, replace
by name), and regenerates `agent.py`.

## Contents
- [Tool kinds + which extra each needs](#tool-kinds)
- [Dependency-free tools (pass 3a)](#dependency-free)
- [Optional-dependency toolsets (pass 3b)](#optional-dependency)
- [Auth (`tools_set_auth`)](#auth)
- [Agent Skills (`skills_*` + the `SkillToolset` tool kind)](#skills)
- [Critical ADK facts](#critical-facts)

## <a name="tool-kinds"></a>Tool kinds + extra

| Tool | Kind | Extra to RUN | Goes in `tools=[...]` as |
|---|---|---|---|
| `tools_add_function` | plain Python function | none (core) | bare `<name>` (ADK auto-wraps to `FunctionTool`) |
| `tools_add_long_running` | long-running function | none (core) | `LongRunningFunctionTool(func=<name>)` |
| `tools_add_builtin` | ADK builtin instance | none (core) | bare name, or `VertexAiSearchTool(...)` |
| `tools_add_agent_tool` | wrap another agent | none (core) | `AgentTool(agent=<target>)` |
| `tools_add_openapi` | `OpenAPIToolset` | none (core) | `<id>` (toolset directly) |
| `tools_add_bigquery` | `BigQueryToolset` | `bigquery` | `<id>` (codegen-only) |
| `tools_add_spanner` | `SpannerToolset` | `spanner` | `<id>` (codegen-only) |
| `tools_add_mcp_toolset` | `McpToolset` | `mcp` (core dep) | `<id>` (codegen-only) |
| `tools_add_apihub` | `APIHubToolset` | core/`gcp` creds | `<id>` (codegen-only) |
| `tools_add_langchain` | `LangchainTool` | `community` | `LangchainTool(tool=<expr>)` |
| `tools_add_crewai` | `CrewaiTool` | `community` | `CrewaiTool(tool=<expr>, name=…, description=…)` |
| `skills_attach` | `SkillToolset` (Agent Skills) | none (core) | `<id>` (toolset directly) — see [Agent Skills](#skills) |
| `tools_set_auth` | attach auth to a toolset | — | `auth_credential=AuthCredential(...)` |
| `tools_list` | list an agent's tools | — | read-only |

> **Codegen-only** means the toolkit emits code importing the extra but never imports it itself.
> You add the extra (`project_add_extra` or `uv add 'google-adk[<extra>]'`) in your venv to RUN.

## <a name="dependency-free"></a>Dependency-free tools (no extra)

### `tools_add_function` / `tools_add_long_running`
```
tools_add_function(path, app_name, agent_name, func_name, params, docstring,
                   returns="dict", body="return {}")
```
- `params` is a list of `{"name": ..., "type": "str", "default": null}`. `default` is a **source
  literal** (e.g. `"0"`, `'"hi"'`) or `null` (no default).
- A plain `def` placed bare in `tools=[...]` is **auto-wrapped to `FunctionTool`** by ADK at runtime
  (no wrapper emitted). `long_running` wraps it in `LongRunningFunctionTool(func=<name>)` for
  human-in-the-loop / async tools.

### `tools_add_builtin`
```
tools_add_builtin(path, app_name, agent_name, kind, args=None)
```
Builtins are **pre-instantiated tool objects** (not classes, not bare functions) — emit the bare name.
Known core builtins (no arg): `google_search`, `url_context`, `load_memory`, `preload_memory`,
`load_artifacts`, `get_user_choice`, `exit_loop`, `transfer_to_agent`, `enterprise_web_search`,
`google_maps_grounding`. The **only** arg-taking builtin is `vertex_ai_search` → pass
`args={"data_store_id": "..."}` (or `search_engine_id`) → renders `VertexAiSearchTool(data_store_id=…)`.

> **`request_input` does NOT exist in google-adk 2.1.0.** For human input use `get_user_choice` or a
> `long_running` function tool. Don't reach for `request_input` — it's not there.

### `tools_add_agent_tool`
```
tools_add_agent_tool(path, app_name, agent_name, target_agent)
```
Wrap an existing agent as a tool: `AgentTool(agent=<target>)`. `target_agent` must exist and differ
from `agent_name`. The target is **not** added as a sub_agent (single-parent rule). Topological codegen
defines the target before the wrapping agent.

### `tools_add_openapi`
```
tools_add_openapi(path, app_name, agent_name, spec, name=None)
```
Renders `<name> = OpenAPIToolset(spec_str=<spec>, spec_str_type="json")` and puts `<name>` **directly**
in `tools=[...]`. **Do not call `.get_tools()`** — a toolset is accepted as-is and expanded lazily by
ADK. `name` defaults to `<agent_name>_openapi`.

## <a name="optional-dependency"></a>Optional-dependency toolsets (codegen-only)

All four GCP/MCP toolsets are `BaseToolset`s → they go **directly** into `tools=[...]`.

- `tools_add_bigquery(… name=None, args=None)` → `<id> = BigQueryToolset(<args>)`. `args` are **source
  expressions** (e.g. `{"credentials_config": "my_creds"}`), not string literals.
- `tools_add_spanner(… name=None, args=None)` → `<id> = SpannerToolset(<args>)`.
- `tools_add_mcp_toolset(… transport, command=None, args=None, url=None, headers=None,
  tool_filter=None, name=None)`:
  - `transport="stdio"` → needs `command` (+ `args`) → `StdioConnectionParams(server_params=
    StdioServerParameters(command=…, args=[…]))`.
  - `transport="sse"` → needs `url` (+ `headers`) → `SseConnectionParams(url=…, headers={…})`.
  - `transport="http"` → needs `url` (+ `headers`) → `StreamableHTTPConnectionParams(url=…, headers={…})`.
  - `tool_filter` restricts exposed tools. This is how you make ANOTHER MCP server's tools available to
    an ADK agent (the reverse direction — exposing ADK tools AS MCP — is `mcp_bridge_*`, see `09-a2a.md`).
- `tools_add_apihub(… apihub_resource_name, name=None)` → `<id> = APIHubToolset(apihub_resource_name=…)`.
  Auth attachable via `tools_set_auth`.
- `tools_add_langchain(… import_line, tool_expr, name=None)` — you supply `import_line` (rendered
  verbatim, e.g. `from langchain_community.tools import WikipediaQueryRun`) and `tool_expr` (the
  construction, e.g. `WikipediaQueryRun(api_wrapper=wrapper)`). Renders `LangchainTool(tool=<tool_expr>)`.
- `tools_add_crewai(… import_line, tool_expr, name, description)` — same idea; `CrewaiTool` **requires**
  `name` (and the toolkit requires `description`). Renders `CrewaiTool(tool=<tool_expr>, name=…, description=…)`.

## <a name="auth"></a>Auth — `tools_set_auth`
```
tools_set_auth(path, app_name, agent_name, tool_name, scheme, credential)
```
- `tool_name` is the **toolset variable name** (the `name` you gave `add_openapi`/`add_apihub`/
  `add_mcp_toolset`). Only those three kinds accept auth (`OpenAPIToolset`/`APIHubToolset`/`McpToolset`
  have `auth_scheme`/`auth_credential`). **`BigQueryToolset`/`SpannerToolset` reject auth** — they use
  their `credentials_config` arg instead.
- `scheme` ∈ {`apikey`, `oauth2`, `service_account`, `bearer`}; `credential` is a dict of fields:
  - `apikey` → `{"api_key": "..."}` → `AuthCredential(auth_type=API_KEY, api_key=…)`
  - `bearer` → `{"token": "..."}` → `http=HttpAuth(scheme="bearer", credentials=HttpCredentials(token=…))`
  - `oauth2` → `{"client_id": "...", "client_secret": "...", "access_token"?}` → `OAuth2Auth(...)`
  - `service_account` → `{...}` → `ServiceAccount(...)`
- Renders only `auth_credential=AuthCredential(...)` (not `auth_scheme=` — ADK's `AuthScheme` is a
  `Union` with no single constructor; toolsets infer the scheme from the credential/spec).

## <a name="skills"></a>Agent Skills — the `skills_*` tools + `SkillToolset`

ADK's **Agent Skill Registry** (`google.adk.skills`) lets you author model-facing **skills** (a
folder with a `SKILL.md` of instructions + optional `references/`/`assets/`/`scripts/`, per the
[Agent Skills spec](https://agentskills.io)) and attach them to an agent via a **`SkillToolset`**.
A `SkillToolset` is a `BaseToolset` → it goes **directly** into `tools=[...]` like `OpenAPIToolset`.
The `skills_*` tools are their own domain, but the attach mechanism is a `tools=[...]` toolset, so
they live here in the tools reference.

| Tool | Key args | Notes |
|---|---|---|
| `skills_create` | `name, description, instruction` | Writes `<app_dir>/skills/<name>/SKILL.md`. `name` is **kebab-case** and **must equal the directory name** (ADK enforces this). `instruction` is the SKILL.md body. |
| `skills_list` | `(path, app_name)` | Lists the project's skills via the real `list_skills_in_dir` (frontmatter only). Read-only. |
| `skills_load` | `name` | Loads one skill fully via the real `load_skill_from_dir` (fields + resources). Read-only. |
| `skills_attach` | `agent_name, skill_names: list[str], name=None` | Adds a `skill_toolset` tool to an `LlmAgent` and regenerates. `name` is the toolset variable. |
| `skills_registry_info` | `(path, app_name)` | Reports the dir-backed registry inventory (id + name + description). |

What the model gets from a `SkillToolset`: the core tools `list_skills`, `load_skill`,
`load_skill_resource`, `run_skill_script` (and `search_skills` **only** if a concrete
`SkillRegistry` is supplied — google-adk 2.1.0 ships none, so the toolkit doesn't wire one). The
generated code loads each skill **from disk at the agent's runtime**:

```python
from pathlib import Path

from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

_ADK_SKILLS_DIR = Path(__file__).parent / "skills"
<var> = SkillToolset(skills=[load_skill_from_dir(_ADK_SKILLS_DIR / "greeter")])
```

and `<var>` goes into `tools=[...]`. Skill content is **never baked into `agent.py`**.

> **Experimental.** `SkillToolset` and the skill tools are `@experimental(FeatureName.SKILL_TOOLSET)`
> and emit a **`UserWarning`** (not a `DeprecationWarning`) when constructed — fine for the toolkit's
> `-W error::DeprecationWarning` gate. `run_skill_script` needs a `code_executor` (toolset- or
> agent-level); without one it returns a `NO_CODE_EXECUTOR` error (it does not crash construction).
> Skills are **local-directory only** in the toolkit (the GCS loaders need the `gcp` extra and are
> not wired).

## <a name="critical-facts"></a>Critical ADK facts (don't get these wrong)

- **Builtins are instances**, dropped in bare. Not classes (except `vertex_ai_search`), not functions
  (except `exit_loop`/`transfer_to_agent`, which the mcp_bridge wraps when converting).
- **`OpenAPIToolset` and all toolsets go directly into `tools=[...]`** — never `.get_tools()`.
- **A plain function** in `tools=[...]` is auto-wrapped to `FunctionTool` by `canonical_tools()` at
  runtime (it shows as a bare `function` until then). The toolkit emits the bare name.
- **`request_input` does not exist** in 2.1.0.
- `langchain`/`crewai` import paths the toolkit emits (`google.adk.tools.langchain_tool` /
  `crewai_tool`) re-export from `google.adk.integrations.*` and emit a deprecation warning when
  imported in your venv — harmless for authoring.
