# adk-toolkit-mcp — Project-wide tool naming conventions

Captured 2026-06-01. fastmcp **3.3.1**, Python 3.12.

## Domain sub-server pattern

Each domain lives in `src/adk_toolkit_mcp/domains/<domain>.py` and exports a
`FastMCP("<domain>")` instance named `<domain>_server`.

```python
# domains/project.py
project_server: FastMCP = FastMCP("project")
```

## Bare tool function names

Tool functions are registered with **bare names** (no domain prefix):

```python
@project_server.tool
def create(...) -> dict[str, Any]: ...

@project_server.tool
def inspect(...) -> dict[str, Any]: ...
```

Bare names like `create`, `run`, `add_function`, `set_env`, etc. are preferred.

## Mount in `server.py` with `namespace=`

Mount each sub-server in `build_server()` using the `namespace=` parameter
(**not** the deprecated `prefix=` parameter):

```python
mcp.mount(project_server, namespace="project")
```

Using `prefix=` emits a `DeprecationWarning` in fastmcp 3.3.1 and must be avoided.

## Exposed tool name = `<domain>_<bare>` (single prefix)

FastMCP concatenates `namespace + "_" + bare_name`, so a bare function `create`
mounted under `namespace="project"` is exposed to MCP clients as `project_create`.

| Domain | Bare function | Exposed name |
|--------|--------------|--------------|
| project | `create` | `project_create` |
| project | `inspect` | `project_inspect` |
| project | `set_env` | `project_set_env` |
| project | `add_extra` | `project_add_extra` |
| project | `agent_config` | `project_agent_config` |

Never name a tool function with the domain prefix already included (e.g.
`project_create`) — that would produce a double-prefixed exposed name
`project_project_create`.

## Envelope `{ok, data, error}` on every tool

Every tool must return the uniform envelope:

```python
{"ok": True,  "data": <payload>, "error": None}   # success
{"ok": False, "data": None,      "error": "<msg>"} # failure
```

Use the helpers from `adk_toolkit_mcp.envelope`:

```python
from ..envelope import err, ok

return ok({"key": "value"})
return err("Something went wrong.")
```

## Summary checklist for new domains

- [ ] `FastMCP("<domain>")` instance named `<domain>_server`
- [ ] Tool functions use bare names (no domain prefix)
- [ ] Every tool returns the `{ok, data, error}` envelope
- [ ] Mounted in `server.py` with `mcp.mount(<domain>_server, namespace="<domain>")`
- [ ] No `prefix=` usage anywhere (deprecated)
- [ ] Tests import bare names directly; mounted-client tests call the exposed
      `<domain>_<bare>` name and assert no double-prefix names exist
