"""`tools` domain: attach ADK tools to agents (code-first, sidecar + regeneration).

A FastMCP sub-server mounted by the root server under the ``tools`` namespace (tools exposed as
``tools_<name>`` on the client side). Functions named with **BARE** names (``add_function``,
``add_long_running``, …) — cf. ``docs/adk-api-notes/conventions.md``.

Each tool operates on ``(path, app_name, agent_name, …)``: it loads the sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, **attaches/replaces** a tool spec on the
``agent_name`` agent ("append unique, replace by name" semantics via
:meth:`~adk_toolkit_mcp.project_model.ToolSpec.ref_key`), rewrites the sidecar, then **fully
regenerates** ``agent.py`` (+ ``__init__.py``). Everything is returned in the
``{ok, data, error}`` envelope; invalid inputs return ``err(...)`` (never an exception).

Pass **3a**: tools **without dependency** (no ``google-adk`` extra required): ``function``,
``long_running``, ``builtin`` (including ``vertex_ai_search``), ``agent_tool``, ``openapi``.

Pass **3b**: toolsets with an **optional dependency** (``google-adk[...]`` extras) —
**codegen-only** (the toolkit never imports these extras; it emits code that the user runs in
their own venv): ``add_bigquery``, ``add_spanner``, ``add_mcp_toolset``, ``add_apihub``,
``add_langchain``, ``add_crewai``, plus ``set_auth`` (attaches an auth sub-spec to a compatible
toolset).

The actual codegen and the semantics live in :mod:`adk_toolkit_mcp.project_model` (pure,
testable). See ``docs/adk-api-notes/tools.md`` for the confirmed ADK signatures (builtins =
instances, toolsets directly in ``tools=[...]``, a function auto-wrapped in a ``FunctionTool`` by
ADK, import paths + auth classes confirmed by introspection).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    BUILTIN_TOOLS,
    AuthSpec,
    ProjectModel,
    ToolSpec,
    add_or_replace_tool,
    add_or_update_agent,
    is_identifier,
    load_model,
    regenerate,
    save_model,
    validate_tool_spec,
)
from ..workspace import Workspace

tools_server: FastMCP = FastMCP("tools")

#: app_name = Python package identifier (both folder AND module name).
_APP_NAME_ERR = (
    "Invalid app_name: expected a Python identifier "
    "(letters, digits, underscore; not starting with a digit)."
)


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _load(path: str, app_name: str) -> ProjectModel | dict[str, Any]:
    """Load the model; return an ``err(...)`` (dict) if the sidecar is corrupt."""
    ws = _app_ws(path, app_name)
    try:
        return load_model(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _commit(path: str, app_name: str, model: ProjectModel) -> dict[str, Any]:
    """Save the sidecar + regenerate ``agent.py``. Converts a cycle into ``err``.

    Returns the common payload ``{app_name, agent, tools, sidecar, regenerated, changed}``.
    """
    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle detected at render time
        return err(str(exc))
    sidecar_changed = save_model(ws, model)
    return ok(
        {
            "app_name": app_name,
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


def _attach(path: str, app_name: str, agent_name: str, tool: ToolSpec) -> dict[str, Any]:
    """Validate then attach/replace ``tool`` on ``agent_name``, and commit. Shared by 1-5.

    Steps: validate ``app_name`` -> load the model -> require an existing ``llm`` agent (only an
    ``LlmAgent`` carries tools) -> validate the spec (with the model, for ``agent_tool``) ->
    attach (append unique / replace by name) -> commit (regenerate).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent_name: {agent_name!r}. Expected a Python identifier.")

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    agent = model.get(agent_name)
    if agent is None:
        return err(f"Agent not found: {agent_name!r}. Create it first (agents domain).")
    if agent.type != "llm":
        return err(
            f"The {agent_name!r} agent is of type {agent.type!r}; only 'llm' agents "
            "(LlmAgent) carry tools."
        )

    tool_error = validate_tool_spec(tool, model, agent_name)
    if tool_error is not None:
        return err(tool_error)

    updated = add_or_replace_tool(agent, tool)
    model = add_or_update_agent(model, updated)
    result = _commit(path, app_name, model)
    if result["ok"]:
        result["data"]["agent"] = agent_name
        result["data"]["tools"] = [t.ref_key() for t in updated.tool_specs()]
    return result


def _parse_params(params: list[dict[str, Any]]) -> tuple[tuple[str, str, str | None], ...] | str:
    """Normalize a ``[{"name","type","default"}]`` list into a typed tuple for ``ToolSpec``.

    Returns an error message (str) if an item is malformed. ``default`` is a **source literal**
    (already rendered) or ``None`` (parameter without default). E.g. ``{"name":"n","type":"int",
    "default":"0"}`` -> ``("n","int","0")``.
    """
    out: list[tuple[str, str, str | None]] = []
    for item in params:
        if not isinstance(item, dict) or "name" not in item:
            return f"Malformed parameter: {item!r}. Expected {{'name','type','default'?}}."
        name = str(item["name"])
        ptype = str(item.get("type", "str"))
        default = item.get("default")
        out.append((name, ptype, None if default is None else str(default)))
    return tuple(out)


def _replace_tool_fields(tool: ToolSpec, **changes: Any) -> ToolSpec:
    """Return an **immutable** copy of ``tool`` with the ``changes`` fields replaced.

    A thin wrapper around :func:`dataclasses.replace` (``ToolSpec`` is ``frozen``) — preserves
    the identity (``ref_key`` unchanged as long as neither ``name``/``kind`` is touched), so
    ``add_or_replace_tool`` indeed replaces the tool in place (no duplicate).
    """
    return replace(tool, **changes)


# --------------------------------------------------------------------------- #
# MCP tools — adding tools by kind
# --------------------------------------------------------------------------- #
@tools_server.tool(tags={"tools"})
def add_function(
    path: str,
    app_name: str,
    agent_name: str,
    func_name: str,
    params: list[dict[str, Any]],
    docstring: str,
    returns: str = "dict",
    body: str = "return {}",
) -> dict[str, Any]:
    """Attach a **function-tool** to ``agent_name``: generates ``def <func_name>(...)`` and
    places the bare name in ``tools=[...]`` (ADK auto-wraps it in a ``FunctionTool``).

    ``params``: list of ``{"name":.., "type":"str", "default":null}`` (``default`` = source
    literal or ``null``). Identifiers and types are validated. "Append unique / replace by name"
    semantics: re-attaching the same ``func_name`` replaces the definition.
    """
    parsed = _parse_params(params)
    if isinstance(parsed, str):
        return err(parsed)
    tool = ToolSpec(
        kind="function",
        name=func_name,
        params=parsed,
        docstring=docstring,
        returns=returns,
        body=body,
    )
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_long_running(
    path: str,
    app_name: str,
    agent_name: str,
    func_name: str,
    params: list[dict[str, Any]],
    docstring: str,
    returns: str = "dict",
    body: str = "return {}",
) -> dict[str, Any]:
    """Like :func:`add_function`, but wraps the function in
    ``LongRunningFunctionTool(func=<func_name>)`` (ADK long-running tool)."""
    parsed = _parse_params(params)
    if isinstance(parsed, str):
        return err(parsed)
    tool = ToolSpec(
        kind="long_running",
        name=func_name,
        params=parsed,
        docstring=docstring,
        returns=returns,
        body=body,
    )
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_builtin(
    path: str,
    app_name: str,
    agent_name: str,
    kind: str,
    args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attach an ADK **builtin** (``google_search``, ``url_context``, …) to ``agent_name``.

    ``kind`` must belong to the confirmed set :data:`BUILTIN_TOOLS`. For ``vertex_ai_search``,
    ``args`` must provide ``data_store_id`` (or ``search_engine_id``) -> rendered as
    ``VertexAiSearchTool(data_store_id="...")``. Core builtins are rendered by their bare name (a
    tool instance already exported by ADK).
    """
    if kind not in BUILTIN_TOOLS:
        return err(f"Unknown builtin: {kind!r}. Known: {', '.join(sorted(BUILTIN_TOOLS))}.")
    arg_pairs = tuple((str(k), str(v)) for k, v in (args or {}).items())
    tool = ToolSpec(kind="builtin", builtin_kind=kind, args=arg_pairs)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_agent_tool(
    path: str,
    app_name: str,
    agent_name: str,
    target_agent: str,
) -> dict[str, Any]:
    """Attach ``AgentTool(agent=<target_agent>)`` to ``agent_name`` (agent-as-tool delegation).

    ``target_agent`` must be an **existing** agent in the model and different from ``agent_name``
    (no self-wrapping). The generation order is topological (target defined before the wrapper);
    the target is **not** added as a ``sub_agent`` (ADK's single-parent rule: an agent wrapped as
    a tool is not a child).
    """
    tool = ToolSpec(kind="agent_tool", target_agent=target_agent)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_openapi(
    path: str,
    app_name: str,
    agent_name: str,
    spec: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Attach an ``OpenAPIToolset`` (built from the ``spec`` string) to ``agent_name``.

    Generates ``<name> = OpenAPIToolset(spec_str=<spec>, spec_str_type="json")`` at module level
    and places ``<name>`` **directly** in ``tools=[...]`` (confirmed: a toolset is accepted as-is,
    no need for ``.get_tools()``). ``name`` defaults to ``<agent_name>_openapi``.
    """
    toolset_name = name if name is not None else f"{agent_name}_openapi"
    if not is_identifier(toolset_name):
        return err(f"Invalid toolset name: {toolset_name!r}. Expected a Python identifier.")
    tool = ToolSpec(kind="openapi", name=toolset_name, spec=spec)
    return _attach(path, app_name, agent_name, tool)


# --------------------------------------------------------------------------- #
# MCP tools — pass 3b (optional-dependency toolsets, codegen-only)
# --------------------------------------------------------------------------- #
def _toolset_name(name: str | None, agent_name: str, suffix: str) -> str:
    """Toolset variable name: ``name`` if provided, otherwise ``<agent_name>_<suffix>``."""
    return name if name is not None else f"{agent_name}_{suffix}"


@tools_server.tool(tags={"tools"})
def add_bigquery(
    path: str,
    app_name: str,
    agent_name: str,
    name: str | None = None,
    args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attach a ``BigQueryToolset`` (``google-adk[bigquery]`` extra) to ``agent_name``.

    Generates ``<name> = BigQueryToolset(<args>)`` at module level and places ``<name>`` directly
    in ``tools=[...]``. ``args`` are **source expressions** (not string literals): e.g.
    ``{"bigquery_tool_config": "my_cfg"}`` references a variable you define elsewhere. ``name``
    defaults to ``<agent_name>_bigquery``. **Codegen-only**: the toolkit does not import the extra
    (cf. ``docs/adk-api-notes/tools.md``).
    """
    toolset_name = _toolset_name(name, agent_name, "bigquery")
    arg_pairs = tuple((str(k), str(v)) for k, v in (args or {}).items())
    tool = ToolSpec(kind="bigquery", name=toolset_name, args=arg_pairs)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_spanner(
    path: str,
    app_name: str,
    agent_name: str,
    name: str | None = None,
    args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attach a ``SpannerToolset`` (``google-adk[spanner]`` extra) to ``agent_name``.

    Like :func:`add_bigquery` but for Spanner: ``<name> = SpannerToolset(<args>)``. ``args`` =
    source expressions (e.g. ``{"credentials_config": "my_creds"}``). ``name`` defaults to
    ``<agent_name>_spanner``. **Codegen-only**.
    """
    toolset_name = _toolset_name(name, agent_name, "spanner")
    arg_pairs = tuple((str(k), str(v)) for k, v in (args or {}).items())
    tool = ToolSpec(kind="spanner", name=toolset_name, args=arg_pairs)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_mcp_toolset(
    path: str,
    app_name: str,
    agent_name: str,
    transport: str,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    tool_filter: list[str] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Attach an ``McpToolset`` (``google-adk[mcp]`` extra) to ``agent_name``.

    ``transport`` ∈ {``stdio``, ``sse``, ``http``}:

    - ``stdio``: ``command`` required (+ optional ``args``) -> ``StdioConnectionParams(
      server_params=StdioServerParameters(command=..., args=[...]))``;
    - ``sse`` / ``http``: ``url`` required (+ optional ``headers``) -> ``SseConnectionParams`` /
      ``StreamableHTTPConnectionParams(url=..., headers={...})``.

    ``tool_filter`` (optional) restricts the exposed tools. ``name`` defaults to
    ``<agent_name>_mcp``. The toolset goes directly into ``tools=[...]``. **Codegen-only**.
    """
    toolset_name = _toolset_name(name, agent_name, "mcp")
    tool = ToolSpec(
        kind="mcp_toolset",
        name=toolset_name,
        transport=transport,
        command=command or "",
        mcp_args=tuple(args or ()),
        url=url or "",
        headers=tuple((str(k), str(v)) for k, v in (headers or {}).items()),
        tool_filter=tuple(tool_filter or ()),
    )
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_apihub(
    path: str,
    app_name: str,
    agent_name: str,
    apihub_resource_name: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Attach an ``APIHubToolset`` to ``agent_name`` (Google Cloud API Hub).

    Generates ``<name> = APIHubToolset(apihub_resource_name="...")`` and places it in
    ``tools=[...]``. ``apihub_resource_name`` is the API Hub resource (e.g.
    ``projects/<p>/locations/<l>/apis/<a>``). ``name`` defaults to ``<agent_name>_apihub``. Auth
    attachable via :func:`set_auth`. **Codegen-only**.
    """
    toolset_name = _toolset_name(name, agent_name, "apihub")
    tool = ToolSpec(kind="apihub", name=toolset_name, apihub_resource_name=apihub_resource_name)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_langchain(
    path: str,
    app_name: str,
    agent_name: str,
    import_line: str,
    tool_expr: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Attach a LangChain tool wrapped via ``LangchainTool`` (``google-adk[community]`` extra).

    The toolkit does not know your LangChain tool: you provide ``import_line`` (rendered
    **verbatim**, e.g. ``from langchain_community.tools import WikipediaQueryRun``) and
    ``tool_expr`` (the construction expression, e.g. ``WikipediaQueryRun(api_wrapper=wrapper)``).
    Rendered: ``LangchainTool(tool=<tool_expr>)`` in ``tools=[...]``. ``name`` is accepted but
    currently not rendered (the LangChain wrapper derives its name). **Codegen-only**.
    """
    tool = ToolSpec(
        kind="langchain",
        import_line=import_line,
        tool_expr=tool_expr,
        name=name or "",
    )
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool(tags={"tools"})
def add_crewai(
    path: str,
    app_name: str,
    agent_name: str,
    import_line: str,
    tool_expr: str,
    name: str,
    description: str,
) -> dict[str, Any]:
    """Attach a CrewAI tool wrapped via ``CrewaiTool`` (``google-adk[community]`` extra).

    Like :func:`add_langchain` but for CrewAI. ``CrewaiTool`` **requires** a ``name``
    (keyword-only, confirmed); ``description`` is required here for an explicit rendering.
    Rendered: ``CrewaiTool(tool=<tool_expr>, name="...", description="...")``. **Codegen-only**.
    """
    tool = ToolSpec(
        kind="crewai",
        import_line=import_line,
        tool_expr=tool_expr,
        name=name,
        description=description,
    )
    return _attach(path, app_name, agent_name, tool)


# --------------------------------------------------------------------------- #
# MCP tool — auth (set_auth): attach an auth sub-spec to an existing toolset
# --------------------------------------------------------------------------- #
@tools_server.tool(tags={"tools"})
def set_auth(
    path: str,
    app_name: str,
    agent_name: str,
    tool_name: str,
    scheme: str,
    credential: dict[str, str],
) -> dict[str, Any]:
    """Attach an **auth** (``scheme`` + ``credential``) to a toolset already present on the agent.

    ``tool_name`` designates the **toolset variable** (the ``name`` passed to ``add_openapi`` /
    ``add_apihub`` / ``add_mcp_toolset``). Only these kinds accept auth (confirmed:
    ``OpenAPIToolset``/``APIHubToolset``/``McpToolset`` have ``auth_scheme``/``auth_credential``;
    ``BigQueryToolset``/``SpannerToolset`` do not -> rejected).

    ``scheme`` ∈ {``apikey``, ``oauth2``, ``service_account``, ``bearer``}. ``credential`` is a
    dict of fields (e.g. ``{"api_key": "..."}``, ``{"token": "..."}``, ``{"client_id": "...",
    "client_secret": "..."}``). Rendered: ``auth_credential=AuthCredential(...)`` on the toolset
    (+ ``google.adk.auth`` imports). Idempotent semantics (replace by name). **Codegen-only**.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent_name: {agent_name!r}. Expected a Python identifier.")

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    agent = model.get(agent_name)
    if agent is None:
        return err(f"Agent not found: {agent_name!r}. Create it first (agents domain).")

    # Find the target toolset (by its ``name`` variable) among the agent's tools.
    target = next(
        (
            t
            for t in agent.tool_specs()
            if t.name == tool_name
            and t.kind in ("openapi", "apihub", "mcp_toolset", "bigquery", "spanner")
        ),
        None,
    )
    if target is None:
        return err(
            f"Toolset not found on {agent_name!r}: {tool_name!r}. "
            "set_auth targets an existing toolset (openapi/apihub/mcp_toolset) by its 'name'."
        )

    cred_pairs = tuple((str(k), str(v)) for k, v in credential.items())
    updated_tool = _replace_tool_fields(target, auth=AuthSpec(scheme=scheme, credential=cred_pairs))

    # Re-validate (rejects auth on bigquery/spanner, unknown scheme, missing required field…).
    tool_error = validate_tool_spec(updated_tool, model, agent_name)
    if tool_error is not None:
        return err(tool_error)

    updated_agent = add_or_replace_tool(agent, updated_tool)
    model = add_or_update_agent(model, updated_agent)
    result = _commit(path, app_name, model)
    if result["ok"]:
        result["data"]["agent"] = agent_name
        result["data"]["tools"] = [t.ref_key() for t in updated_agent.tool_specs()]
    return result


# --------------------------------------------------------------------------- #
# MCP tool — read
# --------------------------------------------------------------------------- #
@tools_server.tool(tags={"tools"}, name="list")
def list_tools_for_agent(path: str, app_name: str, agent_name: str) -> dict[str, Any]:
    """List the tools attached to ``agent_name`` (kind + summary detail). Read-only.

    Named ``list_tools_for_agent`` in Python (so as not to shadow the ``list`` builtin in this
    module), but **registered under the BARE tool name ``list``** -> exposed as ``tools_list``.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent_name: {agent_name!r}. Expected a Python identifier.")

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    agent = model.get(agent_name)
    if agent is None:
        return err(f"Agent not found: {agent_name!r}.")

    return ok(
        {
            "app_name": app_name,
            "agent": agent_name,
            "tools": [_tool_summary(t) for t in agent.tool_specs()],
        }
    )


def _tool_summary(tool: ToolSpec) -> dict[str, Any]:
    """Summary of a ``ToolSpec`` for ``list`` (per kind)."""
    summary: dict[str, Any] = {"kind": tool.kind, "ref_key": tool.ref_key()}
    if tool.kind in ("function", "long_running"):
        summary["name"] = tool.name
        summary["params"] = [{"name": n, "type": t, "default": d} for (n, t, d) in tool.params]
        summary["returns"] = tool.returns
    elif tool.kind == "builtin":
        summary["builtin_kind"] = tool.builtin_kind
        if tool.args:
            summary["args"] = dict(tool.args)
    elif tool.kind == "agent_tool":
        summary["target_agent"] = tool.target_agent
    elif tool.kind in ("openapi", "bigquery", "spanner"):
        summary["name"] = tool.name
        if tool.args:
            summary["args"] = dict(tool.args)
    elif tool.kind == "mcp_toolset":
        summary["name"] = tool.name
        summary["transport"] = tool.transport
    elif tool.kind == "apihub":
        summary["name"] = tool.name
        summary["apihub_resource_name"] = tool.apihub_resource_name
    elif tool.kind in ("langchain", "crewai"):
        summary["tool_expr"] = tool.tool_expr
        if tool.name:
            summary["name"] = tool.name
    if tool.auth is not None:
        summary["auth"] = {"scheme": tool.auth.scheme}
    return summary
