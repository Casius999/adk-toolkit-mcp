"""`mcp_bridge` domain: expose ADK tools **as** MCP tools (P4b).

This domain is the "ADK → MCP" bridge: it converts ADK `BaseTool` objects into **MCP tool
schemas** (`mcp.types.Tool`: ``{name, description, inputSchema}``) via the official function
``google.adk.tools.mcp_tool.conversion_utils.adk_to_mcp_tool_type``. This is the operation
distinct from the P1 ``tools`` domain (which *consumes* MCP servers via ``McpToolset``): here we
make an agent's tools **publishable** as an MCP server.

The ``mcp`` package is a **CORE** dependency (``fastmcp`` depends on it): this domain is therefore
fully testable in CI **without any extra** (unlike ``a2a``). Cf.
``docs/adk-api-notes/a2a-mcp-bridge.md`` for the confirmed signatures and the FUNCTIONAL result
(the MCP schema obtained from a real ADK tool).

Tools exposed under ``namespace="mcp_bridge"`` → ``mcp_bridge_<name>``. BARE names:

- ``expose_adk_tools(path, app_name, agent_name)`` — imports the project's ``root_agent``, locates
  the ``agent_name`` agent (via ``BaseAgent.find_agent``), normalizes its tools to ``BaseTool``
  (via ``await agent.canonical_tools()``, which wraps bare functions in ``FunctionTool``), and
  renders the list of MCP schemas. ROBUST path: we operate on the ACTUALLY built agent (the
  sidecar specs become the real ADK objects), not on a re-derivation of the specs.
- ``convert_builtin(kind)`` — instantiates a single "core" ADK builtin by its ``kind`` (e.g.
  ``google_search``) and returns its MCP schema. Handy to inspect an isolated builtin.

Each tool returns the ``{ok, data, error}`` envelope; invalid inputs → ``err(...)`` (never an
exception that propagates). The ADK imports are **lazy** (at the call site).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import CORE_BUILTINS, is_identifier
from ..run_core import RootAgentImportError, import_root_agent

if TYPE_CHECKING:  # pragma: no cover - hints only, real imports are lazy
    from google.adk.tools import BaseTool

mcp_bridge_server: FastMCP = FastMCP("mcp_bridge")

#: app_name = Python package identifier (both folder AND module name).
_APP_NAME_ERR = (
    "Invalid app_name: expected a Python identifier "
    "(letters, digits, underscore; not starting with a digit)."
)


# --------------------------------------------------------------------------- #
# ADK BaseTool -> MCP schema (mcp.types.Tool) conversion
# --------------------------------------------------------------------------- #
def _to_mcp_schema(tool: BaseTool) -> dict[str, Any]:
    """Convert an ADK ``BaseTool`` into a ``{name, description, inputSchema}`` dict (MCP form).

    Delegates to ``adk_to_mcp_tool_type`` (which returns a ``mcp.types.Tool``) then exposes only
    the three fields we care about. ``inputSchema`` is already a JSON-Schema dict (empty ``{}`` for
    a builtin with no declared parameters, e.g. ``google_search``).
    """
    from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

    mcp_tool = adk_to_mcp_tool_type(tool)
    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description,
        "inputSchema": mcp_tool.inputSchema,
    }


def _builtin_to_base_tool(kind: str) -> BaseTool:
    """Instantiate a "core" builtin (``kind`` ∈ :data:`CORE_BUILTINS`) into a ``BaseTool``.

    Some "core builtins" are already ``BaseTool`` **instances** (e.g. ``google_search`` =
    ``GoogleSearchTool()``); others are plain **functions** (``exit_loop``, ``transfer_to_agent``)
    which we then wrap in a ``FunctionTool`` so they can be converted.
    """
    import google.adk.tools as adk_tools
    from google.adk.tools import BaseTool as _BaseTool
    from google.adk.tools import FunctionTool

    obj = getattr(adk_tools, kind)
    if isinstance(obj, _BaseTool):
        return obj
    # Bare function (exit_loop / transfer_to_agent) -> we wrap it in a FunctionTool.
    return FunctionTool(obj)


# --------------------------------------------------------------------------- #
# MCP tool — convert_builtin
# --------------------------------------------------------------------------- #
@mcp_bridge_server.tool(tags={"mcp_bridge"})
def convert_builtin(kind: str) -> dict[str, Any]:
    """Instantiate a "core" ADK builtin by ``kind`` and return its MCP schema.

    E.g. ``convert_builtin("google_search")`` → ``{name, description, inputSchema}`` (a flattened
    ``mcp.types.Tool``). Only **core** builtins (no required argument) are supported here —
    ``vertex_ai_search`` requires a ``data_store_id`` and must be attached to an agent then exposed
    via :func:`expose_adk_tools`. An unknown ``kind`` → ``err`` listing the known kinds.

    The ``mcp`` package is core → this tool works without any extra (testable in CI).
    """
    if kind not in CORE_BUILTINS:
        return err(
            f"Unknown core builtin: {kind!r}. Known: {', '.join(sorted(CORE_BUILTINS))}. "
            "(Arg-requiring builtins like 'vertex_ai_search': attach them to an agent then "
            "use mcp_bridge_expose_adk_tools.)"
        )
    try:
        tool = _builtin_to_base_tool(kind)
        schema = _to_mcp_schema(tool)
    except Exception as exc:  # noqa: BLE001 - best-effort conversion, we return a clean err
        return err(f"Failed to convert the builtin {kind!r} into an MCP schema: {exc}")
    return ok({"kind": kind, "tool": schema})


# --------------------------------------------------------------------------- #
# MCP tool — expose_adk_tools
# --------------------------------------------------------------------------- #
@mcp_bridge_server.tool(tags={"mcp_bridge"})
async def expose_adk_tools(path: str, app_name: str, agent_name: str) -> dict[str, Any]:
    """Convert a project agent's ADK tools into **MCP tool schemas**.

    Realizes "expose the ADK tools AS MCP tools": imports the project's ``root_agent``
    (``<path>/<app_name>/agent.py``), locates the ``agent_name`` agent in the tree (via
    ``BaseAgent.find_agent`` — also works for the root itself), normalizes its tools to
    ``BaseTool`` (``await agent.canonical_tools()`` wraps bare functions in ``FunctionTool``), then
    converts each via ``adk_to_mcp_tool_type``.

    Returns ``{app_name, agent_name, count, tools: [{name, description, inputSchema}, ...]}``. An
    agent without tools returns an empty list (``count=0``, not an error). Clean errors (``err``)
    if: ``app_name``/``agent_name`` invalid, ``agent.py`` missing/unreadable
    (``RootAgentImportError``), agent not found in the tree, or agent without tool capability (e.g.
    a Sequential/Parallel/Loop workflow agent has no ``canonical_tools``).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent name: {agent_name!r}. Expected a Python identifier.")

    try:
        root_agent = import_root_agent(path, app_name)
    except RootAgentImportError as exc:
        return err(str(exc))

    agent = root_agent.find_agent(agent_name)
    if agent is None:
        return err(
            f"Agent not found in the root_agent tree: {agent_name!r}. "
            "Check the name (the root and all its sub-agents are inspected)."
        )

    # Workflow agents (Sequential/Parallel/Loop) have no tools: no canonical_tools.
    canonical = getattr(agent, "canonical_tools", None)
    if canonical is None:
        return err(
            f"The {agent_name!r} agent (type {type(agent).__name__}) carries no ADK tools "
            "(only LLM-type agents expose tools convertible to MCP)."
        )

    try:
        base_tools = await canonical()
        tools = [_to_mcp_schema(t) for t in base_tools]
    except Exception as exc:  # noqa: BLE001 - we convert any error into an actionable err
        return err(f"Failed to convert {agent_name!r}'s tools into MCP schemas: {exc}")

    return ok(
        {
            "app_name": app_name,
            "agent_name": agent_name,
            "count": len(tools),
            "tools": tools,
        }
    )
