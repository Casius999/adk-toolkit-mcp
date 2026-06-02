"""`agents` domain: ADK multi-agent composition (code-first, sidecar + regeneration).

A FastMCP sub-server mounted by the root server under the ``agents`` namespace (tools exposed as
``agents_<name>`` on the client side). Functions named with **BARE** names (``create_llm``,
``create_sequential``, …) — cf. ``docs/adk-api-notes/conventions.md``.

Each tool operates on ``(path, app_name, …)``: it loads the sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, applies an **immutable** mutation, rewrites it,
then **fully regenerates** ``agent.py`` (+ ``__init__.py``) via
:class:`~adk_toolkit_mcp.workspace.Workspace`. Everything is returned in the ``{ok, data, error}``
envelope; invalid inputs return ``err(...)`` (never an exception).

The actual rendering and the model semantics live in :mod:`adk_toolkit_mcp.project_model` (pure,
testable). See ``docs/adk-api-notes/agents.md`` for the confirmed ADK signatures (and the
deprecation of the workflow agents in 2.1.0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    AgentSpec,
    ProjectModel,
    add_or_update_agent,
    is_identifier,
    load_model,
    regenerate,
    save_model,
    validate_spec,
)
from ..project_model import (
    set_root as _model_set_root,
)
from ..workspace import Workspace

agents_server: FastMCP = FastMCP("agents")

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

    Returns the common payload ``{app_name, agents, root, sidecar, regenerated, changed}``.
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
            "agents": list(model.agent_names()),
            "root": model.root,
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


def _add_spec(path: str, app_name: str, spec: AgentSpec) -> dict[str, Any]:
    """Validate the spec, add/update it in the model, commit. Shared by 1-5."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    spec_error = validate_spec(spec)
    if spec_error is not None:
        return err(spec_error)

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    missing = [s for s in spec.sub_agents if model.get(s) is None and s != spec.name]
    if missing:
        return err(
            f"sub_agents not found: {', '.join(missing)}. "
            "Create them first (creation order is free, but they must exist)."
        )

    model = add_or_update_agent(model, spec)
    return _commit(path, app_name, model)


# --------------------------------------------------------------------------- #
# MCP tools — creation by type
# --------------------------------------------------------------------------- #
@agents_server.tool(tags={"agents"})
def create_llm(
    path: str,
    app_name: str,
    name: str,
    model: str = "gemini-2.5-flash",
    instruction: str = "",
    description: str = "",
    output_key: str | None = None,
) -> dict[str, Any]:
    """Add/update an ``LlmAgent`` agent in the model, then regenerate ``agent.py``."""
    if not model.strip():
        return err("model is empty.")
    spec = AgentSpec(
        name=name,
        type="llm",
        model=model,
        instruction=instruction,
        description=description,
        output_key=output_key,
    )
    return _add_spec(path, app_name, spec)


@agents_server.tool(tags={"agents"})
def create_sequential(
    path: str,
    app_name: str,
    name: str,
    sub_agents: list[str],
    description: str = "",
) -> dict[str, Any]:
    """Add/update a ``SequentialAgent`` orchestrating ``sub_agents`` (which must exist)."""
    spec = AgentSpec(
        name=name,
        type="sequential",
        sub_agents=tuple(sub_agents),
        description=description,
    )
    return _add_spec(path, app_name, spec)


@agents_server.tool(tags={"agents"})
def create_parallel(
    path: str,
    app_name: str,
    name: str,
    sub_agents: list[str],
    description: str = "",
) -> dict[str, Any]:
    """Add/update a ``ParallelAgent`` orchestrating ``sub_agents`` (which must exist)."""
    spec = AgentSpec(
        name=name,
        type="parallel",
        sub_agents=tuple(sub_agents),
        description=description,
    )
    return _add_spec(path, app_name, spec)


@agents_server.tool(tags={"agents"})
def create_loop(
    path: str,
    app_name: str,
    name: str,
    sub_agents: list[str],
    max_iterations: int = 3,
    description: str = "",
) -> dict[str, Any]:
    """Add/update a ``LoopAgent`` (``max_iterations`` > 0 required)."""
    spec = AgentSpec(
        name=name,
        type="loop",
        sub_agents=tuple(sub_agents),
        max_iterations=max_iterations,
        description=description,
    )
    return _add_spec(path, app_name, spec)


@agents_server.tool(tags={"agents"})
def create_custom(
    path: str,
    app_name: str,
    name: str,
    description: str = "",
) -> dict[str, Any]:
    """Add/update a custom agent: ``BaseAgent`` subclass (stub) + instance."""
    spec = AgentSpec(name=name, type="custom", description=description)
    return _add_spec(path, app_name, spec)


# --------------------------------------------------------------------------- #
# MCP tools — composition / root / read
# --------------------------------------------------------------------------- #
@agents_server.tool(tags={"agents"})
def compose(
    path: str,
    app_name: str,
    name: str,
    sub_agents: list[str],
) -> dict[str, Any]:
    """Replace the ``sub_agents`` of an **existing** agent (validates their existence)."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    current = model.get(name)
    if current is None:
        return err(f"Agent not found: {name!r}. Create it before composing.")
    if current.type == "custom":
        return err("A custom agent has no sub_agents managed by the model.")

    missing = [s for s in sub_agents if model.get(s) is None and s != name]
    if missing:
        return err(f"sub_agents not found: {', '.join(missing)}.")
    if name in sub_agents:
        return err(f"An agent cannot reference itself: {name!r}.")

    updated = AgentSpec(
        name=current.name,
        type=current.type,
        model=current.model,
        instruction=current.instruction,
        description=current.description,
        output_key=current.output_key,
        tools=current.tools,
        sub_agents=tuple(sub_agents),
        max_iterations=current.max_iterations,
    )
    model = add_or_update_agent(model, updated)
    return _commit(path, app_name, model)


@agents_server.tool(tags={"agents"})
def set_root(path: str, app_name: str, name: str) -> dict[str, Any]:
    """Designate ``name`` as the sidecar's ``root_agent``, then regenerate ``agent.py``."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    if model.get(name) is None:
        return err(f"Agent not found: {name!r}. Create it before setting it as root.")

    model = _model_set_root(model, name)
    return _commit(path, app_name, model)


@agents_server.tool(tags={"agents"})
def as_tool(path: str, app_name: str, agent_name: str) -> dict[str, Any]:
    """Return the **source snippet** to wrap ``agent_name`` via ``AgentTool``.

    A composition helper (P3 ``tools``): mutates no file. The snippet shows the import and the
    usage ``LlmAgent(..., tools=[AgentTool(agent=<agent_name>)])``.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent name: {agent_name!r}.")

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model
    if model.get(agent_name) is None:
        return err(f"Agent not found: {agent_name!r}.")

    snippet = (
        "from google.adk.tools import AgentTool\n"
        f"{agent_name}_tool = AgentTool(agent={agent_name})\n"
        f"# Then: LlmAgent(..., tools=[{agent_name}_tool])"
    )
    return ok(
        {
            "agent_name": agent_name,
            "import": "from google.adk.tools import AgentTool",
            "expression": f"AgentTool(agent={agent_name})",
            "snippet": snippet,
        }
    )


@agents_server.tool(tags={"agents"}, name="list")
def list_agents(path: str, app_name: str) -> dict[str, Any]:
    """List the sidecar's agents (name, type, root). Read-only.

    Named ``list_agents`` in Python (so as not to shadow the ``list`` builtin in this module),
    but **registered under the BARE tool name ``list``** -> exposed as ``agents_list`` on the
    client side.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    return ok(
        {
            "app_name": app_name,
            "root": model.root,
            "agents": [{"name": a.name, "type": a.type} for a in model.agents],
        }
    )


@agents_server.tool(tags={"agents"})
def get(path: str, app_name: str, name: str) -> dict[str, Any]:
    """Return the full spec of a sidecar agent (as serialized). Read-only."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    spec = model.get(name)
    if spec is None:
        return err(f"Agent not found: {name!r}.")
    return ok(spec.to_dict())
