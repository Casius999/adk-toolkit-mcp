"""`workflow` domain: ADK 2.0 graph-orchestration engine (``google.adk.workflow``).

A FastMCP sub-server mounted by the root server under the ``workflow`` namespace (tools exposed
as ``workflow_<name>`` on the client side). Functions named with **BARE** names (``create``,
``add_node``, ``add_edge``, …) — cf. ``docs/adk-api-notes/conventions.md``.

The engine builds **non-linear / conditional / cyclical** graphs of nodes (agents, functions,
join barriers) wired by edges — distinct from the linear ``SequentialAgent`` / ``ParallelAgent``
/ ``LoopAgent`` (deprecated in 2.1.0). A ``Workflow`` is a ``BaseNode``; the ADK ``AgentLoader``
accepts a ``BaseNode`` as ``root_agent``, so a workflow can be the app root and run via
``adk web`` / ``InMemoryRunner(node=...)`` (cf. ``docs/adk-api-notes/workflow.md``).

Each tool operates on ``(path, app_name, …)``: it loads the sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, applies an **immutable** mutation, validates the
graph, rewrites the sidecar, then **fully regenerates** ``agent.py`` (+ ``__init__.py``).
Everything is returned in the ``{ok, data, error}`` envelope; invalid inputs return ``err(...)``
(never an exception).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    WORKFLOW_START,
    ProjectModel,
    WorkflowEdgeSpec,
    WorkflowNodeSpec,
    WorkflowSpec,
    add_or_replace_edge,
    add_or_replace_node,
    add_or_update_workflow,
    detect_unconditional_cycle,
    is_identifier,
    load_model,
    regenerate,
    save_model,
    validate_workflow_edge_spec,
    validate_workflow_graph,
    validate_workflow_node_spec,
)
from ..project_model import (
    set_workflow_root as _model_set_workflow_root,
)
from ..workspace import Workspace

workflow_server: FastMCP = FastMCP("workflow")

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

    Returns ``{app_name, workflows, root, root_kind, sidecar, regenerated, changed}``.
    """
    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle detected at render time (agents graph)
        return err(str(exc))
    sidecar_changed = save_model(ws, model)
    return ok(
        {
            "app_name": app_name,
            "workflows": list(model.workflow_names()),
            "root": model.root,
            "root_kind": model.root_kind,
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


def _load_with_workflow(
    path: str, app_name: str, workflow: str
) -> tuple[ProjectModel, WorkflowSpec] | dict[str, Any]:
    """Load the model and fetch ``workflow``; return ``err(...)`` (dict) on any problem."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model
    wf = model.get_workflow(workflow)
    if wf is None:
        return err(f"Workflow not found: {workflow!r}. Create it first (workflow_create).")
    return model, wf


def _commit_workflow(
    path: str, app_name: str, model: ProjectModel, wf: WorkflowSpec
) -> dict[str, Any]:
    """Store the workflow in the model and commit (regenerate).

    Per-node / per-edge validity is checked by the callers (``validate_workflow_node_spec`` /
    ``validate_workflow_edge_spec``) BEFORE the mutation. The only **whole-graph** rule enforced
    eagerly here is the **unconditional cycle** (a hard structural error regardless of how
    complete the graph is): adding an unrouted edge that closes a cycle is rejected immediately.

    The remaining whole-graph rules (entry present, reachability, single terminal) are NOT
    enforced here — a graph under construction legitimately passes through incomplete states. They
    gate **rendering** (an incomplete workflow renders as a placeholder, keeping ``agent.py``
    importable) and are enforced at ``set_root`` (when the workflow must be a runnable root).
    """
    cycle_error = detect_unconditional_cycle(wf)
    if cycle_error is not None:
        return err(cycle_error)
    model = add_or_update_workflow(model, wf)
    return _commit(path, app_name, model)


# --------------------------------------------------------------------------- #
# MCP tools — graph construction
# --------------------------------------------------------------------------- #
@workflow_server.tool(tags={"workflow"})
def create(path: str, app_name: str, name: str, description: str = "") -> dict[str, Any]:
    """Create (or reset) an empty ``Workflow`` named ``name`` in the project, then regenerate.

    A workflow starts with no nodes/edges. Add nodes (``add_node``), wire them (``add_edge`` /
    ``set_entry``), then make it the app root (``set_root``).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(name):
        return err(f"Invalid workflow name: {name!r}. Expected a Python identifier.")

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    wf = WorkflowSpec(name=name, description=description)
    model = add_or_update_workflow(model, wf)
    return _commit(path, app_name, model)


@workflow_server.tool(tags={"workflow"})
def add_node(
    path: str,
    app_name: str,
    workflow: str,
    node_name: str,
    kind: str,
    agent: str = "",
    params: list[list[str]] | None = None,
    docstring: str = "",
    returns: str = "dict",
    body: str = "return {}",
) -> dict[str, Any]:
    """Add (or replace) a node in ``workflow``. ``kind`` ∈ ``{agent, function, join}``.

    - ``agent``: wraps an **existing** model agent. ``node_name`` must equal the agent's name
      (or pass ``agent=<existing_agent>`` and use that as ``node_name``); agents ARE ``BaseNode``s.
    - ``function``: a generated ``@node``-decorated ``def`` (a ``FunctionNode``). ``params`` is a
      list of ``[name, type, default|None]`` (besides the implicit ``ctx, node_input``); ``body``
      is the function body (``return <output>`` or ``return "<route>"`` for a routing node).
    - ``join``: a ``JoinNode`` fan-in barrier (waits for all predecessors).
    """
    load_res = _load_with_workflow(path, app_name, workflow)
    if isinstance(load_res, dict):
        return load_res
    model, wf = load_res

    if kind == "agent" and not agent:
        agent = node_name
    parsed_params = tuple(
        (str(p[0]), str(p[1]), (None if len(p) < 3 or p[2] is None else str(p[2])))
        for p in (params or [])
    )
    node = WorkflowNodeSpec(
        name=node_name,
        kind=kind,  # type: ignore[arg-type]  # validated just below
        agent=agent,
        params=parsed_params,
        docstring=docstring,
        returns=returns,
        body=body,
    )
    node_error = validate_workflow_node_spec(node, model)
    if node_error is not None:
        return err(node_error)

    wf = add_or_replace_node(wf, node)
    return _commit_workflow(path, app_name, model, wf)


@workflow_server.tool(tags={"workflow"})
def add_edge(
    path: str,
    app_name: str,
    workflow: str,
    source: str,
    target: str,
    route: str | None = None,
) -> dict[str, Any]:
    """Wire a directed edge ``source -> target`` in ``workflow`` (replace the same pair, else add).

    ``source`` may be ``"START"`` (the graph entry sentinel) — or use ``set_entry``. ``target``
    must be an existing node and may not be ``"START"``. ``route`` (optional) makes the edge
    **conditional**: the engine follows it only when the source node returns that route value
    (enables branching and routed loop-back cycles).
    """
    load_res = _load_with_workflow(path, app_name, workflow)
    if isinstance(load_res, dict):
        return load_res
    model, wf = load_res

    edge = WorkflowEdgeSpec(source=source, target=target, route=route)
    edge_error = validate_workflow_edge_spec(edge, wf)
    if edge_error is not None:
        return err(edge_error)

    wf = add_or_replace_edge(wf, edge)
    return _commit_workflow(path, app_name, model, wf)


@workflow_server.tool(tags={"workflow"})
def set_entry(path: str, app_name: str, workflow: str, node: str) -> dict[str, Any]:
    """Mark ``node`` as a workflow entry: a shortcut adding a ``START -> node`` edge."""
    load_res = _load_with_workflow(path, app_name, workflow)
    if isinstance(load_res, dict):
        return load_res
    model, wf = load_res

    edge = WorkflowEdgeSpec(source=WORKFLOW_START, target=node)
    edge_error = validate_workflow_edge_spec(edge, wf)
    if edge_error is not None:
        return err(edge_error)

    wf = add_or_replace_edge(wf, edge)
    return _commit_workflow(path, app_name, model, wf)


@workflow_server.tool(tags={"workflow"})
def set_root(path: str, app_name: str, name: str) -> dict[str, Any]:
    """Designate workflow ``name`` as the app's ``root_agent`` (a ``Workflow`` is a ``BaseNode``).

    Validates the full graph before committing, so the rendered ``root_agent = <workflow>`` is a
    runnable graph (entry edge present, all nodes reachable, no unconditional cycle, one terminal).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    wf = model.get_workflow(name)
    if wf is None:
        return err(f"Workflow not found: {name!r}. Create it before setting it as root.")
    graph_error = validate_workflow_graph(wf)
    if graph_error is not None:
        return err(graph_error)

    model = _model_set_workflow_root(model, name)
    return _commit(path, app_name, model)


# --------------------------------------------------------------------------- #
# MCP tools — read
# --------------------------------------------------------------------------- #
@workflow_server.tool(tags={"workflow"}, name="list")
def list_workflows(path: str, app_name: str) -> dict[str, Any]:
    """List the sidecar's workflows (name, node/edge counts) + the current root. Read-only.

    Named ``list_workflows`` in Python (so as not to shadow the ``list`` builtin in this module),
    but **registered under the BARE tool name ``list``** -> exposed as ``workflow_list`` on the
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
            "root_kind": model.root_kind,
            "workflows": [
                {"name": w.name, "nodes": len(w.nodes), "edges": len(w.edges)}
                for w in model.workflows
            ],
        }
    )


@workflow_server.tool(tags={"workflow"})
def get(path: str, app_name: str, name: str) -> dict[str, Any]:
    """Return the full spec of a workflow (nodes + edges, as serialized). Read-only."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    wf = model.get_workflow(name)
    if wf is None:
        return err(f"Workflow not found: {name!r}.")
    return ok(wf.to_dict())
