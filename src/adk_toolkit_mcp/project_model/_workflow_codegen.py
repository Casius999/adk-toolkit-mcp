"""Ruff-stable rendering of ``google.adk.workflow`` graphs (internal).

A **private** module (``_`` prefix): it renders a :class:`WorkflowSpec` into the source blocks
that materialize a ``Workflow(name=..., edges=[...])`` in ``agent.py`` — function nodes as
``@node``-decorated ``def``s, join nodes as ``JoinNode(...)`` assignments, and the edge list
(unconditional ``(src, dst)`` tuples + conditional ``(src, {route: dst})`` dicts), grouped so
the output is already in the form ``ruff format`` would produce (idempotent ``format --check``).

Consumed by :mod:`adk_toolkit_mcp.project_model.render` (which assembles the complete module:
imports, agents, then workflows).
"""

from __future__ import annotations

from ._codegen import _py_str, _render_param
from .specs import (
    _WORKFLOW_IMPORT_MODULE,
    LINE_LENGTH,
    WORKFLOW_START,
    WorkflowEdgeSpec,
    WorkflowNodeSpec,
    WorkflowSpec,
)

#: Fixed positional parameters every generated function node receives (cf. workflow.md). User
#: ``params`` (state-bound) are appended after these.
_NODE_FIXED_PARAMS = "ctx, node_input"


def _render_function_node_def(node: WorkflowNodeSpec) -> str:
    """Render a ``@node``-decorated ``def`` block for a ``function`` workflow node.

    Signature: ``def <name>(ctx, node_input[, <params>]) -> <returns>:``; 1-line docstring; body.
    The ``@node`` decorator (imported from ``google.adk.workflow``) turns the callable into a
    ``FunctionNode`` named after the function. Ends with a single ``\\n`` (the module renderer
    handles inter-block spacing PEP 8 / ruff-style).
    """
    extra = ", ".join(_render_param(n, t, d) for (n, t, d) in node.params)
    params = f"{_NODE_FIXED_PARAMS}, {extra}" if extra else _NODE_FIXED_PARAMS
    doc = (node.docstring or node.name).replace("\\", "\\\\").replace('"', '\\"')
    doc_line = f'    """{doc}"""\n'
    body_lines = node.body.splitlines() or ["return {}"]
    body = "".join(f"    {line}\n" for line in body_lines)
    return f"@node\ndef {node.name}({params}) -> {node.returns}:\n{doc_line}{body}"


def _render_join_node_assign(node: WorkflowNodeSpec) -> str:
    """Render ``<name> = JoinNode(name="<name>")`` (a fan-in barrier). Ends with ``\\n``."""
    return f"{node.name} = JoinNode(name={_py_str(node.name)})\n"


def _edge_endpoint_ref(name: str, by_name: dict[str, WorkflowNodeSpec]) -> str:
    """Source expression for an edge endpoint name.

    ``START`` -> the imported ``START`` sentinel; an ``agent`` node -> the wrapped agent variable
    (``agent`` or the node name); ``function``/``join`` nodes -> their own variable name.
    """
    if name == WORKFLOW_START:
        return WORKFLOW_START
    node = by_name.get(name)
    if node is not None and node.kind == "agent":
        return node.agent_ref()
    return name


def _render_edges_value(spec: WorkflowSpec) -> str:
    """Render the ``edges=[...]`` value, ruff-stable, grouping conditional edges by source.

    - Unconditional edges (``route is None``) render as ``(src, dst)`` tuples.
    - Conditional edges sharing a source collapse into a single ``(src, {route: dst, ...})``
      tuple (preserving first-seen order of routes). This matches the ADK dict-edge form and
      keeps the rendered graph compact.

    The list is always emitted multi-line (one edge per line, indent 8, trailing comma) — the
    canonical ``ruff format`` shape for a non-trivial list, so ``format --check`` reformats
    nothing.
    """
    by_name = {n.name: n for n in spec.nodes}

    # Group conditional edges by source (ordered); keep unconditional edges as standalone items,
    # preserving overall first-appearance order of sources.
    order: list[str] = []
    conditional: dict[str, list[WorkflowEdgeSpec]] = {}
    unconditional: list[WorkflowEdgeSpec] = []
    for edge in spec.edges:
        if edge.route is None:
            unconditional.append(edge)
            if edge.source not in order:
                order.append(edge.source)
        else:
            if edge.source not in conditional:
                conditional[edge.source] = []
                if edge.source not in order:
                    order.append(edge.source)
            conditional[edge.source].append(edge)

    items: list[str] = []
    seen_uncond: set[int] = set()
    for src in order:
        if src in conditional:
            mapping = ", ".join(
                f"{_py_str(e.route)}: {_edge_endpoint_ref(e.target, by_name)}"
                for e in conditional[src]
                if e.route is not None
            )
            items.append(f"({_edge_endpoint_ref(src, by_name)}, {{{mapping}}})")
        # Emit unconditional edges from this source (in original order).
        for i, e in enumerate(unconditional):
            if e.source == src and i not in seen_uncond:
                seen_uncond.add(i)
                items.append(
                    f"({_edge_endpoint_ref(src, by_name)}, {_edge_endpoint_ref(e.target, by_name)})"
                )

    if not items:
        return "[]"
    body = "".join(f"        {it},\n" for it in items)
    return f"[\n{body}    ]"


def _render_workflow_call(spec: WorkflowSpec) -> str:
    """Render ``<name> = Workflow(name=..., edges=[...])`` (trailing newline)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("edges", _render_edges_value(spec)))
    kwargs = "".join(f"    {k}={v},\n" for k, v in pairs)
    return f"{spec.name} = Workflow(\n{kwargs})\n"


def workflow_imports(spec: WorkflowSpec) -> tuple[str, ...]:
    """ADK ``google.adk.workflow`` names imported for this workflow (a single ``from`` line).

    Always imports ``Workflow`` and ``START``. Adds ``node`` if any function node exists and
    ``JoinNode`` if any join node exists. The names are sorted by the module renderer
    (isort-clean) — here we just collect the set.
    """
    names = {"START", "Workflow"}
    if any(n.kind == "function" for n in spec.nodes):
        names.add("node")
    if any(n.kind == "join" for n in spec.nodes):
        names.add("JoinNode")
    return (f"from {_WORKFLOW_IMPORT_MODULE} import {', '.join(sorted(names))}",)


def render_workflow_blocks(spec: WorkflowSpec, *, complete: bool) -> tuple[list[str], list[str]]:
    """Render a workflow -> ``(helper_blocks, assign_blocks)``.

    - ``helper_blocks``: the ``@node`` function ``def``s (top-level, emitted before agents/joins).
      Always rendered — a function ``def`` is valid on its own.
    - ``assign_blocks``: join-node assignments followed by the ``Workflow(...)`` assignment
      (emitted after the agents so agent variables referenced in edges already exist).

    ``complete`` reflects whether the graph validates (entry edge present, all nodes reachable,
    single terminal, no unconditional cycle). When **incomplete** (a graph still being built),
    the ``Workflow(...)`` call would raise at construction, so we emit a **placeholder comment**
    instead — keeping the generated ``agent.py`` importable. Once the workflow is wired and set as
    root (which requires a valid graph), the real ``Workflow(...)`` is emitted.

    Function-node ``def``s are separated from the assignment block by the module renderer's PEP 8
    spacing rules (2 blank lines around a ``def``).
    """
    helpers = [_render_function_node_def(n) for n in spec.nodes if n.kind == "function"]
    # Join-node assignments are valid Python on their own — always emit them. Only the
    # ``Workflow(...)`` call can raise at construction on an incomplete graph, so that single
    # block is replaced by a placeholder comment until the graph validates.
    joins = [_render_join_node_assign(n) for n in spec.nodes if n.kind == "join"]
    if complete:
        tail = _render_workflow_call(spec)
    else:
        tail = (
            f"# Workflow {spec.name!r} is not fully wired yet "
            "(needs a START entry, all nodes reachable, one terminal). "
            "Wire it with workflow_add_edge / workflow_set_entry.\n"
        )
    return helpers, [*joins, tail]


# Re-export for callers that fold long single lines (kept consistent with the rest of the
# renderer's width budget).
__all__ = [
    "LINE_LENGTH",
    "render_workflow_blocks",
    "workflow_imports",
]
