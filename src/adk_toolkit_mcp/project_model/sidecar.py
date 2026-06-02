"""Sidecar I/O, immutable mutations and spec validation.

This module carries the **non-rendering logic** of the project model:

- validation: :func:`validate_spec` (agent) and :func:`validate_tool_spec` (tool), plus the
  internal validators ``_validate_mcp`` / ``_validate_auth`` / ``_is_allowed_type``;
- **immutable** mutations: :func:`add_or_update_agent`, :func:`set_root`,
  :func:`add_or_replace_tool` (always return a new object);
- sidecar I/O: :func:`load_model` / :func:`save_model` (read/write of
  ``.adk_toolkit/agents.json`` via a :class:`~adk_toolkit_mcp.workspace.Workspace`).

Imports the dataclasses/constants from :mod:`adk_toolkit_mcp.project_model.specs`. The generation
of ``agent.py`` lives separately in :mod:`adk_toolkit_mcp.project_model.render`.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from ..workspace import Workspace
from .specs import (
    _AGENT_TYPES,
    _ALLOWED_PARAM_TYPES,
    _AUTH_CAPABLE_KINDS,
    _AUTH_SCHEMES,
    _CALLBACK_HOOKS,
    _MCP_TRANSPORTS,
    _POLICY_HOOKS,
    _POLICY_KINDS,
    _TOOL_KINDS,
    _WORKFLOW_NODE_KINDS,
    ARG_BUILTINS,
    BUILTIN_TOOLS,
    SIDECAR_PATH,
    WORKFLOW_START,
    AgentSpec,
    CallbackSpec,
    ProjectModel,
    ToolSpec,
    WorkflowEdgeSpec,
    WorkflowNodeSpec,
    WorkflowSpec,
    is_identifier,
    is_skill_name,
)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_spec(spec: AgentSpec) -> str | None:
    """Return an error message if the spec is invalid, otherwise None."""
    if not is_identifier(spec.name):
        return (
            f"Invalid agent name: {spec.name!r}. Expected a Python identifier "
            "(letters, digits, underscore; not starting with a digit)."
        )
    if spec.type not in _AGENT_TYPES:
        return f"Unknown agent type: {spec.type!r}. Known: {', '.join(sorted(_AGENT_TYPES))}."
    if spec.type == "loop" and spec.max_iterations <= 0:
        return f"max_iterations must be > 0 (received {spec.max_iterations})."
    if spec.type == "remote_a2a" and not spec.agent_card.strip():
        return (
            "remote_a2a: 'agent_card' is required (URL or JSON path of the remote agent-card, "
            "e.g. 'http://host:8001/.well-known/agent-card.json')."
        )
    for sub in spec.sub_agents:
        if not is_identifier(sub):
            return f"Invalid sub_agent: {sub!r}. Expected a Python identifier."
    return None


# --------------------------------------------------------------------------- #
# Immutable mutations
# --------------------------------------------------------------------------- #
def add_or_update_agent(model: ProjectModel, spec: AgentSpec) -> ProjectModel:
    """Add ``spec`` or replace the existing agent of the same name. **Returns a new model.**

    Order is preserved: a replacement stays in its position; an addition is appended.
    """
    found = False
    new_agents: list[AgentSpec] = []
    for a in model.agents:
        if a.name == spec.name:
            new_agents.append(spec)
            found = True
        else:
            new_agents.append(a)
    if not found:
        new_agents.append(spec)
    return replace(model, agents=tuple(new_agents))


def set_root(model: ProjectModel, name: str) -> ProjectModel:
    """Return a new model whose root is an **agent** ``name`` (existence not validated here)."""
    return replace(model, root=name, root_kind="agent")


def set_workflow_root(model: ProjectModel, name: str) -> ProjectModel:
    """Return a new model whose root is a **workflow** ``name`` (existence not validated here)."""
    return replace(model, root=name, root_kind="workflow")


# --------------------------------------------------------------------------- #
# Immutable mutations — workflows (`workflow` domain)
# --------------------------------------------------------------------------- #
def add_or_update_workflow(model: ProjectModel, workflow: WorkflowSpec) -> ProjectModel:
    """Add ``workflow`` or replace the existing one of the same name. **Returns a new model.**

    Order is preserved (replacement keeps its position; addition is appended).
    """
    found = False
    new_workflows: list[WorkflowSpec] = []
    for w in model.workflows:
        if w.name == workflow.name:
            new_workflows.append(workflow)
            found = True
        else:
            new_workflows.append(w)
    if not found:
        new_workflows.append(workflow)
    return replace(model, workflows=tuple(new_workflows))


def add_or_replace_node(workflow: WorkflowSpec, node: WorkflowNodeSpec) -> WorkflowSpec:
    """Attach ``node`` to ``workflow`` (replace by name, else append). **Returns a new spec.**"""
    found = False
    new_nodes: list[WorkflowNodeSpec] = []
    for existing in workflow.nodes:
        if existing.name == node.name:
            new_nodes.append(node)
            found = True
        else:
            new_nodes.append(existing)
    if not found:
        new_nodes.append(node)
    return replace(workflow, nodes=tuple(new_nodes))


def add_or_replace_edge(workflow: WorkflowSpec, edge: WorkflowEdgeSpec) -> WorkflowSpec:
    """Attach ``edge`` to ``workflow`` (replace the same ``(source, target)``, else append).

    Identity is the ordered pair ``(source, target)`` — re-adding the same pair updates its
    ``route`` (position preserved). **Returns a new spec.**
    """
    found = False
    new_edges: list[WorkflowEdgeSpec] = []
    for existing in workflow.edges:
        if existing.source == edge.source and existing.target == edge.target:
            new_edges.append(edge)
            found = True
        else:
            new_edges.append(existing)
    if not found:
        new_edges.append(edge)
    return replace(workflow, edges=tuple(new_edges))


def add_or_replace_tool(spec: AgentSpec, tool: ToolSpec) -> AgentSpec:
    """Attach ``tool`` to ``spec`` following "**append unique, replace by name**".

    If a tool with the same :meth:`ToolSpec.ref_key` already exists, it is **replaced in place**
    (position preserved); otherwise ``tool`` is **appended** to the list. **Returns a new
    ``AgentSpec``** (immutable). Legacy (string) entries are normalized to ``ToolSpec``.
    """
    key = tool.ref_key()
    found = False
    new_tools: list[ToolSpec] = []
    for existing in spec.tool_specs():
        if existing.ref_key() == key:
            new_tools.append(tool)
            found = True
        else:
            new_tools.append(existing)
    if not found:
        new_tools.append(tool)
    return replace(spec, tools=tuple(new_tools))


# --------------------------------------------------------------------------- #
# Sidecar I/O
# --------------------------------------------------------------------------- #
def load_model(ws: Workspace, app_name: str) -> ProjectModel:
    """Load the ``.adk_toolkit/agents.json`` sidecar; return an empty model if absent.

    ``ws`` must point at the **app folder** (``<path>/<app_name>``).
    """
    if not ws.exists(SIDECAR_PATH):
        return ProjectModel(app_name=app_name)
    raw = ws.read(SIDECAR_PATH)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # corrupt sidecar -> clear error
        raise ValueError(f"Invalid sidecar JSON ({SIDECAR_PATH}): {exc}") from exc
    model = ProjectModel.from_dict(data)
    # We force the provided app_name (source of truth = folder).
    return replace(model, app_name=app_name)


def save_model(ws: Workspace, model: ProjectModel) -> bool:
    """Write the sidecar (indented, deterministic JSON). Returns True if modified."""
    payload = json.dumps(model.to_dict(), indent=2, sort_keys=False) + "\n"
    return ws.write(SIDECAR_PATH, payload)


# --------------------------------------------------------------------------- #
# Tool validation
# --------------------------------------------------------------------------- #
def validate_tool_spec(tool: ToolSpec, model: ProjectModel, owner: str) -> str | None:
    """Return an error message if ``tool`` is invalid, otherwise None.

    ``model``/``owner`` are used to validate ``agent_tool`` (existing target and != owner).
    """
    if tool.kind not in _TOOL_KINDS:
        return f"Unknown tool kind: {tool.kind!r}. Known: {', '.join(sorted(_TOOL_KINDS))}."

    # Auth: only openapi/apihub/mcp_toolset accept auth_scheme/auth_credential (confirmed).
    if tool.auth is not None:
        auth_error = _validate_auth(tool)
        if auth_error is not None:
            return auth_error

    if tool.kind in ("function", "long_running"):
        if not is_identifier(tool.name):
            return f"Invalid function name: {tool.name!r}. Expected a Python identifier."
        for pname, ptype, _default in tool.params:
            if not is_identifier(pname):
                return f"Invalid parameter name: {pname!r}. Expected a Python identifier."
            if not _is_allowed_type(ptype):
                return (
                    f"Unsupported parameter type: {ptype!r} (param {pname!r}). "
                    f"Allowed types: {', '.join(sorted(_ALLOWED_PARAM_TYPES))} "
                    "(or ``X | None`` / ``list[X]`` of those)."
                )
        if not _is_allowed_type(tool.returns):
            return f"Unsupported return type: {tool.returns!r}."
        return None

    if tool.kind == "builtin":
        if tool.builtin_kind not in BUILTIN_TOOLS:
            return (
                f"Unknown builtin: {tool.builtin_kind!r}. "
                f"Known: {', '.join(sorted(BUILTIN_TOOLS))}."
            )
        if tool.builtin_kind in ARG_BUILTINS:
            keys = {k for k, _ in tool.args}
            if not ({"data_store_id", "search_engine_id"} & keys):
                return (
                    f"{tool.builtin_kind!r} requires a 'data_store_id' argument "
                    "(or 'search_engine_id')."
                )
        return None

    if tool.kind == "agent_tool":
        if not is_identifier(tool.target_agent):
            return f"Invalid target_agent: {tool.target_agent!r}. Expected a Python identifier."
        if tool.target_agent == owner:
            return f"An agent cannot wrap itself as an AgentTool: {owner!r}."
        if model.get(tool.target_agent) is None:
            return f"Target agent not found: {tool.target_agent!r}. Create it first."
        return None

    if tool.kind == "openapi":
        if not is_identifier(tool.name):
            return f"Invalid OpenAPI toolset name: {tool.name!r} (Python identifier expected)."
        if not tool.spec.strip():
            return "The OpenAPI spec is empty."
        return None

    if tool.kind in ("bigquery", "spanner"):
        if not is_identifier(tool.name):
            return f"Invalid {tool.kind} toolset name: {tool.name!r} (Python identifier expected)."
        return None

    if tool.kind == "mcp_toolset":
        return _validate_mcp(tool)

    if tool.kind == "apihub":
        if not is_identifier(tool.name):
            return f"Invalid APIHub toolset name: {tool.name!r} (Python identifier expected)."
        if not tool.apihub_resource_name.strip():
            return "apihub_resource_name is empty (e.g. 'projects/<p>/locations/<l>/apis/<a>')."
        return None

    if tool.kind in ("langchain", "crewai"):
        if not tool.import_line.strip():
            return f"{tool.kind}: import_line is empty (e.g. 'from x.tools import MyTool')."
        if not tool.tool_expr.strip():
            return f"{tool.kind}: tool_expr is empty (e.g. 'MyTool(arg=...)')."
        if tool.kind == "crewai" and not tool.name.strip():
            return "crewai: 'name' is required (CrewaiTool requires a name, keyword-only)."
        return None

    if tool.kind == "skill_toolset":
        if not is_identifier(tool.name):
            return f"Invalid SkillToolset name: {tool.name!r} (Python identifier expected)."
        if not tool.skill_names:
            return (
                "skill_toolset: at least one skill name is required (create it first via create)."
            )
        for sname in tool.skill_names:
            if not is_skill_name(sname):
                return (
                    f"Invalid skill name: {sname!r}. A skill name must be lowercase kebab-case "
                    "(a-z, 0-9, hyphens), <= 64 chars, and match its directory name."
                )
        return None

    return None  # pragma: no cover


def validate_callback_spec(callback: CallbackSpec) -> str | None:
    """Return an error message if the guardrail (callback) is invalid, otherwise None.

    Checks: known hook, known policy, policy compatible with the hook, and the fields required by
    the policy (``keywords`` / ``max_chars`` (integer > 0) / ``denylist``).
    """
    if callback.hook not in _CALLBACK_HOOKS:
        return (
            f"Unknown callback hook: {callback.hook!r}. "
            f"Known: {', '.join(sorted(_CALLBACK_HOOKS))}."
        )
    if callback.policy not in _POLICY_KINDS:
        return (
            f"Unknown guardrail policy: {callback.policy!r}. "
            f"Known: {', '.join(sorted(_POLICY_KINDS))}."
        )
    allowed = _POLICY_HOOKS[callback.policy]
    if callback.hook not in allowed:
        return (
            f"The {callback.policy!r} policy is not compatible with the {callback.hook!r} hook. "
            f"Valid hooks: {', '.join(sorted(allowed))}."
        )
    if callback.policy == "block_keywords" and not _has_csv(callback.param("keywords")):
        return "block_keywords: 'keywords' is required (comma-separated list)."
    if callback.policy == "block_tool" and not _has_csv(callback.param("denylist")):
        return "block_tool: 'denylist' is required (comma-separated tool names)."
    if callback.policy == "max_input_chars":
        raw = callback.param("max_chars")
        if not _is_positive_int(raw):
            return f"max_input_chars: 'max_chars' must be an integer > 0 (received {raw!r})."
    return None


# --------------------------------------------------------------------------- #
# Workflow validation (`workflow` domain) — mirrors the ADK graph rules so the
# tools fail fast with a clear message instead of letting Workflow(...) raise.
# --------------------------------------------------------------------------- #
def validate_workflow_node_spec(node: WorkflowNodeSpec, model: ProjectModel) -> str | None:
    """Return an error message if the workflow node is invalid, otherwise None.

    Checks: identifier name, known kind, and per-kind requirements:
    - ``agent``: the referenced agent must exist in the model and must NOT be a workflow agent
      type that is deprecated as a graph node? (No — any model agent is a ``BaseNode``; we only
      require existence.) The node ``name`` must equal the agent it wraps (so the rendered edge
      references the agent variable directly).
    - ``function``: identifier params + allowed types (reuses the function-tool type rules).
    - ``join``: nothing extra.
    """
    if node.name == WORKFLOW_START:
        return f"Reserved node name: {WORKFLOW_START!r} is the graph entry sentinel."
    if not is_identifier(node.name):
        return f"Invalid node name: {node.name!r}. Expected a Python identifier."
    if node.kind not in _WORKFLOW_NODE_KINDS:
        return (
            f"Unknown node kind: {node.kind!r}. Known: {', '.join(sorted(_WORKFLOW_NODE_KINDS))}."
        )
    if node.kind == "agent":
        ref = node.agent_ref()
        if not is_identifier(ref):
            return f"Invalid agent reference: {ref!r}. Expected a Python identifier."
        if model.get(ref) is None:
            return f"Agent not found: {ref!r}. Create it first (agents_create_*)."
        if ref != node.name:
            return (
                f"An agent node's name must equal the wrapped agent ({ref!r}), "
                f"got node name {node.name!r}."
            )
        return None
    if node.kind == "function":
        for pname, ptype, _default in node.params:
            if not is_identifier(pname):
                return f"Invalid parameter name: {pname!r}. Expected a Python identifier."
            if not _is_allowed_type(ptype):
                return (
                    f"Unsupported parameter type: {ptype!r} (param {pname!r}). "
                    f"Allowed: {', '.join(sorted(_ALLOWED_PARAM_TYPES))} "
                    "(or ``X | None`` / ``list[X]`` of those)."
                )
        if not _is_allowed_type(node.returns):
            return f"Unsupported return type: {node.returns!r}."
        return None
    # ``join``: no extra checks.
    return None


def validate_workflow_edge_spec(edge: WorkflowEdgeSpec, workflow: WorkflowSpec) -> str | None:
    """Return an error message if the edge is invalid, otherwise None.

    ``source`` may be :data:`WORKFLOW_START`; otherwise both endpoints must be existing nodes.
    ``target`` must NOT be :data:`WORKFLOW_START` (START takes no incoming edge). A route, when
    provided, must be a non-empty string.
    """
    if edge.target == WORKFLOW_START:
        return f"{WORKFLOW_START!r} cannot be an edge target (it takes no incoming edge)."
    if edge.source != WORKFLOW_START and workflow.get_node(edge.source) is None:
        return f"Edge source not found: {edge.source!r}. Add it as a node first."
    if workflow.get_node(edge.target) is None:
        return f"Edge target not found: {edge.target!r}. Add it as a node first."
    if edge.source == edge.target:
        return f"A self-loop edge is not allowed: {edge.source!r} -> {edge.target!r}."
    if edge.route is not None and not edge.route.strip():
        return (
            "route must be a non-empty string when provided (or omit it for an unconditional edge)."
        )
    return None


def validate_workflow_graph(workflow: WorkflowSpec) -> str | None:
    """Return an error message if the full graph is invalid, otherwise None.

    Mirrors ``google.adk.workflow`` graph validation (cf. ``docs/adk-api-notes/workflow.md``):

    - at least one entry edge from :data:`WORKFLOW_START`;
    - every node reachable from ``START``;
    - no **unconditional cycle** (a cycle of ``route=None`` edges loops forever; cycles must
      include at least one routed edge — that is how ReAct-style loops are expressed);
    - at most ONE terminal node (a node with no outgoing edges) — ADK forbids multiple terminal
      outputs. Fan-in to a single ``join`` node keeps a single terminal.
    """
    node_names = set(workflow.node_names())
    start_edges = [e for e in workflow.edges if e.source == WORKFLOW_START]
    if not start_edges:
        return f"The graph has no entry edge from {WORKFLOW_START!r}. Add one (set_entry)."

    # Adjacency (excluding START as a node; it is the virtual entry).
    adjacency: dict[str, list[WorkflowEdgeSpec]] = {n: [] for n in node_names}
    for e in workflow.edges:
        if e.source != WORKFLOW_START:
            adjacency.setdefault(e.source, []).append(e)

    # Reachability from START.
    reachable: set[str] = set()
    stack = [e.target for e in start_edges]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stack.extend(e.target for e in adjacency.get(name, []))
    unreachable = node_names - reachable
    if unreachable:
        return f"Unreachable from {WORKFLOW_START!r}: {', '.join(sorted(unreachable))}."

    cycle_err = detect_unconditional_cycle(workflow)
    if cycle_err is not None:
        return cycle_err

    # Terminal nodes = nodes with no outgoing edge.
    has_outgoing = {e.source for e in workflow.edges if e.source != WORKFLOW_START}
    terminals = sorted(node_names - has_outgoing)
    if len(terminals) > 1:
        return (
            f"A workflow must have at most one terminal node (no outgoing edge); "
            f"found {len(terminals)}: {', '.join(terminals)}. "
            "Wire them into a single 'join' node or chain them."
        )
    return None


def detect_unconditional_cycle(workflow: WorkflowSpec) -> str | None:
    """Return an error if a cycle of ``route=None`` edges exists in ``workflow`` (DFS), else None.

    Independent of the other graph rules (reachability / terminals), so it can be enforced
    **eagerly** during incremental construction: adding an unrouted edge that closes a cycle is a
    hard structural error (an unconditional cycle loops forever) regardless of whether the rest of
    the graph is complete. A cycle is allowed only if at least one edge in it carries a route.
    """
    node_names = set(workflow.node_names())
    unconditional: dict[str, list[str]] = {n: [] for n in node_names}
    for e in workflow.edges:
        if e.source != WORKFLOW_START and e.route is None:
            unconditional.setdefault(e.source, []).append(e.target)

    in_stack: set[str] = set()
    done: set[str] = set()

    def visit(name: str, path: tuple[str, ...]) -> str | None:
        in_stack.add(name)
        for nxt in unconditional.get(name, []):
            if nxt in in_stack:
                cycle = " -> ".join((*path, name, nxt))
                return (
                    f"Unconditional cycle detected: {cycle}. A cycle must include at least one "
                    "conditional (routed) edge to avoid an infinite loop."
                )
            if nxt not in done:
                err_msg = visit(nxt, (*path, name))
                if err_msg is not None:
                    return err_msg
        in_stack.discard(name)
        done.add(name)
        return None

    for n in node_names:
        if n not in done:
            err_msg = visit(n, ())
            if err_msg is not None:
                return err_msg
    return None


def _has_csv(raw: str) -> bool:
    """True if ``raw`` contains at least one non-empty element (CSV list)."""
    return any(s.strip() for s in raw.split(","))


def _is_positive_int(raw: str) -> bool:
    """True if ``raw`` is a strictly positive integer."""
    try:
        return int(raw) > 0
    except (ValueError, TypeError):
        return False


def add_or_replace_callback(spec: AgentSpec, callback: CallbackSpec) -> AgentSpec:
    """Attach ``callback`` to ``spec`` (one callback per hook: replace, otherwise add).

    Returns a new immutable ``AgentSpec``. A hook can only carry ONE generated function (the
    toolkit attaches a single function per kwarg): a second callback on the same hook replaces
    the first (position preserved).
    """
    found = False
    new_callbacks: list[CallbackSpec] = []
    for existing in spec.callbacks:
        if existing.hook == callback.hook:
            new_callbacks.append(callback)
            found = True
        else:
            new_callbacks.append(existing)
    if not found:
        new_callbacks.append(callback)
    return replace(spec, callbacks=tuple(new_callbacks))


def _validate_mcp(tool: ToolSpec) -> str | None:
    """Validate an ``mcp_toolset``: identifier name, known transport, and required fields."""
    if not is_identifier(tool.name):
        return f"Invalid MCP toolset name: {tool.name!r} (Python identifier expected)."
    if tool.transport not in _MCP_TRANSPORTS:
        return (
            f"Unknown MCP transport: {tool.transport!r}. "
            f"Known: {', '.join(sorted(_MCP_TRANSPORTS))}."
        )
    if tool.transport == "stdio":
        if not tool.command.strip():
            return "Transport 'stdio': 'command' is required (e.g. 'npx')."
    elif not tool.url.strip():
        return f"Transport {tool.transport!r}: 'url' is required."
    return None


def _validate_auth(tool: ToolSpec) -> str | None:
    """Validate an ``auth`` sub-spec: auth-capable kind, known scheme, scheme's required fields."""
    if tool.kind not in _AUTH_CAPABLE_KINDS:
        return (
            f"The {tool.kind!r} kind does not accept auth (auth_scheme/auth_credential). "
            f"Compatible kinds: {', '.join(sorted(_AUTH_CAPABLE_KINDS))} "
            "(bigquery/spanner use a credentials_config instead)."
        )
    auth = tool.auth
    assert auth is not None  # guaranteed by the caller
    if auth.scheme not in _AUTH_SCHEMES:
        return f"Unknown auth scheme: {auth.scheme!r}. Known: {', '.join(sorted(_AUTH_SCHEMES))}."
    keys = {k for k, _ in auth.credential}
    required: dict[str, set[str]] = {
        "apikey": {"api_key"},
        "bearer": {"token"},
        "oauth2": {"client_id"},
        "service_account": set(),  # use_default_credential OR scopes: at least one key
    }
    missing = required[auth.scheme] - keys
    if missing:
        fields = ", ".join(sorted(missing))
        return f"Auth {auth.scheme!r}: missing credential field(s): {fields}."
    if auth.scheme == "service_account" and not keys:
        return "Auth 'service_account': provide 'use_default_credential' or 'scopes'."
    return None


def _is_allowed_type(t: str) -> bool:
    """Allowed param/return type: a base type, or a simple composition
    (``X | None``, ``list[X]``, ``dict[X, Y]``, ``Optional[X]``) of base types."""
    t = t.strip()
    if t in _ALLOWED_PARAM_TYPES:
        return True
    # Union with None: ``X | None`` or ``None | X``.
    if "|" in t:
        return all(_is_allowed_type(part) for part in t.split("|"))
    # Simple generics: list[...], dict[...], tuple[...], set[...], Optional[...].
    m = re.fullmatch(r"(list|dict|tuple|set|Optional)\[(.+)\]", t)
    if m is not None:
        inner = m.group(2)
        return all(_is_allowed_type(part) for part in inner.split(","))
    return False
