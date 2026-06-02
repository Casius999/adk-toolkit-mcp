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
    ARG_BUILTINS,
    BUILTIN_TOOLS,
    SIDECAR_PATH,
    AgentSpec,
    CallbackSpec,
    ProjectModel,
    ToolSpec,
    is_identifier,
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
    """Return a new model whose root is ``name`` (without validating existence here)."""
    return replace(model, root=name)


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
