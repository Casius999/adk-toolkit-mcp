"""Code-first ADK project model: JSON sidecar + full regeneration of ``agent.py``.

The toolkit describes the multi-agent composition in a **sidecar file**
``<app_dir>/.adk_toolkit/agents.json`` (where ``<app_dir> = <path>/<app_name>``), then
**fully regenerates** ``agent.py`` from that model. Regenerating rather than patching Python is
more robust (no AST parsing/round-trip, deterministic output).

This package is **pure and unit-testable** (no dependency on google-adk: we only produce a
*source string* that will import ADK at its own runtime). It is split into sub-modules by
responsibility, while keeping the historical public surface imported via
``from adk_toolkit_mcp.project_model import X`` **unchanged**:

- :mod:`.specs` — immutable dataclasses (`ProjectModel`, `AgentSpec`, `ToolSpec`, …), domain
  constants and ``Literal`` aliases;
- :mod:`.sidecar` — sidecar I/O (`load_model`/`save_model`), immutable mutations
  (`add_or_update_agent`/`set_root`/`add_or_replace_tool`) and spec validation;
- :mod:`.render` (+ internal :mod:`._codegen`) — generation of ``agent.py``
  (`render_agent_module`/`regenerate`/`render_tool_ref`/`topological_order` + ruff-stable
  machinery).

See ``docs/adk-api-notes/agents.md`` for the real ADK signatures confirmed by introspection
(and the note on the deprecation of the workflow agents in google-adk 2.1.0).
"""

from __future__ import annotations

from .render import (
    regenerate,
    render_agent_module,
    render_tool_ref,
    topological_order,
)
from .sidecar import (
    add_or_replace_callback,
    add_or_replace_edge,
    add_or_replace_node,
    add_or_replace_tool,
    add_or_update_agent,
    add_or_update_workflow,
    detect_unconditional_cycle,
    load_model,
    save_model,
    set_root,
    set_workflow_root,
    validate_callback_spec,
    validate_spec,
    validate_tool_spec,
    validate_workflow_edge_spec,
    validate_workflow_graph,
    validate_workflow_node_spec,
)
from .specs import (
    ARG_BUILTINS,
    BUILTIN_TOOLS,
    CALLBACK_HOOKS,
    CORE_BUILTINS,
    HARM_BLOCK_THRESHOLDS,
    HARM_CATEGORIES,
    LINE_LENGTH,
    LITELLM_PROVIDERS,
    POLICY_KINDS,
    SIDECAR_DIR,
    SIDECAR_FILE,
    SIDECAR_PATH,
    WORKFLOW_START,
    AgentSpec,
    AgentType,
    AuthSpec,
    CallbackHook,
    CallbackSpec,
    GenerateContentConfigSpec,
    LiteLlmSpec,
    PolicyKind,
    ProjectModel,
    SafetySettingSpec,
    ToolKind,
    ToolRender,
    ToolSpec,
    WorkflowEdgeSpec,
    WorkflowNodeKind,
    WorkflowNodeSpec,
    WorkflowSpec,
    is_identifier,
)

#: Stable public surface. Every name historically importable from
#: ``adk_toolkit_mcp.project_model`` stays exported here (strict backward compatibility).
__all__ = [
    # Dataclasses / specs
    "AgentSpec",
    "AuthSpec",
    "CallbackSpec",
    "GenerateContentConfigSpec",
    "LiteLlmSpec",
    "ProjectModel",
    "SafetySettingSpec",
    "ToolRender",
    "ToolSpec",
    "WorkflowEdgeSpec",
    "WorkflowNodeSpec",
    "WorkflowSpec",
    # Literal aliases
    "AgentType",
    "CallbackHook",
    "PolicyKind",
    "ToolKind",
    "WorkflowNodeKind",
    # Constants
    "ARG_BUILTINS",
    "BUILTIN_TOOLS",
    "CALLBACK_HOOKS",
    "CORE_BUILTINS",
    "HARM_BLOCK_THRESHOLDS",
    "HARM_CATEGORIES",
    "LINE_LENGTH",
    "LITELLM_PROVIDERS",
    "POLICY_KINDS",
    "SIDECAR_DIR",
    "SIDECAR_FILE",
    "SIDECAR_PATH",
    "WORKFLOW_START",
    # Validation
    "is_identifier",
    "validate_callback_spec",
    "validate_spec",
    "validate_tool_spec",
    "validate_workflow_edge_spec",
    "validate_workflow_graph",
    "validate_workflow_node_spec",
    # Immutable mutations
    "add_or_replace_callback",
    "add_or_replace_edge",
    "add_or_replace_node",
    "add_or_replace_tool",
    "add_or_update_agent",
    "add_or_update_workflow",
    "detect_unconditional_cycle",
    "set_root",
    "set_workflow_root",
    # Sidecar I/O
    "load_model",
    "save_model",
    # Rendering
    "regenerate",
    "render_agent_module",
    "render_tool_ref",
    "topological_order",
]
