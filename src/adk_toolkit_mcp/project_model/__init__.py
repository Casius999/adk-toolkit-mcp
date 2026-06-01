"""Modèle de projet ADK code-first : sidecar JSON + régénération complète de ``agent.py``.

Le toolkit décrit la composition multi-agents dans un **fichier sidecar**
``<app_dir>/.adk_toolkit/agents.json`` (où ``<app_dir> = <path>/<app_name>``), puis
**régénère intégralement** ``agent.py`` à partir de ce modèle. Régénérer plutôt que
patcher du Python est plus robuste (pas de parsing/round-trip d'AST, sortie déterministe).

Ce package est **pur et testable unitairement** (aucune dépendance à google-adk : on ne fait
que produire une *chaîne source* qui importera l'ADK à son propre runtime). Il est découpé en
sous-modules par responsabilité, tout en conservant **inchangée** la surface publique
historique importée via ``from adk_toolkit_mcp.project_model import X`` :

- :mod:`.specs` — dataclasses immuables (`ProjectModel`, `AgentSpec`, `ToolSpec`, …),
  constantes de domaine et alias ``Literal`` ;
- :mod:`.sidecar` — I/O du sidecar (`load_model`/`save_model`), mutations immuables
  (`add_or_update_agent`/`set_root`/`add_or_replace_tool`) et validation des specs ;
- :mod:`.render` (+ :mod:`._codegen` interne) — génération de ``agent.py``
  (`render_agent_module`/`regenerate`/`render_tool_ref`/`topological_order` + machinerie
  ruff-stable).

Voir ``docs/adk-api-notes/agents.md`` pour les signatures ADK réelles confirmées par
introspection (et la note sur la dépréciation des agents workflow en google-adk 2.1.0).
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
    add_or_replace_tool,
    add_or_update_agent,
    load_model,
    save_model,
    set_root,
    validate_callback_spec,
    validate_spec,
    validate_tool_spec,
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
    is_identifier,
)

#: Surface publique stable. Tout nom historiquement importable depuis
#: ``adk_toolkit_mcp.project_model`` reste exporté ici (compat ascendante stricte).
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
    # Alias Literal
    "AgentType",
    "CallbackHook",
    "PolicyKind",
    "ToolKind",
    # Constantes
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
    # Validation
    "is_identifier",
    "validate_callback_spec",
    "validate_spec",
    "validate_tool_spec",
    # Mutations immuables
    "add_or_replace_callback",
    "add_or_replace_tool",
    "add_or_update_agent",
    "set_root",
    # I/O sidecar
    "load_model",
    "save_model",
    # Rendu
    "regenerate",
    "render_agent_module",
    "render_tool_ref",
    "topological_order",
]
