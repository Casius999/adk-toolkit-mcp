"""I/O du sidecar, mutations immuables et validation des specs.

Ce module porte la **logique non-rendu** du modèle de projet :

- validation : :func:`validate_spec` (agent) et :func:`validate_tool_spec` (outil), plus les
  validateurs internes ``_validate_mcp`` / ``_validate_auth`` / ``_is_allowed_type`` ;
- mutations **immuables** : :func:`add_or_update_agent`, :func:`set_root`,
  :func:`add_or_replace_tool` (renvoient toujours un nouvel objet) ;
- I/O sidecar : :func:`load_model` / :func:`save_model` (lecture/écriture de
  ``.adk_toolkit/agents.json`` via un :class:`~adk_toolkit_mcp.workspace.Workspace`).

Importe les dataclasses/constantes depuis :mod:`adk_toolkit_mcp.project_model.specs`. La
génération de ``agent.py`` vit séparément dans :mod:`adk_toolkit_mcp.project_model.render`.
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
    _MCP_TRANSPORTS,
    _TOOL_KINDS,
    ARG_BUILTINS,
    BUILTIN_TOOLS,
    SIDECAR_PATH,
    AgentSpec,
    ProjectModel,
    ToolSpec,
    is_identifier,
)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_spec(spec: AgentSpec) -> str | None:
    """Renvoie un message d'erreur si la spec est invalide, sinon None."""
    if not is_identifier(spec.name):
        return (
            f"Nom d'agent invalide : {spec.name!r}. Attendu un identifiant Python "
            "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
        )
    if spec.type not in _AGENT_TYPES:
        return f"Type d'agent inconnu : {spec.type!r}. Connus : {', '.join(sorted(_AGENT_TYPES))}."
    if spec.type == "loop" and spec.max_iterations <= 0:
        return f"max_iterations doit être > 0 (reçu {spec.max_iterations})."
    if spec.type == "remote_a2a" and not spec.agent_card.strip():
        return (
            "remote_a2a : 'agent_card' est requis (URL ou chemin JSON de l'agent-card distant, "
            "ex. 'http://host:8001/.well-known/agent-card.json')."
        )
    for sub in spec.sub_agents:
        if not is_identifier(sub):
            return f"sub_agent invalide : {sub!r}. Attendu un identifiant Python."
    return None


# --------------------------------------------------------------------------- #
# Mutations immuables
# --------------------------------------------------------------------------- #
def add_or_update_agent(model: ProjectModel, spec: AgentSpec) -> ProjectModel:
    """Ajoute ``spec`` ou remplace l'agent existant de même nom. **Renvoie un nouveau modèle.**

    L'ordre est préservé : un remplacement reste à sa position ; un ajout est appended.
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
    """Renvoie un nouveau modèle dont la racine est ``name`` (sans valider l'existence ici)."""
    return replace(model, root=name)


def add_or_replace_tool(spec: AgentSpec, tool: ToolSpec) -> AgentSpec:
    """Attache ``tool`` à ``spec`` selon « **append unique, replace by name** ».

    Si un outil de même :meth:`ToolSpec.ref_key` existe déjà, il est **remplacé en place**
    (position préservée) ; sinon ``tool`` est **ajouté** en fin de liste. **Renvoie un nouvel
    ``AgentSpec``** (immuable). Les entrées héritées (chaîne) sont normalisées en ``ToolSpec``.
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
    """Charge le sidecar ``.adk_toolkit/agents.json`` ; renvoie un modèle vide si absent.

    ``ws`` doit pointer sur le **dossier de l'app** (``<path>/<app_name>``).
    """
    if not ws.exists(SIDECAR_PATH):
        return ProjectModel(app_name=app_name)
    raw = ws.read(SIDECAR_PATH)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # sidecar corrompu -> erreur claire
        raise ValueError(f"Sidecar JSON invalide ({SIDECAR_PATH}) : {exc}") from exc
    model = ProjectModel.from_dict(data)
    # On force app_name fourni (source de vérité = dossier).
    return replace(model, app_name=app_name)


def save_model(ws: Workspace, model: ProjectModel) -> bool:
    """Écrit le sidecar (JSON indenté, déterministe). Renvoie True si modifié."""
    payload = json.dumps(model.to_dict(), indent=2, sort_keys=False) + "\n"
    return ws.write(SIDECAR_PATH, payload)


# --------------------------------------------------------------------------- #
# Validation d'outils
# --------------------------------------------------------------------------- #
def validate_tool_spec(tool: ToolSpec, model: ProjectModel, owner: str) -> str | None:
    """Renvoie un message d'erreur si ``tool`` est invalide, sinon None.

    ``model``/``owner`` servent à valider ``agent_tool`` (cible existante et != propriétaire).
    """
    if tool.kind not in _TOOL_KINDS:
        return f"Genre d'outil inconnu : {tool.kind!r}. Connus : {', '.join(sorted(_TOOL_KINDS))}."

    # Auth : seuls openapi/apihub/mcp_toolset acceptent auth_scheme/auth_credential (confirmé).
    if tool.auth is not None:
        auth_error = _validate_auth(tool)
        if auth_error is not None:
            return auth_error

    if tool.kind in ("function", "long_running"):
        if not is_identifier(tool.name):
            return f"Nom de fonction invalide : {tool.name!r}. Attendu un identifiant Python."
        for pname, ptype, _default in tool.params:
            if not is_identifier(pname):
                return f"Nom de paramètre invalide : {pname!r}. Attendu un identifiant Python."
            if not _is_allowed_type(ptype):
                return (
                    f"Type de paramètre non supporté : {ptype!r} (param {pname!r}). "
                    f"Types autorisés : {', '.join(sorted(_ALLOWED_PARAM_TYPES))} "
                    "(ou ``X | None`` / ``list[X]`` de ceux-ci)."
                )
        if not _is_allowed_type(tool.returns):
            return f"Type de retour non supporté : {tool.returns!r}."
        return None

    if tool.kind == "builtin":
        if tool.builtin_kind not in BUILTIN_TOOLS:
            return (
                f"Builtin inconnu : {tool.builtin_kind!r}. "
                f"Connus : {', '.join(sorted(BUILTIN_TOOLS))}."
            )
        if tool.builtin_kind in ARG_BUILTINS:
            keys = {k for k, _ in tool.args}
            if not ({"data_store_id", "search_engine_id"} & keys):
                return (
                    f"{tool.builtin_kind!r} requiert un argument 'data_store_id' "
                    "(ou 'search_engine_id')."
                )
        return None

    if tool.kind == "agent_tool":
        if not is_identifier(tool.target_agent):
            return f"target_agent invalide : {tool.target_agent!r}. Attendu un identifiant Python."
        if tool.target_agent == owner:
            return f"Un agent ne peut pas s'envelopper lui-même comme AgentTool : {owner!r}."
        if model.get(tool.target_agent) is None:
            return f"Agent cible introuvable : {tool.target_agent!r}. Créez-le d'abord."
        return None

    if tool.kind == "openapi":
        if not is_identifier(tool.name):
            return f"Nom de toolset OpenAPI invalide : {tool.name!r} (identifiant Python attendu)."
        if not tool.spec.strip():
            return "La spec OpenAPI est vide."
        return None

    if tool.kind in ("bigquery", "spanner"):
        if not is_identifier(tool.name):
            return (
                f"Nom de toolset {tool.kind} invalide : {tool.name!r} (identifiant Python attendu)."
            )
        return None

    if tool.kind == "mcp_toolset":
        return _validate_mcp(tool)

    if tool.kind == "apihub":
        if not is_identifier(tool.name):
            return f"Nom de toolset APIHub invalide : {tool.name!r} (identifiant Python attendu)."
        if not tool.apihub_resource_name.strip():
            return "apihub_resource_name est vide (ex. 'projects/<p>/locations/<l>/apis/<a>')."
        return None

    if tool.kind in ("langchain", "crewai"):
        if not tool.import_line.strip():
            return f"{tool.kind} : import_line est vide (ex. 'from x.tools import MyTool')."
        if not tool.tool_expr.strip():
            return f"{tool.kind} : tool_expr est vide (ex. 'MyTool(arg=...)')."
        if tool.kind == "crewai" and not tool.name.strip():
            return "crewai : 'name' est requis (CrewaiTool exige un nom, keyword-only)."
        return None

    return None  # pragma: no cover


def _validate_mcp(tool: ToolSpec) -> str | None:
    """Valide un ``mcp_toolset`` : nom identifiant, transport connu, et champs requis."""
    if not is_identifier(tool.name):
        return f"Nom de toolset MCP invalide : {tool.name!r} (identifiant Python attendu)."
    if tool.transport not in _MCP_TRANSPORTS:
        return (
            f"Transport MCP inconnu : {tool.transport!r}. "
            f"Connus : {', '.join(sorted(_MCP_TRANSPORTS))}."
        )
    if tool.transport == "stdio":
        if not tool.command.strip():
            return "Transport 'stdio' : 'command' est requis (ex. 'npx')."
    elif not tool.url.strip():
        return f"Transport {tool.transport!r} : 'url' est requis."
    return None


def _validate_auth(tool: ToolSpec) -> str | None:
    """Valide une sous-spec ``auth`` : genre auth-capable, schéma connu, champs requis du schéma."""
    if tool.kind not in _AUTH_CAPABLE_KINDS:
        return (
            f"Le genre {tool.kind!r} n'accepte pas d'auth (auth_scheme/auth_credential). "
            f"Genres compatibles : {', '.join(sorted(_AUTH_CAPABLE_KINDS))} "
            "(bigquery/spanner utilisent plutôt un credentials_config)."
        )
    auth = tool.auth
    assert auth is not None  # garanti par l'appelant
    if auth.scheme not in _AUTH_SCHEMES:
        return (
            f"Schéma d'auth inconnu : {auth.scheme!r}. Connus : {', '.join(sorted(_AUTH_SCHEMES))}."
        )
    keys = {k for k, _ in auth.credential}
    required: dict[str, set[str]] = {
        "apikey": {"api_key"},
        "bearer": {"token"},
        "oauth2": {"client_id"},
        "service_account": set(),  # use_default_credential OU scopes : au moins une clé
    }
    missing = required[auth.scheme] - keys
    if missing:
        fields = ", ".join(sorted(missing))
        return f"Auth {auth.scheme!r} : champ(s) credential manquant(s) : {fields}."
    if auth.scheme == "service_account" and not keys:
        return "Auth 'service_account' : fournir 'use_default_credential' ou 'scopes'."
    return None


def _is_allowed_type(t: str) -> bool:
    """Type de param/retour autorisé : un type de base, ou une composition simple
    (``X | None``, ``list[X]``, ``dict[X, Y]``, ``Optional[X]``) de types de base."""
    t = t.strip()
    if t in _ALLOWED_PARAM_TYPES:
        return True
    # Union avec None : ``X | None`` ou ``None | X``.
    if "|" in t:
        return all(_is_allowed_type(part) for part in t.split("|"))
    # Génériques simples : list[...], dict[...], tuple[...], set[...], Optional[...].
    m = re.fullmatch(r"(list|dict|tuple|set|Optional)\[(.+)\]", t)
    if m is not None:
        inner = m.group(2)
        return all(_is_allowed_type(part) for part in inner.split(","))
    return False
