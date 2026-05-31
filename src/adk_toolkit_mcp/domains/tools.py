"""Domaine `tools` : attache des outils ADK aux agents (code-first, sidecar + régénération).

Sous-serveur FastMCP monté par le serveur racine sous le namespace ``tools`` (outils exposés
comme ``tools_<nom>`` côté client). Fonctions nommées avec des noms **BARE** (``add_function``,
``add_long_running``, …) — cf. ``docs/adk-api-notes/conventions.md``.

Chaque outil opère sur ``(path, app_name, agent_name, …)`` : il charge le sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, **attache/remplace** une spec d'outil sur
l'agent ``agent_name`` (sémantique « append unique, replace by name » via
:meth:`~adk_toolkit_mcp.project_model.ToolSpec.ref_key`), réécrit le sidecar, puis **régénère
intégralement** ``agent.py`` (+ ``__init__.py``). Tout est renvoyé dans l'enveloppe
``{ok, data, error}`` ; les entrées invalides renvoient ``err(...)`` (jamais d'exception).

Passe **3a** : outils **sans dépendance** (aucune extra ``google-adk`` requise) :
``function``, ``long_running``, ``builtin`` (dont ``vertex_ai_search``), ``agent_tool``,
``openapi``. Le codegen réel et la sémantique vivent dans
:mod:`adk_toolkit_mcp.project_model` (pur, testable). Voir ``docs/adk-api-notes/tools.md`` pour
les signatures ADK confirmées (builtins = instances, OpenAPIToolset direct dans ``tools=[...]``,
fonction auto-wrappée en ``FunctionTool`` par ADK).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    BUILTIN_TOOLS,
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

#: app_name = identifiant de package Python (nom de dossier ET de module).
_APP_NAME_ERR = (
    "app_name invalide : attendu un identifiant Python "
    "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
)


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _load(path: str, app_name: str) -> ProjectModel | dict[str, Any]:
    """Charge le modèle ; renvoie un ``err(...)`` (dict) si le sidecar est corrompu."""
    ws = _app_ws(path, app_name)
    try:
        return load_model(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _commit(path: str, app_name: str, model: ProjectModel) -> dict[str, Any]:
    """Sauve le sidecar + régénère ``agent.py``. Convertit un cycle en ``err``.

    Renvoie le payload commun ``{app_name, agent, tools, sidecar, regenerated, changed}``.
    """
    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle détecté au rendu
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
    """Valide puis attache/remplace ``tool`` sur ``agent_name``, et commit. Mutualise 1-5.

    Étapes : valide ``app_name`` -> charge le modèle -> exige un agent ``llm`` existant
    (seul un ``LlmAgent`` porte des outils) -> valide la spec (avec le modèle, pour
    ``agent_tool``) -> attache (append unique / replace by name) -> commit (régénère).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"agent_name invalide : {agent_name!r}. Attendu un identifiant Python.")

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    agent = model.get(agent_name)
    if agent is None:
        return err(f"Agent introuvable : {agent_name!r}. Créez-le d'abord (domaine agents).")
    if agent.type != "llm":
        return err(
            f"L'agent {agent_name!r} est de type {agent.type!r} ; seuls les agents 'llm' "
            "(LlmAgent) portent des outils."
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
    """Normalise une liste ``[{"name","type","default"}]`` en tuple typé pour ``ToolSpec``.

    Renvoie un message d'erreur (str) si un item est mal formé. ``default`` est un **littéral
    source** (déjà rendu) ou ``None`` (paramètre sans défaut). Ex. ``{"name":"n","type":"int",
    "default":"0"}`` -> ``("n","int","0")``.
    """
    out: list[tuple[str, str, str | None]] = []
    for item in params:
        if not isinstance(item, dict) or "name" not in item:
            return f"Paramètre mal formé : {item!r}. Attendu {{'name','type','default'?}}."
        name = str(item["name"])
        ptype = str(item.get("type", "str"))
        default = item.get("default")
        out.append((name, ptype, None if default is None else str(default)))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Outils MCP — ajout d'outils par genre
# --------------------------------------------------------------------------- #
@tools_server.tool
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
    """Attache une **function-tool** à ``agent_name`` : génère ``def <func_name>(...)`` et
    place le nom bare dans ``tools=[...]`` (ADK l'auto-wrappe en ``FunctionTool``).

    ``params`` : liste de ``{"name":.., "type":"str", "default":null}`` (``default`` = littéral
    source ou ``null``). Identifiants et types sont validés. Sémantique « append unique /
    replace by name » : ré-attacher le même ``func_name`` remplace la définition.
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


@tools_server.tool
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
    """Comme :func:`add_function`, mais enveloppe la fonction dans
    ``LongRunningFunctionTool(func=<func_name>)`` (outil long-running ADK)."""
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


@tools_server.tool
def add_builtin(
    path: str,
    app_name: str,
    agent_name: str,
    kind: str,
    args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attache un **builtin** ADK (``google_search``, ``url_context``, …) à ``agent_name``.

    ``kind`` doit appartenir à l'ensemble confirmé :data:`BUILTIN_TOOLS`. Pour
    ``vertex_ai_search``, ``args`` doit fournir ``data_store_id`` (ou ``search_engine_id``)
    -> rendu ``VertexAiSearchTool(data_store_id="...")``. Les builtins core sont rendus par
    leur nom bare (instance d'outil déjà exportée par ADK).
    """
    if kind not in BUILTIN_TOOLS:
        return err(f"Builtin inconnu : {kind!r}. Connus : {', '.join(sorted(BUILTIN_TOOLS))}.")
    arg_pairs = tuple((str(k), str(v)) for k, v in (args or {}).items())
    tool = ToolSpec(kind="builtin", builtin_kind=kind, args=arg_pairs)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_agent_tool(
    path: str,
    app_name: str,
    agent_name: str,
    target_agent: str,
) -> dict[str, Any]:
    """Attache ``AgentTool(agent=<target_agent>)`` à ``agent_name`` (délégation agent-as-tool).

    ``target_agent`` doit être un agent **existant** du modèle et différent de ``agent_name``
    (pas d'auto-enveloppe). L'ordre de génération est topologique (cible définie avant
    l'enveloppant) ; la cible **n'est pas** ajoutée comme ``sub_agent`` (règle ADK du parent
    unique : un agent enveloppé en outil n'est pas un enfant).
    """
    tool = ToolSpec(kind="agent_tool", target_agent=target_agent)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_openapi(
    path: str,
    app_name: str,
    agent_name: str,
    spec: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Attache un ``OpenAPIToolset`` (construit depuis la chaîne ``spec``) à ``agent_name``.

    Génère ``<name> = OpenAPIToolset(spec_str=<spec>, spec_str_type="json")`` au niveau module
    et place ``<name>`` **directement** dans ``tools=[...]`` (confirmé : un toolset est accepté
    tel quel, pas besoin de ``.get_tools()``). ``name`` défaut = ``<agent_name>_openapi``.
    """
    toolset_name = name if name is not None else f"{agent_name}_openapi"
    if not is_identifier(toolset_name):
        return err(f"Nom de toolset invalide : {toolset_name!r}. Attendu un identifiant Python.")
    tool = ToolSpec(kind="openapi", name=toolset_name, spec=spec)
    return _attach(path, app_name, agent_name, tool)


# --------------------------------------------------------------------------- #
# Outil MCP — lecture
# --------------------------------------------------------------------------- #
@tools_server.tool(name="list")
def list_tools_for_agent(path: str, app_name: str, agent_name: str) -> dict[str, Any]:
    """Liste les outils attachés à ``agent_name`` (genre + détail synthétique). Lecture seule.

    Nommée ``list_tools_for_agent`` en Python (pour ne pas masquer le builtin ``list`` dans ce
    module), mais **enregistrée sous le nom d'outil BARE ``list``** -> exposée ``tools_list``.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"agent_name invalide : {agent_name!r}. Attendu un identifiant Python.")

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    agent = model.get(agent_name)
    if agent is None:
        return err(f"Agent introuvable : {agent_name!r}.")

    return ok(
        {
            "app_name": app_name,
            "agent": agent_name,
            "tools": [_tool_summary(t) for t in agent.tool_specs()],
        }
    )


def _tool_summary(tool: ToolSpec) -> dict[str, Any]:
    """Résumé synthétique d'un ``ToolSpec`` pour ``list`` (selon le genre)."""
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
    elif tool.kind == "openapi":
        summary["name"] = tool.name
    return summary
