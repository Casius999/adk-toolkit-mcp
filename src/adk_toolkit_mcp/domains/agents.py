"""Domaine `agents` : composition multi-agents ADK (code-first, sidecar + régénération).

Sous-serveur FastMCP monté par le serveur racine sous le namespace ``agents`` (outils
exposés comme ``agents_<nom>`` côté client). Fonctions nommées avec des noms **BARE**
(``create_llm``, ``create_sequential``, …) — cf. ``docs/adk-api-notes/conventions.md``.

Chaque outil opère sur ``(path, app_name, …)`` : il charge le sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, applique une mutation **immuable**, le
réécrit, puis **régénère intégralement** ``agent.py`` (+ ``__init__.py``) via
:class:`~adk_toolkit_mcp.workspace.Workspace`. Tout est renvoyé dans l'enveloppe
``{ok, data, error}`` ; les entrées invalides renvoient ``err(...)`` (jamais d'exception).

Le rendu réel et la sémantique du modèle vivent dans
:mod:`adk_toolkit_mcp.project_model` (pur, testable). Voir ``docs/adk-api-notes/agents.md``
pour les signatures ADK confirmées (et la dépréciation des agents workflow en 2.1.0).
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

    Renvoie le payload commun ``{app_name, agents, root, sidecar, regenerated, changed}``.
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
            "agents": list(model.agent_names()),
            "root": model.root,
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


def _add_spec(path: str, app_name: str, spec: AgentSpec) -> dict[str, Any]:
    """Valide la spec, l'ajoute/met à jour dans le modèle, commit. Mutualise 1-5."""
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
            f"sub_agents introuvables : {', '.join(missing)}. "
            "Créez-les d'abord (l'ordre de création est libre, mais ils doivent exister)."
        )

    model = add_or_update_agent(model, spec)
    return _commit(path, app_name, model)


# --------------------------------------------------------------------------- #
# Outils MCP — création par type
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
    """Ajoute/met à jour un agent ``LlmAgent`` dans le modèle, puis régénère ``agent.py``."""
    if not model.strip():
        return err("model est vide.")
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
    """Ajoute/met à jour un ``SequentialAgent`` orchestrant ``sub_agents`` (qui doivent exister)."""
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
    """Ajoute/met à jour un ``ParallelAgent`` orchestrant ``sub_agents`` (qui doivent exister)."""
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
    """Ajoute/met à jour un ``LoopAgent`` (``max_iterations`` > 0 requis)."""
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
    """Ajoute/met à jour un agent custom : sous-classe ``BaseAgent`` (stub) + instance."""
    spec = AgentSpec(name=name, type="custom", description=description)
    return _add_spec(path, app_name, spec)


# --------------------------------------------------------------------------- #
# Outils MCP — composition / racine / lecture
# --------------------------------------------------------------------------- #
@agents_server.tool(tags={"agents"})
def compose(
    path: str,
    app_name: str,
    name: str,
    sub_agents: list[str],
) -> dict[str, Any]:
    """Remplace les ``sub_agents`` d'un agent **existant** (valide leur existence)."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    current = model.get(name)
    if current is None:
        return err(f"Agent introuvable : {name!r}. Créez-le avant de composer.")
    if current.type == "custom":
        return err("Un agent custom n'a pas de sub_agents gérés par le modèle.")

    missing = [s for s in sub_agents if model.get(s) is None and s != name]
    if missing:
        return err(f"sub_agents introuvables : {', '.join(missing)}.")
    if name in sub_agents:
        return err(f"Un agent ne peut pas se référencer lui-même : {name!r}.")

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
    """Désigne ``name`` comme ``root_agent`` du sidecar, puis régénère ``agent.py``."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    if model.get(name) is None:
        return err(f"Agent introuvable : {name!r}. Créez-le avant de le définir comme racine.")

    model = _model_set_root(model, name)
    return _commit(path, app_name, model)


@agents_server.tool(tags={"agents"})
def as_tool(path: str, app_name: str, agent_name: str) -> dict[str, Any]:
    """Renvoie le **snippet source** pour envelopper ``agent_name`` via ``AgentTool``.

    Helper de composition (P3 ``tools``) : ne mute aucun fichier. Le snippet montre
    l'import et l'usage ``LlmAgent(..., tools=[AgentTool(agent=<agent_name>)])``.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Nom d'agent invalide : {agent_name!r}.")

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model
    if model.get(agent_name) is None:
        return err(f"Agent introuvable : {agent_name!r}.")

    snippet = (
        "from google.adk.tools import AgentTool\n"
        f"{agent_name}_tool = AgentTool(agent={agent_name})\n"
        f"# Puis: LlmAgent(..., tools=[{agent_name}_tool])"
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
    """Liste les agents du sidecar (nom, type, racine). Lecture seule.

    Nommée ``list_agents`` en Python (pour ne pas masquer le builtin ``list`` dans ce
    module), mais **enregistrée sous le nom d'outil BARE ``list``** -> exposée
    ``agents_list`` côté client.
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
    """Renvoie la spec complète d'un agent du sidecar (telle que sérialisée). Lecture seule."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    model = _load(path, app_name)
    if isinstance(model, dict):
        return model

    spec = model.get(name)
    if spec is None:
        return err(f"Agent introuvable : {name!r}.")
    return ok(spec.to_dict())
