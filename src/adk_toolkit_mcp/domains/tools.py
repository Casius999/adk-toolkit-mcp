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
``openapi``.

Passe **3b** : toolsets à **dépendance optionnelle** (extras ``google-adk[...]``) — **codegen-only**
(le toolkit n'importe jamais ces extras ; il émet du code que l'utilisateur exécute dans son propre
venv) : ``add_bigquery``, ``add_spanner``, ``add_mcp_toolset``, ``add_apihub``, ``add_langchain``,
``add_crewai``, plus ``set_auth`` (attache une sous-spec d'auth à un toolset compatible).

Le codegen réel et la sémantique vivent dans :mod:`adk_toolkit_mcp.project_model` (pur, testable).
Voir ``docs/adk-api-notes/tools.md`` pour les signatures ADK confirmées (builtins = instances,
toolsets directs dans ``tools=[...]``, fonction auto-wrappée en ``FunctionTool`` par ADK, chemins
d'import + classes d'auth confirmés par introspection).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    BUILTIN_TOOLS,
    AuthSpec,
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


def _replace_tool_fields(tool: ToolSpec, **changes: Any) -> ToolSpec:
    """Renvoie une copie **immuable** de ``tool`` avec les champs ``changes`` remplacés.

    Fin wrapper de :func:`dataclasses.replace` (``ToolSpec`` est ``frozen``) — préserve l'identité
    (``ref_key`` inchangé tant qu'on ne touche ni ``name``/``kind``), donc ``add_or_replace_tool``
    remplace bien l'outil en place (pas de doublon).
    """
    return replace(tool, **changes)


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
# Outils MCP — passe 3b (toolsets à dépendance optionnelle, codegen-only)
# --------------------------------------------------------------------------- #
def _toolset_name(name: str | None, agent_name: str, suffix: str) -> str:
    """Nom de variable du toolset : ``name`` fourni, sinon ``<agent_name>_<suffix>``."""
    return name if name is not None else f"{agent_name}_{suffix}"


@tools_server.tool
def add_bigquery(
    path: str,
    app_name: str,
    agent_name: str,
    name: str | None = None,
    args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attache un ``BigQueryToolset`` (extra ``google-adk[bigquery]``) à ``agent_name``.

    Génère ``<name> = BigQueryToolset(<args>)`` au niveau module et place ``<name>``
    directement dans ``tools=[...]``. ``args`` sont des **expressions source** (pas des
    littéraux chaîne) : ex. ``{"bigquery_tool_config": "my_cfg"}`` référence une variable que
    vous définissez par ailleurs. ``name`` défaut = ``<agent_name>_bigquery``. **Codegen-only** :
    le toolkit n'importe pas l'extra (cf. ``docs/adk-api-notes/tools.md``).
    """
    toolset_name = _toolset_name(name, agent_name, "bigquery")
    arg_pairs = tuple((str(k), str(v)) for k, v in (args or {}).items())
    tool = ToolSpec(kind="bigquery", name=toolset_name, args=arg_pairs)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_spanner(
    path: str,
    app_name: str,
    agent_name: str,
    name: str | None = None,
    args: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Attache un ``SpannerToolset`` (extra ``google-adk[spanner]``) à ``agent_name``.

    Comme :func:`add_bigquery` mais pour Spanner : ``<name> = SpannerToolset(<args>)``.
    ``args`` = expressions source (ex. ``{"credentials_config": "my_creds"}``). ``name`` défaut
    = ``<agent_name>_spanner``. **Codegen-only**.
    """
    toolset_name = _toolset_name(name, agent_name, "spanner")
    arg_pairs = tuple((str(k), str(v)) for k, v in (args or {}).items())
    tool = ToolSpec(kind="spanner", name=toolset_name, args=arg_pairs)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_mcp_toolset(
    path: str,
    app_name: str,
    agent_name: str,
    transport: str,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    tool_filter: list[str] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Attache un ``McpToolset`` (extra ``google-adk[mcp]``) à ``agent_name``.

    ``transport`` ∈ {``stdio``, ``sse``, ``http``} :

    - ``stdio`` : ``command`` requis (+ ``args`` optionnels) -> ``StdioConnectionParams(
      server_params=StdioServerParameters(command=..., args=[...]))`` ;
    - ``sse`` / ``http`` : ``url`` requis (+ ``headers`` optionnels) -> ``SseConnectionParams`` /
      ``StreamableHTTPConnectionParams(url=..., headers={...})``.

    ``tool_filter`` (optionnel) restreint les outils exposés. ``name`` défaut =
    ``<agent_name>_mcp``. Le toolset entre directement dans ``tools=[...]``. **Codegen-only**.
    """
    toolset_name = _toolset_name(name, agent_name, "mcp")
    tool = ToolSpec(
        kind="mcp_toolset",
        name=toolset_name,
        transport=transport,
        command=command or "",
        mcp_args=tuple(args or ()),
        url=url or "",
        headers=tuple((str(k), str(v)) for k, v in (headers or {}).items()),
        tool_filter=tuple(tool_filter or ()),
    )
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_apihub(
    path: str,
    app_name: str,
    agent_name: str,
    apihub_resource_name: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Attache un ``APIHubToolset`` à ``agent_name`` (API Hub de Google Cloud).

    Génère ``<name> = APIHubToolset(apihub_resource_name="...")`` et le place dans
    ``tools=[...]``. ``apihub_resource_name`` est la ressource API Hub (ex.
    ``projects/<p>/locations/<l>/apis/<a>``). ``name`` défaut = ``<agent_name>_apihub``.
    Auth attachable via :func:`set_auth`. **Codegen-only**.
    """
    toolset_name = _toolset_name(name, agent_name, "apihub")
    tool = ToolSpec(kind="apihub", name=toolset_name, apihub_resource_name=apihub_resource_name)
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_langchain(
    path: str,
    app_name: str,
    agent_name: str,
    import_line: str,
    tool_expr: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Attache un outil LangChain enveloppé via ``LangchainTool`` (extra ``google-adk[community]``).

    Le toolkit ne connaît pas votre outil LangChain : vous fournissez ``import_line`` (rendu
    **verbatim**, ex. ``from langchain_community.tools import WikipediaQueryRun``) et ``tool_expr``
    (l'expression de construction, ex. ``WikipediaQueryRun(api_wrapper=wrapper)``). Rendu :
    ``LangchainTool(tool=<tool_expr>)`` dans ``tools=[...]``. ``name`` est accepté mais
    actuellement non rendu (le wrapper LangChain dérive son nom). **Codegen-only**.
    """
    tool = ToolSpec(
        kind="langchain",
        import_line=import_line,
        tool_expr=tool_expr,
        name=name or "",
    )
    return _attach(path, app_name, agent_name, tool)


@tools_server.tool
def add_crewai(
    path: str,
    app_name: str,
    agent_name: str,
    import_line: str,
    tool_expr: str,
    name: str,
    description: str,
) -> dict[str, Any]:
    """Attache un outil CrewAI enveloppé via ``CrewaiTool`` (extra ``google-adk[community]``).

    Comme :func:`add_langchain` mais pour CrewAI. ``CrewaiTool`` **exige** un ``name``
    (keyword-only, confirmé) ; ``description`` requise ici pour un rendu explicite.
    Rendu : ``CrewaiTool(tool=<tool_expr>, name="...", description="...")``. **Codegen-only**.
    """
    tool = ToolSpec(
        kind="crewai",
        import_line=import_line,
        tool_expr=tool_expr,
        name=name,
        description=description,
    )
    return _attach(path, app_name, agent_name, tool)


# --------------------------------------------------------------------------- #
# Outil MCP — auth (set_auth) : attache une sous-spec auth à un toolset existant
# --------------------------------------------------------------------------- #
@tools_server.tool
def set_auth(
    path: str,
    app_name: str,
    agent_name: str,
    tool_name: str,
    scheme: str,
    credential: dict[str, str],
) -> dict[str, Any]:
    """Attache une **auth** (``scheme`` + ``credential``) à un toolset déjà présent sur l'agent.

    ``tool_name`` désigne la **variable de toolset** (le ``name`` passé à ``add_openapi`` /
    ``add_apihub`` / ``add_mcp_toolset``). Seuls ces genres acceptent l'auth (confirmé :
    ``OpenAPIToolset``/``APIHubToolset``/``McpToolset`` ont ``auth_scheme``/``auth_credential`` ;
    ``BigQueryToolset``/``SpannerToolset`` non -> refus).

    ``scheme`` ∈ {``apikey``, ``oauth2``, ``service_account``, ``bearer``}. ``credential`` est un
    dict de champs (ex. ``{"api_key": "..."}``, ``{"token": "..."}``, ``{"client_id": "...",
    "client_secret": "..."}``). Rendu : ``auth_credential=AuthCredential(...)`` sur le toolset
    (+ imports ``google.adk.auth``). Sémantique idempotente (remplace par nom). **Codegen-only**.
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

    # Cherche le toolset cible (par sa variable ``name``) parmi les outils de l'agent.
    target = next(
        (
            t
            for t in agent.tool_specs()
            if t.name == tool_name
            and t.kind in ("openapi", "apihub", "mcp_toolset", "bigquery", "spanner")
        ),
        None,
    )
    if target is None:
        return err(
            f"Toolset introuvable sur {agent_name!r} : {tool_name!r}. "
            "set_auth cible un toolset existant (openapi/apihub/mcp_toolset) par son 'name'."
        )

    cred_pairs = tuple((str(k), str(v)) for k, v in credential.items())
    updated_tool = _replace_tool_fields(target, auth=AuthSpec(scheme=scheme, credential=cred_pairs))

    # Re-valide (rejette auth sur bigquery/spanner, schéma inconnu, champ requis manquant…).
    tool_error = validate_tool_spec(updated_tool, model, agent_name)
    if tool_error is not None:
        return err(tool_error)

    updated_agent = add_or_replace_tool(agent, updated_tool)
    model = add_or_update_agent(model, updated_agent)
    result = _commit(path, app_name, model)
    if result["ok"]:
        result["data"]["agent"] = agent_name
        result["data"]["tools"] = [t.ref_key() for t in updated_agent.tool_specs()]
    return result


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
    elif tool.kind in ("openapi", "bigquery", "spanner"):
        summary["name"] = tool.name
        if tool.args:
            summary["args"] = dict(tool.args)
    elif tool.kind == "mcp_toolset":
        summary["name"] = tool.name
        summary["transport"] = tool.transport
    elif tool.kind == "apihub":
        summary["name"] = tool.name
        summary["apihub_resource_name"] = tool.apihub_resource_name
    elif tool.kind in ("langchain", "crewai"):
        summary["tool_expr"] = tool.tool_expr
        if tool.name:
            summary["name"] = tool.name
    if tool.auth is not None:
        summary["auth"] = {"scheme": tool.auth.scheme}
    return summary
