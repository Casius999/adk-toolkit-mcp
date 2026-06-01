"""Domaine `mcp_bridge` : exposer des outils ADK **comme** des outils MCP (P4b).

Ce domaine est le pont « ADK → MCP » : il convertit des `BaseTool` ADK en **schémas d'outils
MCP** (`mcp.types.Tool` : ``{name, description, inputSchema}``) via la fonction officielle
``google.adk.tools.mcp_tool.conversion_utils.adk_to_mcp_tool_type``. C'est l'opération distincte
du domaine P1 ``tools`` (qui *consomme* des serveurs MCP via ``McpToolset``) : ici on rend les
outils d'un agent **publiables** comme un serveur MCP.

Le paquet ``mcp`` est une dépendance **CORE** (``fastmcp`` en dépend) : ce domaine est donc
entièrement testable en CI **sans aucun extra** (contrairement à ``a2a``). Cf.
``docs/adk-api-notes/a2a-mcp-bridge.md`` pour les signatures confirmées et le résultat
FONCTIONNEL (le schéma MCP obtenu d'un vrai outil ADK).

Outils exposés sous ``namespace="mcp_bridge"`` → ``mcp_bridge_<nom>``. Noms BARE :

- ``expose_adk_tools(path, app_name, agent_name)`` — importe le ``root_agent`` du projet, localise
  l'agent ``agent_name`` (via ``BaseAgent.find_agent``), normalise ses outils en ``BaseTool`` (via
  ``await agent.canonical_tools()``, qui enveloppe les fonctions nues en ``FunctionTool``), et rend
  la liste de schémas MCP. Chemin ROBUSTE : on opère sur l'agent RÉELLEMENT construit (les specs
  du sidecar deviennent les vrais objets ADK), pas sur une re-dérivation des specs.
- ``convert_builtin(kind)`` — instancie un seul builtin ADK « core » par son ``kind`` (ex.
  ``google_search``) et renvoie son schéma MCP. Pratique pour inspecter un builtin isolé.

Chaque outil renvoie l'enveloppe ``{ok, data, error}`` ; les entrées invalides → ``err(...)``
(jamais d'exception qui remonte). Les imports ADK sont **paresseux** (au point d'appel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import CORE_BUILTINS, is_identifier
from ..run_core import RootAgentImportError, import_root_agent

if TYPE_CHECKING:  # pragma: no cover - hints seulement, imports réels paresseux
    from google.adk.tools import BaseTool

mcp_bridge_server: FastMCP = FastMCP("mcp_bridge")

#: app_name = identifiant de package Python (nom de dossier ET de module).
_APP_NAME_ERR = (
    "app_name invalide : attendu un identifiant Python "
    "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
)


# --------------------------------------------------------------------------- #
# Conversion ADK BaseTool -> schéma MCP (mcp.types.Tool)
# --------------------------------------------------------------------------- #
def _to_mcp_schema(tool: BaseTool) -> dict[str, Any]:
    """Convertit un ``BaseTool`` ADK en dict ``{name, description, inputSchema}`` (forme MCP).

    Délègue à ``adk_to_mcp_tool_type`` (qui renvoie un ``mcp.types.Tool``) puis n'expose que les
    trois champs qui nous intéressent. ``inputSchema`` est déjà un dict JSON-Schema (vide ``{}``
    pour un builtin sans paramètres déclarés, ex. ``google_search``).
    """
    from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

    mcp_tool = adk_to_mcp_tool_type(tool)
    return {
        "name": mcp_tool.name,
        "description": mcp_tool.description,
        "inputSchema": mcp_tool.inputSchema,
    }


def _builtin_to_base_tool(kind: str) -> BaseTool:
    """Instancie un builtin « core » (``kind`` ∈ :data:`CORE_BUILTINS`) en un ``BaseTool``.

    Certains « builtins core » sont déjà des **instances** ``BaseTool`` (ex. ``google_search`` =
    ``GoogleSearchTool()``) ; d'autres sont de simples **fonctions** (``exit_loop``,
    ``transfer_to_agent``) qu'on enveloppe alors en ``FunctionTool`` pour pouvoir les convertir.
    """
    import google.adk.tools as adk_tools
    from google.adk.tools import BaseTool as _BaseTool
    from google.adk.tools import FunctionTool

    obj = getattr(adk_tools, kind)
    if isinstance(obj, _BaseTool):
        return obj
    # Fonction nue (exit_loop / transfer_to_agent) -> on l'enveloppe en FunctionTool.
    return FunctionTool(obj)


# --------------------------------------------------------------------------- #
# Outil MCP — convert_builtin
# --------------------------------------------------------------------------- #
@mcp_bridge_server.tool(tags={"mcp_bridge"})
def convert_builtin(kind: str) -> dict[str, Any]:
    """Instancie un builtin ADK « core » par ``kind`` et renvoie son schéma MCP.

    Ex. ``convert_builtin("google_search")`` → ``{name, description, inputSchema}`` (un
    ``mcp.types.Tool`` aplati). Seuls les builtins **core** (sans argument requis) sont supportés
    ici — ``vertex_ai_search`` exige un ``data_store_id`` et doit être attaché à un agent puis
    exposé via :func:`expose_adk_tools`. Un ``kind`` inconnu → ``err`` listant les kinds connus.

    Le paquet ``mcp`` est core → cet outil fonctionne sans aucun extra (testable en CI).
    """
    if kind not in CORE_BUILTINS:
        return err(
            f"Builtin core inconnu : {kind!r}. Connus : {', '.join(sorted(CORE_BUILTINS))}. "
            "(Les builtins à argument comme 'vertex_ai_search' : attache-les à un agent puis "
            "utilise mcp_bridge_expose_adk_tools.)"
        )
    try:
        tool = _builtin_to_base_tool(kind)
        schema = _to_mcp_schema(tool)
    except Exception as exc:  # noqa: BLE001 - conversion best-effort, on remonte un err propre
        return err(f"Échec de conversion du builtin {kind!r} en schéma MCP : {exc}")
    return ok({"kind": kind, "tool": schema})


# --------------------------------------------------------------------------- #
# Outil MCP — expose_adk_tools
# --------------------------------------------------------------------------- #
@mcp_bridge_server.tool(tags={"mcp_bridge"})
async def expose_adk_tools(path: str, app_name: str, agent_name: str) -> dict[str, Any]:
    """Convertit les outils ADK d'un agent du projet en **schémas d'outils MCP**.

    Réalise « exposer les outils ADK COMME des outils MCP » : importe le ``root_agent`` du projet
    (``<path>/<app_name>/agent.py``), localise l'agent ``agent_name`` dans l'arbre (via
    ``BaseAgent.find_agent`` — fonctionne aussi pour la racine elle-même), normalise ses outils en
    ``BaseTool`` (``await agent.canonical_tools()`` enveloppe les fonctions nues en
    ``FunctionTool``), puis convertit chacun via ``adk_to_mcp_tool_type``.

    Renvoie ``{app_name, agent_name, count, tools: [{name, description, inputSchema}, ...]}``. Un
    agent sans outils renvoie une liste vide (``count=0``, pas une erreur). Erreurs propres
    (``err``) si : ``app_name``/``agent_name`` invalides, ``agent.py`` absent/illisible
    (``RootAgentImportError``), agent introuvable dans l'arbre, ou agent sans capacité d'outils
    (ex. un agent workflow Sequential/Parallel/Loop n'a pas de ``canonical_tools``).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Nom d'agent invalide : {agent_name!r}. Attendu un identifiant Python.")

    try:
        root_agent = import_root_agent(path, app_name)
    except RootAgentImportError as exc:
        return err(str(exc))

    agent = root_agent.find_agent(agent_name)
    if agent is None:
        return err(
            f"Agent introuvable dans l'arbre du root_agent : {agent_name!r}. "
            "Vérifie le nom (la racine et tous ses sous-agents sont inspectés)."
        )

    # Les agents workflow (Sequential/Parallel/Loop) n'ont pas d'outils : pas de canonical_tools.
    canonical = getattr(agent, "canonical_tools", None)
    if canonical is None:
        return err(
            f"L'agent {agent_name!r} (type {type(agent).__name__}) ne porte pas d'outils ADK "
            "(seuls les agents de type LLM exposent des tools convertibles en MCP)."
        )

    try:
        base_tools = await canonical()
        tools = [_to_mcp_schema(t) for t in base_tools]
    except Exception as exc:  # noqa: BLE001 - on convertit toute erreur en err actionnable
        return err(f"Échec de conversion des outils de {agent_name!r} en schémas MCP : {exc}")

    return ok(
        {
            "app_name": app_name,
            "agent_name": agent_name,
            "count": len(tools),
            "tools": tools,
        }
    )
