from __future__ import annotations

import os

from fastmcp import FastMCP

from .domains.a2a import a2a_server
from .domains.agents import agents_server
from .domains.artifacts import artifacts_server
from .domains.deploy import deploy_server
from .domains.dev import dev_server
from .domains.eval import eval_server
from .domains.mcp_bridge import mcp_bridge_server
from .domains.memory import memory_server
from .domains.models import models_server
from .domains.observability import observability_server
from .domains.project import project_server
from .domains.run import run_server
from .domains.safety import safety_server
from .domains.sessions import sessions_server
from .domains.tools import tools_server
from .prompts import register_prompts
from .resources import register_resources

SERVER_NAME = "adk-toolkit-mcp"

#: Valeurs d'env reconnues comme « vrai » pour activer le Code Mode (insensible à la casse).
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})

#: Variable d'env activant le Code Mode au lancement (``main()``).
_CODE_MODE_ENV = "ADK_TOOLKIT_CODE_MODE"


def code_mode_enabled() -> bool:
    """Vrai si la variable d'env ``ADK_TOOLKIT_CODE_MODE`` demande le Code Mode.

    Reconnaît ``1``/``true``/``yes``/``on`` (insensible à la casse). Toute autre valeur
    (ou l'absence de variable) → ``False`` (mode outils-directs par défaut).
    """
    return (os.getenv(_CODE_MODE_ENV) or "").strip().lower() in _TRUTHY


def _apply_code_mode(mcp: FastMCP) -> None:
    """Effondre le catalogue d'outils en une petite surface discovery + execute (Code Mode).

    Applique le VRAI transform FastMCP 3.3.1
    (:class:`fastmcp.experimental.transforms.code_mode.CodeMode`) via
    :meth:`FastMCP.add_transform`. La surface exposée passe alors des 81 outils nommés à
    seulement ``search`` / ``get_schema`` / ``tags`` / ``execute`` (gros gain de tokens pour
    un gros catalogue). Les outils de découverte lisent ``tool.tags`` — d'où l'intérêt
    d'avoir tagué chaque outil par domaine (TASK 1) : ``GetTags`` liste les 15 domaines, puis
    ``search(tags=[...])`` filtre par domaine.

    NB (honnêteté, cf. ``docs/adk-api-notes/fastmcp-codemode.md``) : les outils de découverte
    (``search``/``get_schema``/``tags``) fonctionnent SANS dépendance supplémentaire ; seul
    l'outil ``execute`` (sandbox ``MontySandboxProvider`` par défaut) nécessite le paquet
    optionnel ``pydantic-monty`` (extra ``fastmcp[code-mode]``), importé paresseusement à
    l'appel. Le transform est donc « câblé » ici, mais l'exécution de code requiert l'extra.
    L'import est local pour ne rien coûter au mode outils-directs (par défaut).
    """
    from fastmcp.experimental.transforms.code_mode import CodeMode, GetSchemas, GetTags, Search

    # GetTags est ajouté à la liste par défaut (Search + GetSchemas) car on tague par domaine :
    # le modèle peut parcourir les domaines, puis search(tags=[...]), puis get_schema, puis execute.
    mcp.add_transform(CodeMode(discovery_tools=[Search(), GetSchemas(), GetTags()]))


def build_server(code_mode: bool = False) -> FastMCP:
    """Construit le serveur MCP racine (15 sous-serveurs, 81 outils).

    Par défaut (``code_mode=False``), tous les outils sont exposés par leur nom
    ``<domaine>_<bare>`` (UX outils-directs ; les tests read-through les appellent par nom).

    Si ``code_mode=True``, on applique le transform Code Mode de FastMCP 3.3.1 APRÈS avoir
    monté tous les sous-serveurs : le catalogue est effondré en une surface discovery+execute
    (``search``/``get_schema``/``tags``/``execute``) — économie de tokens pour les 81 outils.
    Voir :func:`_apply_code_mode` et ``docs/adk-api-notes/fastmcp-codemode.md`` (l'outil
    ``execute`` requiert l'extra ``fastmcp[code-mode]`` ; la découverte fonctionne sans).
    """
    mcp = FastMCP(SERVER_NAME)
    register_resources(mcp)
    register_prompts(mcp)
    # P1 domaine 1/4 : project. namespace -> outils exposés comme `project_<nom>`.
    # (`prefix=` est déprécié en fastmcp 3.3.1 ; `namespace=` est l'API courante.)
    mcp.mount(project_server, namespace="project")
    # P1 domaine 2/4 : agents. Outils exposés comme `agents_<nom>`.
    mcp.mount(agents_server, namespace="agents")
    # P3 domaine 3/4 : tools. Outils exposés comme `tools_<nom>`.
    mcp.mount(tools_server, namespace="tools")
    # P1 domaine 4/4 : models. Outils exposés comme `models_<nom>`.
    mcp.mount(models_server, namespace="models")
    # P2 domaine a : sessions (runtime). Outils exposés comme `sessions_<nom>`.
    mcp.mount(sessions_server, namespace="sessions")
    # P2 domaine b : memory (runtime). Outils exposés comme `memory_<nom>`.
    mcp.mount(memory_server, namespace="memory")
    # P2 domaine b : artifacts (runtime). Outils exposés comme `artifacts_<nom>`.
    mcp.mount(artifacts_server, namespace="artifacts")
    # P3 domaine a : run (exécution d'agents). Outils exposés comme `run_<nom>`.
    mcp.mount(run_server, namespace="run")
    # P3 domaine b : eval (évaluation d'agents). Outils exposés comme `eval_<nom>`.
    mcp.mount(eval_server, namespace="eval")
    # P4 domaine a : deploy (construction de commandes adk deploy). Exposés `deploy_<nom>`.
    mcp.mount(deploy_server, namespace="deploy")
    # P4 domaine a : dev (serveurs de dev longue durée + one-shot run). Exposés `dev_<nom>`.
    mcp.mount(dev_server, namespace="dev")
    # P4 domaine b : mcp_bridge (exposer des outils ADK comme MCP). Exposés `mcp_bridge_<nom>`.
    mcp.mount(mcp_bridge_server, namespace="mcp_bridge")
    # P4 domaine b : a2a (consume/expose/agent_card Agent-to-Agent). Exposés `a2a_<nom>`.
    mcp.mount(a2a_server, namespace="a2a")
    # P4 domaine c : safety (callbacks/plugins/réglages de sûreté). Exposés `safety_<nom>`.
    mcp.mount(safety_server, namespace="safety")
    # P4 domaine c : observability (OpenTelemetry/Cloud Trace). Exposés `observability_<nom>`.
    mcp.mount(observability_server, namespace="observability")
    # P6 : Code Mode opt-in — APRÈS tous les mounts (le transform agit sur le catalogue complet).
    if code_mode:
        _apply_code_mode(mcp)
    return mcp


def main() -> None:
    """Point d'entrée CLI : lance le serveur (Code Mode si ``ADK_TOOLKIT_CODE_MODE`` est vrai)."""
    build_server(code_mode=code_mode_enabled()).run()


if __name__ == "__main__":
    main()
