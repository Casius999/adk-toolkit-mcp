from __future__ import annotations

from fastmcp import FastMCP

from .domains.agents import agents_server
from .domains.artifacts import artifacts_server
from .domains.deploy import deploy_server
from .domains.dev import dev_server
from .domains.eval import eval_server
from .domains.mcp_bridge import mcp_bridge_server
from .domains.memory import memory_server
from .domains.models import models_server
from .domains.project import project_server
from .domains.run import run_server
from .domains.sessions import sessions_server
from .domains.tools import tools_server
from .prompts import register_prompts
from .resources import register_resources

SERVER_NAME = "adk-toolkit-mcp"


def build_server() -> FastMCP:
    """Construit le serveur MCP racine. Sous-serveurs montés en P1→P4. Code Mode en P6."""
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
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
