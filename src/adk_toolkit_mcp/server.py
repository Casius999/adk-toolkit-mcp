from __future__ import annotations

from fastmcp import FastMCP

from .domains.agents import agents_server
from .domains.project import project_server
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
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
