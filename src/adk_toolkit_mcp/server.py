from __future__ import annotations

from fastmcp import FastMCP

from .domains.project import project_server
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
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
