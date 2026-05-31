from __future__ import annotations

from fastmcp import FastMCP

from .prompts import register_prompts
from .resources import register_resources

SERVER_NAME = "adk-toolkit-mcp"


def build_server() -> FastMCP:
    """Construit le serveur MCP racine. Sous-serveurs montés en P1→P4. Code Mode en P6."""
    mcp = FastMCP(SERVER_NAME)
    register_resources(mcp)
    register_prompts(mcp)
    return mcp


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
