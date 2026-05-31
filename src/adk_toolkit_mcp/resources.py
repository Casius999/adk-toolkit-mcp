from __future__ import annotations

from fastmcp import FastMCP

from .versions import adk_versions


def register_resources(mcp: FastMCP) -> None:
    """Enregistre les resources lecture-seule. Étendu aux phases suivantes."""

    @mcp.resource("adk://version", mime_type="application/json")
    def version_resource() -> dict[str, str]:
        return adk_versions()
