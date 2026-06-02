from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

from . import __version__


def _safe(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def adk_versions() -> dict[str, str]:
    """Installed versions of the key components (pure, no ADK runtime import)."""
    return {
        "adk_toolkit_mcp": __version__,
        "google_adk": _safe("google-adk"),
        "fastmcp": _safe("fastmcp"),
        "python": platform.python_version(),
    }
