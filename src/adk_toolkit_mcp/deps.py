from __future__ import annotations

import importlib
from types import ModuleType


class MissingDependency(Exception):
    """Optional dependency missing — actionable message."""


def require(module: str, extra: str) -> ModuleType:
    """Import `module` on demand. Raises MissingDependency with the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(
            f"Module '{module}' unavailable. Install the matching extra: "
            f"uv add 'adk-toolkit-mcp[{extra}]'  (or directly 'google-adk[{extra}]')."
        ) from exc
