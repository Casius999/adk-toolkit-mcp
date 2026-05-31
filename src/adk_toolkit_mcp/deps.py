from __future__ import annotations

import importlib
from types import ModuleType


class MissingDependency(Exception):
    """Dépendance optionnelle absente — message orienté action."""


def require(module: str, extra: str) -> ModuleType:
    """Importe `module` à la demande. Lève MissingDependency avec l'extra à installer."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependency(
            f"Module '{module}' indisponible. Installe l'extra correspondant : "
            f"uv add 'adk-toolkit-mcp[{extra}]'  (ou directement 'google-adk[{extra}]')."
        ) from exc
