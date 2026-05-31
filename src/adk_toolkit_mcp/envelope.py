from __future__ import annotations

from typing import Any


def ok(data: Any = None) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def err(message: str) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": message}
