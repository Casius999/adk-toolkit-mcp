"""`project` domain: scaffolding and inspection of ADK apps (code-first).

A FastMCP sub-server mounted by the root server under the ``project`` namespace (the tools are
then exposed as ``project_<name>`` on the client side).

Naming convention: the functions are named with BARE names (``create``, ``inspect``,
``set_env``, …). Mounting with ``namespace="project"`` produces the exposed names
``project_create``, ``project_inspect``, ``project_set_env``, etc. (a single prefix). See
``docs/adk-api-notes/conventions.md``.

Each tool returns the uniform ``{ok, data, error}`` envelope and writes real files via
:class:`~adk_toolkit_mcp.workspace.Workspace`. The layout produced by ``create`` mirrors the real
output of ``adk create`` (see ``docs/adk-api-notes/project.md``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from ..envelope import err, ok
from ..workspace import Workspace

project_server: FastMCP = FastMCP("project")

Backend = Literal["ai_studio", "vertex"]

#: Known ``google-adk`` extras (cf. the root project's pyproject).
KNOWN_EXTRAS: frozenset[str] = frozenset(
    {"gcp", "bigquery", "spanner", "a2a", "eval", "mcp", "community", "litellm"}
)

#: File name of the Agent Config (ADK's no-code path).
AGENT_CONFIG_FILE = "root_agent.yaml"

#: app_name = Python package identifier (serves as both folder AND module name).
_APP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Value displayed in place of any `.env` value (systematic redaction).
_REDACTED = "***"


# --------------------------------------------------------------------------- #
# Pure helpers (indirectly testable via the tools)
# --------------------------------------------------------------------------- #
def _agent_py(app_name: str, model: str) -> str:
    """Source of ``agent.py`` mirroring ``adk create`` (alias imported as LlmAgent)."""
    return (
        "from google.adk.agents import LlmAgent\n"
        "\n"
        "root_agent = LlmAgent(\n"
        f"    model='{model}',\n"
        f"    name='{app_name}',\n"
        "    description='A helpful assistant for user questions.',\n"
        "    instruction='Answer user questions to the best of your knowledge',\n"
        ")\n"
    )


def _env_content(backend: Backend) -> str:
    """`.env` content per backend.

    NB: we write ``FALSE``/``TRUE`` (readable, requested by the spec) where the real scaffolder
    writes ``0``/``1`` — both are accepted by ADK at runtime.
    """
    if backend == "vertex":
        return "GOOGLE_GENAI_USE_VERTEXAI=TRUE\nGOOGLE_CLOUD_PROJECT=\nGOOGLE_CLOUD_LOCATION=\n"
    return "GOOGLE_GENAI_USE_VERTEXAI=FALSE\nGOOGLE_API_KEY=\n"


def _parse_env(text: str) -> dict[str, str]:
    """Minimal parse of a `.env` (``KEY=VALUE`` per line; ignores empties/comments)."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            result[key] = value.strip()
    return result


def _serialize_env(values: dict[str, str]) -> str:
    """Serialize a mapping into a deterministic `.env` (insertion order preserved)."""
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _redact_keys(keys: list[str]) -> dict[str, str]:
    """Map each key to its displayable value (always redacted)."""
    return {key: _REDACTED for key in keys}


def _inject_extra_pyproject(text: str, extra: str) -> tuple[str, bool]:
    """Add the `google-adk[<extra>]` extra to the pyproject dependencies.

    Returns (new_text, changed). Idempotent: if the extra is already present on a
    ``google-adk[...]`` line, touches nothing.
    """
    if re.search(rf"google-adk\[[^\]]*\b{re.escape(extra)}\b[^\]]*\]", text):
        return text, False

    # Case 1: a google-adk line exists -> we graft the extra onto it.
    bare = re.search(r'(["\'])google-adk(?P<spec>[^"\']*)\1', text)
    if bare is not None:
        spec = bare.group("spec")
        if spec.startswith("["):
            close = spec.index("]")
            new_spec = spec[:close] + f",{extra}" + spec[close:]
        else:
            new_spec = f"[{extra}]" + spec
        replacement = f"{bare.group(1)}google-adk{new_spec}{bare.group(1)}"
        return text[: bare.start()] + replacement + text[bare.end() :], True

    # Case 2: no google-adk -> we insert into the dependencies = [ ... ] list.
    dep = re.search(r"dependencies\s*=\s*\[", text)
    if dep is not None:
        insert_at = dep.end()
        line = f'\n    "google-adk[{extra}]",'
        return text[:insert_at] + line + text[insert_at:], True

    return text, False


def _validate_app_name(app_name: str) -> str | None:
    """Return an error message if invalid, otherwise None."""
    name = app_name.strip()
    if not name:
        return "app_name is empty."
    if not _APP_NAME_RE.match(name):
        return (
            f"Invalid app_name: {app_name!r}. Expected a Python identifier "
            "(letters, digits, underscore; not starting with a digit)."
        )
    return None


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@project_server.tool(tags={"project"})
def create(
    path: str,
    app_name: str,
    model: str = "gemini-2.5-flash",
    backend: Backend = "ai_studio",
) -> dict[str, Any]:
    """Scaffold an ADK app in ``<path>/<app_name>/`` (mirrors ``adk create``).

    Writes ``__init__.py``, ``agent.py`` (with a ``root_agent = LlmAgent(...)``) and a ``.env``
    suited to the backend. Idempotent: an identical second call rewrites nothing.
    """
    name_error = _validate_app_name(app_name)
    if name_error is not None:
        return err(name_error)
    if backend not in ("ai_studio", "vertex"):
        return err(f"Invalid backend: {backend!r}. Expected 'ai_studio' or 'vertex'.")
    if not model.strip():
        return err("model is empty.")

    app_name = app_name.strip()
    ws = Workspace(Path(path) / app_name)

    files = {
        "__init__.py": "from . import agent\n",
        "agent.py": _agent_py(app_name, model),
        ".env": _env_content(backend),
    }
    changed = False
    created: list[str] = []
    for relative, content in files.items():
        if ws.write(relative, content):
            changed = True
        created.append(str(ws.path(relative)))

    return ok({"app_name": app_name, "backend": backend, "created": created, "changed": changed})


@project_server.tool(tags={"project"})
def inspect(path: str) -> dict[str, Any]:
    """Inspect an ADK app: presence of ``root_agent``, ``*.py`` files, `.env` keys.

    The `.env` values are never returned (only the key names).
    """
    root = Path(path)
    if not root.exists():
        return err(f"Path not found: {path}")
    if not root.is_dir():
        return err(f"Path is not a directory: {path}")

    ws = Workspace(root)
    py_files = sorted(p.name for p in root.glob("*.py"))

    env_keys: list[str] = []
    if ws.exists(".env"):
        env_keys = sorted(_parse_env(ws.read(".env")).keys())

    return ok(
        {
            "path": str(root),
            "has_root_agent": ws.has_root_agent(),
            "py_files": py_files,
            "env_keys": env_keys,
        }
    )


@project_server.tool(tags={"project"})
def set_env(path: str, values: dict[str, str]) -> dict[str, Any]:
    """Merge ``values`` into the project's `.env` (idempotent, without overwriting the rest).

    Creates the `.env` if it does not exist. Returns the resulting keys (redacted values).
    """
    if not values:
        return err("values is empty: nothing to write.")
    if not all(isinstance(k, str) and k.strip() for k in values):
        return err("All keys in values must be non-empty strings.")

    root = Path(path)
    if not root.exists():
        return err(f"Path not found: {path}")

    ws = Workspace(root)
    merged: dict[str, str] = {}
    if ws.exists(".env"):
        merged.update(_parse_env(ws.read(".env")))
    merged.update({k.strip(): v for k, v in values.items()})

    changed = ws.write(".env", _serialize_env(merged))
    return ok({"env_keys": _redact_keys(sorted(merged.keys())), "changed": changed})


@project_server.tool(tags={"project"})
def add_extra(path: str, extra: str) -> dict[str, Any]:
    """Add a ``google-adk`` extra to the project's dependencies.

    Modifies ``pyproject.toml`` if it exists, otherwise writes a line in ``requirements.txt``.
    Idempotent. Rejects any extra outside KNOWN_EXTRAS.
    """
    extra = extra.strip()
    if extra not in KNOWN_EXTRAS:
        return err(f"Unknown extra: {extra!r}. Known: {', '.join(sorted(KNOWN_EXTRAS))}.")

    root = Path(path)
    if not root.exists():
        return err(f"Path not found: {path}")

    ws = Workspace(root)

    if ws.exists("pyproject.toml"):
        new_text, changed = _inject_extra_pyproject(ws.read("pyproject.toml"), extra)
        if changed:
            ws.write("pyproject.toml", new_text)
        return ok({"target": "pyproject.toml", "extra": extra, "changed": changed})

    # No pyproject -> requirements.txt.
    line = f"google-adk[{extra}]"
    existing = ws.read("requirements.txt") if ws.exists("requirements.txt") else ""
    if re.search(rf"google-adk\[[^\]]*\b{re.escape(extra)}\b[^\]]*\]", existing):
        return ok({"target": "requirements.txt", "extra": extra, "changed": False})
    updated = existing + (line + "\n") if not existing else existing.rstrip("\n") + f"\n{line}\n"
    changed = ws.write("requirements.txt", updated)
    return ok({"target": "requirements.txt", "extra": extra, "changed": changed})


@project_server.tool(tags={"project"})
def agent_config(path: str, yaml_content: str | None = None) -> dict[str, Any]:
    """ADK's Agent Config (no-code) path.

    If ``yaml_content`` is provided, writes ``<path>/root_agent.yaml`` (idempotent). Otherwise,
    returns the expected path and whether the file already exists.
    """
    root = Path(path)
    if not root.exists():
        return err(f"Path not found: {path}")

    ws = Workspace(root)
    if yaml_content is not None:
        changed = ws.write(AGENT_CONFIG_FILE, yaml_content)
        return ok({"path": str(ws.path(AGENT_CONFIG_FILE)), "exists": True, "changed": changed})

    return ok({"path": str(ws.path(AGENT_CONFIG_FILE)), "exists": ws.exists(AGENT_CONFIG_FILE)})
