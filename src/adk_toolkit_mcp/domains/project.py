"""Domaine `project` : scaffolding et inspection d'apps ADK (code-first).

Sous-serveur FastMCP monté par le serveur racine sous le namespace ``project``
(les outils sont alors exposés comme ``project_<nom>`` côté client).

Chaque outil renvoie l'enveloppe uniforme ``{ok, data, error}`` et écrit de vrais
fichiers via :class:`~adk_toolkit_mcp.workspace.Workspace`. Le layout produit par
``project_create`` reflète la sortie réelle de ``adk create`` (voir
``docs/adk-api-notes/project.md``).
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

#: Extras ``google-adk`` connus (cf. pyproject du projet racine).
KNOWN_EXTRAS: frozenset[str] = frozenset(
    {"gcp", "bigquery", "spanner", "a2a", "eval", "mcp", "community", "litellm"}
)

#: Nom de fichier de l'Agent Config (voie no-code d'ADK).
AGENT_CONFIG_FILE = "root_agent.yaml"

#: app_name = identifiant de package Python (sert de nom de dossier ET de module).
_APP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Valeur affichée à la place de toute valeur `.env` (redaction systématique).
_REDACTED = "***"


# --------------------------------------------------------------------------- #
# Helpers purs (testables indirectement via les outils)
# --------------------------------------------------------------------------- #
def _agent_py(app_name: str, model: str) -> str:
    """Source de ``agent.py`` mirroir de ``adk create`` (alias importé sous LlmAgent)."""
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
    """Contenu `.env` selon le backend.

    NB: on écrit ``FALSE``/``TRUE`` (lisible, demandé par la spec) là où le vrai
    scaffolder écrit ``0``/``1`` — les deux sont acceptés par ADK à l'exécution.
    """
    if backend == "vertex":
        return "GOOGLE_GENAI_USE_VERTEXAI=TRUE\nGOOGLE_CLOUD_PROJECT=\nGOOGLE_CLOUD_LOCATION=\n"
    return "GOOGLE_GENAI_USE_VERTEXAI=FALSE\nGOOGLE_API_KEY=\n"


def _parse_env(text: str) -> dict[str, str]:
    """Parse minimal d'un `.env` (``KEY=VALUE`` par ligne; ignore vides/commentaires)."""
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
    """Sérialise un mapping en `.env` déterministe (ordre d'insertion préservé)."""
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _redact_keys(keys: list[str]) -> dict[str, str]:
    """Map chaque clé vers sa valeur affichable (toujours redacted)."""
    return {key: _REDACTED for key in keys}


def _inject_extra_pyproject(text: str, extra: str) -> tuple[str, bool]:
    """Ajoute l'extra `google-adk[<extra>]` aux dépendances pyproject.

    Renvoie (nouveau_texte, changed). Idempotent : si l'extra est déjà présent
    sur une ligne ``google-adk[...]``, ne touche rien.
    """
    if re.search(rf"google-adk\[[^\]]*\b{re.escape(extra)}\b[^\]]*\]", text):
        return text, False

    # Cas 1 : une ligne google-adk existe -> on y greffe l'extra.
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

    # Cas 2 : pas de google-adk -> on insère dans la liste dependencies = [ ... ].
    dep = re.search(r"dependencies\s*=\s*\[", text)
    if dep is not None:
        insert_at = dep.end()
        line = f'\n    "google-adk[{extra}]",'
        return text[:insert_at] + line + text[insert_at:], True

    return text, False


def _validate_app_name(app_name: str) -> str | None:
    """Renvoie un message d'erreur si invalide, sinon None."""
    name = app_name.strip()
    if not name:
        return "app_name est vide."
    if not _APP_NAME_RE.match(name):
        return (
            f"app_name invalide : {app_name!r}. Attendu un identifiant Python "
            "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
        )
    return None


# --------------------------------------------------------------------------- #
# Outils MCP
# --------------------------------------------------------------------------- #
@project_server.tool
def project_create(
    path: str,
    app_name: str,
    model: str = "gemini-2.5-flash",
    backend: Backend = "ai_studio",
) -> dict[str, Any]:
    """Scaffold une app ADK dans ``<path>/<app_name>/`` (mirroir de ``adk create``).

    Écrit ``__init__.py``, ``agent.py`` (avec un ``root_agent = LlmAgent(...)``) et
    ``.env`` adapté au backend. Idempotent : un second appel identique ne réécrit rien.
    """
    name_error = _validate_app_name(app_name)
    if name_error is not None:
        return err(name_error)
    if backend not in ("ai_studio", "vertex"):
        return err(f"backend invalide : {backend!r}. Attendu 'ai_studio' ou 'vertex'.")
    if not model.strip():
        return err("model est vide.")

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


@project_server.tool
def project_inspect(path: str) -> dict[str, Any]:
    """Inspecte une app ADK : présence de ``root_agent``, fichiers ``*.py``, clés `.env`.

    Les valeurs `.env` ne sont jamais renvoyées (seulement les noms de clés).
    """
    root = Path(path)
    if not root.exists():
        return err(f"Chemin introuvable : {path}")
    if not root.is_dir():
        return err(f"Chemin n'est pas un dossier : {path}")

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


@project_server.tool
def project_set_env(path: str, values: dict[str, str]) -> dict[str, Any]:
    """Fusionne ``values`` dans le `.env` du projet (idempotent, sans écraser le reste).

    Crée le `.env` s'il n'existe pas. Renvoie les clés résultantes (valeurs redacted).
    """
    if not values:
        return err("values est vide : rien à écrire.")
    if not all(isinstance(k, str) and k.strip() for k in values):
        return err("Toutes les clés de values doivent être des chaînes non vides.")

    root = Path(path)
    if not root.exists():
        return err(f"Chemin introuvable : {path}")

    ws = Workspace(root)
    merged: dict[str, str] = {}
    if ws.exists(".env"):
        merged.update(_parse_env(ws.read(".env")))
    merged.update({k.strip(): v for k, v in values.items()})

    changed = ws.write(".env", _serialize_env(merged))
    return ok({"env_keys": _redact_keys(sorted(merged.keys())), "changed": changed})


@project_server.tool
def project_add_extra(path: str, extra: str) -> dict[str, Any]:
    """Ajoute un extra ``google-adk`` aux dépendances du projet.

    Modifie ``pyproject.toml`` s'il existe, sinon écrit une ligne dans
    ``requirements.txt``. Idempotent. Rejette tout extra hors de KNOWN_EXTRAS.
    """
    extra = extra.strip()
    if extra not in KNOWN_EXTRAS:
        return err(f"Extra inconnu : {extra!r}. Connus : {', '.join(sorted(KNOWN_EXTRAS))}.")

    root = Path(path)
    if not root.exists():
        return err(f"Chemin introuvable : {path}")

    ws = Workspace(root)

    if ws.exists("pyproject.toml"):
        new_text, changed = _inject_extra_pyproject(ws.read("pyproject.toml"), extra)
        if changed:
            ws.write("pyproject.toml", new_text)
        return ok({"target": "pyproject.toml", "extra": extra, "changed": changed})

    # Pas de pyproject -> requirements.txt.
    line = f"google-adk[{extra}]"
    existing = ws.read("requirements.txt") if ws.exists("requirements.txt") else ""
    if re.search(rf"google-adk\[[^\]]*\b{re.escape(extra)}\b[^\]]*\]", existing):
        return ok({"target": "requirements.txt", "extra": extra, "changed": False})
    updated = existing + (line + "\n") if not existing else existing.rstrip("\n") + f"\n{line}\n"
    changed = ws.write("requirements.txt", updated)
    return ok({"target": "requirements.txt", "extra": extra, "changed": changed})


@project_server.tool
def project_agent_config(path: str, yaml_content: str | None = None) -> dict[str, Any]:
    """Voie Agent Config (no-code) d'ADK.

    Si ``yaml_content`` est fourni, écrit ``<path>/root_agent.yaml`` (idempotent).
    Sinon, renvoie le chemin attendu et si le fichier existe déjà.
    """
    root = Path(path)
    if not root.exists():
        return err(f"Chemin introuvable : {path}")

    ws = Workspace(root)
    if yaml_content is not None:
        changed = ws.write(AGENT_CONFIG_FILE, yaml_content)
        return ok({"path": str(ws.path(AGENT_CONFIG_FILE)), "exists": True, "changed": changed})

    return ok({"path": str(ws.path(AGENT_CONFIG_FILE)), "exists": ws.exists(AGENT_CONFIG_FILE)})
