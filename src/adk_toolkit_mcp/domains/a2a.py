"""Domaine `a2a` : interopérabilité Agent-to-Agent (A2A) autour de google-adk (P4b).

Trois opérations, toutes dans l'enveloppe ``{ok, data, error}`` :

1. ``consume(path, app_name, name, agent_card_url)`` — ajoute un agent ``remote_a2a`` (un proxy
   ``RemoteA2aAgent(name=..., agent_card="<url>")``) au sidecar du projet et régénère ``agent.py``.
   Ce proxy peut ensuite être composé comme ``sub_agent`` d'un autre agent (``agents_compose``).
   **Codegen-only** : le toolkit n'importe jamais ``RemoteA2aAgent`` lui-même (l'extra ``a2a``
   n'est requis qu'au *runtime* de l'``agent.py`` généré).

2. ``expose(path, app_name, port=8001, execute=False)`` — génère ``a2a_app.py`` dans le dossier de
   l'app (``a2a_app = to_a2a(root_agent, port=PORT)``), idempotent via :class:`Workspace`. Par
   défaut (``execute=False``) renvoie le chemin du fichier + la commande de service
   (``uvicorn a2a_app:a2a_app``). Si ``execute=True`` ET l'extra ``a2a`` est présent, démarre un
   process ``uvicorn`` géré (via le registre :mod:`adk_toolkit_mcp.adk_cli`) servant sur ``port`` et
   renvoie l'URL de l'agent-card.

3. ``agent_card(path, app_name)`` — best-effort : construit/inspecte l'``AgentCard`` du
   ``root_agent`` du projet via ``AgentCardBuilder`` (nécessite d'importer l'agent + l'extra
   ``a2a``). Gate proprement si l'extra est absent (``err`` actionnable, jamais de crash).

Outils exposés sous ``namespace="a2a"`` → ``a2a_<nom>``. Noms BARE. Les imports ``a2a`` sont
**paresseux** et gatés par ``importlib.util.find_spec("a2a")``. Cf.
``docs/adk-api-notes/a2a-mcp-bridge.md`` : ``to_a2a(agent, *, host, port, ...) -> Starlette`` sert
``/.well-known/agent-card.json`` ; ``RemoteA2aAgent`` vit dans
``google.adk.agents.remote_a2a_agent`` (PAS ``google.adk.agents``).
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .. import adk_cli
from ..envelope import err, ok
from ..project_model import (
    AgentSpec,
    ProjectModel,
    add_or_update_agent,
    is_identifier,
    load_model,
    regenerate,
    save_model,
    validate_spec,
)
from ..run_core import RootAgentImportError, import_root_agent
from ..workspace import Workspace

a2a_server: FastMCP = FastMCP("a2a")

#: app_name = identifiant de package Python (nom de dossier ET de module).
_APP_NAME_ERR = (
    "app_name invalide : attendu un identifiant Python "
    "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
)

#: Nom du fichier généré par ``expose`` (dans le dossier de l'app).
_A2A_APP_FILE = "a2a_app.py"

#: Chemin du routeur well-known de l'agent-card (constante a2a-sdk confirmée).
_WELL_KNOWN_PATH = "/.well-known/agent-card.json"

#: Bornes de port TCP valides.
_PORT_MIN = 1
_PORT_MAX = 65535

#: Dossier des logs des process gérés (réutilise la convention du domaine dev).
_LOG_DIR = ".adk_toolkit/logs"

#: Message actionnable quand l'extra ``a2a`` est absent.
_A2A_EXTRA_ABSENT = (
    "L'extra 'a2a' n'est pas installé (paquet a2a-sdk introuvable). "
    "Installe-le : uv add 'adk-toolkit-mcp[a2a]' (ou pip install 'google-adk[a2a]')."
)


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _load(path: str, app_name: str) -> ProjectModel | dict[str, Any]:
    """Charge le modèle ; renvoie un ``err(...)`` (dict) si le sidecar est corrompu."""
    try:
        return load_model(_app_ws(path, app_name), app_name)
    except ValueError as exc:
        return err(str(exc))


def _a2a_extra_present() -> bool:
    """Vrai si l'extra ``a2a`` (paquet ``a2a-sdk``) est importable."""
    return find_spec("a2a") is not None


def _validate_port(port: int) -> str | None:
    """Renvoie un message d'erreur si ``port`` est hors bornes TCP, sinon None."""
    if not isinstance(port, int) or not (_PORT_MIN <= port <= _PORT_MAX):
        return f"port invalide : {port!r}. Attendu un entier dans [{_PORT_MIN}, {_PORT_MAX}]."
    return None


def _a2a_app_source(port: int) -> str:
    """Source de ``a2a_app.py`` : ``a2a_app = to_a2a(root_agent, port=PORT)``.

    Importe ``root_agent`` depuis le module sibling ``agent`` (le fichier est servi avec
    ``cwd=<app_dir>`` ⇒ ``uvicorn a2a_app:a2a_app`` résout ``agent`` directement). L'import de
    ``to_a2a`` nécessite l'extra ``a2a`` au runtime — c'est du **codegen-only** (le toolkit ne
    l'importe pas pour générer ce fichier).

    Les deux imports (``agent`` et ``google.adk...``) sont des **tiers** du point de vue d'isort
    (aucun n'est un module first-party connu de ruff) : ils forment donc UN seul groupe, trié
    alphabétiquement et **sans ligne vide** de séparation — la sortie est ``ruff check --select I``
    clean (comme l'``agent.py`` régénéré par ``consume``).
    """
    return (
        '"""Généré par adk-toolkit-mcp (a2a_expose). Sert le root_agent via le protocole A2A.\n'
        "\n"
        "Lance :  uvicorn a2a_app:a2a_app --host localhost --port "
        f"{port}\n"
        f"L'agent-card est servie sur {_WELL_KNOWN_PATH}.\n"
        "Nécessite l'extra a2a : uv add 'adk-toolkit-mcp[a2a]'.\n"
        '"""\n'
        "\n"
        "from agent import root_agent\n"
        "from google.adk.a2a.utils.agent_to_a2a import to_a2a\n"
        "\n"
        f"a2a_app = to_a2a(root_agent, port={port})\n"
    )


def _log_path(path: str, app_name: str, port: int) -> str:
    """Chemin du fichier log du process uvicorn géré (sous le sidecar de l'app)."""
    return str(_app_ws(path, app_name).path(f"{_LOG_DIR}/a2a-{port}.log"))


# --------------------------------------------------------------------------- #
# consume — ajoute un proxy RemoteA2aAgent au modèle
# --------------------------------------------------------------------------- #
@a2a_server.tool
def consume(path: str, app_name: str, name: str, agent_card_url: str) -> dict[str, Any]:
    """Ajoute un agent ``remote_a2a`` (proxy ``RemoteA2aAgent``) au projet et régénère ``agent.py``.

    ``name`` est l'identifiant de la variable d'agent générée ; ``agent_card_url`` est l'URL (ou
    le chemin JSON local) de l'agent-card du service A2A distant. Le proxy n'a pas d'enfants mais
    peut être composé comme ``sub_agent`` d'un autre agent ensuite (via ``agents_compose``).

    Renvoie le payload commun ``{app_name, agents, root, sidecar, regenerated, changed}``. Entrées
    invalides (app_name/name non identifiants, URL vide) → ``err``. Codegen-only : aucune
    dépendance ``a2a`` requise pour AJOUTER le proxy (uniquement pour exécuter l'``agent.py``).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(name):
        return err(f"Nom d'agent invalide : {name!r}. Attendu un identifiant Python.")
    if not agent_card_url.strip():
        return err(
            "agent_card_url est vide. Fournis l'URL (ou le chemin JSON) de l'agent-card distant "
            "(ex. 'http://host:8001/.well-known/agent-card.json')."
        )

    spec = AgentSpec(name=name, type="remote_a2a", agent_card=agent_card_url.strip())
    spec_error = validate_spec(spec)
    if spec_error is not None:
        return err(spec_error)

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    model = add_or_update_agent(model, spec)
    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle (improbable pour un proxy sans enfants)
        return err(str(exc))
    sidecar_changed = save_model(ws, model)
    return ok(
        {
            "app_name": app_name,
            "agents": list(model.agent_names()),
            "root": model.root,
            "remote_agent": {"name": name, "agent_card": agent_card_url.strip()},
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


# --------------------------------------------------------------------------- #
# expose — génère a2a_app.py ; optionnellement sert via uvicorn (process géré)
# --------------------------------------------------------------------------- #
@a2a_server.tool
def expose(path: str, app_name: str, port: int = 8001, execute: bool = False) -> dict[str, Any]:
    """Génère ``a2a_app.py`` (``to_a2a(root_agent, port=PORT)``) ; sert optionnellement via uvicorn.

    Écrit ``<path>/<app_name>/a2a_app.py`` (idempotent via :class:`Workspace`). Puis :

    - ``execute=False`` (défaut) : renvoie ``{file, serve_command, agent_card_path, executed:
      False}`` (la commande ``uvicorn a2a_app:a2a_app`` à lancer depuis le dossier de l'app).
      AUCUNE dépendance ``a2a`` requise (codegen-only).
    - ``execute=True`` : nécessite l'extra ``a2a`` (sinon ``err`` actionnable). Démarre un process
      ``uvicorn a2a_app:a2a_app --host localhost --port <port>`` géré (registre adk_cli, cwd = le
      dossier de l'app pour résoudre l'import ``agent``) et renvoie ``{key, pid, url,
      agent_card_url, executed: True, ...}``.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    port_error = _validate_port(port)
    if port_error is not None:
        return err(port_error)

    ws = _app_ws(path, app_name)
    app_dir = ws.root
    if not app_dir.is_dir():
        return err(
            f"Dossier d'app introuvable : {app_dir}. Scaffolde d'abord l'app (project_create)."
        )
    if not ws.exists("agent.py"):
        return err(
            f"agent.py introuvable dans {app_dir}. Crée d'abord un root_agent "
            "(agents_create_llm + agents_set_root)."
        )

    changed = ws.write(_A2A_APP_FILE, _a2a_app_source(port))
    file_path = str(ws.path(_A2A_APP_FILE))
    serve_command = f"uvicorn a2a_app:a2a_app --host localhost --port {port}"

    if not execute:
        return ok(
            {
                "file": file_path,
                "changed": changed,
                "port": port,
                "serve_command": serve_command,
                "serve_cwd": str(app_dir),
                "agent_card_path": _WELL_KNOWN_PATH,
                "executed": False,
                "note": (
                    "Lance la commande depuis 'serve_cwd' (le dossier de l'app). Nécessite "
                    "l'extra a2a : uv add 'adk-toolkit-mcp[a2a]'."
                ),
            }
        )

    if not _a2a_extra_present():
        return err(_A2A_EXTRA_ABSENT)

    # uvicorn dans le venv courant + l'app importable (cwd = dossier de l'app).
    argv = [
        sys.executable,
        "-m",
        "uvicorn",
        "a2a_app:a2a_app",
        "--host",
        "localhost",
        "--port",
        str(port),
    ]
    key = adk_cli.make_key("a2a", str(app_dir), port)
    try:
        info = adk_cli.start_process(
            key, argv, cwd=str(app_dir), log_path=_log_path(path, app_name, port)
        )
    except adk_cli.ProcessAlreadyRunning as exc:
        return err(str(exc))

    return ok(
        {
            "file": file_path,
            "key": info["key"],
            "pid": info["pid"],
            "running": info["running"],
            "port": port,
            "url": f"http://localhost:{port}",
            "agent_card_url": f"http://localhost:{port}{_WELL_KNOWN_PATH}",
            "log_path": info["log_path"],
            "serve_command": serve_command,
            "executed": True,
        }
    )


# --------------------------------------------------------------------------- #
# agent_card — best-effort build de l'AgentCard du root_agent
# --------------------------------------------------------------------------- #
@a2a_server.tool
async def agent_card(path: str, app_name: str, port: int = 8001) -> dict[str, Any]:
    """Construit/inspecte l'``AgentCard`` du ``root_agent`` du projet (best-effort, gaté ``a2a``).

    Nécessite (a) l'extra ``a2a`` et (b) d'importer le ``root_agent`` du projet. Si l'extra est
    absent → ``err`` actionnable (pas de crash). Sinon, importe l'agent, construit la carte via
    ``AgentCardBuilder(agent=..., rpc_url="http://localhost:<port>/")`` (``build()`` est async) et
    renvoie ``card.model_dump(exclude_none=True)`` (name/description/url/skills/capabilities/...).

    ``port`` ne sert qu'à composer le ``rpc_url`` inscrit dans la carte (aucun serveur n'est lancé).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not _a2a_extra_present():
        return err(_A2A_EXTRA_ABSENT)

    try:
        root_agent = import_root_agent(path, app_name)
    except RootAgentImportError as exc:
        return err(str(exc))

    try:
        from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder

        builder = AgentCardBuilder(agent=root_agent, rpc_url=f"http://localhost:{port}/")
        card = await builder.build()
        card_dict = (
            card.model_dump(exclude_none=True) if hasattr(card, "model_dump") else dict(vars(card))
        )
    except Exception as exc:  # noqa: BLE001 - build best-effort, on remonte un err propre
        return err(f"Échec de construction de l'AgentCard : {exc}")

    return ok(
        {
            "app_name": app_name,
            "agent_card": card_dict,
            "well_known_path": _WELL_KNOWN_PATH,
        }
    )
