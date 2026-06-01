"""Domaine `dev` : boucle de développement local autour des serveurs ``adk`` (P4a).

Gère les serveurs de dev **longue durée** (``adk web`` = UI+API, ``adk api_server`` = API) comme
des process en arrière-plan via le **registre** de :mod:`adk_toolkit_mcp.adk_cli` (``Popen`` +
fichier log), et lance ``adk run`` en **one-shot** non interactif. Aucun de ces serveurs ne peut
être lancé via ``run_adk`` (qui attend la fin du process) : ``web``/``api_server`` bloquent en
servant → on les démarre détachés et on les pilote (status/logs/stop).

Outils exposés sous ``namespace="dev"`` → ``dev_<nom>``. Noms BARE :
- ``web`` / ``api_server`` — démarrent un serveur géré sur le dossier d'agents ; renvoient
  ``{key, pid, port, url, ...}``.
- ``run`` — one-shot ``adk run <agent_dir> <message>`` (le message est un QUERY POSITIONNEL en
  ADK 2.1.0, PAS un flag). Court timeout → jamais de hang. Sans message, renvoie une guidance
  (le mode interactif bloquerait).
- ``stop`` / ``status`` / ``logs`` — pilotent un process démarré, par sa ``key``.

Chaque outil renvoie ``{ok, data, error}``. Cf. ``docs/adk-api-notes/deploy-dev.md`` (AGENTS_DIR
positionnel ; ``--host``/``--port`` réels ; ``adk run`` a un QUERY positionnel, pas ``--input``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from .. import adk_cli
from ..envelope import err, ok

dev_server: FastMCP = FastMCP("dev")

#: Genres de serveur de dev gérés (sous-commandes ``adk``).
ServeKind = Literal["web", "api_server"]

#: Bornes de port TCP valides.
_PORT_MIN = 1
_PORT_MAX = 65535

#: Timeout (s) par défaut d'un ``adk run`` one-shot (jamais de hang, même sans creds).
_RUN_TIMEOUT = 120.0

#: Dossier des fichiers logs des serveurs gérés (sidecar de l'app/dossier d'agents).
_LOG_DIR = ".adk_toolkit/logs"


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _agents_dir(path: str, app_name: str | None) -> str:
    """Résout le positionnel AGENTS_DIR.

    - ``app_name`` fourni → ``<path>/<app_name>`` (un dossier d'agent unique) ;
    - sinon → ``<path>`` (un répertoire d'agents : chaque sous-dossier = un agent).
    """
    return str(Path(path) / app_name) if app_name else str(path)


def _require_dir(target: str) -> str | None:
    """Renvoie un message d'erreur si ``target`` n'est pas un dossier existant, sinon None."""
    if not Path(target).is_dir():
        return f"Dossier introuvable : {target}. Scaffolde d'abord l'app (project_create)."
    return None


def _validate_port(port: int) -> str | None:
    """Renvoie un message d'erreur si ``port`` est hors bornes TCP, sinon None."""
    if not isinstance(port, int) or not (_PORT_MIN <= port <= _PORT_MAX):
        return f"port invalide : {port!r}. Attendu un entier dans [{_PORT_MIN}, {_PORT_MAX}]."
    return None


def _build_serve_argv(
    kind: ServeKind, path: str, app_name: str | None, port: int, host: str
) -> tuple[list[str], str]:
    """Construit l'argv ``adk <kind> --host H --port P AGENTS_DIR`` ; renvoie ``(argv, dir)``.

    ``--host`` et ``--port`` sont des flags réels de ``web``/``api_server`` (cf. notes). AGENTS_DIR
    est le positionnel final.
    """
    agents_dir = _agents_dir(path, app_name)
    argv = [kind, "--host", host, "--port", str(port), agents_dir]
    return argv, agents_dir


def _build_run_argv(path: str, app_name: str, message: str) -> list[str]:
    """Construit l'argv ``adk run AGENT QUERY`` (message = QUERY POSITIONNEL, pas un flag)."""
    return ["run", _agents_dir(path, app_name), message]


def _log_path(path: str, app_name: str | None, kind: str, port: int) -> str:
    """Chemin du fichier log d'un serveur géré (sous le sidecar du dossier d'agents)."""
    base = Path(_agents_dir(path, app_name))
    return str(base / _LOG_DIR / f"{kind}-{port}.log")


def _server_url(host: str, port: int) -> str:
    """URL lisible du serveur (``0.0.0.0`` est affiché comme ``127.0.0.1`` pour un accès local)."""
    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    return f"http://{display_host}:{port}"


def _status_running(key: str) -> bool:
    """True si le process ``key`` est enregistré ET en cours (helper de test/poll)."""
    return bool(adk_cli.process_status(key)["running"])


def _start_serve(
    kind: ServeKind, path: str, app_name: str | None, port: int, host: str
) -> dict[str, Any]:
    """Logique commune à ``web``/``api_server`` : valide, démarre le process géré, renvoie l'info.

    Démarre ``adk <kind> ...`` détaché (stdout/stderr → fichier log) via le registre adk_cli. Une
    clé déjà active (process vivant) → ``err`` (pas d'exception). Renvoie ``{key, pid, port, host,
    url, agents_dir, log_path, argv}``.
    """
    port_error = _validate_port(port)
    if port_error is not None:
        return err(port_error)

    argv, agents_dir = _build_serve_argv(kind, path, app_name, port, host)
    dir_error = _require_dir(agents_dir)
    if dir_error is not None:
        return err(dir_error)

    full_argv = adk_cli.adk_executable() + argv
    key = adk_cli.make_key(kind, agents_dir, port)
    log_path = _log_path(path, app_name, kind, port)

    try:
        info = adk_cli.start_process(key, full_argv, cwd=path, log_path=log_path)
    except adk_cli.ProcessAlreadyRunning as exc:
        return err(str(exc))

    return ok(
        {
            "kind": kind,
            "key": info["key"],
            "pid": info["pid"],
            "running": info["running"],
            "port": port,
            "host": host,
            "url": _server_url(host, port),
            "agents_dir": agents_dir,
            "log_path": info["log_path"],
            "argv": argv,
        }
    )


# --------------------------------------------------------------------------- #
# Outils MCP — serveurs longue durée
# --------------------------------------------------------------------------- #
@dev_server.tool(tags={"dev"})
async def web(
    path: str, app_name: str | None = None, port: int = 8000, host: str = "127.0.0.1"
) -> dict[str, Any]:
    """Démarre ``adk web`` (UI de dev + API) comme process géré sur le dossier d'agents.

    Si ``app_name`` est fourni, sert ce dossier d'agent unique (``<path>/<app_name>``) ; sinon
    sert ``<path>`` comme répertoire d'agents. Renvoie ``{key, pid, port, url, ...}``. Pilote
    ensuite via ``dev_status`` / ``dev_logs`` / ``dev_stop``.

    NB : le Web UI ADK est destiné au dev/test uniquement (pas à la production).
    """
    return _start_serve("web", path, app_name, port, host)


@dev_server.tool(tags={"dev"})
async def api_server(
    path: str, app_name: str | None = None, port: int = 8000, host: str = "127.0.0.1"
) -> dict[str, Any]:
    """Démarre ``adk api_server`` (API FastAPI, sans UI) comme process géré sur le dossier d'agents.

    Mêmes sémantiques d'``app_name``/pilotage que :func:`web`. Renvoie ``{key, pid, port, url,
    ...}``. La doc OpenAPI est servie sur ``<url>/docs`` une fois le serveur prêt.
    """
    return _start_serve("api_server", path, app_name, port, host)


@dev_server.tool(tags={"dev"})
async def run(path: str, app_name: str, message: str | None = None) -> dict[str, Any]:
    """One-shot ``adk run <agent_dir> "<message>"`` (non interactif), avec court timeout.

    En ADK 2.1.0, ``adk run AGENT [QUERY]`` exécute un tour unique quand un QUERY (le message) est
    fourni ; sans QUERY il entre en mode INTERACTIF (qui bloquerait). Donc :
    - ``message`` fourni → exécute la commande (timeout borné) et renvoie ``rc``/``stdout``/
      ``stderr`` en données (un environnement sans creds modèle renvoie un rc non nul / une erreur
      dans la sortie — JAMAIS un hang) ;
    - ``message`` absent → renvoie une guidance (le mode interactif n'est pas scriptable ici).
    """
    dir_error = _require_dir(_agents_dir(path, app_name))
    if dir_error is not None:
        return err(dir_error)

    if message is None or not message.strip():
        return ok(
            {
                "executed": False,
                "guidance": (
                    "Fournis 'message' pour un run one-shot non interactif "
                    '(adk run AGENT "<message>"). Le mode interactif d\'adk run '
                    "bloquerait et n'est pas pilotable via cet outil."
                ),
            }
        )

    argv = _build_run_argv(path, app_name, message)
    result = adk_cli.run_adk(argv, cwd=path, timeout=_RUN_TIMEOUT)
    return ok(
        {
            "executed": True,
            "argv": argv,
            "rc": result.get("rc"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
    )


# --------------------------------------------------------------------------- #
# Outils MCP — pilotage des process gérés
# --------------------------------------------------------------------------- #
@dev_server.tool(tags={"dev"})
async def stop(key: str) -> dict[str, Any]:
    """Arrête (terminaison d'arbre de process) un serveur géré par sa ``key``.

    Idempotent : une clé inconnue renvoie ``found=False`` (toujours ``ok``). Sur Windows, termine
    l'arbre via ``taskkill /T`` ; ailleurs ``terminate()`` puis ``kill()``.
    """
    if not key.strip():
        return err("key est vide.")
    result = adk_cli.stop_process(key)
    return ok(result)


@dev_server.tool(tags={"dev"})
async def status(key: str) -> dict[str, Any]:
    """Renvoie l'état d'un serveur géré : ``{found, running, pid, returncode, log_path, argv}``."""
    if not key.strip():
        return err("key est vide.")
    return ok(adk_cli.process_status(key))


@dev_server.tool(tags={"dev"})
async def logs(key: str, tail: int = 50) -> dict[str, Any]:
    """Renvoie les ``tail`` dernières lignes du log d'un serveur géré.

    ``{found, lines, log_path}``. Une clé inconnue → ``found=False`` + ``lines=[]``.
    """
    if not key.strip():
        return err("key est vide.")
    if tail < 0:
        return err("tail doit être >= 0.")
    return ok(adk_cli.process_logs(key, tail=tail))
