"""Plomberie CLI partagée pour invoquer la commande ``adk`` (P4a).

Les domaines ``deploy`` et ``dev`` s'appuient sur ce module pour :

1. **Résoudre comment invoquer ``adk``** (:func:`adk_executable`) — on préfère le script console
   du venv (``adk``/``adk.exe``), sinon on retombe sur ``[sys.executable, "-m",
   "google.adk.cli"]`` (module RÉEL vérifié : ``google.adk.cli.__main__`` existe en 2.1.0).
2. **Exécuter ``adk <args>`` de façon synchrone** (:func:`run_adk`) en capturant rc/stdout/stderr.
   Liste d'arguments (argv), JAMAIS ``shell=True``. Sert au ``--help`` d'introspection et aux
   vrais déploiements (uniquement quand le domaine ``deploy`` reçoit ``execute=True``).
3. **Lister les flags réellement disponibles** d'une sous-commande (:func:`available_flags`) en
   parsant son ``--help`` — pour que le toolkit ne puisse PAS émettre un flag absent de cette
   version d'ADK (les flags dérivent entre versions).
4. **Un registre de process** pour les serveurs de dev longue durée (``adk web`` /
   ``adk api_server``) : :func:`start_process` / :func:`process_status` / :func:`stop_process` /
   :func:`process_logs`. ``subprocess.Popen`` avec stdout+stderr redirigés vers un fichier log ;
   handles stockés dans un dict module-level keyé par une clé stable (:func:`make_key`).
   ``stop`` termine l'ARBRE de process (sur Windows : ``CREATE_NEW_PROCESS_GROUP`` +
   ``taskkill /T``, sinon ``terminate()`` puis ``kill()``).

Aucune dépendance lourde au chargement ; ``google.adk`` n'est PAS importé ici (on shell vers la
CLI). Cf. ``docs/adk-api-notes/deploy-dev.md`` pour les flags 2.1.0 confirmés.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess  # noqa: S404 - exécution voulue de la CLI adk (argv, jamais shell=True)
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Regex d'un token de flag long dans une sortie ``--help`` (``--flag`` / ``--no-flag``).
_FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9_-]*")

#: Cache des flags par sous-commande (clé = tuple des tokens de sous-commande).
_FLAG_CACHE: dict[tuple[str, ...], set[str]] = {}

#: Délai par défaut (s) d'un ``adk <subcommand> --help``.
_HELP_TIMEOUT = 60.0

#: Vrai si on tourne sous Windows (gestion spécifique de la terminaison d'arbre de process).
_IS_WINDOWS = os.name == "nt"


class ProcessAlreadyRunning(Exception):
    """Levée par :func:`start_process` si la clé est déjà associée à un process vivant."""


# --------------------------------------------------------------------------- #
# Résolution de l'exécutable adk
# --------------------------------------------------------------------------- #
def _venv_script(name: str) -> str | None:
    """Renvoie le chemin d'un script console (``adk``/``adk.exe``) dans le venv courant, ou None.

    On regarde à côté de ``sys.executable`` (``Scripts`` sous Windows, sinon le même dossier
    ``bin``). Permet de préférer l'``adk`` de l'environnement actif sans dépendre du PATH.
    """
    exe_dir = Path(sys.executable).parent
    candidates = [exe_dir / name]
    if _IS_WINDOWS:
        candidates.append(exe_dir / f"{name}.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def adk_executable() -> list[str]:
    """Renvoie l'argv de base pour invoquer ``adk`` (préfixe à compléter avec la sous-commande).

    Ordre de préférence :
    1. script console ``adk``/``adk.exe`` du venv courant (à côté de ``sys.executable``) ;
    2. ``adk`` trouvé sur le PATH (``shutil.which``) ;
    3. fallback ``[sys.executable, "-m", "google.adk.cli"]`` (module réel vérifié).
    """
    script = _venv_script("adk")
    if script is not None:
        return [script]
    found = shutil.which("adk")
    if found:
        return [found]
    return [sys.executable, "-m", "google.adk.cli"]


# --------------------------------------------------------------------------- #
# Exécution synchrone d'adk
# --------------------------------------------------------------------------- #
def run_adk(
    args: list[str], cwd: str | None = None, timeout: float | None = None
) -> dict[str, Any]:
    """Exécute ``adk <args>`` de façon synchrone et renvoie ``{argv, rc, stdout, stderr}``.

    Passe une **liste d'arguments** (jamais ``shell=True``). Capture stdout/stderr en texte.
    Un timeout dépassé renvoie ``rc=-1`` avec un message dans ``stderr`` (jamais de blocage qui
    remonte). Une CLI introuvable lève ``FileNotFoundError`` (cas anormal ; l'appelant décide).
    """
    argv = adk_executable() + list(args)
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, exécution voulue de la CLI adk
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "rc": -1,
            "stdout": exc.stdout or "",
            "stderr": f"adk a dépassé le délai de {timeout}s : {' '.join(args)}",
        }
    return {
        "argv": argv,
        "rc": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


# --------------------------------------------------------------------------- #
# Flags disponibles d'une sous-commande
# --------------------------------------------------------------------------- #
def _parse_flags(help_text: str) -> set[str]:
    """Extrait tous les tokens ``--flag`` (et ``--no-flag``) d'une sortie ``--help``."""
    return set(_FLAG_RE.findall(help_text))


def available_flags(subcommand: list[str]) -> set[str]:
    """Renvoie l'ensemble des flags ``--xxx`` exposés par ``adk <subcommand> --help``.

    Résultat mis en cache par sous-commande (le ``--help`` ne change pas dans un process). Si la
    commande d'aide échoue (rc non nul ou sortie vide), renvoie un set vide (l'appelant traite
    l'absence de flags comme « impossible de valider »).
    """
    key = tuple(subcommand)
    cached = _FLAG_CACHE.get(key)
    if cached is not None:
        return set(cached)
    result = run_adk([*subcommand, "--help"], timeout=_HELP_TIMEOUT)
    flags = _parse_flags(result["stdout"]) if result["rc"] == 0 else set()
    _FLAG_CACHE[key] = flags
    return set(flags)


def clear_flag_cache() -> None:
    """Vide le cache des flags (utile en test pour forcer une réintrospection)."""
    _FLAG_CACHE.clear()


# --------------------------------------------------------------------------- #
# Registre de process longue durée (serveurs de dev)
# --------------------------------------------------------------------------- #
@dataclass
class _ManagedProcess:
    """Handle interne d'un process géré : ``Popen`` + métadonnées + handle de fichier log."""

    key: str
    popen: subprocess.Popen[bytes]
    argv: list[str]
    cwd: str | None
    log_path: str
    log_file: Any  # objet fichier binaire ouvert en écriture


#: Registre module-level des process gérés, protégé par un verrou.
_REGISTRY: dict[str, _ManagedProcess] = {}
_REGISTRY_LOCK = threading.Lock()


def make_key(kind: str, cwd: str, port: int | None) -> str:
    """Construit une clé stable ``"{kind}:{cwd}:{port}"`` pour identifier un process géré."""
    return f"{kind}:{cwd}:{port if port is not None else '-'}"


def _is_running(proc: _ManagedProcess) -> bool:
    """Vrai si le process sous-jacent est encore en cours (``poll()`` renvoie None)."""
    return proc.popen.poll() is None


def start_process(key: str, args: list[str], cwd: str | None, log_path: str) -> dict[str, Any]:
    """Démarre ``args`` en arrière-plan, stdout+stderr redirigés vers ``log_path``.

    Stocke le handle dans le registre sous ``key``. Lève :class:`ProcessAlreadyRunning` si la clé
    est déjà associée à un process VIVANT (un process mort sous cette clé est remplacé). Sur
    Windows, le process est lancé dans un nouveau groupe (``CREATE_NEW_PROCESS_GROUP``) pour
    permettre une terminaison d'arbre fiable.

    Renvoie ``{key, pid, running, log_path, argv}``.
    """
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and _is_running(existing):
            raise ProcessAlreadyRunning(f"Un process est déjà actif pour la clé {key!r}.")
        if existing is not None:
            _close_log(existing)
            del _REGISTRY[key]

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb")  # noqa: SIM115 - fermé via stop_process/_close_log

        creationflags = 0
        if _IS_WINDOWS:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        popen = subprocess.Popen(  # noqa: S603 - argv list, exécution voulue
            list(args),
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        managed = _ManagedProcess(
            key=key, popen=popen, argv=list(args), cwd=cwd, log_path=log_path, log_file=log_file
        )
        _REGISTRY[key] = managed
        return {
            "key": key,
            "pid": popen.pid,
            "running": _is_running(managed),
            "log_path": log_path,
            "argv": list(args),
        }


def process_status(key: str) -> dict[str, Any]:
    """Renvoie l'état d'un process géré : ``{found, running, pid, returncode, log_path, argv}``."""
    with _REGISTRY_LOCK:
        proc = _REGISTRY.get(key)
        if proc is None:
            return {"found": False, "running": False, "pid": None, "returncode": None}
        running = _is_running(proc)
        return {
            "found": True,
            "running": running,
            "pid": proc.popen.pid,
            "returncode": proc.popen.returncode,
            "log_path": proc.log_path,
            "argv": proc.argv,
        }


def process_logs(key: str, tail: int = 50) -> dict[str, Any]:
    """Renvoie les ``tail`` dernières lignes du fichier log d'un process géré.

    ``{found, lines, log_path}``. Une clé inconnue → ``found=False`` + ``lines=[]``. Le fichier
    peut être manquant (process à peine lancé) → lignes vides sans erreur.
    """
    with _REGISTRY_LOCK:
        proc = _REGISTRY.get(key)
        if proc is None:
            return {"found": False, "lines": [], "log_path": None}
        log_path = proc.log_path
    lines = _read_tail(log_path, tail)
    return {"found": True, "lines": lines, "log_path": log_path}


def stop_process(key: str, timeout: float = 10.0) -> dict[str, Any]:
    """Termine l'arbre de process associé à ``key`` et le retire du registre.

    Sur Windows : ``taskkill /F /T /PID`` (tue l'arbre), avec repli sur ``terminate()``/``kill()``.
    Ailleurs : ``terminate()`` (SIGTERM) puis ``kill()`` (SIGKILL) si non terminé dans le délai.
    Idempotent : une clé inconnue → ``found=False``.

    Renvoie ``{found, stopped, returncode}``.
    """
    with _REGISTRY_LOCK:
        proc = _REGISTRY.pop(key, None)
    if proc is None:
        return {"found": False, "stopped": False, "returncode": None}

    _terminate_tree(proc)
    try:
        proc.popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.popen.kill()
        try:
            proc.popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:  # pragma: no cover - cas extrême
            pass
    _close_log(proc)
    stopped = proc.popen.poll() is not None
    return {"found": True, "stopped": stopped, "returncode": proc.popen.returncode}


def stop_all_processes() -> int:
    """Termine TOUS les process gérés (filet de sécurité pour les tests).

    Renvoie le nombre de process effectivement arrêtés.
    """
    with _REGISTRY_LOCK:
        keys = list(_REGISTRY.keys())
    count = 0
    for key in keys:
        if stop_process(key)["found"]:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Helpers internes de terminaison / logs
# --------------------------------------------------------------------------- #
def _terminate_tree(proc: _ManagedProcess) -> None:
    """Demande la terminaison de l'arbre de process (best-effort, sans lever)."""
    popen = proc.popen
    if popen.poll() is not None:
        return
    if _IS_WINDOWS:
        # taskkill termine l'ARBRE (/T) de force (/F). Best-effort : on ignore son rc.
        try:
            subprocess.run(  # noqa: S603,S607 - argv fixe, pas d'entrée utilisateur
                ["taskkill", "/F", "/T", "/PID", str(popen.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - repli rare
            pass
        try:
            popen.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):  # pragma: no cover
            pass
        popen.terminate()
    else:
        popen.terminate()


def _close_log(proc: _ManagedProcess) -> None:
    """Ferme le handle de fichier log (best-effort)."""
    try:
        if not proc.log_file.closed:
            proc.log_file.flush()
            proc.log_file.close()
    except OSError:  # pragma: no cover - fermeture best-effort
        pass


def _read_tail(log_path: str, tail: int) -> list[str]:
    """Renvoie les ``tail`` dernières lignes (sans newline) de ``log_path`` ; [] si absent."""
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - lecture best-effort
        return []
    lines = text.splitlines()
    if tail >= 0:
        return lines[-tail:] if tail else []
    return lines
