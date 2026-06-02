"""Shared CLI plumbing for invoking the ``adk`` command (P4a).

The ``deploy`` and ``dev`` domains rely on this module to:

1. **Resolve how to invoke ``adk``** (:func:`adk_executable`) — we prefer the venv console script
   (``adk``/``adk.exe``), otherwise we fall back to ``[sys.executable, "-m",
   "google.adk.cli"]`` (a REAL verified module: ``google.adk.cli.__main__`` exists in 2.1.0).
2. **Run ``adk <args>`` synchronously** (:func:`run_adk`), capturing rc/stdout/stderr. Argument
   list (argv), NEVER ``shell=True``. Used for ``--help`` introspection and real deployments
   (only when the ``deploy`` domain receives ``execute=True``).
3. **List the flags actually available** for a subcommand (:func:`available_flags`) by parsing
   its ``--help`` — so the toolkit CANNOT emit a flag absent from this version of ADK (flags
   drift between versions).
4. **A process registry** for long-running dev servers (``adk web`` / ``adk api_server``):
   :func:`start_process` / :func:`process_status` / :func:`stop_process` / :func:`process_logs`.
   ``subprocess.Popen`` with stdout+stderr redirected to a log file; handles stored in a
   module-level dict keyed by a stable key (:func:`make_key`). ``stop`` terminates the process
   TREE (on Windows: ``CREATE_NEW_PROCESS_GROUP`` + ``taskkill /T``, otherwise ``terminate()``
   then ``kill()``).

No heavy dependency at load; ``google.adk`` is NOT imported here (we shell out to the CLI). Cf.
``docs/adk-api-notes/deploy-dev.md`` for the confirmed 2.1.0 flags.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess  # noqa: S404 - intentional execution of the adk CLI (argv, never shell=True)
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Regex for a long-flag token in a ``--help`` output (``--flag`` / ``--no-flag``).
_FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9_-]*")

#: Cache of flags per subcommand (key = tuple of subcommand tokens).
_FLAG_CACHE: dict[tuple[str, ...], set[str]] = {}

#: Default timeout (s) for an ``adk <subcommand> --help``.
_HELP_TIMEOUT = 60.0

#: True if running on Windows (specific handling of process-tree termination).
_IS_WINDOWS = os.name == "nt"


class ProcessAlreadyRunning(Exception):
    """Raised by :func:`start_process` if the key is already bound to a live process."""


# --------------------------------------------------------------------------- #
# adk executable resolution
# --------------------------------------------------------------------------- #
def _venv_script(name: str) -> str | None:
    """Return the path of a console script (``adk``/``adk.exe``) in the current venv, or None.

    We look next to ``sys.executable`` (``Scripts`` on Windows, otherwise the same ``bin``
    folder). Lets us prefer the active environment's ``adk`` without relying on the PATH.
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
    """Return the base argv to invoke ``adk`` (prefix to complete with the subcommand).

    Preference order:
    1. ``adk``/``adk.exe`` console script of the current venv (next to ``sys.executable``);
    2. ``adk`` found on the PATH (``shutil.which``);
    3. fallback ``[sys.executable, "-m", "google.adk.cli"]`` (a real verified module).
    """
    script = _venv_script("adk")
    if script is not None:
        return [script]
    found = shutil.which("adk")
    if found:
        return [found]
    return [sys.executable, "-m", "google.adk.cli"]


# --------------------------------------------------------------------------- #
# Synchronous adk execution
# --------------------------------------------------------------------------- #
def run_adk(
    args: list[str], cwd: str | None = None, timeout: float | None = None
) -> dict[str, Any]:
    """Run ``adk <args>`` synchronously and return ``{argv, rc, stdout, stderr}``.

    Passes an **argument list** (never ``shell=True``). Captures stdout/stderr as text. An
    exceeded timeout returns ``rc=-1`` with a message in ``stderr`` (never a blocking hang that
    propagates). A missing CLI raises ``FileNotFoundError`` (abnormal case; the caller decides).
    """
    argv = adk_executable() + list(args)
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, intentional execution of the adk CLI
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
            "stderr": f"adk exceeded the {timeout}s timeout: {' '.join(args)}",
        }
    return {
        "argv": argv,
        "rc": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


# --------------------------------------------------------------------------- #
# Available flags of a subcommand
# --------------------------------------------------------------------------- #
def _parse_flags(help_text: str) -> set[str]:
    """Extract all ``--flag`` (and ``--no-flag``) tokens from a ``--help`` output."""
    return set(_FLAG_RE.findall(help_text))


def available_flags(subcommand: list[str]) -> set[str]:
    """Return the set of ``--xxx`` flags exposed by ``adk <subcommand> --help``.

    Result cached per subcommand (the ``--help`` does not change within a process). If the help
    command fails (non-zero rc or empty output), returns an empty set (the caller treats the
    absence of flags as "cannot validate").
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
    """Clear the flag cache (useful in tests to force re-introspection)."""
    _FLAG_CACHE.clear()


# --------------------------------------------------------------------------- #
# Long-running process registry (dev servers)
# --------------------------------------------------------------------------- #
@dataclass
class _ManagedProcess:
    """Internal handle for a managed process: ``Popen`` + metadata + log file handle."""

    key: str
    popen: subprocess.Popen[bytes]
    argv: list[str]
    cwd: str | None
    log_path: str
    log_file: Any  # binary file object open for writing


#: Module-level registry of managed processes, protected by a lock.
_REGISTRY: dict[str, _ManagedProcess] = {}
_REGISTRY_LOCK = threading.Lock()


def make_key(kind: str, cwd: str, port: int | None) -> str:
    """Build a stable key ``"{kind}:{cwd}:{port}"`` to identify a managed process."""
    return f"{kind}:{cwd}:{port if port is not None else '-'}"


def _is_running(proc: _ManagedProcess) -> bool:
    """True if the underlying process is still running (``poll()`` returns None)."""
    return proc.popen.poll() is None


def start_process(key: str, args: list[str], cwd: str | None, log_path: str) -> dict[str, Any]:
    """Start ``args`` in the background, stdout+stderr redirected to ``log_path``.

    Stores the handle in the registry under ``key``. Raises :class:`ProcessAlreadyRunning` if the
    key is already bound to a LIVE process (a dead process under that key is replaced). On
    Windows, the process is launched in a new group (``CREATE_NEW_PROCESS_GROUP``) to allow
    reliable tree termination.

    Returns ``{key, pid, running, log_path, argv}``.
    """
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None and _is_running(existing):
            raise ProcessAlreadyRunning(f"A process is already active for key {key!r}.")
        if existing is not None:
            _close_log(existing)
            del _REGISTRY[key]

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "wb")  # noqa: SIM115 - closed via stop_process/_close_log

        creationflags = 0
        if _IS_WINDOWS:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        popen = subprocess.Popen(  # noqa: S603 - argv list, intentional execution
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
    """Return a managed process's state: ``{found, running, pid, returncode, log_path, argv}``."""
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
    """Return the last ``tail`` lines of a managed process's log file.

    ``{found, lines, log_path}``. An unknown key → ``found=False`` + ``lines=[]``. The file may
    be missing (process just launched) → empty lines without error.
    """
    with _REGISTRY_LOCK:
        proc = _REGISTRY.get(key)
        if proc is None:
            return {"found": False, "lines": [], "log_path": None}
        log_path = proc.log_path
    lines = _read_tail(log_path, tail)
    return {"found": True, "lines": lines, "log_path": log_path}


def stop_process(key: str, timeout: float = 10.0) -> dict[str, Any]:
    """Terminate the process tree bound to ``key`` and remove it from the registry.

    On Windows: ``taskkill /F /T /PID`` (kills the tree), with a fallback to
    ``terminate()``/``kill()``. Elsewhere: ``terminate()`` (SIGTERM) then ``kill()`` (SIGKILL) if
    not terminated within the timeout. Idempotent: an unknown key → ``found=False``.

    Returns ``{found, stopped, returncode}``.
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
        except subprocess.TimeoutExpired:  # pragma: no cover - extreme case
            pass
    _close_log(proc)
    stopped = proc.popen.poll() is not None
    return {"found": True, "stopped": stopped, "returncode": proc.popen.returncode}


def stop_all_processes() -> int:
    """Terminate ALL managed processes (safety net for the tests).

    Returns the number of processes actually stopped.
    """
    with _REGISTRY_LOCK:
        keys = list(_REGISTRY.keys())
    count = 0
    for key in keys:
        if stop_process(key)["found"]:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Internal termination / log helpers
# --------------------------------------------------------------------------- #
def _terminate_tree(proc: _ManagedProcess) -> None:
    """Request termination of the process tree (best-effort, without raising)."""
    popen = proc.popen
    if popen.poll() is not None:
        return
    if _IS_WINDOWS:
        # taskkill terminates the TREE (/T) forcefully (/F). Best-effort: we ignore its rc.
        try:
            subprocess.run(  # noqa: S603,S607 - fixed argv, no user input
                ["taskkill", "/F", "/T", "/PID", str(popen.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - rare fallback
            pass
        try:
            popen.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):  # pragma: no cover
            pass
        popen.terminate()
    else:
        popen.terminate()


def _close_log(proc: _ManagedProcess) -> None:
    """Close the log file handle (best-effort)."""
    try:
        if not proc.log_file.closed:
            proc.log_file.flush()
            proc.log_file.close()
    except OSError:  # pragma: no cover - best-effort close
        pass


def _read_tail(log_path: str, tail: int) -> list[str]:
    """Return the last ``tail`` lines (without newline) of ``log_path``; [] if absent."""
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - best-effort read
        return []
    lines = text.splitlines()
    if tail >= 0:
        return lines[-tail:] if tail else []
    return lines
