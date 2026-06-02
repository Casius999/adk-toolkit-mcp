"""`dev` domain: local development loop around the ``adk`` servers (P4a).

Manages the **long-running** dev servers (``adk web`` = UI+API, ``adk api_server`` = API) as
background processes via the **registry** of :mod:`adk_toolkit_mcp.adk_cli` (``Popen`` + log
file), and runs ``adk run`` as a non-interactive **one-shot**. None of these servers can be
launched via ``run_adk`` (which waits for the process to finish): ``web``/``api_server`` block
while serving → we start them detached and drive them (status/logs/stop).

Tools exposed under ``namespace="dev"`` → ``dev_<name>``. BARE names:
- ``web`` / ``api_server`` — start a managed server on the agents folder; return
  ``{key, pid, port, url, ...}``.
- ``run`` — one-shot ``adk run <agent_dir> <message>`` (the message is a POSITIONAL QUERY in ADK
  2.1.0, NOT a flag). Short timeout → never a hang. Without a message, returns guidance (the
  interactive mode would block).
- ``stop`` / ``status`` / ``logs`` — drive a started process, by its ``key``.

Each tool returns ``{ok, data, error}``. Cf. ``docs/adk-api-notes/deploy-dev.md`` (positional
AGENTS_DIR; real ``--host``/``--port``; ``adk run`` has a positional QUERY, not ``--input``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from .. import adk_cli
from ..envelope import err, ok

dev_server: FastMCP = FastMCP("dev")

#: Managed dev server kinds (``adk`` subcommands).
ServeKind = Literal["web", "api_server"]

#: Valid TCP port bounds.
_PORT_MIN = 1
_PORT_MAX = 65535

#: Default timeout (s) of a one-shot ``adk run`` (never a hang, even without creds).
_RUN_TIMEOUT = 120.0

#: Folder of the managed servers' log files (the app/agents-folder sidecar).
_LOG_DIR = ".adk_toolkit/logs"


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _agents_dir(path: str, app_name: str | None) -> str:
    """Resolve the positional AGENTS_DIR.

    - ``app_name`` provided → ``<path>/<app_name>`` (a single agent folder);
    - otherwise → ``<path>`` (an agents directory: each subfolder = an agent).
    """
    return str(Path(path) / app_name) if app_name else str(path)


def _require_dir(target: str) -> str | None:
    """Return an error message if ``target`` is not an existing directory, otherwise None."""
    if not Path(target).is_dir():
        return f"Directory not found: {target}. Scaffold the app first (project_create)."
    return None


def _validate_port(port: int) -> str | None:
    """Return an error message if ``port`` is out of the TCP bounds, otherwise None."""
    if not isinstance(port, int) or not (_PORT_MIN <= port <= _PORT_MAX):
        return f"Invalid port: {port!r}. Expected an integer in [{_PORT_MIN}, {_PORT_MAX}]."
    return None


def _build_serve_argv(
    kind: ServeKind, path: str, app_name: str | None, port: int, host: str
) -> tuple[list[str], str]:
    """Build the argv ``adk <kind> --host H --port P AGENTS_DIR``; return ``(argv, dir)``.

    ``--host`` and ``--port`` are real flags of ``web``/``api_server`` (cf. notes). AGENTS_DIR is
    the final positional.
    """
    agents_dir = _agents_dir(path, app_name)
    argv = [kind, "--host", host, "--port", str(port), agents_dir]
    return argv, agents_dir


def _build_run_argv(path: str, app_name: str, message: str) -> list[str]:
    """Build the argv ``adk run AGENT QUERY`` (message = POSITIONAL QUERY, not a flag)."""
    return ["run", _agents_dir(path, app_name), message]


def _log_path(path: str, app_name: str | None, kind: str, port: int) -> str:
    """Path of a managed server's log file (under the agents-folder sidecar)."""
    base = Path(_agents_dir(path, app_name))
    return str(base / _LOG_DIR / f"{kind}-{port}.log")


def _server_url(host: str, port: int) -> str:
    """Readable server URL (``0.0.0.0`` is displayed as ``127.0.0.1`` for local access)."""
    display_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    return f"http://{display_host}:{port}"


def _status_running(key: str) -> bool:
    """True if the ``key`` process is registered AND running (test/poll helper)."""
    return bool(adk_cli.process_status(key)["running"])


def _start_serve(
    kind: ServeKind, path: str, app_name: str | None, port: int, host: str
) -> dict[str, Any]:
    """Logic common to ``web``/``api_server``: validate, start the managed process, return the info.

    Starts ``adk <kind> ...`` detached (stdout/stderr → log file) via the adk_cli registry. An
    already-active key (live process) → ``err`` (no exception). Returns ``{key, pid, port, host,
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
# MCP tools — long-running servers
# --------------------------------------------------------------------------- #
@dev_server.tool(tags={"dev"})
async def web(
    path: str, app_name: str | None = None, port: int = 8000, host: str = "127.0.0.1"
) -> dict[str, Any]:
    """Start ``adk web`` (dev UI + API) as a managed process on the agents folder.

    If ``app_name`` is provided, serves that single agent folder (``<path>/<app_name>``);
    otherwise serves ``<path>`` as an agents directory. Returns ``{key, pid, port, url, ...}``.
    Then drive via ``dev_status`` / ``dev_logs`` / ``dev_stop``.

    NB: the ADK Web UI is intended for dev/test only (not for production).
    """
    return _start_serve("web", path, app_name, port, host)


@dev_server.tool(tags={"dev"})
async def api_server(
    path: str, app_name: str | None = None, port: int = 8000, host: str = "127.0.0.1"
) -> dict[str, Any]:
    """Start ``adk api_server`` (FastAPI API, no UI) as a managed process on the agents folder.

    Same ``app_name``/driving semantics as :func:`web`. Returns ``{key, pid, port, url, ...}``.
    The OpenAPI docs are served at ``<url>/docs`` once the server is ready.
    """
    return _start_serve("api_server", path, app_name, port, host)


@dev_server.tool(tags={"dev"})
async def run(path: str, app_name: str, message: str | None = None) -> dict[str, Any]:
    """One-shot ``adk run <agent_dir> "<message>"`` (non-interactive), with a short timeout.

    In ADK 2.1.0, ``adk run AGENT [QUERY]`` runs a single turn when a QUERY (the message) is
    provided; without a QUERY it enters INTERACTIVE mode (which would block). So:
    - ``message`` provided → runs the command (bounded timeout) and returns ``rc``/``stdout``/
      ``stderr`` in data (an environment without model creds returns a non-zero rc / an error in
      the output — NEVER a hang);
    - ``message`` absent → returns guidance (the interactive mode is not scriptable here).
    """
    dir_error = _require_dir(_agents_dir(path, app_name))
    if dir_error is not None:
        return err(dir_error)

    if message is None or not message.strip():
        return ok(
            {
                "executed": False,
                "guidance": (
                    "Provide 'message' for a non-interactive one-shot run "
                    '(adk run AGENT "<message>"). The interactive mode of adk run '
                    "would block and is not drivable via this tool."
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
# MCP tools — managed process control
# --------------------------------------------------------------------------- #
@dev_server.tool(tags={"dev"})
async def stop(key: str) -> dict[str, Any]:
    """Stop (process-tree termination) a managed server by its ``key``.

    Idempotent: an unknown key returns ``found=False`` (always ``ok``). On Windows, terminates the
    tree via ``taskkill /T``; elsewhere ``terminate()`` then ``kill()``.
    """
    if not key.strip():
        return err("key is empty.")
    result = adk_cli.stop_process(key)
    return ok(result)


@dev_server.tool(tags={"dev"})
async def status(key: str) -> dict[str, Any]:
    """Return a managed server's state: ``{found, running, pid, returncode, log_path, argv}``."""
    if not key.strip():
        return err("key is empty.")
    return ok(adk_cli.process_status(key))


@dev_server.tool(tags={"dev"})
async def logs(key: str, tail: int = 50) -> dict[str, Any]:
    """Return the last ``tail`` lines of a managed server's log.

    ``{found, lines, log_path}``. An unknown key → ``found=False`` + ``lines=[]``.
    """
    if not key.strip():
        return err("key is empty.")
    if tail < 0:
        return err("tail must be >= 0.")
    return ok(adk_cli.process_logs(key, tail=tail))
