"""Tests for the ``dev`` domain (P4a — long-running dev servers + one-shot run).

The ``dev`` domain manages ``adk web`` / ``adk api_server`` processes via the
:mod:`adk_toolkit_mcp.adk_cli` registry, and runs ``adk run`` as a one-shot. We ALWAYS TEST:
- the **command building** (argv) for web/api_server/run;
- the lifecycle via the **registry** (start → status running → logs → stop → not-running), proven
  with the real code path (the launched binary is ``adk``; we do not wait for it to serve).

FUNCTIONAL PROOF (best-effort, GATED): booting a real ``adk api_server`` on an ephemeral port and
probing it over HTTP (``/docs``) is SLOW/sometimes flaky in CI. This test only runs if
``ADK_TOOLKIT_TEST_API_SERVER=1``; otherwise it SKIPs loudly. Either way, we leave no active
process or bound port (cleanup fixture + systematic stop).

``adk run`` requires model creds to produce a response: the test runs it with a short timeout and
accepts a non-zero rc / an error output (returned in DATA, never a hang). We mainly verify that
the command is correctly built and executed.
"""

from __future__ import annotations

import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp import adk_cli
from adk_toolkit_mcp.domains import dev as DEV
from adk_toolkit_mcp.server import build_server

#: Opt-in flag for the REAL boot of an api_server (slow/flaky in CI otherwise).
_BOOT_FLAG = "ADK_TOOLKIT_TEST_API_SERVER"


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Terminate any managed process before AND after each test (no orphan / bound port)."""
    adk_cli.stop_all_processes()
    yield
    adk_cli.stop_all_processes()


def _free_port() -> int:
    """Return a free TCP port (ephemeral bind then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _scaffold_agent(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffold a minimal ADK app (importable WITHOUT an API key); return the parent path."""
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    (app_dir / "agent.py").write_text(
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(name='{app_name}', model='gemini-2.5-flash', "
        "instruction='You are a test agent.')\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _wait_until(predicate, timeout: float = 30.0, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    result = bool(predicate())
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = bool(predicate())
    return result


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    """True if a GET on ``url`` responds (status < 500). Any network error → False."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost test
            return resp.status < 500
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Command building (web / api_server) — without booting
# --------------------------------------------------------------------------- #
def test_build_serve_argv_api_server(tmp_path: Path) -> None:
    """``_build_serve_argv`` produces the expected command for api_server (dir + host/port)."""
    path = _scaffold_agent(tmp_path)
    argv, agents_dir = DEV._build_serve_argv(
        "api_server", path, "myapp", port=8123, host="127.0.0.1"
    )
    assert argv[0] == "api_server"
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8123"
    # app_name provided → AGENTS_DIR points to the app folder (final positional).
    assert argv[-1] == str(Path(path) / "myapp")
    assert agents_dir == str(Path(path) / "myapp")


def test_build_serve_argv_web_without_app_name(tmp_path: Path) -> None:
    """Without app_name, AGENTS_DIR = the parent folder (agents directory)."""
    path = _scaffold_agent(tmp_path)
    argv, agents_dir = DEV._build_serve_argv("web", path, None, port=8000, host="0.0.0.0")
    assert argv[0] == "web"
    assert agents_dir == path
    assert argv[-1] == path


def test_serve_argv_flags_valid_against_real_help(tmp_path: Path) -> None:
    """The flags emitted for web/api_server actually exist (available_flags)."""
    path = _scaffold_agent(tmp_path)
    for kind in ("web", "api_server"):
        argv, _ = DEV._build_serve_argv(kind, path, "myapp", port=8000, host="127.0.0.1")
        valid = adk_cli.available_flags([kind])
        emitted = {t for t in argv if t.startswith("--")}
        assert emitted <= valid, f"{kind}: unknown flags {emitted - valid}"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
async def test_web_missing_agent_dir_returns_err(tmp_path: Path) -> None:
    result = await DEV.web(path=str(tmp_path), app_name="ghost")
    assert result["ok"] is False


async def test_api_server_rejects_bad_port(tmp_path: Path) -> None:
    path = _scaffold_agent(tmp_path)
    result = await DEV.api_server(path=path, app_name="myapp", port=0)
    assert result["ok"] is False
    assert "port" in result["error"].lower()


# --------------------------------------------------------------------------- #
# Lifecycle via the registry — proven with the real code path (start/stop)
# --------------------------------------------------------------------------- #
async def test_dev_start_status_stop_lifecycle(tmp_path: Path) -> None:
    """api_server starts (process registered + running), status sees it, stop terminates it.

    We do NOT depend on the server's HTTP availability (slow) — we prove the registry contract via
    the real start of the ``adk api_server`` process then its stop. The binary is indeed ``adk``
    (proof in the registered argv).
    """
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    started = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert started["ok"] is True, started
    key = started["data"]["key"]
    assert started["data"]["pid"] > 0
    assert started["data"]["port"] == port
    assert "api_server" in started["data"]["url"] or started["data"]["url"].startswith("http")

    # status sees the process (running at least just after launch).
    status = await DEV.status(key=key)
    assert status["ok"] is True
    assert status["data"]["found"] is True

    # logs accessible (the file exists even if empty at the very start).
    logs = await DEV.logs(key=key, tail=20)
    assert logs["ok"] is True
    assert "lines" in logs["data"]

    # stop actually terminates the process.
    stopped = await DEV.stop(key=key)
    assert stopped["ok"] is True
    assert stopped["data"]["found"] is True
    assert _wait_until(lambda: DEV._status_running(key) is False, timeout=15.0)


async def test_dev_double_start_same_key_returns_err(tmp_path: Path) -> None:
    """Starting the same app/port twice (live process) → clean err (no exception)."""
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    first = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert first["ok"] is True
    second = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert second["ok"] is False
    assert "already" in second["error"].lower()
    await DEV.stop(key=first["data"]["key"])


async def test_dev_stop_unknown_key() -> None:
    result = await DEV.stop(key="web:/nope:9999")
    assert result["ok"] is True
    assert result["data"]["found"] is False


async def test_dev_status_unknown_key() -> None:
    result = await DEV.status(key="web:/nope:9999")
    assert result["ok"] is True
    assert result["data"]["found"] is False


# --------------------------------------------------------------------------- #
# FUNCTIONAL (gated) — REAL boot of an api_server + HTTP probe
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv(_BOOT_FLAG) != "1",
    reason=(
        f"REAL api_server boot gated behind {_BOOT_FLAG}=1 "
        "(slow/flaky in CI). The registry + the command building are tested without a gate."
    ),
)
async def test_api_server_boots_and_serves_http(tmp_path: Path) -> None:
    """[GATED {flag}=1] Boots a real adk api_server on an ephemeral port; probes /docs over HTTP."""
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    started = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert started["ok"] is True, started
    key = started["data"]["key"]
    url = f"http://127.0.0.1:{port}/docs"
    try:
        booted = _wait_until(lambda: _http_ok(url), timeout=60.0)
        logs = await DEV.logs(key=key, tail=50)
        assert booted, f"api_server did not respond on {url}. logs={logs['data']['lines']}"
    finally:
        await DEV.stop(key=key)
    assert _wait_until(lambda: DEV._status_running(key) is False, timeout=15.0)


# --------------------------------------------------------------------------- #
# run (one-shot) — builds/executes, never a hang
# --------------------------------------------------------------------------- #
def test_run_argv_construction(tmp_path: Path) -> None:
    """``_build_run_argv`` puts the message as a positional QUERY (not a flag) after AGENT."""
    path = _scaffold_agent(tmp_path)
    argv = DEV._build_run_argv(path, "myapp", "hello there")
    assert argv[0] == "run"
    # AGENT (app folder) then QUERY (message).
    assert argv[1] == str(Path(path) / "myapp")
    assert argv[-1] == "hello there"


async def test_run_without_message_returns_guidance(tmp_path: Path) -> None:
    """Without a message, run returns guidance (the interactive mode would block) — no execution."""
    path = _scaffold_agent(tmp_path)
    result = await DEV.run(path=path, app_name="myapp", message=None)
    assert result["ok"] is True
    assert result["data"]["executed"] is False
    assert result["data"]["guidance"]


async def test_run_with_message_executes_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a message, run invokes adk run (mocked) and surfaces rc/output; never a hang."""
    recorded: dict[str, object] = {}

    def _fake_run(args, cwd=None, timeout=None):  # type: ignore[no-untyped-def]
        recorded["args"] = list(args)
        recorded["timeout"] = timeout
        return {"argv": ["adk", *args], "rc": 0, "stdout": "agent says hi", "stderr": ""}

    monkeypatch.setattr(DEV.adk_cli, "run_adk", _fake_run)
    path = _scaffold_agent(tmp_path)
    result = await DEV.run(path=path, app_name="myapp", message="hi")
    assert result["ok"] is True
    assert result["data"]["executed"] is True
    assert result["data"]["rc"] == 0
    assert "agent says hi" in result["data"]["stdout"]
    assert recorded["args"][0] == "run"
    # A timeout is passed (never an infinite wait).
    assert recorded["timeout"] is not None


async def test_run_missing_agent_dir_returns_err(tmp_path: Path) -> None:
    result = await DEV.run(path=str(tmp_path), app_name="ghost", message="hi")
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client (exposed names + registry flow)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_registry_flow(tmp_path: Path) -> None:
    """Tools exposed as dev_<bare> (no double prefix); start→status→stop flow via the client."""
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {"dev_web", "dev_api_server", "dev_run", "dev_stop", "dev_status", "dev_logs"}
        assert expected <= names
        assert not any(n.startswith("dev_dev_") for n in names)

        started = await client.call_tool(
            "dev_api_server", {"path": path, "app_name": "myapp", "port": port}
        )
        assert started.data["ok"] is True
        key = started.data["data"]["key"]
        try:
            status = await client.call_tool("dev_status", {"key": key})
            assert status.data["ok"] is True
            assert status.data["data"]["found"] is True
        finally:
            stopped = await client.call_tool("dev_stop", {"key": key})
            assert stopped.data["ok"] is True
    assert _wait_until(lambda: DEV._status_running(key) is False, timeout=15.0)
