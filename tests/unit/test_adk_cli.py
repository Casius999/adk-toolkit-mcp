"""Unit tests for ``adk_cli`` (P4a — shared CLI plumbing).

Covers:
- ``adk_executable``: non-empty resolution (venv ``adk[.exe]`` or fallback ``-m google.adk.cli``)
  and the forced fallback branch (emptied PATH).
- ``run_adk``: synchronous execution of a REAL ``--help`` (rc 0, non-empty stdout); NO shell.
- ``available_flags``: against a REAL ``adk deploy cloud_run --help`` returns a non-empty set
  including ``--project``/``--region``/``--service_name``/``--with_ui`` (confirmed 2.1.0 names);
  ``deploy gke`` includes ``--cluster_name``; cache (a second call does not re-launch the process).
- The **process registry**: ``start_process``/``process_status``/``process_logs``/
  ``stop_process`` proven with a TRIVIAL cross-platform process
  (``[sys.executable, "-c", "import time; time.sleep(30)"]``): starts (status running), writes a
  log, and ``stop`` actually terminates it (status not-running after). Systematic cleanup.

No real cloud deployment. These tests touch zero network (only a local ``--help``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from adk_toolkit_mcp import adk_cli

#: Portable trivial process: sleeps 30 s then exits (long enough to observe "running").
_SLEEP_ARGS = [sys.executable, "-c", "import time; time.sleep(30)"]
#: Trivial process that prints then exits fast (to prove log writing + the exit).
_PRINT_ARGS = [sys.executable, "-c", "print('hello-from-child'); import sys; sys.stdout.flush()"]


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Isolate each test: terminate any residual registry process before AND after."""
    adk_cli.stop_all_processes()
    yield
    adk_cli.stop_all_processes()


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll ``predicate`` until True or expiry; return the last boolean result."""
    deadline = time.monotonic() + timeout
    result = bool(predicate())
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = bool(predicate())
    return result


# --------------------------------------------------------------------------- #
# adk_executable / run_adk
# --------------------------------------------------------------------------- #
def test_adk_executable_non_empty() -> None:
    exe = adk_cli.adk_executable()
    assert isinstance(exe, list)
    assert len(exe) >= 1
    assert all(isinstance(part, str) and part for part in exe)


def test_adk_executable_fallback_when_no_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a findable ``adk`` script, we fall back to ``[python, -m, google.adk.cli]``."""
    monkeypatch.setattr(adk_cli.shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(adk_cli, "_venv_script", lambda *_a, **_k: None)
    exe = adk_cli.adk_executable()
    assert exe == [sys.executable, "-m", "google.adk.cli"]


def test_adk_executable_uses_path_when_no_venv_script(monkeypatch: pytest.MonkeyPatch) -> None:
    """No venv script but ``adk`` on the PATH → we use the PATH path."""
    monkeypatch.setattr(adk_cli, "_venv_script", lambda *_a, **_k: None)
    monkeypatch.setattr(adk_cli.shutil, "which", lambda *_a, **_k: "/usr/local/bin/adk")
    assert adk_cli.adk_executable() == ["/usr/local/bin/adk"]


def test_venv_script_returns_none_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_venv_script`` returns None when no candidate exists next to the executable."""
    monkeypatch.setattr(adk_cli.sys, "executable", str(tmp_path / "python.exe"))
    assert adk_cli._venv_script("adk") is None


def test_run_adk_help_real() -> None:
    """``run_adk(['--help'])`` runs (rc 0, non-empty stdout) without a shell."""
    result = adk_cli.run_adk(["--help"], timeout=120)
    assert result["rc"] == 0, result
    assert "Usage" in result["stdout"] or "Commands" in result["stdout"]
    assert isinstance(result["argv"], list)


def test_run_adk_reports_nonzero_rc() -> None:
    """An unknown subcommand → non-zero rc captured (never a propagating exception)."""
    result = adk_cli.run_adk(["this_is_not_a_real_subcommand"], timeout=120)
    assert result["rc"] != 0
    # stderr (Click) mentions the usage error.
    assert result["stderr"] or result["stdout"]


# --------------------------------------------------------------------------- #
# available_flags (against REAL --help)
# --------------------------------------------------------------------------- #
def test_available_flags_cloud_run_real() -> None:
    flags = adk_cli.available_flags(["deploy", "cloud_run"])
    assert isinstance(flags, set)
    assert flags, "available_flags must not be empty for deploy cloud_run"
    # 2.1.0 names confirmed by introspection.
    assert {"--project", "--region", "--service_name", "--app_name", "--with_ui"} <= flags
    assert "--trace_to_cloud" in flags


def test_available_flags_gke_has_cluster_name_real() -> None:
    flags = adk_cli.available_flags(["deploy", "gke"])
    assert "--cluster_name" in flags
    # Sanity: the hypothetical bare old name 'cluster' does not exist.
    assert "--cluster" not in flags


def test_available_flags_web_has_host_port_real() -> None:
    flags = adk_cli.available_flags(["web"])
    assert {"--host", "--port"} <= flags


def test_available_flags_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second call for the same subcommand does NOT re-launch the process."""
    adk_cli.clear_flag_cache()
    calls: list[list[str]] = []
    real_run = adk_cli.run_adk

    def _spy(args, cwd=None, timeout=None):  # type: ignore[no-untyped-def]
        calls.append(list(args))
        return real_run(args, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(adk_cli, "run_adk", _spy)
    first = adk_cli.available_flags(["deploy", "cloud_run"])
    second = adk_cli.available_flags(["deploy", "cloud_run"])
    assert first == second
    # A single underlying call despite two invocations.
    assert len(calls) == 1


def test_parse_flags_extracts_pairs() -> None:
    """``_parse_flags`` extracts both sides of a ``--flag / --no-flag`` pair."""
    sample = (
        "Options:\n"
        "  --project TEXT        Required.\n"
        "  --reload / --no-reload  Optional.\n"
        "  -v, --verbose         Enable.\n"
        "  --help                Show this.\n"
    )
    flags = adk_cli._parse_flags(sample)
    assert {"--project", "--reload", "--no-reload", "--verbose", "--help"} <= flags


# --------------------------------------------------------------------------- #
# Process registry (trivial cross-platform)
# --------------------------------------------------------------------------- #
def test_process_lifecycle_start_status_logs_stop(tmp_path: Path) -> None:
    """Start a sleep, see it running, write/re-read the log, then stop actually terminates it."""
    log_path = str(tmp_path / "proc.log")
    info = adk_cli.start_process("test:sleep", _SLEEP_ARGS, cwd=str(tmp_path), log_path=log_path)
    assert info["key"] == "test:sleep"
    assert isinstance(info["pid"], int) and info["pid"] > 0
    assert info["running"] is True

    status = adk_cli.process_status("test:sleep")
    assert status["found"] is True
    assert status["running"] is True
    assert status["pid"] == info["pid"]
    assert Path(log_path).exists()

    # stop effectively terminates the process.
    stopped = adk_cli.stop_process("test:sleep")
    assert stopped["found"] is True
    assert stopped["stopped"] is True

    # After stop: no longer running.
    assert _wait_until(lambda: adk_cli.process_status("test:sleep")["running"] is False)
    assert adk_cli.process_status("test:sleep")["running"] is False


def test_process_logs_capture_child_output(tmp_path: Path) -> None:
    """A child that prints to stdout has its output captured in the log file."""
    log_path = str(tmp_path / "out.log")
    adk_cli.start_process("test:print", _PRINT_ARGS, cwd=str(tmp_path), log_path=log_path)
    # The child exits fast; we wait for the file to contain the marker.
    assert _wait_until(lambda: "hello-from-child" in _safe_read(log_path), timeout=10.0)
    logs = adk_cli.process_logs("test:print", tail=10)
    assert logs["found"] is True
    assert any("hello-from-child" in line for line in logs["lines"])


def test_process_status_unknown_key() -> None:
    status = adk_cli.process_status("does-not-exist")
    assert status["found"] is False
    assert status["running"] is False


def test_stop_process_unknown_key() -> None:
    stopped = adk_cli.stop_process("nope")
    assert stopped["found"] is False
    assert stopped["stopped"] is False


def test_process_logs_unknown_key() -> None:
    logs = adk_cli.process_logs("nope", tail=5)
    assert logs["found"] is False
    assert logs["lines"] == []


def test_start_process_duplicate_key_rejected(tmp_path: Path) -> None:
    """Starting the same key twice (process still alive) is cleanly rejected."""
    log_path = str(tmp_path / "dup.log")
    adk_cli.start_process("test:dup", _SLEEP_ARGS, cwd=str(tmp_path), log_path=log_path)
    with pytest.raises(adk_cli.ProcessAlreadyRunning):
        adk_cli.start_process("test:dup", _SLEEP_ARGS, cwd=str(tmp_path), log_path=log_path)


def test_start_process_replaces_dead_key(tmp_path: Path) -> None:
    """A key bound to a DEAD process is replaced (no ProcessAlreadyRunning)."""
    log_path = str(tmp_path / "dead.log")
    adk_cli.start_process("test:dead", _PRINT_ARGS, cwd=str(tmp_path), log_path=log_path)
    # We wait for the first process (fast print) to have finished.
    assert _wait_until(lambda: adk_cli.process_status("test:dead")["running"] is False)
    # Re-starting under the same key must succeed (the dead one is replaced).
    info = adk_cli.start_process("test:dead", _SLEEP_ARGS, cwd=str(tmp_path), log_path=log_path)
    assert info["running"] is True
    adk_cli.stop_process("test:dead")


def test_read_tail_negative_returns_all(tmp_path: Path) -> None:
    """``_read_tail`` with a negative tail returns all the lines; tail=0 returns []."""
    log = tmp_path / "lines.log"
    log.write_text("a\nb\nc\n", encoding="utf-8")
    assert adk_cli._read_tail(str(log), -1) == ["a", "b", "c"]
    assert adk_cli._read_tail(str(log), 0) == []
    assert adk_cli._read_tail(str(log), 2) == ["b", "c"]


def test_read_tail_absent_file(tmp_path: Path) -> None:
    assert adk_cli._read_tail(str(tmp_path / "nope.log"), 5) == []


def test_make_key_is_stable() -> None:
    k1 = adk_cli.make_key("web", "/tmp/agents", 8000)
    k2 = adk_cli.make_key("web", "/tmp/agents", 8000)
    k3 = adk_cli.make_key("api_server", "/tmp/agents", 8000)
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("web:")


def _safe_read(path: str) -> str:
    """Read a log file if it exists, otherwise an empty string (the child may not have flushed)."""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
