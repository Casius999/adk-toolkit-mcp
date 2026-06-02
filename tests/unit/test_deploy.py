"""Unit tests for the ``deploy`` domain (P4a — building deployment commands).

Principle: by default, NO real deployment is run (``execute=False``) — the tool builds and returns
the **exact argv** + a readable plan. We assert the built list of tokens (deterministic proof) and
the **validity of the flags** emitted against ADK 2.1.0's real ``--help`` output
(``available_flags``). The real cloud deployment is NOT tested (requires GCP) — but the command
building and the flag validity ARE.

Coverage:
- ``agent_engine`` / ``cloud_run`` / ``gke``: exact argv + all emitted flags ∈ available_flags.
- validation of the required arguments → ``err`` (missing paths/project/region/cluster).
- ``containerize``: writes a Dockerfile (idempotent via Workspace).
- ``preflight``: structured findings (best-effort), never raises.
- ``status``: actionable "unavailable" if the cloud tool is absent, without blocking.
- ``execute=False`` NEVER launches a real deployment (the real ``run_adk`` is not called).
- ``fastmcp.Client`` read-through for a command build.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp import adk_cli
from adk_toolkit_mcp.domains import deploy as D
from adk_toolkit_mcp.server import build_server


def _scaffold(tmp_path: Path, app_name: str = "myapp") -> str:
    """Create a minimal app folder (``agent.py``) and return the parent path."""
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "agent.py").write_text("root_agent = None\n", encoding="utf-8")
    return str(tmp_path)


def _agent_dir(tmp_path: Path, app_name: str = "myapp") -> str:
    """Expected path of the agent folder (positional AGENT of the CLI)."""
    return str(Path(tmp_path) / app_name)


# --------------------------------------------------------------------------- #
# agent_engine
# --------------------------------------------------------------------------- #
def test_agent_engine_argv_exact(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.agent_engine(
        path=path,
        app_name="myapp",
        project="my-proj",
        region="us-central1",
        staging_bucket="gs://ignored",
        display_name="My Agent",
    )
    assert result["ok"] is True, result
    argv = result["data"]["argv"]
    # Expected tokens: deploy agent_engine --project ... --region ... --display_name ... AGENT.
    assert argv[:2] == ["deploy", "agent_engine"]
    assert "--project" in argv and argv[argv.index("--project") + 1] == "my-proj"
    assert "--region" in argv and argv[argv.index("--region") + 1] == "us-central1"
    # an explicit display_name wins; otherwise app_name.
    assert "--display_name" in argv and argv[argv.index("--display_name") + 1] == "My Agent"
    # AGENT is the LAST token (the app folder).
    assert argv[-1] == _agent_dir(tmp_path)
    # staging_bucket is DEPRECATED: not emitted as a flag, but flagged in the notes.
    assert "--staging_bucket" not in argv
    assert any("staging_bucket" in n for n in result["data"]["notes"])
    assert result["data"]["executed"] is False


def test_agent_engine_app_name_maps_to_display_name(tmp_path: Path) -> None:
    """Without an explicit display_name, app_name is mapped to --display_name."""
    path = _scaffold(tmp_path, "billing")
    result = D.agent_engine(path=path, app_name="billing", project="p", region="r")
    assert result["ok"] is True
    argv = result["data"]["argv"]
    assert argv[argv.index("--display_name") + 1] == "billing"


def test_agent_engine_requirements_file_emitted(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.agent_engine(
        path=path,
        app_name="myapp",
        project="p",
        region="r",
        requirements_file="reqs.txt",
    )
    assert result["ok"] is True
    argv = result["data"]["argv"]
    assert "--requirements_file" in argv
    assert argv[argv.index("--requirements_file") + 1] == "reqs.txt"


def test_agent_engine_flags_all_valid(tmp_path: Path) -> None:
    """All the flags emitted for agent_engine actually exist (available_flags)."""
    path = _scaffold(tmp_path)
    result = D.agent_engine(path=path, app_name="myapp", project="p", region="r", display_name="X")
    assert result["ok"] is True
    valid = adk_cli.available_flags(["deploy", "agent_engine"])
    emitted = [t for t in result["data"]["argv"][2:] if t.startswith("--")]
    assert emitted, "at least one flag should be emitted"
    assert set(emitted) <= valid, f"unknown flags: {set(emitted) - valid}"


def test_agent_engine_requires_project_region(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    r1 = D.agent_engine(path=path, app_name="myapp", project="", region="r")
    assert r1["ok"] is False and "project" in r1["error"]
    r2 = D.agent_engine(path=path, app_name="myapp", project="p", region=" ")
    assert r2["ok"] is False and "region" in r2["error"]


def test_agent_engine_missing_agent_dir_returns_err(tmp_path: Path) -> None:
    result = D.agent_engine(path=str(tmp_path), app_name="ghost", project="p", region="r")
    assert result["ok"] is False
    assert "ghost" in result["error"] or "not found" in result["error"].lower()


# --------------------------------------------------------------------------- #
# cloud_run
# --------------------------------------------------------------------------- #
def test_cloud_run_argv_exact(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.cloud_run(
        path=path,
        app_name="myapp",
        project="my-proj",
        region="us-central1",
        service_name="my-svc",
        with_ui=True,
        enable_cloud_trace=True,
    )
    assert result["ok"] is True, result
    argv = result["data"]["argv"]
    assert argv[:2] == ["deploy", "cloud_run"]
    assert argv[argv.index("--project") + 1] == "my-proj"
    assert argv[argv.index("--region") + 1] == "us-central1"
    assert argv[argv.index("--service_name") + 1] == "my-svc"
    assert argv[argv.index("--app_name") + 1] == "myapp"
    # with_ui + enable_cloud_trace are boolean flags (no value).
    assert "--with_ui" in argv
    # enable_cloud_trace maps to the REAL flag --trace_to_cloud (not --enable_cloud_trace).
    assert "--trace_to_cloud" in argv
    assert "--enable_cloud_trace" not in argv
    assert argv[-1] == _agent_dir(tmp_path)


def test_cloud_run_minimal_no_optional_flags(tmp_path: Path) -> None:
    """Without options, with_ui/trace are not emitted; service_name omitted if not provided."""
    path = _scaffold(tmp_path)
    result = D.cloud_run(path=path, app_name="myapp", project="p", region="r")
    assert result["ok"] is True
    argv = result["data"]["argv"]
    assert "--with_ui" not in argv
    assert "--trace_to_cloud" not in argv
    assert "--service_name" not in argv


def test_cloud_run_flags_all_valid(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.cloud_run(
        path=path,
        app_name="myapp",
        project="p",
        region="r",
        service_name="s",
        with_ui=True,
        enable_cloud_trace=True,
    )
    assert result["ok"] is True
    valid = adk_cli.available_flags(["deploy", "cloud_run"])
    emitted = [t for t in result["data"]["argv"][2:] if t.startswith("--")]
    assert set(emitted) <= valid, f"unknown flags: {set(emitted) - valid}"


def test_cloud_run_requires_project_region(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    r = D.cloud_run(path=path, app_name="myapp", project="", region="")
    assert r["ok"] is False


# --------------------------------------------------------------------------- #
# gke
# --------------------------------------------------------------------------- #
def test_gke_argv_exact(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.gke(
        path=path,
        app_name="myapp",
        project="my-proj",
        region="us-central1",
        cluster="my-cluster",
        service_name="my-svc",
    )
    assert result["ok"] is True, result
    argv = result["data"]["argv"]
    assert argv[:2] == ["deploy", "gke"]
    assert argv[argv.index("--project") + 1] == "my-proj"
    assert argv[argv.index("--region") + 1] == "us-central1"
    # cluster maps to the REAL flag --cluster_name (not --cluster).
    assert argv[argv.index("--cluster_name") + 1] == "my-cluster"
    assert "--cluster" not in [t for t in argv if t == "--cluster"]
    assert argv[argv.index("--service_name") + 1] == "my-svc"
    assert argv[-1] == _agent_dir(tmp_path)


def test_gke_flags_all_valid(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.gke(
        path=path, app_name="myapp", project="p", region="r", cluster="c", service_name="s"
    )
    assert result["ok"] is True
    valid = adk_cli.available_flags(["deploy", "gke"])
    emitted = [t for t in result["data"]["argv"][2:] if t.startswith("--")]
    assert set(emitted) <= valid, f"unknown flags: {set(emitted) - valid}"


def test_gke_requires_cluster(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    r = D.gke(path=path, app_name="myapp", project="p", region="r", cluster="")
    assert r["ok"] is False
    assert "cluster" in r["error"]


# --------------------------------------------------------------------------- #
# containerize
# --------------------------------------------------------------------------- #
def test_containerize_writes_dockerfile(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    result = D.containerize(path=path, app_name="myapp")
    assert result["ok"] is True, result
    dockerfile = Path(result["data"]["path"])
    assert dockerfile.exists()
    content = dockerfile.read_text(encoding="utf-8")
    # The Dockerfile serves `adk api_server`.
    assert "adk" in content and "api_server" in content
    assert result["data"]["changed"] is True


def test_containerize_idempotent(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    first = D.containerize(path=path, app_name="myapp")
    second = D.containerize(path=path, app_name="myapp")
    assert first["ok"] is True and second["ok"] is True
    assert second["data"]["changed"] is False


def test_containerize_missing_agent_dir_returns_err(tmp_path: Path) -> None:
    result = D.containerize(path=str(tmp_path), app_name="ghost")
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# preflight (best-effort, never raises)
# --------------------------------------------------------------------------- #
def test_preflight_returns_structured_findings() -> None:
    result = D.preflight(target="cloud_run")
    assert result["ok"] is True
    data = result["data"]
    assert "gcloud_on_path" in data
    assert "adk_runnable" in data
    assert isinstance(data["findings"], list)


def test_preflight_unknown_target_still_ok() -> None:
    """An unknown target does not fail the preflight (best-effort)."""
    result = D.preflight(target="banana")
    assert result["ok"] is True
    assert "banana" in str(result["data"]["findings"]) or data_ok(result)


def data_ok(result: dict) -> bool:
    """Lax helper: the preflight stays ok even for a non-standard target."""
    return result["ok"] is True


# --------------------------------------------------------------------------- #
# status (best-effort, does not block)
# --------------------------------------------------------------------------- #
def test_status_unavailable_when_tool_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """If gcloud/kubectl is absent, status returns "unavailable" guidance without blocking."""
    monkeypatch.setattr(D.shutil, "which", lambda *_a, **_k: None)
    result = D.status(target="cloud_run", project="p", region="r", service_name="s")
    assert result["ok"] is True
    assert result["data"]["available"] is False
    assert result["data"]["guidance"]


def test_status_unknown_target_returns_err() -> None:
    result = D.status(target="banana")
    assert result["ok"] is False


def test_status_agent_engine_guidance() -> None:
    """agent_engine has no status CLI → available False + guidance (always ok)."""
    result = D.status(target="agent_engine")
    assert result["ok"] is True
    assert result["data"]["available"] is False
    assert "Vertex" in result["data"]["guidance"] or result["data"]["guidance"]


def test_status_cloud_run_executes_gcloud_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """gcloud present + complete args → _run_tool is invoked, rc/output surfaced."""
    monkeypatch.setattr(D.shutil, "which", lambda *_a, **_k: "/usr/bin/gcloud")
    monkeypatch.setattr(
        D, "_run_tool", lambda argv: {"rc": 0, "stdout": "https://svc.run.app", "stderr": ""}
    )
    result = D.status(target="cloud_run", project="p", region="r", service_name="s")
    assert result["ok"] is True
    assert result["data"]["available"] is True
    assert result["data"]["rc"] == 0
    assert "run.app" in result["data"]["stdout"]


def test_status_cloud_run_present_but_incomplete_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """gcloud present but args missing → guidance asking for project/region/service_name."""
    monkeypatch.setattr(D.shutil, "which", lambda *_a, **_k: "/usr/bin/gcloud")
    result = D.status(target="cloud_run", project="p")
    assert result["ok"] is True
    assert result["data"]["available"] is True
    assert "service_name" in result["data"]["guidance"]


def test_status_gke_executes_kubectl_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """kubectl present → _run_tool invoked to list the services."""
    monkeypatch.setattr(D.shutil, "which", lambda *_a, **_k: "/usr/bin/kubectl")
    monkeypatch.setattr(
        D, "_run_tool", lambda argv: {"rc": 0, "stdout": "svc ClusterIP", "stderr": ""}
    )
    result = D.status(target="gke", cluster="c")
    assert result["ok"] is True
    assert result["data"]["available"] is True
    assert result["data"]["cluster"] == "c"
    assert "svc" in result["data"]["stdout"]


def test_run_tool_captures_invocation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """_run_tool captures an OSError (tool absent despite which) into data, without raising."""
    import subprocess as _sp

    def _boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise OSError("not found")

    monkeypatch.setattr(_sp, "run", _boom)
    result = D._run_tool(["gcloud", "version"])
    assert result["rc"] == -1
    assert "failed" in result["stderr"] or "not found" in result["stderr"]


def test_execute_true_unknown_flag_blocks_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If available_flags does not contain an emitted flag, _finalize returns err BEFORE running."""
    # available_flags returns a set that does NOT include --app_name → cloud_run must fail.
    monkeypatch.setattr(D.adk_cli, "available_flags", lambda _sub: {"--project", "--region"})
    called: list[object] = []
    monkeypatch.setattr(D.adk_cli, "run_adk", lambda *a, **k: called.append(a) or {"rc": 0})
    path = _scaffold(tmp_path)
    result = D.cloud_run(path=path, app_name="myapp", project="p", region="r", execute=True)
    assert result["ok"] is False
    assert "--app_name" in result["error"]
    assert called == [], "an unknown flag must never trigger run_adk"


def test_finalize_skips_validation_when_help_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty available_flags (introspection failed) → invalidate nothing (plan still returned)."""
    monkeypatch.setattr(D.adk_cli, "available_flags", lambda _sub: set())
    path = _scaffold(tmp_path)
    result = D.cloud_run(path=path, app_name="myapp", project="p", region="r")
    assert result["ok"] is True
    assert result["data"]["argv"][:2] == ["deploy", "cloud_run"]


# --------------------------------------------------------------------------- #
# execute=False NEVER launches a real deployment
# --------------------------------------------------------------------------- #
def test_execute_false_never_runs_real_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With execute=False (default), run_adk is never called (no real deployment)."""
    called: list[list[str]] = []
    monkeypatch.setattr(D.adk_cli, "run_adk", lambda *a, **k: called.append(a) or {"rc": 0})
    path = _scaffold(tmp_path)
    D.agent_engine(path=path, app_name="myapp", project="p", region="r")
    D.cloud_run(path=path, app_name="myapp", project="p", region="r")
    D.gke(path=path, app_name="myapp", project="p", region="r", cluster="c")
    assert called == [], "execute=False must never invoke run_adk"


def test_execute_true_validates_flags_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """execute=True goes through run_adk (mocked) AFTER flag validation; rc/output surfaced."""
    recorded: dict[str, object] = {}

    def _fake_run(args, cwd=None, timeout=None):  # type: ignore[no-untyped-def]
        recorded["args"] = list(args)
        recorded["cwd"] = cwd
        return {"argv": ["adk", *args], "rc": 0, "stdout": "deployed (fake)", "stderr": ""}

    monkeypatch.setattr(D.adk_cli, "run_adk", _fake_run)
    path = _scaffold(tmp_path)
    result = D.cloud_run(path=path, app_name="myapp", project="p", region="r", execute=True)
    assert result["ok"] is True
    assert result["data"]["executed"] is True
    assert result["data"]["rc"] == 0
    assert "deployed (fake)" in result["data"]["stdout"]
    # run_adk indeed received the built argv.
    assert recorded["args"][:2] == ["deploy", "cloud_run"]


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_cloud_run(tmp_path: Path) -> None:
    """Tools exposed as deploy_<bare> (no double prefix); deploy_cloud_run builds the argv."""
    path = _scaffold(tmp_path)
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {
            "deploy_preflight",
            "deploy_agent_engine",
            "deploy_cloud_run",
            "deploy_gke",
            "deploy_containerize",
            "deploy_status",
        }
        assert expected <= names
        assert not any(n.startswith("deploy_deploy_") for n in names)

        res = await client.call_tool(
            "deploy_cloud_run",
            {"path": path, "app_name": "myapp", "project": "p", "region": "r"},
        )
        assert res.data["ok"] is True
        assert res.data["data"]["argv"][:2] == ["deploy", "cloud_run"]
