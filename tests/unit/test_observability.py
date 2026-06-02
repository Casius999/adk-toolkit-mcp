"""Tests for the `observability` domain (P4c) — OpenTelemetry, Cloud Trace, third-party, trace_view.

Covers:
- ``enable_otel``: generates an ast-valid ``otel_setup.py`` (console + otlp); endpoint validation.
  The console setup is also EXECUTED (actually installs a global TracerProvider).
- ``cloud_trace``: returns the real ``--trace_to_cloud`` flag + references the deploy/dev tool.
- ``third_party``: OTLP env variables + a snippet for phoenix/arize/weave/signoz/otlp.
- ``trace_view``: DELEGATES to ``dev_web`` (process registry); tested without a real boot (the
  delegation to an absent app returns ``dev.web``'s ``err``). A real boot stays behind ``dev``'s
  env flag.
- Read via an in-memory ``fastmcp.Client`` (exposed names + ``observability_cloud_trace`` call).
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from adk_toolkit_mcp import adk_cli
from adk_toolkit_mcp.domains import observability as OBS
from adk_toolkit_mcp.domains import observability_setup
from adk_toolkit_mcp.server import build_server


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Terminate any managed process before AND after each test (no orphan / bound port)."""
    adk_cli.stop_all_processes()
    yield
    adk_cli.stop_all_processes()


def _scaffold_app(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffold a minimal ADK app (enough: agent.py present)."""
    from adk_toolkit_mcp.domains.project import create as project_create

    path = str(tmp_path)
    assert project_create(path=path, app_name=app_name)["ok"]
    return path


# --------------------------------------------------------------------------- #
# enable_otel
# --------------------------------------------------------------------------- #
def test_enable_otel_console_generates_ast_valid(tmp_path: Path) -> None:
    """enable_otel console writes an ast-valid otel_setup.py defining setup_otel()."""
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="console")
    assert result["ok"], result.get("error")
    otel_file = Path(result["data"]["otel_setup"])
    assert otel_file.is_file()
    src = otel_file.read_text(encoding="utf-8")
    ast.parse(src)
    assert "def setup_otel()" in src
    assert "ConsoleSpanExporter" in src
    assert "trace.set_tracer_provider(provider)" in src


def test_enable_otel_console_setup_executes(tmp_path: Path) -> None:
    """The generated console setup EXECUTES and actually installs a global TracerProvider."""
    import opentelemetry.trace as ot_trace

    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="console")
    module = _load_module(Path(result["data"]["otel_setup"]), "otel_console_test")
    provider = module.setup_otel()
    try:
        assert type(provider).__name__ == "TracerProvider"
        assert ot_trace.get_tracer_provider() is provider
    finally:
        # Stop the BatchSpanProcessor (background thread) to avoid a flush to a closed stream
        # after the test ends (the provider is a process global).
        provider.shutdown()


def test_enable_otel_otlp_requires_endpoint(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="otlp")
    assert not result["ok"] and "endpoint" in result["error"]


def test_enable_otel_otlp_generates_ast_valid(tmp_path: Path) -> None:
    """enable_otel otlp generates an ast-valid setup lazily importing OTLP (separate package)."""
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(
        path=path, app_name="myapp", exporter="otlp", endpoint="http://localhost:4318/v1/traces"
    )
    assert result["ok"], result.get("error")
    src = Path(result["data"]["otel_setup"]).read_text(encoding="utf-8")
    ast.parse(src)
    # The OTLP import is lazy (inside setup_otel) with an actionable install message.
    assert "from opentelemetry.exporter.otlp" in src
    assert "opentelemetry-exporter-otlp" in src
    assert "http://localhost:4318/v1/traces" in src


def test_enable_otel_unknown_exporter_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="zipkin")
    assert not result["ok"] and "Unknown exporter" in result["error"]


def test_enable_otel_missing_app_errs(tmp_path: Path) -> None:
    result = OBS.enable_otel(path=str(tmp_path), app_name="ghost", exporter="console")
    assert not result["ok"] and "not found" in result["error"]


# --------------------------------------------------------------------------- #
# cloud_trace
# --------------------------------------------------------------------------- #
def test_cloud_trace_returns_real_flag() -> None:
    """cloud_trace returns --trace_to_cloud (real 2.1.0 flag) + the tool that applies it."""
    result = OBS.cloud_trace(target="cloud_run")
    assert result["ok"]
    assert result["data"]["flag"] == "--trace_to_cloud"
    assert result["data"]["otel_flag"] == "--otel_to_cloud"
    assert "deploy_cloud_run" in result["data"]["apply_with"]


def test_cloud_trace_marks_otel_flag_manual_only() -> None:
    """Honesty: --otel_to_cloud is marked 'manual only' (the toolkit does not apply it).

    The toolkit only emits --trace_to_cloud (via deploy_*/dev_*). We must therefore NOT claim to
    apply --otel_to_cloud: the return explicitly qualifies it as manual/not auto-applied, and the
    guidance only requires --trace_to_cloud as the flag applied by the toolkit.
    """
    result = OBS.cloud_trace(target="cloud_run")
    assert result["ok"]
    data = result["data"]
    # The flag applied by the toolkit stays --trace_to_cloud.
    assert data["flag"] == "--trace_to_cloud"
    # --otel_to_cloud is exposed but explicitly marked manual-only / not auto-applied.
    note = data["otel_flag_note"].lower()
    assert "manual" in note
    assert "not applied automatically" in note
    # The guidance does not claim that the toolkit applies --otel_to_cloud.
    guidance = data["guidance"].lower()
    assert "--otel_to_cloud" in guidance
    assert "manual only" in guidance


def test_cloud_trace_all_targets() -> None:
    """All supported targets return the flag (cloud_run/agent_engine/gke/web/api)."""
    for target in ("cloud_run", "agent_engine", "gke", "web", "api_server"):
        result = OBS.cloud_trace(target=target)
        assert result["ok"], target
        assert result["data"]["flag"] == "--trace_to_cloud"


def test_cloud_trace_unknown_target_errs() -> None:
    result = OBS.cloud_trace(target="lambda")
    assert not result["ok"] and "Unknown target" in result["error"]


# --------------------------------------------------------------------------- #
# third_party
# --------------------------------------------------------------------------- #
def test_third_party_phoenix_default_endpoint() -> None:
    """phoenix has a default OTLP endpoint + emits OTEL_EXPORTER_OTLP_ENDPOINT."""
    result = OBS.third_party(provider="phoenix")
    assert result["ok"]
    assert result["data"]["endpoint"] == "http://localhost:6006/v1/traces"
    assert result["data"]["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:6006/v1/traces"
    assert result["data"]["exporter"] == "otlp"


def test_third_party_otlp_requires_endpoint() -> None:
    """generic otlp (no default) requires an explicit endpoint."""
    result = OBS.third_party(provider="otlp")
    assert not result["ok"] and "endpoint" in result["error"]


def test_third_party_custom_endpoint_overrides_default() -> None:
    result = OBS.third_party(provider="phoenix", endpoint="https://my-collector/v1/traces")
    assert result["ok"]
    assert result["data"]["endpoint"] == "https://my-collector/v1/traces"


def test_third_party_headers_emitted_as_env() -> None:
    """The headers (e.g. an API key) are emitted as an OTLP env variable (never hardcoded)."""
    result = OBS.third_party(provider="arize", headers={"api_key": "REDACTED", "space_id": "abc"})
    assert result["ok"]
    env = result["data"]["env"]
    assert "OTEL_EXPORTER_OTLP_HEADERS" in env
    assert "api_key=REDACTED" in env["OTEL_EXPORTER_OTLP_HEADERS"]


def test_third_party_unknown_provider_errs() -> None:
    result = OBS.third_party(provider="datadog")
    assert not result["ok"] and "Unknown provider" in result["error"]


# --------------------------------------------------------------------------- #
# trace_view — delegates to dev_web (process registry), no real boot here
# --------------------------------------------------------------------------- #
async def test_trace_view_delegates_to_dev_web_bad_dir() -> None:
    """trace_view delegates to dev.web: an absent folder returns dev.web's err (no boot)."""
    result = await OBS.trace_view(path="/nonexistent-xyz", app_name="ghost")
    assert not result["ok"]
    assert "not found" in result["error"]


async def test_trace_view_invalid_port_errs(tmp_path: Path) -> None:
    """An out-of-bounds port is rejected by dev.web (delegation)."""
    path = _scaffold_app(tmp_path)
    result = await OBS.trace_view(path=path, app_name="myapp", port=70000)
    assert not result["ok"] and "port" in result["error"]


async def test_trace_view_starts_via_registry_then_stop(tmp_path: Path) -> None:
    """trace_view starts a managed process (same registry as dev_web) and returns key/trace_url.

    We do NOT depend on a real HTTP boot (gated in dev): we verify that the delegation does create
    a drivable registry entry (then we stop it). The process may fail to serve without creds — we
    only test the registry wiring.
    """
    path = _scaffold_app(tmp_path)
    result = await OBS.trace_view(path=path, app_name="myapp", port=_free_port())
    assert result["ok"], result.get("error")
    assert result["data"]["delegated_to"] == "dev_web"
    assert result["data"]["trace_url"] == result["data"]["url"]
    key = result["data"]["key"]
    # The process is registered (drivable via the adk_cli registry).
    assert adk_cli.process_status(key)["found"]
    # Explicit cleanup (the fixture would do it too).
    adk_cli.stop_process(key)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# otel_setup rendering — sanity (both exporters ast-valid + console executable)
# --------------------------------------------------------------------------- #
def test_render_otel_setup_both_exporters_ast_valid() -> None:
    for exporter, endpoint in [("console", None), ("otlp", "http://h/v1/traces")]:
        src = observability_setup.render_otel_setup(
            app_name="myapp", exporter=exporter, endpoint=endpoint
        )
        ast.parse(src)


# --------------------------------------------------------------------------- #
# Read-through fastmcp.Client (exposed names + cloud_trace call)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_cloud_trace() -> None:
    """Tools exposed as observability_<bare> (no double prefix); cloud_trace round-trips."""
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {
            "observability_enable_otel",
            "observability_cloud_trace",
            "observability_third_party",
            "observability_trace_view",
        } <= names
        assert not any(n.startswith("observability_observability_") for n in names)

        called = await client.call_tool("observability_cloud_trace", {"target": "cloud_run"})
        payload = _client_payload(called)
        assert payload["ok"]
        assert payload["data"]["flag"] == "--trace_to_cloud"


def _load_module(file: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client_payload(result: Any) -> dict[str, Any]:
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "ok" in structured:
        return structured
    return json.loads(result.content[0].text)
