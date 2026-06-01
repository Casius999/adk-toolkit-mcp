"""Tests du domaine `observability` (P4c) — OpenTelemetry, Cloud Trace, backends tiers, trace_view.

Couvre :
- ``enable_otel`` : génère un ``otel_setup.py`` ast-valide (console + otlp) ; validation endpoint.
  Le setup console est aussi EXÉCUTÉ (installe réellement un TracerProvider global).
- ``cloud_trace`` : renvoie le vrai flag ``--trace_to_cloud`` + référence l'outil deploy/dev.
- ``third_party`` : variables d'env OTLP + snippet pour phoenix/arize/weave/signoz/otlp.
- ``trace_view`` : DÉLÈGUE à ``dev_web`` (registre de process) ; testé sans boot réel (la
  délégation à une app absente renvoie l'``err`` de ``dev.web``). Un vrai boot reste derrière le
  flag d'env de ``dev``.
- Lecture via ``fastmcp.Client`` en mémoire (noms exposés + appel ``observability_cloud_trace``).
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
    """Termine tout process géré avant ET après chaque test (aucun orphelin / port lié)."""
    adk_cli.stop_all_processes()
    yield
    adk_cli.stop_all_processes()


def _scaffold_app(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffolde une app ADK minimale (suffit : agent.py présent)."""
    from adk_toolkit_mcp.domains.project import create as project_create

    path = str(tmp_path)
    assert project_create(path=path, app_name=app_name)["ok"]
    return path


# --------------------------------------------------------------------------- #
# enable_otel
# --------------------------------------------------------------------------- #
def test_enable_otel_console_generates_ast_valid(tmp_path: Path) -> None:
    """enable_otel console écrit un otel_setup.py ast-valide définissant setup_otel()."""
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
    """Le setup console généré s'EXÉCUTE et installe réellement un TracerProvider global."""
    import opentelemetry.trace as ot_trace

    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="console")
    module = _load_module(Path(result["data"]["otel_setup"]), "otel_console_test")
    provider = module.setup_otel()
    try:
        assert type(provider).__name__ == "TracerProvider"
        assert ot_trace.get_tracer_provider() is provider
    finally:
        # Arrête le BatchSpanProcessor (thread de fond) pour éviter un flush vers un flux fermé
        # après la fin du test (le provider est un global de process).
        provider.shutdown()


def test_enable_otel_otlp_requires_endpoint(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="otlp")
    assert not result["ok"] and "endpoint" in result["error"]


def test_enable_otel_otlp_generates_ast_valid(tmp_path: Path) -> None:
    """enable_otel otlp génère un setup ast-valide important OTLP paresseusement (paquet séparé)."""
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(
        path=path, app_name="myapp", exporter="otlp", endpoint="http://localhost:4318/v1/traces"
    )
    assert result["ok"], result.get("error")
    src = Path(result["data"]["otel_setup"]).read_text(encoding="utf-8")
    ast.parse(src)
    # Import OTLP est paresseux (dans setup_otel) avec un message d'install actionnable.
    assert "from opentelemetry.exporter.otlp" in src
    assert "opentelemetry-exporter-otlp" in src
    assert "http://localhost:4318/v1/traces" in src


def test_enable_otel_unknown_exporter_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = OBS.enable_otel(path=path, app_name="myapp", exporter="zipkin")
    assert not result["ok"] and "exporter inconnu" in result["error"]


def test_enable_otel_missing_app_errs(tmp_path: Path) -> None:
    result = OBS.enable_otel(path=str(tmp_path), app_name="ghost", exporter="console")
    assert not result["ok"] and "introuvable" in result["error"]


# --------------------------------------------------------------------------- #
# cloud_trace
# --------------------------------------------------------------------------- #
def test_cloud_trace_returns_real_flag() -> None:
    """cloud_trace renvoie --trace_to_cloud (flag réel 2.1.0) + l'outil qui l'applique."""
    result = OBS.cloud_trace(target="cloud_run")
    assert result["ok"]
    assert result["data"]["flag"] == "--trace_to_cloud"
    assert result["data"]["otel_flag"] == "--otel_to_cloud"
    assert "deploy_cloud_run" in result["data"]["apply_with"]


def test_cloud_trace_marks_otel_flag_manual_only() -> None:
    """Honnêteté : --otel_to_cloud est marqué 'manuel uniquement' (le toolkit ne l'applique pas).

    Le toolkit n'émet que --trace_to_cloud (via deploy_*/dev_*). On ne doit donc PAS prétendre
    appliquer --otel_to_cloud : le retour le qualifie explicitement de manuel/non-auto-appliqué, et
    la guidance ne réclame que --trace_to_cloud comme flag appliqué par le toolkit.
    """
    result = OBS.cloud_trace(target="cloud_run")
    assert result["ok"]
    data = result["data"]
    # Le flag appliqué par le toolkit reste --trace_to_cloud.
    assert data["flag"] == "--trace_to_cloud"
    # --otel_to_cloud est exposé mais explicitement marqué manuel-only / non auto-appliqué.
    note = data["otel_flag_note"].lower()
    assert "manuel" in note
    assert "non appliqué automatiquement" in note
    # La guidance ne prétend pas que le toolkit applique --otel_to_cloud.
    guidance = data["guidance"].lower()
    assert "--otel_to_cloud" in guidance
    assert "manuel uniquement" in guidance


def test_cloud_trace_all_targets() -> None:
    """Toutes les cibles supportées renvoient le flag (cloud_run/agent_engine/gke/web/api)."""
    for target in ("cloud_run", "agent_engine", "gke", "web", "api_server"):
        result = OBS.cloud_trace(target=target)
        assert result["ok"], target
        assert result["data"]["flag"] == "--trace_to_cloud"


def test_cloud_trace_unknown_target_errs() -> None:
    result = OBS.cloud_trace(target="lambda")
    assert not result["ok"] and "target inconnu" in result["error"]


# --------------------------------------------------------------------------- #
# third_party
# --------------------------------------------------------------------------- #
def test_third_party_phoenix_default_endpoint() -> None:
    """phoenix a un endpoint OTLP par défaut + émet OTEL_EXPORTER_OTLP_ENDPOINT."""
    result = OBS.third_party(provider="phoenix")
    assert result["ok"]
    assert result["data"]["endpoint"] == "http://localhost:6006/v1/traces"
    assert result["data"]["env"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:6006/v1/traces"
    assert result["data"]["exporter"] == "otlp"


def test_third_party_otlp_requires_endpoint() -> None:
    """otlp générique (sans défaut) exige un endpoint explicite."""
    result = OBS.third_party(provider="otlp")
    assert not result["ok"] and "endpoint" in result["error"]


def test_third_party_custom_endpoint_overrides_default() -> None:
    result = OBS.third_party(provider="phoenix", endpoint="https://my-collector/v1/traces")
    assert result["ok"]
    assert result["data"]["endpoint"] == "https://my-collector/v1/traces"


def test_third_party_headers_emitted_as_env() -> None:
    """Les headers (ex. clé API) sont émis comme variable d'env OTLP (jamais figés en dur)."""
    result = OBS.third_party(provider="arize", headers={"api_key": "REDACTED", "space_id": "abc"})
    assert result["ok"]
    env = result["data"]["env"]
    assert "OTEL_EXPORTER_OTLP_HEADERS" in env
    assert "api_key=REDACTED" in env["OTEL_EXPORTER_OTLP_HEADERS"]


def test_third_party_unknown_provider_errs() -> None:
    result = OBS.third_party(provider="datadog")
    assert not result["ok"] and "provider inconnu" in result["error"]


# --------------------------------------------------------------------------- #
# trace_view — délègue à dev_web (registre de process), pas de boot réel ici
# --------------------------------------------------------------------------- #
async def test_trace_view_delegates_to_dev_web_bad_dir() -> None:
    """trace_view délègue à dev.web : un dossier absent renvoie l'err de dev.web (aucun boot)."""
    result = await OBS.trace_view(path="/nonexistent-xyz", app_name="ghost")
    assert not result["ok"]
    assert "introuvable" in result["error"]


async def test_trace_view_invalid_port_errs(tmp_path: Path) -> None:
    """Un port hors bornes est rejeté par dev.web (délégation)."""
    path = _scaffold_app(tmp_path)
    result = await OBS.trace_view(path=path, app_name="myapp", port=70000)
    assert not result["ok"] and "port" in result["error"]


async def test_trace_view_starts_via_registry_then_stop(tmp_path: Path) -> None:
    """trace_view démarre un process géré (même registre que dev_web) et renvoie key/trace_url.

    On NE dépend PAS d'un vrai boot HTTP (gated dans dev) : on vérifie que la délégation crée bien
    une entrée de registre pilotable (puis on l'arrête). Le process peut échouer à servir sans
    creds — on ne teste que le câblage du registre.
    """
    path = _scaffold_app(tmp_path)
    result = await OBS.trace_view(path=path, app_name="myapp", port=_free_port())
    assert result["ok"], result.get("error")
    assert result["data"]["delegated_to"] == "dev_web"
    assert result["data"]["trace_url"] == result["data"]["url"]
    key = result["data"]["key"]
    # Le process est enregistré (pilotable via le registre adk_cli).
    assert adk_cli.process_status(key)["found"]
    # Nettoyage explicite (la fixture le ferait aussi).
    adk_cli.stop_process(key)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# Rendu otel_setup — sanity (les deux exportateurs ast-valides + console exécutable)
# --------------------------------------------------------------------------- #
def test_render_otel_setup_both_exporters_ast_valid() -> None:
    for exporter, endpoint in [("console", None), ("otlp", "http://h/v1/traces")]:
        src = observability_setup.render_otel_setup(
            app_name="myapp", exporter=exporter, endpoint=endpoint
        )
        ast.parse(src)


# --------------------------------------------------------------------------- #
# Read-through fastmcp.Client (noms exposés + appel cloud_trace)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_cloud_trace() -> None:
    """Outils exposés observability_<bare> (pas de double-préfixe) ; cloud_trace round-trip."""
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
