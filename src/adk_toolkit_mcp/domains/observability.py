"""`observability` domain: OpenTelemetry for ADK agents (P4c).

A FastMCP sub-server mounted under ``namespace="observability"`` → tools exposed as
``observability_<name>``. BARE names (``enable_otel``, ``cloud_trace``, ``third_party``,
``trace_view``). Each tool returns the ``{ok, data, error}`` envelope.

**Honesty about overlaps** (cf. ``docs/adk-api-notes/safety-observability.md``) — this domain does
NOT duplicate the logic of the ``deploy``/``dev`` domains:

1. :func:`enable_otel` — generates a REAL ``<app_dir>/<app>/otel_setup.py`` (ast-valid,
   ruff/isort-clean) that configures an OpenTelemetry exporter (``console`` always available;
   ``otlp`` imported lazily — separate package) wired onto the GLOBAL provider that ADK uses.
2. :func:`cloud_trace` — returns the real CLI flag ``--trace_to_cloud`` (confirmed in P4a on
   ``deploy cloud_run``/``agent_engine``/``gke`` + ``web``/``api_server``) and **references** the
   ``deploy``/``dev`` tool that actually applies it (no flag emitted here). ``--otel_to_cloud`` is
   exposed for information but marked **manual only**: no toolkit tool applies it (we do not claim
   to apply a flag we do not emit).
3. :func:`third_party` — returns the OTLP env variables + a setup snippet for a third-party backend
   (phoenix/arize/weave/signoz/otlp).
4. :func:`trace_view` — **delegates** to the same process registry as ``dev_web`` (ADK's Web UI
   hosts the trace view); a real boot is protected by an env flag (like the ``dev`` tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import is_identifier
from . import dev, observability_setup

observability_server: FastMCP = FastMCP("observability")

#: app_name = Python package identifier (both folder AND module name).
_APP_NAME_ERR = (
    "Invalid app_name: expected a Python identifier "
    "(letters, digits, underscore; not starting with a digit)."
)

#: OTel exporters supported by :func:`enable_otel`.
_EXPORTERS: frozenset[str] = frozenset({"console", "otlp"})

#: Name of the generated OTel setup file (in the app's folder).
_OTEL_FILE = "otel_setup.py"

#: The real ADK 2.1.0 CLI flag enabling Cloud Trace (confirmed on deploy/web/api_server).
_CLOUD_TRACE_FLAG = "--trace_to_cloud"

#: Recognized ``cloud_trace`` targets -> the toolkit tool that actually applies the flag.
_CLOUD_TRACE_TARGETS: dict[str, str] = {
    "cloud_run": "deploy_cloud_run(enable_cloud_trace=True)",
    "agent_engine": "deploy_agent_engine",
    "gke": "deploy_gke",
    "web": "dev_web",
    "api_server": "dev_api_server",
}

#: Third-party OTLP backends supported by :func:`third_party` + their default OTLP endpoint (None
#: if no universal default: the user MUST provide ``endpoint``). All speak OTLP/HTTP.
_THIRD_PARTY: dict[str, str | None] = {
    "phoenix": "http://localhost:6006/v1/traces",
    "arize": "https://otlp.arize.com/v1/traces",
    "weave": None,  # W&B Weave: project-specific endpoint (https://trace.wandb.ai/...).
    "signoz": "http://localhost:4318/v1/traces",
    "otlp": None,  # generic OTLP: endpoint required.
}


# --------------------------------------------------------------------------- #
# Tool 1 — enable_otel (generates otel_setup.py)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
def enable_otel(
    path: str,
    app_name: str,
    exporter: str = "console",
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Generate ``<app_dir>/<app>/otel_setup.py`` configuring an OpenTelemetry exporter.

    ``exporter`` ∈ {``console``, ``otlp``}. The file defines ``setup_otel()`` which builds a
    ``TracerProvider`` (with a ``Resource`` ``service.name=<app>``), adds a
    ``BatchSpanProcessor(exporter)`` to it and installs it as the GLOBAL provider
    (``trace.set_tracer_provider``) — this is THE provider that ADK's telemetry uses (cf. notes).
    The user calls ``setup_otel()`` at startup (before running the agent).

    - ``console``: ``ConsoleSpanExporter`` (base OTel SDK package — always available).
    - ``otlp``: ``OTLPSpanExporter`` (HTTP) imported **lazily** (separate package
      ``opentelemetry-exporter-otlp`` — the generated file documents the install). ``endpoint`` is
      then required (e.g. ``http://localhost:4318/v1/traces``).

    The generated code is ast-valid + ruff/isort-clean (the toolkit never imports OTLP itself).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if exporter not in _EXPORTERS:
        return err(f"Unknown exporter: {exporter!r}. Known: {', '.join(sorted(_EXPORTERS))}.")
    if exporter == "otlp" and not (endpoint or "").strip():
        return err(
            "exporter 'otlp': 'endpoint' is required (e.g. 'http://localhost:4318/v1/traces')."
        )

    app_dir = Path(path) / app_name
    if not (app_dir / "agent.py").is_file():
        agent_py = app_dir / "agent.py"
        return err(f"App folder not found: {agent_py}. Scaffold first (project_create).")

    from ..workspace import Workspace

    ws = Workspace(app_dir)
    source = observability_setup.render_otel_setup(
        app_name=app_name, exporter=exporter, endpoint=endpoint
    )
    changed = ws.write(_OTEL_FILE, source)
    return ok(
        {
            "app_name": app_name,
            "exporter": exporter,
            "endpoint": endpoint,
            "otel_setup": str(ws.path(_OTEL_FILE)),
            "usage": "import otel_setup; otel_setup.setup_otel()  # at startup, before the agent",
            "changed": changed,
            "notes": _otel_notes(exporter),
        }
    )


def _otel_notes(exporter: str) -> list[str]:
    """Actionable notes according to the chosen exporter."""
    notes = [
        "Call setup_otel() at startup (before running the agent): it installs the GLOBAL "
        "TracerProvider that ADK uses for its spans.",
    ]
    if exporter == "otlp":
        notes.append(
            "OTLP requires the separate package: pip install opentelemetry-exporter-otlp "
            "(not pulled by google-adk)."
        )
    return notes


# --------------------------------------------------------------------------- #
# Tool 2 — cloud_trace (returns the real flag + references the deploy/dev tool)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
def cloud_trace(target: str) -> dict[str, Any]:
    """Return the CLI flag enabling Cloud Trace for ``target`` + the tool that applies it.

    ``target`` ∈ {cloud_run, agent_engine, gke, web, api_server}. The flag actually APPLIED by the
    toolkit's ``deploy_*`` / ``dev_*`` tools (confirmed in P4a) is ``--trace_to_cloud``. **This
    tool runs nothing**: it references the ``deploy_*`` / ``dev_*`` tool that emits this flag (the
    ``deploy`` domain validates each emitted flag against the real ``--help``; ``deploy_cloud_run``
    exposes the ``enable_cloud_trace`` parameter which maps to ``--trace_to_cloud``). Avoids any
    duplication of the deployment logic.

    The return clearly distinguishes the two flags:
    - ``flag`` = ``--trace_to_cloud``: the flag THIS toolkit applies via ``apply_with``.
    - ``otel_flag`` = ``--otel_to_cloud``: a **manual-only** ADK flag — NO toolkit tool applies it
      automatically (you must pass it yourself to the ``adk`` CLI). So we do not claim the toolkit
      emits it.
    """
    if target not in _CLOUD_TRACE_TARGETS:
        return err(f"Unknown target: {target!r}. Known: {', '.join(sorted(_CLOUD_TRACE_TARGETS))}.")
    return ok(
        {
            "target": target,
            "flag": _CLOUD_TRACE_FLAG,
            "otel_flag": "--otel_to_cloud",
            "otel_flag_note": (
                "manual only — not applied automatically by this toolkit; pass it yourself to the "
                "'adk' CLI to also export the OTel data to Cloud Trace."
            ),
            "apply_with": _CLOUD_TRACE_TARGETS[target],
            "guidance": (
                f"Enable Cloud Trace by passing {_CLOUD_TRACE_FLAG} to '{target}'. "
                f"Use the toolkit tool '{_CLOUD_TRACE_TARGETS[target]}' which emits and validates "
                f"it (do not reimplement the command here). The flag '{_CLOUD_TRACE_FLAG}' is the "
                "ONLY one applied by this toolkit; '--otel_to_cloud' exists on the ADK side but is "
                "NOT auto-applied by the toolkit (manual only)."
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Tool 3 — third_party (OTLP env + snippet for a third-party backend)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
def third_party(
    provider: str,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Emit the OTLP config (env variables + snippet) for a third-party observability backend.

    ``provider`` ∈ {phoenix, arize, weave, signoz, otlp}. All these backends ingest standard OTLP:
    we return the canonical OTel env variables (``OTEL_EXPORTER_OTLP_ENDPOINT`` +
    ``OTEL_EXPORTER_OTLP_HEADERS`` if ``headers``) and a setup snippet pointing to
    ``observability_enable_otel(exporter='otlp', endpoint=...)``. A default ``endpoint`` is
    provided when the backend has one (phoenix/arize/signoz); for ``weave``/``otlp`` it is required.

    No secret is hardcoded: the ``headers`` (e.g. an API key) are emitted as **environment values
    to set** (the snippet reads ``os.environ``), not frozen in the code.
    """
    if provider not in _THIRD_PARTY:
        return err(f"Unknown provider: {provider!r}. Known: {', '.join(sorted(_THIRD_PARTY))}.")

    default_endpoint = _THIRD_PARTY[provider]
    resolved = (endpoint or "").strip() or default_endpoint
    if not resolved:
        return err(
            f"provider {provider!r}: 'endpoint' is required (no universal default). "
            "Provide the collector's OTLP/HTTP URL (e.g. 'https://host/v1/traces')."
        )

    env: dict[str, str] = {"OTEL_EXPORTER_OTLP_ENDPOINT": resolved}
    if headers:
        # OTLP format: "k1=v1,k2=v2". We document that sensitive values come from the env.
        env["OTEL_EXPORTER_OTLP_HEADERS"] = ",".join(f"{k}={v}" for k, v in headers.items())

    snippet = (
        "# 1) Export the env variables above (do NOT hardcode any secret).\n"
        "# 2) Generate the OTLP setup then call it at startup:\n"
        f"#    observability_enable_otel(path, app_name, exporter='otlp', endpoint='{resolved}')\n"
        "#    import otel_setup; otel_setup.setup_otel()"
    )
    return ok(
        {
            "provider": provider,
            "endpoint": resolved,
            "env": env,
            "exporter": "otlp",
            "setup_snippet": snippet,
            "note": (
                f"{provider} ingests standard OTLP. Install 'opentelemetry-exporter-otlp' then "
                "use observability_enable_otel(exporter='otlp', endpoint=...)."
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Tool 4 — trace_view (delegates to dev_web: the ADK UI hosts the trace view)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
async def trace_view(path: str, app_name: str | None = None, port: int = 8000) -> dict[str, Any]:
    """Launch ADK's Web UI (which hosts the **trace view**) by **delegating** to ``dev_web``.

    ADK's dev UI (``adk web``) includes a "Trace" tab visualizing a run's spans. Rather than
    reimplementing a server, this tool delegates to the **same** process registry path as
    ``dev_web`` (``adk_cli``; cf. the ``dev`` domain). The started server is driven via
    ``dev_status`` / ``dev_logs`` / ``dev_stop`` (same keys).

    Returns ``{key, pid, port, url, trace_url, ...}`` (``trace_url`` = the UI's home page where the
    Trace view is accessible). The real boot only happens if a valid agents folder exists (the
    delegation to ``dev.web`` validates everything); the tests keep a real start behind an env
    flag (like ``dev``).
    """
    result = await dev.web(path=path, app_name=app_name, port=port)
    if not result["ok"]:
        return result
    data = dict(result["data"])
    data["delegated_to"] = "dev_web"
    data["trace_url"] = data.get("url")
    data["guidance"] = (
        "ADK's Web UI is started; open the URL and select a session's 'Trace' tab to visualize "
        "the spans. Drive via dev_status/dev_logs/dev_stop (same key)."
    )
    return ok(data)
