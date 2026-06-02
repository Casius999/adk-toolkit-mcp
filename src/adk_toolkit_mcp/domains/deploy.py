"""`deploy` domain: builds (and, on request, runs) the ``adk deploy`` commands (P4a).

Unlike the P1 domains (which *write* code) and P3 (which *runs* an agent), this domain **builds
the ``adk deploy`` CLI argv** from validated arguments. By default, NOTHING is run
(``execute=False``): we return the built argv + a readable plan. The real deployment
(``execute=True``) goes through :func:`adk_toolkit_mcp.adk_cli.run_adk` — it requires GCP and is
NOT exercised in CI. The command **building** and the **flag validity** are.

Central guardrail: each emitted flag is validated against the real ``adk <sub> --help`` output of
this version of ADK (:func:`adk_cli.available_flags`). If the toolkit built a flag absent from the
installed ADK (version drift), the tool returns ``err`` listing the unknown flags — so it can
never emit an invalid command.

A sub-server mounted under ``namespace="deploy"`` → tools exposed as ``deploy_<name>``. BARE
names. Each tool returns the ``{ok, data, error}`` envelope. Cf.
``docs/adk-api-notes/deploy-dev.md`` for the confirmed 2.1.0 flags (and the differences vs.
intuition: ``--trace_to_cloud`` not
``--enable_cloud_trace``; ``--cluster_name`` not ``--cluster``; agent_engine has no ``--app_name``
and ``--staging_bucket`` is deprecated there).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from .. import adk_cli
from ..envelope import err, ok
from ..workspace import Workspace

deploy_server: FastMCP = FastMCP("deploy")

#: Supported deployment targets (``adk deploy`` subcommands).
DeployTarget = Literal["agent_engine", "cloud_run", "gke"]
_DEPLOY_TARGETS: frozenset[str] = frozenset({"agent_engine", "cloud_run", "gke"})

#: Dockerfile serving ``adk api_server`` (port 8080 = Cloud Run / $PORT default).
_DOCKERFILE_NAME = "Dockerfile"


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _agent_dir(path: str, app_name: str) -> Path:
    """Agent source folder = ``<path>/<app_name>`` (the CLI's positional AGENT)."""
    return Path(path) / app_name


def _require_agent_dir(path: str, app_name: str) -> str | None:
    """Return an error message if the agent folder does not exist, otherwise None."""
    agent_dir = _agent_dir(path, app_name)
    if not agent_dir.is_dir():
        return f"Agent folder not found: {agent_dir}. Scaffold the app first (project_create)."
    return None


def _validate_flags(subcommand: list[str], argv: list[str]) -> set[str]:
    """Return the set of emitted flags (in ``argv``) ABSENT from the installed ADK.

    Compares the ``--xxx`` tokens of ``argv`` against :func:`adk_cli.available_flags`. An empty set
    means "all emitted flags are valid". If ``available_flags`` is empty (introspection
    impossible), we invalidate NOTHING (we cannot assert that a flag is unknown).
    """
    valid = adk_cli.available_flags(subcommand)
    if not valid:
        return set()
    emitted = {t for t in argv if t.startswith("--")}
    return emitted - valid


def _finalize(
    subcommand: list[str],
    argv: list[str],
    notes: list[str],
    execute: bool,
    cwd: str | None,
) -> dict[str, Any]:
    """Validate the emitted flags, then return the plan (``execute=False``) or run (otherwise).

    - First validates that every emitted flag exists in the installed ADK (otherwise ``err``).
    - ``execute=False`` (default): returns ``{argv, plan, notes, executed: False, ...}``.
    - ``execute=True``: invokes ``adk_cli.run_adk`` and returns rc/stdout/stderr (real deployment).
    """
    unknown = _validate_flags(subcommand, argv)
    if unknown:
        return err(
            "Flags unknown to the installed ADK: "
            + ", ".join(sorted(unknown))
            + ". (Re-introspect docs/adk-api-notes/deploy-dev.md — version drift?)"
        )

    plan = "adk " + " ".join(argv)
    data: dict[str, Any] = {
        "target": subcommand[-1],
        "argv": argv,
        "plan": plan,
        "notes": notes,
        "executed": execute,
    }
    if not execute:
        return ok(data)

    result = adk_cli.run_adk(argv, cwd=cwd)
    data.update(
        {
            "rc": result.get("rc"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
    )
    return ok(data)


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def preflight(target: str = "cloud_run") -> dict[str, Any]:
    """Best-effort checks before a deployment (never raises/fails).

    Checks whether ``gcloud`` is on the PATH, whether ``adk`` is runnable, and gives findings
    tailored to the target (``agent_engine``/``cloud_run``/``gke``). Always returns ``ok`` — it is
    a diagnostic, not a blocking gate.
    """
    gcloud = shutil.which("gcloud") is not None
    kubectl = shutil.which("kubectl") is not None
    adk_help = adk_cli.run_adk(["--help"], timeout=30)
    adk_runnable = adk_help.get("rc") == 0

    findings: list[str] = []
    if not gcloud:
        findings.append("gcloud not found on the PATH (required for cloud_run/gke/agent_engine).")
    if not adk_runnable:
        findings.append("The adk CLI does not respond to --help (check the google-adk install).")
    if target == "gke" and not kubectl:
        findings.append("kubectl not found (recommended to inspect a GKE deployment).")
    if target not in _DEPLOY_TARGETS:
        findings.append(
            f"Non-standard target: {target!r}. Expected one of {sorted(_DEPLOY_TARGETS)}."
        )
    if not findings:
        findings.append("No obvious problem detected (best-effort checks).")

    return ok(
        {
            "target": target,
            "gcloud_on_path": gcloud,
            "kubectl_on_path": kubectl,
            "adk_runnable": adk_runnable,
            "findings": findings,
        }
    )


# --------------------------------------------------------------------------- #
# agent_engine
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def agent_engine(
    path: str,
    app_name: str,
    project: str,
    region: str,
    staging_bucket: str | None = None,
    display_name: str | None = None,
    requirements_file: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Build ``adk deploy agent_engine`` (Vertex AI Agent Engine).

    Confirmed 2.1.0 flags: ``--project``, ``--region``, ``--display_name``,
    ``--requirements_file``. ``app_name`` is mapped to ``--display_name`` (agent_engine has NO
    ``--app_name``); an explicit ``display_name`` wins. ``staging_bucket`` is **deprecated** (no-op
    in 2.1.0): it is NOT emitted as a flag — only flagged in ``notes``. The positional AGENT =
    ``<path>/<app_name>``.

    By default (``execute=False``), returns the argv + a plan; ``execute=True`` runs the real
    deployment (requires GCP creds — not run in CI).
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)
    if not project.strip():
        return err("project is required for agent_engine (Vertex).")
    if not region.strip():
        return err("region is required for agent_engine (Vertex).")

    name = (display_name or app_name).strip()
    argv: list[str] = [
        "deploy",
        "agent_engine",
        "--project",
        project.strip(),
        "--region",
        region.strip(),
    ]
    argv += ["--display_name", name]
    if requirements_file:
        argv += ["--requirements_file", requirements_file]
    argv.append(str(_agent_dir(path, app_name)))

    notes: list[str] = []
    if staging_bucket:
        notes.append(
            "staging_bucket is deprecated in ADK 2.1.0 (no-op): not passed. "
            "Agent Engine manages its own staging bucket."
        )
    notes.append("Agent Engine deployment: requires Vertex AI credentials (project/region).")

    return _finalize(["deploy", "agent_engine"], argv, notes, execute, cwd=path)


# --------------------------------------------------------------------------- #
# cloud_run
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def cloud_run(
    path: str,
    app_name: str,
    project: str,
    region: str,
    service_name: str | None = None,
    with_ui: bool = False,
    enable_cloud_trace: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Build ``adk deploy cloud_run`` (Google Cloud Run).

    2.1.0 flags: ``--project`` (required), ``--region`` (required), ``--service_name`` (optional),
    ``--app_name`` (= ``app_name``), ``--with_ui`` (boolean), ``--trace_to_cloud`` (boolean — the
    ``enable_cloud_trace`` parameter maps to THIS flag, NOT ``--enable_cloud_trace`` which does not
    exist). Positional AGENT = ``<path>/<app_name>``.
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)
    if not project.strip():
        return err("project is required for cloud_run.")
    if not region.strip():
        return err("region is required for cloud_run.")

    argv: list[str] = [
        "deploy",
        "cloud_run",
        "--project",
        project.strip(),
        "--region",
        region.strip(),
    ]
    if service_name and service_name.strip():
        argv += ["--service_name", service_name.strip()]
    argv += ["--app_name", app_name.strip()]
    if with_ui:
        argv.append("--with_ui")
    if enable_cloud_trace:
        argv.append("--trace_to_cloud")
    argv.append(str(_agent_dir(path, app_name)))

    notes = ["Cloud Run deployment: requires gcloud + a GCP project with Cloud Run enabled."]
    if with_ui:
        notes.append("--with_ui serves the ADK Web UI (dev/test only — not for production).")

    return _finalize(["deploy", "cloud_run"], argv, notes, execute, cwd=path)


# --------------------------------------------------------------------------- #
# gke
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def gke(
    path: str,
    app_name: str,
    project: str,
    region: str,
    cluster: str,
    service_name: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Build ``adk deploy gke`` (Google Kubernetes Engine).

    2.1.0 flags: ``--project`` (required), ``--region`` (required), ``--cluster_name`` (required —
    the ``cluster`` parameter maps to THIS flag, NOT ``--cluster``), ``--service_name`` (optional),
    ``--app_name``. Positional AGENT = ``<path>/<app_name>``.
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)
    if not project.strip():
        return err("project is required for gke.")
    if not region.strip():
        return err("region is required for gke.")
    if not cluster.strip():
        return err("cluster (--cluster_name) is required for gke.")

    argv: list[str] = ["deploy", "gke", "--project", project.strip(), "--region", region.strip()]
    argv += ["--cluster_name", cluster.strip()]
    if service_name and service_name.strip():
        argv += ["--service_name", service_name.strip()]
    argv += ["--app_name", app_name.strip()]
    argv.append(str(_agent_dir(path, app_name)))

    notes = ["GKE deployment: requires gcloud + an existing GKE cluster + configured kubectl."]
    return _finalize(["deploy", "gke"], argv, notes, execute, cwd=path)


# --------------------------------------------------------------------------- #
# containerize
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def containerize(path: str, app_name: str) -> dict[str, Any]:
    """Generate a ``Dockerfile`` for the app (serving ``adk api_server``). Idempotent via Workspace.

    The Dockerfile installs ``google-adk`` + copies the app and runs ``adk api_server`` on
    ``$PORT`` (Cloud Run injects ``PORT``=8080 by default). Written to ``<path>/Dockerfile``.
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)

    content = _dockerfile_content(app_name)
    ws = Workspace(Path(path))
    changed = ws.write(_DOCKERFILE_NAME, content)
    return ok(
        {
            "path": str(ws.path(_DOCKERFILE_NAME)),
            "app_name": app_name,
            "changed": changed,
        }
    )


def _dockerfile_content(app_name: str) -> str:
    """Source of the Dockerfile serving ``adk api_server`` on ``$PORT`` (8080 by default).

    The app lives in ``/app/<app_name>``; ``adk api_server /app`` treats ``/app`` as an agents
    folder (each subfolder = an agent). ``$PORT`` is resolved at runtime via a shell to honor the
    Cloud Run convention.
    """
    return (
        "FROM python:3.12-slim\n"
        "\n"
        "ENV PYTHONUNBUFFERED=1 \\\n"
        "    PIP_NO_CACHE_DIR=1 \\\n"
        "    PORT=8080\n"
        "\n"
        "WORKDIR /app\n"
        "\n"
        "RUN pip install --no-cache-dir google-adk\n"
        "\n"
        f"COPY . /app/{app_name}\n"
        "\n"
        "EXPOSE 8080\n"
        "\n"
        "# Serves the ADK API; /app is the agents folder (each subfolder = an agent).\n"
        'CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT} /app"]\n'
    )


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def status(
    target: str,
    project: str | None = None,
    region: str | None = None,
    service_name: str | None = None,
    cluster: str | None = None,
) -> dict[str, Any]:
    """Best-effort status of a deployment (short timeout; never blocks).

    - ``cloud_run``: shells to ``gcloud run services describe`` if ``gcloud`` is present.
    - ``gke``: shells to ``kubectl get service`` if ``kubectl`` is present.
    - ``agent_engine``: no direct status CLI → guidance.

    If the required tool is absent, returns ``available=False`` + guidance (never a blocking
    error). An unknown target → ``err``.
    """
    if target not in _DEPLOY_TARGETS:
        return err(f"Unknown target: {target!r}. Expected one of {sorted(_DEPLOY_TARGETS)}.")

    if target == "cloud_run":
        return _status_cloud_run(project, region, service_name)
    if target == "gke":
        return _status_gke(cluster)
    return ok(
        {
            "target": "agent_engine",
            "available": False,
            "guidance": (
                "Agent Engine has no dedicated status CLI command. Inspect via the Vertex AI "
                "console or the vertexai SDK (reasoning_engines.list)."
            ),
        }
    )


def _status_cloud_run(
    project: str | None, region: str | None, service_name: str | None
) -> dict[str, Any]:
    """Cloud Run status via ``gcloud`` (short timeout); guidance if gcloud is absent."""
    if shutil.which("gcloud") is None:
        return ok(
            {
                "target": "cloud_run",
                "available": False,
                "guidance": ("gcloud not found. Install the Google Cloud SDK to query Cloud Run."),
            }
        )
    if not (service_name and project and region):
        return ok(
            {
                "target": "cloud_run",
                "available": True,
                "guidance": "Provide project, region and service_name to query the service.",
            }
        )
    args = [
        "run",
        "services",
        "describe",
        service_name,
        "--project",
        project,
        "--region",
        region,
        "--format=value(status.url)",
    ]
    result = _run_tool(["gcloud", *args])
    return ok({"target": "cloud_run", "available": True, **result})


def _status_gke(cluster: str | None) -> dict[str, Any]:
    """GKE status via ``kubectl`` (short timeout); guidance if kubectl is absent."""
    if shutil.which("kubectl") is None:
        return ok(
            {
                "target": "gke",
                "available": False,
                "guidance": (
                    "kubectl not found. Configure it: gcloud container clusters get-credentials."
                ),
            }
        )
    result = _run_tool(["kubectl", "get", "services", "-o", "wide"])
    return ok({"target": "gke", "available": True, "cluster": cluster, **result})


def _run_tool(argv: list[str]) -> dict[str, Any]:
    """Run an external tool (gcloud/kubectl) with a short timeout; return rc/stdout/stderr.

    Best-effort: any failure (tool absent despite ``which``, timeout, error) is captured in data,
    never propagated. Fixed argv (no shell input).
    """
    import subprocess  # noqa: PLC0415 - local import: optional status tool

    try:
        completed = subprocess.run(  # noqa: S603 - argv list, known status tools
            argv,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": -1, "stdout": "", "stderr": f"invocation failed: {exc}"}
    return {
        "rc": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }
