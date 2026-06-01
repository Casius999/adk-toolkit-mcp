"""Domaine `deploy` : construit (et, sur demande, exécute) les commandes ``adk deploy`` (P4a).

Contrairement aux domaines P1 (qui *écrivent* du code) et P3 (qui *exécutent* un agent), ce
domaine **construit l'argv de la CLI ``adk deploy``** à partir d'arguments validés. Par défaut,
RIEN n'est exécuté (``execute=False``) : on renvoie l'argv construit + un plan lisible. Le vrai
déploiement (``execute=True``) passe par :func:`adk_toolkit_mcp.adk_cli.run_adk` — il nécessite
GCP et n'est PAS exercé en CI. La **construction** des commandes et la **validité des flags** le
sont.

Garde-fou central : chaque flag émis est validé contre la vraie sortie ``adk <sub> --help`` de
cette version d'ADK (:func:`adk_cli.available_flags`). Si le toolkit construisait un flag absent
de l'ADK installé (dérive de version), l'outil renvoie ``err`` en listant les flags inconnus —
il ne peut donc jamais émettre une commande invalide.

Sous-serveur monté sous ``namespace="deploy"`` → outils exposés ``deploy_<nom>``. Noms BARE.
Chaque outil renvoie l'enveloppe ``{ok, data, error}``. Cf. ``docs/adk-api-notes/deploy-dev.md``
pour les flags 2.1.0 confirmés (et les écarts vs. l'intuition : ``--trace_to_cloud`` et non
``--enable_cloud_trace`` ; ``--cluster_name`` et non ``--cluster`` ; agent_engine n'a pas
``--app_name`` et ``--staging_bucket`` y est déprécié).
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

#: Cibles de déploiement supportées (sous-commandes ``adk deploy``).
DeployTarget = Literal["agent_engine", "cloud_run", "gke"]
_DEPLOY_TARGETS: frozenset[str] = frozenset({"agent_engine", "cloud_run", "gke"})

#: Dockerfile servant ``adk api_server`` (port 8080 = défaut Cloud Run / $PORT).
_DOCKERFILE_NAME = "Dockerfile"


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _agent_dir(path: str, app_name: str) -> Path:
    """Dossier source de l'agent = ``<path>/<app_name>`` (positionnel AGENT de la CLI)."""
    return Path(path) / app_name


def _require_agent_dir(path: str, app_name: str) -> str | None:
    """Renvoie un message d'erreur si le dossier de l'agent n'existe pas, sinon None."""
    agent_dir = _agent_dir(path, app_name)
    if not agent_dir.is_dir():
        return (
            f"Dossier d'agent introuvable : {agent_dir}. Scaffolde d'abord l'app (project_create)."
        )
    return None


def _validate_flags(subcommand: list[str], argv: list[str]) -> set[str]:
    """Renvoie l'ensemble des flags émis (dans ``argv``) ABSENTS de l'ADK installé.

    Compare les tokens ``--xxx`` de ``argv`` à :func:`adk_cli.available_flags`. Un ensemble vide
    signifie « tous les flags émis sont valides ». Si ``available_flags`` est vide (introspection
    impossible), on n'invalide RIEN (on ne peut pas affirmer qu'un flag est inconnu).
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
    """Valide les flags émis, puis renvoie le plan (``execute=False``) ou exécute (sinon).

    - Valide d'abord que tout flag émis existe dans l'ADK installé (sinon ``err``).
    - ``execute=False`` (défaut) : renvoie ``{argv, plan, notes, executed: False, ...}``.
    - ``execute=True`` : invoque ``adk_cli.run_adk`` et renvoie rc/stdout/stderr (déploiement réel).
    """
    unknown = _validate_flags(subcommand, argv)
    if unknown:
        return err(
            "Flags inconnus de l'ADK installé : "
            + ", ".join(sorted(unknown))
            + ". (Réintrospecte docs/adk-api-notes/deploy-dev.md — drift de version ?)"
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
    """Vérifications best-effort avant un déploiement (ne lève/échoue jamais).

    Contrôle si ``gcloud`` est sur le PATH, si ``adk`` est exécutable, et donne des findings
    orientés selon la cible (``agent_engine``/``cloud_run``/``gke``). Renvoie toujours ``ok`` —
    c'est un diagnostic, pas une porte bloquante.
    """
    gcloud = shutil.which("gcloud") is not None
    kubectl = shutil.which("kubectl") is not None
    adk_help = adk_cli.run_adk(["--help"], timeout=30)
    adk_runnable = adk_help.get("rc") == 0

    findings: list[str] = []
    if not gcloud:
        findings.append("gcloud introuvable sur le PATH (requis pour cloud_run/gke/agent_engine).")
    if not adk_runnable:
        findings.append("La CLI adk ne répond pas à --help (vérifie l'installation google-adk).")
    if target == "gke" and not kubectl:
        findings.append("kubectl introuvable (recommandé pour inspecter un déploiement GKE).")
    if target not in _DEPLOY_TARGETS:
        findings.append(
            f"Cible non standard : {target!r}. Attendu l'une de {sorted(_DEPLOY_TARGETS)}."
        )
    if not findings:
        findings.append("Aucun problème évident détecté (vérifications best-effort).")

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
    """Construit ``adk deploy agent_engine`` (Vertex AI Agent Engine).

    Flags 2.1.0 confirmés : ``--project``, ``--region``, ``--display_name``,
    ``--requirements_file``. ``app_name`` est mappé sur ``--display_name`` (agent_engine n'a PAS
    de ``--app_name``) ; un ``display_name`` explicite l'emporte. ``staging_bucket`` est
    **déprécié** (no-op en 2.1.0) : il N'est PAS émis comme flag — seulement signalé dans
    ``notes``. Le positionnel AGENT = ``<path>/<app_name>``.

    Par défaut (``execute=False``), renvoie l'argv + un plan ; ``execute=True`` lance le vrai
    déploiement (nécessite des creds GCP — non exécuté en CI).
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)
    if not project.strip():
        return err("project est requis pour agent_engine (Vertex).")
    if not region.strip():
        return err("region est requise pour agent_engine (Vertex).")

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
            "staging_bucket est déprécié en ADK 2.1.0 (no-op) : non transmis. "
            "Agent Engine gère son propre bucket de staging."
        )
    notes.append("Déploiement Agent Engine : nécessite des identifiants Vertex AI (projet/région).")

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
    """Construit ``adk deploy cloud_run`` (Google Cloud Run).

    Flags 2.1.0 : ``--project`` (requis), ``--region`` (requis), ``--service_name`` (optionnel),
    ``--app_name`` (= ``app_name``), ``--with_ui`` (booléen), ``--trace_to_cloud`` (booléen — le
    paramètre ``enable_cloud_trace`` mappe sur CE flag, PAS ``--enable_cloud_trace`` qui n'existe
    pas). Positionnel AGENT = ``<path>/<app_name>``.
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)
    if not project.strip():
        return err("project est requis pour cloud_run.")
    if not region.strip():
        return err("region est requise pour cloud_run.")

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

    notes = ["Déploiement Cloud Run : nécessite gcloud + un projet GCP avec Cloud Run activé."]
    if with_ui:
        notes.append("--with_ui sert le Web UI ADK (dev/test uniquement — pas pour la production).")

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
    """Construit ``adk deploy gke`` (Google Kubernetes Engine).

    Flags 2.1.0 : ``--project`` (requis), ``--region`` (requis), ``--cluster_name`` (requis — le
    paramètre ``cluster`` mappe sur CE flag, PAS ``--cluster``), ``--service_name`` (optionnel),
    ``--app_name``. Positionnel AGENT = ``<path>/<app_name>``.
    """
    dir_error = _require_agent_dir(path, app_name)
    if dir_error is not None:
        return err(dir_error)
    if not project.strip():
        return err("project est requis pour gke.")
    if not region.strip():
        return err("region est requise pour gke.")
    if not cluster.strip():
        return err("cluster (--cluster_name) est requis pour gke.")

    argv: list[str] = ["deploy", "gke", "--project", project.strip(), "--region", region.strip()]
    argv += ["--cluster_name", cluster.strip()]
    if service_name and service_name.strip():
        argv += ["--service_name", service_name.strip()]
    argv += ["--app_name", app_name.strip()]
    argv.append(str(_agent_dir(path, app_name)))

    notes = ["Déploiement GKE : nécessite gcloud + un cluster GKE existant + kubectl configuré."]
    return _finalize(["deploy", "gke"], argv, notes, execute, cwd=path)


# --------------------------------------------------------------------------- #
# containerize
# --------------------------------------------------------------------------- #
@deploy_server.tool(tags={"deploy"})
def containerize(path: str, app_name: str) -> dict[str, Any]:
    """Génère un ``Dockerfile`` pour l'app (servant ``adk api_server``). Idempotent via Workspace.

    Le Dockerfile installe ``google-adk`` + copie l'app et lance ``adk api_server`` sur le
    ``$PORT`` (Cloud Run injecte ``PORT``=8080 par défaut). Écrit dans ``<path>/Dockerfile``.
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
    """Source du Dockerfile servant ``adk api_server`` sur ``$PORT`` (8080 par défaut).

    L'app vit dans ``/app/<app_name>`` ; ``adk api_server /app`` traite ``/app`` comme un dossier
    d'agents (chaque sous-dossier = un agent). Le ``$PORT`` est résolu au runtime via un shell
    pour respecter la convention Cloud Run.
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
        "# Sert l'API ADK ; /app est le dossier d'agents (chaque sous-dossier = un agent).\n"
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
    """Statut best-effort d'un déploiement (court timeout ; ne bloque jamais).

    - ``cloud_run`` : shell vers ``gcloud run services describe`` si ``gcloud`` est présent.
    - ``gke`` : shell vers ``kubectl get service`` si ``kubectl`` est présent.
    - ``agent_engine`` : pas de CLI de statut directe → guidance.

    Si l'outil requis est absent, renvoie ``available=False`` + une guidance (jamais d'erreur de
    blocage). Une cible inconnue → ``err``.
    """
    if target not in _DEPLOY_TARGETS:
        return err(f"Cible inconnue : {target!r}. Attendu l'une de {sorted(_DEPLOY_TARGETS)}.")

    if target == "cloud_run":
        return _status_cloud_run(project, region, service_name)
    if target == "gke":
        return _status_gke(cluster)
    return ok(
        {
            "target": "agent_engine",
            "available": False,
            "guidance": (
                "Agent Engine n'a pas de commande de statut CLI dédiée. Inspecte via la console "
                "Vertex AI ou le SDK vertexai (reasoning_engines.list)."
            ),
        }
    )


def _status_cloud_run(
    project: str | None, region: str | None, service_name: str | None
) -> dict[str, Any]:
    """Statut Cloud Run via ``gcloud`` (court timeout) ; guidance si gcloud absent."""
    if shutil.which("gcloud") is None:
        return ok(
            {
                "target": "cloud_run",
                "available": False,
                "guidance": (
                    "gcloud introuvable. Installe le SDK Google Cloud pour interroger Cloud Run."
                ),
            }
        )
    if not (service_name and project and region):
        return ok(
            {
                "target": "cloud_run",
                "available": True,
                "guidance": "Fournis project, region et service_name pour interroger le service.",
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
    """Statut GKE via ``kubectl`` (court timeout) ; guidance si kubectl absent."""
    if shutil.which("kubectl") is None:
        return ok(
            {
                "target": "gke",
                "available": False,
                "guidance": (
                    "kubectl introuvable. Configure-le : gcloud container clusters get-credentials."
                ),
            }
        )
    result = _run_tool(["kubectl", "get", "services", "-o", "wide"])
    return ok({"target": "gke", "available": True, "cluster": cluster, **result})


def _run_tool(argv: list[str]) -> dict[str, Any]:
    """Exécute un outil externe (gcloud/kubectl) avec un court timeout ; renvoie rc/stdout/stderr.

    Best-effort : tout échec (outil absent malgré ``which``, timeout, erreur) est capturé en
    données, jamais propagé. Argv fixe (pas d'entrée shell).
    """
    import subprocess  # noqa: PLC0415 - import local : outil de statut optionnel

    try:
        completed = subprocess.run(  # noqa: S603 - argv list, outils de statut connus
            argv,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"rc": -1, "stdout": "", "stderr": f"échec d'invocation : {exc}"}
    return {
        "rc": completed.returncode,
        "stdout": (completed.stdout or "").strip(),
        "stderr": (completed.stderr or "").strip(),
    }
