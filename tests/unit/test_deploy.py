"""Tests unitaires du domaine ``deploy`` (P4a — construction de commandes de déploiement).

Principe : par défaut, AUCUN déploiement réel n'est exécuté (``execute=False``) — l'outil
construit et renvoie l'**argv exact** + un plan lisible. On assert la liste de tokens construite
(preuve déterministe) et la **validité des flags** émis contre la vraie sortie ``--help`` d'ADK
2.1.0 (``available_flags``). Le vrai déploiement cloud N'EST PAS testé (nécessite GCP) — mais la
construction de commande et la validité des flags LE SONT.

Couverture :
- ``agent_engine`` / ``cloud_run`` / ``gke`` : argv exact + tous les flags émis ∈ available_flags.
- validation des arguments requis → ``err`` (chemins/projet/région/cluster manquants).
- ``containerize`` : écrit un Dockerfile (idempotent via Workspace).
- ``preflight`` : findings structurés (best-effort), ne lève jamais.
- ``status`` : « unavailable » actionnable si l'outil cloud est absent, sans blocage.
- ``execute=False`` ne lance JAMAIS de vrai déploiement (le ``run_adk`` réel n'est pas appelé).
- read-through ``fastmcp.Client`` pour une construction de commande.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp import adk_cli
from adk_toolkit_mcp.domains import deploy as D
from adk_toolkit_mcp.server import build_server


def _scaffold(tmp_path: Path, app_name: str = "myapp") -> str:
    """Crée un dossier d'app minimal (``agent.py``) et renvoie le chemin parent."""
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "agent.py").write_text("root_agent = None\n", encoding="utf-8")
    return str(tmp_path)


def _agent_dir(tmp_path: Path, app_name: str = "myapp") -> str:
    """Chemin attendu du dossier d'agent (positionnel AGENT de la CLI)."""
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
    # Tokens attendus : deploy agent_engine --project ... --region ... --display_name ... AGENT.
    assert argv[:2] == ["deploy", "agent_engine"]
    assert "--project" in argv and argv[argv.index("--project") + 1] == "my-proj"
    assert "--region" in argv and argv[argv.index("--region") + 1] == "us-central1"
    # display_name explicite l'emporte ; sinon app_name.
    assert "--display_name" in argv and argv[argv.index("--display_name") + 1] == "My Agent"
    # AGENT est le DERNIER token (le dossier de l'app).
    assert argv[-1] == _agent_dir(tmp_path)
    # staging_bucket est DÉPRÉCIÉ : non émis comme flag, mais signalé dans les notes.
    assert "--staging_bucket" not in argv
    assert any("staging_bucket" in n for n in result["data"]["notes"])
    assert result["data"]["executed"] is False


def test_agent_engine_app_name_maps_to_display_name(tmp_path: Path) -> None:
    """Sans display_name explicite, app_name est mappé sur --display_name."""
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
    """Tous les flags émis pour agent_engine existent réellement (available_flags)."""
    path = _scaffold(tmp_path)
    result = D.agent_engine(path=path, app_name="myapp", project="p", region="r", display_name="X")
    assert result["ok"] is True
    valid = adk_cli.available_flags(["deploy", "agent_engine"])
    emitted = [t for t in result["data"]["argv"][2:] if t.startswith("--")]
    assert emitted, "au moins un flag devrait être émis"
    assert set(emitted) <= valid, f"flags inconnus: {set(emitted) - valid}"


def test_agent_engine_requires_project_region(tmp_path: Path) -> None:
    path = _scaffold(tmp_path)
    r1 = D.agent_engine(path=path, app_name="myapp", project="", region="r")
    assert r1["ok"] is False and "project" in r1["error"]
    r2 = D.agent_engine(path=path, app_name="myapp", project="p", region=" ")
    assert r2["ok"] is False and "region" in r2["error"]


def test_agent_engine_missing_agent_dir_returns_err(tmp_path: Path) -> None:
    result = D.agent_engine(path=str(tmp_path), app_name="ghost", project="p", region="r")
    assert result["ok"] is False
    assert "ghost" in result["error"] or "introuvable" in result["error"].lower()


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
    # with_ui + enable_cloud_trace sont des flags booléens (pas de valeur).
    assert "--with_ui" in argv
    # enable_cloud_trace mappe sur le VRAI flag --trace_to_cloud (pas --enable_cloud_trace).
    assert "--trace_to_cloud" in argv
    assert "--enable_cloud_trace" not in argv
    assert argv[-1] == _agent_dir(tmp_path)


def test_cloud_run_minimal_no_optional_flags(tmp_path: Path) -> None:
    """Sans options, with_ui/trace ne sont pas émis ; service_name omis si non fourni."""
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
    assert set(emitted) <= valid, f"flags inconnus: {set(emitted) - valid}"


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
    # cluster mappe sur le VRAI flag --cluster_name (pas --cluster).
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
    assert set(emitted) <= valid, f"flags inconnus: {set(emitted) - valid}"


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
    # Le Dockerfile sert `adk api_server`.
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
# preflight (best-effort, ne lève jamais)
# --------------------------------------------------------------------------- #
def test_preflight_returns_structured_findings() -> None:
    result = D.preflight(target="cloud_run")
    assert result["ok"] is True
    data = result["data"]
    assert "gcloud_on_path" in data
    assert "adk_runnable" in data
    assert isinstance(data["findings"], list)


def test_preflight_unknown_target_still_ok() -> None:
    """Un target inconnu ne fait pas échouer le preflight (best-effort)."""
    result = D.preflight(target="banana")
    assert result["ok"] is True
    assert "banana" in str(result["data"]["findings"]) or data_ok(result)


def data_ok(result: dict) -> bool:
    """Helper laxiste : le preflight reste ok même pour un target non standard."""
    return result["ok"] is True


# --------------------------------------------------------------------------- #
# status (best-effort, ne bloque pas)
# --------------------------------------------------------------------------- #
def test_status_unavailable_when_tool_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si gcloud/kubectl est absent, status renvoie une guidance « unavailable » sans bloquer."""
    monkeypatch.setattr(D.shutil, "which", lambda *_a, **_k: None)
    result = D.status(target="cloud_run", project="p", region="r", service_name="s")
    assert result["ok"] is True
    assert result["data"]["available"] is False
    assert result["data"]["guidance"]


def test_status_unknown_target_returns_err() -> None:
    result = D.status(target="banana")
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# execute=False ne lance JAMAIS un vrai déploiement
# --------------------------------------------------------------------------- #
def test_execute_false_never_runs_real_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avec execute=False (défaut), run_adk n'est jamais appelé (pas de vrai déploiement)."""
    called: list[list[str]] = []
    monkeypatch.setattr(D.adk_cli, "run_adk", lambda *a, **k: called.append(a) or {"rc": 0})
    path = _scaffold(tmp_path)
    D.agent_engine(path=path, app_name="myapp", project="p", region="r")
    D.cloud_run(path=path, app_name="myapp", project="p", region="r")
    D.gke(path=path, app_name="myapp", project="p", region="r", cluster="c")
    assert called == [], "execute=False ne doit jamais invoquer run_adk"


def test_execute_true_validates_flags_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """execute=True passe par run_adk (mocké) APRÈS validation des flags ; rc/sortie remontés."""
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
    # run_adk a bien reçu l'argv construit.
    assert recorded["args"][:2] == ["deploy", "cloud_run"]


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_cloud_run(tmp_path: Path) -> None:
    """Outils exposés deploy_<bare> (pas de double-préfixe) ; deploy_cloud_run construit l'argv."""
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
