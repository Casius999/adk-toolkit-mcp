"""Tests du domaine ``dev`` (P4a — serveurs de dev longue durée + one-shot run).

Le domaine ``dev`` gère des process ``adk web`` / ``adk api_server`` via le registre de
:mod:`adk_toolkit_mcp.adk_cli`, et lance ``adk run`` en one-shot. On TESTE TOUJOURS :
- la **construction de la commande** (argv) pour web/api_server/run ;
- le cycle de vie via le **registre** (start → status running → logs → stop → not-running),
  prouvé avec le vrai chemin de code (le binaire lancé est ``adk`` ; on n'attend pas qu'il serve).

PREUVE FONCTIONNELLE (best-effort, GATÉE) : booter un vrai ``adk api_server`` sur un port
éphémère et le sonder en HTTP (``/docs``) est LENT/parfois instable en CI. Ce test n'est exécuté
que si ``ADK_TOOLKIT_TEST_API_SERVER=1`` ; sinon il SKIP bruyamment. Quoi qu'il arrive, on ne
laisse aucun process actif ni port lié (fixture de nettoyage + stop systématique).

``adk run`` nécessite des creds modèle pour produire une réponse : le test l'exécute avec un
court timeout et accepte un rc non nul / une sortie d'erreur (renvoyés en DONNÉES, jamais un
hang). On vérifie surtout que la commande est correctement construite et exécutée.
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

#: Flag d'opt-in pour le boot RÉEL d'un api_server (lent/instable en CI sinon).
_BOOT_FLAG = "ADK_TOOLKIT_TEST_API_SERVER"


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Termine tout process géré avant ET après chaque test (aucun orphelin / port lié)."""
    adk_cli.stop_all_processes()
    yield
    adk_cli.stop_all_processes()


def _free_port() -> int:
    """Renvoie un port TCP libre (bind éphémère puis relâché)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _scaffold_agent(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffolde une app ADK minimale (importable SANS clé API) ; renvoie le chemin parent."""
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
    """True si un GET sur ``url`` répond (statut < 500). Toute erreur réseau → False."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost test
            return resp.status < 500
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Construction de commande (web / api_server) — sans booter
# --------------------------------------------------------------------------- #
def test_build_serve_argv_api_server(tmp_path: Path) -> None:
    """``_build_serve_argv`` produit la commande attendue pour api_server (dir + host/port)."""
    path = _scaffold_agent(tmp_path)
    argv, agents_dir = DEV._build_serve_argv(
        "api_server", path, "myapp", port=8123, host="127.0.0.1"
    )
    assert argv[0] == "api_server"
    assert "--host" in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8123"
    # app_name fourni → AGENTS_DIR pointe sur le dossier de l'app (positionnel final).
    assert argv[-1] == str(Path(path) / "myapp")
    assert agents_dir == str(Path(path) / "myapp")


def test_build_serve_argv_web_without_app_name(tmp_path: Path) -> None:
    """Sans app_name, AGENTS_DIR = le dossier parent (répertoire d'agents)."""
    path = _scaffold_agent(tmp_path)
    argv, agents_dir = DEV._build_serve_argv("web", path, None, port=8000, host="0.0.0.0")
    assert argv[0] == "web"
    assert agents_dir == path
    assert argv[-1] == path


def test_serve_argv_flags_valid_against_real_help(tmp_path: Path) -> None:
    """Les flags émis pour web/api_server existent réellement (available_flags)."""
    path = _scaffold_agent(tmp_path)
    for kind in ("web", "api_server"):
        argv, _ = DEV._build_serve_argv(kind, path, "myapp", port=8000, host="127.0.0.1")
        valid = adk_cli.available_flags([kind])
        emitted = {t for t in argv if t.startswith("--")}
        assert emitted <= valid, f"{kind}: flags inconnus {emitted - valid}"


# --------------------------------------------------------------------------- #
# Validation des entrées
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
# Cycle de vie via le registre — prouvé avec le vrai chemin de code (start/stop)
# --------------------------------------------------------------------------- #
async def test_dev_start_status_stop_lifecycle(tmp_path: Path) -> None:
    """api_server démarre (process enregistré + running), status le voit, stop le termine.

    On NE dépend PAS de la disponibilité HTTP du serveur (lent) — on prouve le contrat du
    registre via le vrai démarrage du process ``adk api_server`` puis son arrêt. Le binaire est
    bien ``adk`` (preuve dans l'argv enregistré).
    """
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    started = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert started["ok"] is True, started
    key = started["data"]["key"]
    assert started["data"]["pid"] > 0
    assert started["data"]["port"] == port
    assert "api_server" in started["data"]["url"] or started["data"]["url"].startswith("http")

    # status voit le process (running au moins juste après le lancement).
    status = await DEV.status(key=key)
    assert status["ok"] is True
    assert status["data"]["found"] is True

    # logs accessibles (le fichier existe même si vide au tout début).
    logs = await DEV.logs(key=key, tail=20)
    assert logs["ok"] is True
    assert "lines" in logs["data"]

    # stop termine effectivement le process.
    stopped = await DEV.stop(key=key)
    assert stopped["ok"] is True
    assert stopped["data"]["found"] is True
    assert _wait_until(lambda: DEV._status_running(key) is False, timeout=15.0)


async def test_dev_double_start_same_key_returns_err(tmp_path: Path) -> None:
    """Démarrer deux fois la même app/port (process vivant) → err propre (pas d'exception)."""
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    first = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert first["ok"] is True
    second = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert second["ok"] is False
    assert "déjà" in second["error"] or "already" in second["error"].lower()
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
# FUNCTIONAL (gaté) — boot RÉEL d'un api_server + sonde HTTP
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.getenv(_BOOT_FLAG) != "1",
    reason=(
        f"Boot api_server RÉEL gaté derrière {_BOOT_FLAG}=1 "
        "(lent/instable en CI). Le registre + la construction de commande sont testés sans gate."
    ),
)
async def test_api_server_boots_and_serves_http(tmp_path: Path) -> None:
    """[GATÉ {flag}=1] Boote un vrai adk api_server sur un port éphémère et sonde /docs en HTTP."""
    path = _scaffold_agent(tmp_path)
    port = _free_port()
    started = await DEV.api_server(path=path, app_name="myapp", port=port)
    assert started["ok"] is True, started
    key = started["data"]["key"]
    url = f"http://127.0.0.1:{port}/docs"
    try:
        booted = _wait_until(lambda: _http_ok(url), timeout=60.0)
        logs = await DEV.logs(key=key, tail=50)
        assert booted, f"api_server n'a pas répondu sur {url}. logs={logs['data']['lines']}"
    finally:
        await DEV.stop(key=key)
    assert _wait_until(lambda: DEV._status_running(key) is False, timeout=15.0)


# --------------------------------------------------------------------------- #
# run (one-shot) — construit/execute, jamais de hang
# --------------------------------------------------------------------------- #
def test_run_argv_construction(tmp_path: Path) -> None:
    """``_build_run_argv`` met le message en QUERY positionnel (pas un flag) après AGENT."""
    path = _scaffold_agent(tmp_path)
    argv = DEV._build_run_argv(path, "myapp", "hello there")
    assert argv[0] == "run"
    # AGENT (dossier d'app) puis QUERY (message).
    assert argv[1] == str(Path(path) / "myapp")
    assert argv[-1] == "hello there"


async def test_run_without_message_returns_guidance(tmp_path: Path) -> None:
    """Sans message, run renvoie une guidance (le mode interactif bloquerait) — pas d'exécution."""
    path = _scaffold_agent(tmp_path)
    result = await DEV.run(path=path, app_name="myapp", message=None)
    assert result["ok"] is True
    assert result["data"]["executed"] is False
    assert result["data"]["guidance"]


async def test_run_with_message_executes_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avec un message, run invoque adk run (mocké) et remonte rc/sortie ; jamais de hang."""
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
    # Un timeout est passé (jamais d'attente infinie).
    assert recorded["timeout"] is not None


async def test_run_missing_agent_dir_returns_err(tmp_path: Path) -> None:
    result = await DEV.run(path=str(tmp_path), app_name="ghost", message="hi")
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client (noms exposés + flux registre)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_registry_flow(tmp_path: Path) -> None:
    """Outils exposés dev_<bare> (pas de double-préfixe) ; flux start→status→stop via le client."""
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
