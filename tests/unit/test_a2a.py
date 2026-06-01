"""Tests du domaine ``a2a`` (P4b — interopérabilité Agent-to-Agent).

Stratégie (honnête sur ce qui est prouvé vs. gaté) :

- ``consume`` est **fonctionnel sans extra** (codegen-only) : on prouve que le sidecar gagne un
  agent ``remote_a2a`` et que ``agent.py`` régénéré contient
  ``from google.adk.agents.remote_a2a_agent import RemoteA2aAgent`` + ``RemoteA2aAgent(...)``. Un
  test **gaté** (``find_spec('a2a')``) lance un subprocess (``-W ignore::DeprecationWarning``) qui
  importe réellement le ``root_agent`` et assert le **vrai type** ``RemoteA2aAgent`` — SKIP si
  l'extra ``a2a`` est absent.
- ``expose`` est **fonctionnel sans extra** : génère ``a2a_app.py`` (``to_a2a(root_agent,
  port=PORT)``), ast valide, et renvoie le chemin + la commande de service ; ``execute=False`` par
  défaut n'exécute rien. Un test **gaté** (``find_spec('a2a')`` ET ``ADK_TOOLKIT_TEST_A2A=1``)
  boote un vrai ``uvicorn`` et GET ``/.well-known/agent-card.json`` puis l'arrête — SKIP bruyamment
  sinon (lent/instable en CI ; nécessite l'extra).
- ``agent_card`` est **gaté** : sans l'extra, renvoie un ``err`` actionnable (testé sans extra) ;
  avec l'extra, construit une vraie ``AgentCard`` (test gaté).

Aucun process/port laissé actif (fixture de nettoyage + stop systématique).
"""

from __future__ import annotations

import ast
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.util import find_spec
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp import adk_cli
from adk_toolkit_mcp.domains import a2a as A2A
from adk_toolkit_mcp.server import build_server

#: Vrai si l'extra ``a2a`` (paquet ``a2a-sdk``) est installé.
_A2A_PRESENT = find_spec("a2a") is not None

#: Flag d'opt-in pour le boot RÉEL d'un serveur a2a uvicorn (lent/instable en CI sinon).
_BOOT_FLAG = "ADK_TOOLKIT_TEST_A2A"


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


def _wait_until(predicate, timeout: float = 30.0, interval: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    result = bool(predicate())
    while not result and time.monotonic() < deadline:
        time.sleep(interval)
        result = bool(predicate())
    return result


def _http_get(url: str, timeout: float = 2.0) -> tuple[int, str] | None:
    """GET ``url`` → ``(status, body)`` ou None si erreur réseau."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - localhost test
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None


def _scaffold_app(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffolde une app ADK minimale (importable SANS clé API) + son sidecar ; renvoie le dir."""
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    (app_dir / "agent.py").write_text(
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(name='{app_name}', model='gemini-2.5-flash', "
        "instruction='You are a test agent.', description='A test agent.')\n",
        encoding="utf-8",
    )
    sidecar = app_dir / ".adk_toolkit"
    sidecar.mkdir(parents=True, exist_ok=True)
    model = {
        "app_name": app_name,
        "root": app_name,
        "agents": [
            {
                "name": app_name,
                "type": "llm",
                "model": "gemini-2.5-flash",
                "instruction": "You are a test agent.",
                "description": "A test agent.",
                "output_key": None,
                "tools": [],
                "sub_agents": [],
            }
        ],
    }
    (sidecar / "agents.json").write_text(json.dumps(model), encoding="utf-8")
    return str(tmp_path)


# --------------------------------------------------------------------------- #
# consume — fonctionnel (codegen-only, no extra)
# --------------------------------------------------------------------------- #
def test_consume_adds_remote_agent_and_regenerates(tmp_path: Path) -> None:
    """``consume`` ajoute un remote_a2a au sidecar et régénère agent.py avec RemoteA2aAgent."""
    path = _scaffold_app(tmp_path)
    result = A2A.consume(
        path=path,
        app_name="myapp",
        name="remote_helper",
        agent_card_url="http://localhost:8002/.well-known/agent-card.json",
    )
    assert result["ok"] is True, result
    assert "remote_helper" in result["data"]["agents"]
    assert result["data"]["remote_agent"]["name"] == "remote_helper"

    agent_txt = (tmp_path / "myapp" / "agent.py").read_text(encoding="utf-8")
    # ⚠️ Import depuis le SOUS-MODULE (pas google.adk.agents) — cf. a2a-mcp-bridge.md.
    assert "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent" in agent_txt
    assert "remote_helper = RemoteA2aAgent(" in agent_txt
    assert 'agent_card="http://localhost:8002/.well-known/agent-card.json"' in agent_txt


def test_consume_then_compose_as_sub_agent(tmp_path: Path) -> None:
    """Le proxy remote_a2a se compose comme sub_agent d'un autre agent (via agents_compose)."""
    from adk_toolkit_mcp.domains import agents as AGENTS

    path = _scaffold_app(tmp_path)
    A2A.consume(
        path=path, app_name="myapp", name="remote_helper", agent_card_url="http://h:8002/a2a"
    )
    composed = AGENTS.compose(
        path=path, app_name="myapp", name="myapp", sub_agents=["remote_helper"]
    )
    assert composed["ok"] is True, composed
    agent_txt = (tmp_path / "myapp" / "agent.py").read_text(encoding="utf-8")
    assert "sub_agents=[remote_helper]" in agent_txt


def test_consume_rejects_empty_url(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = A2A.consume(path=path, app_name="myapp", name="remote_helper", agent_card_url="  ")
    assert result["ok"] is False
    assert "agent_card_url" in result["error"]


def test_consume_rejects_bad_name(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = A2A.consume(
        path=path, app_name="myapp", name="bad name", agent_card_url="http://h/a2a"
    )
    assert result["ok"] is False


def test_consume_rejects_bad_app_name(tmp_path: Path) -> None:
    result = A2A.consume(
        path=str(tmp_path), app_name="bad name", name="r", agent_card_url="http://h/a2a"
    )
    assert result["ok"] is False
    assert "app_name" in result["error"]


def test_consume_corrupted_sidecar_returns_err(tmp_path: Path) -> None:
    """Un sidecar JSON corrompu → err propre (pas d'exception qui remonte)."""
    path = _scaffold_app(tmp_path)
    (tmp_path / "myapp" / ".adk_toolkit" / "agents.json").write_text("{not json", encoding="utf-8")
    result = A2A.consume(path=path, app_name="myapp", name="r", agent_card_url="http://h/a2a")
    assert result["ok"] is False


def test_consume_into_fresh_app_without_sidecar(tmp_path: Path) -> None:
    """consume sur une app SANS sidecar préexistant : crée le modèle + le proxy (pas d'erreur)."""
    app_dir = tmp_path / "fresh"
    app_dir.mkdir()
    result = A2A.consume(
        path=str(tmp_path), app_name="fresh", name="remote_helper", agent_card_url="http://h/a2a"
    )
    assert result["ok"] is True, result
    assert result["data"]["agents"] == ["remote_helper"]
    agent_txt = (app_dir / "agent.py").read_text(encoding="utf-8")
    assert "remote_helper = RemoteA2aAgent(" in agent_txt


@pytest.mark.skipif(
    not _A2A_PRESENT,
    reason="extra 'a2a' absent : la preuve d'instanciation réelle de RemoteA2aAgent est SKIP.",
)
def test_consume_generated_agent_imports_real_remote_a2a_type(tmp_path: Path) -> None:
    """[GATÉ a2a] Le agent.py généré importe et instancie le VRAI RemoteA2aAgent (subprocess).

    Lancé en subprocess avec ``-W ignore::DeprecationWarning`` (RemoteA2aAgent peut toucher des
    surfaces dépréciées). Prouve que le code généré est exécutable AVEC l'extra présent.
    """
    path = _scaffold_app(tmp_path)
    A2A.consume(
        path=path, app_name="myapp", name="remote_helper", agent_card_url="http://h:8002/a2a"
    )
    code = (
        f"import sys; sys.path.insert(0, r'{path}')\n"
        "import importlib; m = importlib.import_module('myapp.agent')\n"
        "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent\n"
        "assert isinstance(m.remote_helper, RemoteA2aAgent), type(m.remote_helper)\n"
        "print('REMOTE_OK', m.remote_helper.name)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "REMOTE_OK remote_helper" in proc.stdout


# --------------------------------------------------------------------------- #
# expose — fonctionnel (codegen-only, no extra) + gaté (live)
# --------------------------------------------------------------------------- #
def test_expose_generates_app_file_no_execute(tmp_path: Path) -> None:
    """``expose`` (execute=False) écrit a2a_app.py (ast valide) + renvoie la commande de service."""
    path = _scaffold_app(tmp_path)
    result = A2A.expose(path=path, app_name="myapp", port=8001, execute=False)
    assert result["ok"] is True, result
    assert result["data"]["executed"] is False
    assert result["data"]["serve_command"] == "uvicorn a2a_app:a2a_app --host localhost --port 8001"
    assert result["data"]["agent_card_path"] == "/.well-known/agent-card.json"

    app_file = tmp_path / "myapp" / "a2a_app.py"
    assert app_file.exists()
    src = app_file.read_text(encoding="utf-8")
    assert "from google.adk.a2a.utils.agent_to_a2a import to_a2a" in src
    assert "from agent import root_agent" in src
    assert "a2a_app = to_a2a(root_agent, port=8001)" in src
    ast.parse(src)  # lève SyntaxError si le rendu est cassé


def test_expose_idempotent(tmp_path: Path) -> None:
    """Réécrire le même a2a_app.py ne le modifie pas (changed=False au 2e appel)."""
    path = _scaffold_app(tmp_path)
    first = A2A.expose(path=path, app_name="myapp", port=8001)
    assert first["data"]["changed"] is True
    second = A2A.expose(path=path, app_name="myapp", port=8001)
    assert second["data"]["changed"] is False


def test_expose_rejects_bad_port(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = A2A.expose(path=path, app_name="myapp", port=0)
    assert result["ok"] is False
    assert "port" in result["error"].lower()


def test_expose_missing_agent_py_returns_err(tmp_path: Path) -> None:
    """Sans agent.py scaffoldé → err actionnable (pas de génération)."""
    (tmp_path / "empty").mkdir()
    result = A2A.expose(path=str(tmp_path), app_name="empty", port=8001)
    assert result["ok"] is False


def test_expose_rejects_bad_app_name(tmp_path: Path) -> None:
    result = A2A.expose(path=str(tmp_path), app_name="bad name", port=8001)
    assert result["ok"] is False
    assert "app_name" in result["error"]


def test_expose_missing_app_dir_returns_err(tmp_path: Path) -> None:
    """Dossier d'app inexistant → err actionnable (jamais de génération hors-cible)."""
    result = A2A.expose(path=str(tmp_path), app_name="ghostapp", port=8001)
    assert result["ok"] is False


def test_expose_execute_without_extra_returns_err(tmp_path: Path) -> None:
    """``execute=True`` sans l'extra a2a → err actionnable (jamais de tentative de boot)."""
    path = _scaffold_app(tmp_path)
    result = A2A.expose(path=path, app_name="myapp", port=_free_port(), execute=True)
    if _A2A_PRESENT:
        # Si l'extra est présent, un process a démarré : on l'arrête et on valide la forme.
        assert result["ok"] is True
        adk_cli.stop_process(result["data"]["key"])
    else:
        assert result["ok"] is False
        assert "a2a" in result["error"].lower()


@pytest.mark.skipif(
    not (_A2A_PRESENT and os.getenv(_BOOT_FLAG) == "1"),
    reason=(
        f"Boot a2a uvicorn RÉEL gaté derrière l'extra 'a2a' ET {_BOOT_FLAG}=1 "
        "(lent/instable en CI). La génération + la commande de service sont testées sans gate."
    ),
)
def test_expose_execute_boots_and_serves_agent_card(tmp_path: Path) -> None:
    """[GATÉ a2a + flag] Boote un vrai uvicorn et GET /.well-known/agent-card.json puis stop."""
    path = _scaffold_app(tmp_path)
    port = _free_port()
    started = A2A.expose(path=path, app_name="myapp", port=port, execute=True)
    assert started["ok"] is True, started
    key = started["data"]["key"]
    url = started["data"]["agent_card_url"]
    try:
        booted = _wait_until(lambda: _http_get(url) is not None, timeout=60.0)
        assert booted, f"a2a app n'a pas répondu sur {url}"
        status, body = _http_get(url)  # type: ignore[misc]
        assert status == 200
        assert "myapp" in body  # la carte porte le nom de l'agent
    finally:
        adk_cli.stop_process(key)
    assert _wait_until(lambda: not adk_cli.process_status(key)["running"], timeout=15.0)


# --------------------------------------------------------------------------- #
# agent_card — gaté (clean err sans extra ; vraie carte avec extra)
# --------------------------------------------------------------------------- #
async def test_agent_card_without_extra_returns_actionable_err(tmp_path: Path) -> None:
    """Sans l'extra a2a, agent_card renvoie un err actionnable (gate gracieuse)."""
    path = _scaffold_app(tmp_path)
    result = await A2A.agent_card(path=path, app_name="myapp")
    if _A2A_PRESENT:
        assert result["ok"] is True
        assert result["data"]["agent_card"]["name"] == "myapp"
    else:
        assert result["ok"] is False
        assert "a2a" in result["error"].lower()


async def test_agent_card_bad_app_name_returns_err(tmp_path: Path) -> None:
    result = await A2A.agent_card(path=str(tmp_path), app_name="bad name")
    assert result["ok"] is False
    assert "app_name" in result["error"]


@pytest.mark.skipif(
    not _A2A_PRESENT,
    reason="extra 'a2a' absent : la construction d'une vraie AgentCard est SKIP.",
)
async def test_agent_card_builds_real_card(tmp_path: Path) -> None:
    """[GATÉ a2a] Construit une vraie AgentCard du root_agent (name/url présents)."""
    path = _scaffold_app(tmp_path)
    result = await A2A.agent_card(path=path, app_name="myapp", port=8001)
    assert result["ok"] is True, result
    card = result["data"]["agent_card"]
    assert card["name"] == "myapp"
    assert card["url"].startswith("http://localhost:8001")


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client (noms exposés + appel consume)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_consume(tmp_path: Path) -> None:
    """Outils exposés a2a_<bare> (pas de double-préfixe) ; a2a_consume round-trip via le client."""
    path = _scaffold_app(tmp_path)
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {"a2a_consume", "a2a_expose", "a2a_agent_card"} <= names
        assert not any(n.startswith("a2a_a2a_") for n in names)

        called = await client.call_tool(
            "a2a_consume",
            {
                "path": path,
                "app_name": "myapp",
                "name": "remote_helper",
                "agent_card_url": "http://h:8002/a2a",
            },
        )
        assert called.data["ok"] is True
        assert "remote_helper" in called.data["data"]["agents"]
