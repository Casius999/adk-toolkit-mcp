"""Tests du domaine ``agents`` : mutation du sidecar, régénération, validation, et
**preuve fonctionnelle** que ``agent.py`` généré instancie de vrais objets ADK.

La preuve fonctionnelle importe le module généré dans un **subprocess** (le venv uv,
``sys.executable``). Le subprocess est lancé avec ``-W ignore::DeprecationWarning`` car
les agents workflow (Sequential/Parallel/Loop) émettent une ``DeprecationWarning`` en
google-adk 2.1.0 (cf. ``docs/adk-api-notes/agents.md``) — on veut prouver l'instanciation,
pas auditer la dépréciation d'ADK ici.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains.agents import (
    as_tool,
    compose,
    create_custom,
    create_llm,
    create_loop,
    create_parallel,
    create_sequential,
    get,
    list_agents,
    set_root,
)
from adk_toolkit_mcp.project_model import SIDECAR_PATH
from adk_toolkit_mcp.server import build_server


# --------------------------------------------------------------------------- #
# Probe fonctionnelle : importe le module généré dans un subprocess
# --------------------------------------------------------------------------- #
def _probe(project_path: str, app_name: str) -> dict[str, object]:
    """Importe ``<app_name>.agent`` dans un subprocess et renvoie un résumé de root_agent."""
    code = (
        "import json,sys;"
        f"sys.path.insert(0, r'{project_path}');"
        f"import {app_name}.agent as m;"
        "ra=m.root_agent;"
        "print(json.dumps({'root_type':type(ra).__name__,'root_name':getattr(ra,'name',None),"
        "'n_sub':len(getattr(ra,'sub_agents',[]) or []),"
        "'sub_types':[type(s).__name__ for s in (getattr(ra,'sub_agents',[]) or [])],"
        "'sub_names':[getattr(s,'name',None) for s in (getattr(ra,'sub_agents',[]) or [])]}))"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        cwd=project_path,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# create_llm
# --------------------------------------------------------------------------- #
def test_create_llm_writes_sidecar_and_agent_py(tmp_path: Path) -> None:
    res = create_llm(str(tmp_path), "demo", "greeter", instruction="Say hi")
    assert res["ok"] is True, res["error"]
    app = tmp_path / "demo"
    assert (app / SIDECAR_PATH).exists()
    agent_txt = (app / "agent.py").read_text(encoding="utf-8")
    assert "greeter = LlmAgent(" in agent_txt
    assert 'instruction="Say hi"' in agent_txt
    # Pas encore de racine -> commentaire, pas d'assignation.
    assert "root_agent = greeter" not in agent_txt


def test_create_llm_rejects_bad_name(tmp_path: Path) -> None:
    res = create_llm(str(tmp_path), "demo", "bad name!")
    assert res["ok"] is False
    assert res["error"]


def test_create_llm_rejects_bad_app_name(tmp_path: Path) -> None:
    res = create_llm(str(tmp_path), "1bad", "agent")
    assert res["ok"] is False


def test_create_llm_rejects_empty_model(tmp_path: Path) -> None:
    res = create_llm(str(tmp_path), "demo", "a", model="   ")
    assert res["ok"] is False


def test_create_llm_update_is_idempotent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a", instruction="v1")
    again = create_llm(str(tmp_path), "demo", "a", instruction="v1")
    assert again["ok"] is True
    assert again["data"]["changed"] is False


# --------------------------------------------------------------------------- #
# create_sequential / parallel / loop : validation des sub_agents
# --------------------------------------------------------------------------- #
def test_create_sequential_requires_existing_sub_agents(tmp_path: Path) -> None:
    res = create_sequential(str(tmp_path), "demo", "pipe", ["missing"])
    assert res["ok"] is False
    assert "introuvable" in res["error"]


def test_create_sequential_succeeds_when_children_exist(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_llm(str(tmp_path), "demo", "b")
    res = create_sequential(str(tmp_path), "demo", "pipe", ["a", "b"], description="Pipeline")
    assert res["ok"] is True, res["error"]
    agent_txt = (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")
    assert "pipe = SequentialAgent(" in agent_txt
    assert "sub_agents=[a, b]" in agent_txt


def test_create_parallel_succeeds(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_llm(str(tmp_path), "demo", "b")
    res = create_parallel(str(tmp_path), "demo", "fan", ["a", "b"])
    assert res["ok"] is True
    assert "fan = ParallelAgent(" in (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")


def test_create_loop_rejects_nonpositive_iterations(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    res = create_loop(str(tmp_path), "demo", "lp", ["a"], max_iterations=0)
    assert res["ok"] is False


def test_create_loop_succeeds(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    res = create_loop(str(tmp_path), "demo", "lp", ["a"], max_iterations=4)
    assert res["ok"] is True
    txt = (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")
    assert "lp = LoopAgent(" in txt
    assert "max_iterations=4" in txt


# --------------------------------------------------------------------------- #
# create_custom
# --------------------------------------------------------------------------- #
def test_create_custom_emits_subclass(tmp_path: Path) -> None:
    res = create_custom(str(tmp_path), "demo", "router", description="Routes")
    assert res["ok"] is True
    txt = (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")
    assert "class RouterAgent(BaseAgent):" in txt
    assert "router = RouterAgent(" in txt


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #
def test_compose_replaces_sub_agents(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_llm(str(tmp_path), "demo", "b")
    create_llm(str(tmp_path), "demo", "c")
    create_sequential(str(tmp_path), "demo", "pipe", ["a"])
    res = compose(str(tmp_path), "demo", "pipe", ["b", "c"])
    assert res["ok"] is True, res["error"]
    spec = get(str(tmp_path), "demo", "pipe")
    assert spec["data"]["sub_agents"] == ["b", "c"]


def test_compose_rejects_missing_agent(tmp_path: Path) -> None:
    res = compose(str(tmp_path), "demo", "ghost", ["a"])
    assert res["ok"] is False


def test_compose_rejects_missing_children(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_sequential(str(tmp_path), "demo", "pipe", ["a"])
    res = compose(str(tmp_path), "demo", "pipe", ["nope"])
    assert res["ok"] is False


def test_compose_rejects_self_reference(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_sequential(str(tmp_path), "demo", "pipe", ["a"])
    res = compose(str(tmp_path), "demo", "pipe", ["pipe"])
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# set_root
# --------------------------------------------------------------------------- #
def test_set_root_rejects_missing(tmp_path: Path) -> None:
    res = set_root(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False


def test_set_root_writes_assignment(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a", instruction="Hi")
    res = set_root(str(tmp_path), "demo", "a")
    assert res["ok"] is True
    assert "root_agent = a" in (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# as_tool / list / get
# --------------------------------------------------------------------------- #
def test_as_tool_returns_snippet(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "helper")
    res = as_tool(str(tmp_path), "demo", "helper")
    assert res["ok"] is True
    assert "AgentTool(agent=helper)" in res["data"]["snippet"]
    assert res["data"]["import"] == "from google.adk.tools import AgentTool"
    # Aucun fichier muté par as_tool (pas de root assigné).
    assert "root_agent = helper" not in (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")


def test_as_tool_rejects_missing_agent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    res = as_tool(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False


def test_list_and_get(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a", instruction="A")
    create_llm(str(tmp_path), "demo", "b")
    create_sequential(str(tmp_path), "demo", "pipe", ["a", "b"])
    set_root(str(tmp_path), "demo", "pipe")

    listing = list_agents(str(tmp_path), "demo")
    assert listing["ok"] is True
    assert listing["data"]["root"] == "pipe"
    names = {a["name"]: a["type"] for a in listing["data"]["agents"]}
    assert names == {"a": "llm", "b": "llm", "pipe": "sequential"}

    got = get(str(tmp_path), "demo", "a")
    assert got["ok"] is True
    assert got["data"]["instruction"] == "A"


def test_get_missing_errors(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    res = get(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False


def test_list_on_fresh_app_is_empty(tmp_path: Path) -> None:
    res = list_agents(str(tmp_path), "demo")
    assert res["ok"] is True
    assert res["data"]["agents"] == []
    assert res["data"]["root"] is None


# --------------------------------------------------------------------------- #
# Cycle au commit -> err (pas d'exception)
# --------------------------------------------------------------------------- #
def test_cycle_via_compose_returns_err(tmp_path: Path) -> None:
    # a -> b puis b -> a (compose) crée un cycle ; le commit doit renvoyer err.
    create_llm(str(tmp_path), "demo", "leaf")
    create_sequential(str(tmp_path), "demo", "a", ["leaf"])
    create_sequential(str(tmp_path), "demo", "b", ["a"])
    res = compose(str(tmp_path), "demo", "a", ["b"])
    assert res["ok"] is False
    assert "ycle" in res["error"]


# --------------------------------------------------------------------------- #
# Sidecar corrompu -> err propagé (jamais d'exception) sur tous les outils
# --------------------------------------------------------------------------- #
def _corrupt_sidecar(tmp_path: Path, app_name: str = "demo") -> str:
    app = tmp_path / app_name / SIDECAR_PATH
    app.parent.mkdir(parents=True, exist_ok=True)
    app.write_text("{ not valid json ]", encoding="utf-8")
    return str(tmp_path)


def test_corrupt_sidecar_create_llm_returns_err(tmp_path: Path) -> None:
    root = _corrupt_sidecar(tmp_path)
    res = create_llm(root, "demo", "a")
    assert res["ok"] is False
    assert "JSON invalide" in res["error"]


def test_corrupt_sidecar_read_tools_return_err(tmp_path: Path) -> None:
    root = _corrupt_sidecar(tmp_path)
    for res in (
        list_agents(root, "demo"),
        get(root, "demo", "a"),
        compose(root, "demo", "a", ["b"]),
        set_root(root, "demo", "a"),
        as_tool(root, "demo", "a"),
    ):
        assert res["ok"] is False
        assert res["error"]


# --------------------------------------------------------------------------- #
# PREUVE FONCTIONNELLE — instanciation réelle des objets ADK (subprocess)
# --------------------------------------------------------------------------- #
def test_functional_single_llm_root_instantiates(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "solo_app", "main", instruction="Answer.")
    set_root(str(tmp_path), "solo_app", "main")
    info = _probe(str(tmp_path), "solo_app")
    assert info["root_type"] == "LlmAgent"
    assert info["root_name"] == "main"
    assert info["n_sub"] == 0


def test_functional_sequential_with_two_llm_children(tmp_path: Path) -> None:
    # Racine = SequentialAgent référençant deux enfants LlmAgent.
    create_llm(str(tmp_path), "pipe_app", "writer", instruction="Write.")
    create_llm(str(tmp_path), "pipe_app", "reviewer", instruction="Review.")
    create_sequential(str(tmp_path), "pipe_app", "pipeline", ["writer", "reviewer"])
    set_root(str(tmp_path), "pipe_app", "pipeline")

    info = _probe(str(tmp_path), "pipe_app")
    assert info["root_type"] == "SequentialAgent"
    assert info["root_name"] == "pipeline"
    assert info["n_sub"] == 2
    assert info["sub_types"] == ["LlmAgent", "LlmAgent"]
    assert info["sub_names"] == ["writer", "reviewer"]


def test_functional_custom_agent_instantiates(tmp_path: Path) -> None:
    create_custom(str(tmp_path), "cust_app", "dispatcher", description="Dispatch")
    set_root(str(tmp_path), "cust_app", "dispatcher")
    info = _probe(str(tmp_path), "cust_app")
    assert info["root_type"] == "DispatcherAgent"
    assert info["root_name"] == "dispatcher"


def test_functional_loop_agent_instantiates(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "loop_app", "step", instruction="Step.")
    create_loop(str(tmp_path), "loop_app", "looper", ["step"], max_iterations=2)
    set_root(str(tmp_path), "loop_app", "looper")
    info = _probe(str(tmp_path), "loop_app")
    assert info["root_type"] == "LoopAgent"
    assert info["n_sub"] == 1


# --------------------------------------------------------------------------- #
# Mount wiring — client in-memory + preuve fonctionnelle bout-en-bout
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_agents_mounted_names_and_functional(tmp_path: Path) -> None:
    mcp = build_server()
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        # Noms exposés à préfixe unique.
        for expected in (
            "agents_create_llm",
            "agents_create_sequential",
            "agents_create_parallel",
            "agents_create_loop",
            "agents_create_custom",
            "agents_compose",
            "agents_as_tool",
            "agents_set_root",
            "agents_list",
            "agents_get",
        ):
            assert expected in tool_names, f"manquant: {expected}"
        # Pas de double préfixe.
        assert not any(n.startswith("agents_agents_") for n in tool_names)

        created = await client.call_tool(
            "agents_create_llm",
            {"path": str(tmp_path), "app_name": "client_app", "name": "root", "instruction": "Hi"},
        )
        assert created.data["ok"] is True
        rooted = await client.call_tool(
            "agents_set_root",
            {"path": str(tmp_path), "app_name": "client_app", "name": "root"},
        )
        assert rooted.data["ok"] is True

    # Hors client : le module généré doit instancier le vrai LlmAgent.
    info = _probe(str(tmp_path), "client_app")
    assert info["root_type"] == "LlmAgent"
    assert info["root_name"] == "root"
