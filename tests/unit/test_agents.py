"""Tests for the ``agents`` domain: sidecar mutation, regeneration, validation, and
**functional proof** that the generated ``agent.py`` instantiates real ADK objects.

The functional proof imports the generated module in a **subprocess** (the uv venv,
``sys.executable``). The subprocess is launched with ``-W ignore::DeprecationWarning`` because the
workflow agents (Sequential/Parallel/Loop) emit a ``DeprecationWarning`` in google-adk 2.1.0 (cf.
``docs/adk-api-notes/agents.md``) — we want to prove the instantiation, not audit ADK's
deprecation here.
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
    set_planner,
    set_root,
)
from adk_toolkit_mcp.project_model import SIDECAR_PATH
from adk_toolkit_mcp.server import build_server


# --------------------------------------------------------------------------- #
# Functional probe: imports the generated module in a subprocess
# --------------------------------------------------------------------------- #
def _probe(project_path: str, app_name: str) -> dict[str, object]:
    """Import ``<app_name>.agent`` in a subprocess and return a summary of root_agent."""
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
    # No root yet -> comment, not an assignment.
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
# create_sequential / parallel / loop: validation of sub_agents
# --------------------------------------------------------------------------- #
def test_create_sequential_requires_existing_sub_agents(tmp_path: Path) -> None:
    res = create_sequential(str(tmp_path), "demo", "pipe", ["missing"])
    assert res["ok"] is False
    assert "not found" in res["error"]


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
    # No file mutated by as_tool (no root assigned).
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
# Cycle at commit -> err (no exception)
# --------------------------------------------------------------------------- #
def test_cycle_via_compose_returns_err(tmp_path: Path) -> None:
    # a -> b then b -> a (compose) creates a cycle; the commit must return err.
    create_llm(str(tmp_path), "demo", "leaf")
    create_sequential(str(tmp_path), "demo", "a", ["leaf"])
    create_sequential(str(tmp_path), "demo", "b", ["a"])
    res = compose(str(tmp_path), "demo", "a", ["b"])
    assert res["ok"] is False
    assert "ycle" in res["error"]


# --------------------------------------------------------------------------- #
# Corrupt sidecar -> err propagated (never an exception) on all the tools
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
    assert "invalid sidecar json" in res["error"].lower()


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
# FUNCTIONAL PROOF — real instantiation of the ADK objects (subprocess)
# --------------------------------------------------------------------------- #
def test_functional_single_llm_root_instantiates(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "solo_app", "main", instruction="Answer.")
    set_root(str(tmp_path), "solo_app", "main")
    info = _probe(str(tmp_path), "solo_app")
    assert info["root_type"] == "LlmAgent"
    assert info["root_name"] == "main"
    assert info["n_sub"] == 0


def test_functional_sequential_with_two_llm_children(tmp_path: Path) -> None:
    # Root = SequentialAgent referencing two LlmAgent children.
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
# Mount wiring — in-memory client + end-to-end functional proof
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_agents_mounted_names_and_functional(tmp_path: Path) -> None:
    mcp = build_server()
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        # Exposed names with a single prefix.
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
            assert expected in tool_names, f"missing: {expected}"
        # No double prefix.
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

    # Outside the client: the generated module must instantiate the real LlmAgent.
    info = _probe(str(tmp_path), "client_app")
    assert info["root_type"] == "LlmAgent"
    assert info["root_name"] == "root"


# --------------------------------------------------------------------------- #
# set_planner — domain behavior + FUNCTIONAL proof (real planner instance)
# --------------------------------------------------------------------------- #
def _probe_planner(project_path: str, app_name: str) -> dict[str, object]:
    """Import ``<app_name>.agent`` in a subprocess and return root_agent.planner info.

    Reports the planner's class name + module (to assert it is the REAL ADK planner type) and,
    for a BuiltInPlanner, the resolved ``thinking_config.thinking_budget``.
    """
    code = (
        "import json,sys;"
        f"sys.path.insert(0, r'{project_path}');"
        f"import {app_name}.agent as m;"
        "ra=m.root_agent;"
        "pl=getattr(ra,'planner',None);"
        "tc=getattr(pl,'thinking_config',None);"
        "print(json.dumps({"
        "'root_type':type(ra).__name__,"
        "'planner_type':type(pl).__name__ if pl is not None else None,"
        "'planner_module':type(pl).__module__ if pl is not None else None,"
        "'thinking_budget':getattr(tc,'thinking_budget',None)}))"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        cwd=project_path,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_set_planner_built_in_writes_render_and_sidecar(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "main", instruction="Think")
    res = set_planner(str(tmp_path), "demo", "main", "built_in", thinking_budget=1024)
    assert res["ok"] is True, res["error"]
    txt = (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")
    assert "from google.adk.planners import BuiltInPlanner" in txt
    assert (
        "planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=1024))" in txt
    )
    # Persisted on the spec.
    spec = get(str(tmp_path), "demo", "main")
    assert spec["data"]["planner"] == {"kind": "built_in", "thinking_budget": 1024}


def test_set_planner_plan_react(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "main")
    res = set_planner(str(tmp_path), "demo", "main", "plan_react")
    assert res["ok"] is True, res["error"]
    txt = (tmp_path / "demo" / "agent.py").read_text(encoding="utf-8")
    assert "from google.adk.planners import PlanReActPlanner" in txt
    assert "planner=PlanReActPlanner()" in txt
    assert get(str(tmp_path), "demo", "main")["data"]["planner"] == {"kind": "plan_react"}


def test_set_planner_rejects_unknown_kind(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "main")
    res = set_planner(str(tmp_path), "demo", "main", "bogus")
    assert res["ok"] is False
    assert "planner kind" in res["error"].lower()


def test_set_planner_rejects_missing_agent(tmp_path: Path) -> None:
    res = set_planner(str(tmp_path), "demo", "ghost", "built_in")
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_set_planner_rejects_non_llm_agent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_sequential(str(tmp_path), "demo", "pipe", ["a"])
    res = set_planner(str(tmp_path), "demo", "pipe", "built_in")
    assert res["ok"] is False
    assert "llm" in res["error"].lower()


def test_set_planner_rejects_nonpositive_budget(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "main")
    res = set_planner(str(tmp_path), "demo", "main", "built_in", thinking_budget=0)
    assert res["ok"] is False


def test_set_planner_rejects_bad_app_name(tmp_path: Path) -> None:
    assert set_planner(str(tmp_path), "1bad", "main", "built_in")["ok"] is False


def test_set_planner_preserves_other_fields(tmp_path: Path) -> None:
    """Attaching a planner must not drop the agent's instruction/output_key."""
    create_llm(str(tmp_path), "demo", "main", instruction="Be precise", output_key="result")
    assert set_planner(str(tmp_path), "demo", "main", "built_in", thinking_budget=64)["ok"]
    spec = get(str(tmp_path), "demo", "main")["data"]
    assert spec["instruction"] == "Be precise"
    assert spec["output_key"] == "result"
    assert spec["planner"] == {"kind": "built_in", "thinking_budget": 64}


def test_functional_built_in_planner_instantiates(tmp_path: Path) -> None:
    """The generated module yields a real BuiltInPlanner with the right thinking budget."""
    create_llm(str(tmp_path), "pl_app", "main", instruction="Think.")
    set_planner(str(tmp_path), "pl_app", "main", "built_in", thinking_budget=2048)
    set_root(str(tmp_path), "pl_app", "main")
    info = _probe_planner(str(tmp_path), "pl_app")
    assert info["root_type"] == "LlmAgent"
    assert info["planner_type"] == "BuiltInPlanner"
    assert info["planner_module"] == "google.adk.planners.built_in_planner"
    assert info["thinking_budget"] == 2048


def test_functional_plan_react_planner_instantiates(tmp_path: Path) -> None:
    """The generated module yields a real PlanReActPlanner."""
    create_llm(str(tmp_path), "pl_app2", "main", instruction="Think.")
    set_planner(str(tmp_path), "pl_app2", "main", "plan_react")
    set_root(str(tmp_path), "pl_app2", "main")
    info = _probe_planner(str(tmp_path), "pl_app2")
    assert info["planner_type"] == "PlanReActPlanner"
    assert info["planner_module"] == "google.adk.planners.plan_re_act_planner"


@pytest.mark.asyncio
async def test_agents_set_planner_mounted(tmp_path: Path) -> None:
    """``agents_set_planner`` is exposed (single prefix) and works via the client."""
    mcp = build_server()
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        assert "agents_set_planner" in tool_names
        assert not any(n.startswith("agents_agents_") for n in tool_names)

        await client.call_tool(
            "agents_create_llm",
            {"path": str(tmp_path), "app_name": "m_app", "name": "main", "instruction": "Hi"},
        )
        res = await client.call_tool(
            "agents_set_planner",
            {
                "path": str(tmp_path),
                "app_name": "m_app",
                "agent_name": "main",
                "kind": "built_in",
                "thinking_budget": 128,
            },
        )
        assert res.data["ok"] is True, res.data["error"]
        rooted = await client.call_tool(
            "agents_set_root",
            {"path": str(tmp_path), "app_name": "m_app", "name": "main"},
        )
        assert rooted.data["ok"] is True

    # Functional: the generated module instantiates the real BuiltInPlanner.
    info = _probe_planner(str(tmp_path), "m_app")
    assert info["planner_type"] == "BuiltInPlanner"
    assert info["thinking_budget"] == 128
