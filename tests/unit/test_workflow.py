"""Tests for the ``workflow`` domain: sidecar mutation, regeneration, validation, and a
**functional proof** that the generated ``agent.py`` instantiates AND runs a real
``google.adk.workflow.Workflow`` offline.

The functional proof imports the generated module in a **subprocess** (the uv venv,
``sys.executable``) — the same pattern as ``test_agents.py``. It instantiates the real
``Workflow`` object, asserts the compiled graph structure, and runs it offline via
``InMemoryRunner(node=...)``:

- a **pure function-node** workflow runs with NO model (deterministic);
- an **agent-node** workflow runs with the node agents' models swapped to ``FakeLlm`` (the
  ``Workflow`` deep-copies its nodes into ``graph.nodes``, so the swap is applied on the graph
  node objects — cf. ``docs/adk-api-notes/workflow.md``).

The subprocess is launched with ``-W ignore::DeprecationWarning`` (consistent with the rest of
the suite; the workflow engine itself emits no deprecation, but FakeLlm-backed agents may pull in
deprecated paths during a run).
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains.workflow import (
    add_edge,
    add_node,
    create,
    get,
    list_workflows,
    set_entry,
    set_root,
)
from adk_toolkit_mcp.project_model import SIDECAR_PATH
from adk_toolkit_mcp.server import build_server

#: Absolute path of the test package dir, so a subprocess can import ``fake_llm`` regardless of
#: its own cwd (the generated app lives in a tmp dir).
_FAKE_LLM_DIR = str(Path(__file__).parent.resolve())


# --------------------------------------------------------------------------- #
# ruff helpers (same proven pattern as test_project_model.py)
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    """Locate the ruff executable in the current environment (venv or PATH)."""
    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _assert_codegen_clean(src: str, tmp_path: Path, label: str) -> None:
    """Assert the generated source is ``ast.parse`` + ``ruff format --check`` + isort clean."""
    ast.parse(src)  # raises SyntaxError if the generated code is invalid

    gen_file = tmp_path / f"{label}.py"
    gen_file.write_text(src, encoding="utf-8")
    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff not found in the environment — format/isort checks ignored")

    fmt = subprocess.run([ruff, "format", "--check", str(gen_file)], capture_output=True, text=True)
    assert fmt.returncode == 0, f"ruff format --check failed ({label}):\n{fmt.stdout}\n{src}"

    isort = subprocess.run(
        [ruff, "check", "--select", "I", str(gen_file)], capture_output=True, text=True
    )
    assert isort.returncode == 0, f"ruff check --select I failed ({label}):\n{isort.stdout}\n{src}"


def _agent_py(tmp_path: Path, app: str) -> str:
    return (tmp_path / app / "agent.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Functional probe: instantiate + RUN the real Workflow in a subprocess
# --------------------------------------------------------------------------- #
def _probe(
    project_path: str, app_name: str, *, swap_fake_llm: bool, message: str
) -> dict[str, object]:
    """Import ``<app>.agent`` in a subprocess, instantiate root_agent, RUN it offline.

    Returns ``{root_type, root_name, node_names, terminal, texts}`` where ``texts`` is the list
    of ``[author, text]`` emitted by ``run_async``. If ``swap_fake_llm`` is True, every
    ``LlmAgent`` in the compiled ``graph.nodes`` has its ``model`` swapped to a ``FakeLlm`` that
    echoes ``<node_name>-out`` (so the run is fully offline, no API key).
    """
    swap = (
        "from fake_llm import FakeLlm\n"
        "for _n in wf.graph.nodes:\n"
        "    if type(_n).__name__ == 'LlmAgent':\n"
        "        _n.model = FakeLlm(model='fake', answer=_n.name + '-out')\n"
        if swap_fake_llm
        else ""
    )
    code = (
        "import asyncio, json, sys\n"
        f"sys.path.insert(0, {project_path!r})\n"
        f"sys.path.insert(0, {_FAKE_LLM_DIR!r})\n"
        f"import {app_name}.agent as m\n"
        "from google.adk.runners import InMemoryRunner\n"
        "from google.genai import types\n"
        "wf = m.root_agent\n"
        f"{swap}"
        "info = {'root_type': type(wf).__name__, 'root_name': getattr(wf, 'name', None),\n"
        "        'node_names': [n.name for n in wf.graph.nodes],\n"
        "        'terminal': sorted(wf.graph._terminal_node_names)}\n"
        "async def go():\n"
        f"    r = InMemoryRunner(node=wf, app_name={app_name!r})\n"
        f"    s = await r.session_service.create_session(app_name={app_name!r}, user_id='u')\n"
        "    texts = []\n"
        f"    msg = types.Content(role='user', parts=[types.Part.from_text(text={message!r})])\n"
        "    async for ev in r.run_async(user_id='u', session_id=s.id, new_message=msg):\n"
        "        if ev.content and ev.content.parts:\n"
        "            for p in ev.content.parts:\n"
        "                if p.text: texts.append([ev.author, p.text])\n"
        "    info['texts'] = texts\n"
        "    print(json.dumps(info))\n"
        "asyncio.run(go())\n"
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
# create
# --------------------------------------------------------------------------- #
def test_create_writes_sidecar_and_placeholder(tmp_path: Path) -> None:
    res = create(str(tmp_path), "demo", "flow", description="My flow")
    assert res["ok"] is True, res["error"]
    app = tmp_path / "demo"
    assert (app / SIDECAR_PATH).exists()
    # An empty workflow is incomplete: it renders a placeholder comment (no Workflow() call yet)
    # so agent.py stays importable. The workflow IS persisted in the sidecar.
    txt = _agent_py(tmp_path, "demo")
    assert "flow = Workflow(" not in txt
    assert "not fully wired yet" in txt
    assert list_workflows(str(tmp_path), "demo")["data"]["workflows"][0]["name"] == "flow"


def test_create_rejects_bad_name(tmp_path: Path) -> None:
    assert create(str(tmp_path), "demo", "bad name!")["ok"] is False


def test_create_rejects_bad_app_name(tmp_path: Path) -> None:
    assert create(str(tmp_path), "1bad", "flow")["ok"] is False


# --------------------------------------------------------------------------- #
# add_node
# --------------------------------------------------------------------------- #
def test_add_function_node_renders_decorated_def(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    res = add_node(
        str(tmp_path),
        "demo",
        "flow",
        "step",
        "function",
        docstring="Do a step.",
        body='return {"x": 1}',
    )
    assert res["ok"] is True, res["error"]
    txt = _agent_py(tmp_path, "demo")
    assert "@node" in txt
    assert "def step(ctx, node_input) -> dict:" in txt
    assert 'return {"x": 1}' in txt


def test_add_join_node_renders_joinnode(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    res = add_node(str(tmp_path), "demo", "flow", "merge", "join")
    assert res["ok"] is True, res["error"]
    txt = _agent_py(tmp_path, "demo")
    assert 'merge = JoinNode(name="merge")' in txt


def test_add_agent_node_requires_existing_agent(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    res = add_node(str(tmp_path), "demo", "flow", "writer", "agent")
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_add_agent_node_succeeds_when_agent_exists(tmp_path: Path) -> None:
    # The agent must exist in the model: scaffold it via the agents domain.
    from adk_toolkit_mcp.domains.agents import create_llm

    create_llm(str(tmp_path), "demo", "writer", instruction="Write.")
    create(str(tmp_path), "demo", "flow")
    res = add_node(str(tmp_path), "demo", "flow", "writer", "agent")
    assert res["ok"] is True, res["error"]


def test_add_node_rejects_unknown_kind(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    res = add_node(str(tmp_path), "demo", "flow", "n", "bogus")
    assert res["ok"] is False
    assert "Unknown node kind" in res["error"]


def test_add_node_rejects_missing_workflow(tmp_path: Path) -> None:
    res = add_node(str(tmp_path), "demo", "ghost", "n", "function")
    assert res["ok"] is False
    assert "Workflow not found" in res["error"]


def test_add_node_rejects_reserved_start_name(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    res = add_node(str(tmp_path), "demo", "flow", "START", "function")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# add_edge / set_entry
# --------------------------------------------------------------------------- #
def test_add_edge_unconditional(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    add_node(str(tmp_path), "demo", "flow", "b", "function")
    set_entry(str(tmp_path), "demo", "flow", "a")
    res = add_edge(str(tmp_path), "demo", "flow", "a", "b")
    assert res["ok"] is True, res["error"]
    txt = _agent_py(tmp_path, "demo")
    assert "(START, a)" in txt
    assert "(a, b)" in txt


def test_add_edge_conditional_groups_by_source(tmp_path: Path) -> None:
    # A complete branching graph (router -> {left, right} -> merge) so the Workflow() is rendered.
    create(str(tmp_path), "demo", "flow")
    add_node(
        str(tmp_path), "demo", "flow", "router", "function", returns="str", body='return "left"'
    )
    add_node(str(tmp_path), "demo", "flow", "left", "function")
    add_node(str(tmp_path), "demo", "flow", "right", "function")
    add_node(str(tmp_path), "demo", "flow", "merge", "join")
    set_entry(str(tmp_path), "demo", "flow", "router")
    add_edge(str(tmp_path), "demo", "flow", "router", "left", route="left")
    add_edge(str(tmp_path), "demo", "flow", "router", "right", route="right")
    add_edge(str(tmp_path), "demo", "flow", "left", "merge")
    add_edge(str(tmp_path), "demo", "flow", "right", "merge")
    res = set_root(str(tmp_path), "demo", "flow")
    assert res["ok"] is True, res["error"]
    txt = _agent_py(tmp_path, "demo")
    # Conditional edges sharing a source collapse into a single dict tuple.
    assert '(router, {"left": left, "right": right})' in txt


def test_add_edge_rejects_target_start(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    res = add_edge(str(tmp_path), "demo", "flow", "a", "START")
    assert res["ok"] is False


def test_add_edge_rejects_missing_node(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    res = add_edge(str(tmp_path), "demo", "flow", "a", "ghost")
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_add_edge_rejects_self_loop(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    res = add_edge(str(tmp_path), "demo", "flow", "a", "a")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# Graph validation surfaced as err (no exception) at set_root / commit
# --------------------------------------------------------------------------- #
def test_set_root_rejects_unreachable_node(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    add_node(str(tmp_path), "demo", "flow", "orphan", "function")  # never wired from START
    set_entry(str(tmp_path), "demo", "flow", "a")
    res = set_root(str(tmp_path), "demo", "flow")
    assert res["ok"] is False
    assert "Unreachable" in res["error"]


def test_commit_rejects_unconditional_cycle(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    add_node(str(tmp_path), "demo", "flow", "b", "function")
    set_entry(str(tmp_path), "demo", "flow", "a")
    add_edge(str(tmp_path), "demo", "flow", "a", "b")
    # b -> a with no route => unconditional cycle, rejected at commit.
    res = add_edge(str(tmp_path), "demo", "flow", "b", "a")
    assert res["ok"] is False
    assert "cycle" in res["error"].lower()


def test_routed_cycle_is_allowed(tmp_path: Path) -> None:
    # A ReAct-style loop: a conditional edge closes the cycle (allowed).
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "reason", "function")
    add_node(
        str(tmp_path), "demo", "flow", "act", "function", returns="str", body='return "continue"'
    )
    add_node(str(tmp_path), "demo", "flow", "finish", "function")
    set_entry(str(tmp_path), "demo", "flow", "reason")
    add_edge(str(tmp_path), "demo", "flow", "reason", "act")
    add_edge(str(tmp_path), "demo", "flow", "act", "reason", route="continue")
    res = add_edge(str(tmp_path), "demo", "flow", "act", "finish", route="stop")
    assert res["ok"] is True, res["error"]


def test_set_root_rejects_multiple_terminals(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    add_node(str(tmp_path), "demo", "flow", "b", "function")
    add_node(str(tmp_path), "demo", "flow", "c", "function")
    set_entry(str(tmp_path), "demo", "flow", "a")
    add_edge(str(tmp_path), "demo", "flow", "a", "b")
    add_edge(str(tmp_path), "demo", "flow", "a", "c")  # b and c are both terminal
    res = set_root(str(tmp_path), "demo", "flow")
    assert res["ok"] is False
    assert "terminal" in res["error"].lower()


def test_set_root_rejects_missing_workflow(tmp_path: Path) -> None:
    res = set_root(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# set_root writes root_agent = <workflow>
# --------------------------------------------------------------------------- #
def test_set_root_writes_workflow_root(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    set_entry(str(tmp_path), "demo", "flow", "a")
    res = set_root(str(tmp_path), "demo", "flow")
    assert res["ok"] is True, res["error"]
    assert res["data"]["root_kind"] == "workflow"
    assert "root_agent = flow" in _agent_py(tmp_path, "demo")


# --------------------------------------------------------------------------- #
# list / get
# --------------------------------------------------------------------------- #
def test_list_and_get(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "flow")
    add_node(str(tmp_path), "demo", "flow", "a", "function")
    set_entry(str(tmp_path), "demo", "flow", "a")

    listing = list_workflows(str(tmp_path), "demo")
    assert listing["ok"] is True
    assert listing["data"]["workflows"] == [{"name": "flow", "nodes": 1, "edges": 1}]

    got = get(str(tmp_path), "demo", "flow")
    assert got["ok"] is True
    assert got["data"]["name"] == "flow"
    assert [n["name"] for n in got["data"]["nodes"]] == ["a"]


def test_get_missing_errors(tmp_path: Path) -> None:
    res = get(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False


def test_list_on_fresh_app_is_empty(tmp_path: Path) -> None:
    res = list_workflows(str(tmp_path), "demo")
    assert res["ok"] is True
    assert res["data"]["workflows"] == []


# --------------------------------------------------------------------------- #
# Corrupt sidecar -> err propagated (never an exception)
# --------------------------------------------------------------------------- #
def _corrupt_sidecar(tmp_path: Path, app_name: str = "demo") -> str:
    app = tmp_path / app_name / SIDECAR_PATH
    app.parent.mkdir(parents=True, exist_ok=True)
    app.write_text("{ not valid json ]", encoding="utf-8")
    return str(tmp_path)


def test_corrupt_sidecar_returns_err(tmp_path: Path) -> None:
    root = _corrupt_sidecar(tmp_path)
    for res in (
        create(root, "demo", "flow"),
        add_node(root, "demo", "flow", "a", "function"),
        add_edge(root, "demo", "flow", "a", "b"),
        set_entry(root, "demo", "flow", "a"),
        set_root(root, "demo", "flow"),
        list_workflows(root, "demo"),
        get(root, "demo", "flow"),
    ):
        assert res["ok"] is False
        assert res["error"]


# --------------------------------------------------------------------------- #
# Codegen cleanliness (ast.parse + ruff format --check + ruff check --select I)
# --------------------------------------------------------------------------- #
def test_generated_function_workflow_is_codegen_clean(tmp_path: Path) -> None:
    create(str(tmp_path), "clean_app", "triage")
    add_node(
        str(tmp_path),
        "clean_app",
        "triage",
        "classify",
        "function",
        returns="str",
        body='return "urgent"',
    )
    add_node(str(tmp_path), "clean_app", "triage", "urgent", "function")
    add_node(str(tmp_path), "clean_app", "triage", "normal", "function")
    set_entry(str(tmp_path), "clean_app", "triage", "classify")
    add_edge(str(tmp_path), "clean_app", "triage", "classify", "urgent", route="urgent")
    add_edge(str(tmp_path), "clean_app", "triage", "classify", "normal", route="normal")
    set_root(str(tmp_path), "clean_app", "triage")
    _assert_codegen_clean(_agent_py(tmp_path, "clean_app"), tmp_path, "fn_workflow")


def test_generated_agent_workflow_is_codegen_clean(tmp_path: Path) -> None:
    from adk_toolkit_mcp.domains.agents import create_llm

    create_llm(str(tmp_path), "clean2", "writer", instruction="Write.")
    create_llm(str(tmp_path), "clean2", "reviewer", instruction="Review.")
    create(str(tmp_path), "clean2", "editorial")
    add_node(str(tmp_path), "clean2", "editorial", "writer", "agent")
    add_node(str(tmp_path), "clean2", "editorial", "reviewer", "agent")
    set_entry(str(tmp_path), "clean2", "editorial", "writer")
    add_edge(str(tmp_path), "clean2", "editorial", "writer", "reviewer")
    set_root(str(tmp_path), "clean2", "editorial")
    _assert_codegen_clean(_agent_py(tmp_path, "clean2"), tmp_path, "agent_workflow")


# --------------------------------------------------------------------------- #
# FUNCTIONAL PROOF — instantiate AND run a real Workflow offline (subprocess)
# --------------------------------------------------------------------------- #
def test_functional_function_node_workflow_runs_offline(tmp_path: Path) -> None:
    """A pure function-node graph instantiates as a real Workflow and runs with no model."""
    create(str(tmp_path), "fn_run", "pipe")
    add_node(
        str(tmp_path), "fn_run", "pipe", "first", "function", docstring="First.", body='return "go"'
    )
    add_node(
        str(tmp_path),
        "fn_run",
        "pipe",
        "second",
        "function",
        docstring="Second.",
        body='return {"done": True}',
    )
    set_entry(str(tmp_path), "fn_run", "pipe", "first")
    add_edge(str(tmp_path), "fn_run", "pipe", "first", "second")
    set_root(str(tmp_path), "fn_run", "pipe")

    info = _probe(str(tmp_path), "fn_run", swap_fake_llm=False, message="hello")
    assert info["root_type"] == "Workflow"
    assert info["root_name"] == "pipe"
    assert info["node_names"] == ["__START__", "first", "second"]
    assert info["terminal"] == ["second"]
    # The graph executed (no exception); it is a real, runnable Workflow.


def test_functional_agent_node_workflow_runs_offline_with_fakellm(tmp_path: Path) -> None:
    """An agent-node graph instantiates as a real Workflow and runs both LlmAgents (FakeLlm)."""
    from adk_toolkit_mcp.domains.agents import create_llm

    create_llm(str(tmp_path), "ed_run", "writer", instruction="Write.")
    create_llm(str(tmp_path), "ed_run", "reviewer", instruction="Review.")
    create(str(tmp_path), "ed_run", "editorial")
    add_node(str(tmp_path), "ed_run", "editorial", "writer", "agent")
    add_node(str(tmp_path), "ed_run", "editorial", "reviewer", "agent")
    set_entry(str(tmp_path), "ed_run", "editorial", "writer")
    add_edge(str(tmp_path), "ed_run", "editorial", "writer", "reviewer")
    set_root(str(tmp_path), "ed_run", "editorial")

    info = _probe(str(tmp_path), "ed_run", swap_fake_llm=True, message="draft please")
    assert info["root_type"] == "Workflow"
    assert info["node_names"] == ["__START__", "writer", "reviewer"]
    # Both agent nodes ran, in graph order (writer then reviewer).
    assert info["texts"] == [["writer", "writer-out"], ["reviewer", "reviewer-out"]]


# --------------------------------------------------------------------------- #
# Mount wiring — in-memory client + end-to-end functional proof
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_workflow_mounted_names_and_functional(tmp_path: Path) -> None:
    mcp = build_server()
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        for expected in (
            "workflow_create",
            "workflow_add_node",
            "workflow_add_edge",
            "workflow_set_entry",
            "workflow_set_root",
            "workflow_list",
            "workflow_get",
        ):
            assert expected in tool_names, f"missing: {expected}"
        # No double prefix.
        assert not any(n.startswith("workflow_workflow_") for n in tool_names)

        path = str(tmp_path)
        assert (
            await client.call_tool(
                "workflow_create", {"path": path, "app_name": "client_wf", "name": "pipe"}
            )
        ).data["ok"] is True
        assert (
            await client.call_tool(
                "workflow_add_node",
                {
                    "path": path,
                    "app_name": "client_wf",
                    "workflow": "pipe",
                    "node_name": "only",
                    "kind": "function",
                    "body": 'return {"ok": 1}',
                },
            )
        ).data["ok"] is True
        assert (
            await client.call_tool(
                "workflow_set_entry",
                {"path": path, "app_name": "client_wf", "workflow": "pipe", "node": "only"},
            )
        ).data["ok"] is True
        rooted = await client.call_tool(
            "workflow_set_root", {"path": path, "app_name": "client_wf", "name": "pipe"}
        )
        assert rooted.data["ok"] is True
        assert rooted.data["data"]["root_kind"] == "workflow"

    # Outside the client: the generated module must instantiate + run the real Workflow.
    info = _probe(str(tmp_path), "client_wf", swap_fake_llm=False, message="go")
    assert info["root_type"] == "Workflow"
    assert info["node_names"] == ["__START__", "only"]
