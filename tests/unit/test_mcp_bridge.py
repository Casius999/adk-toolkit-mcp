"""Tests for the ``mcp_bridge`` domain (P4b — exposing ADK tools AS MCP tools).

These tests are **FUNCTIONAL and runnable in CI without any extra**: the ``mcp`` package is a core
dependency of ``fastmcp``, so ``adk_to_mcp_tool_type`` is always available. We prove:

- ``convert_builtin("google_search")`` returns a dict in ``mcp.types.Tool`` form
  (``{name, description, inputSchema}``) — we assert the STRUCTURE on a REAL ADK tool;
- ``expose_adk_tools`` on a real (scaffolded) agent carrying a builtin + a function-tool returns
  their MCP schemas (the function-tool has a real JSON-Schema ``properties``/``required``);
- the error paths (unknown kind, invalid app/agent, absent agent, agent without tools) → clean
  ``err``, never an exception;
- read-through via an in-memory ``fastmcp.Client``: exposed names ``mcp_bridge_<bare>`` (no double
  prefix) and the ``mcp_bridge_convert_builtin`` call round-trips.

Cf. ``docs/adk-api-notes/a2a-mcp-bridge.md`` (signatures + functional result confirmed).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastmcp import Client

from adk_toolkit_mcp.domains import mcp_bridge as MB
from adk_toolkit_mcp.server import build_server


@contextmanager
def _ignore_workflow_deprecation() -> Iterator[None]:
    """LOCALLY filter the workflow agents' ``DeprecationWarning`` (Sequential/Parallel/Loop).

    These agents are deprecated in ADK 2.1.0 but remain functional (cf. PROGRESS/agents.md);
    building them in-process (via ``import_root_agent`` which ``exec``s the scaffolded ``agent.py``)
    emits a ``DeprecationWarning`` that ``-W error::DeprecationWarning`` would turn into an error.
    We NEUTRALIZE it only for the duration of the call (narrow scope, our code stays strict).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*is deprecated and will be removed.*",
            category=DeprecationWarning,
        )
        yield


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scaffold_agent_with_tools(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffold an ADK app (importable WITHOUT an API key) with a builtin + a function-tool.

    The agent carries ``google_search`` (a builtin, empty inputSchema) and ``add_numbers`` (a bare
    function that ``canonical_tools`` wraps in a FunctionTool → real JSON-Schema). Returns the
    parent path.
    """
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    (app_dir / "agent.py").write_text(
        "from google.adk.agents import LlmAgent\n"
        "from google.adk.tools import google_search\n"
        "\n"
        "\n"
        "def add_numbers(a: int, b: int) -> int:\n"
        '    """Add two integers and return the sum."""\n'
        "    return a + b\n"
        "\n"
        "\n"
        f"root_agent = LlmAgent(name='{app_name}', model='gemini-2.5-flash', "
        "instruction='Help.', tools=[google_search, add_numbers])\n",
        encoding="utf-8",
    )
    return str(tmp_path)


def _scaffold_sequential(tmp_path: Path, app_name: str = "wf") -> str:
    """Scaffold an app whose root_agent is a SequentialAgent (no canonical_tools)."""
    app_dir = tmp_path / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    (app_dir / "agent.py").write_text(
        "from google.adk.agents import LlmAgent, SequentialAgent\n"
        "\n"
        "child = LlmAgent(name='child', model='gemini-2.5-flash', instruction='Hi.')\n"
        f"root_agent = SequentialAgent(name='{app_name}', sub_agents=[child])\n",
        encoding="utf-8",
    )
    return str(tmp_path)


# --------------------------------------------------------------------------- #
# convert_builtin — FUNCTIONAL (no extra)
# --------------------------------------------------------------------------- #
def test_convert_builtin_google_search_structure() -> None:
    """``convert_builtin('google_search')`` → MCP schema {name, description, inputSchema}."""
    result = MB.convert_builtin("google_search")
    assert result["ok"] is True, result
    tool = result["data"]["tool"]
    # Flattened mcp.types.Tool form: the three expected keys, with the right type.
    assert set(tool.keys()) == {"name", "description", "inputSchema"}
    assert tool["name"] == "google_search"
    assert isinstance(tool["description"], str)
    # google_search has no declared parameters -> inputSchema is a (empty) dict.
    assert isinstance(tool["inputSchema"], dict)


def test_convert_builtin_other_core_builtins() -> None:
    """Other core builtins also convert (BaseTool instances or wrapped functions)."""
    for kind in ("url_context", "load_memory", "exit_loop"):
        result = MB.convert_builtin(kind)
        assert result["ok"] is True, (kind, result)
        assert result["data"]["tool"]["name"] == kind
        assert isinstance(result["data"]["tool"]["inputSchema"], dict)


def test_convert_builtin_unknown_kind_returns_err() -> None:
    result = MB.convert_builtin("not_a_builtin")
    assert result["ok"] is False
    assert "not_a_builtin" in result["error"]


def test_convert_builtin_arg_builtin_rejected_with_guidance() -> None:
    """``vertex_ai_search`` (arg-requiring) is not a *core* builtin → err pointing to expose."""
    result = MB.convert_builtin("vertex_ai_search")
    assert result["ok"] is False
    assert "expose_adk_tools" in result["error"]


# --------------------------------------------------------------------------- #
# expose_adk_tools — FUNCTIONAL (no extra)
# --------------------------------------------------------------------------- #
async def test_expose_adk_tools_returns_mcp_schemas(tmp_path: Path) -> None:
    """A real agent (builtin + function-tool) → their MCP schemas (function = real schema)."""
    path = _scaffold_agent_with_tools(tmp_path)
    result = await MB.expose_adk_tools(path=path, app_name="myapp", agent_name="myapp")
    assert result["ok"] is True, result
    data = result["data"]
    assert data["count"] == 2
    by_name = {t["name"]: t for t in data["tools"]}
    assert "google_search" in by_name
    assert "add_numbers" in by_name
    # The function-tool has a REAL JSON-Schema (properties a/b, required).
    schema = by_name["add_numbers"]["inputSchema"]
    assert schema["type"] == "object"
    assert set(schema["properties"].keys()) == {"a", "b"}
    assert set(schema["required"]) == {"a", "b"}
    assert by_name["add_numbers"]["description"] == "Add two integers and return the sum."


async def test_expose_adk_tools_unknown_agent_returns_err(tmp_path: Path) -> None:
    path = _scaffold_agent_with_tools(tmp_path)
    result = await MB.expose_adk_tools(path=path, app_name="myapp", agent_name="ghost")
    assert result["ok"] is False
    assert "ghost" in result["error"]


async def test_expose_adk_tools_missing_app_returns_err(tmp_path: Path) -> None:
    """No scaffolded agent.py → clean err (RootAgentImportError), no exception."""
    result = await MB.expose_adk_tools(path=str(tmp_path), app_name="ghostapp", agent_name="x")
    assert result["ok"] is False


async def test_expose_adk_tools_invalid_app_name_returns_err(tmp_path: Path) -> None:
    result = await MB.expose_adk_tools(path=str(tmp_path), app_name="bad name", agent_name="x")
    assert result["ok"] is False
    assert "app_name" in result["error"]


async def test_expose_adk_tools_invalid_agent_name_returns_err(tmp_path: Path) -> None:
    result = await MB.expose_adk_tools(path=str(tmp_path), app_name="myapp", agent_name="bad name")
    assert result["ok"] is False


async def test_expose_adk_tools_workflow_agent_has_no_tools(tmp_path: Path) -> None:
    """A workflow agent (Sequential) without canonical_tools → actionable err (not a crash)."""
    path = _scaffold_sequential(tmp_path)
    with _ignore_workflow_deprecation():
        result = await MB.expose_adk_tools(path=path, app_name="wf", agent_name="wf")
    assert result["ok"] is False
    assert "LLM" in result["error"] or "tools" in result["error"].lower()


async def test_expose_adk_tools_sub_agent_found_in_tree(tmp_path: Path) -> None:
    """``find_agent`` locates a SUB-agent (not just the root) — the LLM child has its tools."""
    path = _scaffold_sequential(tmp_path)
    # 'child' is an LlmAgent (without tools) nested under the root SequentialAgent.
    with _ignore_workflow_deprecation():
        result = await MB.expose_adk_tools(path=path, app_name="wf", agent_name="child")
    assert result["ok"] is True, result
    # No tool attached → empty list, this is NOT an error.
    assert result["data"]["count"] == 0
    assert result["data"]["tools"] == []


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client (exposed names + call)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_convert_builtin() -> None:
    """Tools exposed as ``mcp_bridge_<bare>`` (no double prefix); convert_builtin round-trips."""
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {"mcp_bridge_convert_builtin", "mcp_bridge_expose_adk_tools"} <= names
        assert not any(n.startswith("mcp_bridge_mcp_bridge_") for n in names)

        called = await client.call_tool("mcp_bridge_convert_builtin", {"kind": "google_search"})
        assert called.data["ok"] is True
        assert called.data["data"]["tool"]["name"] == "google_search"
