"""Root server tests: direct-tools mode (default) vs Code Mode (P6a), and per-domain tags.

- ``build_server()`` (default) exposes the 94 tools by their ``<domain>_<bare>`` name (no
  regression) and each tool carries its domain tag.
- ``build_server(code_mode=True)`` applies the REAL FastMCP 3.3.1 transform and collapses the
  surface into discovery tools + ``execute`` — demonstrating the token reduction (94 → a handful).
- ``code_mode_enabled()`` reads the ``ADK_TOOLKIT_CODE_MODE`` env variable.

The server-side lists (``mcp.list_tools()``) return ``fastmcp.tools.Tool`` objects that carry
``.tags``; the client read-through (``fastmcp.Client``) confirms the exposed surface and the tag
surfaced via ``_meta.fastmcp.tags``.
"""

from __future__ import annotations

import os

import pytest
from fastmcp import Client, FastMCP

from adk_toolkit_mcp.server import build_server, code_mode_enabled, main

#: Exact number of tools exposed in direct-tools mode (non-regression contract).
_EXPECTED_TOOL_COUNT = 94

#: The 17 mounted domains (namespace prefix -> expected tag).
_DOMAINS = (
    "project",
    "agents",
    "tools",
    "models",
    "sessions",
    "memory",
    "artifacts",
    "run",
    "eval",
    "deploy",
    "dev",
    "a2a",
    "mcp_bridge",
    "safety",
    "observability",
    "workflow",
    "skills",
)

#: Sample of tool names that MUST exist in direct-tools mode (one per key domain).
_SAMPLE_NAMES = {
    "project_create",
    "agents_create_llm",
    "agents_set_root",
    "agents_set_planner",
    "tools_add_function",
    "models_set",
    "sessions_create",
    "memory_search",
    "artifacts_save",
    "run_agent",
    "eval_run",
    "deploy_cloud_run",
    "dev_web",
    "a2a_consume",
    "mcp_bridge_convert_builtin",
    "safety_add_callback",
    "observability_enable_otel",
    "workflow_create",
    "workflow_add_node",
    "skills_create",
    "skills_attach",
}


def _domain_of(tool_name: str) -> str:
    """Return the domain of an exposed tool name (handles the compound ``mcp_bridge`` namespace)."""
    if tool_name.startswith("mcp_bridge_"):
        return "mcp_bridge"
    return tool_name.split("_", 1)[0]


# --------------------------------------------------------------------------- #
# Basic construction
# --------------------------------------------------------------------------- #
def test_build_server_returns_fastmcp() -> None:
    assert isinstance(build_server(), FastMCP)


def test_build_server_code_mode_returns_fastmcp() -> None:
    assert isinstance(build_server(code_mode=True), FastMCP)


def test_main_is_callable() -> None:
    assert callable(main)


# --------------------------------------------------------------------------- #
# Direct-tools mode (default): 94 tools, stable names, per-domain tags
# --------------------------------------------------------------------------- #
async def test_default_mode_exposes_all_94_tools_by_name() -> None:
    """Default: 94 tools exposed by name (no regression) + a sample present."""
    async with Client(build_server()) as client:
        names = {t.name for t in await client.list_tools()}
    assert len(names) == _EXPECTED_TOOL_COUNT
    assert _SAMPLE_NAMES <= names
    # No double prefix (e.g. project_project_create).
    assert not any(n.startswith(f"{d}_{d}_") for d in _DOMAINS for n in names)


async def test_every_tool_carries_its_domain_tag() -> None:
    """Each tool carries exactly its domain tag (server-side ``.tags`` inspection)."""
    tools = await build_server().list_tools()
    assert len(tools) == _EXPECTED_TOOL_COUNT
    mismatched = [
        (t.name, sorted(t.tags or [])) for t in tools if _domain_of(t.name) not in (t.tags or set())
    ]
    assert mismatched == []


async def test_domain_tags_surface_to_client_via_meta() -> None:
    """The domain tag surfaces to the MCP client via ``_meta.fastmcp.tags``."""
    async with Client(build_server()) as client:
        tools = await client.list_tools()
    by_name = {t.name: t for t in tools}
    meta = by_name["project_create"].meta or {}
    assert "project" in (meta.get("fastmcp", {}).get("tags") or [])


# --------------------------------------------------------------------------- #
# Code Mode (opt-in): collapsed + reachable surface
# --------------------------------------------------------------------------- #
async def test_code_mode_collapses_surface_to_discovery_and_execute() -> None:
    """code_mode=True: the surface goes from 94 tools to a handful of discovery + execute."""
    async with Client(build_server(code_mode=True)) as client:
        names = {t.name for t in await client.list_tools()}
    # Sharp surface reduction (big token saving) — demonstrated quantitatively.
    assert len(names) < 10
    # Execution tool still present + at least one discovery tool.
    assert "execute" in names
    assert {"search", "get_schema"} <= names
    # The 94 direct names are NO LONGER exposed at the top level.
    assert "run_agent" not in names
    assert "project_create" not in names


async def test_code_mode_reduces_tool_surface_vs_default() -> None:
    """Demonstrates the reduction: Code Mode surface << direct-tools surface (94)."""
    async with Client(build_server()) as direct_client:
        direct = {t.name for t in await direct_client.list_tools()}
    async with Client(build_server(code_mode=True)) as cm_client:
        code_mode = {t.name for t in await cm_client.list_tools()}
    assert len(direct) == _EXPECTED_TOOL_COUNT
    # At least 90% fewer tools at the top level.
    assert len(code_mode) <= len(direct) // 10


async def test_code_mode_tags_discovery_tool_present() -> None:
    """Since we tag by domain, the ``tags`` discovery tool is added and reachable in Code Mode."""
    async with Client(build_server(code_mode=True)) as client:
        names = {t.name for t in await client.list_tools()}
        assert "tags" in names
        # The ``tags`` discovery lists the tagged domains (reads the catalog, without monty).
        result = await client.call_tool("tags", {"detail": "brief"})
    rendered = "\n".join(block.text for block in result.content if getattr(block, "text", None))
    # A few known domains appear in the tags rendering.
    assert "agents" in rendered
    assert "deploy" in rendered


# --------------------------------------------------------------------------- #
# Environment-variable toggle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nope", False),
    ],
)
def test_code_mode_enabled_reads_env(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """``code_mode_enabled`` recognizes the truthy/falsy values of ``ADK_TOOLKIT_CODE_MODE``."""
    monkeypatch.setenv("ADK_TOOLKIT_CODE_MODE", value)
    assert code_mode_enabled() is expected


def test_code_mode_enabled_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an env variable, Code Mode is disabled (direct-tools mode by default)."""
    monkeypatch.delenv("ADK_TOOLKIT_CODE_MODE", raising=False)
    assert code_mode_enabled() is False
    # Sanity: the env did not leak from another test.
    assert os.getenv("ADK_TOOLKIT_CODE_MODE") is None
