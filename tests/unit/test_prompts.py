"""Tests for the 5 workflow prompts (P6a).

Read via an in-memory ``fastmcp.Client`` (``list_prompts`` / ``get_prompt`` — the REAL fastmcp
3.3.1 client API). We verify:

- the 5 expected prompts are registered (with their arguments);
- each renders a NON-empty, actionable string referencing ``<domain>_*`` tools;
- **load-bearing cross-check**: every ``<domain>_<name>`` token cited in a prompt actually
  exists in the server's real tool catalog (no invented tool name);
- each prompt carries the ``workflow`` tag.
"""

from __future__ import annotations

import re

import pytest
from fastmcp import Client

from adk_toolkit_mcp.server import build_server

#: The 5 expected workflow prompts -> example arguments for rendering.
_PROMPT_ARGS: dict[str, dict[str, str]] = {
    "scaffold_multi_agent": {"goal": "triage support tickets"},
    "add_guardrail": {"agent": "router", "concern": "block PII"},
    "write_evalset": {"agent": "router"},
    "deploy_checklist": {"target": "cloud_run"},
    "debug_agent": {"symptom": "no response"},
}

#: The 15 mounted domains (to spot the ``<domain>_<name>`` tokens in the prompt text).
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
)

#: Spots a token that looks like an exposed tool name: ``<domain>_<snake_suffix>``.
_TOOL_TOKEN = re.compile(r"\b(?:" + "|".join(_DOMAINS) + r")_[a-z][a-z_]*\b")


async def _real_tool_names() -> set[str]:
    """Set of the tool names actually exposed (direct-tools mode)."""
    return {t.name for t in await build_server().list_tools()}


async def _render(client: Client, name: str) -> str:
    """Render a prompt and return the text of its single message."""
    result = await client.get_prompt(name, _PROMPT_ARGS[name])
    return result.messages[0].content.text


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
async def test_all_five_prompts_registered() -> None:
    """The 5 expected workflow prompts are registered on the server."""
    async with Client(build_server()) as client:
        names = {p.name for p in await client.list_prompts()}
    assert set(_PROMPT_ARGS) <= names


async def test_prompts_declare_their_arguments() -> None:
    """Each prompt declares its arguments (derived from the function signature)."""
    async with Client(build_server()) as client:
        by_name = {p.name: p for p in await client.list_prompts()}
    expected_args = {
        "scaffold_multi_agent": {"goal"},
        "add_guardrail": {"agent", "concern"},
        "write_evalset": {"agent"},
        "deploy_checklist": {"target"},
        "debug_agent": {"symptom"},
    }
    for name, args in expected_args.items():
        declared = {a.name for a in (by_name[name].arguments or [])}
        assert declared == args, (name, declared)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_PROMPT_ARGS))
async def test_prompt_renders_non_empty_actionable_text(name: str) -> None:
    """Each prompt renders a non-empty, substantial string citing the toolkit's tools."""
    async with Client(build_server()) as client:
        text = await _render(client, name)
    assert isinstance(text, str)
    assert len(text.strip()) > 200
    # An actionable prompt cites at least one ``<domain>_*`` tool.
    assert _TOOL_TOKEN.search(text) is not None


async def test_prompt_interpolates_its_arguments() -> None:
    """The passed arguments are interpolated into the rendered text (parameterized template)."""
    async with Client(build_server()) as client:
        scaffold = await _render(client, "scaffold_multi_agent")
        guardrail = await _render(client, "add_guardrail")
    assert "triage support tickets" in scaffold
    assert "router" in guardrail
    assert "block PII" in guardrail


# --------------------------------------------------------------------------- #
# Cross-check: every cited tool actually exists (no invented name)
# --------------------------------------------------------------------------- #
async def test_every_cited_tool_token_is_a_real_tool() -> None:
    """Cross-check: every ``<domain>_<name>`` token cited in a prompt is a real tool."""
    real = await _real_tool_names()
    cited: set[str] = set()
    async with Client(build_server()) as client:
        for name in _PROMPT_ARGS:
            cited |= set(_TOOL_TOKEN.findall(await _render(client, name)))
    # The cross-check only makes sense if the prompts actually cite tools.
    assert cited, "no tool token cited — the prompts should reference tools"
    bogus = sorted(token for token in cited if token not in real)
    assert bogus == [], f"tokens cited but nonexistent as tools: {bogus}"


async def test_key_workflow_tools_are_cited() -> None:
    """Each prompt cites the pivotal tool of its workflow (key-path coverage)."""
    async with Client(build_server()) as client:
        rendered = {name: await _render(client, name) for name in _PROMPT_ARGS}
    assert "project_create" in rendered["scaffold_multi_agent"]
    assert "agents_create_llm" in rendered["scaffold_multi_agent"]
    assert "run_agent" in rendered["scaffold_multi_agent"]
    assert "safety_add_callback" in rendered["add_guardrail"]
    assert "safety_add_plugin" in rendered["add_guardrail"]
    assert "eval_create_set" in rendered["write_evalset"]
    assert "eval_run" in rendered["write_evalset"]
    assert "deploy_preflight" in rendered["deploy_checklist"]
    assert "deploy_cloud_run" in rendered["deploy_checklist"]
    assert "run_inspect_events" in rendered["debug_agent"]


# --------------------------------------------------------------------------- #
# Tag workflow
# --------------------------------------------------------------------------- #
async def test_prompts_carry_workflow_tag() -> None:
    """Each workflow prompt carries the ``workflow`` tag (parity with the tools' tagging)."""
    prompts = await build_server().list_prompts()
    by_name = {p.name: p for p in prompts}
    for name in _PROMPT_ARGS:
        assert "workflow" in (by_name[name].tags or set()), name
