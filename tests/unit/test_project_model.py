"""Unit tests for the pure ``project_model`` renderer (no in-process ADK construction).

We assert on the generated **source string** (safe under ``-W error::DeprecationWarning``,
since we build no deprecated workflow agent here). The functional proof (real
instantiation of the ADK objects) is done in ``test_agents.py`` via a subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from adk_toolkit_mcp.project_model import (
    SIDECAR_PATH,
    AgentSpec,
    AuthSpec,
    CallbackSpec,
    ProjectModel,
    ToolRender,
    ToolSpec,
    add_or_replace_callback,
    add_or_update_agent,
    load_model,
    regenerate,
    render_agent_module,
    render_tool_ref,
    save_model,
    set_root,
    topological_order,
    validate_callback_spec,
    validate_spec,
    validate_tool_spec,
)
from adk_toolkit_mcp.workspace import Workspace


# --------------------------------------------------------------------------- #
# Dataclasses + (de)serialization
# --------------------------------------------------------------------------- #
def test_agentspec_is_frozen() -> None:
    spec = AgentSpec(name="a", type="llm")
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "b"  # type: ignore[misc]


def test_agentspec_roundtrip_llm() -> None:
    spec = AgentSpec(
        name="writer",
        type="llm",
        model="gemini-2.5-flash",
        instruction="Write.",
        description="A writer.",
        output_key="draft",
        tools=("google_search",),
    )
    restored = AgentSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_projectmodel_roundtrip() -> None:
    model = ProjectModel(
        app_name="demo",
        root="pipe",
        agents=(
            AgentSpec(name="a", type="llm"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("a",)),
        ),
    )
    restored = ProjectModel.from_dict(model.to_dict())
    assert restored == model


# --------------------------------------------------------------------------- #
# Immutable mutations
# --------------------------------------------------------------------------- #
def test_add_or_update_agent_is_immutable() -> None:
    model = ProjectModel(app_name="demo")
    new = add_or_update_agent(model, AgentSpec(name="a", type="llm"))
    assert model.agents == ()  # original intact
    assert new.agent_names() == ("a",)
    assert new is not model


def test_add_or_update_agent_replaces_in_place() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(AgentSpec(name="a", type="llm"), AgentSpec(name="b", type="llm")),
    )
    updated = add_or_update_agent(model, AgentSpec(name="a", type="llm", instruction="new"))
    # Position preserved, no duplicate.
    assert updated.agent_names() == ("a", "b")
    a = updated.get("a")
    assert a is not None and a.instruction == "new"


def test_set_root_immutable() -> None:
    model = ProjectModel(app_name="demo", agents=(AgentSpec(name="a", type="llm"),))
    new = set_root(model, "a")
    assert model.root is None
    assert new.root == "a"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_rejects_bad_identifier() -> None:
    assert validate_spec(AgentSpec(name="bad name", type="llm")) is not None
    assert validate_spec(AgentSpec(name="1abc", type="llm")) is not None


def test_validate_rejects_unknown_type() -> None:
    assert validate_spec(AgentSpec(name="a", type="nope")) is not None  # type: ignore[arg-type]


def test_validate_rejects_nonpositive_max_iterations() -> None:
    assert validate_spec(AgentSpec(name="a", type="loop", max_iterations=0)) is not None
    assert validate_spec(AgentSpec(name="a", type="loop", max_iterations=-1)) is not None


def test_validate_accepts_good_llm() -> None:
    assert validate_spec(AgentSpec(name="good_agent", type="llm")) is None


# --------------------------------------------------------------------------- #
# Topological sort + cycles
# --------------------------------------------------------------------------- #
def test_topological_order_child_before_parent() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="pipe", type="sequential", sub_agents=("a", "b")),
            AgentSpec(name="a", type="llm"),
            AgentSpec(name="b", type="llm"),
        ),
    )
    ordered = [s.name for s in topological_order(model)]
    assert ordered.index("a") < ordered.index("pipe")
    assert ordered.index("b") < ordered.index("pipe")


def test_topological_order_detects_direct_cycle() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="x", type="sequential", sub_agents=("y",)),
            AgentSpec(name="y", type="sequential", sub_agents=("x",)),
        ),
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        topological_order(model)


def test_topological_order_detects_self_cycle() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(AgentSpec(name="x", type="sequential", sub_agents=("x",)),),
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        topological_order(model)


# --------------------------------------------------------------------------- #
# Source rendering — by type
# --------------------------------------------------------------------------- #
def test_render_llm_minimal_omits_empty_kwargs() -> None:
    model = ProjectModel(
        app_name="demo",
        root="solo",
        agents=(AgentSpec(name="solo", type="llm", instruction="Hi"),),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents import LlmAgent" in src
    assert "solo = LlmAgent(" in src
    assert 'name="solo"' in src
    assert 'model="gemini-2.5-flash"' in src
    assert 'instruction="Hi"' in src
    # empty description / output_key None / empty tools / empty sub_agents -> omitted.
    assert "description=" not in src
    assert "output_key=" not in src
    assert "tools=" not in src
    assert "sub_agents=" not in src
    assert src.rstrip().endswith("root_agent = solo")


def test_render_llm_includes_output_key_and_tools() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(
                name="searcher",
                type="llm",
                description="Searches.",
                output_key="results",
                tools=("google_search", "my_tool"),
            ),
        ),
    )
    src = render_agent_module(model)
    assert 'output_key="results"' in src
    assert "tools=[google_search, my_tool]" in src
    assert 'description="Searches."' in src


def test_render_sequential_and_parallel() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="a", type="llm"),
            AgentSpec(name="b", type="llm"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("a", "b")),
            AgentSpec(name="fan", type="parallel", sub_agents=("a", "b")),
        ),
    )
    src = render_agent_module(model)
    # The imported names are sorted (isort ``I001``), not in the canonical ADK order:
    # ``LlmAgent, ParallelAgent, SequentialAgent`` (alphabetical).
    assert "from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent" in src
    assert "pipe = SequentialAgent(" in src
    assert "sub_agents=[a, b]" in src
    assert "fan = ParallelAgent(" in src


def test_render_loop_includes_max_iterations() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="step", type="llm"),
            AgentSpec(name="lp", type="loop", sub_agents=("step",), max_iterations=5),
        ),
    )
    src = render_agent_module(model)
    assert "lp = LoopAgent(" in src
    assert "max_iterations=5" in src
    assert "sub_agents=[step]" in src


def test_render_custom_emits_baseagent_subclass_and_instance() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(AgentSpec(name="my_custom", type="custom", description="D"),),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents import BaseAgent" in src
    assert "class MyCustomAgent(BaseAgent):" in src
    assert "async def _run_async_impl(self, ctx):" in src
    # async generator no-op: return followed by an unreachable yield.
    assert "yield" in src
    assert 'my_custom = MyCustomAgent(name="my_custom", description="D")' in src


# --------------------------------------------------------------------------- #
# remote_a2a (P4b) — proxy RemoteA2aAgent
# --------------------------------------------------------------------------- #
def test_render_remote_a2a_emits_call_and_submodule_import() -> None:
    """``remote_a2a`` renders ``RemoteA2aAgent(name=..., agent_card="...")`` + the submodule import.

    Warning: in 2.1.0, ``RemoteA2aAgent`` is NOT in ``google.adk.agents``: the import MUST be
    ``from google.adk.agents.remote_a2a_agent import RemoteA2aAgent`` (cf. a2a-mcp-bridge.md).
    """
    model = ProjectModel(
        app_name="demo",
        root="remote_helper",
        agents=(
            AgentSpec(
                name="remote_helper",
                type="remote_a2a",
                agent_card="http://localhost:8001/.well-known/agent-card.json",
                description="Remote helper.",
            ),
        ),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent" in src
    # NO erroneous import from google.adk.agents (RemoteA2aAgent is not there).
    assert "from google.adk.agents import RemoteA2aAgent" not in src
    assert "remote_helper = RemoteA2aAgent(" in src
    assert 'name="remote_helper"' in src
    assert 'agent_card="http://localhost:8001/.well-known/agent-card.json"' in src
    assert 'description="Remote helper."' in src
    assert "root_agent = remote_helper" in src


def test_render_remote_a2a_composes_as_sub_agent_topo_order() -> None:
    """A ``remote_a2a`` can be a ``sub_agent`` of another agent; defined BEFORE the parent."""
    model = ProjectModel(
        app_name="demo",
        root="router",
        agents=(
            # Declared AFTER the parent in the model to prove the topological sort.
            AgentSpec(
                name="router", type="llm", instruction="Route.", sub_agents=("remote_helper",)
            ),
            AgentSpec(
                name="remote_helper",
                type="remote_a2a",
                agent_card="http://localhost:8001/a2a",
            ),
        ),
    )
    src = render_agent_module(model)
    # Two agent classes each imported from their module (isort sorts by module).
    assert "from google.adk.agents import LlmAgent" in src
    assert "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent" in src
    # The proxy definition precedes the parent's (dependency before dependent).
    assert src.index("remote_helper = RemoteA2aAgent(") < src.index("router = LlmAgent(")
    assert "sub_agents=[remote_helper]" in src


def test_render_remote_a2a_no_description_omits_kwarg() -> None:
    """Without a description, the ``description=`` kwarg is omitted (only name + agent_card)."""
    model = ProjectModel(
        app_name="demo",
        root="r",
        agents=(AgentSpec(name="r", type="remote_a2a", agent_card="http://h/a2a"),),
    )
    src = render_agent_module(model)
    assert "r = RemoteA2aAgent(" in src
    assert "description=" not in src


def test_remote_a2a_spec_roundtrip_serializes_agent_card() -> None:
    """The ``agent_card`` field survives a to_dict/from_dict round-trip (sidecar form)."""
    spec = AgentSpec(
        name="remote_helper",
        type="remote_a2a",
        agent_card="http://localhost:8001/.well-known/agent-card.json",
        description="D",
    )
    d = spec.to_dict()
    assert d["type"] == "remote_a2a"
    assert d["agent_card"] == "http://localhost:8001/.well-known/agent-card.json"
    back = AgentSpec.from_dict(d)
    assert back.type == "remote_a2a"
    assert back.agent_card == spec.agent_card
    assert back.description == "D"


def test_validate_remote_a2a_requires_agent_card() -> None:
    """``remote_a2a`` without agent_card is rejected (actionable message)."""
    bad = AgentSpec(name="r", type="remote_a2a", agent_card="")
    error = validate_spec(bad)
    assert error is not None
    assert "agent_card" in error
    good = AgentSpec(name="r", type="remote_a2a", agent_card="http://h/a2a")
    assert validate_spec(good) is None


def test_render_remote_a2a_format_and_isort_stable(tmp_path: Path) -> None:
    """The module generated with a remote_a2a + a parent llm is format- AND isort-clean."""
    model = ProjectModel(
        app_name="demo",
        root="router",
        agents=(
            AgentSpec(
                name="remote_helper",
                type="remote_a2a",
                agent_card="http://localhost:8001/.well-known/agent-card.json",
                description="Remote helper.",
            ),
            AgentSpec(
                name="router", type="llm", instruction="Route.", sub_agents=("remote_helper",)
            ),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "remote_a2a")


def test_render_empty_model_has_no_root() -> None:
    src = render_agent_module(ProjectModel(app_name="demo"))
    assert "root_agent =" not in src.replace("# root_agent", "")
    assert "No agent" in src


def test_render_root_missing_emits_comment_not_assignment() -> None:
    model = ProjectModel(
        app_name="demo",
        root="ghost",  # does not exist
        agents=(AgentSpec(name="real", type="llm"),),
    )
    src = render_agent_module(model)
    assert "root_agent = ghost" not in src
    assert "not found" in src


def test_render_imports_only_used_classes() -> None:
    # Only llms -> does not import Sequential/Parallel/Loop/BaseAgent.
    model = ProjectModel(app_name="demo", agents=(AgentSpec(name="a", type="llm"),))
    src = render_agent_module(model)
    line = next(line for line in src.splitlines() if line.startswith("from google.adk.agents"))
    assert "LlmAgent" in line
    assert "SequentialAgent" not in line
    assert "BaseAgent" not in line


# --------------------------------------------------------------------------- #
# Tool rendering — render_tool_ref (pass 3a)
# --------------------------------------------------------------------------- #
def test_render_tool_ref_function_emits_def_and_bare_ref() -> None:
    tool = ToolSpec(
        kind="function",
        name="add",
        params=(("a", "int", None), ("b", "int", "0")),
        docstring="Add two ints.",
        returns="dict",
        body="return {'sum': a + b}",
    )
    tr = render_tool_ref(tool)
    assert isinstance(tr, ToolRender)
    assert tr.ref == "add"  # ADK auto-wraps the function in a FunctionTool.
    assert tr.imports == ()  # a plain function imports nothing.
    assert len(tr.helpers) == 1
    helper = tr.helpers[0]
    assert helper.startswith("def add(a: int, b: int = 0) -> dict:")
    assert '"""Add two ints."""' in helper
    assert "return {'sum': a + b}" in helper


def test_render_tool_ref_long_running_wraps_func() -> None:
    tool = ToolSpec(kind="long_running", name="slow", docstring="Slow op.")
    tr = render_tool_ref(tool)
    assert tr.ref == "LongRunningFunctionTool(func=slow)"
    assert "from google.adk.tools import LongRunningFunctionTool" in tr.imports
    assert tr.helpers[0].startswith("def slow() -> dict:")


def test_render_tool_ref_builtin_core_is_bare_name() -> None:
    tr = render_tool_ref(ToolSpec(kind="builtin", builtin_kind="google_search"))
    assert tr.ref == "google_search"
    assert tr.imports == ("from google.adk.tools import google_search",)
    assert tr.helpers == ()


def test_render_tool_ref_builtin_vertex_ai_search_needs_arg() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="builtin",
            builtin_kind="vertex_ai_search",
            args=(("data_store_id", "projects/p/dataStores/d"),),
        )
    )
    assert tr.ref == 'VertexAiSearchTool(data_store_id="projects/p/dataStores/d")'
    assert tr.imports == ("from google.adk.tools import VertexAiSearchTool",)


def test_render_tool_ref_agent_tool_wraps_target() -> None:
    tr = render_tool_ref(ToolSpec(kind="agent_tool", target_agent="helper"))
    assert tr.ref == "AgentTool(agent=helper)"
    assert tr.imports == ("from google.adk.tools import AgentTool",)
    assert tr.helpers == ()


def test_render_tool_ref_openapi_builds_toolset_and_refs_it() -> None:
    tr = render_tool_ref(ToolSpec(kind="openapi", name="petstore", spec='{"openapi": "3.0.0"}'))
    assert tr.ref == "petstore"
    assert tr.imports == ("from google.adk.tools.openapi_tool import OpenAPIToolset",)
    assert len(tr.helpers) == 1
    assert tr.helpers[0].startswith("petstore = OpenAPIToolset(spec_str=")
    assert 'spec_str_type="json"' in tr.helpers[0]


def test_render_tool_ref_legacy_string_is_bare_passthrough() -> None:
    # Legacy P1 form: a string stays a bare reference, no import or helper.
    tr = render_tool_ref("already_imported_tool")
    assert tr.ref == "already_imported_tool"
    assert tr.imports == ()
    assert tr.helpers == ()


# --------------------------------------------------------------------------- #
# Tool rendering — render_tool_ref (pass 3b: optional-dependency toolsets)
# --------------------------------------------------------------------------- #
def test_render_tool_ref_bigquery_builds_toolset() -> None:
    tr = render_tool_ref(ToolSpec(kind="bigquery", name="bq"))
    assert tr.ref == "bq"
    assert tr.imports == ("from google.adk.tools.bigquery import BigQueryToolset",)
    assert tr.helpers[0].startswith("bq = BigQueryToolset(")


def test_render_tool_ref_bigquery_with_args() -> None:
    tr = render_tool_ref(
        ToolSpec(kind="bigquery", name="bq", args=(("bigquery_tool_config", "my_cfg"),))
    )
    # The args are source EXPRESSIONS (not string literals) -> rendered as is.
    assert "bigquery_tool_config=my_cfg" in tr.helpers[0]


def test_render_tool_ref_spanner_builds_toolset() -> None:
    tr = render_tool_ref(ToolSpec(kind="spanner", name="sp"))
    assert tr.ref == "sp"
    assert tr.imports == ("from google.adk.tools.spanner import SpannerToolset",)
    assert tr.helpers[0].startswith("sp = SpannerToolset(")


def test_render_tool_ref_mcp_stdio_single_arg_no_trailing_comma() -> None:
    # The stdio toolset nests 3 calls: even minimal it exceeds 100 cols and folds.
    # ruff rule: a SINGLE-argument call that folds does NOT add a trailing comma.
    tr = render_tool_ref(ToolSpec(kind="mcp_toolset", name="fs", transport="stdio", command="srv"))
    assert tr.ref == "fs"
    imps = "\n".join(tr.imports)
    assert "from google.adk.tools.mcp_tool import" in imps
    assert "McpToolset" in imps and "StdioConnectionParams" in imps
    assert "from mcp import StdioServerParameters" in imps
    helper = tr.helpers[0]
    # Exact ``ruff format`` shape: single argument exploded, no trailing comma.
    assert helper == (
        "fs = McpToolset(\n"
        "    connection_params=StdioConnectionParams(\n"
        '        server_params=StdioServerParameters(command="srv", args=[])\n'
        "    )\n"
        ")\n"
    )


def test_render_tool_ref_mcp_stdio_long_folds_ruff_stable() -> None:
    # Long case: the renderer folds (recursively); we check the key fragments + the order.
    tr = render_tool_ref(
        ToolSpec(
            kind="mcp_toolset",
            name="fs",
            transport="stdio",
            command="npx",
            mcp_args=("-y", "@modelcontextprotocol/server-filesystem", "/tmp"),
            tool_filter=("read_file", "list_directory"),
        )
    )
    helper = tr.helpers[0]
    assert helper.startswith("fs = McpToolset(\n")
    assert "connection_params=StdioConnectionParams(" in helper
    assert "server_params=StdioServerParameters(" in helper
    assert 'command="npx"' in helper
    assert 'args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]' in helper
    assert 'tool_filter=["read_file", "list_directory"]' in helper
    # No line exceeds the ruff limit.
    assert all(len(line) <= 100 for line in helper.splitlines())


def test_render_tool_ref_mcp_sse() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="mcp_toolset",
            name="remote",
            transport="sse",
            url="https://example.com/sse",
            headers=(("Authorization", "Bearer x"),),
        )
    )
    imps = "\n".join(tr.imports)
    assert "SseConnectionParams" in imps
    assert "StdioServerParameters" not in imps  # no stdio import for sse
    helper = tr.helpers[0]
    assert "SseConnectionParams(" in helper
    assert 'url="https://example.com/sse"' in helper
    assert 'headers={"Authorization": "Bearer x"}' in helper


def test_render_tool_ref_mcp_http() -> None:
    tr = render_tool_ref(
        ToolSpec(kind="mcp_toolset", name="h", transport="http", url="https://api.example.com/mcp")
    )
    imps = "\n".join(tr.imports)
    assert "StreamableHTTPConnectionParams" in imps
    helper = tr.helpers[0]
    assert 'StreamableHTTPConnectionParams(url="https://api.example.com/mcp")' in helper


def test_render_tool_ref_apihub_builds_toolset() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="apihub",
            name="hub",
            apihub_resource_name="projects/p/locations/l/apis/a",
        )
    )
    assert tr.ref == "hub"
    assert tr.imports == ("from google.adk.tools.apihub_tool import APIHubToolset",)
    helper = tr.helpers[0]
    assert helper.startswith("hub = APIHubToolset(")
    assert 'apihub_resource_name="projects/p/locations/l/apis/a"' in helper


def test_render_tool_ref_langchain_wraps_expr_and_renders_import_line() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="langchain",
            import_line="from langchain_community.tools import WikipediaQueryRun",
            tool_expr="WikipediaQueryRun(api_wrapper=wrapper)",
        )
    )
    assert tr.ref == "LangchainTool(tool=WikipediaQueryRun(api_wrapper=wrapper))"
    imps = "\n".join(tr.imports)
    assert "from google.adk.tools.langchain_tool import LangchainTool" in imps
    assert "from langchain_community.tools import WikipediaQueryRun" in imps
    assert tr.helpers == ()


def test_render_tool_ref_crewai_wraps_expr_with_name_and_description() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="crewai",
            import_line="from crewai_tools import SerperDevTool",
            tool_expr="SerperDevTool()",
            name="serper",
            description="Web search via Serper.",
        )
    )
    assert tr.ref == (
        'CrewaiTool(tool=SerperDevTool(), name="serper", description="Web search via Serper.")'
    )
    imps = "\n".join(tr.imports)
    assert "from google.adk.tools.crewai_tool import CrewaiTool" in imps
    assert "from crewai_tools import SerperDevTool" in imps


# --------------------------------------------------------------------------- #
# Auth rendering (set_auth) — auth_credential= on compatible toolsets
# --------------------------------------------------------------------------- #
def test_render_tool_ref_openapi_with_apikey_auth() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="openapi",
            name="api",
            spec='{"openapi": "3.0.0"}',
            auth=AuthSpec(scheme="apikey", credential=(("api_key", "secret123"),)),
        )
    )
    helper = tr.helpers[0]
    assert "auth_credential=AuthCredential(" in helper
    assert "auth_type=AuthCredentialTypes.API_KEY" in helper
    assert 'api_key="secret123"' in helper
    imps = "\n".join(tr.imports)
    assert "from google.adk.auth import AuthCredential, AuthCredentialTypes" in imps


def test_render_tool_ref_apihub_with_bearer_auth() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="apihub",
            name="hub",
            apihub_resource_name="projects/p/apis/a",
            auth=AuthSpec(scheme="bearer", credential=(("token", "tok"),)),
        )
    )
    helper = tr.helpers[0]
    assert "auth_credential=AuthCredential(" in helper
    assert "auth_type=AuthCredentialTypes.HTTP" in helper
    assert 'http=HttpAuth(scheme="bearer", credentials=HttpCredentials(token="tok"))' in helper
    imps = "\n".join(tr.imports)
    assert "from google.adk.auth import AuthCredential, AuthCredentialTypes" in imps
    assert "from google.adk.auth.auth_credential import HttpAuth, HttpCredentials" in imps


def test_render_tool_ref_mcp_with_oauth2_auth() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="mcp_toolset",
            name="m",
            transport="http",
            url="https://x/mcp",
            auth=AuthSpec(
                scheme="oauth2",
                credential=(("client_id", "cid"), ("client_secret", "csec")),
            ),
        )
    )
    helper = tr.helpers[0]
    assert "auth_credential=AuthCredential(" in helper
    assert "auth_type=AuthCredentialTypes.OAUTH2" in helper
    assert 'oauth2=OAuth2Auth(client_id="cid", client_secret="csec")' in helper
    imps = "\n".join(tr.imports)
    assert "from google.adk.auth.auth_credential import OAuth2Auth" in imps


def test_render_tool_ref_service_account_auth() -> None:
    tr = render_tool_ref(
        ToolSpec(
            kind="apihub",
            name="hub",
            apihub_resource_name="projects/p/apis/a",
            auth=AuthSpec(
                scheme="service_account", credential=(("use_default_credential", "true"),)
            ),
        )
    )
    helper = tr.helpers[0]
    assert "auth_type=AuthCredentialTypes.SERVICE_ACCOUNT" in helper
    assert "service_account=ServiceAccount(use_default_credential=True)" in helper
    imps = "\n".join(tr.imports)
    assert "from google.adk.auth.auth_credential import ServiceAccount" in imps


# --------------------------------------------------------------------------- #
# (De)serialization of ToolSpec
# --------------------------------------------------------------------------- #
def test_toolspec_roundtrip_function() -> None:
    tool = ToolSpec(
        kind="function",
        name="f",
        params=(("x", "str", None), ("n", "int", "1")),
        docstring="Doc.",
        returns="dict",
        body="return {}",
    )
    assert ToolSpec.from_dict(tool.to_dict()) == tool


def test_toolspec_roundtrip_builtin_with_args() -> None:
    tool = ToolSpec(
        kind="builtin",
        builtin_kind="vertex_ai_search",
        args=(("data_store_id", "ds"),),
    )
    assert ToolSpec.from_dict(tool.to_dict()) == tool


def test_toolspec_roundtrip_agent_tool_and_openapi() -> None:
    at = ToolSpec(kind="agent_tool", target_agent="t")
    assert ToolSpec.from_dict(at.to_dict()) == at
    oa = ToolSpec(kind="openapi", name="ts", spec="{}")
    assert ToolSpec.from_dict(oa.to_dict()) == oa


def test_toolspec_from_legacy_string_maps_to_builtin() -> None:
    spec = ToolSpec.from_dict("google_search")
    assert spec.kind == "builtin"
    assert spec.builtin_kind == "google_search"


def test_toolspec_roundtrip_bigquery_and_spanner() -> None:
    bq = ToolSpec(kind="bigquery", name="bq", args=(("bigquery_tool_config", "cfg"),))
    assert ToolSpec.from_dict(bq.to_dict()) == bq
    sp = ToolSpec(kind="spanner", name="sp")
    assert ToolSpec.from_dict(sp.to_dict()) == sp


def test_toolspec_roundtrip_mcp_toolset() -> None:
    mcp = ToolSpec(
        kind="mcp_toolset",
        name="fs",
        transport="stdio",
        command="npx",
        mcp_args=("-y", "server"),
        tool_filter=("read",),
    )
    assert ToolSpec.from_dict(mcp.to_dict()) == mcp
    sse = ToolSpec(
        kind="mcp_toolset",
        name="r",
        transport="sse",
        url="https://x/sse",
        headers=(("A", "B"),),
    )
    assert ToolSpec.from_dict(sse.to_dict()) == sse


def test_toolspec_roundtrip_apihub_langchain_crewai() -> None:
    hub = ToolSpec(kind="apihub", name="h", apihub_resource_name="projects/p/apis/a")
    assert ToolSpec.from_dict(hub.to_dict()) == hub
    lc = ToolSpec(kind="langchain", import_line="from x import Y", tool_expr="Y()")
    assert ToolSpec.from_dict(lc.to_dict()) == lc
    cw = ToolSpec(
        kind="crewai", import_line="from x import Z", tool_expr="Z()", name="z", description="d"
    )
    assert ToolSpec.from_dict(cw.to_dict()) == cw


def test_toolspec_roundtrip_with_auth() -> None:
    tool = ToolSpec(
        kind="apihub",
        name="h",
        apihub_resource_name="projects/p/apis/a",
        auth=AuthSpec(scheme="apikey", credential=(("api_key", "k"),)),
    )
    restored = ToolSpec.from_dict(tool.to_dict())
    assert restored == tool
    assert restored.auth is not None
    assert restored.auth.scheme == "apikey"


def test_agentspec_with_toolspecs_roundtrips_via_sidecar(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        root="r",
        agents=(
            AgentSpec(name="child", type="llm"),
            AgentSpec(
                name="r",
                type="llm",
                tools=(
                    ToolSpec(kind="function", name="f", docstring="d"),
                    ToolSpec(kind="builtin", builtin_kind="google_search"),
                    ToolSpec(kind="agent_tool", target_agent="child"),
                ),
            ),
        ),
    )
    assert save_model(ws, model)
    reloaded = load_model(ws, "app")
    assert reloaded == model


# --------------------------------------------------------------------------- #
# Tool validation
# --------------------------------------------------------------------------- #
def _model_with(*names: str) -> ProjectModel:
    return ProjectModel(app_name="m", agents=tuple(AgentSpec(name=n, type="llm") for n in names))


def test_validate_tool_rejects_bad_function_name() -> None:
    err = validate_tool_spec(ToolSpec(kind="function", name="1bad"), _model_with("owner"), "owner")
    assert err is not None


def test_validate_tool_rejects_bad_param_type() -> None:
    tool = ToolSpec(kind="function", name="f", params=(("x", "Banana", None),))
    assert validate_tool_spec(tool, _model_with("owner"), "owner") is not None


def test_validate_tool_accepts_union_and_generic_types() -> None:
    tool = ToolSpec(
        kind="function",
        name="f",
        params=(("x", "str | None", None), ("y", "list[int]", None)),
        returns="dict",
    )
    assert validate_tool_spec(tool, _model_with("owner"), "owner") is None


def test_validate_tool_rejects_unknown_builtin() -> None:
    assert (
        validate_tool_spec(ToolSpec(kind="builtin", builtin_kind="nope"), _model_with("o"), "o")
        is not None
    )


def test_validate_tool_vertex_requires_arg() -> None:
    assert (
        validate_tool_spec(
            ToolSpec(kind="builtin", builtin_kind="vertex_ai_search"), _model_with("o"), "o"
        )
        is not None
    )


def test_validate_tool_agent_tool_target_must_exist() -> None:
    model = _model_with("owner")  # no 'ghost'
    err = validate_tool_spec(ToolSpec(kind="agent_tool", target_agent="ghost"), model, "owner")
    assert err is not None


def test_validate_tool_agent_tool_no_self_wrap() -> None:
    model = _model_with("owner")
    err = validate_tool_spec(ToolSpec(kind="agent_tool", target_agent="owner"), model, "owner")
    assert err is not None


def test_validate_tool_openapi_rejects_empty_spec() -> None:
    err = validate_tool_spec(ToolSpec(kind="openapi", name="ts", spec="  "), _model_with("o"), "o")
    assert err is not None


# --- 3b: validation of the new kinds ---------------------------------------- #
def test_validate_tool_bigquery_requires_valid_name() -> None:
    assert validate_tool_spec(ToolSpec(kind="bigquery", name="1bad"), _model_with("o"), "o")
    assert validate_tool_spec(ToolSpec(kind="bigquery", name="bq"), _model_with("o"), "o") is None


def test_validate_tool_spanner_requires_valid_name() -> None:
    assert validate_tool_spec(ToolSpec(kind="spanner", name="bad name"), _model_with("o"), "o")
    assert validate_tool_spec(ToolSpec(kind="spanner", name="sp"), _model_with("o"), "o") is None


def test_validate_tool_mcp_transport_must_be_known() -> None:
    bad = ToolSpec(kind="mcp_toolset", name="m", transport="carrier_pigeon", url="x")
    assert validate_tool_spec(bad, _model_with("o"), "o") is not None


def test_validate_tool_mcp_stdio_requires_command() -> None:
    no_cmd = ToolSpec(kind="mcp_toolset", name="m", transport="stdio")
    assert validate_tool_spec(no_cmd, _model_with("o"), "o") is not None
    ok_cmd = ToolSpec(kind="mcp_toolset", name="m", transport="stdio", command="npx")
    assert validate_tool_spec(ok_cmd, _model_with("o"), "o") is None


def test_validate_tool_mcp_sse_http_require_url() -> None:
    no_url = ToolSpec(kind="mcp_toolset", name="m", transport="sse")
    assert validate_tool_spec(no_url, _model_with("o"), "o") is not None
    ok_url = ToolSpec(kind="mcp_toolset", name="m", transport="http", url="https://x/mcp")
    assert validate_tool_spec(ok_url, _model_with("o"), "o") is None


def test_validate_tool_apihub_requires_resource_name() -> None:
    no_res = ToolSpec(kind="apihub", name="h", apihub_resource_name="")
    assert validate_tool_spec(no_res, _model_with("o"), "o") is not None
    ok_res = ToolSpec(kind="apihub", name="h", apihub_resource_name="projects/p/apis/a")
    assert validate_tool_spec(ok_res, _model_with("o"), "o") is None


def test_validate_tool_langchain_requires_import_and_expr() -> None:
    assert validate_tool_spec(
        ToolSpec(kind="langchain", import_line="", tool_expr="X()"), _model_with("o"), "o"
    )
    assert validate_tool_spec(
        ToolSpec(kind="langchain", import_line="from x import X", tool_expr=""),
        _model_with("o"),
        "o",
    )
    good = ToolSpec(kind="langchain", import_line="from x import X", tool_expr="X()")
    assert validate_tool_spec(good, _model_with("o"), "o") is None


def test_validate_tool_crewai_requires_name() -> None:
    no_name = ToolSpec(kind="crewai", import_line="from x import X", tool_expr="X()", name="")
    assert validate_tool_spec(no_name, _model_with("o"), "o") is not None
    good = ToolSpec(
        kind="crewai", import_line="from x import X", tool_expr="X()", name="x", description="d"
    )
    assert validate_tool_spec(good, _model_with("o"), "o") is None


def test_validate_tool_auth_rejected_on_bigquery_spanner() -> None:
    # bigquery/spanner do not accept auth_scheme/auth_credential (they have credentials_config).
    auth = AuthSpec(scheme="apikey", credential=(("api_key", "k"),))
    assert validate_tool_spec(
        ToolSpec(kind="bigquery", name="bq", auth=auth), _model_with("o"), "o"
    )
    assert validate_tool_spec(ToolSpec(kind="spanner", name="sp", auth=auth), _model_with("o"), "o")


def test_validate_tool_auth_scheme_must_be_known() -> None:
    bad = ToolSpec(
        kind="apihub",
        name="h",
        apihub_resource_name="projects/p/apis/a",
        auth=AuthSpec(scheme="telepathy", credential=(("k", "v"),)),
    )
    assert validate_tool_spec(bad, _model_with("o"), "o") is not None


def test_validate_tool_auth_apikey_requires_api_key_field() -> None:
    bad = ToolSpec(
        kind="apihub",
        name="h",
        apihub_resource_name="projects/p/apis/a",
        auth=AuthSpec(scheme="apikey", credential=(("wrong", "v"),)),
    )
    assert validate_tool_spec(bad, _model_with("o"), "o") is not None


def test_validate_tool_auth_bearer_requires_token_field() -> None:
    bad = ToolSpec(
        kind="mcp_toolset",
        name="m",
        transport="http",
        url="https://x/mcp",
        auth=AuthSpec(scheme="bearer", credential=(("nope", "v"),)),
    )
    assert validate_tool_spec(bad, _model_with("o"), "o") is not None


# --------------------------------------------------------------------------- #
# Module rendering with tools — helpers BEFORE agents, deduped imports
# --------------------------------------------------------------------------- #
def test_render_module_emits_helpers_before_agents_and_dedups_imports() -> None:
    model = ProjectModel(
        app_name="demo",
        root="root",
        agents=(
            AgentSpec(name="child", type="llm", instruction="c"),
            AgentSpec(
                name="root",
                type="llm",
                instruction="use",
                tools=(
                    ToolSpec(kind="function", name="add", docstring="Add."),
                    ToolSpec(kind="builtin", builtin_kind="google_search"),
                    # Two google_search -> deduped import.
                    ToolSpec(kind="builtin", builtin_kind="google_search"),
                    ToolSpec(kind="agent_tool", target_agent="child"),
                ),
            ),
        ),
    )
    src = render_agent_module(model)
    # The tool's def appears before the root agent's definition.
    assert src.index("def add(") < src.index("root = LlmAgent(")
    # google_search appears only once in the import section (deduped/merged).
    import_section = src.split("def add(")[0]
    assert import_section.count("google_search") == 1
    # Imported from the tools root package.
    assert "from google.adk.tools import" in src
    assert "google_search" in src
    # AgentTool references the existing child agent.
    assert "AgentTool(agent=child)" in src
    # The function is referenced bare (ADK auto-wraps it in a FunctionTool).
    assert "tools=[" in src and "add" in src


def test_render_module_topo_orders_agent_tool_target_first() -> None:
    # The agent wrapped by AgentTool must be defined before the agent that wraps it.
    model = ProjectModel(
        app_name="demo",
        root="boss",
        agents=(
            AgentSpec(
                name="boss",
                type="llm",
                instruction="delegate",
                tools=(ToolSpec(kind="agent_tool", target_agent="worker"),),
            ),
            AgentSpec(name="worker", type="llm", instruction="work"),
        ),
    )
    src = render_agent_module(model)
    assert src.index("worker = LlmAgent(") < src.index("boss = LlmAgent(")


# --------------------------------------------------------------------------- #
# Sidecar I/O + regenerate (on disk)
# --------------------------------------------------------------------------- #
def test_load_model_absent_returns_empty(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = load_model(ws, "app")
    assert model.app_name == "app"
    assert model.agents == ()
    assert model.root is None


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        root="a",
        agents=(AgentSpec(name="a", type="llm", instruction="Hi"),),
    )
    assert save_model(ws, model) is True
    assert ws.exists(SIDECAR_PATH)
    reloaded = load_model(ws, "app")
    assert reloaded == model


def test_save_model_idempotent(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(app_name="app", agents=(AgentSpec(name="a", type="llm"),))
    assert save_model(ws, model) is True
    assert save_model(ws, model) is False  # identical content -> nothing changed


def test_load_model_corrupt_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    ws.write(SIDECAR_PATH, "{ not valid json ]")
    with pytest.raises(ValueError, match="Invalid sidecar JSON"):
        load_model(ws, "app")


def test_regenerate_writes_agent_and_init(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        root="solo",
        agents=(AgentSpec(name="solo", type="llm", instruction="Hi"),),
    )
    result = regenerate(ws, model)
    assert result["changed"] is True
    assert ws.exists("agent.py")
    assert ws.exists("__init__.py")
    assert ws.read("__init__.py") == "from . import agent\n"
    assert "root_agent = solo" in ws.read("agent.py")


def test_regenerate_idempotent(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(app_name="app", agents=(AgentSpec(name="a", type="llm"),))
    first = regenerate(ws, model)
    assert first["changed"] is True
    second = regenerate(ws, model)
    assert second["changed"] is False


def test_regenerate_cycle_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        agents=(
            AgentSpec(name="x", type="sequential", sub_agents=("y",)),
            AgentSpec(name="y", type="sequential", sub_agents=("x",)),
        ),
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        regenerate(ws, model)


# --------------------------------------------------------------------------- #
# ruff format stability — the generated file must already be formatted
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    """Locate the ruff executable in the current environment (venv or PATH)."""
    # Prefer the ruff that lives next to the current Python interpreter (venv).
    import sys

    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _assert_ruff_isort_clean(src: str, tmp_path: Path, label: str) -> None:
    """Check that ``ruff check --select I`` (isort) passes (exit 0) on *src*.

    The generated ``agent.py`` must not only be *format*-clean: its import lines must also
    be *isort*-clean (names sorted inside each ``from X import ...``, module order). Before
    the fix, the line ``from google.adk.agents import LlmAgent, BaseAgent`` (canonical ADK
    order) triggered ``I001``.
    """
    gen_file = tmp_path / f"{label}_isort.py"
    gen_file.write_text(src, encoding="utf-8")

    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff not found in the environment — isort test ignored")

    result = subprocess.run(
        [ruff, "check", "--select", "I", str(gen_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff check --select I (isort) failed for case '{label}'.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Generated source:\n{src}"
    )


def _assert_ruff_format_stable(src: str, tmp_path: Path, label: str) -> None:
    """Check that the generated output is **already formatted AND isort-clean**.

    Runs ``ruff format --check`` (formatting idempotence) *and* ``ruff check --select I``
    (import sorting) on *src* — both must pass (exit 0).
    """
    gen_file = tmp_path / f"{label}.py"
    gen_file.write_text(src, encoding="utf-8")

    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff not found in the environment — format test ignored")

    result = subprocess.run(
        [ruff, "format", "--check", str(gen_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff format --check failed for case '{label}'.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Generated source:\n{src}"
    )

    # The generated file must also be isort-clean (not only format-clean).
    _assert_ruff_isort_clean(src, tmp_path, label)


def test_render_format_stable_custom_llm_workflow(tmp_path: Path) -> None:
    """The module generated with a custom + llm + workflow is stable for ruff format."""
    model = ProjectModel(
        app_name="demo",
        root="pipe",
        agents=(
            AgentSpec(name="my_custom", type="custom", description="Custom agent"),
            AgentSpec(name="llm_one", type="llm", instruction="Think"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("my_custom", "llm_one")),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "custom_llm_workflow")


def test_render_agent_import_line_names_sorted_for_isort() -> None:
    """The agent-class import line sorts its names (custom + llm -> BaseAgent, LlmAgent).

    Regression: ``_needed_agent_imports`` returns the canonical ADK order
    (``LlmAgent, ..., BaseAgent``); the emission must nonetheless sort the names to satisfy
    isort (``BaseAgent, LlmAgent``).
    """
    model = ProjectModel(
        app_name="demo",
        root="router",
        agents=(
            AgentSpec(name="writer", type="llm", instruction="Write"),
            AgentSpec(name="router", type="custom", description="Custom router"),
        ),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents import BaseAgent, LlmAgent\n" in src
    # The unsorted canonical order must NOT appear.
    assert "import LlmAgent, BaseAgent" not in src


def test_render_isort_clean_custom_plus_llm(tmp_path: Path) -> None:
    """Dedicated case: a ``custom`` agent + an ``llm`` (combo ``LlmAgent`` + ``BaseAgent``).

    Before the fix, the unsorted canonical order ``import LlmAgent, BaseAgent`` triggered
    ``I001``. After the fix, the output is sorted (``BaseAgent, LlmAgent``). We check
    explicitly that ``ruff check --select I`` is clean (exit 0) on the generated output.
    """
    model = ProjectModel(
        app_name="demo",
        root="router",
        agents=(
            AgentSpec(name="writer", type="llm", instruction="Write"),
            AgentSpec(name="router", type="custom", description="Custom router"),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_isort_clean(src, tmp_path, "custom_plus_llm")


def test_render_format_stable_llm_only(tmp_path: Path) -> None:
    """The module generated with llm agents only is stable for ruff format."""
    model = ProjectModel(
        app_name="demo",
        root="solo",
        agents=(
            AgentSpec(
                name="solo",
                type="llm",
                instruction="Hi",
                description="A solo agent",
                output_key="result",
                tools=("google_search",),
            ),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "llm_only")


def test_render_format_stable_function_tools_and_custom(tmp_path: Path) -> None:
    """Function tools (top-level defs) + custom agent + agent_tool: stable for ruff format."""
    model = ProjectModel(
        app_name="demo",
        root="root",
        agents=(
            AgentSpec(name="aux", type="custom", description="Aux agent"),
            AgentSpec(name="child", type="llm", instruction="child"),
            AgentSpec(
                name="root",
                type="llm",
                instruction="Coordinate.",
                description="Root coordinator",
                output_key="out",
                tools=(
                    ToolSpec(
                        kind="function",
                        name="add",
                        params=(("a", "int", None), ("b", "int", "0")),
                        docstring="Add two integers.",
                        returns="dict",
                        # The body is rendered verbatim: it must already be ruff-clean (the
                        # toolkit does not reformat user code). Double quotes -> stable.
                        body='return {"sum": a + b}',
                    ),
                    ToolSpec(kind="long_running", name="poll", docstring="Poll a job."),
                    ToolSpec(kind="builtin", builtin_kind="google_search"),
                    ToolSpec(kind="agent_tool", target_agent="child"),
                ),
            ),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "function_tools_and_custom")


def test_render_format_stable_all_tool_kinds(tmp_path: Path) -> None:
    """The six tool kinds (3a) together: output already formatted for ruff."""
    model = ProjectModel(
        app_name="demo",
        root="root",
        agents=(
            AgentSpec(name="child", type="llm", instruction="child"),
            AgentSpec(
                name="root",
                type="llm",
                instruction="Use every tool kind.",
                tools=(
                    ToolSpec(
                        kind="function",
                        name="compute",
                        params=(("value", "str", None),),
                        docstring="Compute.",
                        returns="dict",
                        body="return {}",
                    ),
                    ToolSpec(kind="long_running", name="watch", docstring="Watch."),
                    ToolSpec(kind="builtin", builtin_kind="google_search"),
                    ToolSpec(
                        kind="builtin",
                        builtin_kind="vertex_ai_search",
                        args=(("data_store_id", "projects/p/locations/l/dataStores/d"),),
                    ),
                    ToolSpec(kind="agent_tool", target_agent="child"),
                    ToolSpec(
                        kind="openapi",
                        name="petstore",
                        spec='{"openapi": "3.0.0", "info": {"title": "t", "version": "1"}}',
                    ),
                ),
            ),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "all_tool_kinds")


def _all_3b_model() -> ProjectModel:
    """Model exercising all the 3b kinds + auth, shared by ast.parse and ruff format."""
    return ProjectModel(
        app_name="demo",
        root="root",
        agents=(
            AgentSpec(
                name="root",
                type="llm",
                instruction="Use 3b toolsets.",
                tools=(
                    ToolSpec(kind="bigquery", name="bq"),
                    ToolSpec(kind="spanner", name="sp"),
                    ToolSpec(
                        kind="mcp_toolset",
                        name="fs",
                        transport="stdio",
                        command="npx",
                        mcp_args=("-y", "@modelcontextprotocol/server-filesystem", "/data"),
                        tool_filter=("read_file", "list_directory"),
                    ),
                    ToolSpec(
                        kind="mcp_toolset",
                        name="remote",
                        transport="http",
                        url="https://api.example.com/mcp",
                        headers=(("Authorization", "Bearer tok"),),
                    ),
                    ToolSpec(
                        kind="apihub",
                        name="hub",
                        apihub_resource_name="projects/p/locations/l/apis/a",
                        auth=AuthSpec(scheme="apikey", credential=(("api_key", "secret"),)),
                    ),
                    ToolSpec(
                        kind="openapi",
                        name="petstore",
                        spec='{"openapi": "3.0.0", "info": {"title": "t", "version": "1"}}',
                        auth=AuthSpec(scheme="bearer", credential=(("token", "tok"),)),
                    ),
                    ToolSpec(
                        kind="langchain",
                        import_line="from langchain_community.tools import WikipediaQueryRun",
                        tool_expr="WikipediaQueryRun(api_wrapper=wiki_wrapper)",
                    ),
                    ToolSpec(
                        kind="crewai",
                        import_line="from crewai_tools import SerperDevTool",
                        tool_expr="SerperDevTool()",
                        name="serper",
                        description="Web search.",
                    ),
                ),
            ),
        ),
    )


def test_render_3b_module_is_valid_python_ast() -> None:
    """The module generated with all the 3b kinds + auth is valid Python (ast.parse).

    We do NOT import it (the extras are not installed in CI); we just check that it parses
    syntactically and that the expected imports/refs are present.
    """
    import ast

    src = render_agent_module(_all_3b_model())
    ast.parse(src)  # raises SyntaxError if the rendering is broken
    # Key imports present.
    assert "from google.adk.tools.bigquery import BigQueryToolset" in src
    assert "from google.adk.tools.spanner import SpannerToolset" in src
    assert "from google.adk.tools.mcp_tool import" in src
    assert "from mcp import StdioServerParameters" in src
    assert "from google.adk.tools.apihub_tool import APIHubToolset" in src
    assert "from google.adk.tools.langchain_tool import LangchainTool" in src
    assert "from google.adk.tools.crewai_tool import CrewaiTool" in src
    assert "from google.adk.auth import AuthCredential, AuthCredentialTypes" in src
    # The toolset helpers are defined before the root agent.
    assert src.index("bq = BigQueryToolset(") < src.index("root = LlmAgent(")
    # The user import_lines appear (verbatim).
    assert "from langchain_community.tools import WikipediaQueryRun" in src
    assert "from crewai_tools import SerperDevTool" in src


def test_render_format_stable_all_3b_kinds(tmp_path: Path) -> None:
    """All the 3b kinds + auth together: output already formatted for ruff."""
    src = render_agent_module(_all_3b_model())
    _assert_ruff_format_stable(src, tmp_path, "all_3b_kinds")


# --------------------------------------------------------------------------- #
# Callbacks (guardrails, P4c) — (de)serialization, validation, mutation, rendering
# --------------------------------------------------------------------------- #
def _kw_callback(refusal: str = "No.") -> CallbackSpec:
    return CallbackSpec(
        hook="before_model",
        policy="block_keywords",
        params=(("keywords", "bomb,hack"), ("refusal", refusal)),
    )


def test_callbackspec_roundtrip() -> None:
    """CallbackSpec.to_dict / from_dict round-trip (hook + policy.kind + params)."""
    cb = _kw_callback()
    data = cb.to_dict()
    assert data == {
        "hook": "before_model",
        "policy": {"kind": "block_keywords", "keywords": "bomb,hack", "refusal": "No."},
    }
    assert CallbackSpec.from_dict(data) == cb


def test_callbackspec_kwarg_name() -> None:
    """kwarg_name() maps the hook to the real LlmAgent kwarg (_callback suffix)."""
    assert _kw_callback().kwarg_name() == "before_model_callback"
    assert CallbackSpec(hook="before_tool", policy="block_tool").kwarg_name() == (
        "before_tool_callback"
    )


def test_agentspec_serializes_callbacks_and_max_llm_calls() -> None:
    """An LlmAgent serializes callbacks + max_llm_calls; from_dict re-reads them."""
    spec = AgentSpec(name="a", type="llm", callbacks=(_kw_callback(),), max_llm_calls=42)
    data = spec.to_dict()
    assert data["callbacks"] == [_kw_callback().to_dict()]
    assert data["max_llm_calls"] == 42
    back = AgentSpec.from_dict(data)
    assert back.callbacks == (_kw_callback(),)
    assert back.max_llm_calls == 42


def test_agentspec_omits_empty_callbacks_and_max_llm_calls() -> None:
    """Without a callback or cap, the keys are NOT emitted (backward compat)."""
    data = AgentSpec(name="a", type="llm").to_dict()
    assert "callbacks" not in data
    assert "max_llm_calls" not in data


def test_validate_callback_spec_ok_and_errors() -> None:
    """validate_callback_spec accepts valid policies and rejects invalid ones."""
    assert validate_callback_spec(_kw_callback()) is None
    # Hook incompatible with the policy (block_keywords is before_model only).
    bad_hook = CallbackSpec(
        hook="before_tool", policy="block_keywords", params=(("keywords", "x"),)
    )
    assert "is not compatible" in (validate_callback_spec(bad_hook) or "")
    # block_keywords without keywords.
    no_kw = CallbackSpec(hook="before_model", policy="block_keywords")
    assert "keywords" in (validate_callback_spec(no_kw) or "")
    # max_input_chars with a non-integer max_chars.
    bad_max = CallbackSpec(
        hook="before_model", policy="max_input_chars", params=(("max_chars", "abc"),)
    )
    assert "max_chars" in (validate_callback_spec(bad_max) or "")
    # block_tool without denylist.
    no_dl = CallbackSpec(hook="before_tool", policy="block_tool")
    assert "denylist" in (validate_callback_spec(no_dl) or "")


def test_add_or_replace_callback_one_per_hook() -> None:
    """A second callback on the same hook REPLACES the first (one kwarg per hook)."""
    spec = AgentSpec(name="a", type="llm")
    spec = add_or_replace_callback(spec, _kw_callback(refusal="first"))
    spec = add_or_replace_callback(spec, _kw_callback(refusal="second"))
    assert len(spec.callbacks) == 1
    assert spec.callbacks[0].param("refusal") == "second"
    # A different hook is added (does not replace).
    spec = add_or_replace_callback(
        spec, CallbackSpec(hook="before_tool", policy="block_tool", params=(("denylist", "rm"),))
    )
    assert {c.hook for c in spec.callbacks} == {"before_model", "before_tool"}


def test_render_block_keywords_callback() -> None:
    """block_keywords renders a before_model function attached via the real kwarg + helpers."""
    spec = AgentSpec(name="guarded", type="llm", callbacks=(_kw_callback(),))
    src = render_agent_module(ProjectModel(app_name="app", root="guarded", agents=(spec,)))
    # The guardrail function is defined and attached via the real kwarg.
    assert "def _guard_before_model_guarded(callback_context, llm_request):" in src
    assert "before_model_callback=_guard_before_model_guarded" in src
    # Shared helpers emitted.
    assert "def _user_text(llm_request) -> str:" in src
    assert "def _refuse(message: str) -> LlmResponse:" in src
    # The list of blocked words + the refusal are present.
    assert '["bomb", "hack"]' in src
    assert 'return _refuse("No.")' in src


def test_render_block_tool_callback() -> None:
    """block_tool renders a before_tool function short-circuiting the tool (dict)."""
    cb = CallbackSpec(
        hook="before_tool", policy="block_tool", params=(("denylist", "delete_db,drop"),)
    )
    spec = AgentSpec(name="guarded", type="llm", callbacks=(cb,))
    src = render_agent_module(ProjectModel(app_name="app", root="guarded", agents=(spec,)))
    assert "def _guard_before_tool_guarded(tool, args, tool_context):" in src
    assert "before_tool_callback=_guard_before_tool_guarded" in src
    assert '["delete_db", "drop"]' in src
    assert "if tool.name in denylist:" in src
    # block_tool does NOT need the before_model helpers.
    assert "_refuse" not in src


def test_render_max_input_chars_callback() -> None:
    """max_input_chars renders a before_model function refusing beyond N characters."""
    cb = CallbackSpec(hook="before_model", policy="max_input_chars", params=(("max_chars", "500"),))
    spec = AgentSpec(name="g", type="llm", callbacks=(cb,))
    src = render_agent_module(ProjectModel(app_name="app", root="g", agents=(spec,)))
    assert "max_chars = 500" in src
    assert "if len(_user_text(llm_request)) > max_chars:" in src


def test_max_llm_calls_not_rendered_in_agent_py() -> None:
    """max_llm_calls is a RunConfig setting: it must NOT appear in agent.py."""
    spec = AgentSpec(name="a", type="llm", max_llm_calls=7)
    src = render_agent_module(ProjectModel(app_name="app", root="a", agents=(spec,)))
    assert "max_llm_calls" not in src


def test_render_callbacks_ast_parse(tmp_path: Path) -> None:
    """The module with the three policies (separate agents) is ast-parseable."""
    import ast

    a1 = AgentSpec(name="kw", type="llm", callbacks=(_kw_callback(),))
    a2 = AgentSpec(
        name="mx",
        type="llm",
        callbacks=(
            CallbackSpec(
                hook="before_model", policy="max_input_chars", params=(("max_chars", "9"),)
            ),
        ),
    )
    a3 = AgentSpec(
        name="tl",
        type="llm",
        callbacks=(
            CallbackSpec(hook="before_tool", policy="block_tool", params=(("denylist", "rm"),)),
        ),
    )
    src = render_agent_module(ProjectModel(app_name="app", root="kw", agents=(a1, a2, a3)))
    ast.parse(src)  # does not raise


def test_render_callbacks_format_stable(tmp_path: Path) -> None:
    """The code generated with guardrails is already ruff-format + isort clean (the 3 policies)."""
    a1 = AgentSpec(name="kw", type="llm", callbacks=(_kw_callback(refusal="I cannot help."),))
    a2 = AgentSpec(
        name="mx",
        type="llm",
        callbacks=(
            CallbackSpec(
                hook="before_model", policy="max_input_chars", params=(("max_chars", "2000"),)
            ),
        ),
    )
    a3 = AgentSpec(
        name="tl",
        type="llm",
        callbacks=(
            CallbackSpec(
                hook="before_tool", policy="block_tool", params=(("denylist", "rm,drop"),)
            ),
        ),
    )
    src = render_agent_module(ProjectModel(app_name="app", root="kw", agents=(a1, a2, a3)))
    _assert_ruff_format_stable(src, tmp_path, "callbacks_all_policies")


def test_render_callbacks_with_tools_and_gcc_format_stable(tmp_path: Path) -> None:
    """An agent with tool + gcc + callback stays ruff-stable (import merge)."""
    tool = ToolSpec(
        kind="function",
        name="greet",
        params=(("name", "str", None),),
        docstring="Greet",
        returns="str",
        body='return f"hi {name}"',
    )
    spec = AgentSpec(
        name="full",
        type="llm",
        instruction="Be helpful",
        tools=(tool,),
        callbacks=(_kw_callback(refusal="Refused."),),
    )
    src = render_agent_module(ProjectModel(app_name="app", root="full", agents=(spec,)))
    _assert_ruff_format_stable(src, tmp_path, "callbacks_with_tools")
