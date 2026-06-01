"""Tests unitaires du renderer pur ``project_model`` (aucune construction ADK in-process).

On assert sur la **chaîne source** générée (sûr sous ``-W error::DeprecationWarning``,
puisqu'on ne construit aucun agent workflow déprécié ici). La preuve fonctionnelle
(instanciation réelle des objets ADK) est faite dans ``test_agents.py`` via un subprocess.
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
    ProjectModel,
    ToolRender,
    ToolSpec,
    add_or_update_agent,
    load_model,
    regenerate,
    render_agent_module,
    render_tool_ref,
    save_model,
    set_root,
    topological_order,
    validate_spec,
    validate_tool_spec,
)
from adk_toolkit_mcp.workspace import Workspace


# --------------------------------------------------------------------------- #
# Dataclasses + (dé)sérialisation
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
# Mutations immuables
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
    # Position préservée, pas de doublon.
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
# Tri topologique + cycles
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
# Rendu source — par type
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
    # description vide / output_key None / tools vide / sub_agents vide -> omis.
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
    # Les noms importés sont triés (isort ``I001``), pas dans l'ordre canonique ADK :
    # ``LlmAgent, ParallelAgent, SequentialAgent`` (alphabétique).
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
    # async generator no-op : return suivi d'un yield inatteignable.
    assert "yield" in src
    assert 'my_custom = MyCustomAgent(name="my_custom", description="D")' in src


# --------------------------------------------------------------------------- #
# remote_a2a (P4b) — proxy RemoteA2aAgent
# --------------------------------------------------------------------------- #
def test_render_remote_a2a_emits_call_and_submodule_import() -> None:
    """``remote_a2a`` rend ``RemoteA2aAgent(name=..., agent_card="...")`` + l'import sous-module.

    ⚠️ En 2.1.0, ``RemoteA2aAgent`` N'EST PAS dans ``google.adk.agents`` : l'import DOIT être
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
    # PAS d'import erroné depuis google.adk.agents (RemoteA2aAgent n'y est pas).
    assert "from google.adk.agents import RemoteA2aAgent" not in src
    assert "remote_helper = RemoteA2aAgent(" in src
    assert 'name="remote_helper"' in src
    assert 'agent_card="http://localhost:8001/.well-known/agent-card.json"' in src
    assert 'description="Remote helper."' in src
    assert "root_agent = remote_helper" in src


def test_render_remote_a2a_composes_as_sub_agent_topo_order() -> None:
    """Un ``remote_a2a`` peut être ``sub_agent`` d'un autre agent ; défini AVANT le parent."""
    model = ProjectModel(
        app_name="demo",
        root="router",
        agents=(
            # Déclaré APRÈS le parent dans le modèle pour prouver le tri topologique.
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
    # Deux classes d'agents importées chacune depuis leur module (isort trie par module).
    assert "from google.adk.agents import LlmAgent" in src
    assert "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent" in src
    # La définition du proxy précède celle du parent (dépendance avant dépendant).
    assert src.index("remote_helper = RemoteA2aAgent(") < src.index("router = LlmAgent(")
    assert "sub_agents=[remote_helper]" in src


def test_render_remote_a2a_no_description_omits_kwarg() -> None:
    """Sans description, le kwarg ``description=`` est omis (uniquement name + agent_card)."""
    model = ProjectModel(
        app_name="demo",
        root="r",
        agents=(AgentSpec(name="r", type="remote_a2a", agent_card="http://h/a2a"),),
    )
    src = render_agent_module(model)
    assert "r = RemoteA2aAgent(" in src
    assert "description=" not in src


def test_remote_a2a_spec_roundtrip_serializes_agent_card() -> None:
    """Le champ ``agent_card`` survit à un aller-retour to_dict/from_dict (forme sidecar)."""
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
    """``remote_a2a`` sans agent_card est rejeté (message actionnable)."""
    bad = AgentSpec(name="r", type="remote_a2a", agent_card="")
    error = validate_spec(bad)
    assert error is not None
    assert "agent_card" in error
    good = AgentSpec(name="r", type="remote_a2a", agent_card="http://h/a2a")
    assert validate_spec(good) is None


def test_render_remote_a2a_format_and_isort_stable(tmp_path: Path) -> None:
    """Le module généré avec un remote_a2a + un llm parent est format- ET isort-clean."""
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
    assert "Aucun agent" in src


def test_render_root_missing_emits_comment_not_assignment() -> None:
    model = ProjectModel(
        app_name="demo",
        root="ghost",  # n'existe pas
        agents=(AgentSpec(name="real", type="llm"),),
    )
    src = render_agent_module(model)
    assert "root_agent = ghost" not in src
    assert "introuvable" in src


def test_render_imports_only_used_classes() -> None:
    # Seulement des llm -> n'importe pas Sequential/Parallel/Loop/BaseAgent.
    model = ProjectModel(app_name="demo", agents=(AgentSpec(name="a", type="llm"),))
    src = render_agent_module(model)
    line = next(line for line in src.splitlines() if line.startswith("from google.adk.agents"))
    assert "LlmAgent" in line
    assert "SequentialAgent" not in line
    assert "BaseAgent" not in line


# --------------------------------------------------------------------------- #
# Rendu des outils — render_tool_ref (passe 3a)
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
    assert tr.ref == "add"  # ADK auto-wrappe la fonction en FunctionTool.
    assert tr.imports == ()  # un plain function n'importe rien.
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
    # Forme héritée P1 : une chaîne reste une référence bare, sans import ni helper.
    tr = render_tool_ref("already_imported_tool")
    assert tr.ref == "already_imported_tool"
    assert tr.imports == ()
    assert tr.helpers == ()


# --------------------------------------------------------------------------- #
# Rendu des outils — render_tool_ref (passe 3b : toolsets à dépendance optionnelle)
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
    # Les args sont des EXPRESSIONS source (pas des chaînes littérales) -> rendues telles quelles.
    assert "bigquery_tool_config=my_cfg" in tr.helpers[0]


def test_render_tool_ref_spanner_builds_toolset() -> None:
    tr = render_tool_ref(ToolSpec(kind="spanner", name="sp"))
    assert tr.ref == "sp"
    assert tr.imports == ("from google.adk.tools.spanner import SpannerToolset",)
    assert tr.helpers[0].startswith("sp = SpannerToolset(")


def test_render_tool_ref_mcp_stdio_single_arg_no_trailing_comma() -> None:
    # Le toolset stdio imbrique 3 appels : même minimal il dépasse 100 cols et se replie.
    # Règle ruff : un call à argument UNIQUE qui se replie n'ajoute PAS de virgule finale.
    tr = render_tool_ref(ToolSpec(kind="mcp_toolset", name="fs", transport="stdio", command="srv"))
    assert tr.ref == "fs"
    imps = "\n".join(tr.imports)
    assert "from google.adk.tools.mcp_tool import" in imps
    assert "McpToolset" in imps and "StdioConnectionParams" in imps
    assert "from mcp import StdioServerParameters" in imps
    helper = tr.helpers[0]
    # Forme exacte ``ruff format`` : argument unique éclaté, sans virgule finale.
    assert helper == (
        "fs = McpToolset(\n"
        "    connection_params=StdioConnectionParams(\n"
        '        server_params=StdioServerParameters(command="srv", args=[])\n'
        "    )\n"
        ")\n"
    )


def test_render_tool_ref_mcp_stdio_long_folds_ruff_stable() -> None:
    # Cas long : le renderer replie (récursivement) ; on vérifie les fragments clés + l'ordre.
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
    # Aucune ligne ne dépasse la limite ruff.
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
    assert "StdioServerParameters" not in imps  # pas de stdio import pour sse
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
# Rendu de l'auth (set_auth) — auth_credential= sur les toolsets compatibles
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
# (Dé)sérialisation des ToolSpec
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
# Validation des outils
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
    model = _model_with("owner")  # pas de 'ghost'
    err = validate_tool_spec(ToolSpec(kind="agent_tool", target_agent="ghost"), model, "owner")
    assert err is not None


def test_validate_tool_agent_tool_no_self_wrap() -> None:
    model = _model_with("owner")
    err = validate_tool_spec(ToolSpec(kind="agent_tool", target_agent="owner"), model, "owner")
    assert err is not None


def test_validate_tool_openapi_rejects_empty_spec() -> None:
    err = validate_tool_spec(ToolSpec(kind="openapi", name="ts", spec="  "), _model_with("o"), "o")
    assert err is not None


# --- 3b : validation des nouveaux genres ----------------------------------- #
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
    # bigquery/spanner n'acceptent pas auth_scheme/auth_credential (ils ont credentials_config).
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
# Rendu de module avec outils — helpers AVANT les agents, imports dédupés
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
                    # Deux google_search -> import dédupé.
                    ToolSpec(kind="builtin", builtin_kind="google_search"),
                    ToolSpec(kind="agent_tool", target_agent="child"),
                ),
            ),
        ),
    )
    src = render_agent_module(model)
    # Le def de l'outil apparaît avant la définition de l'agent root.
    assert src.index("def add(") < src.index("root = LlmAgent(")
    # google_search n'apparaît qu'une seule fois dans la section d'imports (dédupé/fusionné).
    import_section = src.split("def add(")[0]
    assert import_section.count("google_search") == 1
    # Importé depuis le package root des outils.
    assert "from google.adk.tools import" in src
    assert "google_search" in src
    # AgentTool référence l'agent enfant existant.
    assert "AgentTool(agent=child)" in src
    # La fonction est référencée bare (ADK l'auto-wrappe en FunctionTool).
    assert "tools=[" in src and "add" in src


def test_render_module_topo_orders_agent_tool_target_first() -> None:
    # L'agent enveloppé par AgentTool doit être défini avant l'agent qui l'enveloppe.
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
# Sidecar I/O + regenerate (sur disque)
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
    assert save_model(ws, model) is False  # contenu identique -> rien changé


def test_load_model_corrupt_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    ws.write(SIDECAR_PATH, "{ not valid json ]")
    with pytest.raises(ValueError, match="JSON invalide"):
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
# Stabilité de format ruff — le fichier généré doit être déjà formaté
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    """Localise l'exécutable ruff dans l'environnement courant (venv ou PATH)."""
    # Prefer the ruff that lives next to the current Python interpreter (venv).
    import sys

    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _assert_ruff_isort_clean(src: str, tmp_path: Path, label: str) -> None:
    """Vérifie que ``ruff check --select I`` (isort) passe (exit 0) sur *src*.

    Le ``agent.py`` généré ne doit pas seulement être *format*-clean : ses lignes d'import
    doivent aussi être *isort*-clean (noms triés à l'intérieur de chaque ``from X import ...``,
    ordre des modules). Avant le correctif, la ligne ``from google.adk.agents import LlmAgent,
    BaseAgent`` (ordre canonique ADK) déclenchait ``I001``.
    """
    gen_file = tmp_path / f"{label}_isort.py"
    gen_file.write_text(src, encoding="utf-8")

    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff introuvable dans l'environnement — test isort ignoré")

    result = subprocess.run(
        [ruff, "check", "--select", "I", str(gen_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff check --select I (isort) a échoué pour le cas '{label}'.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Source générée :\n{src}"
    )


def _assert_ruff_format_stable(src: str, tmp_path: Path, label: str) -> None:
    """Vérifie que la sortie générée est **déjà formatée ET isort-clean**.

    Lance ``ruff format --check`` (idempotence du formatage) *et* ``ruff check --select I``
    (tri des imports) sur *src* — les deux doivent passer (exit 0).
    """
    gen_file = tmp_path / f"{label}.py"
    gen_file.write_text(src, encoding="utf-8")

    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff introuvable dans l'environnement — test de format ignoré")

    result = subprocess.run(
        [ruff, "format", "--check", str(gen_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff format --check a échoué pour le cas '{label}'.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Source générée :\n{src}"
    )

    # Le fichier généré doit aussi être isort-clean (pas seulement format-clean).
    _assert_ruff_isort_clean(src, tmp_path, label)


def test_render_format_stable_custom_llm_workflow(tmp_path: Path) -> None:
    """Le module généré avec un custom + llm + workflow est stable pour ruff format."""
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
    """La ligne d'import des classes d'agents trie ses noms (custom + llm -> BaseAgent, LlmAgent).

    Régression : ``_needed_agent_imports`` renvoie l'ordre canonique ADK
    (``LlmAgent, ..., BaseAgent``) ; l'émission doit néanmoins trier les noms pour satisfaire
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
    # L'ordre canonique non trié ne doit PAS apparaître.
    assert "import LlmAgent, BaseAgent" not in src


def test_render_isort_clean_custom_plus_llm(tmp_path: Path) -> None:
    """Cas dédié : un agent ``custom`` + un ``llm`` (combo ``LlmAgent`` + ``BaseAgent``).

    Avant le correctif, l'ordre canonique non trié ``import LlmAgent, BaseAgent`` déclenchait
    ``I001``. Après correctif, la sortie est triée (``BaseAgent, LlmAgent``). On vérifie
    explicitement que ``ruff check --select I`` est clean (exit 0) sur la sortie générée.
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
    """Le module généré avec des agents llm uniquement est stable pour ruff format."""
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
    """Function tools (defs top-level) + agent custom + agent_tool : stable pour ruff format."""
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
                        # Le corps est rendu verbatim : il doit déjà être ruff-clean (le toolkit
                        # ne reformate pas le code utilisateur). Guillemets doubles -> stable.
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
    """Les six genres d'outils (3a) ensemble : sortie déjà formatée pour ruff."""
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
    """Modèle exerçant tous les genres 3b + l'auth, partagé par ast.parse et ruff format."""
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
    """Le module généré avec tous les genres 3b + auth est du Python valide (ast.parse).

    On NE l'importe PAS (les extras ne sont pas installés en CI) ; on vérifie juste qu'il
    s'analyse syntaxiquement et que les imports/refs attendus sont présents.
    """
    import ast

    src = render_agent_module(_all_3b_model())
    ast.parse(src)  # lève SyntaxError si le rendu est cassé
    # Imports clés présents.
    assert "from google.adk.tools.bigquery import BigQueryToolset" in src
    assert "from google.adk.tools.spanner import SpannerToolset" in src
    assert "from google.adk.tools.mcp_tool import" in src
    assert "from mcp import StdioServerParameters" in src
    assert "from google.adk.tools.apihub_tool import APIHubToolset" in src
    assert "from google.adk.tools.langchain_tool import LangchainTool" in src
    assert "from google.adk.tools.crewai_tool import CrewaiTool" in src
    assert "from google.adk.auth import AuthCredential, AuthCredentialTypes" in src
    # Les helpers de toolset sont définis avant l'agent root.
    assert src.index("bq = BigQueryToolset(") < src.index("root = LlmAgent(")
    # Les user import_lines apparaissent (verbatim).
    assert "from langchain_community.tools import WikipediaQueryRun" in src
    assert "from crewai_tools import SerperDevTool" in src


def test_render_format_stable_all_3b_kinds(tmp_path: Path) -> None:
    """Tous les genres 3b + auth ensemble : sortie déjà formatée pour ruff."""
    src = render_agent_module(_all_3b_model())
    _assert_ruff_format_stable(src, tmp_path, "all_3b_kinds")
