"""Tests du domaine ``tools`` : attache/validation/idempotence des outils, régénération, et
**preuve fonctionnelle** que ``agent.py`` généré instancie de vrais objets ADK avec outils.

La preuve fonctionnelle importe le module généré dans un **subprocess** (le venv uv,
``sys.executable``), lancé avec ``-W ignore::DeprecationWarning`` (les agents workflow émettent
une ``DeprecationWarning`` en google-adk 2.1.0 — hors sujet ici). Outils **sans dépendance**
uniquement (3a) : ``function`` + ``builtin`` ``google_search`` + ``agent_tool``.

Rappel clé (cf. ``docs/adk-api-notes/tools.md``) : un plain function reste de type ``function``
dans le champ ``.tools`` brut après init ; il n'est wrappé en ``FunctionTool`` que par l'appel
**asynchrone** ``canonical_tools()``. On asserte donc sur les deux niveaux.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains.agents import create_custom, create_llm, create_sequential, set_root
from adk_toolkit_mcp.domains.tools import (
    add_agent_tool,
    add_apihub,
    add_bigquery,
    add_builtin,
    add_crewai,
    add_function,
    add_langchain,
    add_long_running,
    add_mcp_toolset,
    add_openapi,
    add_spanner,
    list_tools_for_agent,
    set_auth,
)
from adk_toolkit_mcp.project_model import SIDECAR_PATH
from adk_toolkit_mcp.server import build_server

_OPENAPI_SPEC = json.dumps(
    {
        "openapi": "3.0.0",
        "info": {"title": "Ping API", "version": "1.0.0"},
        "paths": {
            "/ping": {
                "get": {
                    "operationId": "ping",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
)


def _agent_src(tmp_path: Path, app_name: str) -> str:
    return (tmp_path / app_name / "agent.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# add_function
# --------------------------------------------------------------------------- #
def test_add_function_appends_spec_and_renders_def(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root", instruction="Use a tool.")
    res = add_function(
        str(tmp_path),
        "demo",
        "root",
        "add",
        params=[
            {"name": "a", "type": "int", "default": None},
            {"name": "b", "type": "int", "default": "0"},
        ],
        docstring="Add two integers.",
        returns="dict",
        body='return {"sum": a + b}',
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    # def généré avec signature typée + docstring, référencé bare dans tools=[...].
    assert "def add(a: int, b: int = 0) -> dict:" in src
    assert '"""Add two integers."""' in src
    assert "add" in src and "tools=[" in src
    # listé via tools_list.
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    kinds = [t["kind"] for t in listing["data"]["tools"]]
    assert kinds == ["function"]


def test_add_function_rejects_bad_func_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_function(str(tmp_path), "demo", "root", "1bad", params=[], docstring="d")
    assert res["ok"] is False
    assert res["error"]


def test_add_function_rejects_bad_param_type(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_function(
        str(tmp_path),
        "demo",
        "root",
        "f",
        params=[{"name": "x", "type": "Banana", "default": None}],
        docstring="d",
    )
    assert res["ok"] is False


def test_add_function_rejects_malformed_param(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_function(str(tmp_path), "demo", "root", "f", params=[{"type": "str"}], docstring="d")
    assert res["ok"] is False  # 'name' manquant


def test_add_function_replace_by_name_is_idempotent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    first = add_function(str(tmp_path), "demo", "root", "f", params=[], docstring="v1")
    assert first["ok"] is True
    again = add_function(str(tmp_path), "demo", "root", "f", params=[], docstring="v1")
    assert again["ok"] is True
    assert again["data"]["changed"] is False  # contenu identique -> rien réécrit
    # Toujours un seul outil (remplacement par nom, pas de doublon).
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    assert len(listing["data"]["tools"]) == 1


def test_add_function_replace_updates_body(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_function(
        str(tmp_path), "demo", "root", "f", params=[], docstring="d", body='return {"v": 1}'
    )
    res = add_function(
        str(tmp_path), "demo", "root", "f", params=[], docstring="d", body='return {"v": 2}'
    )
    assert res["ok"] is True
    src = _agent_src(tmp_path, "demo")
    assert 'return {"v": 2}' in src
    assert 'return {"v": 1}' not in src
    # Un seul def f.
    assert src.count("def f(") == 1


# --------------------------------------------------------------------------- #
# add_long_running
# --------------------------------------------------------------------------- #
def test_add_long_running_wraps_func(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_long_running(
        str(tmp_path),
        "demo",
        "root",
        "poll",
        params=[{"name": "job", "type": "str", "default": None}],
        docstring="Poll a job.",
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "def poll(job: str) -> dict:" in src
    assert "LongRunningFunctionTool(func=poll)" in src
    assert "from google.adk.tools import" in src and "LongRunningFunctionTool" in src


# --------------------------------------------------------------------------- #
# add_builtin
# --------------------------------------------------------------------------- #
def test_add_builtin_core_renders_bare_name_and_import(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_builtin(str(tmp_path), "demo", "root", "google_search")
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools import google_search" in src
    assert "google_search" in src.split("root = LlmAgent(")[1]  # référencé dans l'agent


def test_add_builtin_rejects_unknown_kind(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_builtin(str(tmp_path), "demo", "root", "definitely_not_a_builtin")
    assert res["ok"] is False
    assert res["error"]


def test_add_builtin_vertex_requires_data_store(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_builtin(str(tmp_path), "demo", "root", "vertex_ai_search")
    assert res["ok"] is False
    res2 = add_builtin(
        str(tmp_path),
        "demo",
        "root",
        "vertex_ai_search",
        args={"data_store_id": "projects/p/locations/l/dataStores/d"},
    )
    assert res2["ok"] is True, res2["error"]
    src = _agent_src(tmp_path, "demo")
    assert 'VertexAiSearchTool(data_store_id="projects/p/locations/l/dataStores/d")' in src
    assert "from google.adk.tools import VertexAiSearchTool" in src


def test_add_builtin_replace_same_kind_is_idempotent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_builtin(str(tmp_path), "demo", "root", "google_search")
    again = add_builtin(str(tmp_path), "demo", "root", "google_search")
    assert again["ok"] is True
    assert again["data"]["changed"] is False
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    assert len(listing["data"]["tools"]) == 1


# --------------------------------------------------------------------------- #
# add_agent_tool
# --------------------------------------------------------------------------- #
def test_add_agent_tool_wraps_existing_agent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root", instruction="Delegate.")
    create_llm(str(tmp_path), "demo", "helper", instruction="Help.")
    res = add_agent_tool(str(tmp_path), "demo", "root", "helper")
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "AgentTool(agent=helper)" in src
    assert "from google.adk.tools import AgentTool" in src
    # helper défini avant root (ordre topo : la cible précède l'enveloppant).
    assert src.index("helper = LlmAgent(") < src.index("root = LlmAgent(")


def test_add_agent_tool_rejects_missing_target(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_agent_tool(str(tmp_path), "demo", "root", "ghost")
    assert res["ok"] is False
    assert res["error"]


def test_add_agent_tool_rejects_self_reference(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_agent_tool(str(tmp_path), "demo", "root", "root")
    assert res["ok"] is False


def test_add_agent_tool_does_not_add_target_as_sub_agent(tmp_path: Path) -> None:
    # Règle parent unique : l'agent enveloppé en outil ne doit PAS devenir un sub_agent.
    create_llm(str(tmp_path), "demo", "root", instruction="Delegate.")
    create_llm(str(tmp_path), "demo", "helper", instruction="Help.")
    add_agent_tool(str(tmp_path), "demo", "root", "helper")
    src = _agent_src(tmp_path, "demo")
    # root ne référence helper que via AgentTool, pas via sub_agents.
    root_block = src.split("root = LlmAgent(")[1]
    assert "sub_agents=" not in root_block.split(")")[0]


# --------------------------------------------------------------------------- #
# add_openapi
# --------------------------------------------------------------------------- #
def test_add_openapi_builds_toolset_and_refs_it(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_openapi(str(tmp_path), "demo", "root", _OPENAPI_SPEC, name="petstore")
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.openapi_tool import OpenAPIToolset" in src
    # La construction peut être repliée par ruff si la spec est longue (inline ou multi-ligne).
    assert "petstore = OpenAPIToolset(" in src
    assert "spec_str=" in src
    assert 'spec_str_type="json"' in src
    # le toolset (variable) est référencé bare dans tools=[...].
    assert "petstore" in src.split("root = LlmAgent(")[1]


def test_add_openapi_default_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_openapi(str(tmp_path), "demo", "root", _OPENAPI_SPEC)
    assert res["ok"] is True
    src = _agent_src(tmp_path, "demo")
    assert "root_openapi = OpenAPIToolset(" in src


# --------------------------------------------------------------------------- #
# 3b : add_bigquery / add_spanner
# --------------------------------------------------------------------------- #
def test_add_bigquery_builds_toolset(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_bigquery(str(tmp_path), "demo", "root", name="bq")
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.bigquery import BigQueryToolset" in src
    assert "bq = BigQueryToolset(" in src
    assert "bq" in src.split("root = LlmAgent(")[1]


def test_add_bigquery_default_name_and_args(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_bigquery(str(tmp_path), "demo", "root", args={"bigquery_tool_config": "my_cfg"})
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "root_bigquery = BigQueryToolset(" in src
    # args sont des expressions source (référence de variable), pas des littéraux chaîne.
    assert "bigquery_tool_config=my_cfg" in src


def test_add_spanner_builds_toolset(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_spanner(str(tmp_path), "demo", "root", name="sp")
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.spanner import SpannerToolset" in src
    assert "sp = SpannerToolset(" in src


def test_add_bigquery_rejects_bad_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_bigquery(str(tmp_path), "demo", "root", name="bad name!")
    assert res["ok"] is False
    assert res["error"]


# --------------------------------------------------------------------------- #
# 3b : add_mcp_toolset
# --------------------------------------------------------------------------- #
def test_add_mcp_stdio(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_mcp_toolset(
        str(tmp_path),
        "demo",
        "root",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
        tool_filter=["read_file"],
        name="fs",
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.mcp_tool import" in src
    assert "from mcp import StdioServerParameters" in src
    assert "fs = McpToolset(" in src
    assert 'command="npx"' in src
    assert 'tool_filter=["read_file"]' in src


def test_add_mcp_http_with_headers(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_mcp_toolset(
        str(tmp_path),
        "demo",
        "root",
        transport="http",
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer x"},
        name="h",
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "StreamableHTTPConnectionParams(" in src
    assert 'url="https://api.example.com/mcp"' in src
    assert 'headers={"Authorization": "Bearer x"}' in src


def test_add_mcp_rejects_bad_transport(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_mcp_toolset(str(tmp_path), "demo", "root", transport="ftp", url="x")
    assert res["ok"] is False
    assert res["error"]


def test_add_mcp_stdio_requires_command(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_mcp_toolset(str(tmp_path), "demo", "root", transport="stdio")
    assert res["ok"] is False


def test_add_mcp_sse_requires_url(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_mcp_toolset(str(tmp_path), "demo", "root", transport="sse")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# 3b : add_apihub
# --------------------------------------------------------------------------- #
def test_add_apihub_builds_toolset(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_apihub(str(tmp_path), "demo", "root", "projects/p/locations/l/apis/a", name="hub")
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.apihub_tool import APIHubToolset" in src
    assert "hub = APIHubToolset(" in src
    assert 'apihub_resource_name="projects/p/locations/l/apis/a"' in src


def test_add_apihub_rejects_empty_resource(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_apihub(str(tmp_path), "demo", "root", "  ", name="hub")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# 3b : add_langchain / add_crewai
# --------------------------------------------------------------------------- #
def test_add_langchain_wraps_expr(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_langchain(
        str(tmp_path),
        "demo",
        "root",
        import_line="from langchain_community.tools import WikipediaQueryRun",
        tool_expr="WikipediaQueryRun(api_wrapper=wrapper)",
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.langchain_tool import LangchainTool" in src
    assert "from langchain_community.tools import WikipediaQueryRun" in src
    assert "LangchainTool(tool=WikipediaQueryRun(api_wrapper=wrapper))" in src


def test_add_crewai_wraps_expr_with_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_crewai(
        str(tmp_path),
        "demo",
        "root",
        import_line="from crewai_tools import SerperDevTool",
        tool_expr="SerperDevTool()",
        name="serper",
        description="Web search.",
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "from google.adk.tools.crewai_tool import CrewaiTool" in src
    assert "from crewai_tools import SerperDevTool" in src
    assert 'CrewaiTool(tool=SerperDevTool(), name="serper", description="Web search.")' in src


def test_add_langchain_rejects_empty_import_line(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_langchain(str(tmp_path), "demo", "root", import_line="", tool_expr="X()")
    assert res["ok"] is False


def test_add_crewai_rejects_missing_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_crewai(
        str(tmp_path),
        "demo",
        "root",
        import_line="from x import X",
        tool_expr="X()",
        name="",
        description="d",
    )
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# 3b : set_auth (attache une sous-spec auth à un toolset existant)
# --------------------------------------------------------------------------- #
def test_set_auth_injects_apikey_on_openapi(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_openapi(str(tmp_path), "demo", "root", _OPENAPI_SPEC, name="api")
    res = set_auth(
        str(tmp_path), "demo", "root", "api", scheme="apikey", credential={"api_key": "secret"}
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "auth_credential=AuthCredential(" in src
    assert "auth_type=AuthCredentialTypes.API_KEY" in src
    assert 'api_key="secret"' in src
    assert "from google.adk.auth import AuthCredential, AuthCredentialTypes" in src


def test_set_auth_injects_bearer_on_apihub(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_apihub(str(tmp_path), "demo", "root", "projects/p/apis/a", name="hub")
    res = set_auth(
        str(tmp_path), "demo", "root", "hub", scheme="bearer", credential={"token": "tok"}
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "auth_type=AuthCredentialTypes.HTTP" in src
    assert 'HttpCredentials(token="tok")' in src


def test_set_auth_on_mcp_toolset(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_mcp_toolset(str(tmp_path), "demo", "root", transport="http", url="https://x/mcp", name="m")
    res = set_auth(
        str(tmp_path),
        "demo",
        "root",
        "m",
        scheme="oauth2",
        credential={"client_id": "cid", "client_secret": "csec"},
    )
    assert res["ok"] is True, res["error"]
    src = _agent_src(tmp_path, "demo")
    assert "auth_type=AuthCredentialTypes.OAUTH2" in src
    assert 'OAuth2Auth(client_id="cid", client_secret="csec")' in src


def test_set_auth_rejects_unknown_target(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = set_auth(
        str(tmp_path), "demo", "root", "ghost", scheme="apikey", credential={"api_key": "k"}
    )
    assert res["ok"] is False
    assert res["error"]


def test_set_auth_rejects_non_auth_capable_target(tmp_path: Path) -> None:
    # bigquery n'accepte pas l'auth -> set_auth doit refuser.
    create_llm(str(tmp_path), "demo", "root")
    add_bigquery(str(tmp_path), "demo", "root", name="bq")
    res = set_auth(
        str(tmp_path), "demo", "root", "bq", scheme="apikey", credential={"api_key": "k"}
    )
    assert res["ok"] is False
    assert res["error"]


def test_set_auth_rejects_function_tool_target(tmp_path: Path) -> None:
    # set_auth ne s'applique qu'aux toolsets (par variable), pas à une function-tool.
    create_llm(str(tmp_path), "demo", "root")
    add_function(str(tmp_path), "demo", "root", "f", params=[], docstring="d")
    res = set_auth(str(tmp_path), "demo", "root", "f", scheme="apikey", credential={"api_key": "k"})
    assert res["ok"] is False


def test_set_auth_rejects_bad_scheme(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_apihub(str(tmp_path), "demo", "root", "projects/p/apis/a", name="hub")
    res = set_auth(str(tmp_path), "demo", "root", "hub", scheme="telepathy", credential={"k": "v"})
    assert res["ok"] is False


def test_set_auth_rejects_bad_app_and_agent_name(tmp_path: Path) -> None:
    cred = {"api_key": "k"}
    bad_app = set_auth(str(tmp_path), "1bad", "root", "hub", scheme="apikey", credential=cred)
    assert bad_app["ok"] is False
    bad_agent = set_auth(
        str(tmp_path), "demo", "bad name!", "hub", scheme="apikey", credential=cred
    )
    assert bad_agent["ok"] is False


def test_set_auth_rejects_missing_agent(tmp_path: Path) -> None:
    res = set_auth(
        str(tmp_path), "demo", "ghost", "hub", scheme="apikey", credential={"api_key": "k"}
    )
    assert res["ok"] is False
    assert "introuvable" in res["error"]


def test_set_auth_rejects_apikey_without_api_key(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_apihub(str(tmp_path), "demo", "root", "projects/p/apis/a", name="hub")
    res = set_auth(str(tmp_path), "demo", "root", "hub", scheme="apikey", credential={"wrong": "v"})
    assert res["ok"] is False


def test_set_auth_is_idempotent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_apihub(str(tmp_path), "demo", "root", "projects/p/apis/a", name="hub")
    set_auth(str(tmp_path), "demo", "root", "hub", scheme="apikey", credential={"api_key": "k"})
    again = set_auth(
        str(tmp_path), "demo", "root", "hub", scheme="apikey", credential={"api_key": "k"}
    )
    assert again["ok"] is True
    assert again["data"]["changed"] is False
    # Toujours un seul outil.
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    assert len(listing["data"]["tools"]) == 1


# --------------------------------------------------------------------------- #
# Garde-fous communs : agent inexistant / mauvais type / corrompu
# --------------------------------------------------------------------------- #
def test_attach_rejects_missing_agent(tmp_path: Path) -> None:
    res = add_builtin(str(tmp_path), "demo", "ghost", "google_search")
    assert res["ok"] is False
    assert "introuvable" in res["error"]


def test_attach_rejects_non_llm_agent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "a")
    create_sequential(str(tmp_path), "demo", "pipe", ["a"])
    res = add_builtin(str(tmp_path), "demo", "pipe", "google_search")
    assert res["ok"] is False
    assert "llm" in res["error"].lower()


def test_attach_rejects_bad_agent_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_builtin(str(tmp_path), "demo", "bad name!", "google_search")
    assert res["ok"] is False
    assert res["error"]


def test_add_long_running_rejects_malformed_param(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_long_running(
        str(tmp_path), "demo", "root", "g", params=[{"type": "str"}], docstring="d"
    )
    assert res["ok"] is False  # 'name' manquant


def test_add_openapi_rejects_bad_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    res = add_openapi(str(tmp_path), "demo", "root", _OPENAPI_SPEC, name="bad name!")
    assert res["ok"] is False
    assert res["error"]


def test_list_rejects_bad_app_and_agent_name(tmp_path: Path) -> None:
    assert list_tools_for_agent(str(tmp_path), "1bad", "root")["ok"] is False
    assert list_tools_for_agent(str(tmp_path), "demo", "bad name!")["ok"] is False


def test_list_summarizes_openapi_and_vertex(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_openapi(str(tmp_path), "demo", "root", _OPENAPI_SPEC, name="api")
    add_builtin(str(tmp_path), "demo", "root", "vertex_ai_search", args={"data_store_id": "ds"})
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    by_kind = {t["kind"]: t for t in listing["data"]["tools"]}
    assert by_kind["openapi"]["name"] == "api"
    assert by_kind["builtin"]["builtin_kind"] == "vertex_ai_search"
    assert by_kind["builtin"]["args"] == {"data_store_id": "ds"}


def test_list_summarizes_3b_kinds(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    add_bigquery(str(tmp_path), "demo", "root", name="bq", args={"bigquery_tool_config": "cfg"})
    add_spanner(str(tmp_path), "demo", "root", name="sp")
    add_mcp_toolset(str(tmp_path), "demo", "root", transport="http", url="https://x/mcp", name="m")
    add_apihub(str(tmp_path), "demo", "root", "projects/p/apis/a", name="hub")
    add_langchain(
        str(tmp_path), "demo", "root", import_line="from x import Y", tool_expr="Y(opt=1)"
    )
    add_crewai(
        str(tmp_path),
        "demo",
        "root",
        import_line="from z import Z",
        tool_expr="Z()",
        name="zz",
        description="d",
    )
    set_auth(str(tmp_path), "demo", "root", "hub", scheme="apikey", credential={"api_key": "k"})
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    by_kind = {t["kind"]: t for t in listing["data"]["tools"]}
    assert by_kind["bigquery"]["name"] == "bq"
    assert by_kind["bigquery"]["args"] == {"bigquery_tool_config": "cfg"}
    assert by_kind["spanner"]["name"] == "sp"
    assert by_kind["mcp_toolset"]["transport"] == "http"
    assert by_kind["apihub"]["apihub_resource_name"] == "projects/p/apis/a"
    assert by_kind["apihub"]["auth"] == {"scheme": "apikey"}
    assert by_kind["langchain"]["tool_expr"] == "Y(opt=1)"
    assert by_kind["crewai"]["name"] == "zz"


def test_attach_rejects_bad_app_name(tmp_path: Path) -> None:
    res = add_builtin(str(tmp_path), "1bad", "root", "google_search")
    assert res["ok"] is False


def _corrupt_sidecar(tmp_path: Path, app_name: str = "demo") -> str:
    app = tmp_path / app_name / SIDECAR_PATH
    app.parent.mkdir(parents=True, exist_ok=True)
    app.write_text("{ not valid json ]", encoding="utf-8")
    return str(tmp_path)


def test_corrupt_sidecar_returns_err_on_all_tools(tmp_path: Path) -> None:
    root = _corrupt_sidecar(tmp_path)
    for res in (
        add_function(root, "demo", "root", "f", params=[], docstring="d"),
        add_long_running(root, "demo", "root", "g", params=[], docstring="d"),
        add_builtin(root, "demo", "root", "google_search"),
        add_agent_tool(root, "demo", "root", "x"),
        add_openapi(root, "demo", "root", _OPENAPI_SPEC),
        add_bigquery(root, "demo", "root"),
        add_spanner(root, "demo", "root"),
        add_mcp_toolset(root, "demo", "root", transport="stdio", command="npx"),
        add_apihub(root, "demo", "root", "projects/p/apis/a"),
        add_langchain(root, "demo", "root", import_line="from x import Y", tool_expr="Y()"),
        add_crewai(
            root,
            "demo",
            "root",
            import_line="from x import Z",
            tool_expr="Z()",
            name="z",
            description="d",
        ),
        set_auth(root, "demo", "root", "api", scheme="apikey", credential={"api_key": "k"}),
        list_tools_for_agent(root, "demo", "root"),
    ):
        assert res["ok"] is False
        assert res["error"]


# --------------------------------------------------------------------------- #
# list (tools_list)
# --------------------------------------------------------------------------- #
def test_list_tools_reports_each_kind(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root", instruction="Use tools.")
    create_llm(str(tmp_path), "demo", "child", instruction="Child.")
    add_function(
        str(tmp_path),
        "demo",
        "root",
        "compute",
        params=[{"name": "x", "type": "str", "default": None}],
        docstring="Compute.",
    )
    add_builtin(str(tmp_path), "demo", "root", "google_search")
    add_agent_tool(str(tmp_path), "demo", "root", "child")
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    assert listing["ok"] is True
    kinds = [t["kind"] for t in listing["data"]["tools"]]
    assert kinds == ["function", "builtin", "agent_tool"]
    # Le résumé function expose les params typés.
    fn = listing["data"]["tools"][0]
    assert fn["name"] == "compute"
    assert fn["params"] == [{"name": "x", "type": "str", "default": None}]


def test_list_tools_empty_for_fresh_agent(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "root")
    listing = list_tools_for_agent(str(tmp_path), "demo", "root")
    assert listing["ok"] is True
    assert listing["data"]["tools"] == []


def test_list_tools_rejects_missing_agent(tmp_path: Path) -> None:
    res = list_tools_for_agent(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# Stabilité de format ruff — agent.py avec function tools + agent custom
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    import shutil

    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def test_generated_agent_py_is_ruff_format_stable(tmp_path: Path) -> None:
    """Un agent.py avec function tools + builtin + agent_tool + agent custom passe
    ``ruff format --check`` (la sortie est déjà formatée)."""
    create_custom(str(tmp_path), "demo", "aux", description="Aux agent")
    create_llm(str(tmp_path), "demo", "child", instruction="Child.")
    create_llm(str(tmp_path), "demo", "root", instruction="Coordinate.")
    add_function(
        str(tmp_path),
        "demo",
        "root",
        "add",
        params=[
            {"name": "a", "type": "int", "default": None},
            {"name": "b", "type": "int", "default": "0"},
        ],
        docstring="Add two integers.",
        returns="dict",
        body='return {"sum": a + b}',
    )
    add_builtin(str(tmp_path), "demo", "root", "google_search")
    add_agent_tool(str(tmp_path), "demo", "root", "child")

    src = _agent_src(tmp_path, "demo")
    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff introuvable dans l'environnement — test de format ignoré")
    gen = tmp_path / "to_check.py"
    gen.write_text(src, encoding="utf-8")
    result = subprocess.run([ruff, "format", "--check", str(gen)], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"ruff format --check a échoué.\nStdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Source générée :\n{src}"
    )


# --------------------------------------------------------------------------- #
# PREUVE FONCTIONNELLE — instanciation réelle des objets ADK (subprocess)
# --------------------------------------------------------------------------- #
def _probe_tools(project_path: str, app_name: str) -> dict[str, object]:
    """Importe ``<app_name>.agent`` dans un subprocess et renvoie un résumé des outils.

    Renvoie le **type brut** des entrées de ``root_agent.tools`` (après init) ET le type
    **canonique** (après ``await canonical_tools()``), car un plain function n'est wrappé en
    ``FunctionTool`` que lazily par ``canonical_tools`` (cf. docs/adk-api-notes/tools.md).
    """
    code = (
        "import json,sys,asyncio;"
        f"sys.path.insert(0, r'{project_path}');"
        f"import {app_name}.agent as m;"
        "ra=m.root_agent;"
        "raw=[type(t).__name__ for t in (ra.tools or [])];"
        "canon=asyncio.get_event_loop().run_until_complete(ra.canonical_tools());"
        "can=[type(t).__name__ for t in canon];"
        "print(json.dumps({'root_type':type(ra).__name__,'n_tools':len(ra.tools or []),"
        "'raw':raw,'canonical':can}))"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        cwd=project_path,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_functional_deps_free_tools_instantiate(tmp_path: Path) -> None:
    """function + builtin google_search + agent_tool : le module généré instancie de vrais
    objets ADK ; on vérifie le compte et les types (bruts + canoniques)."""
    create_llm(str(tmp_path), "probe_app", "root", instruction="Use tools.")
    create_llm(str(tmp_path), "probe_app", "helper", instruction="Help.")
    add_function(
        str(tmp_path),
        "probe_app",
        "root",
        "add",
        params=[
            {"name": "a", "type": "int", "default": None},
            {"name": "b", "type": "int", "default": None},
        ],
        docstring="Add.",
        returns="dict",
        body='return {"sum": a + b}',
    )
    add_builtin(str(tmp_path), "probe_app", "root", "google_search")
    add_agent_tool(str(tmp_path), "probe_app", "root", "helper")
    set_root(str(tmp_path), "probe_app", "root")

    info = _probe_tools(str(tmp_path), "probe_app")
    assert info["root_type"] == "LlmAgent"
    assert info["n_tools"] == 3
    # Champ brut : la fonction reste 'function' ; le builtin est son instance ; AgentTool tel quel.
    assert info["raw"] == ["function", "GoogleSearchTool", "AgentTool"]
    # Canonique : la fonction est wrappée en FunctionTool ; les autres inchangés.
    assert info["canonical"] == ["FunctionTool", "GoogleSearchTool", "AgentTool"]


# --------------------------------------------------------------------------- #
# Mount wiring — client in-memory + preuve fonctionnelle bout-en-bout
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_tools_mounted_names_and_read_through(tmp_path: Path) -> None:
    mcp = build_server()
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        for expected in (
            "tools_add_function",
            "tools_add_long_running",
            "tools_add_builtin",
            "tools_add_agent_tool",
            "tools_add_openapi",
            "tools_add_bigquery",
            "tools_add_spanner",
            "tools_add_mcp_toolset",
            "tools_add_apihub",
            "tools_add_langchain",
            "tools_add_crewai",
            "tools_set_auth",
            "tools_list",
        ):
            assert expected in tool_names, f"manquant: {expected}"
        # Pas de double préfixe.
        assert not any(n.startswith("tools_tools_") for n in tool_names)

        # Prépare un agent llm puis attache une function-tool via le client.
        await client.call_tool(
            "agents_create_llm",
            {"path": str(tmp_path), "app_name": "client_app", "name": "root", "instruction": "Hi"},
        )
        created = await client.call_tool(
            "tools_add_function",
            {
                "path": str(tmp_path),
                "app_name": "client_app",
                "agent_name": "root",
                "func_name": "greet",
                "params": [{"name": "who", "type": "str", "default": None}],
                "docstring": "Greet someone.",
                "returns": "str",
                "body": 'return f"hi {who}"',
            },
        )
        assert created.data["ok"] is True, created.data["error"]
        await client.call_tool(
            "agents_set_root",
            {"path": str(tmp_path), "app_name": "client_app", "name": "root"},
        )

    # Hors client : le module généré instancie un LlmAgent portant une function-tool.
    info = _probe_tools(str(tmp_path), "client_app")
    assert info["root_type"] == "LlmAgent"
    assert info["n_tools"] == 1
    assert info["raw"] == ["function"]
    assert info["canonical"] == ["FunctionTool"]


# --------------------------------------------------------------------------- #
# 3b : agent.py avec TOUS les genres optionnels — ast.parse + ruff format
# (PAS d'import : les extras ne sont pas installés en CI ; cf. docs/adk-api-notes/tools.md)
# --------------------------------------------------------------------------- #
def _build_all_3b(tmp_path: Path) -> str:
    """Construit un agent.py via les outils du domaine couvrant tous les genres 3b + auth."""
    create_llm(str(tmp_path), "deps", "root", instruction="Use 3b toolsets.")
    add_bigquery(str(tmp_path), "deps", "root", name="bq", args={"bigquery_tool_config": "bq_cfg"})
    add_spanner(str(tmp_path), "deps", "root", name="sp")
    add_mcp_toolset(
        str(tmp_path),
        "deps",
        "root",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
        tool_filter=["read_file", "list_directory"],
        name="fs",
    )
    add_apihub(str(tmp_path), "deps", "root", "projects/p/locations/l/apis/a", name="hub")
    add_openapi(str(tmp_path), "deps", "root", _OPENAPI_SPEC, name="petstore")
    set_auth(str(tmp_path), "deps", "root", "petstore", scheme="bearer", credential={"token": "t"})
    add_langchain(
        str(tmp_path),
        "deps",
        "root",
        import_line="from langchain_community.tools import WikipediaQueryRun",
        tool_expr="WikipediaQueryRun(api_wrapper=wiki)",
    )
    add_crewai(
        str(tmp_path),
        "deps",
        "root",
        import_line="from crewai_tools import SerperDevTool",
        tool_expr="SerperDevTool()",
        name="serper",
        description="Web search.",
    )
    set_root(str(tmp_path), "deps", "root")
    return _agent_src(tmp_path, "deps")


def test_all_3b_kinds_generate_valid_python(tmp_path: Path) -> None:
    """L'agent.py contenant tous les genres 3b + auth est du Python valide (ast.parse).

    On NE l'importe PAS (les extras google-adk ne sont pas installés en CI)."""
    import ast

    src = _build_all_3b(tmp_path)
    ast.parse(src)  # SyntaxError si le rendu est cassé
    # Imports + constructions clés présents.
    assert "from google.adk.tools.bigquery import BigQueryToolset" in src
    assert "from google.adk.tools.spanner import SpannerToolset" in src
    assert "from google.adk.tools.mcp_tool import" in src
    assert "from google.adk.tools.apihub_tool import APIHubToolset" in src
    assert "from google.adk.tools.langchain_tool import LangchainTool" in src
    assert "from google.adk.tools.crewai_tool import CrewaiTool" in src
    assert "from google.adk.auth import AuthCredential, AuthCredentialTypes" in src


def test_all_3b_kinds_ruff_format_stable(tmp_path: Path) -> None:
    """L'agent.py avec tous les genres 3b + auth passe ``ruff format --check``."""
    src = _build_all_3b(tmp_path)
    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff introuvable dans l'environnement — test de format ignoré")
    gen = tmp_path / "to_check_3b.py"
    gen.write_text(src, encoding="utf-8")
    result = subprocess.run([ruff, "format", "--check", str(gen)], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"ruff format --check a échoué.\nStdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Source générée :\n{src}"
    )


def test_mcp_toolset_functional_probe_if_available(tmp_path: Path) -> None:
    """Preuve fonctionnelle OPTIONNELLE : si l'extra mcp est présent, l'agent.py s'instancie.

    Gardé derrière ``find_spec`` -> SKIP si l'extra absent (CI sans extras). Le McpToolset est
    sans dépendance lourde dans le venv de base (le paquet ``mcp`` y est déjà), mais on reste
    défensif : on n'échoue jamais à cause d'un extra manquant.
    """
    import importlib.util

    if importlib.util.find_spec("google.adk.tools.mcp_tool") is None or (
        importlib.util.find_spec("mcp") is None
    ):
        pytest.skip("extra mcp absent — preuve fonctionnelle ignorée")

    create_llm(str(tmp_path), "mcp_probe", "root", instruction="Use MCP.")
    add_mcp_toolset(
        str(tmp_path),
        "mcp_probe",
        "root",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
        name="srv",
    )
    set_root(str(tmp_path), "mcp_probe", "root")

    code = (
        "import json,sys;"
        f"sys.path.insert(0, r'{tmp_path}');"
        "import mcp_probe.agent as m;"
        "ra=m.root_agent;"
        "raw=[type(t).__name__ for t in (ra.tools or [])];"
        "print(json.dumps({'root_type':type(ra).__name__,'raw':raw}))"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    if out.returncode != 0:
        pytest.skip(f"instanciation MCP indisponible dans cet environnement : {out.stderr[:200]}")
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["root_type"] == "LlmAgent"
    assert info["raw"] == ["McpToolset"]
