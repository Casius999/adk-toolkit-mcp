"""Tests du domaine ``mcp_bridge`` (P4b — exposer des outils ADK COMME des outils MCP).

Ces tests sont **FONCTIONNELS et exécutables en CI sans aucun extra** : le paquet ``mcp`` est une
dépendance core de ``fastmcp``, donc ``adk_to_mcp_tool_type`` est toujours disponible. On prouve :

- ``convert_builtin("google_search")`` renvoie un dict en forme ``mcp.types.Tool``
  (``{name, description, inputSchema}``) — on assert la STRUCTURE sur un VRAI outil ADK ;
- ``expose_adk_tools`` sur un agent réel (scaffoldé) portant un builtin + une function-tool
  renvoie leurs schémas MCP (la function-tool a un vrai JSON-Schema ``properties``/``required``) ;
- les chemins d'erreur (kind inconnu, app/agent invalides, agent absent, agent sans outils) →
  ``err`` propre, jamais d'exception ;
- read-through via un ``fastmcp.Client`` en mémoire : noms exposés ``mcp_bridge_<bare>`` (pas de
  double-préfixe) et l'appel ``mcp_bridge_convert_builtin`` round-trip.

Cf. ``docs/adk-api-notes/a2a-mcp-bridge.md`` (signatures + résultat fonctionnel confirmés).
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
    """Filtre LOCAL la ``DeprecationWarning`` des agents workflow (Sequential/Parallel/Loop).

    Ces agents sont dépréciés en ADK 2.1.0 mais restent fonctionnels (cf. PROGRESS/agents.md) ;
    les construire in-process (via ``import_root_agent`` qui ``exec`` l'``agent.py`` scaffoldé)
    émet une ``DeprecationWarning`` que ``-W error::DeprecationWarning`` transformerait en erreur.
    On la NEUTRALISE uniquement le temps de l'appel (scope étroit, notre code reste strict).
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
    """Scaffolde une app ADK (importable SANS clé API) avec un builtin + une function-tool.

    L'agent porte ``google_search`` (un builtin, inputSchema vide) et ``add_numbers`` (une
    fonction nue qu'``canonical_tools`` enveloppe en FunctionTool → vrai JSON-Schema). Renvoie le
    chemin parent.
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
    """Scaffolde une app dont le root_agent est un SequentialAgent (pas de canonical_tools)."""
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
# convert_builtin — FONCTIONNEL (no extra)
# --------------------------------------------------------------------------- #
def test_convert_builtin_google_search_structure() -> None:
    """``convert_builtin('google_search')`` → schéma MCP {name, description, inputSchema}."""
    result = MB.convert_builtin("google_search")
    assert result["ok"] is True, result
    tool = result["data"]["tool"]
    # Forme mcp.types.Tool aplatie : les trois clés attendues, dans le bon type.
    assert set(tool.keys()) == {"name", "description", "inputSchema"}
    assert tool["name"] == "google_search"
    assert isinstance(tool["description"], str)
    # google_search n'a pas de paramètres déclarés -> inputSchema est un dict (vide).
    assert isinstance(tool["inputSchema"], dict)


def test_convert_builtin_other_core_builtins() -> None:
    """D'autres builtins core convertissent aussi (instances BaseTool ou fonctions enveloppées)."""
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
    """``vertex_ai_search`` (à argument) n'est pas un builtin *core* → err pointant vers expose."""
    result = MB.convert_builtin("vertex_ai_search")
    assert result["ok"] is False
    assert "expose_adk_tools" in result["error"]


# --------------------------------------------------------------------------- #
# expose_adk_tools — FONCTIONNEL (no extra)
# --------------------------------------------------------------------------- #
async def test_expose_adk_tools_returns_mcp_schemas(tmp_path: Path) -> None:
    """Un agent réel (builtin + function-tool) → leurs schémas MCP (function = vrai schéma)."""
    path = _scaffold_agent_with_tools(tmp_path)
    result = await MB.expose_adk_tools(path=path, app_name="myapp", agent_name="myapp")
    assert result["ok"] is True, result
    data = result["data"]
    assert data["count"] == 2
    by_name = {t["name"]: t for t in data["tools"]}
    assert "google_search" in by_name
    assert "add_numbers" in by_name
    # La function-tool a un VRAI JSON-Schema (properties a/b, required).
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
    """Pas d'agent.py scaffoldé → err propre (RootAgentImportError), pas d'exception."""
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
    """Un agent workflow (Sequential) sans canonical_tools → err actionnable (pas un crash)."""
    path = _scaffold_sequential(tmp_path)
    with _ignore_workflow_deprecation():
        result = await MB.expose_adk_tools(path=path, app_name="wf", agent_name="wf")
    assert result["ok"] is False
    assert "LLM" in result["error"] or "tools" in result["error"].lower()


async def test_expose_adk_tools_sub_agent_found_in_tree(tmp_path: Path) -> None:
    """``find_agent`` localise un SOUS-agent (pas que la racine) — l'enfant LLM a ses outils."""
    path = _scaffold_sequential(tmp_path)
    # 'child' est un LlmAgent (sans outils) niché sous le SequentialAgent racine.
    with _ignore_workflow_deprecation():
        result = await MB.expose_adk_tools(path=path, app_name="wf", agent_name="child")
    assert result["ok"] is True, result
    # Aucun outil attaché → liste vide, ce n'est PAS une erreur.
    assert result["data"]["count"] == 0
    assert result["data"]["tools"] == []


# --------------------------------------------------------------------------- #
# read-through fastmcp.Client (noms exposés + appel)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_convert_builtin() -> None:
    """Outils exposés ``mcp_bridge_<bare>`` (pas de double-préfixe) ; convert_builtin round-trip."""
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {"mcp_bridge_convert_builtin", "mcp_bridge_expose_adk_tools"} <= names
        assert not any(n.startswith("mcp_bridge_mcp_bridge_") for n in names)

        called = await client.call_tool("mcp_bridge_convert_builtin", {"kind": "google_search"})
        assert called.data["ok"] is True
        assert called.data["data"]["tool"]["name"] == "google_search"
