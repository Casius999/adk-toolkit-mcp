"""Tests du serveur racine : mode outils-directs (défaut) vs Code Mode (P6a), et tags par domaine.

- ``build_server()`` (défaut) expose les 81 outils par leur nom ``<domaine>_<bare>`` (aucune
  régression) et chaque outil porte son tag de domaine.
- ``build_server(code_mode=True)`` applique le VRAI transform FastMCP 3.3.1 et effondre la surface
  en outils de découverte + ``execute`` — démontrant la réduction de tokens (81 → poignée).
- ``code_mode_enabled()`` lit la variable d'env ``ADK_TOOLKIT_CODE_MODE``.

Les listes server-side (``mcp.list_tools()``) renvoient des ``fastmcp.tools.Tool`` qui portent
``.tags`` ; le read-through client (``fastmcp.Client``) confirme la surface exposée et le tag
remonté via ``_meta.fastmcp.tags``.
"""

from __future__ import annotations

import os

import pytest
from fastmcp import Client, FastMCP

from adk_toolkit_mcp.server import build_server, code_mode_enabled, main

#: Nombre exact d'outils exposés en mode outils-directs (contrat de non-régression).
_EXPECTED_TOOL_COUNT = 81

#: Les 15 domaines montés (préfixe de namespace -> tag attendu).
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

#: Échantillon de noms d'outils qui DOIVENT exister en mode outils-directs (un par domaine clé).
_SAMPLE_NAMES = {
    "project_create",
    "agents_create_llm",
    "agents_set_root",
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
}


def _domain_of(tool_name: str) -> str:
    """Renvoie le domaine d'un nom d'outil exposé (gère le namespace composé ``mcp_bridge``)."""
    if tool_name.startswith("mcp_bridge_"):
        return "mcp_bridge"
    return tool_name.split("_", 1)[0]


# --------------------------------------------------------------------------- #
# Construction de base
# --------------------------------------------------------------------------- #
def test_build_server_returns_fastmcp() -> None:
    assert isinstance(build_server(), FastMCP)


def test_build_server_code_mode_returns_fastmcp() -> None:
    assert isinstance(build_server(code_mode=True), FastMCP)


def test_main_is_callable() -> None:
    assert callable(main)


# --------------------------------------------------------------------------- #
# Mode outils-directs (défaut) : 81 outils, noms stables, tags par domaine
# --------------------------------------------------------------------------- #
async def test_default_mode_exposes_all_81_tools_by_name() -> None:
    """Défaut : 81 outils exposés par nom (aucune régression) + un échantillon présent."""
    async with Client(build_server()) as client:
        names = {t.name for t in await client.list_tools()}
    assert len(names) == _EXPECTED_TOOL_COUNT
    assert _SAMPLE_NAMES <= names
    # Pas de double-préfixe (ex. project_project_create).
    assert not any(n.startswith(f"{d}_{d}_") for d in _DOMAINS for n in names)


async def test_every_tool_carries_its_domain_tag() -> None:
    """Chaque outil porte exactement son tag de domaine (inspection server-side ``.tags``)."""
    tools = await build_server().list_tools()
    assert len(tools) == _EXPECTED_TOOL_COUNT
    mismatched = [
        (t.name, sorted(t.tags or [])) for t in tools if _domain_of(t.name) not in (t.tags or set())
    ]
    assert mismatched == []


async def test_domain_tags_surface_to_client_via_meta() -> None:
    """Le tag de domaine remonte au client MCP via ``_meta.fastmcp.tags``."""
    async with Client(build_server()) as client:
        tools = await client.list_tools()
    by_name = {t.name: t for t in tools}
    meta = by_name["project_create"].meta or {}
    assert "project" in (meta.get("fastmcp", {}).get("tags") or [])


# --------------------------------------------------------------------------- #
# Code Mode (opt-in) : surface effondrée + atteignable
# --------------------------------------------------------------------------- #
async def test_code_mode_collapses_surface_to_discovery_and_execute() -> None:
    """code_mode=True : la surface passe des 81 outils à une poignée discovery + execute."""
    async with Client(build_server(code_mode=True)) as client:
        names = {t.name for t in await client.list_tools()}
    # Réduction franche de la surface (gros gain de tokens) — démontrée quantitativement.
    assert len(names) < 10
    # Outil d'exécution toujours présent + au moins un outil de découverte.
    assert "execute" in names
    assert {"search", "get_schema"} <= names
    # Les 81 noms directs ne sont PLUS exposés au top-level.
    assert "run_agent" not in names
    assert "project_create" not in names


async def test_code_mode_reduces_tool_surface_vs_default() -> None:
    """Démontre la réduction : surface Code Mode << surface outils-directs (81)."""
    async with Client(build_server()) as direct_client:
        direct = {t.name for t in await direct_client.list_tools()}
    async with Client(build_server(code_mode=True)) as cm_client:
        code_mode = {t.name for t in await cm_client.list_tools()}
    assert len(direct) == _EXPECTED_TOOL_COUNT
    # Au moins 90% d'outils en moins au top-level.
    assert len(code_mode) <= len(direct) // 10


async def test_code_mode_tags_discovery_tool_present() -> None:
    """Comme on tague par domaine, le discovery ``tags`` est ajouté et atteignable en Code Mode."""
    async with Client(build_server(code_mode=True)) as client:
        names = {t.name for t in await client.list_tools()}
        assert "tags" in names
        # Le discovery ``tags`` liste les domaines tagués (lecture du catalogue, sans monty).
        result = await client.call_tool("tags", {"detail": "brief"})
    rendered = "\n".join(block.text for block in result.content if getattr(block, "text", None))
    # Quelques domaines connus apparaissent dans le rendu des tags.
    assert "agents" in rendered
    assert "deploy" in rendered


# --------------------------------------------------------------------------- #
# Bascule par variable d'environnement
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
    """``code_mode_enabled`` reconnaît les valeurs vraies/fausses de ``ADK_TOOLKIT_CODE_MODE``."""
    monkeypatch.setenv("ADK_TOOLKIT_CODE_MODE", value)
    assert code_mode_enabled() is expected


def test_code_mode_enabled_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans variable d'env, le Code Mode est désactivé (mode outils-directs par défaut)."""
    monkeypatch.delenv("ADK_TOOLKIT_CODE_MODE", raising=False)
    assert code_mode_enabled() is False
    # Sanity : l'env n'a pas fui d'un autre test.
    assert os.getenv("ADK_TOOLKIT_CODE_MODE") is None
