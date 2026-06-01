"""Tests des 5 prompts de workflow (P6a).

Lus via un ``fastmcp.Client`` in-memory (``list_prompts`` / ``get_prompt`` — la VRAIE API client
de fastmcp 3.3.1). On vérifie :

- les 5 prompts attendus sont enregistrés (avec leurs arguments) ;
- chacun rend une chaîne NON vide et actionnable référençant des outils ``<domaine>_*`` ;
- **cross-check load-bearing** : tout token ``<domaine>_<nom>`` cité dans un prompt existe bien
  dans le catalogue réel d'outils du serveur (aucun nom d'outil inventé) ;
- chaque prompt porte le tag ``workflow``.
"""

from __future__ import annotations

import re

import pytest
from fastmcp import Client

from adk_toolkit_mcp.server import build_server

#: Les 5 prompts de workflow attendus -> arguments d'exemple pour le rendu.
_PROMPT_ARGS: dict[str, dict[str, str]] = {
    "scaffold_multi_agent": {"goal": "trier des tickets de support"},
    "add_guardrail": {"agent": "router", "concern": "bloquer les PII"},
    "write_evalset": {"agent": "router"},
    "deploy_checklist": {"target": "cloud_run"},
    "debug_agent": {"symptom": "aucune réponse"},
}

#: Les 15 domaines montés (pour repérer les tokens ``<domaine>_<nom>`` dans le texte des prompts).
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

#: Repère un token ressemblant à un nom d'outil exposé : ``<domaine>_<suffixe_snake>``.
_TOOL_TOKEN = re.compile(r"\b(?:" + "|".join(_DOMAINS) + r")_[a-z][a-z_]*\b")


async def _real_tool_names() -> set[str]:
    """Ensemble des noms d'outils réellement exposés (mode outils-directs)."""
    return {t.name for t in await build_server().list_tools()}


async def _render(client: Client, name: str) -> str:
    """Rend un prompt et renvoie le texte de son unique message."""
    result = await client.get_prompt(name, _PROMPT_ARGS[name])
    return result.messages[0].content.text


# --------------------------------------------------------------------------- #
# Enregistrement
# --------------------------------------------------------------------------- #
async def test_all_five_prompts_registered() -> None:
    """Les 5 prompts de workflow attendus sont enregistrés sur le serveur."""
    async with Client(build_server()) as client:
        names = {p.name for p in await client.list_prompts()}
    assert set(_PROMPT_ARGS) <= names


async def test_prompts_declare_their_arguments() -> None:
    """Chaque prompt déclare ses arguments (dérivés de la signature de la fonction)."""
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
# Rendu
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(_PROMPT_ARGS))
async def test_prompt_renders_non_empty_actionable_text(name: str) -> None:
    """Chaque prompt rend une chaîne non vide, substantielle et citant des outils du toolkit."""
    async with Client(build_server()) as client:
        text = await _render(client, name)
    assert isinstance(text, str)
    assert len(text.strip()) > 200
    # Un prompt actionnable cite au moins un outil ``<domaine>_*``.
    assert _TOOL_TOKEN.search(text) is not None


async def test_prompt_interpolates_its_arguments() -> None:
    """Les arguments passés sont interpolés dans le texte rendu (template réellement paramétré)."""
    async with Client(build_server()) as client:
        scaffold = await _render(client, "scaffold_multi_agent")
        guardrail = await _render(client, "add_guardrail")
    assert "trier des tickets de support" in scaffold
    assert "router" in guardrail
    assert "bloquer les PII" in guardrail


# --------------------------------------------------------------------------- #
# Cross-check : tout outil cité existe réellement (aucun nom inventé)
# --------------------------------------------------------------------------- #
async def test_every_cited_tool_token_is_a_real_tool() -> None:
    """Cross-check : chaque token ``<domaine>_<nom>`` cité dans un prompt est un outil réel."""
    real = await _real_tool_names()
    cited: set[str] = set()
    async with Client(build_server()) as client:
        for name in _PROMPT_ARGS:
            cited |= set(_TOOL_TOKEN.findall(await _render(client, name)))
    # Le cross-check n'a de sens que si les prompts citent vraiment des outils.
    assert cited, "aucun token d'outil cité — les prompts devraient référencer des outils"
    bogus = sorted(token for token in cited if token not in real)
    assert bogus == [], f"tokens cités mais inexistants comme outils : {bogus}"


async def test_key_workflow_tools_are_cited() -> None:
    """Chaque prompt cite l'outil pivot de son workflow (couverture du parcours-clé)."""
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
    """Chaque prompt de workflow porte le tag ``workflow`` (parité avec le tagging des outils)."""
    prompts = await build_server().list_prompts()
    by_name = {p.name: p for p in prompts}
    for name in _PROMPT_ARGS:
        assert "workflow" in (by_name[name].tags or set()), name
