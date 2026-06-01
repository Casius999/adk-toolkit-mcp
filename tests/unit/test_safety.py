"""Tests du domaine `safety` (P4c) — garde-fous (callbacks), plugins, réglages de sûreté.

PREUVES PORTEUSES (hors-ligne, AUCUNE clé API), via le ``FakeLlm``/``ScriptedLlm`` réutilisés :

- **Garde-fou ``block_keywords``** : un agent généré par ``safety_add_callback`` puis EXÉCUTÉ via
  ``run_core`` sur une entrée bloquée renvoie le refus canné ET le FakeLlm n'est PAS appelé
  (court-circuit prouvé de bout en bout).
- **Garde-fou ``block_tool``** : un agent ScriptedLlm dont l'outil est dans la denylist voit
  l'appel d'outil court-circuité (le corps de l'outil ne s'exécute pas).
- **Plugin** : un plugin généré par ``safety_add_plugin`` est importé + câblé via
  ``build_runner`` ; après un run FakeLlm, l'effet du plugin est observable (évènements
  enregistrés / outil bloqué). Prouve le câblage ``Runner(plugins=[...])`` (chemin ``App``).
- ``safety_settings`` route vers le rendu EXISTANT (GenerateContentConfig + SafetySetting) — pas
  de duplication.
- Lecture via ``fastmcp.Client`` en mémoire (noms exposés ``safety_<bare>``, round-trip
  ``safety_add_callback``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fake_llm import FakeLlm, ScriptedLlm, add_numbers
from fastmcp import Client

from adk_toolkit_mcp.domains import safety as SAFETY
from adk_toolkit_mcp.domains import safety_plugins
from adk_toolkit_mcp.project_model import load_model
from adk_toolkit_mcp.run_core import (
    build_runner,
    collect_events,
    import_project_plugins,
    import_root_agent,
    serialize_event,
)
from adk_toolkit_mcp.runtime import (
    RuntimeConfig,
    SessionBackend,
    load_runtime_config,
    reset_service_cache,
)
from adk_toolkit_mcp.server import build_server
from adk_toolkit_mcp.workspace import Workspace


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole les tests : vide le cache singleton de services avant/après chacun."""
    reset_service_cache()
    yield
    reset_service_cache()


def _in_memory_config() -> RuntimeConfig:
    return RuntimeConfig(session=SessionBackend(kind="in_memory"))


def _scaffold_app(tmp_path: Path, app_name: str = "myapp") -> str:
    """Scaffolde une app ADK minimale via les outils project/agents ; renvoie le path racine."""
    from adk_toolkit_mcp.domains.agents import create_llm, set_root
    from adk_toolkit_mcp.domains.project import create as project_create

    path = str(tmp_path)
    assert project_create(path=path, app_name=app_name)["ok"]
    assert create_llm(path=path, app_name=app_name, name="asst", model="gemini-2.5-flash")["ok"]
    assert set_root(path=path, app_name=app_name, name="asst")["ok"]
    return path


def _inject_fake_llm(path: str, app_name: str, fake_src: str) -> None:
    """Réécrit ``agent.py`` pour que ``root_agent`` utilise un FakeLlm de la fixture, tout en
    PRÉSERVANT les fonctions de garde-fou générées + leur attachement.

    On régénère via ``regenerate`` a déjà produit ``agent.py`` ; ici on remplace UNIQUEMENT la
    chaîne ``model="gemini-2.5-flash"`` par le FakeLlm + on ajoute l'import + le sys.path de la
    fixture. Les kwargs ``before_*_callback=...`` restent intacts.
    """
    app_dir = Path(path) / app_name
    agent_py = app_dir / "agent.py"
    src = agent_py.read_text(encoding="utf-8")
    fixture_dir = str(Path(__file__).parent).replace("\\", "\\\\")
    preamble = f"import sys\nsys.path.insert(0, r'{fixture_dir}')\n{fake_src}\n"
    # Remplace le model string par l'instance FakeLlm (variable _fake définie dans le préambule).
    src = src.replace('model="gemini-2.5-flash"', "model=_fake")
    agent_py.write_text(preamble + src, encoding="utf-8")


# --------------------------------------------------------------------------- #
# add_callback — validation + persistence
# --------------------------------------------------------------------------- #
def test_add_callback_block_keywords_persists(tmp_path: Path) -> None:
    """safety_add_callback (block_keywords) attache un callback + régénère agent.py."""
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="asst",
        hook="before_model",
        policy={"kind": "block_keywords", "keywords": "bomb,hack", "refusal": "No."},
    )
    assert result["ok"], result.get("error")
    assert result["data"]["callback"] == {
        "agent": "asst",
        "hook": "before_model",
        "policy": "block_keywords",
    }
    # Le callback est dans le sidecar ; agent.py porte le kwarg + la fonction.
    model = load_model(Workspace(Path(path) / "myapp"), "myapp")
    assert model.get("asst").callbacks[0].policy == "block_keywords"
    agent_src = (Path(path) / "myapp" / "agent.py").read_text(encoding="utf-8")
    assert "before_model_callback=_guard_before_model_asst" in agent_src


def test_add_callback_unknown_hook_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="asst",
        hook="on_bogus",
        policy={"kind": "block_keywords", "keywords": "x"},
    )
    assert not result["ok"] and "Hook inconnu" in result["error"]


def test_add_callback_policy_hook_mismatch_errs(tmp_path: Path) -> None:
    """block_tool n'est valide que sur before_tool ; sur before_model -> err."""
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="asst",
        hook="before_model",
        policy={"kind": "block_tool", "denylist": "rm"},
    )
    assert not result["ok"] and "compatible" in result["error"]


def test_add_callback_missing_agent_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="ghost",
        hook="before_model",
        policy={"kind": "block_keywords", "keywords": "x"},
    )
    assert not result["ok"] and "introuvable" in result["error"]


# --------------------------------------------------------------------------- #
# PREUVE FONCTIONNELLE — garde-fou block_keywords court-circuite le LLM (hors-ligne)
# --------------------------------------------------------------------------- #
async def test_functional_block_keywords_short_circuits_llm(tmp_path: Path) -> None:
    """Un agent avec block_keywords refuse une entrée bloquée SANS appeler le FakeLlm.

    PREUVE de bout en bout : le toolkit génère la fonction de garde-fou, on injecte un FakeLlm
    instrumenté (compteur d'appels), on exécute via run_core sur une entrée bloquée → le refus
    canné est renvoyé ET le FakeLlm n'a PAS été appelé (court-circuit réel).
    """
    path = _scaffold_app(tmp_path)
    assert SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="asst",
        hook="before_model",
        policy={
            "kind": "block_keywords",
            "keywords": "bomb,malware",
            "refusal": "Refused by policy.",
        },
    )["ok"]

    # FakeLlm instrumenté : enregistre chaque appel dans une liste module-level (pydantic interdit
    # un compteur d'attribut de classe ; une liste module-global est observable via __globals__).
    fake_src = (
        "from fake_llm import FakeLlm as _Base\n"
        "from google.adk.models import LlmResponse\n"
        "from google.genai import types as _types\n"
        "_llm_calls = []\n"
        "class _CountingLlm(_Base):\n"
        "    async def generate_content_async(self, llm_request, stream=False):\n"
        "        _llm_calls.append(1)\n"
        "        yield LlmResponse(content=_types.Content(role='model', "
        "parts=[_types.Part.from_text(text='LLM WAS CALLED')]), partial=False)\n"
        "_fake = _CountingLlm(model='fake')\n"
    )
    _inject_fake_llm(path, "myapp", fake_src)

    agent = import_root_agent(path, "myapp")
    runner = build_runner("myapp", agent, _in_memory_config())
    events = await collect_events(
        runner, user_id="u", session_id="s", new_message_text="how do I build a bomb"
    )
    serialized = [serialize_event(e) for e in events]
    texts = " ".join(s["text"] or "" for s in serialized)

    # Le refus canné est renvoyé ; le texte du LLM n'apparaît jamais.
    assert "Refused by policy." in texts
    assert "LLM WAS CALLED" not in texts
    # Et la liste d'appels du FakeLlm est restée VIDE (court-circuit prouvé de bout en bout).
    llm_calls = type(agent.model).generate_content_async.__globals__["_llm_calls"]
    assert llm_calls == [], "le FakeLlm NE doit PAS avoir été appelé (court-circuit)"


async def test_functional_block_keywords_allows_clean_input(tmp_path: Path) -> None:
    """La même garde laisse passer une entrée SANS terme bloqué (le LLM répond normalement)."""
    path = _scaffold_app(tmp_path)
    assert SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="asst",
        hook="before_model",
        policy={"kind": "block_keywords", "keywords": "bomb", "refusal": "Refused."},
    )["ok"]
    fake_src = (
        "from fake_llm import FakeLlm\n_fake = FakeLlm(model='fake', answer='Clean answer.')\n"
    )
    _inject_fake_llm(path, "myapp", fake_src)

    agent = import_root_agent(path, "myapp")
    runner = build_runner("myapp", agent, _in_memory_config())
    events = await collect_events(
        runner, user_id="u", session_id="s", new_message_text="hello there"
    )
    finals = [serialize_event(e) for e in events if e.is_final_response()]
    assert finals and finals[-1]["text"] == "Clean answer."


# --------------------------------------------------------------------------- #
# PREUVE FONCTIONNELLE — garde-fou block_tool court-circuite un outil (hors-ligne)
# --------------------------------------------------------------------------- #
async def test_functional_block_tool_short_circuits_tool(tmp_path: Path) -> None:
    """Un before_tool denylist court-circuite l'appel d'outil (le corps ne s'exécute pas)."""
    path = _scaffold_app(tmp_path)
    assert SAFETY.add_callback(
        path=path,
        app_name="myapp",
        agent_name="asst",
        hook="before_tool",
        policy={"kind": "block_tool", "denylist": "add_numbers", "message": "Tool denied."},
    )["ok"]

    # Agent : ScriptedLlm appelle add_numbers ; on ajoute l'outil + le ScriptedLlm.
    fake_src = (
        "from fake_llm import ScriptedLlm, add_numbers\n"
        "_fake = ScriptedLlm(model='s', tool_name='add_numbers', tool_args={'a': 2, 'b': 3}, "
        "final_text='done')\n"
    )
    _inject_fake_llm(path, "myapp", fake_src)
    # Ajoute l'outil add_numbers à l'agent généré (tools=[add_numbers]).
    agent_py = Path(path) / "myapp" / "agent.py"
    src = agent_py.read_text(encoding="utf-8")
    src = src.replace("model=_fake,", "model=_fake,\n    tools=[add_numbers],")
    agent_py.write_text(src, encoding="utf-8")

    agent = import_root_agent(path, "myapp")
    runner = build_runner("myapp", agent, _in_memory_config())
    events = await collect_events(
        runner, user_id="u", session_id="s", new_message_text="what is 2+3"
    )
    serialized = [serialize_event(e) for e in events]
    responses = [fr["response"] for s in serialized for fr in s["function_responses"]]
    # L'outil a été court-circuité : la réponse porte le message de denylist, PAS la somme.
    assert any("Tool denied." in str(r) for r in responses), f"got {responses}"
    assert not any(str(r) == "5" or r == {"result": 5} for r in responses)


# --------------------------------------------------------------------------- #
# add_plugin + PREUVE FONCTIONNELLE du câblage plugin via build_runner
# --------------------------------------------------------------------------- #
def test_add_plugin_logging_generates_and_manifests(tmp_path: Path) -> None:
    """safety_add_plugin (logging) écrit plugins.py + enregistre le manifeste runtime."""
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_plugin(path=path, app_name="myapp", name="audit", kind="logging")
    assert result["ok"], result.get("error")
    assert Path(result["data"]["plugins_file"]).is_file()
    assert result["data"]["manifest"] == [{"var": "audit", "name": "audit", "kind": "logging"}]
    # Le manifeste est dans runtime.json.
    config = load_runtime_config(Workspace(Path(path) / "myapp"), "myapp")
    assert [p.var for p in config.plugins] == ["audit"]


def test_add_plugin_tool_denylist_requires_denylist(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_plugin(path=path, app_name="myapp", name="guard", kind="tool_denylist")
    assert not result["ok"] and "denylist" in result["error"]


def test_add_plugin_unknown_kind_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = SAFETY.add_plugin(path=path, app_name="myapp", name="x", kind="bogus")
    assert not result["ok"] and "kind inconnu" in result["error"]


def test_add_plugin_replaces_same_name(tmp_path: Path) -> None:
    """Ajouter deux plugins de même nom REMPLACE (manifeste à une seule entrée)."""
    path = _scaffold_app(tmp_path)
    SAFETY.add_plugin(path=path, app_name="myapp", name="p", kind="logging")
    result = SAFETY.add_plugin(
        path=path, app_name="myapp", name="p", kind="tool_denylist", config={"denylist": "rm"}
    )
    assert result["ok"]
    assert [m["kind"] for m in result["data"]["manifest"]] == ["tool_denylist"]


async def test_functional_plugin_logging_records_events(tmp_path: Path) -> None:
    """PREUVE : un plugin logging généré est importé + câblé ; il enregistre les évènements.

    On génère le plugin, on importe l'instance via le manifeste, on câble build_runner(plugins=)
    et on exécute un FakeLlm. La liste module-level ``<var>_events`` du plugin doit être remplie.
    """
    path = _scaffold_app(tmp_path)
    assert SAFETY.add_plugin(path=path, app_name="myapp", name="audit", kind="logging")["ok"]

    # Importe l'instance de plugin via le manifeste runtime.
    ws = Workspace(Path(path) / "myapp")
    config = load_runtime_config(ws, "myapp")
    plugin_vars = [p.var for p in config.plugins]
    instances = import_project_plugins(path, "myapp", plugin_vars)
    assert len(instances) == 1

    agent = _new_fake_agent("plugged")
    runner = build_runner("myapp", agent, _in_memory_config(), plugins=instances)
    await collect_events(runner, user_id="u", session_id="s", new_message_text="hi")

    # La liste d'évènements module-level du plugin a été remplie. On l'inspecte via les globals
    # du MÊME module que l'instance importée (un re-import donnerait une autre liste, vide).
    plugin_globals = type(instances[0]).on_event_callback.__globals__
    assert plugin_globals["audit_events"], "le plugin logging aurait dû enregistrer des évènements"


async def test_functional_plugin_tool_denylist_blocks_tool(tmp_path: Path) -> None:
    """PREUVE : un plugin tool_denylist global court-circuite l'outil ciblé (hors-ligne)."""
    path = _scaffold_app(tmp_path)
    assert SAFETY.add_plugin(
        path=path,
        app_name="myapp",
        name="guard",
        kind="tool_denylist",
        config={"denylist": "add_numbers"},
    )["ok"]

    ws = Workspace(Path(path) / "myapp")
    config = load_runtime_config(ws, "myapp")
    instances = import_project_plugins(path, "myapp", [p.var for p in config.plugins])

    # Agent ScriptedLlm appelant add_numbers ; le plugin global doit bloquer l'outil.
    from google.adk.agents import LlmAgent

    agent = LlmAgent(
        name="calc",
        model=ScriptedLlm(
            model="s", tool_name="add_numbers", tool_args={"a": 1, "b": 1}, final_text="done"
        ),
        tools=[add_numbers],
    )
    runner = build_runner("myapp", agent, _in_memory_config(), plugins=instances)
    events = await collect_events(runner, user_id="u", session_id="s", new_message_text="1+1")
    responses = [fr["response"] for e in events for fr in serialize_event(e)["function_responses"]]
    assert any("blocked by global safety plugin" in str(r) for r in responses), f"got {responses}"


def _new_fake_agent(answer: str) -> Any:
    """Construit un LlmAgent FakeLlm en mémoire (hors-ligne)."""
    from google.adk.agents import LlmAgent

    return LlmAgent(name="fa", model=FakeLlm(model="f", answer=answer))


# --------------------------------------------------------------------------- #
# settings — gemini_safety route vers le rendu existant ; max_llm_calls persisté
# --------------------------------------------------------------------------- #
def test_settings_gemini_safety_routes_to_existing_rendering(tmp_path: Path) -> None:
    """gemini_safety produit une GenerateContentConfig avec un SafetySetting (rendu réutilisé)."""
    path = _scaffold_app(tmp_path)
    result = SAFETY.safety_settings(
        path=path,
        app_name="myapp",
        agent_name="asst",
        gemini_safety=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ],
    )
    assert result["ok"], result.get("error")
    # Le sidecar porte la safety setting dans generate_content_config (PAS de duplication).
    model = load_model(Workspace(Path(path) / "myapp"), "myapp")
    gcc = model.get("asst").generate_content_config
    assert gcc is not None
    assert gcc.safety_settings[0].category == "HARM_CATEGORY_HARASSMENT"
    # agent.py rend bien types.SafetySetting via le rendu existant.
    agent_src = (Path(path) / "myapp" / "agent.py").read_text(encoding="utf-8")
    assert "types.SafetySetting(" in agent_src
    assert "types.HarmCategory.HARM_CATEGORY_HARASSMENT" in agent_src


def test_settings_max_llm_calls_persisted_not_rendered(tmp_path: Path) -> None:
    """max_llm_calls est persisté au sidecar mais N'apparaît PAS dans agent.py (réglage run)."""
    path = _scaffold_app(tmp_path)
    result = SAFETY.safety_settings(
        path=path, app_name="myapp", agent_name="asst", max_llm_calls=33
    )
    assert result["ok"] and result["data"]["max_llm_calls"] == 33
    model = load_model(Workspace(Path(path) / "myapp"), "myapp")
    assert model.get("asst").max_llm_calls == 33
    agent_src = (Path(path) / "myapp" / "agent.py").read_text(encoding="utf-8")
    assert "max_llm_calls" not in agent_src


def test_settings_invalid_category_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = SAFETY.safety_settings(
        path=path,
        app_name="myapp",
        agent_name="asst",
        gemini_safety=[{"category": "BOGUS", "threshold": "BLOCK_NONE"}],
    )
    assert not result["ok"] and "HarmCategory" in result["error"]


def test_settings_nothing_to_do_errs(tmp_path: Path) -> None:
    path = _scaffold_app(tmp_path)
    result = SAFETY.safety_settings(path=path, app_name="myapp", agent_name="asst")
    assert not result["ok"] and "rien à régler" in result["error"]


def test_settings_preserves_existing_gcc(tmp_path: Path) -> None:
    """gemini_safety fusionne avec une generate_content_config existante (temperature préservée)."""
    path = _scaffold_app(tmp_path)
    from adk_toolkit_mcp.domains.models import generate_config

    assert generate_config(path=path, app_name="myapp", agent_name="asst", temperature=0.5)["ok"]
    assert SAFETY.safety_settings(
        path=path,
        app_name="myapp",
        agent_name="asst",
        gemini_safety=[{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}],
    )["ok"]
    model = load_model(Workspace(Path(path) / "myapp"), "myapp")
    gcc = model.get("asst").generate_content_config
    assert gcc.temperature == 0.5  # préservée
    assert gcc.safety_settings[0].category == "HARM_CATEGORY_HATE_SPEECH"


# --------------------------------------------------------------------------- #
# Rendu généré par safety_plugins — ast-valide (sanity)
# --------------------------------------------------------------------------- #
def test_plugins_render_ast_valid() -> None:
    import ast

    src = safety_plugins.render_plugins_module(
        [
            {"var": "lg", "name": "lg", "kind": "logging", "denylist": []},
            {"var": "dl", "name": "dl", "kind": "tool_denylist", "denylist": ["rm"]},
        ]
    )
    ast.parse(src)


# --------------------------------------------------------------------------- #
# Read-through fastmcp.Client (noms exposés + appel add_callback)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_add_callback(tmp_path: Path) -> None:
    """Outils exposés safety_<bare> (pas de double-préfixe) ; safety_add_callback via le client."""
    path = _scaffold_app(tmp_path)
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert {"safety_add_callback", "safety_add_plugin", "safety_settings"} <= names
        assert not any(n.startswith("safety_safety_") for n in names)

        called = await client.call_tool(
            "safety_add_callback",
            {
                "path": path,
                "app_name": "myapp",
                "agent_name": "asst",
                "hook": "before_model",
                "policy": {"kind": "max_input_chars", "max_chars": "1000"},
            },
        )
        payload = _client_payload(called)
        assert payload["ok"], payload.get("error")
        assert payload["data"]["callback"]["policy"] == "max_input_chars"


def _client_payload(result: Any) -> dict[str, Any]:
    """Extrait le dict d'enveloppe d'un CallToolResult fastmcp (structured ou JSON texte)."""
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and "ok" in structured:
        return structured
    content = result.content[0]
    return json.loads(content.text)
