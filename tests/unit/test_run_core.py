"""Tests unitaires du cœur d'exécution ``run_core`` (P3a).

PREUVE PORTEUSE (sans aucune clé API) : un ``FakeLlm(BaseLlm)`` permet d'exécuter une boucle
d'agent ADK complète hors-ligne, via ``build_runner`` + ``collect_events`` sur un
``RuntimeConfig`` in-memory. On prouve :
- réponse texte finale == texte canned (un seul événement final) ;
- boucle tool-call : function_call → function_response (tool exécuté par ADK) → texte final.

Couverture complémentaire :
- ``import_root_agent`` : import d'un ``root_agent`` ; reload après édition (nom de module
  unique → pas de cache périmé) ; erreurs (fichier absent, root_agent manquant, module cassé).
- ``serialize_event`` : aplatissement d'événements synthétiques.
- ``build_run_config`` : modes valides + mode invalide (ValueError).
- ``collect_events`` avec callback ``progress`` : awaité une fois par événement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ``tests/unit`` n'est pas un package (pas d'__init__.py) ; en mode d'import pytest par défaut
# (prepend), le dossier du test est sur sys.path → ``fake_llm`` est importable en top-level.
from fake_llm import FakeLlm, ScriptedLlm, add_numbers

from adk_toolkit_mcp.run_core import (
    PluginsImportError,
    RootAgentImportError,
    build_run_config,
    build_runner,
    collect_events,
    import_project_plugins,
    import_root_agent,
    serialize_event,
    streaming_mode_names,
)
from adk_toolkit_mcp.runtime import RuntimeConfig, SessionBackend, reset_service_cache


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole les tests : vide le cache singleton de services avant/après chacun."""
    reset_service_cache()
    yield
    reset_service_cache()


def _in_memory_config() -> RuntimeConfig:
    """RuntimeConfig minimal : sessions in_memory, pas de memory/artifacts."""
    return RuntimeConfig(session=SessionBackend(kind="in_memory"))


def _llm_agent(name: str, model: object, tools: list | None = None) -> object:
    """Construit un LlmAgent DIRECTEMENT (sans import de fichier) avec un FakeLlm."""
    from google.adk.agents import LlmAgent

    return LlmAgent(name=name, model=model, tools=tools or [])


# --------------------------------------------------------------------------- #
# PREUVE FONCTIONNELLE — exécution hors-ligne via FakeLlm
# --------------------------------------------------------------------------- #
async def test_functional_final_text_offline() -> None:
    """Un FakeLlm renvoyant un texte final fait produire au Runner un événement final == texte."""
    agent = _llm_agent("fake_agent", FakeLlm(model="fake", answer="Bonjour offline!"))
    runner = build_runner("app", agent, _in_memory_config())

    events = await collect_events(runner, user_id="u1", session_id="s1", new_message_text="salut")
    assert events, "au moins un événement attendu"
    serialized = [serialize_event(e) for e in events]
    finals = [s for s in serialized if s["is_final"]]
    assert finals, "un événement final attendu"
    assert finals[-1]["text"] == "Bonjour offline!"


async def test_functional_tool_call_loop_offline() -> None:
    """Boucle complète offline : function_call → function_response → texte final.

    PREUVE que le câblage Runner du toolkit exécute une vraie boucle d'agent (LLM → appel
    d'outil → exécution de l'outil par ADK → réponse finale), sans aucune clé API.
    """
    agent = _llm_agent(
        "calc",
        ScriptedLlm(
            model="scripted",
            tool_name="add_numbers",
            tool_args={"a": 2, "b": 3},
            final_text="The sum is 5.",
        ),
        tools=[add_numbers],
    )
    runner = build_runner("app", agent, _in_memory_config())

    events = await collect_events(
        runner, user_id="u1", session_id="s1", new_message_text="what is 2+3"
    )
    serialized = [serialize_event(e) for e in events]

    # 1) Un événement portant un function_call vers add_numbers(a=2,b=3).
    call_events = [s for s in serialized if s["function_calls"]]
    assert call_events, f"un function_call attendu, got {serialized}"
    fc = call_events[0]["function_calls"][0]
    assert fc["name"] == "add_numbers"
    assert fc["args"] == {"a": 2, "b": 3}

    # 2) Un événement portant la function_response (ADK a exécuté l'outil).
    resp_events = [s for s in serialized if s["function_responses"]]
    assert resp_events, f"une function_response attendue, got {serialized}"
    assert resp_events[0]["function_responses"][0]["name"] == "add_numbers"

    # 3) Un événement final portant le texte canned.
    finals = [s for s in serialized if s["is_final"]]
    assert finals, "un événement final attendu"
    assert finals[-1]["text"] == "The sum is 5."

    # Ordre : le function_call précède la function_response qui précède le final.
    call_idx = next(i for i, s in enumerate(serialized) if s["function_calls"])
    resp_idx = next(i for i, s in enumerate(serialized) if s["function_responses"])
    final_idx = next(i for i, s in enumerate(serialized) if s["is_final"])
    assert call_idx < resp_idx < final_idx


async def test_build_runner_wires_memory_and_artifacts() -> None:
    """build_runner passe les services memory/artifacts au Runner quand ils sont configurés.

    Prouve le câblage des trois services issus de runtime.py (sinon ces branches resteraient
    non exercées). Avec des backends in_memory configurés, le Runner doit exposer les instances.
    """
    from adk_toolkit_mcp.runtime import (
        ArtifactBackend,
        MemoryBackend,
        get_artifact_service,
        get_memory_service,
    )

    config = RuntimeConfig(
        session=SessionBackend(kind="in_memory"),
        memory=MemoryBackend(kind="in_memory"),
        artifacts=ArtifactBackend(kind="in_memory"),
    )
    agent = _llm_agent("fake_agent", FakeLlm(model="fake", answer="wired"))
    runner = build_runner("app", agent, config)

    # Les services câblés sont les mêmes instances (singleton) que les fabriques renvoient.
    assert runner.memory_service is get_memory_service(config.memory)
    assert runner.artifact_service is get_artifact_service(config.artifacts)

    # Et l'agent s'exécute toujours hors-ligne avec ce câblage complet.
    events = await collect_events(runner, user_id="u1", session_id="s1", new_message_text="hi")
    finals = [serialize_event(e) for e in events if e.is_final_response()]
    assert finals and finals[-1]["text"] == "wired"


async def test_collect_events_creates_missing_session() -> None:
    """collect_events crée la session si elle n'existe pas (auto_create_session=False côté ADK)."""
    agent = _llm_agent("fake_agent", FakeLlm(model="fake"))
    runner = build_runner("app", agent, _in_memory_config())
    # Aucune session pré-créée ; collect_events doit la créer puis exécuter sans erreur.
    events = await collect_events(
        runner, user_id="u1", session_id="brand-new", new_message_text="hi"
    )
    assert events


async def test_collect_events_progress_called_per_event() -> None:
    """Le callback progress est awaité une fois par événement, avec (index, event sérialisé)."""
    agent = _llm_agent(
        "calc",
        ScriptedLlm(model="scripted", tool_name="add_numbers", tool_args={"a": 1, "b": 1}),
        tools=[add_numbers],
    )
    runner = build_runner("app", agent, _in_memory_config())

    seen: list[tuple[int, dict]] = []

    async def _progress(index: int, event: dict) -> None:
        seen.append((index, event))

    events = await collect_events(
        runner,
        user_id="u1",
        session_id="s1",
        new_message_text="go",
        progress=_progress,
    )
    # Un appel de progress par événement, indices 1-based contigus.
    assert len(seen) == len(events)
    assert [i for i, _ in seen] == list(range(1, len(events) + 1))
    # Les payloads de progress sont des events sérialisés (clés attendues présentes).
    assert all("is_final" in payload and "author" in payload for _, payload in seen)


# --------------------------------------------------------------------------- #
# import_root_agent
# --------------------------------------------------------------------------- #
def _write_agent_py(root: Path, app_name: str, body: str) -> None:
    """Écrit ``<root>/<app_name>/agent.py`` avec le corps donné."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "agent.py").write_text(body, encoding="utf-8")


def test_import_root_agent_returns_root_agent(tmp_path: Path) -> None:
    """import_root_agent renvoie l'objet root_agent défini dans agent.py."""
    _write_agent_py(
        tmp_path,
        "myapp",
        "class _A:\n    name = 'root_agent'\n\nroot_agent = _A()\n",
    )
    agent = import_root_agent(str(tmp_path), "myapp")
    assert getattr(agent, "name", None) == "root_agent"


def test_import_root_agent_reload_picks_up_edits(tmp_path: Path) -> None:
    """Une édition d'agent.py est reprise (nom de module unique → pas de cache sys.modules)."""
    _write_agent_py(tmp_path, "myapp", "root_agent = 'v1'\n")
    first = import_root_agent(str(tmp_path), "myapp")
    assert first == "v1"

    # Édite le fichier puis ré-importe : doit refléter la NOUVELLE valeur.
    _write_agent_py(tmp_path, "myapp", "root_agent = 'v2'\n")
    second = import_root_agent(str(tmp_path), "myapp")
    assert second == "v2"


def test_import_root_agent_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RootAgentImportError, match="introuvable"):
        import_root_agent(str(tmp_path), "ghost")


def test_import_root_agent_missing_symbol_raises(tmp_path: Path) -> None:
    _write_agent_py(tmp_path, "myapp", "x = 1\n")  # pas de root_agent
    with pytest.raises(RootAgentImportError, match="root_agent"):
        import_root_agent(str(tmp_path), "myapp")


def test_import_root_agent_broken_module_raises(tmp_path: Path) -> None:
    _write_agent_py(tmp_path, "myapp", "raise RuntimeError('boom')\n")
    with pytest.raises(RootAgentImportError, match="Échec de l'import"):
        import_root_agent(str(tmp_path), "myapp")


async def test_import_then_run_offline(tmp_path: Path) -> None:
    """Bout-en-bout offline : agent.py importe un FakeLlm de la fixture → run produit le texte.

    Prouve qu'un agent CHARGÉ DEPUIS UN FICHIER (et non construit en mémoire) s'exécute
    hors-ligne via le câblage du toolkit. On rend la fixture importable via sys.path.
    """
    fixture_dir = str(Path(__file__).parent)
    body = (
        "import sys\n"
        f"sys.path.insert(0, r'{fixture_dir}')\n"
        "from fake_llm import FakeLlm\n"
        "from google.adk.agents import LlmAgent\n"
        "root_agent = LlmAgent(name='filed', model=FakeLlm(model='fake', answer='From file!'))\n"
    )
    _write_agent_py(tmp_path, "myapp", body)

    agent = import_root_agent(str(tmp_path), "myapp")
    runner = build_runner("myapp", agent, _in_memory_config())
    events = await collect_events(runner, user_id="u1", session_id="s1", new_message_text="hi")
    finals = [serialize_event(e) for e in events if e.is_final_response()]
    assert finals and finals[-1]["text"] == "From file!"


# --------------------------------------------------------------------------- #
# serialize_event (événements synthétiques)
# --------------------------------------------------------------------------- #
def test_serialize_event_text_only() -> None:
    """Un event texte simple → text rempli, listes vides, non final si partial."""
    from google.adk.events import Event
    from google.genai import types

    ev = Event(
        author="assistant",
        content=types.Content(role="model", parts=[types.Part.from_text(text="hello world")]),
        partial=True,
    )
    s = serialize_event(ev)
    assert s["author"] == "assistant"
    assert s["text"] == "hello world"
    assert s["function_calls"] == []
    assert s["function_responses"] == []
    assert s["state_delta"] == {}
    assert s["partial"] is True


def test_serialize_event_function_call_and_state_delta() -> None:
    """Un event function_call + state_delta + transfer → champs correctement extraits."""
    from google.adk.events import Event, EventActions
    from google.genai import types

    ev = Event(
        author="planner",
        content=types.Content(
            role="model", parts=[types.Part.from_function_call(name="search", args={"q": "adk"})]
        ),
        actions=EventActions(state_delta={"app:hits": 3}, transfer_to_agent="worker"),
    )
    s = serialize_event(ev)
    assert s["function_calls"] == [{"name": "search", "args": {"q": "adk"}}]
    assert s["text"] is None  # un part function_call n'a pas de texte
    assert s["state_delta"] == {"app:hits": 3}
    assert s["transfer_to_agent"] == "worker"


def test_serialize_event_no_content() -> None:
    """Un event sans content → text None, listes vides (pas d'exception)."""
    from google.adk.events import Event

    s = serialize_event(Event(author="user"))
    assert s["text"] is None
    assert s["function_calls"] == []
    assert s["is_final"] in (True, False)


# --------------------------------------------------------------------------- #
# build_run_config
# --------------------------------------------------------------------------- #
def test_build_run_config_valid_modes() -> None:
    """NONE/SSE/BIDI (insensible à la casse) construisent un RunConfig avec le bon mode."""
    from google.adk.agents.run_config import StreamingMode

    assert build_run_config("NONE").streaming_mode == StreamingMode.NONE
    assert build_run_config("sse").streaming_mode == StreamingMode.SSE
    assert build_run_config("Bidi").streaming_mode == StreamingMode.BIDI


def test_build_run_config_max_llm_calls_forwarded() -> None:
    """max_llm_calls fourni est transmis ; None laisse le défaut ADK (500)."""
    assert build_run_config("NONE", max_llm_calls=7).max_llm_calls == 7
    assert build_run_config("NONE", max_llm_calls=None).max_llm_calls == 500


def test_build_run_config_response_modalities() -> None:
    """response_modalities est transmis quand fourni."""
    rc = build_run_config("NONE", response_modalities=["TEXT"])
    assert rc.response_modalities == ["TEXT"]


def test_build_run_config_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="streaming_mode invalide"):
        build_run_config("TURBO")


def test_streaming_mode_names() -> None:
    names = streaming_mode_names()
    assert set(names) == {"NONE", "SSE", "BIDI"}


# --------------------------------------------------------------------------- #
# Plugins (P4c) — build_runner via App + import_project_plugins
# --------------------------------------------------------------------------- #
def _rec_plugin() -> object:
    """Construit un BasePlugin qui enregistre l'auteur de chaque évènement (preuve hors-ligne)."""
    from google.adk.plugins import BasePlugin

    class _RecPlugin(BasePlugin):
        def __init__(self, name: str) -> None:
            super().__init__(name=name)
            self.seen: list[str] = []

        async def on_event_callback(self, *, invocation_context, event):  # noqa: ANN001
            self.seen.append(event.author)
            return None

    return _RecPlugin(name="rec")


async def test_functional_plugin_wired_via_build_runner() -> None:
    """PREUVE : un plugin passé à build_runner câble Runner(app=App(plugins=[...])) et s'exécute.

    On lance un FakeLlm hors-ligne ; le plugin enregistre les évènements. Prouve le câblage
    Runner(plugins) via le chemin App (non déprécié) de bout en bout, sans clé API.
    """
    plugin = _rec_plugin()
    agent = _llm_agent("fa", FakeLlm(model="f", answer="plugged"))
    runner = build_runner("app", agent, _in_memory_config(), plugins=[plugin])

    # app_name est dérivé de App.name (chemin App).
    assert runner.app_name == "app"

    events = await collect_events(runner, user_id="u", session_id="s", new_message_text="hi")
    finals = [serialize_event(e) for e in events if e.is_final_response()]
    assert finals and finals[-1]["text"] == "plugged"
    # Le plugin a bien vu des évènements (hook on_event_callback déclenché).
    assert plugin.seen, "le plugin aurait dû enregistrer au moins un évènement"


def test_build_runner_no_plugins_unchanged() -> None:
    """Sans plugins, build_runner garde le chemin Runner(app_name=, agent=) (compat ascendante)."""
    agent = _llm_agent("fa", FakeLlm(model="f"))
    runner = build_runner("app", agent, _in_memory_config())
    assert runner.app_name == "app"
    # Aucun plugin câblé.
    assert not getattr(runner, "plugin_manager", None) or not runner.plugin_manager.plugins


def _write_plugins_py(root: Path, app_name: str, body: str) -> None:
    """Écrit ``<root>/<app_name>/plugins.py`` avec le corps donné."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "plugins.py").write_text(body, encoding="utf-8")


def test_import_project_plugins_returns_instances(tmp_path: Path) -> None:
    """import_project_plugins renvoie les instances nommées dans plugins.py (ordre préservé)."""
    _write_plugins_py(
        tmp_path,
        "myapp",
        "from google.adk.plugins import BasePlugin\n"
        "p1 = BasePlugin(name='one')\n"
        "p2 = BasePlugin(name='two')\n",
    )
    instances = import_project_plugins(str(tmp_path), "myapp", ["p1", "p2"])
    assert [p.name for p in instances] == ["one", "two"]


def test_import_project_plugins_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PluginsImportError, match="introuvable"):
        import_project_plugins(str(tmp_path), "ghost", ["p"])


def test_import_project_plugins_missing_var_raises(tmp_path: Path) -> None:
    _write_plugins_py(
        tmp_path,
        "myapp",
        "from google.adk.plugins import BasePlugin\np1 = BasePlugin(name='one')\n",
    )
    with pytest.raises(PluginsImportError, match="ne définit pas la variable"):
        import_project_plugins(str(tmp_path), "myapp", ["missing"])


def test_import_project_plugins_broken_module_raises(tmp_path: Path) -> None:
    _write_plugins_py(tmp_path, "myapp", "raise RuntimeError('boom')\n")
    with pytest.raises(PluginsImportError, match="Échec de l'import"):
        import_project_plugins(str(tmp_path), "myapp", ["p"])
