"""Unit tests for the ``run_core`` execution core (P3a).

LOAD-BEARING PROOF (without any API key): a ``FakeLlm(BaseLlm)`` lets us run a complete ADK agent
loop offline, via ``build_runner`` + ``collect_events`` on an in-memory ``RuntimeConfig``. We
prove:
- final text response == canned text (a single final event);
- tool-call loop: function_call → function_response (tool executed by ADK) → final text.

Complementary coverage:
- ``import_root_agent``: importing a ``root_agent``; reload after an edit (unique module name → no
  stale cache); errors (missing file, missing root_agent, broken module).
- ``serialize_event``: flattening of synthetic events.
- ``build_run_config``: valid modes + invalid mode (ValueError).
- ``collect_events`` with a ``progress`` callback: awaited once per event.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ``tests/unit`` is not a package (no __init__.py); in pytest's default import mode (prepend), the
# test's folder is on sys.path → ``fake_llm`` is importable at the top level.
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
    """Isolate the tests: clear the singleton service cache before/after each."""
    reset_service_cache()
    yield
    reset_service_cache()


def _in_memory_config() -> RuntimeConfig:
    """Minimal RuntimeConfig: in_memory sessions, no memory/artifacts."""
    return RuntimeConfig(session=SessionBackend(kind="in_memory"))


def _llm_agent(name: str, model: object, tools: list | None = None) -> object:
    """Build an LlmAgent DIRECTLY (no file import) with a FakeLlm."""
    from google.adk.agents import LlmAgent

    return LlmAgent(name=name, model=model, tools=tools or [])


# --------------------------------------------------------------------------- #
# FUNCTIONAL PROOF — offline execution via FakeLlm
# --------------------------------------------------------------------------- #
async def test_functional_final_text_offline() -> None:
    """A FakeLlm returning final text makes the Runner produce a final event == that text."""
    agent = _llm_agent("fake_agent", FakeLlm(model="fake", answer="Hello offline!"))
    runner = build_runner("app", agent, _in_memory_config())

    events = await collect_events(runner, user_id="u1", session_id="s1", new_message_text="hi")
    assert events, "at least one event expected"
    serialized = [serialize_event(e) for e in events]
    finals = [s for s in serialized if s["is_final"]]
    assert finals, "a final event expected"
    assert finals[-1]["text"] == "Hello offline!"


async def test_functional_tool_call_loop_offline() -> None:
    """Complete offline loop: function_call → function_response → final text.

    PROOF that the toolkit's Runner wiring runs a real agent loop (LLM → tool call → tool
    execution by ADK → final response), without any API key.
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

    # 1) An event carrying a function_call to add_numbers(a=2,b=3).
    call_events = [s for s in serialized if s["function_calls"]]
    assert call_events, f"a function_call expected, got {serialized}"
    fc = call_events[0]["function_calls"][0]
    assert fc["name"] == "add_numbers"
    assert fc["args"] == {"a": 2, "b": 3}

    # 2) An event carrying the function_response (ADK executed the tool).
    resp_events = [s for s in serialized if s["function_responses"]]
    assert resp_events, f"a function_response expected, got {serialized}"
    assert resp_events[0]["function_responses"][0]["name"] == "add_numbers"

    # 3) A final event carrying the canned text.
    finals = [s for s in serialized if s["is_final"]]
    assert finals, "a final event expected"
    assert finals[-1]["text"] == "The sum is 5."

    # Order: the function_call precedes the function_response which precedes the final.
    call_idx = next(i for i, s in enumerate(serialized) if s["function_calls"])
    resp_idx = next(i for i, s in enumerate(serialized) if s["function_responses"])
    final_idx = next(i for i, s in enumerate(serialized) if s["is_final"])
    assert call_idx < resp_idx < final_idx


async def test_build_runner_wires_memory_and_artifacts() -> None:
    """build_runner passes the memory/artifacts services to the Runner when they are configured.

    Proves the wiring of the three services from runtime.py (otherwise those branches would stay
    unexercised). With configured in_memory backends, the Runner must expose the instances.
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

    # The wired services are the same (singleton) instances that the factories return.
    assert runner.memory_service is get_memory_service(config.memory)
    assert runner.artifact_service is get_artifact_service(config.artifacts)

    # And the agent still runs offline with this complete wiring.
    events = await collect_events(runner, user_id="u1", session_id="s1", new_message_text="hi")
    finals = [serialize_event(e) for e in events if e.is_final_response()]
    assert finals and finals[-1]["text"] == "wired"


async def test_collect_events_creates_missing_session() -> None:
    """collect_events creates the session if it doesn't exist (auto_create_session=False in ADK)."""
    agent = _llm_agent("fake_agent", FakeLlm(model="fake"))
    runner = build_runner("app", agent, _in_memory_config())
    # No pre-created session; collect_events must create it then run without error.
    events = await collect_events(
        runner, user_id="u1", session_id="brand-new", new_message_text="hi"
    )
    assert events


async def test_collect_events_progress_called_per_event() -> None:
    """The progress callback is awaited once per event, with (index, serialized event)."""
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
    # One progress call per event, contiguous 1-based indices.
    assert len(seen) == len(events)
    assert [i for i, _ in seen] == list(range(1, len(events) + 1))
    # The progress payloads are serialized events (expected keys present).
    assert all("is_final" in payload and "author" in payload for _, payload in seen)


# --------------------------------------------------------------------------- #
# import_root_agent
# --------------------------------------------------------------------------- #
def _write_agent_py(root: Path, app_name: str, body: str) -> None:
    """Write ``<root>/<app_name>/agent.py`` with the given body."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "agent.py").write_text(body, encoding="utf-8")


def test_import_root_agent_returns_root_agent(tmp_path: Path) -> None:
    """import_root_agent returns the root_agent object defined in agent.py."""
    _write_agent_py(
        tmp_path,
        "myapp",
        "class _A:\n    name = 'root_agent'\n\nroot_agent = _A()\n",
    )
    agent = import_root_agent(str(tmp_path), "myapp")
    assert getattr(agent, "name", None) == "root_agent"


def test_import_root_agent_reload_picks_up_edits(tmp_path: Path) -> None:
    """An edit to agent.py is picked up (unique module name → no sys.modules cache)."""
    _write_agent_py(tmp_path, "myapp", "root_agent = 'v1'\n")
    first = import_root_agent(str(tmp_path), "myapp")
    assert first == "v1"

    # Edit the file then re-import: must reflect the NEW value.
    _write_agent_py(tmp_path, "myapp", "root_agent = 'v2'\n")
    second = import_root_agent(str(tmp_path), "myapp")
    assert second == "v2"


def test_import_root_agent_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RootAgentImportError, match="not found"):
        import_root_agent(str(tmp_path), "ghost")


def test_import_root_agent_missing_symbol_raises(tmp_path: Path) -> None:
    _write_agent_py(tmp_path, "myapp", "x = 1\n")  # no root_agent
    with pytest.raises(RootAgentImportError, match="root_agent"):
        import_root_agent(str(tmp_path), "myapp")


def test_import_root_agent_broken_module_raises(tmp_path: Path) -> None:
    _write_agent_py(tmp_path, "myapp", "raise RuntimeError('boom')\n")
    with pytest.raises(RootAgentImportError, match="Failed to import"):
        import_root_agent(str(tmp_path), "myapp")


async def test_import_then_run_offline(tmp_path: Path) -> None:
    """Offline end-to-end: agent.py imports a FakeLlm from the fixture → run produces the text.

    Proves that an agent LOADED FROM A FILE (not built in memory) runs offline via the toolkit's
    wiring. We make the fixture importable via sys.path.
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
# serialize_event (synthetic events)
# --------------------------------------------------------------------------- #
def test_serialize_event_text_only() -> None:
    """A simple text event → text filled, empty lists, not final if partial."""
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
    """A function_call + state_delta + transfer event → fields correctly extracted."""
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
    assert s["text"] is None  # a function_call part has no text
    assert s["state_delta"] == {"app:hits": 3}
    assert s["transfer_to_agent"] == "worker"


def test_serialize_event_no_content() -> None:
    """An event without content → text None, empty lists (no exception)."""
    from google.adk.events import Event

    s = serialize_event(Event(author="user"))
    assert s["text"] is None
    assert s["function_calls"] == []
    assert s["is_final"] in (True, False)


# --------------------------------------------------------------------------- #
# build_run_config
# --------------------------------------------------------------------------- #
def test_build_run_config_valid_modes() -> None:
    """NONE/SSE/BIDI (case-insensitive) build a RunConfig with the right mode."""
    from google.adk.agents.run_config import StreamingMode

    assert build_run_config("NONE").streaming_mode == StreamingMode.NONE
    assert build_run_config("sse").streaming_mode == StreamingMode.SSE
    assert build_run_config("Bidi").streaming_mode == StreamingMode.BIDI


def test_build_run_config_max_llm_calls_forwarded() -> None:
    """A provided max_llm_calls is passed through; None leaves the ADK default (500)."""
    assert build_run_config("NONE", max_llm_calls=7).max_llm_calls == 7
    assert build_run_config("NONE", max_llm_calls=None).max_llm_calls == 500


def test_build_run_config_response_modalities() -> None:
    """response_modalities is passed through when provided."""
    rc = build_run_config("NONE", response_modalities=["TEXT"])
    assert rc.response_modalities == ["TEXT"]


def test_build_run_config_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="invalid streaming_mode"):
        build_run_config("TURBO")


def test_streaming_mode_names() -> None:
    names = streaming_mode_names()
    assert set(names) == {"NONE", "SSE", "BIDI"}


# --------------------------------------------------------------------------- #
# Plugins (P4c) — build_runner via App + import_project_plugins
# --------------------------------------------------------------------------- #
def _rec_plugin() -> object:
    """Build a BasePlugin that records the author of each event (offline proof)."""
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
    """PROOF: a plugin passed to build_runner wires Runner(app=App(plugins=[...])) and runs.

    We run a FakeLlm offline; the plugin records the events. Proves the Runner(plugins) wiring via
    the App (non-deprecated) path end to end, without an API key.
    """
    plugin = _rec_plugin()
    agent = _llm_agent("fa", FakeLlm(model="f", answer="plugged"))
    runner = build_runner("app", agent, _in_memory_config(), plugins=[plugin])

    # app_name is derived from App.name (App path).
    assert runner.app_name == "app"

    events = await collect_events(runner, user_id="u", session_id="s", new_message_text="hi")
    finals = [serialize_event(e) for e in events if e.is_final_response()]
    assert finals and finals[-1]["text"] == "plugged"
    # The plugin did see events (the on_event_callback hook fired).
    assert plugin.seen, "the plugin should have recorded at least one event"


def test_build_runner_no_plugins_unchanged() -> None:
    """Without plugins, build_runner keeps the Runner(app_name=, agent=) path (backward compat)."""
    agent = _llm_agent("fa", FakeLlm(model="f"))
    runner = build_runner("app", agent, _in_memory_config())
    assert runner.app_name == "app"
    # No plugin wired.
    assert not getattr(runner, "plugin_manager", None) or not runner.plugin_manager.plugins


def _write_plugins_py(root: Path, app_name: str, body: str) -> None:
    """Write ``<root>/<app_name>/plugins.py`` with the given body."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "plugins.py").write_text(body, encoding="utf-8")


def test_import_project_plugins_returns_instances(tmp_path: Path) -> None:
    """import_project_plugins returns the instances named in plugins.py (order preserved)."""
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
    with pytest.raises(PluginsImportError, match="not found"):
        import_project_plugins(str(tmp_path), "ghost", ["p"])


def test_import_project_plugins_missing_var_raises(tmp_path: Path) -> None:
    _write_plugins_py(
        tmp_path,
        "myapp",
        "from google.adk.plugins import BasePlugin\np1 = BasePlugin(name='one')\n",
    )
    with pytest.raises(PluginsImportError, match="does not define the plugin variable"):
        import_project_plugins(str(tmp_path), "myapp", ["missing"])


def test_import_project_plugins_broken_module_raises(tmp_path: Path) -> None:
    _write_plugins_py(tmp_path, "myapp", "raise RuntimeError('boom')\n")
    with pytest.raises(PluginsImportError, match="Failed to import"):
        import_project_plugins(str(tmp_path), "myapp", ["p"])
