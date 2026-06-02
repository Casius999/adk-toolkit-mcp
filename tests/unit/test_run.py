"""Unit tests for the ``run`` domain (P3a — ADK agent execution).

The tools are **async** (``asyncio_mode=auto``). We call the bare functions directly and, for the
read-through, via an in-memory ``fastmcp.Client``.

FUNCTIONAL PROOF (without an API key): we scaffold an app whose ``agent.py`` imports a ``FakeLlm``
from the fixture (via ``sys.path``) and builds an ``LlmAgent``. ``run_agent`` then runs a real
agent loop offline and returns the canned final text — proving the mounted tool runs an agent end
to end without network.

Complementary coverage:
- validations (empty user_id/session_id/message) and clean errors (missing agent.py →
  RootAgentImportError converted to err; corrupt config → err).
- ``run_config_build``: valid + invalid modes.
- ``run_inspect_events`` (PURE): summary of synthetic + invalid events.
- ``run_stream``: the progress callback is invoked per event (proven via ``collect_events`` with a
  callback, and via a ``fastmcp.Client`` that captures ctx.report_progress).
- ``run_live``: returns an actionable err when the Live capability/key is absent (no blocking).
- ``fastmcp.Client`` read-through for ``run_agent`` against a FakeLlm agent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import run as R
from adk_toolkit_mcp.runtime import reset_service_cache
from adk_toolkit_mcp.server import build_server

#: Test fixtures folder (contains ``fake_llm.py``) — injected into the generated agent.py.
_FIXTURE_DIR = str(Path(__file__).parent)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isolate the tests: clear the singleton service cache before/after each."""
    reset_service_cache()
    yield
    reset_service_cache()


def _scaffold_fake_agent(
    root: Path, app_name: str = "myapp", answer: str = "Hello offline!"
) -> str:
    """Write an app whose ``agent.py`` builds an LlmAgent + FakeLlm (offline). Returns path."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "import sys\n"
        f"sys.path.insert(0, r'{_FIXTURE_DIR}')\n"
        "from fake_llm import FakeLlm\n"
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(\n"
        f"    name='{app_name}', model=FakeLlm(model='fake', answer={answer!r})\n"
        ")\n"
    )
    (app_dir / "agent.py").write_text(body, encoding="utf-8")
    return str(root)


def _scaffold_tool_agent(root: Path, app_name: str = "calc") -> str:
    """Write an app whose ``agent.py`` builds a ScriptedLlm agent + tool (offline)."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "import sys\n"
        f"sys.path.insert(0, r'{_FIXTURE_DIR}')\n"
        "from fake_llm import ScriptedLlm, add_numbers\n"
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(name='{app_name}', "
        "model=ScriptedLlm(model='scripted', tool_name='add_numbers', "
        "tool_args={'a': 2, 'b': 3}, final_text='The sum is 5.'), tools=[add_numbers])\n"
    )
    (app_dir / "agent.py").write_text(body, encoding="utf-8")
    return str(root)


def _scaffold_fake_workflow_agent(root: Path, app_name: str = "ed_run") -> str:
    """Write an app whose ``root_agent`` is a **Workflow** of two FakeLlm ``LlmAgent`` nodes.

    Proves the ``run`` domain wires a workflow (``BaseNode``) root via ``node=`` (NOT ``agent=``):
    the run executes offline (no key), emitting the two node outputs in graph order.
    """
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "import sys\n"
        f"sys.path.insert(0, r'{_FIXTURE_DIR}')\n"
        "from fake_llm import FakeLlm\n"
        "from google.adk.agents import LlmAgent\n"
        "from google.adk.workflow import START, Workflow\n"
        "writer = LlmAgent("
        "name='writer', model=FakeLlm(model='fake', answer='draft text'), instruction='Write.')\n"
        "reviewer = LlmAgent("
        "name='reviewer', model=FakeLlm(model='fake', answer='reviewed text'), "
        "instruction='Review.')\n"
        "root_agent = Workflow("
        "name='editorial', edges=[(START, writer), (writer, reviewer)])\n"
    )
    (app_dir / "agent.py").write_text(body, encoding="utf-8")
    return str(root)


def _persist_max_llm_calls(path: str, app_name: str, value: int) -> None:
    """Persist ``max_llm_calls=value`` on the ROOT agent via the REAL ``safety_settings`` tool.

    We first create the root agent in the sidecar (``agents_create_llm`` + ``agents_set_root``),
    then call ``safety_settings(max_llm_calls=value)`` — exactly the user path. ``safety_settings``
    regenerates ``agent.py`` (Gemini model), so the caller then REWRITES it with a FakeLlm to stay
    runnable offline (the ``agents.json`` sidecar — from which the persisted value is re-read — is
    not affected by this ``agent.py`` rewrite).
    """
    from adk_toolkit_mcp.domains import agents as AGENTS
    from adk_toolkit_mcp.domains import safety as SAFETY

    assert AGENTS.create_llm(path=path, app_name=app_name, name=app_name)["ok"]
    assert AGENTS.set_root(path=path, app_name=app_name, name=app_name)["ok"]
    res = SAFETY.safety_settings(
        path=path, app_name=app_name, agent_name=app_name, max_llm_calls=value
    )
    assert res["ok"], res
    assert res["data"]["max_llm_calls"] == value


# --------------------------------------------------------------------------- #
# FUNCTIONAL — run_agent runs a FakeLlm agent offline
# --------------------------------------------------------------------------- #
async def test_run_agent_functional_offline(tmp_path: Path) -> None:
    """run_agent (mounted tool) runs a FakeLlm agent loaded from agent.py → final text."""
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="42 is the answer")
    result = await R.agent(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s1",
        message="what is the answer?",
    )
    assert result["ok"] is True, result
    assert result["data"]["final_text"] == "42 is the answer"
    assert result["data"]["event_count"] >= 1
    # The events are serialized (expected keys).
    ev = result["data"]["events"][0]
    assert {"author", "text", "is_final", "function_calls"} <= set(ev)


async def test_run_agent_functional_tool_loop_offline(tmp_path: Path) -> None:
    """run_agent proves a complete tool-call loop offline: call → response → final."""
    path = _scaffold_tool_agent(tmp_path, "calc")
    result = await R.agent(
        path=path, app_name="calc", user_id="u1", session_id="s1", message="2+3?"
    )
    assert result["ok"] is True, result
    events = result["data"]["events"]
    assert any(e["function_calls"] for e in events), events
    assert any(e["function_responses"] for e in events), events
    assert result["data"]["final_text"] == "The sum is 5."


async def test_run_agent_reuses_session_across_calls(tmp_path: Path) -> None:
    """Two run_agent on the same session_id: the second sees the first's events."""
    path = _scaffold_fake_agent(tmp_path, "myapp")
    first = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert first["ok"] is True
    first_count = first["data"]["event_count"]

    # Check via the sessions domain that the session accumulates events.
    from adk_toolkit_mcp.domains import sessions as S

    got = await S.get(path=path, app_name="myapp", user_id="u1", session_id="s1")
    assert got["ok"] is True
    assert got["data"]["event_count"] >= first_count


# --------------------------------------------------------------------------- #
# FUNCTIONAL — run_agent works for a WORKFLOW (BaseNode) root, offline
# --------------------------------------------------------------------------- #
async def test_run_agent_runs_workflow_root_offline(tmp_path: Path) -> None:
    """run_agent drives a Workflow-rooted app end to end offline (node= path).

    The root is a ``Workflow`` (a ``BaseNode``, NOT a ``BaseAgent``) of two FakeLlm ``LlmAgent``
    nodes. ``build_runner`` must wire it via ``node=`` (not ``agent=``); the two nodes run in
    graph order and real events come back — proving workflow roots are runnable via the run domain.
    """
    path = _scaffold_fake_workflow_agent(tmp_path, "ed_run")
    result = await R.agent(
        path=path, app_name="ed_run", user_id="u1", session_id="s1", message="draft please"
    )
    assert result["ok"] is True, result
    events = result["data"]["events"]
    texts = [(e["author"], e["text"]) for e in events if e["text"]]
    assert texts == [("writer", "draft text"), ("reviewer", "reviewed text")]
    assert result["data"]["final_text"] == "reviewed text"


async def test_run_agent_runs_function_node_workflow_root_via_real_tools(tmp_path: Path) -> None:
    """A workflow built via the REAL workflow tools (pure function nodes) runs via run_agent.

    End-to-end through the toolkit: ``workflow_create``/``add_node``/``add_edge``/``set_entry``/
    ``set_root`` produce a ``root_kind="workflow"`` app whose ``agent.py`` is a function-node
    ``Workflow``; ``run_agent`` then executes it offline (no LLM at all) and returns real events.
    """
    from adk_toolkit_mcp.domains import workflow as WF

    p = str(tmp_path)
    WF.create(p, "fn_run", "pipe")
    WF.add_node(p, "fn_run", "pipe", "first", "function", docstring="First.", body='return "go"')
    WF.add_node(
        p, "fn_run", "pipe", "second", "function", docstring="Second.", body='return {"d": 1}'
    )
    WF.set_entry(p, "fn_run", "pipe", "first")
    WF.add_edge(p, "fn_run", "pipe", "first", "second")
    rooted = WF.set_root(p, "fn_run", "pipe")
    assert rooted["ok"] is True, rooted["error"]
    assert rooted["data"]["root_kind"] == "workflow"

    result = await R.agent(
        path=p, app_name="fn_run", user_id="u1", session_id="s1", message="hello"
    )
    assert result["ok"] is True, result
    # Two function nodes executed → real events authored by the workflow.
    assert result["data"]["event_count"] >= 2
    assert all(e["author"] == "pipe" for e in result["data"]["events"])


# --------------------------------------------------------------------------- #
# FUNCTIONAL — persisted max_llm_calls (safety_settings) is honored by run_*
# --------------------------------------------------------------------------- #
async def test_run_agent_uses_persisted_max_llm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_agent WITHOUT an explicit max_llm_calls applies the root agent's PERSISTED cap (=7).

    We persist ``safety_settings(..., max_llm_calls=7)`` on the root (real user path), rewrite
    ``agent.py`` to a FakeLlm (offline), then call ``run_agent`` WITHOUT passing ``max_llm_calls``.
    We capture the ``RunConfig`` actually built via a seam: we monkeypatch ``R.build_run_config``
    to record the received ``max_llm_calls`` argument and the returned ``RunConfig`` (delegating to
    the real factory so the run succeeds).

    This test FAILS before the fix (run_* ignored the persisted value → build_run_config received
    ``None`` instead of 7).
    """
    from adk_toolkit_mcp import run_core

    path = _scaffold_fake_agent(tmp_path, "myapp", answer="capped")
    _persist_max_llm_calls(path, "myapp", 7)
    # safety_settings regenerated agent.py (Gemini): we put it back to FakeLlm (offline run).
    _scaffold_fake_agent(tmp_path, "myapp", answer="capped")

    seen: dict[str, object] = {}
    real_build = run_core.build_run_config

    def _spy_build_run_config(**kwargs: object) -> object:
        seen["max_llm_calls"] = kwargs.get("max_llm_calls")
        cfg = real_build(**kwargs)  # type: ignore[arg-type]
        seen["run_config"] = cfg
        return cfg

    monkeypatch.setattr(R, "build_run_config", _spy_build_run_config)

    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is True, result
    # The persisted cap (7) was resolved and passed to build_run_config…
    assert seen["max_llm_calls"] == 7
    # …and the RunConfig actually used by the runner indeed carries max_llm_calls == 7.
    assert seen["run_config"].max_llm_calls == 7  # type: ignore[attr-defined]


async def test_run_agent_explicit_max_llm_calls_overrides_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit caller value WINS over the persisted one (7 persisted, 3 explicit → 3)."""
    from adk_toolkit_mcp import run_core

    path = _scaffold_fake_agent(tmp_path, "myapp", answer="capped")
    _persist_max_llm_calls(path, "myapp", 7)
    _scaffold_fake_agent(tmp_path, "myapp", answer="capped")

    seen: dict[str, object] = {}
    real_build = run_core.build_run_config

    def _spy_build_run_config(**kwargs: object) -> object:
        seen["max_llm_calls"] = kwargs.get("max_llm_calls")
        return real_build(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(R, "build_run_config", _spy_build_run_config)

    result = await R.agent(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s1",
        message="hi",
        max_llm_calls=3,
    )
    assert result["ok"] is True, result
    # The explicit (3) overrides the persisted (7).
    assert seen["max_llm_calls"] == 3


async def test_run_agent_without_sidecar_uses_adk_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a sidecar or explicit value, max_llm_calls stays None → ADK default (no regression).

    Guarantees that the persisted-value enrichment is best-effort: an app scaffolded without a
    sidecar ``agents.json`` (historical case for these tests) keeps the previous behavior
    (``None``).
    """
    from adk_toolkit_mcp import run_core

    path = _scaffold_fake_agent(tmp_path, "myapp", answer="default")
    seen: dict[str, object] = {}
    real_build = run_core.build_run_config

    def _spy_build_run_config(**kwargs: object) -> object:
        seen["max_llm_calls"] = kwargs.get("max_llm_calls")
        return real_build(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(R, "build_run_config", _spy_build_run_config)

    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is True, result
    assert seen["max_llm_calls"] is None


# --------------------------------------------------------------------------- #
# Validation + error paths
# --------------------------------------------------------------------------- #
async def test_run_agent_rejects_empty_message(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="  ")
    assert result["ok"] is False
    assert "message" in result["error"]


async def test_run_agent_rejects_empty_user_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(path=path, app_name="myapp", user_id="  ", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "user_id" in result["error"]


async def test_run_agent_rejects_empty_session_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id=" ", message="hi")
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_run_agent_missing_agent_py_returns_err(tmp_path: Path) -> None:
    """Pas d'agent.py → RootAgentImportError convertie en err actionnable (pas d'exception)."""
    result = await R.agent(
        path=str(tmp_path), app_name="ghost", user_id="u1", session_id="s1", message="hi"
    )
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


async def test_run_agent_broken_agent_py_returns_err(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir(parents=True)
    (app_dir / "agent.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    result = await R.agent(
        path=str(tmp_path), app_name="myapp", user_id="u1", session_id="s1", message="hi"
    )
    assert result["ok"] is False
    assert result["error"]


async def test_run_agent_invalid_streaming_mode_returns_err(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s1",
        message="hi",
        streaming_mode="TURBO",
    )
    assert result["ok"] is False
    assert "streaming_mode" in result["error"]


async def test_run_agent_corrupt_config_returns_err(tmp_path: Path) -> None:
    """corrupt runtime.json → clean err (the config loads before the agent import)."""
    path = _scaffold_fake_agent(tmp_path, "myapp")
    cfg_dir = tmp_path / "myapp" / ".adk_toolkit"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert result["error"]


# --------------------------------------------------------------------------- #
# run_stream — per-event progress
# --------------------------------------------------------------------------- #
async def test_run_stream_offline_no_ctx(tmp_path: Path) -> None:
    """run_stream works without ctx (no-op progress) and returns the final text."""
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="streamed!")
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id="s1", message="go", ctx=None
    )
    assert result["ok"] is True
    assert result["data"]["streaming_mode"] == "SSE"
    assert result["data"]["final_text"] == "streamed!"


async def test_run_stream_rejects_empty_message(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id="s1", message="", ctx=None
    )
    assert result["ok"] is False


async def test_run_stream_rejects_empty_user_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.stream(
        path=path, app_name="myapp", user_id=" ", session_id="s1", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert "user_id" in result["error"]


async def test_run_stream_rejects_empty_session_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id=" ", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_run_stream_missing_agent_returns_err(tmp_path: Path) -> None:
    """run_stream on an app without agent.py → err (import failed), no exception."""
    result = await R.stream(
        path=str(tmp_path), app_name="ghost", user_id="u1", session_id="s1", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert result["error"]


async def test_run_stream_invalid_backend_returns_err(tmp_path: Path) -> None:
    """Invalid backend (database without db_url, hand-edited) → clean err via run_stream.

    Covers the ValueError branch of run_stream (non-instantiable backend). run_stream forces
    streaming_mode='SSE', so the only ValueError here comes from the backend.
    """
    path = _scaffold_fake_agent(tmp_path, "myapp")
    cfg_dir = tmp_path / "myapp" / ".adk_toolkit"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "runtime.json").write_text(
        '{"session": {"kind": "database", "db_url": null}}', encoding="utf-8"
    )
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert "db_url" in result["error"]


async def test_run_agent_invalid_backend_returns_err(tmp_path: Path) -> None:
    """run_agent on a non-instantiable backend (database without db_url) → clean err."""
    path = _scaffold_fake_agent(tmp_path, "myapp")
    cfg_dir = tmp_path / "myapp" / ".adk_toolkit"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "runtime.json").write_text(
        '{"session": {"kind": "database", "db_url": null}}', encoding="utf-8"
    )
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "db_url" in result["error"]


async def test_run_stream_progress_via_client(tmp_path: Path) -> None:
    """Via a fastmcp.Client, run_stream reports progress: the handler receives calls.

    We capture ``progress`` on the client side (FastMCP injects the Context and relays
    report_progress). At least one event → at least one progress call.
    """
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="progress proof")
    mcp = build_server()

    progress_calls: list[tuple[float, float | None, str | None]] = []

    async def _handler(progress: float, total: float | None, message: str | None) -> None:
        progress_calls.append((progress, total, message))

    async with Client(mcp, progress_handler=_handler) as client:
        res = await client.call_tool(
            "run_stream",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": "s1",
                "message": "hi",
            },
        )
        assert res.data["ok"] is True
        assert res.data["data"]["final_text"] == "progress proof"

    # The progress handler was invoked at least once (one final event at minimum).
    assert progress_calls, "report_progress should have been relayed to the client"


# --------------------------------------------------------------------------- #
# run_live — actionable degradation without key/capability (no blocking)
# --------------------------------------------------------------------------- #
async def test_run_live_without_credentials_returns_actionable_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a Live key, run_live returns an immediate actionable err (never a hang)."""
    # Neutralize any Live creds possibly present in the environment.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "GOOGLE_API_KEY" in result["error"]


async def test_run_live_with_key_but_non_live_model_returns_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a key but a non-live-capable model (FakeLlm), run_live returns a clear err.

    Proves the second guard: even with creds, a FakeLlm (connect not overridden) cannot stream in
    Live → actionable err, still without network blocking.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-not-used")
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "Live" in result["error"]


async def test_run_live_vertex_credentials_recognized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vertex creds (USE_VERTEXAI=TRUE + PROJECT) pass the 1st guard; failure on the model.

    Covers the Vertex branch of _has_live_credentials: without an AI Studio key but with Vertex
    configured, creds detection succeeds → we hit the model capability guard
    (FakeLlm not live-capable) → err ``Live``, still without a hang.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")

    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    # 1st guard passed (Vertex creds recognized) → we hit the model guard (Live).
    assert "Live" in result["error"]


def test_model_supports_live_handles_missing_canonical_model() -> None:
    """_model_supports_live returns False (without raising) if the agent has no canonical_model."""

    class _Bare:
        pass

    assert R._model_supports_live(_Bare()) is False  # type: ignore[arg-type]


def test_model_supports_live_swallows_exceptions() -> None:
    """An error while accessing the model → False (defensive detection, never a raise)."""

    class _Exploding:
        @property
        def canonical_model(self) -> object:
            raise RuntimeError("boom on access")

    assert R._model_supports_live(_Exploding()) is False  # type: ignore[arg-type]


async def test_run_live_rejects_empty_message(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="  ")
    assert result["ok"] is False
    assert "message" in result["error"]


async def test_run_live_rejects_empty_session_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="  ", message="hi")
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_run_live_missing_agent_returns_err(tmp_path: Path) -> None:
    result = await R.live(
        path=str(tmp_path), app_name="ghost", user_id="u1", session_id="s1", message="hi"
    )
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# run_config_build (validation pure)
# --------------------------------------------------------------------------- #
def test_config_build_valid() -> None:
    result = R.config_build(streaming_mode="SSE", max_llm_calls=10)
    assert result["ok"] is True
    assert result["data"]["streaming_mode"] == "SSE"
    assert result["data"]["max_llm_calls"] == 10
    assert set(result["data"]["streaming_options"]) == {"NONE", "SSE", "BIDI"}


def test_config_build_default_max_llm_calls() -> None:
    """max_llm_calls None → ADK default (500) reflected in the descriptor."""
    result = R.config_build(streaming_mode="NONE")
    assert result["ok"] is True
    assert result["data"]["max_llm_calls"] == 500


def test_config_build_invalid_mode() -> None:
    result = R.config_build(streaming_mode="WARP")
    assert result["ok"] is False
    assert "streaming_mode" in result["error"]


def test_config_build_response_modalities() -> None:
    result = R.config_build(streaming_mode="NONE", response_modalities=["TEXT"])
    assert result["ok"] is True
    assert result["data"]["response_modalities"] == ["TEXT"]


# --------------------------------------------------------------------------- #
# run_inspect_events (PUR)
# --------------------------------------------------------------------------- #
def _synthetic_events() -> list[dict]:
    """List of synthetic serialized events to test the summary."""
    return [
        {
            "author": "planner",
            "text": None,
            "function_calls": [{"name": "search", "args": {"q": "x"}}],
            "function_responses": [],
            "state_delta": {"app:hits": 1},
            "transfer_to_agent": "worker",
            "is_final": False,
            "partial": None,
        },
        {
            "author": "worker",
            "text": None,
            "function_calls": [{"name": "fetch", "args": {}}, {"name": "search", "args": {}}],
            "function_responses": [{"name": "search", "response": {"r": 1}}],
            "state_delta": {"user:seen": True},
            "transfer_to_agent": None,
            "is_final": False,
            "partial": None,
        },
        {
            "author": "worker",
            "text": "Done.",
            "function_calls": [],
            "function_responses": [],
            "state_delta": {},
            "transfer_to_agent": None,
            "is_final": True,
            "partial": None,
        },
    ]


def test_inspect_events_summary() -> None:
    result = R.inspect_events(_synthetic_events())
    assert result["ok"] is True
    data = result["data"]
    assert data["event_count"] == 3
    assert data["function_call_count"] == 3
    assert data["function_response_count"] == 1
    # Unique tools, first-appearance order preserved.
    assert data["tool_names"] == ["search", "fetch"]
    assert data["transfers"] == ["worker"]
    assert data["state_delta_keys"] == ["app:hits", "user:seen"]
    assert data["final_text"] == "Done."


def test_inspect_events_empty_list() -> None:
    result = R.inspect_events([])
    assert result["ok"] is True
    assert result["data"]["event_count"] == 0
    assert result["data"]["final_text"] is None
    assert result["data"]["tool_names"] == []


def test_inspect_events_rejects_non_list() -> None:
    result = R.inspect_events({"not": "a list"})  # type: ignore[arg-type]
    assert result["ok"] is False


def test_inspect_events_rejects_non_dict_item() -> None:
    result = R.inspect_events(["not a dict"])  # type: ignore[list-item]
    assert result["ok"] is False


async def test_inspect_events_consumes_run_agent_output(tmp_path: Path) -> None:
    """End-to-end: run_agent's events output is summarizable by run_inspect_events."""
    path = _scaffold_tool_agent(tmp_path, "calc")
    run = await R.agent(path=path, app_name="calc", user_id="u1", session_id="s1", message="2+3?")
    assert run["ok"] is True
    summary = R.inspect_events(run["data"]["events"])
    assert summary["ok"] is True
    assert "add_numbers" in summary["data"]["tool_names"]
    assert summary["data"]["final_text"] == "The sum is 5."


# --------------------------------------------------------------------------- #
# In-memory fastmcp.Client read-through (exposed names + double-prefix guard)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_run_agent(tmp_path: Path) -> None:
    """The tools are exposed as run_<bare> (no double prefix) and run_agent runs."""
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="client says hi")
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {"run_agent", "run_stream", "run_live", "run_config_build", "run_inspect_events"}
        assert expected <= names
        assert not any(n.startswith("run_run_") for n in names)

        res = await client.call_tool(
            "run_agent",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": "s1",
                "message": "hi",
            },
        )
        assert res.data["ok"] is True
        assert res.data["data"]["final_text"] == "client says hi"


async def test_client_run_config_build(tmp_path: Path) -> None:
    """run_config_build accessible via the client (pure validation)."""
    mcp = build_server()
    async with Client(mcp) as client:
        res = await client.call_tool("run_config_build", {"streaming_mode": "SSE"})
        assert res.data["ok"] is True
        assert res.data["data"]["streaming_mode"] == "SSE"
