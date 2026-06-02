"""`run` domain: RUNS an ADK agent via a ``Runner`` (P3a — execution core).

Unlike the P1 domains (which *write* ``agent.py``) and like the P2 domains (which call real ADK
services), this domain **imports the ``root_agent``** of an app, wires it into a ``Runner`` on the
configured session/memory/artifacts services (``runtime.json``), and collects the produced
``Event`` objects. All the reusable machinery lives in :mod:`adk_toolkit_mcp.run_core` (tested
offline via a ``FakeLlm`` — no key required).

A FastMCP sub-server mounted under ``namespace="run"`` → tools exposed as ``run_<name>``.
Functions with **BARE** names. ``agent`` is registered under the bare tool name ``agent``
(exposed as ``run_agent``); ``stream`` → ``run_stream``; ``live`` → ``run_live``;
``config_build`` → ``run_config_build``; ``inspect_events`` → ``run_inspect_events``.

Each tool returns the ``{ok, data, error}`` envelope; invalid inputs, a corrupt config, a failed
``root_agent`` import and a missing Live capability return ``err(...)`` (never an exception that
propagates, never a network hang).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import Context, FastMCP

from ..envelope import err, ok
from ..project_model import load_model
from ..run_core import (
    RootAgentImportError,
    build_run_config,
    build_runner,
    collect_events,
    import_root_agent,
    serialize_event,
    streaming_mode_names,
)
from ..runtime import RuntimeConfig, load_runtime_config
from ..workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints only
    from google.adk.agents import BaseAgent

run_server: FastMCP = FastMCP("run")


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _config_for(path: str, app_name: str) -> RuntimeConfig | dict[str, Any]:
    """Load the app's runtime config, or return an ``err(...)`` if corrupt.

    An app without a ``runtime.json`` receives the default config (``in_memory`` sessions) — so an
    agent can be run without having called ``sessions_service_set`` beforehand.
    """
    ws = _app_ws(path, app_name)
    try:
        return load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _prepare(path: str, app_name: str) -> tuple[BaseAgent, RuntimeConfig] | dict[str, Any]:
    """Load the config and import ``root_agent``; return ``(agent, config)`` or an ``err``.

    Centralizes the two failures converted into ``err``: a corrupt config (``ValueError``) and the
    ``root_agent`` import (``RootAgentImportError``: missing file, broken module, missing symbol).
    """
    config = _config_for(path, app_name)
    if isinstance(config, dict):
        return config
    try:
        root_agent = import_root_agent(path, app_name)
    except RootAgentImportError as exc:
        return err(str(exc))
    return root_agent, config


def _final_text(serialized: list[dict[str, Any]]) -> str | None:
    """Return the text of the LAST final event (the agent's response), or ``None``."""
    finals = [s for s in serialized if s["is_final"] and s["text"]]
    return finals[-1]["text"] if finals else None


def _resolve_max_llm_calls(path: str, app_name: str, caller_value: int | None) -> int | None:
    """Resolve the effective LLM call cap for a run.

    Precedence: an explicit caller value (``caller_value is not None``) **always wins**.
    Otherwise, we fall back to the value **persisted** by ``safety_settings(..., max_llm_calls=N)``:
    the project's ROOT agent ``AgentSpec.max_llm_calls`` (``model.root``), read from the sidecar
    ``.adk_toolkit/agents.json`` via :func:`load_model`. If nothing is persisted (or no root, or no
    sidecar), we return ``None`` → ADK default (500), as before.

    Best-effort and non-blocking: a corrupt sidecar (``ValueError``) does NOT fail the run (the
    ``run`` domain historically did not read ``agents.json``) — we simply fall back to ``None``. A
    corrupt runtime config, on the other hand, is still handled upstream by ``_config_for``.
    """
    if caller_value is not None:
        return caller_value
    try:
        model = load_model(_app_ws(path, app_name), app_name)
    except ValueError:
        return None
    if model.root is None:
        return None
    root_spec = model.get(model.root)
    return root_spec.max_llm_calls if root_spec is not None else None


def _model_supports_live(agent: BaseAgent) -> bool:
    """Indicate whether the agent's model supports the Live connection (``connect`` overridden).

    The base ``BaseLlm.connect`` raises ``NotImplementedError``; only a live-capable model (e.g.
    ``Gemini``) overrides it. We therefore compare the ``connect`` method of the resolved model's
    class against ``BaseLlm``'s. Any resolution error → ``False`` (cautious).
    """
    try:
        from google.adk.models import BaseLlm

        model = getattr(agent, "canonical_model", None)
        if model is None:
            return False
        return type(model).connect is not BaseLlm.connect
    except Exception:  # noqa: BLE001 - defensive detection: a failure = no Live
        return False


def _has_live_credentials() -> bool:
    """Indicate whether credentials enabling the Live API are present in the environment.

    AI Studio: ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``). Vertex: ``GOOGLE_GENAI_USE_VERTEXAI``
    truthy + ``GOOGLE_CLOUD_PROJECT``. No value is read/logged — only presence matters.
    """
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return True
    use_vertex = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    if use_vertex in {"1", "true", "yes"} and os.getenv("GOOGLE_CLOUD_PROJECT"):
        return True
    return False


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@run_server.tool(tags={"run"})
async def agent(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    message: str,
    max_llm_calls: int | None = None,
    streaming_mode: str = "NONE",
) -> dict[str, Any]:
    """Run the app's ``root_agent`` on ``message`` and return the events + final text.

    Imports ``root_agent`` (from ``<path>/<app_name>/agent.py``), wires it into a ``Runner`` on the
    configured services, creates the session if needed, runs the agent loop, then returns the list
    of **serialized** events and the text of the final response.

    ``streaming_mode`` ∈ {``NONE``, ``SSE``, ``BIDI``} (default ``NONE``: a single final
    ``LlmResponse`` per turn). ``max_llm_calls`` caps the number of LLM calls: an explicit value
    **wins**; if ``None``, we fall back to the cap **persisted** by
    ``safety_settings(..., max_llm_calls=N)`` (the sidecar root agent's ``max_llm_calls``); failing
    that, to the ADK default (500).
    """
    if not user_id.strip():
        return err("user_id is empty.")
    if not session_id.strip():
        return err("session_id is empty.")
    if not message.strip():
        return err("message is empty.")

    prepared = _prepare(path, app_name)
    if isinstance(prepared, dict):
        return prepared
    root_agent, config = prepared

    # Effective cap: explicit caller value, otherwise the persisted value (root spec).
    resolved_max_llm_calls = _resolve_max_llm_calls(path, app_name, max_llm_calls)

    try:
        run_config = build_run_config(
            streaming_mode=streaming_mode, max_llm_calls=resolved_max_llm_calls
        )
        runner = build_runner(app_name, root_agent, config)
        events = await collect_events(
            runner,
            user_id=user_id,
            session_id=session_id,
            new_message_text=message,
            run_config=run_config,
        )
    except ValueError as exc:
        # invalid streaming_mode OR invalid backend (missing required field / missing gcp extra).
        return err(str(exc))

    serialized = [serialize_event(e) for e in events]
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "streaming_mode": streaming_mode.strip().upper(),
            "event_count": len(serialized),
            "events": serialized,
            "final_text": _final_text(serialized),
        }
    )


@run_server.tool(tags={"run"})
async def stream(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    message: str,
    max_llm_calls: int | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Like ``agent`` but in SSE mode, reporting per-event progress via ``ctx``.

    Forces ``streaming_mode="SSE"``. For each produced event, reports progress to the MCP client
    (``ctx.report_progress`` + ``ctx.info``) — useful for real-time tracking. Returns the same data
    as ``agent`` (serialized events + final text). ``max_llm_calls`` follows the same precedence as
    ``agent`` (explicit > persisted root cap > ADK default 500).
    """
    if not user_id.strip():
        return err("user_id is empty.")
    if not session_id.strip():
        return err("session_id is empty.")
    if not message.strip():
        return err("message is empty.")

    prepared = _prepare(path, app_name)
    if isinstance(prepared, dict):
        return prepared
    root_agent, config = prepared

    async def _progress(index: int, event: dict[str, Any]) -> None:
        """Report an event to the client (silent no-op if ``ctx`` is absent)."""
        if ctx is None:
            return
        label = event.get("author") or "event"
        await ctx.report_progress(index, message=f"event {index} ({label})")
        await ctx.info(f"[run.stream] event {index}: author={label} final={event['is_final']}")

    # Effective cap: explicit caller value, otherwise the persisted value (root spec).
    resolved_max_llm_calls = _resolve_max_llm_calls(path, app_name, max_llm_calls)

    try:
        run_config = build_run_config(streaming_mode="SSE", max_llm_calls=resolved_max_llm_calls)
        runner = build_runner(app_name, root_agent, config)
        events = await collect_events(
            runner,
            user_id=user_id,
            session_id=session_id,
            new_message_text=message,
            run_config=run_config,
            progress=_progress,
        )
    except ValueError as exc:
        return err(str(exc))

    serialized = [serialize_event(e) for e in events]
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "streaming_mode": "SSE",
            "event_count": len(serialized),
            "events": serialized,
            "final_text": _final_text(serialized),
        }
    )


@run_server.tool(tags={"run"})
async def live(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    message: str,
    max_llm_calls: int | None = None,
) -> dict[str, Any]:
    """[EXPERIMENTAL] Live/BIDI execution (Gemini Live API) — requires a key + live-capable model.

    The Live path uses ``BaseLlm.connect`` (websocket to the Gemini Live API), NOT
    ``generate_content_async``: it requires a real key (``GOOGLE_API_KEY`` or Vertex creds) AND a
    live-capable model, and CANNOT run in CI. This tool performs the faithful wiring (importing the
    ``root_agent``, a BIDI ``RunConfig``) but **detects the missing capability** and returns an
    actionable ``err`` BEFORE any connection — it never blocks.

    With the prerequisites present, it would open a ``LiveRequestQueue``, push ``message`` onto it,
    and stream the events from ``runner.run_live(...)``. ``max_llm_calls`` follows the same
    precedence as ``agent`` (explicit > persisted root cap > ADK default 500).
    """
    if not user_id.strip() or not session_id.strip():
        return err("user_id and session_id are required.")
    if not message.strip():
        return err("message is empty.")

    prepared = _prepare(path, app_name)
    if isinstance(prepared, dict):
        return prepared
    root_agent, config = prepared

    # Capability detection BEFORE any network connection (otherwise the call would block/fail).
    if not _has_live_credentials():
        return err(
            "run_live requires the Gemini Live API: set GOOGLE_API_KEY (AI Studio) or "
            "GOOGLE_GENAI_USE_VERTEXAI=TRUE + GOOGLE_CLOUD_PROJECT (Vertex). "
            "Experimental tool — not runnable without a key/websocket (e.g. in CI)."
        )
    if not _model_supports_live(root_agent):
        model_name = getattr(getattr(root_agent, "canonical_model", None), "model", "?")
        return err(
            f"The agent's model ({model_name!r}) does not support the Live connection "
            "(BaseLlm.connect not overridden). Use a live-capable Gemini model."
        )

    # Effective cap: explicit caller value, otherwise the persisted value (root spec).
    resolved_max_llm_calls = _resolve_max_llm_calls(path, app_name, max_llm_calls)

    # Prerequisites present: faithful wiring of the Live path (not covered in CI).
    try:  # pragma: no cover - requires a real Live API + websocket
        from google.adk.agents.live_request_queue import LiveRequestQueue
        from google.genai import types

        run_config = build_run_config(streaming_mode="BIDI", max_llm_calls=resolved_max_llm_calls)
        runner = build_runner(app_name, root_agent, config)
        session_service = runner.session_service
        session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        if session is None:
            session = await session_service.create_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        queue = LiveRequestQueue()
        queue.send_content(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        queue.close()
        events = [
            serialize_event(event)
            async for event in runner.run_live(
                user_id=user_id,
                session_id=session_id,
                live_request_queue=queue,
                run_config=run_config,
            )
        ]
        return ok(
            {
                "app_name": app_name,
                "session_id": session_id,
                "streaming_mode": "BIDI",
                "event_count": len(events),
                "events": events,
                "final_text": _final_text(events),
            }
        )
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - Live path not testable in CI
        # Any Live-path failure (network, model, websocket) → actionable err, never a raise.
        return err(f"Live execution failed: {exc}")


@run_server.tool(tags={"run"})
def config_build(
    streaming_mode: str = "NONE",
    max_llm_calls: int | None = None,
    response_modalities: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and describe a ``RunConfig`` (without running an agent).

    Returns a descriptor ``{streaming_mode, max_llm_calls, response_modalities}`` and the list of
    valid modes (``streaming_options``). An unknown ``streaming_mode`` returns ``err``.
    """
    try:
        run_config = build_run_config(
            streaming_mode=streaming_mode,
            max_llm_calls=max_llm_calls,
            response_modalities=response_modalities,
        )
    except ValueError as exc:
        return err(str(exc))

    return ok(
        {
            "streaming_mode": run_config.streaming_mode.name,
            "max_llm_calls": run_config.max_llm_calls,
            "response_modalities": run_config.response_modalities,
            "streaming_options": streaming_mode_names(),
        }
    )


@run_server.tool(tags={"run"})
def inspect_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of serialized events (as returned by ``run_agent``).

    A PURE tool (no I/O): counts the function_calls, lists the tools used, the agent transfers,
    the state_delta keys, and extracts the final text. ``events`` must be a list of dicts in the
    :func:`serialize_event` format.
    """
    if not isinstance(events, list):
        return err("events must be a list of serialized event dicts.")

    tool_names: list[str] = []
    function_call_count = 0
    function_response_count = 0
    transfers: list[str] = []
    state_delta_keys: set[str] = set()
    final_texts: list[str] = []

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            return err(f"events[{index}] is not a serialized event dict.")
        for call in event.get("function_calls") or []:
            function_call_count += 1
            name = call.get("name") if isinstance(call, dict) else None
            if name:
                tool_names.append(name)
        function_response_count += len(event.get("function_responses") or [])
        transfer = event.get("transfer_to_agent")
        if transfer:
            transfers.append(transfer)
        for key in event.get("state_delta") or {}:
            state_delta_keys.add(key)
        if event.get("is_final") and event.get("text"):
            final_texts.append(event["text"])

    # Unique tools, preserving first-appearance order.
    unique_tools = list(dict.fromkeys(tool_names))
    return ok(
        {
            "event_count": len(events),
            "function_call_count": function_call_count,
            "function_response_count": function_response_count,
            "tool_names": unique_tools,
            "transfers": transfers,
            "state_delta_keys": sorted(state_delta_keys),
            "final_text": final_texts[-1] if final_texts else None,
        }
    )
