"""ADK agent execution core (P3a) — offline-testable helpers.

This module factors out all the machinery for running an ADK agent so it can be **proven
offline with a FakeLlm** (no API key required). The ``run`` domain is just a thin layer of MCP
tools on top of these helpers.

Contents (cf. ``docs/adk-api-notes/runtime-run.md`` for the confirmed ADK signatures):

- :func:`build_runner` — wires a ``google.adk.runners.Runner`` onto the session/memory/artifacts
  services from :mod:`adk_toolkit_mcp.runtime` (same singleton factories).
- :func:`collect_events` — ensures the session exists (creates it as needed), runs
  ``run_async`` and collects the ``Event`` objects; an optional ``progress`` callback is *awaited*
  per event (SSE support).
- :func:`serialize_event` — flattens an ``Event`` into a simple dict ``{author, text,
  function_calls, function_responses, state_delta, transfer_to_agent, is_final, partial}``.
- :func:`import_root_agent` — imports ``<path>/<app_name>/agent.py`` and returns ``root_agent``
  via ``importlib`` with a UNIQUE module name on each call (no stale ``sys.modules`` cache: an
  edit to ``agent.py`` is picked up). Raises :class:`RootAgentImportError`.
- :func:`build_run_config` — validates ``streaming_mode`` against the real ``StreamingMode`` enum
  and builds a ``RunConfig``.

No ADK import at module load (everything is lazy), to stay consistent with the rest of the
toolkit and keep the tests fast.
"""

from __future__ import annotations

import importlib.util
import itertools
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .runtime import (
    RuntimeConfig,
    get_artifact_service,
    get_memory_service,
    get_session_service,
)

if TYPE_CHECKING:  # pragma: no cover - hints only, real imports are lazy
    from google.adk.agents import BaseAgent, RunConfig
    from google.adk.events import Event
    from google.adk.runners import Runner
    from google.adk.workflow import BaseNode

#: Monotonic counter guaranteeing a unique module name per ``import_root_agent``.
_IMPORT_COUNTER = itertools.count()

#: Monotonic counter for a unique module name per ``import_project_plugins`` (same reason as
#: for ``import_root_agent``: no stale ``sys.modules`` cache after an edit).
_PLUGINS_IMPORT_COUNTER = itertools.count()

#: Progress callback: receives ``(index_1_based, serialized_event)`` and is awaited.
ProgressCallback = Callable[[int, dict[str, Any]], Awaitable[None]]


class RootAgentImportError(Exception):
    """Failure importing ``root_agent`` (missing file, execution error, missing attribute).

    The ``run`` domain converts this exception into ``err(...)`` with an actionable message.
    """


class PluginsImportError(Exception):
    """Failure importing the project plugins (``plugins.py`` missing/broken, missing variable).

    The domains convert this exception into ``err(...)`` with an actionable message.
    """


# --------------------------------------------------------------------------- #
# Import root_agent (unique module name → no stale cache)
# --------------------------------------------------------------------------- #
def import_root_agent(path: str, app_name: str) -> BaseAgent | BaseNode:
    """Import ``<path>/<app_name>/agent.py`` and return its ``root_agent``.

    Uses a module name **unique on each call** (suffix via a monotonic counter) so that an edit
    to ``agent.py`` between two calls is picked up (never served from a stale ``sys.modules``).
    The module is intentionally NOT inserted into ``sys.modules``.

    The returned ``root_agent`` is a ``BaseAgent`` for an agent-rooted project, OR a ``BaseNode``
    (e.g. a ``Workflow`` graph root) for a workflow-rooted one — the ADK ``AgentLoader`` accepts
    both as a module-level ``root_agent`` (cf. ``docs/adk-api-notes/workflow.md``).
    :func:`build_runner` dispatches on the kind (``agent=`` vs ``node=``).

    Raises :class:`RootAgentImportError` if the file is missing, if its execution fails, or if
    ``root_agent`` is not defined in it.
    """
    agent_file = Path(path) / app_name / "agent.py"
    if not agent_file.is_file():
        raise RootAgentImportError(
            f"agent.py not found: {agent_file}. Scaffold the app first (project_create)."
        )

    module_name = f"_adk_toolkit_root_agent_{app_name}_{next(_IMPORT_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    if spec is None:  # pragma: no cover - degenerate importlib case
        raise RootAgentImportError(f"Unable to prepare the import of {agent_file}.")

    module = importlib.util.module_from_spec(spec)
    # We read the source and COMPILE/EXECUTE it directly rather than via
    # ``spec.loader.exec_module``: the ``SourceFileLoader`` caches bytecode by (path, mtime), and
    # on Windows two writes within the same mtime tick return a STALE version — an edit to
    # ``agent.py`` would then not be picked up. Reading+compiling on each call guarantees
    # freshness (in addition to the unique module name).
    try:
        source = agent_file.read_text(encoding="utf-8")
        code = compile(source, str(agent_file), "exec")
        exec(code, module.__dict__)  # noqa: S102 - intentional execution of user code (agent.py)
    except Exception as exc:  # noqa: BLE001 - we wrap any module execution error
        raise RootAgentImportError(f"Failed to import {agent_file}: {exc}") from exc

    root_agent = getattr(module, "root_agent", None)
    if root_agent is None:
        raise RootAgentImportError(
            f"{agent_file} does not define 'root_agent'. Define a root_agent = LlmAgent(...)."
        )
    return root_agent


def import_project_plugins(path: str, app_name: str, plugin_vars: list[str]) -> list[Any]:
    """Import ``<path>/<app_name>/plugins.py`` and return the instances named in the list.

    ``plugin_vars`` is the list of **module-level variable names** (from the ``runtime.json``
    manifest). Each name must refer to a plugin instance declared in ``plugins.py``. Returns the
    instances in the order of ``plugin_vars`` (empty if the list is empty — called only when at
    least one plugin is declared).

    Like :func:`import_root_agent`, we read+``compile()``+``exec()`` the source under a **unique**
    module name (no stale ``sys.modules`` cache after an edit). Raises
    :class:`PluginsImportError` (missing file, failed execution, missing variable).
    """
    plugins_file = Path(path) / app_name / "plugins.py"
    if not plugins_file.is_file():
        raise PluginsImportError(
            f"plugins.py not found: {plugins_file}. Declare a plugin (safety_add_plugin)."
        )

    module_name = f"_adk_toolkit_plugins_{app_name}_{next(_PLUGINS_IMPORT_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, plugins_file)
    if spec is None:  # pragma: no cover - degenerate importlib case
        raise PluginsImportError(f"Unable to prepare the import of {plugins_file}.")

    module = importlib.util.module_from_spec(spec)
    try:
        source = plugins_file.read_text(encoding="utf-8")
        code = compile(source, str(plugins_file), "exec")
        exec(code, module.__dict__)  # noqa: S102 - intentional execution of user code (plugins.py)
    except Exception as exc:  # noqa: BLE001 - we wrap any module execution error
        raise PluginsImportError(f"Failed to import {plugins_file}: {exc}") from exc

    instances: list[Any] = []
    for var in plugin_vars:
        instance = getattr(module, var, None)
        if instance is None:
            raise PluginsImportError(
                f"{plugins_file} does not define the plugin variable {var!r}. "
                "Check the runtime.json manifest ('plugins' key)."
            )
        instances.append(instance)
    return instances


# --------------------------------------------------------------------------- #
# Runner construction (services from runtime.py)
# --------------------------------------------------------------------------- #
def is_workflow_node_root(root: BaseAgent | BaseNode) -> bool:
    """True if ``root`` is a workflow node root (a ``BaseNode`` that is NOT a ``BaseAgent``).

    A ``Workflow`` (and any ``BaseNode`` graph root) is a ``BaseNode`` but **not** a ``BaseAgent``,
    so it must be wired into the ``Runner`` via ``node=`` rather than ``agent=`` (cf.
    ``docs/adk-api-notes/workflow.md``). An ``LlmAgent``/``SequentialAgent``/… is a ``BaseAgent``
    (the historical agent path). The ``google.adk.workflow`` import is lazy; if it is unavailable
    (older ADK without the workflow engine), nothing can be a node root → ``False``.
    """
    try:
        from google.adk.agents import BaseAgent
        from google.adk.workflow import BaseNode
    except ImportError:  # pragma: no cover - workflow engine always present in 2.x
        return False
    return isinstance(root, BaseNode) and not isinstance(root, BaseAgent)


def build_runner(
    app_name: str,
    root_agent: BaseAgent | BaseNode,
    runtime_config: RuntimeConfig,
    plugins: list[Any] | None = None,
) -> Runner:
    """Build a ``Runner`` wired onto the services of ``runtime_config``.

    The **sessions** service is always required (singleton factory ``get_session_service``). The
    **memory** and **artifacts** services are only passed if a backend is configured (otherwise
    omitted: ADK tolerates ``None``). We use ``Runner`` (and NOT ``InMemoryRunner``, which would
    recreate its own services and bypass the toolkit's config and singleton cache).

    **Root type (agent vs workflow node)**: the ``Runner`` accepts EITHER ``agent=`` (a
    ``BaseAgent``) OR ``node=`` (a ``BaseNode`` — e.g. a ``Workflow`` graph root, which is a
    ``BaseNode`` but NOT a ``BaseAgent``). We detect the kind via :func:`is_workflow_node_root`
    (``isinstance``) and pick the matching kwarg. The **agent path is unchanged** (backward
    compatible); a workflow-rooted project (``root_kind="workflow"``) is wired via ``node=``
    (verified to run offline end-to-end — cf. ``docs/adk-api-notes/workflow.md``).

    **Plugins (P4c)**: if ``plugins`` is non-empty, we take the NON-deprecated path
    ``Runner(app=App(name=app_name, root_agent=root_agent, plugins=[...]), ...)`` — ``Runner``'s
    direct ``plugins=`` argument is DEPRECATED in 2.1.0 (``DeprecationWarning``), whereas ``App``
    triggers no warning (verified by introspection). ``App.root_agent`` accepts a ``BaseNode`` too
    (its annotation is ``BaseAgent | Any | None`` — verified), so a workflow root + plugins also
    goes through ``App``. Without a plugin (default), we keep the historical path
    ``Runner(app_name=, agent=, ...)`` for an agent root — strictly unchanged behavior.

    Backend errors (``ValueError``: missing required field / missing extra) propagate to the
    caller, which converts them into ``err(...)``.
    """
    from google.adk.runners import Runner

    session_service = get_session_service(runtime_config.session)
    kwargs: dict[str, Any] = {"session_service": session_service}
    if runtime_config.memory is not None:
        kwargs["memory_service"] = get_memory_service(runtime_config.memory)
    if runtime_config.artifacts is not None:
        kwargs["artifact_service"] = get_artifact_service(runtime_config.artifacts)

    if plugins:
        # Non-deprecated path: App carries name/root_agent/plugins; Runner derives app_name from it.
        # App.root_agent accepts a BaseNode (workflow) root as well as a BaseAgent.
        from google.adk.apps import App

        app = App(name=app_name, root_agent=root_agent, plugins=list(plugins))
        return Runner(app=app, **kwargs)

    kwargs["app_name"] = app_name
    if is_workflow_node_root(root_agent):
        # Workflow/BaseNode root: wired via node= (NOT agent=, which only accepts a BaseAgent).
        kwargs["node"] = root_agent
        return Runner(**kwargs)
    kwargs["agent"] = root_agent
    return Runner(**kwargs)


# --------------------------------------------------------------------------- #
# RunConfig (streaming_mode validation against the real enum)
# --------------------------------------------------------------------------- #
def build_run_config(
    streaming_mode: str = "NONE",
    max_llm_calls: int | None = None,
    response_modalities: list[str] | None = None,
) -> RunConfig:
    """Build a ``RunConfig``; validate ``streaming_mode`` against the ``StreamingMode`` enum.

    ``streaming_mode`` is resolved **by name** (case-insensitive): ``NONE`` / ``SSE`` / ``BIDI``.
    An unknown name raises ``ValueError`` (actionable message). ``max_llm_calls=None`` leaves the
    ADK default (500) in place; an integer is passed through as-is. ``response_modalities`` (e.g.
    ``["TEXT"]``) is only passed if provided.
    """
    from google.adk.agents import RunConfig

    mode = _resolve_streaming_mode(streaming_mode)
    kwargs: dict[str, Any] = {"streaming_mode": mode}
    if max_llm_calls is not None:
        kwargs["max_llm_calls"] = max_llm_calls
    if response_modalities is not None:
        kwargs["response_modalities"] = response_modalities
    return RunConfig(**kwargs)


def streaming_mode_names() -> list[str]:
    """Return the valid ``StreamingMode`` names (for descriptors/errors)."""
    from google.adk.agents.run_config import StreamingMode

    return [m.name for m in StreamingMode]


def _resolve_streaming_mode(streaming_mode: str) -> Any:
    """Resolve a mode name (case-insensitive) to a ``StreamingMode`` member.

    Raises ``ValueError`` with the list of valid names if the mode is unknown.
    """
    from google.adk.agents.run_config import StreamingMode

    try:
        return StreamingMode[streaming_mode.strip().upper()]
    except KeyError as exc:
        valid = ", ".join(m.name for m in StreamingMode)
        raise ValueError(
            f"invalid streaming_mode: {streaming_mode!r}. Expected one of: {valid}."
        ) from exc


# --------------------------------------------------------------------------- #
# Event serialization
# --------------------------------------------------------------------------- #
def serialize_event(event: Event) -> dict[str, Any]:
    """Flatten an ADK ``Event`` into a simple JSON-serializable dict.

    Fields: ``author``; ``text`` (concatenation of the text parts, ``None`` if none);
    ``function_calls`` (``[{name, args}]``); ``function_responses`` (``[{name, response}]``);
    ``state_delta`` (``event.actions.state_delta``); ``transfer_to_agent``; ``is_final``
    (``event.is_final_response()``); ``partial``.
    """
    content = event.content
    parts = list(content.parts or []) if content is not None else []
    text = "".join(p.text for p in parts if p.text)

    function_calls = [
        {"name": fc.name, "args": dict(fc.args or {})} for fc in event.get_function_calls()
    ]
    function_responses = [
        {"name": fr.name, "response": fr.response} for fr in event.get_function_responses()
    ]

    actions = event.actions
    state_delta = dict(actions.state_delta or {}) if actions is not None else {}
    transfer_to_agent = actions.transfer_to_agent if actions is not None else None

    return {
        "author": event.author,
        "text": text or None,
        "function_calls": function_calls,
        "function_responses": function_responses,
        "state_delta": state_delta,
        "transfer_to_agent": transfer_to_agent,
        "is_final": event.is_final_response(),
        "partial": event.partial,
    }


# --------------------------------------------------------------------------- #
# Run + event collection
# --------------------------------------------------------------------------- #
async def collect_events(
    runner: Runner,
    *,
    user_id: str,
    session_id: str,
    new_message_text: str,
    run_config: RunConfig | None = None,
    progress: ProgressCallback | None = None,
) -> list[Event]:
    """Run the agent and collect all the ``Event`` objects produced.

    First ensures the session exists (creates it if ``get_session`` returns ``None`` —
    ``Runner.auto_create_session`` is ``False`` by default). Builds ``new_message`` as a
    ``types.Content`` with role ``"user"`` carrying ``new_message_text``, then iterates
    ``run_async``. If ``progress`` is provided, it is **awaited** per event (with the 1-based
    index and the serialized event) — used for the SSE progress of ``run_stream``.

    Returns the list of raw ``Event`` objects (the caller serializes via :func:`serialize_event`).
    """
    from google.genai import types

    session_service = runner.session_service
    app_name = runner.app_name
    existing = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if existing is None:
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )

    new_message = types.Content(role="user", parts=[types.Part.from_text(text=new_message_text)])

    events: list[Event] = []
    index = 0
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
        run_config=run_config,
    ):
        events.append(event)
        if progress is not None:
            index += 1
            await progress(index, serialize_event(event))
    return events
