"""`safety` domain: agent guardrails (callbacks), global plugins and safety settings (P4c).

A FastMCP sub-server mounted under ``namespace="safety"`` → tools exposed as ``safety_<name>``.
BARE names (``add_callback``, ``add_plugin``, ``settings``). The domain operates on a project
``(path, app_name, …)``: it updates the sidecar ``.adk_toolkit/agents.json`` (callbacks /
settings) or writes ``plugins.py`` + the ``runtime.json`` manifest (plugins), then **regenerates**
``agent.py``. Everything is returned in the ``{ok, data, error}`` envelope.

Three surfaces (cf. ``docs/adk-api-notes/safety-observability.md`` for the confirmed ADK APIs):

1. :func:`add_callback` — attaches a guardrail (``block_keywords`` / ``max_input_chars`` /
   ``block_tool``) to an ``LlmAgent`` via the real kwarg (``before_model_callback`` /
   ``before_tool_callback``). Rendered as a **real function** by ``project_model``; returning
   non-``None`` short-circuits the LLM/the tool (proven offline).
2. :func:`add_plugin` — generates/extends ``<app_dir>/<app>/plugins.py`` with a ``BasePlugin``
   subclass (a real global policy: ``logging`` via ``on_event_callback``, or ``tool_denylist`` via
   ``before_tool_callback``), registered in the ``runtime.json`` manifest so that
   ``run_core.build_runner`` wires it onto the ``Runner`` (via ``App``).
3. :func:`settings` — fine convenience: ``gemini_safety`` routes to the EXISTING rendering of
   ``generate_content_config`` (reuses ``project_model`` — no duplication); ``max_llm_calls`` is
   persisted as the agent's default execution cap and **actually applied** by the ``run_*`` tools
   when the call passes no explicit ``max_llm_calls`` (the ``run`` domain reads the root agent's
   persisted value → ``RunConfig.max_llm_calls``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    CALLBACK_HOOKS,
    HARM_BLOCK_THRESHOLDS,
    HARM_CATEGORIES,
    POLICY_KINDS,
    AgentSpec,
    CallbackHook,
    CallbackSpec,
    GenerateContentConfigSpec,
    PolicyKind,
    ProjectModel,
    SafetySettingSpec,
    add_or_replace_callback,
    add_or_update_agent,
    is_identifier,
    load_model,
    regenerate,
    save_model,
    validate_callback_spec,
)
from ..runtime import (
    PluginSpec,
    load_runtime_config,
    save_runtime_config,
)
from ..workspace import Workspace
from . import safety_plugins

safety_server: FastMCP = FastMCP("safety")

#: app_name = Python package identifier (both folder AND module name).
_APP_NAME_ERR = (
    "Invalid app_name: expected a Python identifier "
    "(letters, digits, underscore; not starting with a digit)."
)

#: Generated plugin kinds (real global policies).
_PLUGIN_KINDS: frozenset[str] = frozenset({"logging", "tool_denylist"})

#: Name of the generated plugins file (in the app's folder).
_PLUGINS_FILE = "plugins.py"


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _load(path: str, app_name: str) -> ProjectModel | dict[str, Any]:
    """Load the model; return an ``err(...)`` (dict) if the sidecar is corrupt."""
    ws = _app_ws(path, app_name)
    try:
        return load_model(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _commit(path: str, app_name: str, model: ProjectModel) -> dict[str, Any]:
    """Save the sidecar + regenerate ``agent.py``. Converts a cycle into ``err``.

    Returns the common payload ``{app_name, agents, root, sidecar, regenerated, changed}``.
    """
    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle detected at render time
        return err(str(exc))
    sidecar_changed = save_model(ws, model)
    return ok(
        {
            "app_name": app_name,
            "agents": list(model.agent_names()),
            "root": model.root,
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


def _resolve_llm_agent(
    path: str, app_name: str, agent_name: str
) -> tuple[ProjectModel, AgentSpec] | dict[str, Any]:
    """Load the model and resolve an existing ``LlmAgent``. Returns ``(model, spec)`` or ``err``."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent_name: {agent_name!r} (Python identifier expected).")

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    spec = model.get(agent_name)
    if spec is None:
        return err(f"Agent not found: {agent_name!r}.")
    if spec.type != "llm":
        return err(
            f"The {agent_name!r} agent is of type {spec.type!r}; only LlmAgent agents "
            "(type='llm') support callbacks and safety settings."
        )
    return model, spec


# --------------------------------------------------------------------------- #
# Tool 1 — add_callback (guardrail attached to the agent)
# --------------------------------------------------------------------------- #
@safety_server.tool(tags={"safety"})
def add_callback(
    path: str,
    app_name: str,
    agent_name: str,
    hook: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Attach a guardrail (callback) to an ``LlmAgent`` then regenerate ``agent.py``.

    ``hook`` ∈ {before_model, after_model, before_tool, after_tool, before_agent, after_agent}.
    ``policy`` is a dict ``{"kind": "<policy>", ...params}``:

    - ``block_keywords`` (before_model): ``{"kind": "block_keywords", "keywords": "bomb,hack",
      "refusal": "..."}`` — refuses (short-circuits the LLM) if the user text contains a blocked
      term.
    - ``max_input_chars`` (before_model): ``{"kind": "max_input_chars", "max_chars": "2000"}`` —
      refuses if the input exceeds N characters.
    - ``block_tool`` (before_tool): ``{"kind": "block_tool", "denylist": "delete_db",
      "message": "..."}`` — short-circuits the tool if its name is in the denylist.

    The policy is rendered as a **real function** attached via the real kwarg
    (``before_model_callback=…``). One callback per hook (a second one replaces).
    """
    if hook not in CALLBACK_HOOKS:
        return err(f"Unknown hook: {hook!r}. Known: {', '.join(sorted(CALLBACK_HOOKS))}.")
    if not isinstance(policy, dict):
        return err("policy must be an object {'kind': '<policy>', ...}.")
    kind = str(policy.get("kind", ""))
    if kind not in POLICY_KINDS:
        return err(f"Unknown policy: {kind!r}. Known: {', '.join(sorted(POLICY_KINDS))}.")

    params = tuple((str(k), str(v)) for k, v in policy.items() if k != "kind")
    # hook/kind are validated above against the allowed sets -> cast to the Literals.
    callback = CallbackSpec(
        hook=cast("CallbackHook", hook), policy=cast("PolicyKind", kind), params=params
    )
    cb_error = validate_callback_spec(callback)
    if cb_error is not None:
        return err(cb_error)

    result = _resolve_llm_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    model, spec = result

    updated = add_or_replace_callback(spec, callback)
    model = add_or_update_agent(model, updated)
    out = _commit(path, app_name, model)
    if out["ok"]:
        out["data"]["callback"] = {"agent": agent_name, "hook": hook, "policy": kind}
    return out


# --------------------------------------------------------------------------- #
# Tool 2 — add_plugin (global policy via BasePlugin + runtime manifest)
# --------------------------------------------------------------------------- #
@safety_server.tool(tags={"safety"})
def add_plugin(
    path: str,
    app_name: str,
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate/extend ``plugins.py`` with a ``BasePlugin`` + register it in the runtime manifest.

    Real global policies (``kind``):

    - ``logging``: records each event via ``on_event_callback`` in a module-level ``<var>_events``
      list (offline-inspectable) and logs via ``logging``.
    - ``tool_denylist``: globally short-circuits any tool call whose name is in
      ``config={"denylist": "delete_db,drop_table"}`` (via ``before_tool_callback``).

    The plugin is declared as a module-level variable ``<name>`` in ``plugins.py`` and registered
    in ``runtime.json`` (``plugins`` key) so that ``run_core.build_runner`` wires it onto the
    ``Runner`` (via ``App``). ``name`` must be a Python identifier (serves as the variable + the
    plugin's logical name). Idempotent: a plugin with the same ``name`` is replaced.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(name):
        return err(f"Invalid name: {name!r} (Python identifier expected, serves as variable).")
    if kind not in _PLUGIN_KINDS:
        return err(f"Unknown kind: {kind!r}. Known: {', '.join(sorted(_PLUGIN_KINDS))}.")

    cfg = config or {}
    if kind == "tool_denylist":
        denylist = [s.strip() for s in str(cfg.get("denylist", "")).split(",") if s.strip()]
        if not denylist:
            return err("tool_denylist: config={'denylist': 'tool1,tool2'} is required (≥ 1 tool).")
    else:  # logging
        denylist = []

    ws = _app_ws(path, app_name)
    if not ws.path("agent.py").is_file():
        agent_py = ws.path("agent.py")
        return err(f"App folder not found: {agent_py}. Scaffold first (project_create).")

    # Load the runtime config, update the manifest (replace a plugin with the same var).
    try:
        config_rt = load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))

    new_spec = PluginSpec(var=name, name=name, kind=kind)
    others = [p for p in config_rt.plugins if p.var != name]
    updated_specs = (*others, new_spec)
    config_rt = replace(config_rt, plugins=updated_specs)

    # (Re)generate plugins.py from the full manifest (deterministic, idempotent).
    plugin_payloads = _plugin_payloads(updated_specs, denylist_for=name, denylist=denylist, ws=ws)
    source = safety_plugins.render_plugins_module(plugin_payloads)
    plugins_changed = ws.write(_PLUGINS_FILE, source)
    runtime_changed = save_runtime_config(ws, config_rt)

    return ok(
        {
            "app_name": app_name,
            "plugin": {"name": name, "kind": kind},
            "plugins_file": str(ws.path(_PLUGINS_FILE)),
            "manifest": [p.to_dict() for p in updated_specs],
            "changed": plugins_changed or runtime_changed,
        }
    )


def _plugin_payloads(
    specs: tuple[PluginSpec, ...],
    *,
    denylist_for: str,
    denylist: list[str],
    ws: Workspace,
) -> list[dict[str, Any]]:
    """Build the render payloads for ALL the manifest's plugins (full regeneration).

    To preserve the config of an already-present ``tool_denylist`` plugin (other than the one
    being added), we re-read its ``denylist`` from the existing ``plugins.py`` (best-effort). The
    plugin being added (``denylist_for``) uses the freshly provided ``denylist``.
    """
    existing = (
        safety_plugins.parse_existing_denylists(ws.read(_PLUGINS_FILE))
        if ws.exists(_PLUGINS_FILE)
        else {}
    )
    payloads: list[dict[str, Any]] = []
    for spec in specs:
        dl = denylist if spec.var == denylist_for else existing.get(spec.var, [])
        payloads.append({"var": spec.var, "name": spec.name, "kind": spec.kind, "denylist": dl})
    return payloads


# --------------------------------------------------------------------------- #
# Tool 3 — settings (gemini_safety -> existing rendering; max_llm_calls -> cap)
# --------------------------------------------------------------------------- #
@safety_server.tool(tags={"safety"}, name="settings")
def safety_settings(
    path: str,
    app_name: str,
    agent_name: str,
    max_llm_calls: int | None = None,
    gemini_safety: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Safety settings of an ``LlmAgent``: Gemini safety settings + LLM call cap.

    Named ``safety_settings`` in Python but **registered under the BARE name ``settings``** →
    exposed as ``safety_settings`` on the client side.

    - ``gemini_safety``: list of ``{"category": "<HarmCategory>", "threshold": "<Threshold>"}``.
      **Routes to the EXISTING rendering** of ``generate_content_config`` (reuses
      ``project_model.GenerateContentConfigSpec`` + the ``types.SafetySetting`` rendering — NO
      duplication of the ``models`` domain's safety logic). Merges with an existing
      ``generate_content_config`` (preserves temperature, etc.).
    - ``max_llm_calls``: stored as the agent's **default** LLM call cap, persisted in the sidecar
      (``AgentSpec.max_llm_calls``). It is **actually used** by the ``run_*`` tools
      (``run_agent``/``run_stream``/``run_live``) when the call passes NO explicit
      ``max_llm_calls``: the ``run`` domain reads the ROOT agent's persisted value and passes it to
      ``RunConfig.max_llm_calls``. An explicit caller value always wins. It is not an ``LlmAgent``
      kwarg — so it is not rendered in ``agent.py``.

    Calling with neither of the two is an error (nothing to do).
    """
    if max_llm_calls is None and not gemini_safety:
        return err("Provide 'gemini_safety' and/or 'max_llm_calls' (nothing to set otherwise).")
    if max_llm_calls is not None and max_llm_calls <= 0:
        return err(f"max_llm_calls must be > 0 (received {max_llm_calls}).")

    # Validate the safety settings against the enums (same constants as the models domain).
    parsed_ss: list[SafetySettingSpec] = []
    for ss in gemini_safety or []:
        cat = ss.get("category", "")
        thr = ss.get("threshold", "")
        if cat not in HARM_CATEGORIES:
            return err(
                f"Unknown HarmCategory: {cat!r}. Known: {', '.join(sorted(HARM_CATEGORIES))}."
            )
        if thr not in HARM_BLOCK_THRESHOLDS:
            return err(
                f"Unknown HarmBlockThreshold: {thr!r}. "
                f"Known: {', '.join(sorted(HARM_BLOCK_THRESHOLDS))}."
            )
        parsed_ss.append(SafetySettingSpec(category=cat, threshold=thr))

    result = _resolve_llm_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    model, spec = result

    updated = spec
    # gemini_safety: merges with the existing generate_content_config (reuses the rendering).
    if parsed_ss:
        current = spec.generate_content_config or GenerateContentConfigSpec()
        merged = replace(current, safety_settings=tuple(parsed_ss))
        updated = replace(updated, generate_content_config=merged)

    out_extra: dict[str, Any] = {"agent": agent_name}
    if parsed_ss:
        out_extra["gemini_safety"] = [s.to_dict() for s in parsed_ss]

    # max_llm_calls: persists on the spec (serialized to the sidecar, NOT rendered in agent.py —
    # it is a RunConfig setting exposed by the run domain).
    if max_llm_calls is not None:
        updated = replace(updated, max_llm_calls=max_llm_calls)
        out_extra["max_llm_calls"] = max_llm_calls

    model = add_or_update_agent(model, updated)
    out = _commit(path, app_name, model)
    if out["ok"]:
        out["data"].update(out_extra)
    return out
