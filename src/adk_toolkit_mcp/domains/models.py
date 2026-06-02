"""`models` domain: model configuration of an ADK agent (code-first, sidecar + regeneration).

A FastMCP sub-server mounted by the root server under the ``models`` namespace (tools exposed as
``models_<name>`` on the client side). Functions named with **BARE** names (``set``,
``configure_litellm``, ``generate_config``) — cf. ``docs/adk-api-notes/conventions.md``.

Each tool operates on ``(path, app_name, agent_name, …)``: it loads the sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, updates the model / generate_content config spec,
rewrites the sidecar, then **fully regenerates** ``agent.py`` (+ ``__init__.py``). Everything is
returned in the ``{ok, data, error}`` envelope; invalid inputs return ``err(...)`` (never an
exception).

See ``docs/adk-api-notes/models.md`` for the confirmed ADK signatures (LiteLlm.__init__,
GenerateContentConfig, HarmCategory + HarmBlockThreshold enum members).

Security note: API keys are **never** hardcoded in the generated code. If ``api_key_env`` is
provided, the generated code uses ``os.getenv("<ENV>")``. Otherwise, LiteLLM reads the
provider's env variables automatically.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    HARM_BLOCK_THRESHOLDS,
    HARM_CATEGORIES,
    LITELLM_PROVIDERS,
    GenerateContentConfigSpec,
    LiteLlmSpec,
    ProjectModel,
    SafetySettingSpec,
    add_or_update_agent,
    is_identifier,
    load_model,
    regenerate,
    save_model,
)
from ..workspace import Workspace

models_server: FastMCP = FastMCP("models")

#: app_name = Python package identifier (both folder AND module name).
_APP_NAME_ERR = (
    "Invalid app_name: expected a Python identifier "
    "(letters, digits, underscore; not starting with a digit)."
)


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


def _resolve_agent(
    path: str, app_name: str, agent_name: str
) -> tuple[ProjectModel, Any] | dict[str, Any]:
    """Load the model and resolve the agent. Returns ``(model, agent_spec)`` or ``err(...)``."""
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
            f"The {agent_name!r} agent is of type {spec.type!r}; "
            "only LlmAgent agents (type='llm') support model configuration."
        )
    return model, spec


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@models_server.tool(tags={"models"}, name="set")
def set_model(
    path: str,
    app_name: str,
    agent_name: str,
    model: str,
) -> dict[str, Any]:
    """Set a Gemini model (string) on an existing ``LlmAgent`` agent.

    Named ``set_model`` in Python (so as not to shadow the ``set`` builtin in this module), but
    **registered under the BARE tool name ``set``** -> exposed as ``models_set`` on the client
    side.

    ``model``: non-empty string (e.g. ``"gemini-2.5-flash"``, ``"gemini-2.0-flash-lite"``).
    Clears any previously set LiteLlm ``model_spec``.
    """
    if not model.strip():
        return err("model is empty.")

    result = _resolve_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    pm, spec = result

    updated = replace(spec, model=model, model_spec=None)
    pm = add_or_update_agent(pm, updated)
    return _commit(path, app_name, pm)


@models_server.tool(tags={"models"})
def configure_litellm(
    path: str,
    app_name: str,
    agent_name: str,
    provider: str,
    model: str,
    api_base: str = "",
    api_key_env: str = "",
) -> dict[str, Any]:
    """Configure a LiteLlm model on an existing ``LlmAgent`` agent.

    ``provider`` ∈ {openai, anthropic, ollama, ollama_chat, openrouter, vllm, lm_studio, gemini}.
    ``model``: the model name at the provider (e.g. ``"gpt-4o"``, ``"llama3"``, ``"mistral"``).
    ``api_base``: endpoint URL (optional; for ``lm_studio`` it defaults to
    ``http://127.0.0.1:1234/v1`` if absent).
    ``api_key_env``: the name of the env variable holding the API key. **The key is never
    hardcoded**: if provided, the generated code includes ``api_key=os.getenv("<ENV>")``.

    Generates ``model=LiteLlm(model="<provider>/<model>"[, api_base=...][, api_key=...])`` in
    ``agent.py``.
    """
    if provider not in LITELLM_PROVIDERS:
        return err(
            f"Unknown provider: {provider!r}. Supported: {', '.join(sorted(LITELLM_PROVIDERS))}."
        )
    if not model.strip():
        return err("model is empty.")

    result = _resolve_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    pm, spec = result

    model_spec = LiteLlmSpec(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
    )
    updated = replace(spec, model_spec=model_spec)
    pm = add_or_update_agent(pm, updated)
    return _commit(path, app_name, pm)


@models_server.tool(tags={"models"})
def generate_config(
    path: str,
    app_name: str,
    agent_name: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    top_p: float | None = None,
    top_k: float | None = None,
    safety_settings: list[dict[str, str]] | None = None,
    response_modalities: list[str] | None = None,
) -> dict[str, Any]:
    """Set the ``generate_content_config`` of an existing ``LlmAgent`` agent.

    Generates ``generate_content_config=types.GenerateContentConfig(...)`` in ``agent.py``.

    ``safety_settings``: list of ``{"category": "<HarmCategory>", "threshold": "<Threshold>"}``.
    The values are validated against the members of the ``HarmCategory`` / ``HarmBlockThreshold``
    enums (confirmed by google-genai introspection).

    All parameters are optional; only the provided (non-None) ones are included. Calling with all
    None clears the existing config (idempotent).
    """
    # Validate the safety_settings.
    parsed_ss: list[SafetySettingSpec] = []
    for ss in safety_settings or []:
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

    result = _resolve_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    pm, spec = result

    # If everything is None and no safety_settings, we clear the config.
    has_any = any(
        v is not None for v in [temperature, max_output_tokens, top_p, top_k, response_modalities]
    ) or bool(parsed_ss)

    gcc: GenerateContentConfigSpec | None = None
    if has_any:
        gcc = GenerateContentConfigSpec(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
            safety_settings=tuple(parsed_ss),
            response_modalities=tuple(response_modalities or []),
        )

    updated = replace(spec, generate_content_config=gcc)
    pm = add_or_update_agent(pm, updated)
    return _commit(path, app_name, pm)
