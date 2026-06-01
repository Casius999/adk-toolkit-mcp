"""Domaine `models` : configuration du modèle d'un agent ADK (code-first, sidecar + régénération).

Sous-serveur FastMCP monté par le serveur racine sous le namespace ``models`` (outils exposés
comme ``models_<nom>`` côté client). Fonctions nommées avec des noms **BARE** (``set``,
``configure_litellm``, ``generate_config``) — cf. ``docs/adk-api-notes/conventions.md``.

Chaque outil opère sur ``(path, app_name, agent_name, …)`` : il charge le sidecar
``<path>/<app_name>/.adk_toolkit/agents.json``, met à jour la spec du modèle / de la config
generate_content, réécrit le sidecar, puis **régénère intégralement** ``agent.py``
(+ ``__init__.py``). Tout est renvoyé dans l'enveloppe ``{ok, data, error}`` ; les entrées
invalides renvoient ``err(...)`` (jamais d'exception).

Voir ``docs/adk-api-notes/models.md`` pour les signatures ADK confirmées (LiteLlm.__init__,
GenerateContentConfig, HarmCategory + HarmBlockThreshold enum members).

Note sécurité : les clés API ne sont **jamais** écrites en dur dans le code généré. Si
``api_key_env`` est fourni, le code généré utilise ``os.getenv("<ENV>")``. Sinon, LiteLLM
lit les variables d'env du provider automatiquement.
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

#: app_name = identifiant de package Python (nom de dossier ET de module).
_APP_NAME_ERR = (
    "app_name invalide : attendu un identifiant Python "
    "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
)


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _load(path: str, app_name: str) -> ProjectModel | dict[str, Any]:
    """Charge le modèle ; renvoie un ``err(...)`` (dict) si le sidecar est corrompu."""
    ws = _app_ws(path, app_name)
    try:
        return load_model(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _commit(path: str, app_name: str, model: ProjectModel) -> dict[str, Any]:
    """Sauve le sidecar + régénère ``agent.py``. Convertit un cycle en ``err``.

    Renvoie le payload commun ``{app_name, agents, root, sidecar, regenerated, changed}``.
    """
    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle détecté au rendu
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
    """Charge le modèle et résout l'agent. Renvoie ``(model, agent_spec)`` ou ``err(...)``."""
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"agent_name invalide : {agent_name!r} (identifiant Python attendu).")

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    spec = model.get(agent_name)
    if spec is None:
        return err(f"Agent introuvable : {agent_name!r}.")
    if spec.type != "llm":
        return err(
            f"L'agent {agent_name!r} est de type {spec.type!r} ; "
            "seuls les agents LlmAgent (type='llm') supportent la configuration du modèle."
        )
    return model, spec


# --------------------------------------------------------------------------- #
# Outils MCP
# --------------------------------------------------------------------------- #
@models_server.tool(name="set")
def set_model(
    path: str,
    app_name: str,
    agent_name: str,
    model: str,
) -> dict[str, Any]:
    """Définit un modèle Gemini (chaîne) sur un agent ``LlmAgent`` existant.

    Nommée ``set_model`` en Python (pour ne pas masquer le builtin ``set`` dans ce module),
    mais **enregistrée sous le nom d'outil BARE ``set``** -> exposée ``models_set`` côté
    client.

    ``model`` : chaîne non vide (ex. ``"gemini-2.5-flash"``, ``"gemini-2.0-flash-lite"``).
    Efface tout ``model_spec`` LiteLlm précédemment défini.
    """
    if not model.strip():
        return err("model est vide.")

    result = _resolve_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    pm, spec = result

    updated = replace(spec, model=model, model_spec=None)
    pm = add_or_update_agent(pm, updated)
    return _commit(path, app_name, pm)


@models_server.tool
def configure_litellm(
    path: str,
    app_name: str,
    agent_name: str,
    provider: str,
    model: str,
    api_base: str = "",
    api_key_env: str = "",
) -> dict[str, Any]:
    """Configure un modèle LiteLlm sur un agent ``LlmAgent`` existant.

    ``provider`` ∈ {openai, anthropic, ollama, ollama_chat, openrouter, vllm, lm_studio, gemini}.
    ``model`` : nom du modèle chez le provider (ex. ``"gpt-4o"``, ``"llama3"``, ``"mistral"``).
    ``api_base`` : URL d'endpoint (optionnel ; pour ``lm_studio`` vaut par défaut
    ``http://127.0.0.1:1234/v1`` si absent).
    ``api_key_env`` : nom de la variable d'env portant la clé API. **La clé n'est jamais
    écrite en dur** : si fourni, le code généré inclut ``api_key=os.getenv("<ENV>")``.

    Génère ``model=LiteLlm(model="<provider>/<model>"[, api_base=...][, api_key=...])``
    dans ``agent.py``.
    """
    if provider not in LITELLM_PROVIDERS:
        return err(
            f"Provider inconnu : {provider!r}. Supportés : {', '.join(sorted(LITELLM_PROVIDERS))}."
        )
    if not model.strip():
        return err("model est vide.")

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


@models_server.tool
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
    """Définit la ``generate_content_config`` d'un agent ``LlmAgent`` existant.

    Génère ``generate_content_config=types.GenerateContentConfig(...)`` dans ``agent.py``.

    ``safety_settings`` : liste de ``{"category": "<HarmCategory>", "threshold": "<Threshold>"}``.
    Les valeurs sont validées contre les membres des enums ``HarmCategory`` /
    ``HarmBlockThreshold`` (confirmés par introspection google-genai).

    Tous les paramètres sont optionnels ; seuls ceux fournis (non-None) sont inclus.
    Appeler avec tous None efface la config existante (idempotent).
    """
    # Valider les safety_settings.
    parsed_ss: list[SafetySettingSpec] = []
    for ss in safety_settings or []:
        cat = ss.get("category", "")
        thr = ss.get("threshold", "")
        if cat not in HARM_CATEGORIES:
            return err(
                f"HarmCategory inconnue : {cat!r}. Connues : {', '.join(sorted(HARM_CATEGORIES))}."
            )
        if thr not in HARM_BLOCK_THRESHOLDS:
            return err(
                f"HarmBlockThreshold inconnu : {thr!r}. "
                f"Connus : {', '.join(sorted(HARM_BLOCK_THRESHOLDS))}."
            )
        parsed_ss.append(SafetySettingSpec(category=cat, threshold=thr))

    result = _resolve_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    pm, spec = result

    # Si tout est None et pas de safety_settings, on efface la config.
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
