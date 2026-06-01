"""Domaine `safety` : garde-fous d'agent (callbacks), plugins globaux et réglages de sûreté (P4c).

Sous-serveur FastMCP monté sous ``namespace="safety"`` → outils exposés ``safety_<nom>``. Noms
BARE (``add_callback``, ``add_plugin``, ``settings``). Le domaine opère sur un projet
``(path, app_name, …)`` : il met à jour le sidecar ``.adk_toolkit/agents.json`` (callbacks /
réglages) ou écrit ``plugins.py`` + le manifeste ``runtime.json`` (plugins), puis **régénère**
``agent.py``. Tout est renvoyé dans l'enveloppe ``{ok, data, error}``.

Trois surfaces (cf. ``docs/adk-api-notes/safety-observability.md`` pour les API ADK confirmées) :

1. :func:`add_callback` — attache un garde-fou (``block_keywords`` / ``max_input_chars`` /
   ``block_tool``) à un ``LlmAgent`` via le vrai kwarg (``before_model_callback`` /
   ``before_tool_callback``). Rendu comme une **vraie fonction** par ``project_model`` ; renvoyer
   non-``None`` court-circuite le LLM/l'outil (prouvé hors-ligne).
2. :func:`add_plugin` — génère/étend ``<app_dir>/<app>/plugins.py`` avec une sous-classe
   ``BasePlugin`` (politique globale réelle : ``logging`` via ``on_event_callback``, ou
   ``tool_denylist`` via ``before_tool_callback``), enregistrée dans le manifeste ``runtime.json``
   pour que ``run_core.build_runner`` la câble sur le ``Runner`` (via ``App``).
3. :func:`settings` — fine convenance : ``gemini_safety`` route vers le rendu EXISTANT de
   ``generate_content_config`` (réutilise ``project_model`` — pas de duplication) ;
   ``max_llm_calls`` est persisté comme plafond d'exécution par défaut de l'agent et **réellement
   appliqué** par les outils ``run_*`` quand l'appel ne passe pas de ``max_llm_calls`` explicite
   (le domaine ``run`` lit la valeur persistée de l'agent root → ``RunConfig.max_llm_calls``).
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

#: app_name = identifiant de package Python (nom de dossier ET de module).
_APP_NAME_ERR = (
    "app_name invalide : attendu un identifiant Python "
    "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
)

#: Genres de plugin générés (politiques globales réelles).
_PLUGIN_KINDS: frozenset[str] = frozenset({"logging", "tool_denylist"})

#: Nom du fichier de plugins généré (dans le dossier de l'app).
_PLUGINS_FILE = "plugins.py"


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


def _resolve_llm_agent(
    path: str, app_name: str, agent_name: str
) -> tuple[ProjectModel, AgentSpec] | dict[str, Any]:
    """Charge le modèle et résout un ``LlmAgent`` existant. Renvoie ``(model, spec)`` ou ``err``."""
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
            f"L'agent {agent_name!r} est de type {spec.type!r} ; seuls les agents LlmAgent "
            "(type='llm') supportent les callbacks et les réglages de sûreté."
        )
    return model, spec


# --------------------------------------------------------------------------- #
# Outil 1 — add_callback (garde-fou attaché à l'agent)
# --------------------------------------------------------------------------- #
@safety_server.tool
def add_callback(
    path: str,
    app_name: str,
    agent_name: str,
    hook: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Attache un garde-fou (callback) à un ``LlmAgent`` puis régénère ``agent.py``.

    ``hook`` ∈ {before_model, after_model, before_tool, after_tool, before_agent, after_agent}.
    ``policy`` est un dict ``{"kind": "<policy>", ...params}`` :

    - ``block_keywords`` (before_model) : ``{"kind": "block_keywords", "keywords": "bomb,hack",
      "refusal": "..."}`` — refuse (court-circuite le LLM) si le texte utilisateur contient un
      terme bloqué.
    - ``max_input_chars`` (before_model) : ``{"kind": "max_input_chars", "max_chars": "2000"}`` —
      refuse si l'entrée dépasse N caractères.
    - ``block_tool`` (before_tool) : ``{"kind": "block_tool", "denylist": "delete_db",
      "message": "..."}`` — court-circuite l'outil si son nom est dans la denylist.

    La politique est rendue comme une **vraie fonction** attachée via le kwarg réel
    (``before_model_callback=…``). Un seul callback par hook (un second remplace).
    """
    if hook not in CALLBACK_HOOKS:
        return err(f"Hook inconnu : {hook!r}. Connus : {', '.join(sorted(CALLBACK_HOOKS))}.")
    if not isinstance(policy, dict):
        return err("policy doit être un objet {'kind': '<policy>', ...}.")
    kind = str(policy.get("kind", ""))
    if kind not in POLICY_KINDS:
        return err(f"Politique inconnue : {kind!r}. Connues : {', '.join(sorted(POLICY_KINDS))}.")

    params = tuple((str(k), str(v)) for k, v in policy.items() if k != "kind")
    # hook/kind sont validés ci-dessus contre les ensembles autorisés -> cast vers les Literal.
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
# Outil 2 — add_plugin (politique globale via BasePlugin + manifeste runtime)
# --------------------------------------------------------------------------- #
@safety_server.tool
def add_plugin(
    path: str,
    app_name: str,
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Génère/étend ``plugins.py`` avec un plugin ``BasePlugin`` + l'inscrit au manifeste runtime.

    Politiques globales réelles (``kind``) :

    - ``logging`` : enregistre chaque évènement via ``on_event_callback`` dans une liste
      module-level ``<var>_events`` (inspectable hors-ligne) et journalise via ``logging``.
    - ``tool_denylist`` : court-circuite globalement tout appel d'outil dont le nom est dans
      ``config={"denylist": "delete_db,drop_table"}`` (via ``before_tool_callback``).

    Le plugin est déclaré comme variable module-level ``<name>`` dans ``plugins.py`` et enregistré
    dans ``runtime.json`` (clé ``plugins``) afin que ``run_core.build_runner`` le câble sur le
    ``Runner`` (via ``App``). ``name`` doit être un identifiant Python (sert de variable + de nom
    logique du plugin). Idempotent : un plugin de même ``name`` est remplacé.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(name):
        return err(f"name invalide : {name!r} (identifiant Python attendu, sert de variable).")
    if kind not in _PLUGIN_KINDS:
        return err(f"kind inconnu : {kind!r}. Connus : {', '.join(sorted(_PLUGIN_KINDS))}.")

    cfg = config or {}
    if kind == "tool_denylist":
        denylist = [s.strip() for s in str(cfg.get("denylist", "")).split(",") if s.strip()]
        if not denylist:
            return err("tool_denylist : config={'denylist': 'tool1,tool2'} est requis (≥ 1 outil).")
    else:  # logging
        denylist = []

    ws = _app_ws(path, app_name)
    if not ws.path("agent.py").is_file():
        agent_py = ws.path("agent.py")
        return err(f"Dossier d'app introuvable : {agent_py}. Scaffolde d'abord (project_create).")

    # Charge la config runtime, met à jour le manifeste (remplace un plugin de même var).
    try:
        config_rt = load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))

    new_spec = PluginSpec(var=name, name=name, kind=kind)
    others = [p for p in config_rt.plugins if p.var != name]
    updated_specs = (*others, new_spec)
    config_rt = replace(config_rt, plugins=updated_specs)

    # (Ré)génère plugins.py à partir du manifeste complet (déterministe, idempotent).
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
    """Construit les payloads de rendu pour TOUS les plugins du manifeste (régénération complète).

    Pour conserver la config d'un plugin ``tool_denylist`` déjà présent (autre que celui qu'on
    ajoute), on relit son ``denylist`` depuis le ``plugins.py`` existant (best-effort). Le plugin
    en cours d'ajout (``denylist_for``) utilise le ``denylist`` fraîchement fourni.
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
# Outil 3 — settings (gemini_safety -> rendu existant ; max_llm_calls -> plafond)
# --------------------------------------------------------------------------- #
@safety_server.tool(name="settings")
def safety_settings(
    path: str,
    app_name: str,
    agent_name: str,
    max_llm_calls: int | None = None,
    gemini_safety: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Réglages de sûreté d'un ``LlmAgent`` : safety settings Gemini + plafond d'appels LLM.

    Nommée ``safety_settings`` en Python mais **enregistrée sous le nom BARE ``settings``** →
    exposée ``safety_settings`` côté client.

    - ``gemini_safety`` : liste de ``{"category": "<HarmCategory>", "threshold": "<Threshold>"}``.
      **Route vers le rendu EXISTANT** de ``generate_content_config`` (réutilise
      ``project_model.GenerateContentConfigSpec`` + le rendu des ``types.SafetySetting`` — AUCUNE
      duplication de la logique de sûreté du domaine ``models``). Fusionne avec une
      ``generate_content_config`` existante (préserve temperature, etc.).
    - ``max_llm_calls`` : stocké comme plafond d'appels LLM **par défaut de l'agent**, persisté
      dans le sidecar (``AgentSpec.max_llm_calls``). Il est **réellement utilisé** par les outils
      ``run_*`` (``run_agent``/``run_stream``/``run_live``) quand l'appel ne passe PAS de
      ``max_llm_calls`` explicite : le domaine ``run`` lit la valeur persistée de l'agent ROOT et
      la transmet à ``RunConfig.max_llm_calls``. Une valeur d'appelant explicite prime toujours.
      Ce n'est pas un kwarg d'``LlmAgent`` — donc non rendu dans ``agent.py``.

    Appeler sans aucun des deux est une erreur (rien à faire).
    """
    if max_llm_calls is None and not gemini_safety:
        return err("Fournis 'gemini_safety' et/ou 'max_llm_calls' (rien à régler sinon).")
    if max_llm_calls is not None and max_llm_calls <= 0:
        return err(f"max_llm_calls doit être > 0 (reçu {max_llm_calls}).")

    # Valider les safety settings contre les enums (mêmes constantes que le domaine models).
    parsed_ss: list[SafetySettingSpec] = []
    for ss in gemini_safety or []:
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

    result = _resolve_llm_agent(path, app_name, agent_name)
    if isinstance(result, dict):
        return result
    model, spec = result

    updated = spec
    # gemini_safety : fusionne avec la generate_content_config existante (réutilise le rendu).
    if parsed_ss:
        current = spec.generate_content_config or GenerateContentConfigSpec()
        merged = replace(current, safety_settings=tuple(parsed_ss))
        updated = replace(updated, generate_content_config=merged)

    out_extra: dict[str, Any] = {"agent": agent_name}
    if parsed_ss:
        out_extra["gemini_safety"] = [s.to_dict() for s in parsed_ss]

    # max_llm_calls : persiste sur la spec (sérialisé au sidecar, NON rendu dans agent.py — c'est
    # un réglage de RunConfig exposé par le domaine run).
    if max_llm_calls is not None:
        updated = replace(updated, max_llm_calls=max_llm_calls)
        out_extra["max_llm_calls"] = max_llm_calls

    model = add_or_update_agent(model, updated)
    out = _commit(path, app_name, model)
    if out["ok"]:
        out["data"].update(out_extra)
    return out
