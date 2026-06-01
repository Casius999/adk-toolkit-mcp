"""Cœur d'exécution d'agents ADK (P3a) — helpers testables hors-ligne.

Ce module factorise toute la mécanique d'exécution d'un agent ADK de façon à pouvoir la
**prouver hors-ligne avec un FakeLlm** (aucune clé API requise). Le domaine ``run`` n'est qu'une
fine couche d'outils MCP au-dessus de ces helpers.

Contenu (cf. ``docs/adk-api-notes/runtime-run.md`` pour les signatures ADK confirmées) :

- :func:`build_runner` — câble un ``google.adk.runners.Runner`` sur les services
  session/memory/artifacts issus de :mod:`adk_toolkit_mcp.runtime` (mêmes fabriques singleton).
- :func:`collect_events` — garantit l'existence de la session (la crée au besoin), lance
  ``run_async`` et collecte les ``Event`` ; un callback ``progress`` optionnel est *awaité* par
  événement (support SSE).
- :func:`serialize_event` — aplati un ``Event`` en dict simple ``{author, text,
  function_calls, function_responses, state_delta, transfer_to_agent, is_final, partial}``.
- :func:`import_root_agent` — importe ``<path>/<app_name>/agent.py`` et renvoie ``root_agent``
  via ``importlib`` avec un nom de module UNIQUE à chaque appel (pas de cache ``sys.modules``
  périmé : une édition d'``agent.py`` est reprise). Lève :class:`RootAgentImportError`.
- :func:`build_run_config` — valide ``streaming_mode`` contre la vraie enum ``StreamingMode``
  et construit un ``RunConfig``.

Aucun import ADK au chargement du module (tout est paresseux), pour rester cohérent avec le
reste du toolkit et garder les tests rapides.
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

if TYPE_CHECKING:  # pragma: no cover - hints seulement, imports réels paresseux
    from google.adk.agents import BaseAgent, RunConfig
    from google.adk.events import Event
    from google.adk.runners import Runner

#: Compteur monotone garantissant un nom de module unique par ``import_root_agent``.
_IMPORT_COUNTER = itertools.count()

#: Compteur monotone pour un nom de module unique par ``import_project_plugins`` (même raison
#: que pour ``import_root_agent`` : pas de cache ``sys.modules`` périmé après édition).
_PLUGINS_IMPORT_COUNTER = itertools.count()

#: Callback de progression : reçoit ``(index_1_based, serialized_event)`` et est awaité.
ProgressCallback = Callable[[int, dict[str, Any]], Awaitable[None]]


class RootAgentImportError(Exception):
    """Échec d'import de ``root_agent`` (fichier absent, erreur d'exécution, attribut manquant).

    Le domaine ``run`` convertit cette exception en ``err(...)`` avec un message actionnable.
    """


class PluginsImportError(Exception):
    """Échec d'import des plugins de projet (``plugins.py`` absent/cassé, variable manquante).

    Les domaines convertissent cette exception en ``err(...)`` avec un message actionnable.
    """


# --------------------------------------------------------------------------- #
# Import de root_agent (nom de module unique → pas de cache périmé)
# --------------------------------------------------------------------------- #
def import_root_agent(path: str, app_name: str) -> BaseAgent:
    """Importe ``<path>/<app_name>/agent.py`` et renvoie son ``root_agent``.

    Utilise un nom de module **unique à chaque appel** (suffixe via un compteur monotone) afin
    qu'une édition d'``agent.py`` entre deux appels soit bien reprise (jamais servie depuis un
    ``sys.modules`` périmé). Le module n'est volontairement PAS inséré dans ``sys.modules``.

    Lève :class:`RootAgentImportError` si le fichier est absent, si son exécution échoue, ou si
    ``root_agent`` n'y est pas défini.
    """
    agent_file = Path(path) / app_name / "agent.py"
    if not agent_file.is_file():
        raise RootAgentImportError(
            f"agent.py introuvable : {agent_file}. Scaffolde d'abord l'app (project_create)."
        )

    module_name = f"_adk_toolkit_root_agent_{app_name}_{next(_IMPORT_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    if spec is None:  # pragma: no cover - cas dégénéré d'importlib
        raise RootAgentImportError(f"Impossible de préparer l'import de {agent_file}.")

    module = importlib.util.module_from_spec(spec)
    # On lit la source et on la COMPILE/EXÉCUTE directement plutôt que via
    # ``spec.loader.exec_module`` : le ``SourceFileLoader`` met en cache le bytecode par
    # (chemin, mtime), et sur Windows deux écritures dans le même tick de mtime renvoient une
    # version PÉRIMÉE — une édition d'``agent.py`` ne serait alors pas reprise. Lire+compiler à
    # chaque appel garantit la fraîcheur (en plus du nom de module unique).
    try:
        source = agent_file.read_text(encoding="utf-8")
        code = compile(source, str(agent_file), "exec")
        exec(code, module.__dict__)  # noqa: S102 - exécution voulue du code utilisateur (agent.py)
    except Exception as exc:  # noqa: BLE001 - on enveloppe toute erreur d'exécution du module
        raise RootAgentImportError(f"Échec de l'import de {agent_file} : {exc}") from exc

    root_agent = getattr(module, "root_agent", None)
    if root_agent is None:
        raise RootAgentImportError(
            f"{agent_file} ne définit pas 'root_agent'. Définis un root_agent = LlmAgent(...)."
        )
    return root_agent


def import_project_plugins(path: str, app_name: str, plugin_vars: list[str]) -> list[Any]:
    """Importe ``<path>/<app_name>/plugins.py`` et renvoie les instances nommées dans la liste.

    ``plugin_vars`` est la liste des **noms de variables module-level** (issus du manifeste
    ``runtime.json``). Chaque nom doit désigner une instance de plugin déclarée dans
    ``plugins.py``. Renvoie les instances dans l'ordre de ``plugin_vars`` (vide si la liste est
    vide — appelée seulement quand au moins un plugin est déclaré).

    Comme :func:`import_root_agent`, on lit+``compile()``+``exec()`` la source sous un nom de
    module **unique** (pas de cache ``sys.modules`` périmé après édition). Lève
    :class:`PluginsImportError` (fichier absent, exécution échouée, variable manquante).
    """
    plugins_file = Path(path) / app_name / "plugins.py"
    if not plugins_file.is_file():
        raise PluginsImportError(
            f"plugins.py introuvable : {plugins_file}. Déclare un plugin (safety_add_plugin)."
        )

    module_name = f"_adk_toolkit_plugins_{app_name}_{next(_PLUGINS_IMPORT_COUNTER)}"
    spec = importlib.util.spec_from_file_location(module_name, plugins_file)
    if spec is None:  # pragma: no cover - cas dégénéré d'importlib
        raise PluginsImportError(f"Impossible de préparer l'import de {plugins_file}.")

    module = importlib.util.module_from_spec(spec)
    try:
        source = plugins_file.read_text(encoding="utf-8")
        code = compile(source, str(plugins_file), "exec")
        exec(code, module.__dict__)  # noqa: S102 - exécution voulue du code utilisateur (plugins.py)
    except Exception as exc:  # noqa: BLE001 - on enveloppe toute erreur d'exécution du module
        raise PluginsImportError(f"Échec de l'import de {plugins_file} : {exc}") from exc

    instances: list[Any] = []
    for var in plugin_vars:
        instance = getattr(module, var, None)
        if instance is None:
            raise PluginsImportError(
                f"{plugins_file} ne définit pas la variable de plugin {var!r}. "
                "Vérifie le manifeste runtime.json (clé 'plugins')."
            )
        instances.append(instance)
    return instances


# --------------------------------------------------------------------------- #
# Construction du Runner (services issus de runtime.py)
# --------------------------------------------------------------------------- #
def build_runner(
    app_name: str,
    root_agent: BaseAgent,
    runtime_config: RuntimeConfig,
    plugins: list[Any] | None = None,
) -> Runner:
    """Construit un ``Runner`` câblé sur les services de ``runtime_config``.

    Le service de **sessions** est toujours requis (fabrique singleton ``get_session_service``).
    Les services de **mémoire** et d'**artifacts** ne sont passés que si un backend est
    configuré (sinon omis : ADK tolère ``None``). On utilise ``Runner`` (et NON
    ``InMemoryRunner``, qui recréerait ses propres services et court-circuiterait la config et
    le cache singleton du toolkit).

    **Plugins (P4c)** : si ``plugins`` est non vide, on emprunte le chemin NON déprécié
    ``Runner(app=App(name=app_name, root_agent=root_agent, plugins=[...]), ...)`` — l'argument
    ``plugins=`` direct de ``Runner`` est DÉPRÉCIÉ en 2.1.0 (``DeprecationWarning``), tandis que
    ``App`` ne déclenche aucun warning (vérifié par introspection). Sans plugin (défaut), on
    garde le chemin historique ``Runner(app_name=, agent=, ...)`` — comportement strictement
    inchangé (compat ascendante).

    Les erreurs de backend (``ValueError`` : champ requis manquant / extra absent) remontent à
    l'appelant, qui les convertit en ``err(...)``.
    """
    from google.adk.runners import Runner

    session_service = get_session_service(runtime_config.session)
    kwargs: dict[str, Any] = {"session_service": session_service}
    if runtime_config.memory is not None:
        kwargs["memory_service"] = get_memory_service(runtime_config.memory)
    if runtime_config.artifacts is not None:
        kwargs["artifact_service"] = get_artifact_service(runtime_config.artifacts)

    if plugins:
        # Chemin non déprécié : App porte name/root_agent/plugins ; Runner en dérive app_name.
        from google.adk.apps import App

        app = App(name=app_name, root_agent=root_agent, plugins=list(plugins))
        return Runner(app=app, **kwargs)

    kwargs["app_name"] = app_name
    kwargs["agent"] = root_agent
    return Runner(**kwargs)


# --------------------------------------------------------------------------- #
# RunConfig (validation du streaming_mode contre la vraie enum)
# --------------------------------------------------------------------------- #
def build_run_config(
    streaming_mode: str = "NONE",
    max_llm_calls: int | None = None,
    response_modalities: list[str] | None = None,
) -> RunConfig:
    """Construit un ``RunConfig`` ; valide ``streaming_mode`` contre l'enum ``StreamingMode``.

    ``streaming_mode`` est résolu **par nom** (insensible à la casse) :
    ``NONE`` / ``SSE`` / ``BIDI``. Un nom inconnu lève ``ValueError`` (message actionnable).
    ``max_llm_calls=None`` laisse le défaut ADK (500) en place ; un entier est transmis tel
    quel. ``response_modalities`` (ex. ``["TEXT"]``) n'est passé que s'il est fourni.
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
    """Renvoie les noms valides de ``StreamingMode`` (pour les descripteurs/erreurs)."""
    from google.adk.agents.run_config import StreamingMode

    return [m.name for m in StreamingMode]


def _resolve_streaming_mode(streaming_mode: str) -> Any:
    """Résout un nom de mode (insensible à la casse) en membre ``StreamingMode``.

    Lève ``ValueError`` avec la liste des noms valides si le mode est inconnu.
    """
    from google.adk.agents.run_config import StreamingMode

    try:
        return StreamingMode[streaming_mode.strip().upper()]
    except KeyError as exc:
        valid = ", ".join(m.name for m in StreamingMode)
        raise ValueError(
            f"streaming_mode invalide : {streaming_mode!r}. Attendu l'un de : {valid}."
        ) from exc


# --------------------------------------------------------------------------- #
# Sérialisation d'un Event
# --------------------------------------------------------------------------- #
def serialize_event(event: Event) -> dict[str, Any]:
    """Aplati un ``Event`` ADK en dict simple sérialisable JSON.

    Champs : ``author`` ; ``text`` (concaténation des parts textuelles, ``None`` si aucune) ;
    ``function_calls`` (``[{name, args}]``) ; ``function_responses`` (``[{name, response}]``) ;
    ``state_delta`` (``event.actions.state_delta``) ; ``transfer_to_agent`` ; ``is_final``
    (``event.is_final_response()``) ; ``partial``.
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
# Exécution + collecte des événements
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
    """Lance l'agent et collecte tous les ``Event`` produits.

    Garantit d'abord l'existence de la session (la crée si ``get_session`` renvoie ``None`` —
    ``Runner.auto_create_session`` vaut ``False`` par défaut). Construit ``new_message`` comme
    un ``types.Content`` rôle ``"user"`` portant ``new_message_text``, puis itère
    ``run_async``. Si ``progress`` est fourni, il est **awaité** par événement (avec l'index
    1-based et l'événement sérialisé) — utilisé pour la progression SSE de ``run_stream``.

    Renvoie la liste des ``Event`` bruts (l'appelant sérialise via :func:`serialize_event`).
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
