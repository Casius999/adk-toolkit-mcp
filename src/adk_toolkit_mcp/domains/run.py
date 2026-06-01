"""Domaine `run` : EXÉCUTE un agent ADK via un ``Runner`` (P3a — cœur d'exécution).

Contrairement aux domaines P1 (qui *écrivent* ``agent.py``) et comme les domaines P2 (qui
appellent de vrais services ADK), ce domaine **importe le ``root_agent``** d'une app, le câble
dans un ``Runner`` sur les services session/memory/artifacts configurés (``runtime.json``), et
collecte les ``Event`` produits. Toute la mécanique réutilisable vit dans
:mod:`adk_toolkit_mcp.run_core` (testée hors-ligne via un ``FakeLlm`` — aucune clé requise).

Sous-serveur FastMCP monté sous ``namespace="run"`` → outils exposés ``run_<nom>``.
Fonctions à noms **BARE**. ``agent`` est enregistrée sous le nom d'outil bare ``agent``
(exposé ``run_agent``) ; ``stream`` → ``run_stream`` ; ``live`` → ``run_live`` ;
``config_build`` → ``run_config_build`` ; ``inspect_events`` → ``run_inspect_events``.

Chaque outil renvoie l'enveloppe ``{ok, data, error}`` ; entrées invalides, config corrompue,
import de ``root_agent`` échoué et capacité Live absente renvoient ``err(...)`` (jamais
d'exception qui remonte, jamais de blocage réseau).
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

if TYPE_CHECKING:  # pragma: no cover - hints seulement
    from google.adk.agents import BaseAgent

run_server: FastMCP = FastMCP("run")


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _config_for(path: str, app_name: str) -> RuntimeConfig | dict[str, Any]:
    """Charge la config runtime de l'app, ou renvoie un ``err(...)`` si corrompue.

    Une app sans ``runtime.json`` reçoit la config par défaut (sessions ``in_memory``) — on peut
    donc exécuter un agent sans avoir appelé ``sessions_service_set`` au préalable.
    """
    ws = _app_ws(path, app_name)
    try:
        return load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _prepare(path: str, app_name: str) -> tuple[BaseAgent, RuntimeConfig] | dict[str, Any]:
    """Charge la config et importe ``root_agent`` ; renvoie ``(agent, config)`` ou un ``err``.

    Centralise les deux échecs convertis en ``err`` : config corrompue (``ValueError``) et
    import de ``root_agent`` (``RootAgentImportError`` : fichier absent, module cassé, symbole
    manquant).
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
    """Renvoie le texte du DERNIER événement final (réponse de l'agent), ou ``None``."""
    finals = [s for s in serialized if s["is_final"] and s["text"]]
    return finals[-1]["text"] if finals else None


def _resolve_max_llm_calls(path: str, app_name: str, caller_value: int | None) -> int | None:
    """Résout le plafond effectif d'appels LLM pour un run.

    Précédence : une valeur d'appelant explicite (``caller_value is not None``) **prime toujours**.
    Sinon, on retombe sur la valeur **persistée** par ``safety_settings(..., max_llm_calls=N)`` :
    le ``AgentSpec.max_llm_calls`` de l'agent ROOT du projet (``model.root``), lu dans le sidecar
    ``.adk_toolkit/agents.json`` via :func:`load_model`. Si rien n'est persisté (ou pas de root,
    ou pas de sidecar), on renvoie ``None`` → défaut ADK (500), comme avant.

    Best-effort et non bloquant : un sidecar corrompu (``ValueError``) ne fait PAS échouer le run
    (le domaine ``run`` ne lisait historiquement pas ``agents.json``) — on retombe simplement sur
    ``None``. La config runtime corrompue, elle, reste gérée en amont par ``_config_for``.
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
    """Indique si le modèle de l'agent supporte la connexion Live (``connect`` surchargé).

    Le ``BaseLlm.connect`` de base lève ``NotImplementedError`` ; seul un modèle live-capable
    (ex. ``Gemini``) le surcharge. On compare donc la méthode ``connect`` de la classe du modèle
    résolu à celle de ``BaseLlm``. Toute erreur de résolution → ``False`` (prudence).
    """
    try:
        from google.adk.models import BaseLlm

        model = getattr(agent, "canonical_model", None)
        if model is None:
            return False
        return type(model).connect is not BaseLlm.connect
    except Exception:  # noqa: BLE001 - détection défensive : un échec = pas de Live
        return False


def _has_live_credentials() -> bool:
    """Indique si des identifiants permettant l'API Live sont présents dans l'environnement.

    AI Studio : ``GOOGLE_API_KEY`` (ou ``GEMINI_API_KEY``). Vertex : ``GOOGLE_GENAI_USE_VERTEXAI``
    vrai + ``GOOGLE_CLOUD_PROJECT``. Aucune valeur n'est lue/loggée — seule la présence compte.
    """
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return True
    use_vertex = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    if use_vertex in {"1", "true", "yes"} and os.getenv("GOOGLE_CLOUD_PROJECT"):
        return True
    return False


# --------------------------------------------------------------------------- #
# Outils MCP
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
    """Exécute le ``root_agent`` de l'app sur ``message`` et renvoie les événements + texte final.

    Importe ``root_agent`` (depuis ``<path>/<app_name>/agent.py``), le câble dans un ``Runner``
    sur les services configurés, crée la session si besoin, lance la boucle d'agent, puis renvoie
    la liste des événements **sérialisés** et le texte de la réponse finale.

    ``streaming_mode`` ∈ {``NONE``, ``SSE``, ``BIDI``} (par défaut ``NONE`` : un seul
    ``LlmResponse`` final par tour). ``max_llm_calls`` borne le nombre d'appels LLM : une valeur
    explicite **prime** ; si ``None``, on retombe sur le plafond **persisté** par
    ``safety_settings(..., max_llm_calls=N)`` (``AgentSpec.max_llm_calls`` de l'agent root du
    sidecar) ; à défaut, sur le défaut ADK (500).
    """
    if not user_id.strip():
        return err("user_id est vide.")
    if not session_id.strip():
        return err("session_id est vide.")
    if not message.strip():
        return err("message est vide.")

    prepared = _prepare(path, app_name)
    if isinstance(prepared, dict):
        return prepared
    root_agent, config = prepared

    # Plafond effectif : valeur d'appelant explicite, sinon valeur persistée (root spec).
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
        # streaming_mode invalide OU backend invalide (champ requis manquant / extra gcp absent).
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
    """Comme ``agent`` mais en mode SSE, rapportant la progression par événement via ``ctx``.

    Force ``streaming_mode="SSE"``. Pour chaque événement produit, rapporte la progression au
    client MCP (``ctx.report_progress`` + ``ctx.info``) — utile pour un suivi en temps réel.
    Renvoie les mêmes données que ``agent`` (événements sérialisés + texte final). ``max_llm_calls``
    suit la même précédence que ``agent`` (explicite > plafond persisté du root > défaut ADK 500).
    """
    if not user_id.strip():
        return err("user_id est vide.")
    if not session_id.strip():
        return err("session_id est vide.")
    if not message.strip():
        return err("message est vide.")

    prepared = _prepare(path, app_name)
    if isinstance(prepared, dict):
        return prepared
    root_agent, config = prepared

    async def _progress(index: int, event: dict[str, Any]) -> None:
        """Rapporte un événement au client (no-op silencieux si ``ctx`` absent)."""
        if ctx is None:
            return
        label = event.get("author") or "event"
        await ctx.report_progress(index, message=f"event {index} ({label})")
        await ctx.info(f"[run.stream] event {index}: author={label} final={event['is_final']}")

    # Plafond effectif : valeur d'appelant explicite, sinon valeur persistée (root spec).
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
    """[EXPÉRIMENTAL] Exécution Live/BIDI (Gemini Live API) — nécessite clé + modèle live-capable.

    La voie Live utilise ``BaseLlm.connect`` (websocket vers l'API Gemini Live), PAS
    ``generate_content_async`` : elle exige une vraie clé (``GOOGLE_API_KEY`` ou creds Vertex)
    ET un modèle live-capable, et NE PEUT PAS s'exécuter en CI. Cet outil effectue le câblage
    fidèle (import du ``root_agent``, ``RunConfig`` BIDI) mais **détecte l'absence de capacité**
    et renvoie un ``err`` actionnable AVANT toute connexion — il ne bloque jamais.

    En présence des prérequis, il ouvrirait une ``LiveRequestQueue``, y pousserait ``message``,
    et streamerait les événements de ``runner.run_live(...)``. ``max_llm_calls`` suit la même
    précédence que ``agent`` (explicite > plafond persisté du root > défaut ADK 500).
    """
    if not user_id.strip() or not session_id.strip():
        return err("user_id et session_id sont requis.")
    if not message.strip():
        return err("message est vide.")

    prepared = _prepare(path, app_name)
    if isinstance(prepared, dict):
        return prepared
    root_agent, config = prepared

    # Détection de capacité AVANT toute connexion réseau (sinon l'appel bloquerait/échouerait).
    if not _has_live_credentials():
        return err(
            "run_live requiert l'API Gemini Live : définis GOOGLE_API_KEY (AI Studio) ou "
            "GOOGLE_GENAI_USE_VERTEXAI=TRUE + GOOGLE_CLOUD_PROJECT (Vertex). "
            "Outil expérimental — non exécutable sans clé/websocket (ex. en CI)."
        )
    if not _model_supports_live(root_agent):
        model_name = getattr(getattr(root_agent, "canonical_model", None), "model", "?")
        return err(
            f"Le modèle de l'agent ({model_name!r}) ne supporte pas la connexion Live "
            "(BaseLlm.connect non surchargé). Utilise un modèle Gemini live-capable."
        )

    # Plafond effectif : valeur d'appelant explicite, sinon valeur persistée (root spec).
    resolved_max_llm_calls = _resolve_max_llm_calls(path, app_name, max_llm_calls)

    # Prérequis présents : câblage fidèle de la voie Live (non couvert en CI).
    try:  # pragma: no cover - nécessite une vraie API Live + websocket
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
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - voie Live non testable en CI
        # Tout échec de la voie Live (réseau, modèle, websocket) → err actionnable, jamais de raise.
        return err(f"Échec de l'exécution Live : {exc}")


@run_server.tool(tags={"run"})
def config_build(
    streaming_mode: str = "NONE",
    max_llm_calls: int | None = None,
    response_modalities: list[str] | None = None,
) -> dict[str, Any]:
    """Valide et décrit un ``RunConfig`` (sans exécuter d'agent).

    Renvoie un descripteur ``{streaming_mode, max_llm_calls, response_modalities}`` et la liste
    des modes valides (``streaming_options``). Un ``streaming_mode`` inconnu renvoie ``err``.
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
    """Résume une liste d'événements sérialisés (telle que renvoyée par ``run_agent``).

    Outil PUR (aucune I/O) : compte les function_calls, recense les outils utilisés, les
    transferts d'agents, les clés de state_delta, et extrait le texte final. ``events`` doit être
    une liste de dicts au format de :func:`serialize_event`.
    """
    if not isinstance(events, list):
        return err("events doit être une liste de dicts d'événements sérialisés.")

    tool_names: list[str] = []
    function_call_count = 0
    function_response_count = 0
    transfers: list[str] = []
    state_delta_keys: set[str] = set()
    final_texts: list[str] = []

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            return err(f"events[{index}] n'est pas un dict d'événement sérialisé.")
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

    # Outils uniques en préservant l'ordre de première apparition.
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
