"""Fabrique partagée des services runtime ADK (sessions / memory / artifacts / run).

Les domaines P2 n'écrivent pas de code (contrairement à P1) : ils **instancient de vrais
objets de service ADK** et les appellent (de façon asynchrone). Ce module centralise :

1. La **configuration des backends** (``SessionBackend`` + ``RuntimeConfig``) persistée dans
   ``<app_dir>/.adk_toolkit/runtime.json`` (``load_runtime_config`` / ``save_runtime_config``).
   ``RuntimeConfig`` prévoit déjà des emplacements pour memory et artifacts (P2b les étendra).
2. Un **cache singleton au niveau du process** : ``get_session_service(backend)`` importe
   paresseusement ``google.adk`` et renvoie TOUJOURS la même instance pour une clé de backend
   stable. C'est indispensable pour ``InMemorySessionService``, dont l'état vit en mémoire :
   deux appels d'outils partageant le même backend ``in_memory`` doivent voir le même état.

Aucune dépendance optionnelle n'est importée au chargement du module ; ``sqlalchemy`` (extra
``db``) n'est requis qu'à l'instanciation effective d'un ``DatabaseSessionService``.

Voir ``docs/adk-api-notes/sessions.md`` pour l'API ADK confirmée (services async, mutation
d'état via ``append_event``, pilote async requis pour SQLite : ``sqlite+aiosqlite:///``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from .workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints seulement, import réel paresseux
    from google.adk.sessions import BaseSessionService

#: Fichier de configuration runtime (dans le sidecar ``.adk_toolkit`` de l'app).
RUNTIME_CONFIG_FILE = ".adk_toolkit/runtime.json"

#: Genres de backend de session supportés.
SessionKind = Literal["in_memory", "database", "vertex"]

#: Ensemble des genres valides (validation côté outil).
SESSION_KINDS: frozenset[str] = frozenset({"in_memory", "database", "vertex"})


# --------------------------------------------------------------------------- #
# Configuration des backends
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionBackend:
    """Backend du service de sessions ADK.

    - ``in_memory`` : ``InMemorySessionService`` (état en mémoire process ; mis en cache par
      une clé stable pour survivre entre appels d'outils).
    - ``database`` : ``DatabaseSessionService`` (nécessite l'extra ``db`` = sqlalchemy ;
      ``db_url`` doit utiliser un pilote async, ex. ``sqlite+aiosqlite:///chemin.db``).
    - ``vertex`` : ``VertexAiSessionService`` (nécessite ``project`` et ``location``).

    Gelé (hashable) afin de servir directement de clé de cache.
    """

    kind: SessionKind = "in_memory"
    db_url: str | None = None
    project: str | None = None
    location: str | None = None

    def cache_key(self) -> tuple[str, str | None, str | None, str | None]:
        """Clé stable de cache d'instance (mêmes valeurs → même instance de service)."""
        return (self.kind, self.db_url, self.project, self.location)


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration runtime complète d'une app (sessions + emplacements P2b).

    Seul ``session`` est exploité en P2a. ``memory`` et ``artifacts`` sont réservés (laissés
    à ``None``) pour que P2b les renseigne sans changer le format de fichier ni casser la
    lecture des configs déjà écrites.
    """

    session: SessionBackend = field(default_factory=SessionBackend)
    #: Réservé P2b (MemoryBackend) — gardé opaque pour compat ascendante.
    memory: dict[str, Any] | None = None
    #: Réservé P2b (ArtifactBackend) — gardé opaque pour compat ascendante.
    artifacts: dict[str, Any] | None = None


def _backend_from_dict(data: dict[str, Any] | None) -> SessionBackend:
    """Construit un ``SessionBackend`` depuis un dict JSON (tolérant aux clés inconnues)."""
    if not data:
        return SessionBackend()
    kind = data.get("kind", "in_memory")
    if kind not in SESSION_KINDS:
        kind = "in_memory"
    return SessionBackend(
        kind=kind,
        db_url=data.get("db_url"),
        project=data.get("project"),
        location=data.get("location"),
    )


def load_runtime_config(ws: Workspace, app_name: str) -> RuntimeConfig:
    """Charge la config runtime de l'app, ou renvoie une config par défaut si absente.

    ``app_name`` est accepté pour symétrie avec ``load_model`` (et usage futur) ; la config
    vit dans le sidecar de l'app pointé par ``ws``. Une config corrompue lève ``ValueError``.
    """
    _ = app_name  # symétrie d'API ; le sidecar est déjà résolu par ``ws``.
    if not ws.exists(RUNTIME_CONFIG_FILE):
        return RuntimeConfig()
    try:
        raw = json.loads(ws.read(RUNTIME_CONFIG_FILE))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime.json illisible (JSON invalide) : {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("runtime.json invalide : objet JSON attendu.")
    return RuntimeConfig(
        session=_backend_from_dict(raw.get("session")),
        memory=raw.get("memory"),
        artifacts=raw.get("artifacts"),
    )


def save_runtime_config(ws: Workspace, config: RuntimeConfig) -> bool:
    """Persiste la config runtime (JSON déterministe). Renvoie True si écrit/modifié.

    Idempotent via ``Workspace.write`` (n'écrit pas si le contenu est identique).
    """
    payload = {
        "session": asdict(config.session),
        "memory": config.memory,
        "artifacts": config.artifacts,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return ws.write(RUNTIME_CONFIG_FILE, text)


# --------------------------------------------------------------------------- #
# Cache singleton des services (au niveau du process)
# --------------------------------------------------------------------------- #
#: Cache des instances de service de session, clé = ``SessionBackend.cache_key()``.
_SESSION_SERVICES: dict[tuple[str, str | None, str | None, str | None], Any] = {}

#: Verrou protégeant la création d'instances (création paresseuse thread-safe).
_SESSION_LOCK = Lock()


def _build_session_service(backend: SessionBackend) -> BaseSessionService:
    """Instancie un service de session ADK selon le backend (import paresseux).

    Lève ``ValueError`` pour une config invalide (genre inconnu, champ requis manquant).
    L'``ImportError`` de ``DatabaseSessionService`` (sqlalchemy absent) est convertie en
    ``ValueError`` avec un message orienté action (extra ``db``).
    """
    if backend.kind == "in_memory":
        from google.adk.sessions import InMemorySessionService

        return InMemorySessionService()

    if backend.kind == "database":
        if not backend.db_url:
            raise ValueError("Backend 'database' : 'db_url' est requis.")
        try:
            from google.adk.sessions import DatabaseSessionService
        except ImportError as exc:
            raise ValueError(
                "DatabaseSessionService indisponible : SQLAlchemy manquant. "
                "Installe l'extra 'db' : uv add 'adk-toolkit-mcp[db]' "
                "(ou 'google-adk[db]')."
            ) from exc
        return DatabaseSessionService(db_url=backend.db_url)

    if backend.kind == "vertex":
        if not backend.project or not backend.location:
            raise ValueError("Backend 'vertex' : 'project' et 'location' sont requis.")
        try:
            from google.adk.sessions import VertexAiSessionService
        except ImportError as exc:  # pragma: no cover - dépend de l'extra gcp
            raise ValueError(
                "VertexAiSessionService indisponible : installe l'extra 'gcp' "
                "(uv add 'adk-toolkit-mcp[gcp]')."
            ) from exc
        return VertexAiSessionService(project=backend.project, location=backend.location)

    raise ValueError(
        f"Genre de backend de session inconnu : {backend.kind!r}. "
        f"Attendu l'un de : {', '.join(sorted(SESSION_KINDS))}."
    )


def get_session_service(backend: SessionBackend) -> BaseSessionService:
    """Renvoie l'instance (mise en cache) du service de session pour ``backend``.

    INVARIANT CLÉ : deux appels avec un ``backend`` de même ``cache_key()`` renvoient la
    **même** instance — l'état d'``InMemorySessionService`` survit donc entre appels d'outils
    dans le même process. Les services Database sont aussi cachés par ``db_url``.
    """
    key = backend.cache_key()
    cached = _SESSION_SERVICES.get(key)
    if cached is not None:
        return cached
    with _SESSION_LOCK:
        cached = _SESSION_SERVICES.get(key)
        if cached is None:
            cached = _build_session_service(backend)
            _SESSION_SERVICES[key] = cached
        return cached


def reset_service_cache() -> None:
    """Vide le cache d'instances (réservé aux tests pour l'isolation)."""
    with _SESSION_LOCK:
        _SESSION_SERVICES.clear()
