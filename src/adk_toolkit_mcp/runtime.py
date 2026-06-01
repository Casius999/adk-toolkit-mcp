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
    from google.adk.artifacts import BaseArtifactService
    from google.adk.memory import BaseMemoryService
    from google.adk.sessions import BaseSessionService

#: Fichier de configuration runtime (dans le sidecar ``.adk_toolkit`` de l'app).
RUNTIME_CONFIG_FILE = ".adk_toolkit/runtime.json"

#: Genres de backend de session supportés.
SessionKind = Literal["in_memory", "database", "vertex"]

#: Ensemble des genres valides (validation côté outil).
SESSION_KINDS: frozenset[str] = frozenset({"in_memory", "database", "vertex"})

#: Genres de backend de mémoire supportés.
MemoryKind = Literal["in_memory", "vertex_rag", "vertex_memory_bank"]

#: Ensemble des genres de mémoire valides (validation côté outil).
MEMORY_KINDS: frozenset[str] = frozenset({"in_memory", "vertex_rag", "vertex_memory_bank"})

#: Genres de backend d'artifacts supportés.
ArtifactKind = Literal["in_memory", "gcs"]

#: Ensemble des genres d'artifacts valides (validation côté outil).
ARTIFACT_KINDS: frozenset[str] = frozenset({"in_memory", "gcs"})

#: Message orienté action commun aux services nécessitant l'extra ``gcp`` (Vertex / GCS).
_GCP_EXTRA_HINT = "installe l'extra 'gcp' (uv add 'adk-toolkit-mcp[gcp]')."


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
class MemoryBackend:
    """Backend du service de MÉMOIRE ADK.

    - ``in_memory`` : ``InMemoryMemoryService`` (rappel par mots-clés ; état en mémoire
      process, mis en cache par clé stable pour survivre entre appels d'outils).
    - ``vertex_rag`` : ``VertexAiRagMemoryService`` (extra ``gcp`` ; ``rag_corpus`` requis,
      nom de corpus RAG complet ``projects/…/locations/…/ragCorpora/…``).
    - ``vertex_memory_bank`` : ``VertexAiMemoryBankService`` (extra ``gcp`` ; ``project``,
      ``location`` et ``agent_engine_id`` requis).

    Gelé (hashable) pour servir directement de clé de cache.
    """

    kind: MemoryKind = "in_memory"
    project: str | None = None
    location: str | None = None
    rag_corpus: str | None = None
    agent_engine_id: str | None = None

    def cache_key(self) -> tuple[str, str | None, str | None, str | None, str | None]:
        """Clé stable de cache d'instance (mêmes valeurs → même instance de service)."""
        return (self.kind, self.project, self.location, self.rag_corpus, self.agent_engine_id)


@dataclass(frozen=True)
class ArtifactBackend:
    """Backend du service d'ARTIFACTS ADK.

    - ``in_memory`` : ``InMemoryArtifactService`` (état en mémoire process, mis en cache).
    - ``gcs`` : ``GcsArtifactService`` (extra ``gcp`` ; ``bucket`` requis).

    Gelé (hashable) pour servir directement de clé de cache.
    """

    kind: ArtifactKind = "in_memory"
    bucket: str | None = None

    def cache_key(self) -> tuple[str, str | None]:
        """Clé stable de cache d'instance (mêmes valeurs → même instance de service)."""
        return (self.kind, self.bucket)


@dataclass(frozen=True)
class PluginSpec:
    """Manifeste d'un plugin de projet (P4c) déclaré dans ``<app_dir>/<app>/plugins.py``.

    ``var`` est le **nom de la variable module-level** portant l'instance de plugin dans
    ``plugins.py`` (ex. ``logging_plugin``). ``name`` (optionnel) est le nom logique du plugin
    (``BasePlugin.name``) et ``kind`` une étiquette descriptive (``logging`` / ``tool_denylist``)
    — tous deux purement informatifs ; seul ``var`` est utilisé par ``build_runner`` pour
    récupérer l'instance et la passer à ``Runner`` (via ``App``).

    Gelé (immuable) pour rester cohérent avec le reste de la config.
    """

    var: str
    name: str = ""
    kind: str = ""

    def to_dict(self) -> dict[str, str]:
        d: dict[str, str] = {"var": self.var}
        if self.name:
            d["name"] = self.name
        if self.kind:
            d["kind"] = self.kind
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginSpec:
        return cls(
            var=str(data.get("var", "")),
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration runtime complète d'une app (sessions + memory + artifacts + plugins).

    En P2a, seul ``session`` était exploité (memory/artifacts étaient des dicts opaques
    réservés). P2b les remplace par de vrais dataclasses. ``memory`` et ``artifacts`` restent
    ``None`` tant qu'aucun backend n'a été choisi — ce qui préserve la compat ascendante avec
    une ``runtime.json`` écrite par P2a (où ces champs valaient ``null``).

    P4c ajoute ``plugins`` : un tuple (éventuellement vide) de :class:`PluginSpec` listant les
    plugins ``BasePlugin`` déclarés dans ``plugins.py``. Vide -> aucun plugin (``build_runner``
    inchangé, compat ascendante stricte avec une ``runtime.json`` sans clé ``plugins``).
    """

    session: SessionBackend = field(default_factory=SessionBackend)
    memory: MemoryBackend | None = None
    artifacts: ArtifactBackend | None = None
    plugins: tuple[PluginSpec, ...] = ()


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


def _memory_from_dict(data: dict[str, Any] | None) -> MemoryBackend | None:
    """Construit un ``MemoryBackend`` depuis un dict JSON, ou ``None`` si absent.

    Tolérant : un genre inconnu retombe sur ``in_memory`` (cohérent avec les sessions).
    """
    if not data:
        return None
    kind = data.get("kind", "in_memory")
    if kind not in MEMORY_KINDS:
        kind = "in_memory"
    return MemoryBackend(
        kind=kind,
        project=data.get("project"),
        location=data.get("location"),
        rag_corpus=data.get("rag_corpus"),
        agent_engine_id=data.get("agent_engine_id"),
    )


def _artifacts_from_dict(data: dict[str, Any] | None) -> ArtifactBackend | None:
    """Construit un ``ArtifactBackend`` depuis un dict JSON, ou ``None`` si absent.

    Tolérant : un genre inconnu retombe sur ``in_memory``.
    """
    if not data:
        return None
    kind = data.get("kind", "in_memory")
    if kind not in ARTIFACT_KINDS:
        kind = "in_memory"
    return ArtifactBackend(kind=kind, bucket=data.get("bucket"))


def _plugins_from_list(data: Any) -> tuple[PluginSpec, ...]:
    """Construit le tuple de ``PluginSpec`` depuis une liste JSON (vide/absente -> ``()``).

    Tolérant : les entrées sans ``var`` non vide sont ignorées (rien à importer).
    """
    if not isinstance(data, list):
        return ()
    specs = [PluginSpec.from_dict(d) for d in data if isinstance(d, dict)]
    return tuple(s for s in specs if s.var)


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
        memory=_memory_from_dict(raw.get("memory")),
        artifacts=_artifacts_from_dict(raw.get("artifacts")),
        plugins=_plugins_from_list(raw.get("plugins")),
    )


def save_runtime_config(ws: Workspace, config: RuntimeConfig) -> bool:
    """Persiste la config runtime (JSON déterministe). Renvoie True si écrit/modifié.

    Idempotent via ``Workspace.write`` (n'écrit pas si le contenu est identique). ``memory`` et
    ``artifacts`` non configurés restent sérialisés à ``null`` (format identique à P2a). La clé
    ``plugins`` n'est émise QUE si au moins un plugin est déclaré (compat ascendante stricte :
    une ``runtime.json`` sans plugin reste byte-identique au format P2a/P2b).
    """
    payload: dict[str, Any] = {
        "session": asdict(config.session),
        "memory": asdict(config.memory) if config.memory is not None else None,
        "artifacts": asdict(config.artifacts) if config.artifacts is not None else None,
    }
    if config.plugins:
        payload["plugins"] = [p.to_dict() for p in config.plugins]
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


# --------------------------------------------------------------------------- #
# Service de MÉMOIRE (même schéma de cache singleton)
# --------------------------------------------------------------------------- #
#: Cache des instances de service de mémoire, clé = ``MemoryBackend.cache_key()``.
_MEMORY_SERVICES: dict[tuple[str, str | None, str | None, str | None, str | None], Any] = {}

#: Verrou protégeant la création des services de mémoire.
_MEMORY_LOCK = Lock()


def _build_memory_service(backend: MemoryBackend) -> BaseMemoryService:
    """Instancie un service de mémoire ADK selon le backend (import paresseux).

    Lève ``ValueError`` pour une config invalide (genre inconnu, champ requis manquant).
    L'``ImportError`` des services Vertex (extra ``gcp`` absent) est convertie en ``ValueError``
    orienté action.
    """
    if backend.kind == "in_memory":
        from google.adk.memory import InMemoryMemoryService

        return InMemoryMemoryService()

    if backend.kind == "vertex_rag":
        if not backend.rag_corpus:
            raise ValueError("Backend 'vertex_rag' : 'rag_corpus' est requis.")
        # NB : l'ImportError de l'extra gcp est levée DANS le constructeur (import paresseux de
        # vertexai), pas à l'import de la classe — on enveloppe donc aussi la construction.
        try:
            from google.adk.memory import VertexAiRagMemoryService

            return VertexAiRagMemoryService(rag_corpus=backend.rag_corpus)
        except ImportError as exc:  # pragma: no cover - dépend de l'extra gcp
            raise ValueError(f"VertexAiRagMemoryService indisponible : {_GCP_EXTRA_HINT}") from exc

    if backend.kind == "vertex_memory_bank":
        if not (backend.project and backend.location and backend.agent_engine_id):
            raise ValueError(
                "Backend 'vertex_memory_bank' : 'project', 'location' et 'agent_engine_id' "
                "sont requis."
            )
        try:
            from google.adk.memory import VertexAiMemoryBankService

            return VertexAiMemoryBankService(
                project=backend.project,
                location=backend.location,
                agent_engine_id=backend.agent_engine_id,
            )
        except ImportError as exc:  # pragma: no cover - dépend de l'extra gcp
            raise ValueError(f"VertexAiMemoryBankService indisponible : {_GCP_EXTRA_HINT}") from exc

    raise ValueError(
        f"Genre de backend de mémoire inconnu : {backend.kind!r}. "
        f"Attendu l'un de : {', '.join(sorted(MEMORY_KINDS))}."
    )


def get_memory_service(backend: MemoryBackend) -> BaseMemoryService:
    """Renvoie l'instance (mise en cache) du service de mémoire pour ``backend``.

    Même invariant que ``get_session_service`` : un backend ``in_memory`` de même
    ``cache_key()`` renvoie TOUJOURS la même instance (l'état mémoire survit entre appels).
    """
    key = backend.cache_key()
    cached = _MEMORY_SERVICES.get(key)
    if cached is not None:
        return cached
    with _MEMORY_LOCK:
        cached = _MEMORY_SERVICES.get(key)
        if cached is None:
            cached = _build_memory_service(backend)
            _MEMORY_SERVICES[key] = cached
        return cached


# --------------------------------------------------------------------------- #
# Service d'ARTIFACTS (même schéma de cache singleton)
# --------------------------------------------------------------------------- #
#: Cache des instances de service d'artifacts, clé = ``ArtifactBackend.cache_key()``.
_ARTIFACT_SERVICES: dict[tuple[str, str | None], Any] = {}

#: Verrou protégeant la création des services d'artifacts.
_ARTIFACT_LOCK = Lock()


def _build_artifact_service(backend: ArtifactBackend) -> BaseArtifactService:
    """Instancie un service d'artifacts ADK selon le backend (import paresseux).

    Lève ``ValueError`` pour une config invalide. L'``ImportError`` de ``GcsArtifactService``
    (extra ``gcp`` absent) est convertie en ``ValueError`` orienté action.
    """
    if backend.kind == "in_memory":
        from google.adk.artifacts import InMemoryArtifactService

        return InMemoryArtifactService()

    if backend.kind == "gcs":
        if not backend.bucket:
            raise ValueError("Backend 'gcs' : 'bucket' est requis.")
        # GcsArtifactService importe google.cloud.storage DANS __init__ : un ModuleNotFoundError
        # (sous-classe d'ImportError) survient à la construction si l'extra gcp est absent — on
        # enveloppe donc l'import ET la construction dans le même try.
        try:
            from google.adk.artifacts import GcsArtifactService

            return GcsArtifactService(bucket_name=backend.bucket)
        except ImportError as exc:  # pragma: no cover - dépend de l'extra gcp
            raise ValueError(f"GcsArtifactService indisponible : {_GCP_EXTRA_HINT}") from exc

    raise ValueError(
        f"Genre de backend d'artifacts inconnu : {backend.kind!r}. "
        f"Attendu l'un de : {', '.join(sorted(ARTIFACT_KINDS))}."
    )


def get_artifact_service(backend: ArtifactBackend) -> BaseArtifactService:
    """Renvoie l'instance (mise en cache) du service d'artifacts pour ``backend``.

    Même invariant que ``get_session_service`` : un backend ``in_memory`` de même
    ``cache_key()`` renvoie TOUJOURS la même instance.
    """
    key = backend.cache_key()
    cached = _ARTIFACT_SERVICES.get(key)
    if cached is not None:
        return cached
    with _ARTIFACT_LOCK:
        cached = _ARTIFACT_SERVICES.get(key)
        if cached is None:
            cached = _build_artifact_service(backend)
            _ARTIFACT_SERVICES[key] = cached
        return cached


def reset_service_cache() -> None:
    """Vide TOUS les caches d'instances (sessions + memory + artifacts).

    Réservé aux tests pour l'isolation : après appel, le prochain ``get_*_service`` recrée une
    instance neuve (utile pour prouver une persistance qui ne dépend pas de l'état en mémoire).
    """
    with _SESSION_LOCK:
        _SESSION_SERVICES.clear()
    with _MEMORY_LOCK:
        _MEMORY_SERVICES.clear()
    with _ARTIFACT_LOCK:
        _ARTIFACT_SERVICES.clear()
