"""Shared factory for ADK runtime services (sessions / memory / artifacts / run).

The P2 domains do not write code (unlike P1): they **instantiate real ADK service objects** and
call them (asynchronously). This module centralizes:

1. The **backend configuration** (``SessionBackend`` + ``RuntimeConfig``) persisted in
   ``<app_dir>/.adk_toolkit/runtime.json`` (``load_runtime_config`` / ``save_runtime_config``).
   ``RuntimeConfig`` already reserves slots for memory and artifacts (P2b extends them).
2. A **process-level singleton cache**: ``get_session_service(backend)`` lazily imports
   ``google.adk`` and ALWAYS returns the same instance for a stable backend key. This is
   essential for ``InMemorySessionService``, whose state lives in memory: two tool calls sharing
   the same ``in_memory`` backend must see the same state.

No optional dependency is imported at module load; ``sqlalchemy`` (extra ``db``) is only
required when a ``DatabaseSessionService`` is actually instantiated.

See ``docs/adk-api-notes/sessions.md`` for the confirmed ADK API (async services, state mutation
via ``append_event``, async driver required for SQLite: ``sqlite+aiosqlite:///``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from .workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints only, real import is lazy
    from google.adk.artifacts import BaseArtifactService
    from google.adk.memory import BaseMemoryService
    from google.adk.sessions import BaseSessionService

#: Runtime configuration file (in the app's ``.adk_toolkit`` sidecar).
RUNTIME_CONFIG_FILE = ".adk_toolkit/runtime.json"

#: Supported session backend kinds.
SessionKind = Literal["in_memory", "database", "vertex"]

#: Set of valid kinds (tool-side validation).
SESSION_KINDS: frozenset[str] = frozenset({"in_memory", "database", "vertex"})

#: Supported memory backend kinds.
MemoryKind = Literal["in_memory", "vertex_rag", "vertex_memory_bank"]

#: Set of valid memory kinds (tool-side validation).
MEMORY_KINDS: frozenset[str] = frozenset({"in_memory", "vertex_rag", "vertex_memory_bank"})

#: Supported artifact backend kinds.
ArtifactKind = Literal["in_memory", "gcs"]

#: Set of valid artifact kinds (tool-side validation).
ARTIFACT_KINDS: frozenset[str] = frozenset({"in_memory", "gcs"})

#: Actionable message shared by services requiring the ``gcp`` extra (Vertex / GCS).
_GCP_EXTRA_HINT = "install the 'gcp' extra (uv add 'adk-toolkit-mcp[gcp]')."


# --------------------------------------------------------------------------- #
# Backend configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionBackend:
    """ADK session service backend.

    - ``in_memory``: ``InMemorySessionService`` (process-memory state; cached by a stable key
      to survive across tool calls).
    - ``database``: ``DatabaseSessionService`` (requires the ``db`` extra = sqlalchemy;
      ``db_url`` must use an async driver, e.g. ``sqlite+aiosqlite:///path.db``).
    - ``vertex``: ``VertexAiSessionService`` (requires ``project`` and ``location``).

    Frozen (hashable) so it can serve directly as a cache key.
    """

    kind: SessionKind = "in_memory"
    db_url: str | None = None
    project: str | None = None
    location: str | None = None

    def cache_key(self) -> tuple[str, str | None, str | None, str | None]:
        """Stable instance cache key (same values → same service instance)."""
        return (self.kind, self.db_url, self.project, self.location)


@dataclass(frozen=True)
class MemoryBackend:
    """ADK MEMORY service backend.

    - ``in_memory``: ``InMemoryMemoryService`` (keyword recall; process-memory state, cached by
      a stable key to survive across tool calls).
    - ``vertex_rag``: ``VertexAiRagMemoryService`` (``gcp`` extra; ``rag_corpus`` required, the
      full RAG corpus name ``projects/…/locations/…/ragCorpora/…``).
    - ``vertex_memory_bank``: ``VertexAiMemoryBankService`` (``gcp`` extra; ``project``,
      ``location`` and ``agent_engine_id`` required).

    Frozen (hashable) so it can serve directly as a cache key.
    """

    kind: MemoryKind = "in_memory"
    project: str | None = None
    location: str | None = None
    rag_corpus: str | None = None
    agent_engine_id: str | None = None

    def cache_key(self) -> tuple[str, str | None, str | None, str | None, str | None]:
        """Stable instance cache key (same values → same service instance)."""
        return (self.kind, self.project, self.location, self.rag_corpus, self.agent_engine_id)


@dataclass(frozen=True)
class ArtifactBackend:
    """ADK ARTIFACTS service backend.

    - ``in_memory``: ``InMemoryArtifactService`` (process-memory state, cached).
    - ``gcs``: ``GcsArtifactService`` (``gcp`` extra; ``bucket`` required).

    Frozen (hashable) so it can serve directly as a cache key.
    """

    kind: ArtifactKind = "in_memory"
    bucket: str | None = None

    def cache_key(self) -> tuple[str, str | None]:
        """Stable instance cache key (same values → same service instance)."""
        return (self.kind, self.bucket)


@dataclass(frozen=True)
class PluginSpec:
    """Manifest of a project plugin (P4c) declared in ``<app_dir>/<app>/plugins.py``.

    ``var`` is the **module-level variable name** holding the plugin instance in ``plugins.py``
    (e.g. ``logging_plugin``). ``name`` (optional) is the plugin's logical name
    (``BasePlugin.name``) and ``kind`` a descriptive label (``logging`` / ``tool_denylist``) —
    both purely informational; only ``var`` is used by ``build_runner`` to fetch the instance
    and pass it to ``Runner`` (via ``App``).

    Frozen (immutable) to stay consistent with the rest of the config.
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
    """Full runtime configuration of an app (sessions + memory + artifacts + plugins).

    In P2a, only ``session`` was used (memory/artifacts were reserved opaque dicts). P2b replaces
    them with real dataclasses. ``memory`` and ``artifacts`` stay ``None`` until a backend has
    been chosen — which preserves backward compatibility with a ``runtime.json`` written by P2a
    (where those fields were ``null``).

    P4c adds ``plugins``: a (possibly empty) tuple of :class:`PluginSpec` listing the
    ``BasePlugin`` plugins declared in ``plugins.py``. Empty -> no plugin (``build_runner``
    unchanged, strict backward compatibility with a ``runtime.json`` without a ``plugins`` key).
    """

    session: SessionBackend = field(default_factory=SessionBackend)
    memory: MemoryBackend | None = None
    artifacts: ArtifactBackend | None = None
    plugins: tuple[PluginSpec, ...] = ()


def _backend_from_dict(data: dict[str, Any] | None) -> SessionBackend:
    """Build a ``SessionBackend`` from a JSON dict (tolerant of unknown keys)."""
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
    """Build a ``MemoryBackend`` from a JSON dict, or ``None`` if absent.

    Tolerant: an unknown kind falls back to ``in_memory`` (consistent with sessions).
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
    """Build an ``ArtifactBackend`` from a JSON dict, or ``None`` if absent.

    Tolerant: an unknown kind falls back to ``in_memory``.
    """
    if not data:
        return None
    kind = data.get("kind", "in_memory")
    if kind not in ARTIFACT_KINDS:
        kind = "in_memory"
    return ArtifactBackend(kind=kind, bucket=data.get("bucket"))


def _plugins_from_list(data: Any) -> tuple[PluginSpec, ...]:
    """Build the ``PluginSpec`` tuple from a JSON list (empty/absent -> ``()``).

    Tolerant: entries without a non-empty ``var`` are ignored (nothing to import).
    """
    if not isinstance(data, list):
        return ()
    specs = [PluginSpec.from_dict(d) for d in data if isinstance(d, dict)]
    return tuple(s for s in specs if s.var)


def load_runtime_config(ws: Workspace, app_name: str) -> RuntimeConfig:
    """Load the app's runtime config, or return a default config if absent.

    ``app_name`` is accepted for symmetry with ``load_model`` (and future use); the config
    lives in the app sidecar pointed to by ``ws``. A corrupt config raises ``ValueError``.
    """
    _ = app_name  # API symmetry; the sidecar is already resolved by ``ws``.
    if not ws.exists(RUNTIME_CONFIG_FILE):
        return RuntimeConfig()
    try:
        raw = json.loads(ws.read(RUNTIME_CONFIG_FILE))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime.json unreadable (invalid JSON): {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("invalid runtime.json: expected a JSON object.")
    return RuntimeConfig(
        session=_backend_from_dict(raw.get("session")),
        memory=_memory_from_dict(raw.get("memory")),
        artifacts=_artifacts_from_dict(raw.get("artifacts")),
        plugins=_plugins_from_list(raw.get("plugins")),
    )


def save_runtime_config(ws: Workspace, config: RuntimeConfig) -> bool:
    """Persist the runtime config (deterministic JSON). Returns True if written/modified.

    Idempotent via ``Workspace.write`` (does not write if the content is identical). Unconfigured
    ``memory`` and ``artifacts`` stay serialized as ``null`` (same format as P2a). The
    ``plugins`` key is emitted ONLY if at least one plugin is declared (strict backward
    compatibility: a ``runtime.json`` without a plugin stays byte-identical to the P2a/P2b
    format).
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
# Singleton service cache (process-level)
# --------------------------------------------------------------------------- #
#: Cache of session service instances, key = ``SessionBackend.cache_key()``.
_SESSION_SERVICES: dict[tuple[str, str | None, str | None, str | None], Any] = {}

#: Lock protecting instance creation (thread-safe lazy creation).
_SESSION_LOCK = Lock()


def _build_session_service(backend: SessionBackend) -> BaseSessionService:
    """Instantiate an ADK session service for the backend (lazy import).

    Raises ``ValueError`` for an invalid config (unknown kind, missing required field). The
    ``ImportError`` of ``DatabaseSessionService`` (sqlalchemy missing) is converted to a
    ``ValueError`` with an actionable message (``db`` extra).
    """
    if backend.kind == "in_memory":
        from google.adk.sessions import InMemorySessionService

        return InMemorySessionService()

    if backend.kind == "database":
        if not backend.db_url:
            raise ValueError("Backend 'database': 'db_url' is required.")
        try:
            from google.adk.sessions import DatabaseSessionService
        except ImportError as exc:
            raise ValueError(
                "DatabaseSessionService unavailable: SQLAlchemy missing. "
                "Install the 'db' extra: uv add 'adk-toolkit-mcp[db]' "
                "(or 'google-adk[db]')."
            ) from exc
        return DatabaseSessionService(db_url=backend.db_url)

    if backend.kind == "vertex":
        if not backend.project or not backend.location:
            raise ValueError("Backend 'vertex': 'project' and 'location' are required.")
        try:
            from google.adk.sessions import VertexAiSessionService
        except ImportError as exc:  # pragma: no cover - depends on the gcp extra
            raise ValueError(
                "VertexAiSessionService unavailable: install the 'gcp' extra "
                "(uv add 'adk-toolkit-mcp[gcp]')."
            ) from exc
        return VertexAiSessionService(project=backend.project, location=backend.location)

    raise ValueError(
        f"Unknown session backend kind: {backend.kind!r}. "
        f"Expected one of: {', '.join(sorted(SESSION_KINDS))}."
    )


def get_session_service(backend: SessionBackend) -> BaseSessionService:
    """Return the (cached) session service instance for ``backend``.

    KEY INVARIANT: two calls with a ``backend`` of the same ``cache_key()`` return the **same**
    instance — so ``InMemorySessionService`` state survives across tool calls within the same
    process. Database services are also cached by ``db_url``.
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
# MEMORY service (same singleton cache scheme)
# --------------------------------------------------------------------------- #
#: Cache of memory service instances, key = ``MemoryBackend.cache_key()``.
_MEMORY_SERVICES: dict[tuple[str, str | None, str | None, str | None, str | None], Any] = {}

#: Lock protecting memory service creation.
_MEMORY_LOCK = Lock()


def _build_memory_service(backend: MemoryBackend) -> BaseMemoryService:
    """Instantiate an ADK memory service for the backend (lazy import).

    Raises ``ValueError`` for an invalid config (unknown kind, missing required field). The
    ``ImportError`` of the Vertex services (``gcp`` extra missing) is converted to an actionable
    ``ValueError``.
    """
    if backend.kind == "in_memory":
        from google.adk.memory import InMemoryMemoryService

        return InMemoryMemoryService()

    if backend.kind == "vertex_rag":
        if not backend.rag_corpus:
            raise ValueError("Backend 'vertex_rag': 'rag_corpus' is required.")
        # NB: the gcp extra's ImportError is raised INSIDE the constructor (lazy import of
        # vertexai), not at class import — so we wrap the construction too.
        try:
            from google.adk.memory import VertexAiRagMemoryService

            return VertexAiRagMemoryService(rag_corpus=backend.rag_corpus)
        except ImportError as exc:  # pragma: no cover - depends on the gcp extra
            raise ValueError(f"VertexAiRagMemoryService unavailable: {_GCP_EXTRA_HINT}") from exc

    if backend.kind == "vertex_memory_bank":
        if not (backend.project and backend.location and backend.agent_engine_id):
            raise ValueError(
                "Backend 'vertex_memory_bank': 'project', 'location' and 'agent_engine_id' "
                "are required."
            )
        try:
            from google.adk.memory import VertexAiMemoryBankService

            return VertexAiMemoryBankService(
                project=backend.project,
                location=backend.location,
                agent_engine_id=backend.agent_engine_id,
            )
        except ImportError as exc:  # pragma: no cover - depends on the gcp extra
            raise ValueError(f"VertexAiMemoryBankService unavailable: {_GCP_EXTRA_HINT}") from exc

    raise ValueError(
        f"Unknown memory backend kind: {backend.kind!r}. "
        f"Expected one of: {', '.join(sorted(MEMORY_KINDS))}."
    )


def get_memory_service(backend: MemoryBackend) -> BaseMemoryService:
    """Return the (cached) memory service instance for ``backend``.

    Same invariant as ``get_session_service``: an ``in_memory`` backend of the same
    ``cache_key()`` ALWAYS returns the same instance (memory state survives across calls).
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
# ARTIFACTS service (same singleton cache scheme)
# --------------------------------------------------------------------------- #
#: Cache of artifact service instances, key = ``ArtifactBackend.cache_key()``.
_ARTIFACT_SERVICES: dict[tuple[str, str | None], Any] = {}

#: Lock protecting artifact service creation.
_ARTIFACT_LOCK = Lock()


def _build_artifact_service(backend: ArtifactBackend) -> BaseArtifactService:
    """Instantiate an ADK artifact service for the backend (lazy import).

    Raises ``ValueError`` for an invalid config. The ``ImportError`` of ``GcsArtifactService``
    (``gcp`` extra missing) is converted to an actionable ``ValueError``.
    """
    if backend.kind == "in_memory":
        from google.adk.artifacts import InMemoryArtifactService

        return InMemoryArtifactService()

    if backend.kind == "gcs":
        if not backend.bucket:
            raise ValueError("Backend 'gcs': 'bucket' is required.")
        # GcsArtifactService imports google.cloud.storage INSIDE __init__: a ModuleNotFoundError
        # (an ImportError subclass) occurs at construction if the gcp extra is missing — so we
        # wrap the import AND the construction in the same try.
        try:
            from google.adk.artifacts import GcsArtifactService

            return GcsArtifactService(bucket_name=backend.bucket)
        except ImportError as exc:  # pragma: no cover - depends on the gcp extra
            raise ValueError(f"GcsArtifactService unavailable: {_GCP_EXTRA_HINT}") from exc

    raise ValueError(
        f"Unknown artifact backend kind: {backend.kind!r}. "
        f"Expected one of: {', '.join(sorted(ARTIFACT_KINDS))}."
    )


def get_artifact_service(backend: ArtifactBackend) -> BaseArtifactService:
    """Return the (cached) artifact service instance for ``backend``.

    Same invariant as ``get_session_service``: an ``in_memory`` backend of the same
    ``cache_key()`` ALWAYS returns the same instance.
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
    """Clear ALL instance caches (sessions + memory + artifacts).

    Reserved for tests, for isolation: after calling, the next ``get_*_service`` recreates a
    fresh instance (useful to prove a persistence that does not depend on in-memory state).
    """
    with _SESSION_LOCK:
        _SESSION_SERVICES.clear()
    with _MEMORY_LOCK:
        _MEMORY_SERVICES.clear()
    with _ARTIFACT_LOCK:
        _ARTIFACT_SERVICES.clear()
