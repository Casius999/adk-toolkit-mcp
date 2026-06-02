"""Unit tests for the shared runtime factory (``adk_toolkit_mcp.runtime``).

Covers:
- Round-trip of the runtime config (persist → load), including the ``database`` backend.
- Tolerance: absent config → ``in_memory`` default; corrupt JSON → ``ValueError``.
- Idempotence of ``save_runtime_config``.
- Singleton INVARIANT: the same ``in_memory`` backend → the SAME service instance;
  different backends → different instances; ``database`` service cached by ``db_url``.
- ``reset_service_cache`` properly breaks the cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adk_toolkit_mcp.runtime import (
    RUNTIME_CONFIG_FILE,
    ArtifactBackend,
    MemoryBackend,
    PluginSpec,
    RuntimeConfig,
    SessionBackend,
    get_artifact_service,
    get_memory_service,
    get_session_service,
    load_runtime_config,
    reset_service_cache,
    save_runtime_config,
)
from adk_toolkit_mcp.workspace import Workspace


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isolate each test: the service-instance cache is cleared before/after."""
    reset_service_cache()
    yield
    reset_service_cache()


# --------------------------------------------------------------------------- #
# Config persist/load round-trip
# --------------------------------------------------------------------------- #
def test_load_returns_default_when_absent(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    config = load_runtime_config(ws, "myapp")
    assert config.session.kind == "in_memory"
    assert config.session.db_url is None
    assert config.memory is None
    assert config.artifacts is None


def test_save_then_load_in_memory_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    config = RuntimeConfig(session=SessionBackend(kind="in_memory"))
    assert save_runtime_config(ws, config) is True

    loaded = load_runtime_config(ws, "myapp")
    assert loaded.session.kind == "in_memory"


def test_save_then_load_database_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    backend = SessionBackend(kind="database", db_url="sqlite+aiosqlite:///x.db")
    save_runtime_config(ws, RuntimeConfig(session=backend))

    loaded = load_runtime_config(ws, "myapp")
    assert loaded.session.kind == "database"
    assert loaded.session.db_url == "sqlite+aiosqlite:///x.db"


def test_save_then_load_vertex_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    backend = SessionBackend(kind="vertex", project="p", location="us-central1")
    save_runtime_config(ws, RuntimeConfig(session=backend))

    loaded = load_runtime_config(ws, "myapp")
    assert loaded.session.kind == "vertex"
    assert loaded.session.project == "p"
    assert loaded.session.location == "us-central1"


def test_save_is_idempotent(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    config = RuntimeConfig(
        session=SessionBackend(kind="database", db_url="sqlite+aiosqlite:///x.db")
    )
    assert save_runtime_config(ws, config) is True
    # Second write with identical content -> no change.
    assert save_runtime_config(ws, config) is False


def test_corrupt_runtime_json_raises_value_error(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(".adk_toolkit/runtime.json", "{ not json")
    with pytest.raises(ValueError):
        load_runtime_config(ws, "myapp")


def test_non_object_runtime_json_raises_value_error(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(".adk_toolkit/runtime.json", "[1, 2, 3]")
    with pytest.raises(ValueError):
        load_runtime_config(ws, "myapp")


def test_unknown_kind_in_file_falls_back_to_in_memory(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(".adk_toolkit/runtime.json", '{"session": {"kind": "bogus"}}')
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.session.kind == "in_memory"


def test_memory_artifacts_default_none(tmp_path: Path) -> None:
    """Without a chosen backend, memory/artifacts stay ``None`` (serialized as ``null``)."""
    ws = Workspace(tmp_path)
    save_runtime_config(ws, RuntimeConfig(session=SessionBackend()))
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.memory is None
    assert loaded.artifacts is None


def test_memory_in_memory_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    config = RuntimeConfig(session=SessionBackend(), memory=MemoryBackend(kind="in_memory"))
    save_runtime_config(ws, config)
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.memory == MemoryBackend(kind="in_memory")


def test_memory_vertex_rag_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    backend = MemoryBackend(kind="vertex_rag", rag_corpus="projects/p/locations/us/ragCorpora/1")
    save_runtime_config(ws, RuntimeConfig(session=SessionBackend(), memory=backend))
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.memory is not None
    assert loaded.memory.kind == "vertex_rag"
    assert loaded.memory.rag_corpus == "projects/p/locations/us/ragCorpora/1"


def test_memory_vertex_memory_bank_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    backend = MemoryBackend(
        kind="vertex_memory_bank", project="p", location="us-central1", agent_engine_id="123"
    )
    save_runtime_config(ws, RuntimeConfig(session=SessionBackend(), memory=backend))
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.memory is not None
    assert loaded.memory.kind == "vertex_memory_bank"
    assert loaded.memory.project == "p"
    assert loaded.memory.location == "us-central1"
    assert loaded.memory.agent_engine_id == "123"


def test_artifacts_in_memory_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    config = RuntimeConfig(session=SessionBackend(), artifacts=ArtifactBackend(kind="in_memory"))
    save_runtime_config(ws, config)
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.artifacts == ArtifactBackend(kind="in_memory")


def test_artifacts_gcs_round_trip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    backend = ArtifactBackend(kind="gcs", bucket="my-bucket")
    save_runtime_config(ws, RuntimeConfig(session=SessionBackend(), artifacts=backend))
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.artifacts is not None
    assert loaded.artifacts.kind == "gcs"
    assert loaded.artifacts.bucket == "my-bucket"


def test_full_config_round_trip(tmp_path: Path) -> None:
    """Session + memory + artifacts together survive the round-trip."""
    ws = Workspace(tmp_path)
    config = RuntimeConfig(
        session=SessionBackend(kind="database", db_url="sqlite+aiosqlite:///x.db"),
        memory=MemoryBackend(kind="in_memory"),
        artifacts=ArtifactBackend(kind="in_memory"),
    )
    save_runtime_config(ws, config)
    loaded = load_runtime_config(ws, "myapp")
    assert loaded == config


def test_backward_compat_with_p2a_runtime_json(tmp_path: Path) -> None:
    """A ``runtime.json`` written by P2a (memory/artifacts = null) loads cleanly.

    P2a serialized memory/artifacts as ``null``; they must load as ``None`` without error, and the
    session must stay intact.
    """
    ws = Workspace(tmp_path)
    ws.write(
        ".adk_toolkit/runtime.json",
        '{"session": {"kind": "in_memory", "db_url": null, "project": null, '
        '"location": null}, "memory": null, "artifacts": null}\n',
    )
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.session.kind == "in_memory"
    assert loaded.memory is None
    assert loaded.artifacts is None


def test_unknown_memory_kind_in_file_falls_back(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(".adk_toolkit/runtime.json", '{"memory": {"kind": "bogus"}}')
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.memory is not None
    assert loaded.memory.kind == "in_memory"


def test_unknown_artifact_kind_in_file_falls_back(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    ws.write(".adk_toolkit/runtime.json", '{"artifacts": {"kind": "bogus"}}')
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.artifacts is not None
    assert loaded.artifacts.kind == "in_memory"


# --------------------------------------------------------------------------- #
# Singleton invariant
# --------------------------------------------------------------------------- #
def test_in_memory_singleton_identity() -> None:
    """Same in_memory backend → SAME instance (state survives across calls)."""
    backend = SessionBackend(kind="in_memory")
    svc_a = get_session_service(backend)
    svc_b = get_session_service(SessionBackend(kind="in_memory"))
    assert svc_a is svc_b


def test_in_memory_singleton_survives_state(tmp_path: Path) -> None:
    """Concretely prove that the shared instance keeps the state (create then get)."""
    import asyncio

    backend = SessionBackend(kind="in_memory")

    async def scenario() -> None:
        svc1 = get_session_service(backend)
        created = await svc1.create_session(app_name="app", user_id="u1")
        # Another tool call would fetch the service via the same key.
        svc2 = get_session_service(SessionBackend(kind="in_memory"))
        fetched = await svc2.get_session(app_name="app", user_id="u1", session_id=created.id)
        assert fetched is not None
        assert fetched.id == created.id

    asyncio.run(scenario())


def test_database_service_keyed_by_url(tmp_path: Path) -> None:
    """Database service cached by db_url: same url → same instance."""
    url = f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}"
    svc_a = get_session_service(SessionBackend(kind="database", db_url=url))
    svc_b = get_session_service(SessionBackend(kind="database", db_url=url))
    assert svc_a is svc_b


def test_database_service_different_url_different_instance(tmp_path: Path) -> None:
    url1 = f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}"
    url2 = f"sqlite+aiosqlite:///{(tmp_path / 'b.db').as_posix()}"
    svc_a = get_session_service(SessionBackend(kind="database", db_url=url1))
    svc_b = get_session_service(SessionBackend(kind="database", db_url=url2))
    assert svc_a is not svc_b


def test_in_memory_and_database_are_distinct(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'a.db').as_posix()}"
    in_mem = get_session_service(SessionBackend(kind="in_memory"))
    db = get_session_service(SessionBackend(kind="database", db_url=url))
    assert in_mem is not db


def test_database_without_url_raises() -> None:
    with pytest.raises(ValueError, match="db_url"):
        get_session_service(SessionBackend(kind="database", db_url=None))


def test_vertex_without_project_raises() -> None:
    with pytest.raises(ValueError, match="project"):
        get_session_service(SessionBackend(kind="vertex"))


def test_reset_service_cache_invalidates() -> None:
    backend = SessionBackend(kind="in_memory")
    svc_a = get_session_service(backend)
    reset_service_cache()
    svc_b = get_session_service(backend)
    assert svc_a is not svc_b


# --------------------------------------------------------------------------- #
# Memory / artifact singleton invariants + config validation
# --------------------------------------------------------------------------- #
def test_memory_in_memory_singleton_identity() -> None:
    """Same in_memory memory backend → SAME instance (state shared across calls)."""
    svc_a = get_memory_service(MemoryBackend(kind="in_memory"))
    svc_b = get_memory_service(MemoryBackend(kind="in_memory"))
    assert svc_a is svc_b


def test_artifact_in_memory_singleton_identity() -> None:
    svc_a = get_artifact_service(ArtifactBackend(kind="in_memory"))
    svc_b = get_artifact_service(ArtifactBackend(kind="in_memory"))
    assert svc_a is svc_b


def test_memory_and_artifact_caches_are_independent() -> None:
    """The caches are distinct: a memory service is not an artifact service."""
    mem = get_memory_service(MemoryBackend(kind="in_memory"))
    art = get_artifact_service(ArtifactBackend(kind="in_memory"))
    assert mem is not art


def test_reset_clears_memory_and_artifact_caches() -> None:
    mem_a = get_memory_service(MemoryBackend(kind="in_memory"))
    art_a = get_artifact_service(ArtifactBackend(kind="in_memory"))
    reset_service_cache()
    assert get_memory_service(MemoryBackend(kind="in_memory")) is not mem_a
    assert get_artifact_service(ArtifactBackend(kind="in_memory")) is not art_a


def test_memory_vertex_rag_without_corpus_raises() -> None:
    with pytest.raises(ValueError, match="rag_corpus"):
        get_memory_service(MemoryBackend(kind="vertex_rag"))


def test_memory_vertex_memory_bank_without_fields_raises() -> None:
    with pytest.raises(ValueError, match="agent_engine_id"):
        get_memory_service(MemoryBackend(kind="vertex_memory_bank", project="p"))


def test_artifact_gcs_without_bucket_raises() -> None:
    with pytest.raises(ValueError, match="bucket"):
        get_artifact_service(ArtifactBackend(kind="gcs"))


# --------------------------------------------------------------------------- #
# Plugins manifest (P4c) — serialization + backward compatibility
# --------------------------------------------------------------------------- #
def test_runtime_plugins_roundtrip(tmp_path: Path) -> None:
    """The plugins manifest is persisted then re-read (var/name/kind)."""
    ws = Workspace(tmp_path)
    config = RuntimeConfig(
        session=SessionBackend(kind="in_memory"),
        plugins=(
            PluginSpec(var="logging_plugin", name="logging_plugin", kind="logging"),
            PluginSpec(var="guard", name="guard", kind="tool_denylist"),
        ),
    )
    save_runtime_config(ws, config)
    loaded = load_runtime_config(ws, "app")
    assert [(p.var, p.kind) for p in loaded.plugins] == [
        ("logging_plugin", "logging"),
        ("guard", "tool_denylist"),
    ]


def test_runtime_no_plugins_key_when_empty(tmp_path: Path) -> None:
    """Without a plugin, the 'plugins' key is NOT emitted (runtime.json stays P2a/P2b compat)."""
    ws = Workspace(tmp_path)
    save_runtime_config(ws, RuntimeConfig(session=SessionBackend(kind="in_memory")))
    raw = ws.read(RUNTIME_CONFIG_FILE)
    assert "plugins" not in raw
    # And a file without a 'plugins' key re-reads with plugins == () (no error).
    assert load_runtime_config(ws, "app").plugins == ()


def test_runtime_plugins_ignores_entries_without_var(tmp_path: Path) -> None:
    """A manifest entry without a non-empty 'var' is ignored (tolerance)."""
    import json

    ws = Workspace(tmp_path)
    payload = {
        "session": {"kind": "in_memory"},
        "memory": None,
        "artifacts": None,
        "plugins": [{"var": "ok", "kind": "logging"}, {"kind": "logging"}, {"var": ""}],
    }
    ws.write(RUNTIME_CONFIG_FILE, json.dumps(payload))
    loaded = load_runtime_config(ws, "app")
    assert [p.var for p in loaded.plugins] == ["ok"]
