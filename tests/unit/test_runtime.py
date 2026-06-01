"""Tests unitaires de la fabrique runtime partagée (``adk_toolkit_mcp.runtime``).

Couvre :
- Round-trip de la config runtime (persist → load), y compris backend ``database``.
- Tolérance : config absente → défaut ``in_memory`` ; JSON corrompu → ``ValueError``.
- Idempotence de ``save_runtime_config``.
- INVARIANT singleton : même backend ``in_memory`` → MÊME instance de service ;
  backends différents → instances différentes ; service ``database`` caché par ``db_url``.
- ``reset_service_cache`` casse bien le cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adk_toolkit_mcp.runtime import (
    RuntimeConfig,
    SessionBackend,
    get_session_service,
    load_runtime_config,
    reset_service_cache,
    save_runtime_config,
)
from adk_toolkit_mcp.workspace import Workspace


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole chaque test : le cache d'instances de service est vidé avant/après."""
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


def test_reserved_memory_artifacts_round_trip(tmp_path: Path) -> None:
    """Les emplacements P2b (memory/artifacts) sont préservés tels quels."""
    ws = Workspace(tmp_path)
    config = RuntimeConfig(
        session=SessionBackend(),
        memory={"kind": "in_memory"},
        artifacts={"kind": "in_memory"},
    )
    save_runtime_config(ws, config)
    loaded = load_runtime_config(ws, "myapp")
    assert loaded.memory == {"kind": "in_memory"}
    assert loaded.artifacts == {"kind": "in_memory"}


# --------------------------------------------------------------------------- #
# Singleton invariant
# --------------------------------------------------------------------------- #
def test_in_memory_singleton_identity() -> None:
    """Même backend in_memory → MÊME instance (l'état survit entre appels)."""
    backend = SessionBackend(kind="in_memory")
    svc_a = get_session_service(backend)
    svc_b = get_session_service(SessionBackend(kind="in_memory"))
    assert svc_a is svc_b


def test_in_memory_singleton_survives_state(tmp_path: Path) -> None:
    """Prouve concrètement que l'instance partagée conserve l'état (create puis get)."""
    import asyncio

    backend = SessionBackend(kind="in_memory")

    async def scenario() -> None:
        svc1 = get_session_service(backend)
        created = await svc1.create_session(app_name="app", user_id="u1")
        # Un autre appel d'outil récupérerait le service via la même clé.
        svc2 = get_session_service(SessionBackend(kind="in_memory"))
        fetched = await svc2.get_session(app_name="app", user_id="u1", session_id=created.id)
        assert fetched is not None
        assert fetched.id == created.id

    asyncio.run(scenario())


def test_database_service_keyed_by_url(tmp_path: Path) -> None:
    """Service database mis en cache par db_url : même url → même instance."""
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
