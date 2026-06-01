"""Tests unitaires du domaine ``sessions`` (P2a — services runtime ADK).

Les outils sont **async** (``asyncio_mode=auto`` dans pyproject). On appelle les fonctions
bare directement (le décorateur ``@tool`` de cette version de FastMCP renvoie la fonction
d'origine) et, pour le read-through, via un ``fastmcp.Client`` in-memory.

Couverture clé :
- ``service_set`` : persiste le backend ; validations (kind, db_url, vertex).
- ``create`` → ``get`` round-trip ; ``list`` / ``delete`` ; ``append_event`` incrémente.
- ``state_set`` puis ``state_get`` pour CHAQUE scope (session/app/user/temp).
- Correction des préfixes : app/user/temp stockés sous ``app:``/``user:``/``temp:``.
- PERSISTANCE FONCTIONNELLE avec un backend ``database`` sur un fichier SQLite : l'état écrit
  par un appel d'outil est relu par un appel ultérieur (preuve via DatabaseSessionService).
- Read-through ``fastmcp.Client`` : service_set → create → state_set → state_get.

Note ADK (cf. docs/adk-api-notes/sessions.md) : l'état ``temp:`` n'est PAS persisté par
``get_session``. ``state_set`` lit donc l'état sur la session mutée (où ``temp`` est visible) ;
un ``state_get`` ultérieur sur ``temp`` renvoie ``found=False`` — comportement attendu, testé.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import sessions as S
from adk_toolkit_mcp.runtime import reset_service_cache
from adk_toolkit_mcp.server import build_server

#: SQLAlchemy est requis pour DatabaseSessionService (extra ``db`` / ``dev``).
_HAS_SQLALCHEMY = importlib.util.find_spec("sqlalchemy") is not None


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole les tests : vide le cache singleton de services avant/après chacun."""
    reset_service_cache()
    yield
    reset_service_cache()


def _db_url(tmp_path: Path) -> str:
    """URL SQLite à pilote ASYNC (ADK utilise create_async_engine ; pysqlite échouerait)."""
    return f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"


async def _setup_in_memory(tmp_path: Path, app_name: str = "myapp") -> str:
    """Configure un backend in_memory et renvoie le ``path`` racine (string)."""
    path = str(tmp_path)
    result = S.service_set(path=path, app_name=app_name, kind="in_memory")
    assert result["ok"] is True
    return path


# --------------------------------------------------------------------------- #
# service_set : persistance + validations
# --------------------------------------------------------------------------- #
async def test_service_set_in_memory_persists_config(tmp_path: Path) -> None:
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert result["ok"] is True
    assert result["data"]["kind"] == "in_memory"
    config_path = Path(result["data"]["config_path"])
    assert config_path.exists()
    assert config_path.name == "runtime.json"


async def test_service_set_database_persists_url(tmp_path: Path) -> None:
    url = _db_url(tmp_path)
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="database", db_url=url)
    assert result["ok"] is True
    assert result["data"]["kind"] == "database"
    assert result["data"]["db_url"] == url


async def test_service_set_rejects_unknown_kind(tmp_path: Path) -> None:
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="bogus")
    assert result["ok"] is False
    assert "kind" in result["error"].lower()


async def test_service_set_database_requires_url(tmp_path: Path) -> None:
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="database")
    assert result["ok"] is False
    assert "db_url" in result["error"]


async def test_service_set_vertex_requires_project_location(tmp_path: Path) -> None:
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="vertex", project="p")
    assert result["ok"] is False
    assert "location" in result["error"] or "vertex" in result["error"]


async def test_service_set_is_idempotent(tmp_path: Path) -> None:
    first = S.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    second = S.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert first["data"]["changed"] is True
    assert second["data"]["changed"] is False


# --------------------------------------------------------------------------- #
# create / get / list / delete
# --------------------------------------------------------------------------- #
async def test_create_then_get_round_trip(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1", state={"foo": "bar"})
    assert created["ok"] is True
    sid = created["data"]["session_id"]
    assert created["data"]["state"] == {"foo": "bar"}

    got = await S.get(path=path, app_name="myapp", user_id="u1", session_id=sid)
    assert got["ok"] is True
    assert got["data"]["session_id"] == sid
    assert got["data"]["state"] == {"foo": "bar"}
    assert got["data"]["event_count"] == 0


async def test_create_with_explicit_session_id(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1", session_id="fixed-id")
    assert created["data"]["session_id"] == "fixed-id"


async def test_create_rejects_empty_user_id(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    result = await S.create(path=path, app_name="myapp", user_id="  ")
    assert result["ok"] is False
    assert "user_id" in result["error"]


async def test_get_missing_session_returns_err(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    result = await S.get(path=path, app_name="myapp", user_id="u1", session_id="nope")
    assert result["ok"] is False
    assert "introuvable" in result["error"].lower()


async def test_list_sessions(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    a = await S.create(path=path, app_name="myapp", user_id="u1")
    b = await S.create(path=path, app_name="myapp", user_id="u1")
    listed = await S.list_sessions_tool(path=path, app_name="myapp", user_id="u1")
    assert listed["ok"] is True
    ids = set(listed["data"]["session_ids"])
    assert {a["data"]["session_id"], b["data"]["session_id"]} <= ids


async def test_delete_removes_session(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    deleted = await S.delete(path=path, app_name="myapp", user_id="u1", session_id=sid)
    assert deleted["ok"] is True
    assert deleted["data"]["deleted"] == sid

    got = await S.get(path=path, app_name="myapp", user_id="u1", session_id=sid)
    assert got["ok"] is False


async def test_delete_rejects_empty_session_id(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    result = await S.delete(path=path, app_name="myapp", user_id="u1", session_id="")
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# state_set / state_get for EACH scope + prefix correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("scope", "expected_prefix"),
    [
        ("session", ""),
        ("app", "app:"),
        ("user", "user:"),
        ("temp", "temp:"),
    ],
)
async def test_state_set_returns_value_for_each_scope(
    tmp_path: Path, scope: str, expected_prefix: str
) -> None:
    """state_set renvoie la valeur pour CHAQUE scope (y compris temp, sur l'objet muté)."""
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    result = await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="mykey",
        value="myval",
        scope=scope,
    )
    assert result["ok"] is True
    # Préfixe correct selon le scope (constantes State.*_PREFIX).
    assert result["data"]["stored_key"] == f"{expected_prefix}mykey"
    # La valeur est lisible dans l'état renvoyé (vrai pour les 4 scopes ici).
    assert result["data"]["state"][f"{expected_prefix}mykey"] == "myval"


@pytest.mark.parametrize("scope", ["session", "app", "user"])
async def test_state_set_then_get_persisted_scopes(tmp_path: Path, scope: str) -> None:
    """Pour session/app/user, state_get (refetch) retrouve la valeur posée par state_set."""
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="k",
        value="v",
        scope=scope,
    )
    got = await S.state_get(
        path=path, app_name="myapp", user_id="u1", session_id=sid, key="k", scope=scope
    )
    assert got["ok"] is True
    assert got["data"]["found"] is True
    assert got["data"]["value"] == "v"


async def test_state_get_temp_not_persisted_across_calls(tmp_path: Path) -> None:
    """L'état temp posé par un appel n'est PAS retrouvé par un state_get ultérieur (ADK)."""
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    set_result = await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="tk",
        value="tv",
        scope="temp",
    )
    # Visible sur l'objet muté retourné par state_set...
    assert set_result["data"]["state"]["temp:tk"] == "tv"
    # ...mais absent après refetch (sémantique ADK : temp non persisté).
    got = await S.state_get(
        path=path, app_name="myapp", user_id="u1", session_id=sid, key="tk", scope="temp"
    )
    assert got["ok"] is True
    assert got["data"]["found"] is False
    assert got["data"]["value"] is None


async def test_state_set_rejects_bad_scope(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    result = await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="k",
        value="v",
        scope="galaxy",
    )
    assert result["ok"] is False
    assert "scope" in result["error"].lower()


async def test_state_get_rejects_bad_scope(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    result = await S.state_get(
        path=path, app_name="myapp", user_id="u1", session_id=sid, key="k", scope="galaxy"
    )
    assert result["ok"] is False
    assert "scope" in result["error"].lower()


async def test_state_set_rejects_empty_key(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    result = await S.state_set(
        path=path, app_name="myapp", user_id="u1", session_id=sid, key="  ", value="v"
    )
    assert result["ok"] is False
    assert "key" in result["error"]


async def test_state_set_missing_session_returns_err(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    result = await S.state_set(
        path=path, app_name="myapp", user_id="u1", session_id="nope", key="k", value="v"
    )
    assert result["ok"] is False
    assert "introuvable" in result["error"].lower()


# --------------------------------------------------------------------------- #
# append_event
# --------------------------------------------------------------------------- #
async def test_append_event_increments_count(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    assert created["data"]["event_count"] == 0

    first = await S.append_event(
        path=path, app_name="myapp", user_id="u1", session_id=sid, author="user", text="hi"
    )
    assert first["ok"] is True
    assert first["data"]["event_count"] == 1

    second = await S.append_event(
        path=path, app_name="myapp", user_id="u1", session_id=sid, author="assistant", text="yo"
    )
    assert second["data"]["event_count"] == 2


async def test_append_event_with_state_delta(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    result = await S.append_event(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        author="user",
        state_delta={"counter": 1, "app:shared": "x"},
    )
    assert result["ok"] is True
    assert result["data"]["state"]["counter"] == 1
    assert result["data"]["state"]["app:shared"] == "x"


async def test_append_event_rejects_empty_author(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    result = await S.append_event(
        path=path, app_name="myapp", user_id="u1", session_id=sid, author="  "
    )
    assert result["ok"] is False
    assert "author" in result["error"]


async def test_append_event_missing_session_returns_err(tmp_path: Path) -> None:
    path = await _setup_in_memory(tmp_path)
    result = await S.append_event(
        path=path, app_name="myapp", user_id="u1", session_id="nope", author="user"
    )
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# Bad-config handling (corrupt runtime.json -> clean err, no exception)
# --------------------------------------------------------------------------- #
async def test_operations_on_corrupt_config_return_err(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = await S.create(path=str(tmp_path), app_name="myapp", user_id="u1")
    assert result["ok"] is False
    assert result["error"]


async def test_operations_on_invalid_backend_return_err(tmp_path: Path) -> None:
    """runtime.json valide mais backend invalide (database sans db_url) -> err propre.

    Simule un fichier édité à la main : la config se charge mais l'instanciation du service
    échoue (ValueError) ; l'outil doit renvoyer err sans laisser remonter l'exception.
    """
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text(
        '{"session": {"kind": "database", "db_url": null}}', encoding="utf-8"
    )
    result = await S.create(path=str(tmp_path), app_name="myapp", user_id="u1")
    assert result["ok"] is False
    assert "db_url" in result["error"]


# --------------------------------------------------------------------------- #
# FUNCTIONAL PERSISTENCE — real DatabaseSessionService over a SQLite file
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_SQLALCHEMY, reason="sqlalchemy non installé (extra 'db'/'dev')")
async def test_database_backend_state_persists_across_calls(tmp_path: Path) -> None:
    """PREUVE de persistance : state_set via un appel, relu par un state_get ultérieur.

    Backend ``database`` sur un fichier SQLite (pilote async aiosqlite). On vide même le cache
    singleton entre l'écriture et la lecture pour forcer une NOUVELLE instance de service —
    la valeur ne peut alors provenir que de la base, pas de l'état en mémoire.
    """
    path = str(tmp_path)
    url = _db_url(tmp_path)
    set_cfg = S.service_set(path=path, app_name="myapp", kind="database", db_url=url)
    assert set_cfg["ok"] is True

    created = await S.create(path=path, app_name="myapp", user_id="u1")
    assert created["ok"] is True, created
    sid = created["data"]["session_id"]

    set_state = await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="persisted",
        value="survives",
        scope="session",
    )
    assert set_state["ok"] is True

    # Force une instance de service neuve : la lecture suivante traverse la base.
    reset_service_cache()

    got = await S.state_get(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="persisted",
        scope="session",
    )
    assert got["ok"] is True
    assert got["data"]["found"] is True
    assert got["data"]["value"] == "survives"


@pytest.mark.skipif(not _HAS_SQLALCHEMY, reason="sqlalchemy non installé (extra 'db'/'dev')")
async def test_database_backend_app_user_prefixes_persist(tmp_path: Path) -> None:
    """app:/user: sont relus depuis la base sous leur nom préfixé réel."""
    path = str(tmp_path)
    url = _db_url(tmp_path)
    S.service_set(path=path, app_name="myapp", kind="database", db_url=url)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="ak",
        value="av",
        scope="app",
    )
    await S.state_set(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        key="uk",
        value="uv",
        scope="user",
    )
    reset_service_cache()

    got = await S.get(path=path, app_name="myapp", user_id="u1", session_id=sid)
    assert got["ok"] is True
    assert got["data"]["state"]["app:ak"] == "av"
    assert got["data"]["state"]["user:uk"] == "uv"


# --------------------------------------------------------------------------- #
# In-memory fastmcp.Client read-through (exposed names + double-prefix guard)
# --------------------------------------------------------------------------- #
async def test_client_read_through_full_flow(tmp_path: Path) -> None:
    """sessions_service_set → sessions_create → sessions_state_set → sessions_state_get."""
    mcp = build_server()
    path = str(tmp_path)
    async with Client(mcp) as client:
        # Aucun nom double-préfixé (sessions_sessions_*) exposé.
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert "sessions_create" in names
        assert "sessions_state_set" in names
        assert "sessions_list" in names
        assert not any(n.startswith("sessions_sessions_") for n in names)

        set_cfg = await client.call_tool(
            "sessions_service_set",
            {"path": path, "app_name": "myapp", "kind": "in_memory"},
        )
        assert set_cfg.data["ok"] is True

        created = await client.call_tool(
            "sessions_create", {"path": path, "app_name": "myapp", "user_id": "u1"}
        )
        assert created.data["ok"] is True
        sid = created.data["data"]["session_id"]

        set_state = await client.call_tool(
            "sessions_state_set",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": sid,
                "key": "k",
                "value": "v",
                "scope": "user",
            },
        )
        assert set_state.data["ok"] is True
        assert set_state.data["data"]["stored_key"] == "user:k"

        got = await client.call_tool(
            "sessions_state_get",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": sid,
                "key": "k",
                "scope": "user",
            },
        )
        assert got.data["ok"] is True
        assert got.data["data"]["found"] is True
        assert got.data["data"]["value"] == "v"
