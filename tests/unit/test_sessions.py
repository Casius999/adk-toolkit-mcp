"""Unit tests for the ``sessions`` domain (P2a — ADK runtime services).

The tools are **async** (``asyncio_mode=auto`` in pyproject). We call the bare functions directly
(this FastMCP version's ``@tool`` decorator returns the original function) and, for the
read-through, via an in-memory ``fastmcp.Client``.

Key coverage:
- ``service_set``: persists the backend; validations (kind, db_url, vertex).
- ``create`` → ``get`` round-trip; ``list`` / ``delete``; ``append_event`` increments.
- ``state_set`` then ``state_get`` for EACH scope (session/app/user/temp).
- Prefix correctness: app/user/temp stored under ``app:``/``user:``/``temp:``.
- FUNCTIONAL PERSISTENCE with a ``database`` backend on a SQLite file: state written by one tool
  call is re-read by a later call (proof via DatabaseSessionService).
- ``fastmcp.Client`` read-through: service_set → create → state_set → state_get.
- Security: ``service_set`` must NOT expose the credentials (user:password) in the MCP response;
  ``_redact_db_url`` masks the userinfo while keeping the scheme/host/db.

ADK note (cf. docs/adk-api-notes/sessions.md): ``temp:`` state is NOT persisted by
``get_session``. ``state_set`` therefore reads the state on the mutated session (where ``temp`` is
visible); a later ``state_get`` on ``temp`` returns ``found=False`` — expected behavior, tested.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import sessions as S
from adk_toolkit_mcp.domains.sessions import _redact_db_url
from adk_toolkit_mcp.runtime import reset_service_cache
from adk_toolkit_mcp.server import build_server

#: SQLAlchemy is required for DatabaseSessionService (``db`` / ``dev`` extra).
_HAS_SQLALCHEMY = importlib.util.find_spec("sqlalchemy") is not None


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isolate the tests: clear the singleton service cache before/after each."""
    reset_service_cache()
    yield
    reset_service_cache()


def _db_url(tmp_path: Path) -> str:
    """ASYNC-driver SQLite URL (ADK uses create_async_engine; pysqlite would fail)."""
    return f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"


async def _setup_in_memory(tmp_path: Path, app_name: str = "myapp") -> str:
    """Configure an in_memory backend and return the root ``path`` (string)."""
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
    # The returned db_url is redacted (no credentials); the raw URL is preserved in runtime.json.
    assert result["data"]["db_url"] == url  # SQLite has no credentials → returned unchanged


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
    assert "not found" in result["error"].lower()


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
    """state_set returns the value for EACH scope (including temp, on the mutated object)."""
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
    # Correct prefix per scope (State.*_PREFIX constants).
    assert result["data"]["stored_key"] == f"{expected_prefix}mykey"
    # The value is readable in the returned state (true for all 4 scopes here).
    assert result["data"]["state"][f"{expected_prefix}mykey"] == "myval"


@pytest.mark.parametrize("scope", ["session", "app", "user"])
async def test_state_set_then_get_persisted_scopes(tmp_path: Path, scope: str) -> None:
    """For session/app/user, state_get (refetch) finds the value set by state_set."""
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
    """temp state set by one call is NOT found by a later state_get (ADK)."""
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
    # Visible on the mutated object returned by state_set...
    assert set_result["data"]["state"]["temp:tk"] == "tv"
    # ...but absent after refetch (ADK semantics: temp not persisted).
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
    assert "not found" in result["error"].lower()


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
    """Valid runtime.json but invalid backend (database without db_url) -> clean err.

    Simulates a hand-edited file: the config loads but the service instantiation fails
    (ValueError); the tool must return err without letting the exception propagate.
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
@pytest.mark.skipif(not _HAS_SQLALCHEMY, reason="sqlalchemy not installed ('db'/'dev' extra)")
async def test_database_backend_state_persists_across_calls(tmp_path: Path) -> None:
    """Persistence PROOF: state_set via one call, re-read by a later state_get.

    ``database`` backend on a SQLite file (async aiosqlite driver). We even clear the singleton
    cache between the write and the read to force a NEW service instance — the value can then only
    come from the database, not from in-memory state.
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

    # Force a NEW service instance: the next read goes through the database.
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


@pytest.mark.skipif(not _HAS_SQLALCHEMY, reason="sqlalchemy not installed ('db'/'dev' extra)")
async def test_database_backend_app_user_prefixes_persist(tmp_path: Path) -> None:
    """app:/user: are re-read from the database under their real prefixed name."""
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
        # No double-prefixed name (sessions_sessions_*) exposed.
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


# --------------------------------------------------------------------------- #
# _redact_db_url unit tests
# --------------------------------------------------------------------------- #
def test_redact_db_url_masks_password() -> None:
    """postgresql+asyncpg://user:secret@host/db → credentials replaced by ***."""
    url = "postgresql+asyncpg://user:s3cret@host:5432/db"
    redacted = _redact_db_url(url)
    assert "s3cret" not in redacted
    assert "***" in redacted
    # Scheme, host, port and database name are preserved.
    assert redacted.startswith("postgresql+asyncpg://")
    assert "host:5432" in redacted
    assert "/db" in redacted


def test_redact_db_url_masks_user_and_password() -> None:
    """Both username and password are hidden behind the single *** token."""
    url = "postgresql+asyncpg://admin:p@ssw0rd@db.example.com/mydb"
    redacted = _redact_db_url(url)
    assert "admin" not in redacted
    assert "p@ssw0rd" not in redacted
    assert "***@db.example.com" in redacted
    assert "/mydb" in redacted


def test_redact_db_url_sqlite_no_credentials_unchanged() -> None:
    """sqlite+aiosqlite:///path/to.db has no credentials → returned as-is."""
    url = "sqlite+aiosqlite:///path/to/my.db"
    assert _redact_db_url(url) == url


def test_redact_db_url_sqlite_relative_unchanged() -> None:
    """sqlite+aiosqlite:///relative.db (no host) → returned as-is."""
    url = "sqlite+aiosqlite:///relative.db"
    assert _redact_db_url(url) == url


# --------------------------------------------------------------------------- #
# service_set credential-redaction integration tests
# --------------------------------------------------------------------------- #
async def test_service_set_database_does_not_expose_password(tmp_path: Path) -> None:
    """service_set with kind='database' and a URL containing credentials must NOT
    return the plain password in the MCP response payload (security rule)."""
    secret = "s3cret"
    url = f"postgresql+asyncpg://user:{secret}@host:5432/db"
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="database", db_url=url)
    assert result["ok"] is True
    returned_url: str = result["data"]["db_url"]
    # Password must not appear in the returned db_url.
    assert secret not in returned_url, (
        f"Credential leaked in service_set response: {returned_url!r}"
    )
    # The redacted marker must be present.
    assert "***" in returned_url


async def test_service_set_database_redacted_url_preserves_structure(tmp_path: Path) -> None:
    """The redacted URL keeps scheme, host, port and database name — just credentials masked."""
    url = "postgresql+asyncpg://user:s3cret@host:5432/mydb"
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="database", db_url=url)
    assert result["ok"] is True
    returned_url: str = result["data"]["db_url"]
    assert returned_url.startswith("postgresql+asyncpg://")
    assert "host:5432" in returned_url
    assert "/mydb" in returned_url
    assert "***@host:5432" in returned_url


async def test_service_set_database_runtime_json_stores_real_url(tmp_path: Path) -> None:
    """runtime.json must persist the RAW (unredacted) URL so the backend can actually connect."""
    import json

    secret = "s3cret"
    url = f"postgresql+asyncpg://user:{secret}@host:5432/db"
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="database", db_url=url)
    assert result["ok"] is True
    config_path = Path(result["data"]["config_path"])
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    # The file must store the real URL (needed to connect later).
    assert raw["session"]["db_url"] == url, (
        "runtime.json must store the unredacted URL for actual DB connections"
    )


async def test_service_set_sqlite_credential_free_url_returned_intact(tmp_path: Path) -> None:
    """SQLite URLs have no credentials — the returned db_url is the original string."""
    url = _db_url(tmp_path)
    result = S.service_set(path=str(tmp_path), app_name="myapp", kind="database", db_url=url)
    assert result["ok"] is True
    # No credentials in a SQLite URL → returned unchanged.
    assert result["data"]["db_url"] == url
