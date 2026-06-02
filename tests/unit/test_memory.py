"""Unit tests for the ``memory`` domain (P2b — ADK runtime memory service).

The tools are **async** (``asyncio_mode=auto``). We call the bare functions directly and, for the
read-through, via an in-memory ``fastmcp.Client``.

Key coverage:
- ``service_set``: persists the backend; validations (kind, vertex_rag, vertex_memory_bank);
  preserves the session/artifacts backends already written.
- FUNCTIONAL (in_memory): create a session (via the sessions domain), add text-CARRYING events to
  it, ``add_session`` into memory, then ``search`` with a query that MUST match
  (InMemoryMemoryService's keyword recall) → non-empty result containing the expected text; a
  non-matching query → 0 results.
- Clean errors: no configured backend, session not found, empty query/inputs.
- Vertex branches: shape validation + (depending on the extra's presence) an actionable error.
- ``fastmcp.Client`` read-through for a complete memory flow.

ADK reminder (cf. docs/adk-api-notes/memory-artifacts.md): only events with textual
``content.parts`` are indexed; recall is by KEYWORD (not semantic).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import memory as M
from adk_toolkit_mcp.domains import sessions as S
from adk_toolkit_mcp.runtime import reset_service_cache
from adk_toolkit_mcp.server import build_server


def _has_module(name: str) -> bool:
    """Tolerant ``find_spec``: an absent parent namespace raises ModuleNotFoundError."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


#: The Vertex memory services require the ``gcp`` extra (google-cloud-aiplatform).
_HAS_VERTEX = _has_module("vertexai")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isolate the tests: clear the singleton service cache before/after each."""
    reset_service_cache()
    yield
    reset_service_cache()


async def _setup(tmp_path: Path, app_name: str = "myapp") -> str:
    """Configure both the session AND memory in_memory backends; return the root ``path``."""
    path = str(tmp_path)
    assert S.service_set(path=path, app_name=app_name, kind="in_memory")["ok"] is True
    assert M.service_set(path=path, app_name=app_name, kind="in_memory")["ok"] is True
    return path


# --------------------------------------------------------------------------- #
# service_set : persistance + validations
# --------------------------------------------------------------------------- #
async def test_service_set_in_memory_persists(tmp_path: Path) -> None:
    result = M.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert result["ok"] is True
    assert result["data"]["kind"] == "in_memory"
    config_path = Path(result["data"]["config_path"])
    assert config_path.exists()
    assert config_path.name == "runtime.json"


async def test_service_set_rejects_unknown_kind(tmp_path: Path) -> None:
    result = M.service_set(path=str(tmp_path), app_name="myapp", kind="bogus")
    assert result["ok"] is False
    assert "kind" in result["error"].lower()


async def test_service_set_vertex_rag_requires_corpus(tmp_path: Path) -> None:
    result = M.service_set(path=str(tmp_path), app_name="myapp", kind="vertex_rag")
    assert result["ok"] is False
    assert "rag_corpus" in result["error"]


async def test_service_set_vertex_memory_bank_requires_fields(tmp_path: Path) -> None:
    result = M.service_set(
        path=str(tmp_path), app_name="myapp", kind="vertex_memory_bank", project="p"
    )
    assert result["ok"] is False
    assert "agent_engine_id" in result["error"] or "location" in result["error"]


async def test_service_set_vertex_rag_persists_corpus(tmp_path: Path) -> None:
    result = M.service_set(
        path=str(tmp_path),
        app_name="myapp",
        kind="vertex_rag",
        rag_corpus="projects/p/locations/us/ragCorpora/1",
    )
    assert result["ok"] is True
    assert result["data"]["rag_corpus"] == "projects/p/locations/us/ragCorpora/1"


async def test_service_set_is_idempotent(tmp_path: Path) -> None:
    first = M.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    second = M.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert first["data"]["changed"] is True
    assert second["data"]["changed"] is False


async def test_service_set_preserves_session_backend(tmp_path: Path) -> None:
    """Choosing the memory backend must not overwrite an already-written session backend."""
    path = str(tmp_path)
    S.service_set(path=path, app_name="myapp", kind="database", db_url="sqlite+aiosqlite:///x.db")
    M.service_set(path=path, app_name="myapp", kind="in_memory")
    # The session backend must persist.
    from adk_toolkit_mcp.runtime import load_runtime_config
    from adk_toolkit_mcp.workspace import Workspace

    cfg = load_runtime_config(Workspace(Path(path) / "myapp"), "myapp")
    assert cfg.session.kind == "database"
    assert cfg.memory is not None
    assert cfg.memory.kind == "in_memory"


# --------------------------------------------------------------------------- #
# add_session / search — error paths
# --------------------------------------------------------------------------- #
async def test_search_without_configured_service_returns_err(tmp_path: Path) -> None:
    """No memory_service_set → explicit err (and not an exception)."""
    path = str(tmp_path)
    # Configure only the sessions, not the memory.
    S.service_set(path=path, app_name="myapp", kind="in_memory")
    result = await M.search(path=path, app_name="myapp", user_id="u1", query="hi")
    assert result["ok"] is False
    assert "memory_service_set" in result["error"]


async def test_add_session_without_configured_service_returns_err(tmp_path: Path) -> None:
    path = str(tmp_path)
    S.service_set(path=path, app_name="myapp", kind="in_memory")
    result = await M.add_session(path=path, app_name="myapp", user_id="u1", session_id="s")
    assert result["ok"] is False
    assert "memory_service_set" in result["error"]


async def test_add_session_missing_session_returns_err(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    result = await M.add_session(path=path, app_name="myapp", user_id="u1", session_id="nope")
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


async def test_add_session_rejects_empty_session_id(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    result = await M.add_session(path=path, app_name="myapp", user_id="u1", session_id="  ")
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_search_rejects_empty_query(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    result = await M.search(path=path, app_name="myapp", user_id="u1", query="  ")
    assert result["ok"] is False
    assert "query" in result["error"]


async def test_search_on_corrupt_config_returns_err(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = await M.search(path=str(tmp_path), app_name="myapp", user_id="u1", query="x")
    assert result["ok"] is False
    assert result["error"]


async def test_add_session_on_corrupt_config_returns_err(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = await M.add_session(path=str(tmp_path), app_name="myapp", user_id="u1", session_id="s")
    assert result["ok"] is False
    assert result["error"]


async def test_service_set_overwrites_corrupt_config(tmp_path: Path) -> None:
    """service_set tolerates a corrupt runtime.json (starts from a default config)."""
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = M.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert result["ok"] is True
    assert result["data"]["kind"] == "in_memory"


# --------------------------------------------------------------------------- #
# FUNCTIONAL — real InMemoryMemoryService keyword recall
# --------------------------------------------------------------------------- #
async def test_functional_add_and_search_hits(tmp_path: Path) -> None:
    """Complete flow: session + text events → add_session → search finds the memory.

    InMemoryMemoryService indexes only the text-carrying events and does a keyword recall. We add
    two events mentioning "Paris" then search for "Paris".
    """
    path = await _setup(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    # Text-CARRYING events (otherwise not indexed).
    await S.append_event(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        author="user",
        text="The capital of France is Paris",
    )
    await S.append_event(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        author="assistant",
        text="Paris has many museums",
    )

    added = await M.add_session(path=path, app_name="myapp", user_id="u1", session_id=sid)
    assert added["ok"] is True
    assert added["data"]["event_count"] == 2

    hit = await M.search(path=path, app_name="myapp", user_id="u1", query="Paris")
    assert hit["ok"] is True
    assert hit["data"]["count"] >= 1
    joined = " ".join(m["text"] for m in hit["data"]["memories"])
    assert "Paris" in joined
    # Each memory exposes author + timestamp + serialized content.
    first = hit["data"]["memories"][0]
    assert first["author"] in {"user", "assistant"}
    assert first["timestamp"]
    assert first["content"]["parts"][0]["text"]


async def test_functional_search_no_match_returns_empty(tmp_path: Path) -> None:
    """A query without a common keyword returns no memory (count == 0)."""
    path = await _setup(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    await S.append_event(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        author="user",
        text="hello world",
    )
    await M.add_session(path=path, app_name="myapp", user_id="u1", session_id=sid)

    miss = await M.search(path=path, app_name="myapp", user_id="u1", query="zzzznomatch")
    assert miss["ok"] is True
    assert miss["data"]["count"] == 0
    assert miss["data"]["memories"] == []


async def test_functional_state_only_event_not_recalled(tmp_path: Path) -> None:
    """An event without text (state_delta only) is NOT indexed → search does not find it."""
    path = await _setup(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    # Event without textual content.
    await S.append_event(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id=sid,
        author="user",
        state_delta={"secret": "treasure"},
    )
    await M.add_session(path=path, app_name="myapp", user_id="u1", session_id=sid)

    res = await M.search(path=path, app_name="myapp", user_id="u1", query="treasure")
    assert res["ok"] is True
    assert res["data"]["count"] == 0


# --------------------------------------------------------------------------- #
# Vertex branches: validate config + actionable error when extra absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_HAS_VERTEX, reason="gcp extra present: no dependency error")
async def test_vertex_rag_search_errors_without_extra(tmp_path: Path) -> None:
    """With a vertex_rag backend but without the gcp extra, search returns an actionable err."""
    path = str(tmp_path)
    S.service_set(path=path, app_name="myapp", kind="in_memory")
    M.service_set(
        path=path,
        app_name="myapp",
        kind="vertex_rag",
        rag_corpus="projects/p/locations/us/ragCorpora/1",
    )
    result = await M.search(path=path, app_name="myapp", user_id="u1", query="x")
    assert result["ok"] is False
    assert "gcp" in result["error"]


# --------------------------------------------------------------------------- #
# In-memory fastmcp.Client read-through (exposed names + double-prefix guard)
# --------------------------------------------------------------------------- #
async def test_client_read_through_memory_flow(tmp_path: Path) -> None:
    """memory_service_set → sessions_* (seed) → memory_add_session → memory_search."""
    mcp = build_server()
    path = str(tmp_path)
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert "memory_service_set" in names
        assert "memory_add_session" in names
        assert "memory_search" in names
        assert not any(n.startswith("memory_memory_") for n in names)

        await client.call_tool(
            "sessions_service_set", {"path": path, "app_name": "myapp", "kind": "in_memory"}
        )
        await client.call_tool(
            "memory_service_set", {"path": path, "app_name": "myapp", "kind": "in_memory"}
        )
        created = await client.call_tool(
            "sessions_create", {"path": path, "app_name": "myapp", "user_id": "u1"}
        )
        sid = created.data["data"]["session_id"]
        await client.call_tool(
            "sessions_append_event",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": sid,
                "author": "user",
                "text": "Banana bread recipe",
            },
        )
        added = await client.call_tool(
            "memory_add_session",
            {"path": path, "app_name": "myapp", "user_id": "u1", "session_id": sid},
        )
        assert added.data["ok"] is True

        found = await client.call_tool(
            "memory_search",
            {"path": path, "app_name": "myapp", "user_id": "u1", "query": "banana"},
        )
        assert found.data["ok"] is True
        assert found.data["data"]["count"] >= 1
