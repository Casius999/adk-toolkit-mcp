"""Tests unitaires du domaine ``memory`` (P2b — service de mémoire runtime ADK).

Les outils sont **async** (``asyncio_mode=auto``). On appelle les fonctions bare directement
et, pour le read-through, via un ``fastmcp.Client`` in-memory.

Couverture clé :
- ``service_set`` : persiste le backend ; validations (kind, vertex_rag, vertex_memory_bank) ;
  préserve les backends session/artifacts déjà écrits.
- FONCTIONNEL (in_memory) : créer une session (via le domaine sessions), y ajouter des
  événements PORTANT du texte, ``add_session`` dans la mémoire, puis ``search`` avec une
  requête qui DOIT matcher (rappel par mots-clés d'InMemoryMemoryService) → résultat non vide
  contenant le texte attendu ; une requête sans correspondance → 0 résultat.
- Erreurs propres : pas de backend configuré, session introuvable, query/inputs vides.
- Branches Vertex : validation de forme + (selon présence de l'extra) erreur orientée action.
- Read-through ``fastmcp.Client`` pour un flux mémoire complet.

Rappel ADK (cf. docs/adk-api-notes/memory-artifacts.md) : seuls les événements avec
``content.parts`` textuels sont indexés ; le rappel est par MOTS-CLÉS (pas sémantique).
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
    """``find_spec`` tolérant : un namespace parent absent lève ModuleNotFoundError."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


#: Les services Vertex mémoire nécessitent l'extra ``gcp`` (google-cloud-aiplatform).
_HAS_VERTEX = _has_module("vertexai")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole les tests : vide le cache singleton de services avant/après chacun."""
    reset_service_cache()
    yield
    reset_service_cache()


async def _setup(tmp_path: Path, app_name: str = "myapp") -> str:
    """Configure les backends session ET mémoire in_memory ; renvoie le ``path`` racine."""
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
    """Choisir le backend mémoire ne doit pas écraser le backend session déjà écrit."""
    path = str(tmp_path)
    S.service_set(path=path, app_name="myapp", kind="database", db_url="sqlite+aiosqlite:///x.db")
    M.service_set(path=path, app_name="myapp", kind="in_memory")
    # Le backend session doit subsister.
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
    """Pas de memory_service_set → err explicite (et non une exception)."""
    path = str(tmp_path)
    # Configure seulement les sessions, pas la mémoire.
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
    assert "introuvable" in result["error"].lower()


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
    """service_set tolère une runtime.json corrompue (repart d'une config par défaut)."""
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
    """Flux complet : session + événements texte → add_session → search trouve le souvenir.

    InMemoryMemoryService indexe uniquement les événements porteurs de texte et fait un rappel
    par mots-clés. On ajoute deux événements mentionnant « Paris » puis on cherche « Paris ».
    """
    path = await _setup(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]

    # Événements PORTANT du texte (sinon non indexés).
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
    # Chaque souvenir expose author + timestamp + content sérialisé.
    first = hit["data"]["memories"][0]
    assert first["author"] in {"user", "assistant"}
    assert first["timestamp"]
    assert first["content"]["parts"][0]["text"]


async def test_functional_search_no_match_returns_empty(tmp_path: Path) -> None:
    """Une requête sans mot-clé commun ne renvoie aucun souvenir (count == 0)."""
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
    """Un événement sans texte (state_delta seul) n'est PAS indexé → search ne le trouve pas."""
    path = await _setup(tmp_path)
    created = await S.create(path=path, app_name="myapp", user_id="u1")
    sid = created["data"]["session_id"]
    # Événement sans content textuel.
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
@pytest.mark.skipif(_HAS_VERTEX, reason="extra gcp présent : pas d'erreur de dépendance")
async def test_vertex_rag_search_errors_without_extra(tmp_path: Path) -> None:
    """Avec un backend vertex_rag mais sans l'extra gcp, search renvoie une err orientée action."""
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
