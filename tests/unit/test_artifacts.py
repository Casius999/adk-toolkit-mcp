"""Tests unitaires du domaine ``artifacts`` (P2b — service d'artifacts runtime ADK).

Les outils sont **async** (``asyncio_mode=auto``). On appelle les fonctions bare directement
(``list`` est exposé via la fonction Python ``list_artifacts_tool``) et, pour le read-through,
via un ``fastmcp.Client`` in-memory.

Couverture clé :
- ``service_set`` : persiste le backend ; validations (kind, gcs) ; préserve session/memory.
- FONCTIONNEL (in_memory) : save texte → version 0 ; re-save → version 1 ; load (dernière +
  version précise) restitue le contenu exact ; list montre le fichier ; versions = [0, 1] ;
  delete supprime. Nom ``user:``-préfixé accepté. Round-trip base64 (mime binaire).
- Erreurs propres : pas de backend, ni text ni bytes_b64 (ou les deux), base64 invalide,
  artifact absent, filename vide.
- Branche GCS : validation de forme + (selon l'extra) erreur orientée action.
- Read-through ``fastmcp.Client`` pour un flux artifacts complet.

Rappel ADK (cf. docs/adk-api-notes/memory-artifacts.md) : save renvoie une version 0-indexée ;
load renvoie une ``Part`` (``.text`` ou ``.inline_data``) ou ``None`` si absente.
"""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import artifacts as A
from adk_toolkit_mcp.runtime import reset_service_cache
from adk_toolkit_mcp.server import build_server


def _has_module(name: str) -> bool:
    """``find_spec`` tolérant : un namespace parent absent lève ModuleNotFoundError."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


#: GcsArtifactService nécessite l'extra ``gcp`` (google.cloud.storage).
_HAS_GCS = _has_module("google.cloud.storage")


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole les tests : vide le cache singleton de services avant/après chacun."""
    reset_service_cache()
    yield
    reset_service_cache()


async def _setup(tmp_path: Path, app_name: str = "myapp") -> str:
    """Configure un backend artifacts in_memory ; renvoie le ``path`` racine (string)."""
    path = str(tmp_path)
    assert A.service_set(path=path, app_name=app_name, kind="in_memory")["ok"] is True
    return path


# --------------------------------------------------------------------------- #
# service_set : persistance + validations
# --------------------------------------------------------------------------- #
async def test_service_set_in_memory_persists(tmp_path: Path) -> None:
    result = A.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert result["ok"] is True
    assert result["data"]["kind"] == "in_memory"
    assert Path(result["data"]["config_path"]).name == "runtime.json"


async def test_service_set_rejects_unknown_kind(tmp_path: Path) -> None:
    result = A.service_set(path=str(tmp_path), app_name="myapp", kind="bogus")
    assert result["ok"] is False
    assert "kind" in result["error"].lower()


async def test_service_set_gcs_requires_bucket(tmp_path: Path) -> None:
    result = A.service_set(path=str(tmp_path), app_name="myapp", kind="gcs")
    assert result["ok"] is False
    assert "bucket" in result["error"]


async def test_service_set_gcs_persists_bucket(tmp_path: Path) -> None:
    result = A.service_set(path=str(tmp_path), app_name="myapp", kind="gcs", bucket="my-bucket")
    assert result["ok"] is True
    assert result["data"]["bucket"] == "my-bucket"


async def test_service_set_is_idempotent(tmp_path: Path) -> None:
    first = A.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    second = A.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert first["data"]["changed"] is True
    assert second["data"]["changed"] is False


async def test_service_set_preserves_session_and_memory(tmp_path: Path) -> None:
    """Choisir le backend artifacts ne doit pas écraser session/memory déjà écrits."""
    from adk_toolkit_mcp.domains import memory as M
    from adk_toolkit_mcp.domains import sessions as S
    from adk_toolkit_mcp.runtime import load_runtime_config
    from adk_toolkit_mcp.workspace import Workspace

    path = str(tmp_path)
    S.service_set(path=path, app_name="myapp", kind="in_memory")
    M.service_set(path=path, app_name="myapp", kind="in_memory")
    A.service_set(path=path, app_name="myapp", kind="gcs", bucket="b")

    cfg = load_runtime_config(Workspace(Path(path) / "myapp"), "myapp")
    assert cfg.session.kind == "in_memory"
    assert cfg.memory is not None and cfg.memory.kind == "in_memory"
    assert cfg.artifacts is not None and cfg.artifacts.kind == "gcs"


# --------------------------------------------------------------------------- #
# save validation / error paths
# --------------------------------------------------------------------------- #
async def test_save_without_configured_service_returns_err(tmp_path: Path) -> None:
    result = await A.save(
        path=str(tmp_path),
        app_name="myapp",
        user_id="u1",
        session_id="s",
        filename="f.txt",
        text="hi",
    )
    assert result["ok"] is False
    assert "artifacts_service_set" in result["error"]


async def test_save_requires_exactly_one_of_text_or_bytes(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    # Aucun des deux.
    neither = await A.save(
        path=path, app_name="myapp", user_id="u1", session_id="s", filename="f.txt"
    )
    assert neither["ok"] is False
    assert "EXACTEMENT" in neither["error"]
    # Les deux.
    both = await A.save(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s",
        filename="f.txt",
        text="hi",
        bytes_b64=base64.b64encode(b"x").decode("ascii"),
    )
    assert both["ok"] is False
    assert "EXACTEMENT" in both["error"]


async def test_save_rejects_invalid_base64(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    result = await A.save(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s",
        filename="f.bin",
        bytes_b64="!!!not base64!!!",
        mime_type="application/octet-stream",
    )
    assert result["ok"] is False
    assert "base64" in result["error"]


async def test_save_rejects_empty_filename(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    result = await A.save(
        path=path, app_name="myapp", user_id="u1", session_id="s", filename="  ", text="hi"
    )
    assert result["ok"] is False
    assert "filename" in result["error"]


async def test_load_missing_artifact_returns_err(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    result = await A.load(
        path=path, app_name="myapp", user_id="u1", session_id="s", filename="nope.txt"
    )
    assert result["ok"] is False
    assert "introuvable" in result["error"].lower()


@pytest.mark.parametrize("filename", ["", "   "])
async def test_load_delete_versions_reject_empty_filename(tmp_path: Path, filename: str) -> None:
    path = await _setup(tmp_path)
    common = {"path": path, "app_name": "myapp", "user_id": "u1", "session_id": "s"}
    for fn in (
        A.load(filename=filename, **common),
        A.delete(filename=filename, **common),
        A.versions(filename=filename, **common),
    ):
        result = await fn
        assert result["ok"] is False
        assert "filename" in result["error"]


async def test_all_tools_on_corrupt_config_return_err(tmp_path: Path) -> None:
    """runtime.json corrompue → chaque outil renvoie une err propre (pas d'exception)."""
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text("{ broken", encoding="utf-8")
    common = {"path": str(tmp_path), "app_name": "myapp", "user_id": "u1", "session_id": "s"}

    save_res = await A.save(filename="f.txt", text="hi", **common)
    load_res = await A.load(filename="f.txt", **common)
    list_res = await A.list_artifacts_tool(**common)
    del_res = await A.delete(filename="f.txt", **common)
    ver_res = await A.versions(filename="f.txt", **common)
    for res in (save_res, load_res, list_res, del_res, ver_res):
        assert res["ok"] is False
        assert res["error"]


async def test_service_set_overwrites_corrupt_config(tmp_path: Path) -> None:
    """service_set tolère une runtime.json corrompue (repart d'une config par défaut)."""
    app_dir = tmp_path / "myapp"
    (app_dir / ".adk_toolkit").mkdir(parents=True)
    (app_dir / ".adk_toolkit" / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = A.service_set(path=str(tmp_path), app_name="myapp", kind="in_memory")
    assert result["ok"] is True
    assert result["data"]["kind"] == "in_memory"


# --------------------------------------------------------------------------- #
# FUNCTIONAL — full text lifecycle: save/version, load latest+specific, list, versions, delete
# --------------------------------------------------------------------------- #
async def test_functional_text_versions_and_roundtrip(tmp_path: Path) -> None:
    path = await _setup(tmp_path)
    common = {"app_name": "myapp", "user_id": "u1", "session_id": "s", "filename": "note.txt"}

    v0 = await A.save(path=path, text="first version", **common)
    assert v0["ok"] is True
    assert v0["data"]["version"] == 0

    v1 = await A.save(path=path, text="second version", **common)
    assert v1["data"]["version"] == 1

    # Load latest → second version, exact content.
    latest = await A.load(path=path, **common)
    assert latest["ok"] is True
    assert latest["data"]["encoding"] == "text"
    assert latest["data"]["text"] == "second version"
    assert latest["data"]["mime_type"] == "text/plain"

    # Load specific version 0 → first version.
    first = await A.load(path=path, version=0, **common)
    assert first["data"]["text"] == "first version"

    # list shows the filename.
    listed = await A.list_artifacts_tool(path=path, app_name="myapp", user_id="u1", session_id="s")
    assert listed["ok"] is True
    assert "note.txt" in listed["data"]["filenames"]

    # versions lists [0, 1].
    vers = await A.versions(path=path, **common)
    assert vers["ok"] is True
    assert vers["data"]["versions"] == [0, 1]

    # delete removes it.
    deleted = await A.delete(path=path, **common)
    assert deleted["ok"] is True
    assert deleted["data"]["deleted"] == "note.txt"

    gone = await A.load(path=path, **common)
    assert gone["ok"] is False
    listed_after = await A.list_artifacts_tool(
        path=path, app_name="myapp", user_id="u1", session_id="s"
    )
    assert "note.txt" not in listed_after["data"]["filenames"]


async def test_functional_user_prefixed_filename_accepted(tmp_path: Path) -> None:
    """Un nom ``user:``-préfixé (user-scoped) est accepté et round-trip correctement."""
    path = await _setup(tmp_path)
    common = {
        "app_name": "myapp",
        "user_id": "u1",
        "session_id": "s",
        "filename": "user:profile.txt",
    }
    saved = await A.save(path=path, text="user profile data", **common)
    assert saved["ok"] is True
    assert saved["data"]["version"] == 0

    loaded = await A.load(path=path, **common)
    assert loaded["ok"] is True
    assert loaded["data"]["text"] == "user profile data"


async def test_functional_base64_binary_roundtrip(tmp_path: Path) -> None:
    """Round-trip d'un artifact binaire (mime non texte) via base64."""
    path = await _setup(tmp_path)
    raw = bytes(range(256))  # données binaires arbitraires
    b64 = base64.b64encode(raw).decode("ascii")
    common = {"app_name": "myapp", "user_id": "u1", "session_id": "s", "filename": "blob.bin"}

    saved = await A.save(path=path, bytes_b64=b64, mime_type="application/octet-stream", **common)
    assert saved["ok"] is True
    assert saved["data"]["version"] == 0
    assert saved["data"]["mime_type"] == "application/octet-stream"

    loaded = await A.load(path=path, **common)
    assert loaded["ok"] is True
    assert loaded["data"]["encoding"] == "base64"
    assert loaded["data"]["mime_type"] == "application/octet-stream"
    assert loaded["data"]["text"] is None
    # Décodage → octets identiques.
    assert base64.b64decode(loaded["data"]["bytes_b64"]) == raw


async def test_functional_versions_empty_for_unknown(tmp_path: Path) -> None:
    """``versions`` sur un fichier inconnu renvoie une liste vide (pas d'erreur)."""
    path = await _setup(tmp_path)
    vers = await A.versions(
        path=path, app_name="myapp", user_id="u1", session_id="s", filename="ghost.txt"
    )
    assert vers["ok"] is True
    assert vers["data"]["versions"] == []


# --------------------------------------------------------------------------- #
# GCS branch: validate config + actionable error when extra absent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(_HAS_GCS, reason="extra gcp présent : pas d'erreur de dépendance")
async def test_gcs_save_errors_without_extra(tmp_path: Path) -> None:
    """Backend gcs sans l'extra gcp → save renvoie une err orientée action (pas d'exception)."""
    path = str(tmp_path)
    A.service_set(path=path, app_name="myapp", kind="gcs", bucket="my-bucket")
    result = await A.save(
        path=path, app_name="myapp", user_id="u1", session_id="s", filename="f.txt", text="hi"
    )
    assert result["ok"] is False
    assert "gcp" in result["error"]


# --------------------------------------------------------------------------- #
# In-memory fastmcp.Client read-through (exposed names + double-prefix guard)
# --------------------------------------------------------------------------- #
async def test_client_read_through_artifacts_flow(tmp_path: Path) -> None:
    """artifacts_service_set → artifacts_save → artifacts_load → artifacts_list."""
    mcp = build_server()
    path = str(tmp_path)
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert "artifacts_service_set" in names
        assert "artifacts_save" in names
        assert "artifacts_list" in names  # enregistré sous le nom bare `list`
        assert "artifacts_versions" in names
        assert not any(n.startswith("artifacts_artifacts_") for n in names)

        await client.call_tool(
            "artifacts_service_set", {"path": path, "app_name": "myapp", "kind": "in_memory"}
        )
        saved = await client.call_tool(
            "artifacts_save",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": "s",
                "filename": "hello.txt",
                "text": "hello via client",
            },
        )
        assert saved.data["ok"] is True
        assert saved.data["data"]["version"] == 0

        loaded = await client.call_tool(
            "artifacts_load",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": "s",
                "filename": "hello.txt",
            },
        )
        assert loaded.data["ok"] is True
        assert loaded.data["data"]["text"] == "hello via client"

        listed = await client.call_tool(
            "artifacts_list",
            {"path": path, "app_name": "myapp", "user_id": "u1", "session_id": "s"},
        )
        assert listed.data["ok"] is True
        assert "hello.txt" in listed.data["data"]["filenames"]
