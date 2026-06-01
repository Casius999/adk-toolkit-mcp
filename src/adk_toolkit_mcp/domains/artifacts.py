"""Domaine `artifacts` : opère le service d'ARTIFACTS runtime d'ADK (P2b).

Comme `sessions`/`memory`, ce domaine **instancie un vrai service d'artifacts ADK** et
l'appelle de façon asynchrone. Le service concret (``InMemoryArtifactService`` /
``GcsArtifactService``) est choisi par le backend persisté dans
``<app_dir>/.adk_toolkit/runtime.json`` et fourni par la fabrique singleton
:mod:`adk_toolkit_mcp.runtime` (l'instance ``in_memory`` est partagée entre appels d'outils).

Sous-serveur FastMCP monté sous ``namespace="artifacts"`` → outils exposés ``artifacts_<nom>``.
Fonctions à noms **BARE** (``service_set``, ``save``, ``load``, ``delete``, ``versions``).
``list`` est un builtin Python : la fonction s'appelle ``list_artifacts_tool`` mais est
enregistrée sous le nom d'outil bare ``list`` → exposée ``artifacts_list`` côté client.

Rappel ADK (cf. ``docs/adk-api-notes/memory-artifacts.md``) :
- ``save_artifact(*, app_name, user_id, session_id, filename, artifact) -> int`` (version
  0-indexée) ; ``artifact`` est une ``types.Part`` (construite via ``Part.from_text`` /
  ``Part.from_bytes``).
- ``load_artifact(..., version=None) -> Optional[Part]`` : ``.text`` pour du texte, sinon
  ``.inline_data`` (``data`` bytes + ``mime_type``). ``None`` ⇒ artifact absent.
- ``list_artifact_keys`` / ``list_versions`` / ``delete_artifact``.
- Un nom de fichier préfixé ``user:`` rend l'artifact **user-scoped** (partagé entre sessions).

Chaque outil renvoie l'enveloppe ``{ok, data, error}`` ; les entrées invalides (ex. ni
``text`` ni ``bytes_b64``, ou les deux), la config corrompue et les artifacts absents
renvoient ``err(...)`` (jamais d'exception qui remonte).
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..runtime import (
    ARTIFACT_KINDS,
    ArtifactBackend,
    RuntimeConfig,
    get_artifact_service,
    load_runtime_config,
    save_runtime_config,
)
from ..workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints seulement
    from google.adk.artifacts import BaseArtifactService
    from google.genai import types

artifacts_server: FastMCP = FastMCP("artifacts")


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _artifact_service_for(path: str, app_name: str) -> BaseArtifactService | dict[str, Any]:
    """Renvoie le service d'artifacts (caché) configuré pour l'app, ou un ``err(...)``.

    ``err`` si la config est corrompue, si aucun backend artifacts n'a été choisi
    (``artifacts_service_set`` non appelé), ou si le backend est invalide (champ requis
    manquant / extra ``gcp`` absent).
    """
    ws = _app_ws(path, app_name)
    try:
        config = load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))
    if config.artifacts is None:
        return err(
            "Aucun service d'artifacts configuré pour cette app. "
            "Appelle d'abord artifacts_service_set."
        )
    try:
        return get_artifact_service(config.artifacts)
    except ValueError as exc:
        return err(str(exc))


def _part_to_payload(part: types.Part, version: int | None) -> dict[str, Any]:
    """Sérialise une ``Part`` chargée en payload : texte si dispo, sinon base64 + mime.

    - Partie texte (``part.text``) → ``{"text": …, "mime_type": "text/plain", "encoding":
      "text"}``.
    - Partie binaire (``part.inline_data``) → ``{"bytes_b64": …, "mime_type": …, "encoding":
      "base64"}``.
    """
    inline = getattr(part, "inline_data", None)
    if part.text is not None:
        return {
            "version": version,
            "encoding": "text",
            "mime_type": "text/plain",
            "text": part.text,
            "bytes_b64": None,
        }
    if inline is not None and inline.data is not None:
        return {
            "version": version,
            "encoding": "base64",
            "mime_type": inline.mime_type,
            "text": None,
            "bytes_b64": base64.b64encode(inline.data).decode("ascii"),
        }
    # Part sans texte ni données inline (cas dégénéré) : renvoie une enveloppe minimale.
    return {
        "version": version,
        "encoding": "empty",
        "mime_type": None,
        "text": None,
        "bytes_b64": None,
    }


# --------------------------------------------------------------------------- #
# Outils MCP
# --------------------------------------------------------------------------- #
@artifacts_server.tool
def service_set(path: str, app_name: str, kind: str, bucket: str | None = None) -> dict[str, Any]:
    """Choisit et persiste le backend du service d'artifacts de l'app (``runtime.json``).

    ``kind`` ∈ {``in_memory``, ``gcs``}. ``gcs`` exige ``bucket`` (extra ``gcp``).

    N'instancie PAS le service (validation de forme seulement) ; préserve les backends session
    et memory déjà écrits. Renvoie la config artifacts persistée.
    """
    if kind not in ARTIFACT_KINDS:
        return err(
            f"kind invalide : {kind!r}. Attendu l'un de : {', '.join(sorted(ARTIFACT_KINDS))}."
        )
    if kind == "gcs" and not (bucket and bucket.strip()):
        return err("kind='gcs' nécessite 'bucket' (nom du bucket GCS).")

    ws = _app_ws(path, app_name)
    backend = ArtifactBackend(kind=kind, bucket=bucket)  # type: ignore[arg-type]  # validé
    try:
        existing = load_runtime_config(ws, app_name)
    except ValueError:
        existing = RuntimeConfig()
    config = RuntimeConfig(session=existing.session, memory=existing.memory, artifacts=backend)
    changed = save_runtime_config(ws, config)

    return ok(
        {
            "app_name": app_name,
            "kind": backend.kind,
            "bucket": backend.bucket,
            "config_path": str(ws.path(".adk_toolkit/runtime.json")),
            "changed": changed,
        }
    )


@artifacts_server.tool
async def save(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    filename: str,
    text: str | None = None,
    bytes_b64: str | None = None,
    mime_type: str = "text/plain",
) -> dict[str, Any]:
    """Enregistre un artifact (nouvelle version) et renvoie son numéro de version (int).

    Fournir EXACTEMENT un parmi :
    - ``text`` : construit ``Part.from_text(text=…)`` (``mime_type`` ignoré, toujours texte) ;
    - ``bytes_b64`` : base64 décodé → ``Part.from_bytes(data=…, mime_type=…)``.

    Un ``filename`` préfixé ``user:`` rend l'artifact user-scoped (partagé entre sessions).
    """
    if not filename.strip():
        return err("filename est vide.")
    if (text is None) == (bytes_b64 is None):
        return err("Fournis EXACTEMENT un de 'text' ou 'bytes_b64' (pas les deux, pas aucun).")

    service = _artifact_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    from google.genai import types

    if text is not None:
        part = types.Part.from_text(text=text)
        stored_mime = "text/plain"
    else:
        assert bytes_b64 is not None  # garanti par la validation XOR ci-dessus
        try:
            raw = base64.b64decode(bytes_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            return err(f"bytes_b64 n'est pas du base64 valide : {exc}")
        part = types.Part.from_bytes(data=raw, mime_type=mime_type)
        stored_mime = mime_type

    version = await service.save_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        artifact=part,
    )
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "filename": filename,
            "version": version,
            "mime_type": stored_mime,
        }
    )


@artifacts_server.tool
async def load(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    filename: str,
    version: int | None = None,
) -> dict[str, Any]:
    """Charge un artifact (dernière version par défaut, ou ``version`` précise).

    Renvoie ``{version, encoding, mime_type, text, bytes_b64}`` : ``text`` si la part est
    textuelle, sinon ``bytes_b64`` (base64) + ``mime_type``. Un artifact absent → ``err(...)``.
    """
    if not filename.strip():
        return err("filename est vide.")

    service = _artifact_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    part = await service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
        version=version,
    )
    if part is None:
        return err(
            f"Artifact introuvable : {filename!r} "
            f"(app={app_name}, user={user_id}, session={session_id}, version={version})."
        )

    payload = _part_to_payload(part, version)
    payload.update({"app_name": app_name, "filename": filename})
    return ok(payload)


@artifacts_server.tool(name="list")
async def list_artifacts_tool(
    path: str, app_name: str, user_id: str, session_id: str
) -> dict[str, Any]:
    """Liste les noms d'artifacts pour ``(app_name, user_id, session_id)``.

    Nommée ``list_artifacts_tool`` en Python (``list`` est un builtin) mais enregistrée sous le
    nom d'outil bare ``list`` → exposée ``artifacts_list`` côté client.
    """
    service = _artifact_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    keys = await service.list_artifact_keys(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "filenames": list(keys),
        }
    )


@artifacts_server.tool
async def delete(
    path: str, app_name: str, user_id: str, session_id: str, filename: str
) -> dict[str, Any]:
    """Supprime toutes les versions d'un artifact. Renvoie le nom supprimé (idempotent côté
    service)."""
    if not filename.strip():
        return err("filename est vide.")

    service = _artifact_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    await service.delete_artifact(
        app_name=app_name, user_id=user_id, session_id=session_id, filename=filename
    )
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "deleted": filename,
        }
    )


@artifacts_server.tool
async def versions(
    path: str, app_name: str, user_id: str, session_id: str, filename: str
) -> dict[str, Any]:
    """Renvoie la liste des numéros de version d'un artifact (``list_versions``)."""
    if not filename.strip():
        return err("filename est vide.")

    service = _artifact_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    vers = await service.list_versions(
        app_name=app_name, user_id=user_id, session_id=session_id, filename=filename
    )
    return ok(
        {
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "filename": filename,
            "versions": list(vers),
        }
    )
