"""`artifacts` domain: operates ADK's runtime ARTIFACTS service (P2b).

Like `sessions`/`memory`, this domain **instantiates a real ADK artifact service** and calls it
asynchronously. The concrete service (``InMemoryArtifactService`` / ``GcsArtifactService``) is
chosen by the backend persisted in ``<app_dir>/.adk_toolkit/runtime.json`` and provided by the
singleton factory :mod:`adk_toolkit_mcp.runtime` (the ``in_memory`` instance is shared across tool
calls).

A FastMCP sub-server mounted under ``namespace="artifacts"`` → tools exposed as
``artifacts_<name>``. Functions with **BARE** names (``service_set``, ``save``, ``load``,
``delete``, ``versions``). ``list`` is a Python builtin: the function is called
``list_artifacts_tool`` but is registered under the bare tool name ``list`` → exposed as
``artifacts_list`` on the client side.

ADK reminder (cf. ``docs/adk-api-notes/memory-artifacts.md``):
- ``save_artifact(*, app_name, user_id, session_id, filename, artifact) -> int`` (0-indexed
  version); ``artifact`` is a ``types.Part`` (built via ``Part.from_text`` / ``Part.from_bytes``).
- ``load_artifact(..., version=None) -> Optional[Part]``: ``.text`` for text, otherwise
  ``.inline_data`` (``data`` bytes + ``mime_type``). ``None`` ⇒ artifact absent.
- ``list_artifact_keys`` / ``list_versions`` / ``delete_artifact``.
- A filename prefixed with ``user:`` makes the artifact **user-scoped** (shared across sessions).

Each tool returns the ``{ok, data, error}`` envelope; invalid inputs (e.g. neither ``text`` nor
``bytes_b64``, or both), a corrupt config and absent artifacts return ``err(...)`` (never an
exception that propagates).
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

if TYPE_CHECKING:  # pragma: no cover - hints only
    from google.adk.artifacts import BaseArtifactService
    from google.genai import types

artifacts_server: FastMCP = FastMCP("artifacts")


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _artifact_service_for(path: str, app_name: str) -> BaseArtifactService | dict[str, Any]:
    """Return the (cached) artifact service configured for the app, or an ``err(...)``.

    ``err`` if the config is corrupt, if no artifacts backend has been chosen
    (``artifacts_service_set`` not called), or if the backend is invalid (missing required field /
    missing ``gcp`` extra).
    """
    ws = _app_ws(path, app_name)
    try:
        config = load_runtime_config(ws, app_name)
    except ValueError as exc:
        return err(str(exc))
    if config.artifacts is None:
        return err("No artifact service configured for this app. Call artifacts_service_set first.")
    try:
        return get_artifact_service(config.artifacts)
    except ValueError as exc:
        return err(str(exc))


def _part_to_payload(part: types.Part, version: int | None) -> dict[str, Any]:
    """Serialize a loaded ``Part`` into a payload: text if available, otherwise base64 + mime.

    - Text part (``part.text``) → ``{"text": …, "mime_type": "text/plain", "encoding": "text"}``.
    - Binary part (``part.inline_data``) → ``{"bytes_b64": …, "mime_type": …, "encoding":
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
    # Part with neither text nor inline data (degenerate case): return a minimal envelope.
    return {
        "version": version,
        "encoding": "empty",
        "mime_type": None,
        "text": None,
        "bytes_b64": None,
    }


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@artifacts_server.tool(tags={"artifacts"})
def service_set(path: str, app_name: str, kind: str, bucket: str | None = None) -> dict[str, Any]:
    """Choose and persist the app's artifact service backend (``runtime.json``).

    ``kind`` ∈ {``in_memory``, ``gcs``}. ``gcs`` requires ``bucket`` (``gcp`` extra).

    Does NOT instantiate the service (shape validation only); preserves the session and memory
    backends already written. Returns the persisted artifacts config.
    """
    if kind not in ARTIFACT_KINDS:
        return err(f"Invalid kind: {kind!r}. Expected one of: {', '.join(sorted(ARTIFACT_KINDS))}.")
    if kind == "gcs" and not (bucket and bucket.strip()):
        return err("kind='gcs' requires 'bucket' (the GCS bucket name).")

    ws = _app_ws(path, app_name)
    backend = ArtifactBackend(kind=kind, bucket=bucket)  # type: ignore[arg-type]  # validated
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


@artifacts_server.tool(tags={"artifacts"})
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
    """Save an artifact (new version) and return its version number (int).

    Provide EXACTLY one of:
    - ``text``: builds ``Part.from_text(text=…)`` (``mime_type`` ignored, always text);
    - ``bytes_b64``: base64 decoded → ``Part.from_bytes(data=…, mime_type=…)``.

    A ``filename`` prefixed with ``user:`` makes the artifact user-scoped (shared across sessions).
    """
    if not filename.strip():
        return err("filename is empty.")
    if (text is None) == (bytes_b64 is None):
        return err("Provide EXACTLY one of 'text' or 'bytes_b64' (not both, not neither).")

    service = _artifact_service_for(path, app_name)
    if isinstance(service, dict):
        return service

    from google.genai import types

    if text is not None:
        part = types.Part.from_text(text=text)
        stored_mime = "text/plain"
    else:
        assert bytes_b64 is not None  # guaranteed by the XOR validation above
        try:
            raw = base64.b64decode(bytes_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            return err(f"bytes_b64 is not valid base64: {exc}")
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


@artifacts_server.tool(tags={"artifacts"})
async def load(
    path: str,
    app_name: str,
    user_id: str,
    session_id: str,
    filename: str,
    version: int | None = None,
) -> dict[str, Any]:
    """Load an artifact (latest version by default, or a specific ``version``).

    Returns ``{version, encoding, mime_type, text, bytes_b64}``: ``text`` if the part is textual,
    otherwise ``bytes_b64`` (base64) + ``mime_type``. An absent artifact → ``err(...)``.
    """
    if not filename.strip():
        return err("filename is empty.")

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
            f"Artifact not found: {filename!r} "
            f"(app={app_name}, user={user_id}, session={session_id}, version={version})."
        )

    payload = _part_to_payload(part, version)
    payload.update({"app_name": app_name, "filename": filename})
    return ok(payload)


@artifacts_server.tool(tags={"artifacts"}, name="list")
async def list_artifacts_tool(
    path: str, app_name: str, user_id: str, session_id: str
) -> dict[str, Any]:
    """List the artifact names for ``(app_name, user_id, session_id)``.

    Named ``list_artifacts_tool`` in Python (``list`` is a builtin) but registered under the bare
    tool name ``list`` → exposed as ``artifacts_list`` on the client side.
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


@artifacts_server.tool(tags={"artifacts"})
async def delete(
    path: str, app_name: str, user_id: str, session_id: str, filename: str
) -> dict[str, Any]:
    """Delete all versions of an artifact. Returns the deleted name (idempotent on the service
    side)."""
    if not filename.strip():
        return err("filename is empty.")

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


@artifacts_server.tool(tags={"artifacts"})
async def versions(
    path: str, app_name: str, user_id: str, session_id: str, filename: str
) -> dict[str, Any]:
    """Return the list of an artifact's version numbers (``list_versions``)."""
    if not filename.strip():
        return err("filename is empty.")

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
