from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains.project import (
    project_add_extra,
    project_agent_config,
    project_create,
    project_inspect,
    project_set_env,
)
from adk_toolkit_mcp.server import build_server


# --------------------------------------------------------------------------- #
# project_create
# --------------------------------------------------------------------------- #
def test_project_create_writes_expected_files_ai_studio(tmp_path: Path) -> None:
    res = project_create(str(tmp_path), "demo_app")
    assert res["ok"] is True
    app = tmp_path / "demo_app"

    init_txt = (app / "__init__.py").read_text(encoding="utf-8")
    assert init_txt == "from . import agent\n"

    agent_txt = (app / "agent.py").read_text(encoding="utf-8")
    assert "from google.adk.agents import LlmAgent" in agent_txt
    assert "root_agent = LlmAgent(" in agent_txt
    assert "name='demo_app'" in agent_txt or 'name="demo_app"' in agent_txt
    assert "gemini-2.5-flash" in agent_txt

    env_txt = (app / ".env").read_text(encoding="utf-8")
    assert "GOOGLE_GENAI_USE_VERTEXAI=FALSE" in env_txt
    assert "GOOGLE_API_KEY=" in env_txt
    assert "GOOGLE_CLOUD_PROJECT" not in env_txt

    # Returned paths point at the three real files.
    created = res["data"]["created"]
    assert any(p.endswith("agent.py") for p in created)
    assert any(p.endswith("__init__.py") for p in created)
    assert any(p.endswith(".env") for p in created)


def test_project_create_vertex_backend_env(tmp_path: Path) -> None:
    res = project_create(str(tmp_path), "vtx", model="gemini-2.0-pro", backend="vertex")
    assert res["ok"] is True
    env_txt = (tmp_path / "vtx" / ".env").read_text(encoding="utf-8")
    assert "GOOGLE_GENAI_USE_VERTEXAI=TRUE" in env_txt
    assert "GOOGLE_CLOUD_PROJECT=" in env_txt
    assert "GOOGLE_CLOUD_LOCATION=" in env_txt
    assert "GOOGLE_API_KEY" not in env_txt
    assert "gemini-2.0-pro" in (tmp_path / "vtx" / "agent.py").read_text(encoding="utf-8")


def test_project_create_is_idempotent(tmp_path: Path) -> None:
    first = project_create(str(tmp_path), "demo_app")
    assert first["data"]["changed"] is True
    second = project_create(str(tmp_path), "demo_app")
    assert second["ok"] is True
    # Second call writes identical content -> nothing changed.
    assert second["data"]["changed"] is False


def test_project_create_rejects_bad_backend(tmp_path: Path) -> None:
    res = project_create(str(tmp_path), "demo_app", backend="nope")  # type: ignore[arg-type]
    assert res["ok"] is False
    assert res["error"]


def test_project_create_rejects_empty_app_name(tmp_path: Path) -> None:
    res = project_create(str(tmp_path), "   ")
    assert res["ok"] is False
    assert res["error"]


def test_project_create_rejects_invalid_app_name(tmp_path: Path) -> None:
    res = project_create(str(tmp_path), "bad name!")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# project_inspect
# --------------------------------------------------------------------------- #
def test_project_inspect_detects_root_agent_and_env(tmp_path: Path) -> None:
    project_create(str(tmp_path), "demo_app")
    app = tmp_path / "demo_app"
    res = project_inspect(str(app))
    assert res["ok"] is True
    data = res["data"]
    assert data["has_root_agent"] is True
    assert "agent.py" in data["py_files"]
    assert "__init__.py" in data["py_files"]
    # .env keys reported, values redacted.
    assert "GOOGLE_GENAI_USE_VERTEXAI" in data["env_keys"]
    assert "GOOGLE_API_KEY" in data["env_keys"]


def test_project_inspect_no_root_agent(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("x = 1\n", encoding="utf-8")
    res = project_inspect(str(tmp_path))
    assert res["ok"] is True
    assert res["data"]["has_root_agent"] is False


def test_project_inspect_missing_path_errors(tmp_path: Path) -> None:
    res = project_inspect(str(tmp_path / "does_not_exist"))
    assert res["ok"] is False
    assert res["error"]


def test_project_inspect_env_ignores_comments_and_blanks(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# a comment\n\nGOOGLE_API_KEY=secret\nMALFORMED_NO_EQUALS\n",
        encoding="utf-8",
    )
    res = project_inspect(str(tmp_path))
    assert res["ok"] is True
    keys = res["data"]["env_keys"]
    assert keys == ["GOOGLE_API_KEY"]


# --------------------------------------------------------------------------- #
# project_set_env
# --------------------------------------------------------------------------- #
def test_project_set_env_merges_without_clobbering(tmp_path: Path) -> None:
    project_create(str(tmp_path), "demo_app")
    app = tmp_path / "demo_app"
    res = project_set_env(str(app), {"GOOGLE_API_KEY": "sk-123", "EXTRA_FLAG": "yes"})
    assert res["ok"] is True
    env_txt = (app / ".env").read_text(encoding="utf-8")
    # Pre-existing unrelated key preserved.
    assert "GOOGLE_GENAI_USE_VERTEXAI=FALSE" in env_txt
    # Updated/added keys present.
    assert "GOOGLE_API_KEY=sk-123" in env_txt
    assert "EXTRA_FLAG=yes" in env_txt
    # Returned keys are redacted (no secret value leaks).
    assert "GOOGLE_API_KEY" in res["data"]["env_keys"]
    assert "sk-123" not in str(res["data"])


def test_project_set_env_creates_env_when_absent(tmp_path: Path) -> None:
    res = project_set_env(str(tmp_path), {"FOO": "bar"})
    assert res["ok"] is True
    assert (tmp_path / ".env").exists()
    assert "FOO=bar" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_project_set_env_rejects_empty_values(tmp_path: Path) -> None:
    res = project_set_env(str(tmp_path), {})
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# project_add_extra
# --------------------------------------------------------------------------- #
def test_project_add_extra_updates_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = [\n    "google-adk>=2.0",\n]\n',
        encoding="utf-8",
    )
    res = project_add_extra(str(tmp_path), "bigquery")
    assert res["ok"] is True
    txt = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert "google-adk[bigquery]" in txt


def test_project_add_extra_rejects_unknown(tmp_path: Path) -> None:
    res = project_add_extra(str(tmp_path), "definitely-not-real")
    assert res["ok"] is False
    assert res["error"]


def test_project_add_extra_idempotent(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = [\n    "google-adk[bigquery]>=2.0",\n]\n',
        encoding="utf-8",
    )
    res = project_add_extra(str(tmp_path), "bigquery")
    assert res["ok"] is True
    assert res["data"]["changed"] is False


def test_project_add_extra_grafts_onto_existing_bracket(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = [\n    "google-adk[gcp]>=2.0",\n]\n',
        encoding="utf-8",
    )
    res = project_add_extra(str(tmp_path), "bigquery")
    assert res["ok"] is True
    assert res["data"]["changed"] is True
    txt = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    # Both extras coexist in one bracket.
    assert "gcp" in txt and "bigquery" in txt
    assert "google-adk[gcp,bigquery]" in txt


def test_project_add_extra_inserts_when_no_adk_line(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = [\n    "pydantic>=2",\n]\n',
        encoding="utf-8",
    )
    res = project_add_extra(str(tmp_path), "eval")
    assert res["ok"] is True
    assert res["data"]["changed"] is True
    assert "google-adk[eval]" in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")


def test_project_add_extra_pyproject_without_dependencies_no_change(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    res = project_add_extra(str(tmp_path), "spanner")
    assert res["ok"] is True
    # No dependencies array and no google-adk line -> nothing to change.
    assert res["data"]["changed"] is False


def test_project_add_extra_requirements_fallback(tmp_path: Path) -> None:
    res = project_add_extra(str(tmp_path), "mcp")
    assert res["ok"] is True
    req = tmp_path / "requirements.txt"
    assert req.exists()
    assert "google-adk[mcp]" in req.read_text(encoding="utf-8")


def test_project_add_extra_requirements_appends_to_existing(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("pydantic>=2\n", encoding="utf-8")
    res = project_add_extra(str(tmp_path), "a2a")
    assert res["ok"] is True
    txt = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "pydantic>=2" in txt
    assert "google-adk[a2a]" in txt


def test_project_add_extra_requirements_idempotent(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("google-adk[litellm]\n", encoding="utf-8")
    res = project_add_extra(str(tmp_path), "litellm")
    assert res["ok"] is True
    assert res["data"]["changed"] is False


def test_project_add_extra_missing_path_errors(tmp_path: Path) -> None:
    res = project_add_extra(str(tmp_path / "nope"), "gcp")
    assert res["ok"] is False


def test_project_set_env_missing_path_errors(tmp_path: Path) -> None:
    res = project_set_env(str(tmp_path / "nope"), {"FOO": "bar"})
    assert res["ok"] is False


def test_project_set_env_rejects_blank_key(tmp_path: Path) -> None:
    res = project_set_env(str(tmp_path), {"  ": "bar"})
    assert res["ok"] is False


def test_project_inspect_rejects_file_path(tmp_path: Path) -> None:
    f = tmp_path / "afile.txt"
    f.write_text("x", encoding="utf-8")
    res = project_inspect(str(f))
    assert res["ok"] is False


def test_project_agent_config_missing_path_errors(tmp_path: Path) -> None:
    res = project_agent_config(str(tmp_path / "nope"))
    assert res["ok"] is False


def test_project_create_rejects_empty_model(tmp_path: Path) -> None:
    res = project_create(str(tmp_path), "demo_app", model="  ")
    assert res["ok"] is False


# --------------------------------------------------------------------------- #
# project_agent_config
# --------------------------------------------------------------------------- #
def test_project_agent_config_writes_yaml(tmp_path: Path) -> None:
    yaml = "name: root_agent\nmodel: gemini-2.5-flash\n"
    res = project_agent_config(str(tmp_path), yaml_content=yaml)
    assert res["ok"] is True
    cfg = tmp_path / "root_agent.yaml"
    assert cfg.exists()
    assert cfg.read_text(encoding="utf-8") == yaml
    assert res["data"]["exists"] is True


def test_project_agent_config_reports_when_absent(tmp_path: Path) -> None:
    res = project_agent_config(str(tmp_path))
    assert res["ok"] is True
    assert res["data"]["exists"] is False
    assert res["data"]["path"].endswith("root_agent.yaml")


# --------------------------------------------------------------------------- #
# Mount wiring (in-memory client read-through)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_project_create_through_mounted_client(tmp_path: Path) -> None:
    mcp = build_server()
    async with Client(mcp) as client:
        result = await client.call_tool(
            "project_project_create",
            {"path": str(tmp_path), "app_name": "client_app"},
        )
    assert result.data["ok"] is True
    assert (tmp_path / "client_app" / "agent.py").exists()
