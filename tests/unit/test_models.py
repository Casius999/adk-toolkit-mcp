"""Unit tests for the ``models`` domain (P1 domain 4/4).

Covers:
- ``set`` (bare ``models_set``): sets a Gemini string model on an agent.
- ``configure_litellm`` (bare ``models_configure_litellm``): configures LiteLlm.
- ``generate_config`` (bare ``models_generate_config``): configures GenerateContentConfig.
- Source rendering: LiteLlm / os / types imports, api_key never hardcoded.
- ruff format stability.
- Functional probe: a Gemini agent + generate_content_config importable in a subprocess.
- LiteLlm probe (skipped if litellm is absent in CI).
- ``adk://models`` resource (read via in-memory Client).
- Client read-through: ``models_configure_litellm`` returns ``{ok: True}``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from adk_toolkit_mcp.domains.models import models_server
from adk_toolkit_mcp.project_model import (
    HARM_BLOCK_THRESHOLDS,
    HARM_CATEGORIES,
    LITELLM_PROVIDERS,
    AgentSpec,
    GenerateContentConfigSpec,
    LiteLlmSpec,
    ProjectModel,
    SafetySettingSpec,
    render_agent_module,
)
from adk_toolkit_mcp.resources import register_resources
from adk_toolkit_mcp.server import build_server


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    """Locate the ruff executable in the current environment."""
    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _assert_ruff_format_stable(src: str, tmp_path: Path, label: str) -> None:
    gen_file = tmp_path / f"{label}.py"
    gen_file.write_text(src, encoding="utf-8")
    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff not found in the environment — format test ignored")
    result = subprocess.run(
        [ruff, "format", "--check", str(gen_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff format --check failed for case '{label}'.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Generated source:\n{src}"
    )


def _simple_llm_model(
    agent_name: str = "my_agent",
    model: str = "gemini-2.5-flash",
    model_spec: LiteLlmSpec | None = None,
    generate_content_config: GenerateContentConfigSpec | None = None,
) -> ProjectModel:
    spec = AgentSpec(
        name=agent_name,
        type="llm",
        model=model,
        model_spec=model_spec,
        generate_content_config=generate_content_config,
    )
    model_obj = ProjectModel(app_name="myapp", agents=(spec,), root=agent_name)
    return model_obj


# --------------------------------------------------------------------------- #
# Constants + dataclasses
# --------------------------------------------------------------------------- #
def test_litellm_providers_set_is_correct() -> None:
    assert "openai" in LITELLM_PROVIDERS
    assert "anthropic" in LITELLM_PROVIDERS
    assert "lm_studio" in LITELLM_PROVIDERS
    assert "ollama" in LITELLM_PROVIDERS


def test_harm_categories_not_empty() -> None:
    assert "HARM_CATEGORY_HARASSMENT" in HARM_CATEGORIES
    assert "HARM_CATEGORY_DANGEROUS_CONTENT" in HARM_CATEGORIES


def test_harm_block_thresholds_not_empty() -> None:
    assert "BLOCK_NONE" in HARM_BLOCK_THRESHOLDS
    assert "BLOCK_MEDIUM_AND_ABOVE" in HARM_BLOCK_THRESHOLDS


def test_litellm_spec_roundtrip() -> None:
    spec = LiteLlmSpec(provider="openai", model="gpt-4o", api_base="http://x/v1", api_key_env="K")
    restored = LiteLlmSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_safety_setting_spec_roundtrip() -> None:
    ss = SafetySettingSpec(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE")
    restored = SafetySettingSpec.from_dict(ss.to_dict())
    assert restored == ss


def test_generate_content_config_spec_roundtrip() -> None:
    gcc = GenerateContentConfigSpec(
        temperature=0.7,
        max_output_tokens=2048,
        top_p=0.9,
        top_k=40.0,
        safety_settings=(SafetySettingSpec("HARM_CATEGORY_HARASSMENT", "BLOCK_MEDIUM_AND_ABOVE"),),
        response_modalities=("TEXT",),
    )
    restored = GenerateContentConfigSpec.from_dict(gcc.to_dict())
    assert restored == gcc


def test_agent_spec_with_model_spec_and_gcc_roundtrip() -> None:
    spec = AgentSpec(
        name="a",
        type="llm",
        model_spec=LiteLlmSpec(provider="openai", model="gpt-4o"),
        generate_content_config=GenerateContentConfigSpec(temperature=0.5),
    )
    restored = AgentSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.model_spec is not None
    assert restored.model_spec.provider == "openai"
    assert restored.generate_content_config is not None
    assert restored.generate_content_config.temperature == 0.5


# --------------------------------------------------------------------------- #
# Gemini string rendering (backward compat)
# --------------------------------------------------------------------------- #
def test_render_gemini_string_model_unchanged() -> None:
    model_obj = _simple_llm_model(model="gemini-2.5-pro")
    src = render_agent_module(model_obj)
    assert 'model="gemini-2.5-pro"' in src
    # No LiteLlm or types import if only a Gemini string.
    assert "LiteLlm" not in src
    assert "from google.genai import types" not in src


# --------------------------------------------------------------------------- #
# LiteLlm rendering
# --------------------------------------------------------------------------- #
def test_render_litellm_basic_openai() -> None:
    model_obj = _simple_llm_model(model_spec=LiteLlmSpec(provider="openai", model="gpt-4o"))
    src = render_agent_module(model_obj)
    assert "from google.adk.models.lite_llm import LiteLlm" in src
    assert 'model=LiteLlm(model="openai/gpt-4o")' in src
    # No hardcoded api_key.
    assert "api_key=" not in src


def test_render_litellm_with_api_base() -> None:
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(provider="openai", model="gpt-4o", api_base="https://my.api/v1")
    )
    src = render_agent_module(model_obj)
    assert 'api_base="https://my.api/v1"' in src
    assert "api_key=" not in src


def test_render_litellm_api_key_uses_os_getenv() -> None:
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(provider="openai", model="gpt-4o", api_key_env="MY_API_KEY")
    )
    src = render_agent_module(model_obj)
    # The key is read via os.getenv, never hardcoded.
    assert 'api_key=os.getenv("MY_API_KEY")' in src
    assert "import os" in src
    # No literal key value.
    assert "api_key=" in src  # present...
    # ...but only as os.getenv (no literal key string).
    import re

    assert not re.search(r'api_key\s*=\s*"[a-zA-Z0-9_-]+"', src)


def test_render_litellm_no_hardcoded_api_key_ever() -> None:
    """Security invariant: no key must ever appear hardcoded."""
    for provider in LITELLM_PROVIDERS:
        spec = LiteLlmSpec(provider=provider, model="model-x")
        model_obj = _simple_llm_model(model_spec=spec)
        src = render_agent_module(model_obj)
        # Without api_key_env: no api_key= at all.
        assert "api_key=" not in src, (
            f"api_key found in the generated code for provider={provider!r} without api_key_env"
        )


def test_render_litellm_lm_studio_defaults_provider_and_api_base() -> None:
    model_obj = _simple_llm_model(model_spec=LiteLlmSpec(provider="lm_studio", model="llama3"))
    src = render_agent_module(model_obj)
    # lm_studio -> provider rendered as openai.
    assert '"openai/llama3"' in src
    # Default LM Studio api_base.
    assert '"http://127.0.0.1:1234/v1"' in src
    # No api_key without api_key_env.
    assert "api_key=" not in src


def test_render_litellm_lm_studio_custom_api_base_overrides_default() -> None:
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(
            provider="lm_studio", model="llama3", api_base="http://127.0.0.1:9999/v1"
        )
    )
    src = render_agent_module(model_obj)
    assert '"http://127.0.0.1:9999/v1"' in src
    assert "1234" not in src


def test_render_litellm_anthropic_no_api_base() -> None:
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(provider="anthropic", model="claude-opus-4-5")
    )
    src = render_agent_module(model_obj)
    assert '"anthropic/claude-opus-4-5"' in src
    assert "api_base=" not in src


def test_render_litellm_with_api_key_env_import_os() -> None:
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(
            provider="anthropic", model="claude-opus-4-5", api_key_env="ANTHROPIC_API_KEY"
        )
    )
    src = render_agent_module(model_obj)
    assert 'api_key=os.getenv("ANTHROPIC_API_KEY")' in src
    assert "import os" in src


# --------------------------------------------------------------------------- #
# GenerateContentConfig rendering
# --------------------------------------------------------------------------- #
def test_render_generate_content_config_temperature_only() -> None:
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(temperature=0.7)
    )
    src = render_agent_module(model_obj)
    assert "from google.genai import types" in src
    assert "generate_content_config=types.GenerateContentConfig(" in src
    assert "temperature=0.7" in src
    # Other fields absent (not provided).
    assert "max_output_tokens=" not in src
    assert "safety_settings=" not in src


def test_render_generate_content_config_all_scalar_fields() -> None:
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(
            temperature=0.5,
            max_output_tokens=1024,
            top_p=0.9,
            top_k=40.0,
        )
    )
    src = render_agent_module(model_obj)
    assert "temperature=0.5" in src
    assert "max_output_tokens=1024" in src
    assert "top_p=0.9" in src
    assert "top_k=40.0" in src


def test_render_generate_content_config_safety_settings() -> None:
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(
            safety_settings=(
                SafetySettingSpec("HARM_CATEGORY_HARASSMENT", "BLOCK_MEDIUM_AND_ABOVE"),
                SafetySettingSpec("HARM_CATEGORY_HATE_SPEECH", "BLOCK_ONLY_HIGH"),
            )
        )
    )
    src = render_agent_module(model_obj)
    assert "safety_settings=" in src
    assert "types.SafetySetting(" in src
    assert "types.HarmCategory.HARM_CATEGORY_HARASSMENT" in src
    assert "types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE" in src
    assert "types.HarmCategory.HARM_CATEGORY_HATE_SPEECH" in src
    assert "types.HarmBlockThreshold.BLOCK_ONLY_HIGH" in src


def test_render_generate_content_config_response_modalities() -> None:
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(response_modalities=("TEXT", "IMAGE"))
    )
    src = render_agent_module(model_obj)
    assert 'response_modalities=["TEXT", "IMAGE"]' in src


def test_render_generate_content_config_with_litellm_together() -> None:
    """LiteLlm + GenerateContentConfig together: correct imports."""
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY"),
        generate_content_config=GenerateContentConfigSpec(temperature=0.3, max_output_tokens=512),
    )
    src = render_agent_module(model_obj)
    assert "from google.adk.models.lite_llm import LiteLlm" in src
    assert "from google.genai import types" in src
    assert "import os" in src
    assert "LiteLlm(" in src
    assert "types.GenerateContentConfig(" in src


# --------------------------------------------------------------------------- #
# Valid Python (ast.parse) — the generated modules must be syntactically correct
# --------------------------------------------------------------------------- #
def test_render_litellm_module_is_valid_python_ast() -> None:
    model_obj = _simple_llm_model(
        model_spec=LiteLlmSpec(provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY"),
        generate_content_config=GenerateContentConfigSpec(
            temperature=0.5,
            safety_settings=(SafetySettingSpec("HARM_CATEGORY_HARASSMENT", "BLOCK_NONE"),),
        ),
    )
    src = render_agent_module(model_obj)
    ast.parse(src)  # raises SyntaxError if invalid


def test_render_gcc_only_module_is_valid_python_ast() -> None:
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(
            temperature=0.7,
            max_output_tokens=2048,
            safety_settings=(
                SafetySettingSpec("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_LOW_AND_ABOVE"),
                SafetySettingSpec("HARM_CATEGORY_SEXUALLY_EXPLICIT", "OFF"),
            ),
            response_modalities=("TEXT",),
        )
    )
    src = render_agent_module(model_obj)
    ast.parse(src)


# --------------------------------------------------------------------------- #
# ruff format stability (format-stable generated code)
# --------------------------------------------------------------------------- #
def test_render_format_stable_litellm_with_gcc(tmp_path: Path) -> None:
    """LiteLlm + GenerateContentConfig + safety_settings module: stable for ruff format."""
    model_obj = ProjectModel(
        app_name="myapp",
        root="my_agent",
        agents=(
            AgentSpec(
                name="my_agent",
                type="llm",
                instruction="You are a helpful assistant.",
                model_spec=LiteLlmSpec(
                    provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY"
                ),
                generate_content_config=GenerateContentConfigSpec(
                    temperature=0.7,
                    max_output_tokens=2048,
                    top_p=0.9,
                    safety_settings=(
                        SafetySettingSpec("HARM_CATEGORY_HARASSMENT", "BLOCK_MEDIUM_AND_ABOVE"),
                        SafetySettingSpec("HARM_CATEGORY_DANGEROUS_CONTENT", "BLOCK_ONLY_HIGH"),
                    ),
                    response_modalities=("TEXT",),
                ),
            ),
        ),
    )
    src = render_agent_module(model_obj)
    _assert_ruff_format_stable(src, tmp_path, "litellm_with_gcc")


def test_render_format_stable_gcc_only(tmp_path: Path) -> None:
    """Module with only GenerateContentConfig (Gemini string): stable for ruff format."""
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(
            temperature=0.5,
            max_output_tokens=1024,
            safety_settings=(SafetySettingSpec("HARM_CATEGORY_HATE_SPEECH", "BLOCK_NONE"),),
        )
    )
    src = render_agent_module(model_obj)
    _assert_ruff_format_stable(src, tmp_path, "gcc_only")


def test_render_format_stable_lm_studio(tmp_path: Path) -> None:
    """LM Studio (provider normalized to openai + default api_base): stable for ruff format."""
    model_obj = _simple_llm_model(model_spec=LiteLlmSpec(provider="lm_studio", model="mistral"))
    src = render_agent_module(model_obj)
    _assert_ruff_format_stable(src, tmp_path, "lm_studio")


# --------------------------------------------------------------------------- #
# Functional probe: Gemini string + GenerateContentConfig importable in a subprocess
# (google-genai is core → installed, no extra needed)
# --------------------------------------------------------------------------- #
def test_functional_probe_gemini_string_with_gcc(tmp_path: Path) -> None:
    """Functional probe: generate agent.py + import it in a subprocess; check the live types."""
    model_obj = ProjectModel(
        app_name="probe_app",
        root="probe_agent",
        agents=(
            AgentSpec(
                name="probe_agent",
                type="llm",
                model="gemini-2.0-flash",
                instruction="Test probe.",
                generate_content_config=GenerateContentConfigSpec(
                    temperature=0.42,
                    max_output_tokens=512,
                ),
            ),
        ),
    )
    src = render_agent_module(model_obj)
    app_dir = tmp_path / "probe_app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    (app_dir / "agent.py").write_text(src, encoding="utf-8")

    probe_script = tmp_path / "probe.py"
    probe_script.write_text(
        """
import sys
sys.path.insert(0, sys.argv[1])
import probe_app.agent as m
from google.genai import types

agent = m.root_agent
assert agent.model == "gemini-2.0-flash", f"Expected gemini-2.0-flash, got {agent.model!r}"
gcc = agent.generate_content_config
assert gcc is not None, "generate_content_config should not be None"
assert isinstance(gcc, types.GenerateContentConfig), (
    f"Expected GenerateContentConfig, got {type(gcc)}"
)
assert gcc.temperature == 0.42, f"Expected temperature=0.42, got {gcc.temperature!r}"
assert gcc.max_output_tokens == 512, (
    f"Expected max_output_tokens=512, got {gcc.max_output_tokens!r}"
)
print("PROBE OK")
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(probe_script), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Probe subprocess failed.\nstdout: {result.stdout}\nstderr: {result.stderr}\n"
        f"Generated agent.py:\n{src}"
    )
    assert "PROBE OK" in result.stdout


# --------------------------------------------------------------------------- #
# LiteLlm probe (conditional — skipped if litellm is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(importlib.util.find_spec("litellm") is None, reason="litellm not installed")
def test_functional_probe_litellm(tmp_path: Path) -> None:
    """Functional LiteLlm probe: generate + import; check that model is a LiteLlm."""
    model_obj = ProjectModel(
        app_name="litellm_probe",
        root="llm_agent",
        agents=(
            AgentSpec(
                name="llm_agent",
                type="llm",
                model_spec=LiteLlmSpec(provider="openai", model="gpt-4o"),
            ),
        ),
    )
    src = render_agent_module(model_obj)
    app_dir = tmp_path / "litellm_probe"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    (app_dir / "agent.py").write_text(src, encoding="utf-8")

    probe_script = tmp_path / "litellm_probe_script.py"
    probe_script.write_text(
        """
import sys
sys.path.insert(0, sys.argv[1])
import litellm_probe.agent as m
from google.adk.models.lite_llm import LiteLlm

agent = m.root_agent
assert isinstance(agent.model, LiteLlm), f"Expected LiteLlm, got {type(agent.model)}"
print("LITELLM PROBE OK")
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe_script), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"LiteLlm probe failed.\nstdout: {result.stdout}\nstderr: {result.stderr}\n"
        f"Generated agent.py:\n{src}"
    )
    assert "LITELLM PROBE OK" in result.stdout


# --------------------------------------------------------------------------- #
# adk://models resource
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_resource_read_through() -> None:
    """Read ``adk://models`` via an in-memory Client and check the content."""
    mcp = FastMCP("t")
    register_resources(mcp)
    async with Client(mcp) as client:
        result = await client.read_resource("adk://models")
    payload = json.loads(result[0].text)
    assert "gemini_models" in payload
    assert "litellm_providers" in payload
    assert "harm_categories" in payload
    assert "harm_block_thresholds" in payload
    # The known providers are present.
    providers = payload["litellm_providers"]["supported"]
    assert "openai" in providers
    assert "lm_studio" in providers
    # The ADK categories are present.
    assert "HARM_CATEGORY_HARASSMENT" in payload["harm_categories"]
    assert "BLOCK_NONE" in payload["harm_block_thresholds"]


# --------------------------------------------------------------------------- #
# In-memory client read-through: models_configure_litellm
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_configure_litellm_client_readthrough(tmp_path: Path) -> None:
    """In-memory call to models_configure_litellm via Client; checks {ok: True}."""
    # Prepare a project with an llm agent.
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        root="my_agent",
        agents=(AgentSpec(name="my_agent", type="llm", model="gemini-2.5-flash"),),
    )
    save_model(ws, model_obj)

    # Call models_configure_litellm via an in-memory Client.
    async with Client(models_server) as client:
        result = await client.call_tool(
            "configure_litellm",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "my_agent",
                "provider": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
            },
        )
    # CallToolResult.data directly contains the dict.
    payload = result.data
    assert payload["ok"] is True, f"Expected ok=True, got: {payload}"
    assert payload["error"] is None

    # Check that the generated code contains LiteLlm and os.getenv.
    agent_py = app_dir / "agent.py"
    assert agent_py.exists()
    generated = agent_py.read_text(encoding="utf-8")
    assert "LiteLlm" in generated
    assert 'os.getenv("OPENAI_API_KEY")' in generated
    # Security invariant: no hardcoded key.
    assert "OPENAI_API_KEY" in generated  # the env var name is fine...
    import re

    assert not re.search(r'api_key\s*=\s*"[a-zA-Z0-9_+/=.-]+"', generated)


# --------------------------------------------------------------------------- #
# Domain tool tests — input validation
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_set_rejects_empty_model(tmp_path: Path) -> None:
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        agents=(AgentSpec(name="a", type="llm"),),
    )
    save_model(ws, model_obj)

    async with Client(models_server) as client:
        result = await client.call_tool(
            "set",
            {"path": str(tmp_path), "app_name": "myapp", "agent_name": "a", "model": "  "},
        )
    payload = result.data
    assert payload["ok"] is False
    assert "empty" in payload["error"].lower()


@pytest.mark.asyncio
async def test_models_configure_litellm_rejects_unknown_provider(tmp_path: Path) -> None:
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        agents=(AgentSpec(name="a", type="llm"),),
    )
    save_model(ws, model_obj)

    async with Client(models_server) as client:
        result = await client.call_tool(
            "configure_litellm",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "a",
                "provider": "unknown_provider_xyz",
                "model": "some-model",
            },
        )
    payload = result.data
    assert payload["ok"] is False
    assert "provider" in payload["error"].lower()


@pytest.mark.asyncio
async def test_models_generate_config_rejects_bad_safety_category(tmp_path: Path) -> None:
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        agents=(AgentSpec(name="a", type="llm"),),
    )
    save_model(ws, model_obj)

    async with Client(models_server) as client:
        result = await client.call_tool(
            "generate_config",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "a",
                "safety_settings": [{"category": "BAD_CATEGORY", "threshold": "BLOCK_NONE"}],
            },
        )
    payload = result.data
    assert payload["ok"] is False
    assert "harmcategory" in payload["error"].lower() or "harm" in payload["error"].lower()


@pytest.mark.asyncio
async def test_models_generate_config_rejects_bad_threshold(tmp_path: Path) -> None:
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        agents=(AgentSpec(name="a", type="llm"),),
    )
    save_model(ws, model_obj)

    async with Client(models_server) as client:
        result = await client.call_tool(
            "generate_config",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "a",
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BAD_THRESHOLD",
                    }
                ],
            },
        )
    payload = result.data
    assert payload["ok"] is False


@pytest.mark.asyncio
async def test_models_generate_config_rejects_non_llm_agent(tmp_path: Path) -> None:
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        agents=(
            AgentSpec(name="child", type="llm"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("child",)),
        ),
    )
    save_model(ws, model_obj)

    async with Client(models_server) as client:
        result = await client.call_tool(
            "generate_config",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "pipe",
                "temperature": 0.5,
            },
        )
    payload = result.data
    assert payload["ok"] is False
    assert "llm" in payload["error"].lower()


@pytest.mark.asyncio
async def test_models_set_idempotent(tmp_path: Path) -> None:
    """Calling set twice with the same model must be idempotent."""
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    model_obj = ProjectModel(
        app_name="myapp",
        agents=(AgentSpec(name="a", type="llm"),),
    )
    save_model(ws, model_obj)

    for _ in range(2):
        async with Client(models_server) as client:
            result = await client.call_tool(
                "set",
                {
                    "path": str(tmp_path),
                    "app_name": "myapp",
                    "agent_name": "a",
                    "model": "gemini-2.5-pro",
                },
            )
        payload = result.data
        assert payload["ok"] is True

    agent_py = app_dir / "agent.py"
    assert 'model="gemini-2.5-pro"' in agent_py.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Double-prefix guard test: the exposed tools must not contain a double prefix
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_exposed_tool_names_no_double_prefix() -> None:
    """The models_* tools are exposed without a double prefix (not models_models_*)."""
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
    models_tools = [n for n in names if n.startswith("models_")]
    assert models_tools, "No models_* tool found"
    for name in models_tools:
        assert not name.startswith("models_models_"), f"Double prefix detected: {name!r}"
    # Expected tools.
    assert "models_set" in models_tools
    assert "models_configure_litellm" in models_tools
    assert "models_generate_config" in models_tools


# --------------------------------------------------------------------------- #
# Additional error-path coverage (models domain)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_set_invalid_app_name(tmp_path: Path) -> None:
    """invalid app_name -> err."""
    async with Client(models_server) as client:
        result = await client.call_tool(
            "set",
            {
                "path": str(tmp_path),
                "app_name": "bad app",  # space -> invalid
                "agent_name": "a",
                "model": "gemini-2.5-flash",
            },
        )
    assert result.data["ok"] is False
    assert "app_name" in result.data["error"].lower()


@pytest.mark.asyncio
async def test_models_set_invalid_agent_name(tmp_path: Path) -> None:
    """invalid agent_name -> err."""
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    save_model(ws, ProjectModel(app_name="myapp", agents=(AgentSpec(name="a", type="llm"),)))

    async with Client(models_server) as client:
        result = await client.call_tool(
            "set",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "bad name",  # space -> invalid
                "model": "gemini-2.5-flash",
            },
        )
    assert result.data["ok"] is False


@pytest.mark.asyncio
async def test_models_generate_config_clears_when_all_none(tmp_path: Path) -> None:
    """generate_config with all None clears the existing config (idempotent)."""
    from adk_toolkit_mcp.project_model import (
        AgentSpec,
        GenerateContentConfigSpec,
        ProjectModel,
        save_model,
    )
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    # Agent with an existing config.
    save_model(
        ws,
        ProjectModel(
            app_name="myapp",
            agents=(
                AgentSpec(
                    name="a",
                    type="llm",
                    generate_content_config=GenerateContentConfigSpec(temperature=0.5),
                ),
            ),
        ),
    )

    async with Client(models_server) as client:
        result = await client.call_tool(
            "generate_config",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "a",
                # No parameters -> clears the config.
            },
        )
    assert result.data["ok"] is True
    # Check that the config was cleared.
    agent_py = (app_dir / "agent.py").read_text(encoding="utf-8")
    assert "generate_content_config=" not in agent_py


@pytest.mark.asyncio
async def test_models_set_missing_agent(tmp_path: Path) -> None:
    """Nonexistent agent -> err."""
    from adk_toolkit_mcp.project_model import ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    save_model(ws, ProjectModel(app_name="myapp"))

    async with Client(models_server) as client:
        result = await client.call_tool(
            "set",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "ghost",
                "model": "gemini-2.5-flash",
            },
        )
    assert result.data["ok"] is False
    assert "not found" in result.data["error"].lower()


@pytest.mark.asyncio
async def test_models_generate_config_success_all_fields(tmp_path: Path) -> None:
    """generate_config with all fields -> ok + correct code."""
    from adk_toolkit_mcp.project_model import AgentSpec, ProjectModel, save_model
    from adk_toolkit_mcp.workspace import Workspace

    app_dir = tmp_path / "myapp"
    app_dir.mkdir()
    ws = Workspace(app_dir)
    save_model(
        ws,
        ProjectModel(app_name="myapp", agents=(AgentSpec(name="a", type="llm"),)),
    )

    async with Client(models_server) as client:
        result = await client.call_tool(
            "generate_config",
            {
                "path": str(tmp_path),
                "app_name": "myapp",
                "agent_name": "a",
                "temperature": 0.8,
                "max_output_tokens": 512,
                "top_p": 0.95,
                "top_k": 32.0,
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                    }
                ],
                "response_modalities": ["TEXT"],
            },
        )
    assert result.data["ok"] is True
    agent_py = (app_dir / "agent.py").read_text(encoding="utf-8")
    assert "types.GenerateContentConfig(" in agent_py
    assert "temperature=0.8" in agent_py
    assert "HARM_CATEGORY_HARASSMENT" in agent_py
