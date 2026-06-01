"""Tests unitaires du domaine ``models`` (P1 domaine 4/4).

Couvre :
- ``set`` (bare ``models_set``) : définit un modèle Gemini string sur un agent.
- ``configure_litellm`` (bare ``models_configure_litellm``) : configure LiteLlm.
- ``generate_config`` (bare ``models_generate_config``) : configure GenerateContentConfig.
- Rendu de source : imports LiteLlm / os / types, api_key jamais hardcodé.
- Stabilité ruff format.
- Probe fonctionnel : agent Gemini + generate_content_config importable en subprocess.
- Probe LiteLlm (skipé si litellm absent en CI).
- Ressource ``adk://models`` (lecture via Client in-memory).
- Read-through client : ``models_configure_litellm`` renvoie ``{ok: True}``.
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
# Helpers partagés
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    """Localise l'exécutable ruff dans l'environnement courant."""
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
        pytest.skip("ruff introuvable dans l'environnement — test de format ignoré")
    result = subprocess.run(
        [ruff, "format", "--check", str(gen_file)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"ruff format --check a échoué pour le cas '{label}'.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}\n"
        f"Source générée :\n{src}"
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
# Constantes + dataclasses
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
# Rendu Gemini string (compat ascendante)
# --------------------------------------------------------------------------- #
def test_render_gemini_string_model_unchanged() -> None:
    model_obj = _simple_llm_model(model="gemini-2.5-pro")
    src = render_agent_module(model_obj)
    assert 'model="gemini-2.5-pro"' in src
    # Pas d'import LiteLlm ni types si seulement un string Gemini.
    assert "LiteLlm" not in src
    assert "from google.genai import types" not in src


# --------------------------------------------------------------------------- #
# Rendu LiteLlm
# --------------------------------------------------------------------------- #
def test_render_litellm_basic_openai() -> None:
    model_obj = _simple_llm_model(model_spec=LiteLlmSpec(provider="openai", model="gpt-4o"))
    src = render_agent_module(model_obj)
    assert "from google.adk.models.lite_llm import LiteLlm" in src
    assert 'model=LiteLlm(model="openai/gpt-4o")' in src
    # Pas d'api_key hardcodée.
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
    # La clé est lue via os.getenv, jamais hardcodée.
    assert 'api_key=os.getenv("MY_API_KEY")' in src
    assert "import os" in src
    # Aucune valeur de clé littérale.
    assert "api_key=" in src  # présent...
    # ...mais uniquement sous forme os.getenv (pas de chaîne littérale de clé).
    import re

    assert not re.search(r'api_key\s*=\s*"[a-zA-Z0-9_-]+"', src)


def test_render_litellm_no_hardcoded_api_key_ever() -> None:
    """Invariant de sécurité : aucune clé ne doit jamais apparaître en dur."""
    for provider in LITELLM_PROVIDERS:
        spec = LiteLlmSpec(provider=provider, model="model-x")
        model_obj = _simple_llm_model(model_spec=spec)
        src = render_agent_module(model_obj)
        # Sans api_key_env : pas de api_key= du tout.
        assert "api_key=" not in src, (
            f"api_key trouvé dans le code généré pour provider={provider!r} sans api_key_env"
        )


def test_render_litellm_lm_studio_defaults_provider_and_api_base() -> None:
    model_obj = _simple_llm_model(model_spec=LiteLlmSpec(provider="lm_studio", model="llama3"))
    src = render_agent_module(model_obj)
    # lm_studio -> provider rendu comme openai.
    assert '"openai/llama3"' in src
    # api_base par défaut LM Studio.
    assert '"http://127.0.0.1:1234/v1"' in src
    # Pas d'api_key sans api_key_env.
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
# Rendu GenerateContentConfig
# --------------------------------------------------------------------------- #
def test_render_generate_content_config_temperature_only() -> None:
    model_obj = _simple_llm_model(
        generate_content_config=GenerateContentConfigSpec(temperature=0.7)
    )
    src = render_agent_module(model_obj)
    assert "from google.genai import types" in src
    assert "generate_content_config=types.GenerateContentConfig(" in src
    assert "temperature=0.7" in src
    # Autres champs absents (pas fournis).
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
    """LiteLlm + GenerateContentConfig ensemble : imports corrects."""
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
# Python valide (ast.parse) — les modules générés doivent être syntaxiquement corrects
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
    ast.parse(src)  # lève SyntaxError si invalide


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
# Stabilité ruff format (format-stable generated code)
# --------------------------------------------------------------------------- #
def test_render_format_stable_litellm_with_gcc(tmp_path: Path) -> None:
    """Module LiteLlm + GenerateContentConfig + safety_settings : stable pour ruff format."""
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
    """Module avec seulement GenerateContentConfig (Gemini string) : stable pour ruff format."""
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
    """LM Studio (provider normalisé openai + api_base par défaut) : stable pour ruff format."""
    model_obj = _simple_llm_model(model_spec=LiteLlmSpec(provider="lm_studio", model="mistral"))
    src = render_agent_module(model_obj)
    _assert_ruff_format_stable(src, tmp_path, "lm_studio")


# --------------------------------------------------------------------------- #
# Probe fonctionnel : Gemini string + GenerateContentConfig importable en subprocess
# (google-genai est core → installé, pas besoin d'extra)
# --------------------------------------------------------------------------- #
def test_functional_probe_gemini_string_with_gcc(tmp_path: Path) -> None:
    """Probe fonctionnel : génère agent.py + l'importe en subprocess ; vérifie les types live."""
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
# Probe LiteLlm (conditionnel — skipé si litellm absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(importlib.util.find_spec("litellm") is None, reason="litellm not installed")
def test_functional_probe_litellm(tmp_path: Path) -> None:
    """Probe fonctionnel LiteLlm : génère + importe ; vérifie que model est un LiteLlm."""
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
# Ressource adk://models
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_resource_read_through() -> None:
    """Lit ``adk://models`` via Client in-memory et vérifie le contenu."""
    mcp = FastMCP("t")
    register_resources(mcp)
    async with Client(mcp) as client:
        result = await client.read_resource("adk://models")
    payload = json.loads(result[0].text)
    assert "gemini_models" in payload
    assert "litellm_providers" in payload
    assert "harm_categories" in payload
    assert "harm_block_thresholds" in payload
    # Les providers connus sont présents.
    providers = payload["litellm_providers"]["supported"]
    assert "openai" in providers
    assert "lm_studio" in providers
    # Les catégories ADK sont présentes.
    assert "HARM_CATEGORY_HARASSMENT" in payload["harm_categories"]
    assert "BLOCK_NONE" in payload["harm_block_thresholds"]


# --------------------------------------------------------------------------- #
# In-memory client read-through : models_configure_litellm
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_configure_litellm_client_readthrough(tmp_path: Path) -> None:
    """Appel in-memory de models_configure_litellm via Client ; vérifie {ok: True}."""
    # Prépare un projet avec un agent llm.
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

    # Appelle models_configure_litellm via Client in-memory.
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
    # CallToolResult.data contient directement le dict.
    payload = result.data
    assert payload["ok"] is True, f"Expected ok=True, got: {payload}"
    assert payload["error"] is None

    # Vérifie que le code généré contient LiteLlm et os.getenv.
    agent_py = app_dir / "agent.py"
    assert agent_py.exists()
    generated = agent_py.read_text(encoding="utf-8")
    assert "LiteLlm" in generated
    assert 'os.getenv("OPENAI_API_KEY")' in generated
    # Invariant sécurité : aucune clé hardcodée.
    assert "OPENAI_API_KEY" in generated  # le nom de la var env est ok...
    import re

    assert not re.search(r'api_key\s*=\s*"[a-zA-Z0-9_+/=.-]+"', generated)


# --------------------------------------------------------------------------- #
# Tests des outils domaine — validation d'entrées
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
    assert "vide" in payload["error"].lower()


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
    assert "provider" in payload["error"].lower() or "inconnu" in payload["error"].lower()


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
    """Appeler set deux fois avec le même modèle doit être idempotent."""
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
# Test double-prefix guard : les outils exposés ne doivent pas contenir de double prefix
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_exposed_tool_names_no_double_prefix() -> None:
    """Les outils models_* sont exposés sans double-prefix (pas models_models_*)."""
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
    models_tools = [n for n in names if n.startswith("models_")]
    assert models_tools, "Aucun outil models_* trouvé"
    for name in models_tools:
        assert not name.startswith("models_models_"), f"Double-prefix détecté : {name!r}"
    # Outils attendus.
    assert "models_set" in models_tools
    assert "models_configure_litellm" in models_tools
    assert "models_generate_config" in models_tools


# --------------------------------------------------------------------------- #
# Couverture supplémentaire des chemins d'erreur (domaine models)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_models_set_invalid_app_name(tmp_path: Path) -> None:
    """app_name invalide -> err."""
    async with Client(models_server) as client:
        result = await client.call_tool(
            "set",
            {
                "path": str(tmp_path),
                "app_name": "bad app",  # espace -> invalide
                "agent_name": "a",
                "model": "gemini-2.5-flash",
            },
        )
    assert result.data["ok"] is False
    assert "app_name" in result.data["error"].lower()


@pytest.mark.asyncio
async def test_models_set_invalid_agent_name(tmp_path: Path) -> None:
    """agent_name invalide -> err."""
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
                "agent_name": "bad name",  # espace -> invalide
                "model": "gemini-2.5-flash",
            },
        )
    assert result.data["ok"] is False


@pytest.mark.asyncio
async def test_models_generate_config_clears_when_all_none(tmp_path: Path) -> None:
    """generate_config avec tout None efface la config existante (idempotent)."""
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
    # Agent avec une config existante.
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
                # Pas de paramètres -> efface la config.
            },
        )
    assert result.data["ok"] is True
    # Vérifier que la config a été effacée.
    agent_py = (app_dir / "agent.py").read_text(encoding="utf-8")
    assert "generate_content_config=" not in agent_py


@pytest.mark.asyncio
async def test_models_set_missing_agent(tmp_path: Path) -> None:
    """Agent inexistant -> err."""
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
    assert "introuvable" in result.data["error"].lower()


@pytest.mark.asyncio
async def test_models_generate_config_success_all_fields(tmp_path: Path) -> None:
    """generate_config avec tous les champs -> ok + code correct."""
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
