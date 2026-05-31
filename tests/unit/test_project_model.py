"""Tests unitaires du renderer pur ``project_model`` (aucune construction ADK in-process).

On assert sur la **chaîne source** générée (sûr sous ``-W error::DeprecationWarning``,
puisqu'on ne construit aucun agent workflow déprécié ici). La preuve fonctionnelle
(instanciation réelle des objets ADK) est faite dans ``test_agents.py`` via un subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from adk_toolkit_mcp.project_model import (
    SIDECAR_PATH,
    AgentSpec,
    ProjectModel,
    add_or_update_agent,
    load_model,
    regenerate,
    render_agent_module,
    save_model,
    set_root,
    topological_order,
    validate_spec,
)
from adk_toolkit_mcp.workspace import Workspace


# --------------------------------------------------------------------------- #
# Dataclasses + (dé)sérialisation
# --------------------------------------------------------------------------- #
def test_agentspec_is_frozen() -> None:
    spec = AgentSpec(name="a", type="llm")
    with pytest.raises((AttributeError, TypeError)):
        spec.name = "b"  # type: ignore[misc]


def test_agentspec_roundtrip_llm() -> None:
    spec = AgentSpec(
        name="writer",
        type="llm",
        model="gemini-2.5-flash",
        instruction="Write.",
        description="A writer.",
        output_key="draft",
        tools=("google_search",),
    )
    restored = AgentSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_projectmodel_roundtrip() -> None:
    model = ProjectModel(
        app_name="demo",
        root="pipe",
        agents=(
            AgentSpec(name="a", type="llm"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("a",)),
        ),
    )
    restored = ProjectModel.from_dict(model.to_dict())
    assert restored == model


# --------------------------------------------------------------------------- #
# Mutations immuables
# --------------------------------------------------------------------------- #
def test_add_or_update_agent_is_immutable() -> None:
    model = ProjectModel(app_name="demo")
    new = add_or_update_agent(model, AgentSpec(name="a", type="llm"))
    assert model.agents == ()  # original intact
    assert new.agent_names() == ("a",)
    assert new is not model


def test_add_or_update_agent_replaces_in_place() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(AgentSpec(name="a", type="llm"), AgentSpec(name="b", type="llm")),
    )
    updated = add_or_update_agent(model, AgentSpec(name="a", type="llm", instruction="new"))
    # Position préservée, pas de doublon.
    assert updated.agent_names() == ("a", "b")
    a = updated.get("a")
    assert a is not None and a.instruction == "new"


def test_set_root_immutable() -> None:
    model = ProjectModel(app_name="demo", agents=(AgentSpec(name="a", type="llm"),))
    new = set_root(model, "a")
    assert model.root is None
    assert new.root == "a"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_rejects_bad_identifier() -> None:
    assert validate_spec(AgentSpec(name="bad name", type="llm")) is not None
    assert validate_spec(AgentSpec(name="1abc", type="llm")) is not None


def test_validate_rejects_unknown_type() -> None:
    assert validate_spec(AgentSpec(name="a", type="nope")) is not None  # type: ignore[arg-type]


def test_validate_rejects_nonpositive_max_iterations() -> None:
    assert validate_spec(AgentSpec(name="a", type="loop", max_iterations=0)) is not None
    assert validate_spec(AgentSpec(name="a", type="loop", max_iterations=-1)) is not None


def test_validate_accepts_good_llm() -> None:
    assert validate_spec(AgentSpec(name="good_agent", type="llm")) is None


# --------------------------------------------------------------------------- #
# Tri topologique + cycles
# --------------------------------------------------------------------------- #
def test_topological_order_child_before_parent() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="pipe", type="sequential", sub_agents=("a", "b")),
            AgentSpec(name="a", type="llm"),
            AgentSpec(name="b", type="llm"),
        ),
    )
    ordered = [s.name for s in topological_order(model)]
    assert ordered.index("a") < ordered.index("pipe")
    assert ordered.index("b") < ordered.index("pipe")


def test_topological_order_detects_direct_cycle() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="x", type="sequential", sub_agents=("y",)),
            AgentSpec(name="y", type="sequential", sub_agents=("x",)),
        ),
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        topological_order(model)


def test_topological_order_detects_self_cycle() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(AgentSpec(name="x", type="sequential", sub_agents=("x",)),),
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        topological_order(model)


# --------------------------------------------------------------------------- #
# Rendu source — par type
# --------------------------------------------------------------------------- #
def test_render_llm_minimal_omits_empty_kwargs() -> None:
    model = ProjectModel(
        app_name="demo",
        root="solo",
        agents=(AgentSpec(name="solo", type="llm", instruction="Hi"),),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents import LlmAgent" in src
    assert "solo = LlmAgent(" in src
    assert 'name="solo"' in src
    assert 'model="gemini-2.5-flash"' in src
    assert 'instruction="Hi"' in src
    # description vide / output_key None / tools vide / sub_agents vide -> omis.
    assert "description=" not in src
    assert "output_key=" not in src
    assert "tools=" not in src
    assert "sub_agents=" not in src
    assert src.rstrip().endswith("root_agent = solo")


def test_render_llm_includes_output_key_and_tools() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(
                name="searcher",
                type="llm",
                description="Searches.",
                output_key="results",
                tools=("google_search", "my_tool"),
            ),
        ),
    )
    src = render_agent_module(model)
    assert 'output_key="results"' in src
    assert "tools=[google_search, my_tool]" in src
    assert 'description="Searches."' in src


def test_render_sequential_and_parallel() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="a", type="llm"),
            AgentSpec(name="b", type="llm"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("a", "b")),
            AgentSpec(name="fan", type="parallel", sub_agents=("a", "b")),
        ),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent" in src
    assert "pipe = SequentialAgent(" in src
    assert "sub_agents=[a, b]" in src
    assert "fan = ParallelAgent(" in src


def test_render_loop_includes_max_iterations() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(
            AgentSpec(name="step", type="llm"),
            AgentSpec(name="lp", type="loop", sub_agents=("step",), max_iterations=5),
        ),
    )
    src = render_agent_module(model)
    assert "lp = LoopAgent(" in src
    assert "max_iterations=5" in src
    assert "sub_agents=[step]" in src


def test_render_custom_emits_baseagent_subclass_and_instance() -> None:
    model = ProjectModel(
        app_name="demo",
        agents=(AgentSpec(name="my_custom", type="custom", description="D"),),
    )
    src = render_agent_module(model)
    assert "from google.adk.agents import BaseAgent" in src
    assert "class MyCustomAgent(BaseAgent):" in src
    assert "async def _run_async_impl(self, ctx):" in src
    # async generator no-op : return suivi d'un yield inatteignable.
    assert "yield" in src
    assert 'my_custom = MyCustomAgent(name="my_custom", description="D")' in src


def test_render_empty_model_has_no_root() -> None:
    src = render_agent_module(ProjectModel(app_name="demo"))
    assert "root_agent =" not in src.replace("# root_agent", "")
    assert "Aucun agent" in src


def test_render_root_missing_emits_comment_not_assignment() -> None:
    model = ProjectModel(
        app_name="demo",
        root="ghost",  # n'existe pas
        agents=(AgentSpec(name="real", type="llm"),),
    )
    src = render_agent_module(model)
    assert "root_agent = ghost" not in src
    assert "introuvable" in src


def test_render_imports_only_used_classes() -> None:
    # Seulement des llm -> n'importe pas Sequential/Parallel/Loop/BaseAgent.
    model = ProjectModel(app_name="demo", agents=(AgentSpec(name="a", type="llm"),))
    src = render_agent_module(model)
    line = next(line for line in src.splitlines() if line.startswith("from google.adk.agents"))
    assert "LlmAgent" in line
    assert "SequentialAgent" not in line
    assert "BaseAgent" not in line


# --------------------------------------------------------------------------- #
# Sidecar I/O + regenerate (sur disque)
# --------------------------------------------------------------------------- #
def test_load_model_absent_returns_empty(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = load_model(ws, "app")
    assert model.app_name == "app"
    assert model.agents == ()
    assert model.root is None


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        root="a",
        agents=(AgentSpec(name="a", type="llm", instruction="Hi"),),
    )
    assert save_model(ws, model) is True
    assert ws.exists(SIDECAR_PATH)
    reloaded = load_model(ws, "app")
    assert reloaded == model


def test_save_model_idempotent(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(app_name="app", agents=(AgentSpec(name="a", type="llm"),))
    assert save_model(ws, model) is True
    assert save_model(ws, model) is False  # contenu identique -> rien changé


def test_load_model_corrupt_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    ws.write(SIDECAR_PATH, "{ not valid json ]")
    with pytest.raises(ValueError, match="JSON invalide"):
        load_model(ws, "app")


def test_regenerate_writes_agent_and_init(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        root="solo",
        agents=(AgentSpec(name="solo", type="llm", instruction="Hi"),),
    )
    result = regenerate(ws, model)
    assert result["changed"] is True
    assert ws.exists("agent.py")
    assert ws.exists("__init__.py")
    assert ws.read("__init__.py") == "from . import agent\n"
    assert "root_agent = solo" in ws.read("agent.py")


def test_regenerate_idempotent(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(app_name="app", agents=(AgentSpec(name="a", type="llm"),))
    first = regenerate(ws, model)
    assert first["changed"] is True
    second = regenerate(ws, model)
    assert second["changed"] is False


def test_regenerate_cycle_raises(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "app")
    model = ProjectModel(
        app_name="app",
        agents=(
            AgentSpec(name="x", type="sequential", sub_agents=("y",)),
            AgentSpec(name="y", type="sequential", sub_agents=("x",)),
        ),
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        regenerate(ws, model)


# --------------------------------------------------------------------------- #
# Stabilité de format ruff — le fichier généré doit être déjà formaté
# --------------------------------------------------------------------------- #
def _ruff_exe() -> str | None:
    """Localise l'exécutable ruff dans l'environnement courant (venv ou PATH)."""
    # Prefer the ruff that lives next to the current Python interpreter (venv).
    import sys

    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _assert_ruff_format_stable(src: str, tmp_path: Path, label: str) -> None:
    """Écrit *src* dans un fichier temporaire et vérifie que ``ruff format --check`` passe."""
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


def test_render_format_stable_custom_llm_workflow(tmp_path: Path) -> None:
    """Le module généré avec un custom + llm + workflow est stable pour ruff format."""
    model = ProjectModel(
        app_name="demo",
        root="pipe",
        agents=(
            AgentSpec(name="my_custom", type="custom", description="Custom agent"),
            AgentSpec(name="llm_one", type="llm", instruction="Think"),
            AgentSpec(name="pipe", type="sequential", sub_agents=("my_custom", "llm_one")),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "custom_llm_workflow")


def test_render_format_stable_llm_only(tmp_path: Path) -> None:
    """Le module généré avec des agents llm uniquement est stable pour ruff format."""
    model = ProjectModel(
        app_name="demo",
        root="solo",
        agents=(
            AgentSpec(
                name="solo",
                type="llm",
                instruction="Hi",
                description="A solo agent",
                output_key="result",
                tools=("google_search",),
            ),
        ),
    )
    src = render_agent_module(model)
    _assert_ruff_format_stable(src, tmp_path, "llm_only")
