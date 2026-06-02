"""Tests for the ``skills`` domain (Agent Skill Registry, ``google.adk.skills``).

Three layers, mirroring ``test_workflow.py``:

1. **Schema conformance via the REAL ADK loaders.** ``create`` writes the on-disk skill layout;
   ``list``/``load``/``registry_info`` round-trip it through the real
   ``google.adk.skills.list_skills_in_dir`` / ``load_skill_from_dir`` — proving the written
   ``SKILL.md`` conforms to the spec (not a guess). We assert real ``Skill`` object fields (name,
   description, instructions, frontmatter, resources).
2. **Codegen.** ``attach`` renders a ``SkillToolset`` into ``agent.py``; the generated source passes
   ``ast.parse`` + ``ruff format --check`` + ``ruff check --select I``.
3. **Functional proof (subprocess).** The generated ``agent.py`` is imported in a subprocess (the
   uv venv) and its ``SkillToolset`` instantiates, **loads the real skills from disk**, and exposes
   the real ADK skill tools (``list_skills``/``load_skill``/``load_skill_resource``/
   ``run_skill_script``) — offline, no API key. The ``SkillToolset`` is ``@experimental``, so it
   emits a ``UserWarning`` at import (NOT a ``DeprecationWarning``): the subprocess filters it and
   the suite's ``-W error::DeprecationWarning`` gate is unaffected.

Plus validation/error paths (all return the ``{ok, data, error}`` envelope, never an exception) and
an in-memory ``fastmcp.Client`` read-through of one full skills flow.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains.agents import create_llm
from adk_toolkit_mcp.domains.skills import (
    attach,
    create,
    list_skills,
    load,
    registry_info,
)
from adk_toolkit_mcp.project_model import SIDECAR_PATH
from adk_toolkit_mcp.server import build_server


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _agent_py(tmp_path: Path, app: str) -> str:
    return (tmp_path / app / "agent.py").read_text(encoding="utf-8")


def _skill_md(tmp_path: Path, app: str, name: str) -> Path:
    return tmp_path / app / "skills" / name / "SKILL.md"


def _ruff_exe() -> str | None:
    """Locate the ruff executable in the current environment (venv or PATH)."""
    venv_bin = Path(sys.executable).parent
    for candidate in (venv_bin / "ruff", venv_bin / "ruff.exe"):
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def _assert_codegen_clean(src: str, tmp_path: Path, label: str) -> None:
    """Assert the generated source is ``ast.parse`` + ``ruff format --check`` + isort clean."""
    ast.parse(src)  # raises SyntaxError if the generated code is invalid

    gen_file = tmp_path / f"{label}.py"
    gen_file.write_text(src, encoding="utf-8")
    ruff = _ruff_exe()
    if ruff is None:
        pytest.skip("ruff not found in the environment — format/isort checks ignored")

    fmt = subprocess.run([ruff, "format", "--check", str(gen_file)], capture_output=True, text=True)
    assert fmt.returncode == 0, f"ruff format --check failed ({label}):\n{fmt.stdout}\n{src}"

    isort = subprocess.run(
        [ruff, "check", "--select", "I", str(gen_file)], capture_output=True, text=True
    )
    assert isort.returncode == 0, f"ruff check --select I failed ({label}):\n{isort.stdout}\n{src}"


def _make_agent_with_skills(tmp_path: Path, app: str, *skill_names: str) -> None:
    """Scaffold an ``LlmAgent`` + one skill per ``skill_names``, then attach them all."""
    create_llm(str(tmp_path), app, "assistant", instruction="Help the user.")
    for name in skill_names:
        res = create(
            str(tmp_path),
            app,
            name,
            description=f"The {name} skill: does {name} things.",
            instruction=f"# {name}\nFollow the {name} steps.",
        )
        assert res["ok"] is True, res["error"]
    attached = attach(str(tmp_path), app, "assistant", list(skill_names))
    assert attached["ok"] is True, attached["error"]


# --------------------------------------------------------------------------- #
# Functional probe: import the generated agent.py in a subprocess and confirm
# the SkillToolset loads the real skills from disk and exposes the real tools.
# --------------------------------------------------------------------------- #
def _probe_toolset(project_path: str, app_name: str, toolset_var: str) -> dict[str, object]:
    """Import ``<app>.agent`` in a subprocess; return the SkillToolset's type/tools/skills.

    The subprocess filters the ``@experimental`` ``UserWarning`` emitted by ``SkillToolset`` (it is
    not a ``DeprecationWarning``, so the suite gate is unaffected) and is launched with
    ``-W ignore::DeprecationWarning`` (consistent with the rest of the suite).
    """
    code = (
        "import asyncio, json, sys, warnings\n"
        "warnings.simplefilter('ignore', UserWarning)\n"
        f"sys.path.insert(0, {project_path!r})\n"
        f"import {app_name}.agent as m\n"
        f"ts = m.{toolset_var}\n"
        "tools = asyncio.run(ts.get_tools())\n"
        "info = {\n"
        "    'toolset_type': type(ts).__name__,\n"
        "    'tool_names': sorted(t.name for t in tools),\n"
        "    'skills': sorted(ts._skills.keys()),\n"
        "}\n"
        "print(json.dumps(info))\n"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
        cwd=project_path,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# create — writes the on-disk skill layout ADK expects
# --------------------------------------------------------------------------- #
def test_create_writes_skill_md_with_frontmatter(tmp_path: Path) -> None:
    res = create(
        str(tmp_path),
        "demo",
        "greeter",
        description="Greets the user warmly.",
        instruction="# Greeter\nGreet warmly in the user's language.",
    )
    assert res["ok"] is True, res["error"]
    md = _skill_md(tmp_path, "demo", "greeter")
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: greeter" in text
    assert 'description: "Greets the user warmly."' in text
    assert "# Greeter" in text
    assert res["data"]["skill"] == "greeter"


def test_create_is_idempotent_overwrite(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "greeter", description="v1.", instruction="A")
    res = create(str(tmp_path), "demo", "greeter", description="v2.", instruction="B")
    assert res["ok"] is True
    text = _skill_md(tmp_path, "demo", "greeter").read_text(encoding="utf-8")
    assert 'description: "v2."' in text
    assert text.rstrip().endswith("B")


def test_create_rejects_non_kebab_name(tmp_path: Path) -> None:
    res = create(str(tmp_path), "demo", "Bad_Name", description="d", instruction="i")
    assert res["ok"] is False
    assert "kebab-case" in res["error"]


def test_create_rejects_bad_app_name(tmp_path: Path) -> None:
    assert create(str(tmp_path), "1bad", "greeter", description="d", instruction="i")["ok"] is False


def test_create_rejects_empty_description(tmp_path: Path) -> None:
    res = create(str(tmp_path), "demo", "greeter", description="   ", instruction="i")
    assert res["ok"] is False
    assert "description is required" in res["error"]


def test_create_rejects_overlong_description(tmp_path: Path) -> None:
    res = create(str(tmp_path), "demo", "greeter", description="x" * 1025, instruction="i")
    assert res["ok"] is False
    assert "1024" in res["error"]


# --------------------------------------------------------------------------- #
# list / load / registry_info — round-trip through the REAL ADK loaders
# --------------------------------------------------------------------------- #
def test_list_round_trips_through_real_loader(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    create(str(tmp_path), "demo", "summarizer", description="Summarizes.", instruction="# S")
    res = list_skills(str(tmp_path), "demo")
    assert res["ok"] is True
    ids = [s["id"] for s in res["data"]["skills"]]
    assert ids == ["greeter", "summarizer"]  # sorted
    by_id = {s["id"]: s for s in res["data"]["skills"]}
    assert by_id["greeter"]["name"] == "greeter"
    assert by_id["greeter"]["description"] == "Greets."


def test_list_missing_dir_is_empty_not_error(tmp_path: Path) -> None:
    res = list_skills(str(tmp_path), "demo")
    assert res["ok"] is True
    assert res["data"]["skills"] == []


def test_load_returns_real_skill_fields(tmp_path: Path) -> None:
    create(
        str(tmp_path),
        "demo",
        "greeter",
        description="Greets the user warmly.",
        instruction="# Greeter\nStep 1.\nStep 2.",
    )
    res = load(str(tmp_path), "demo", "greeter")
    assert res["ok"] is True, res["error"]
    data = res["data"]
    # These fields come from a REAL google.adk.skills.Skill object.
    assert data["name"] == "greeter"
    assert data["description"] == "Greets the user warmly."
    assert data["instructions"].startswith("# Greeter")
    assert "Step 1." in data["instructions"]
    assert data["frontmatter"]["name"] == "greeter"
    assert data["resources"] == {"references": [], "assets": [], "scripts": []}


def test_load_with_references_resource(tmp_path: Path) -> None:
    """A reference file dropped into the skill dir is surfaced by the REAL loader."""
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    refs = tmp_path / "demo" / "skills" / "greeter" / "references"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "langs.md").write_text("en: Hello\nfr: Bonjour\n", encoding="utf-8")
    res = load(str(tmp_path), "demo", "greeter")
    assert res["ok"] is True
    assert res["data"]["resources"]["references"] == ["langs.md"]


def test_load_missing_skill_returns_err(tmp_path: Path) -> None:
    res = load(str(tmp_path), "demo", "ghost")
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_load_invalid_skill_md_returns_err(tmp_path: Path) -> None:
    """A SKILL.md without frontmatter is rejected by the REAL loader (surfaced as err)."""
    skill_dir = tmp_path / "demo" / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("no frontmatter here", encoding="utf-8")
    res = load(str(tmp_path), "demo", "broken")
    assert res["ok"] is False
    assert "frontmatter" in res["error"].lower()


def test_load_name_dir_mismatch_returns_err(tmp_path: Path) -> None:
    """ADK enforces frontmatter name == directory name; a mismatch surfaces as err."""
    skill_dir = tmp_path / "demo" / "skills" / "greeter"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        '---\nname: other\ndescription: "Mismatched name."\n---\n# X\n', encoding="utf-8"
    )
    res = load(str(tmp_path), "demo", "greeter")
    assert res["ok"] is False
    assert "does not match directory" in res["error"]


def test_registry_info_reports_directory_inventory(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    create(str(tmp_path), "demo", "summarizer", description="Summarizes.", instruction="# S")
    res = registry_info(str(tmp_path), "demo")
    assert res["ok"] is True
    assert res["data"]["registry_kind"] == "directory"
    assert res["data"]["count"] == 2
    assert [r["id"] for r in res["data"]["registered"]] == ["greeter", "summarizer"]


def test_registry_info_empty_when_no_skills(tmp_path: Path) -> None:
    res = registry_info(str(tmp_path), "demo")
    assert res["ok"] is True
    assert res["data"]["count"] == 0


# --------------------------------------------------------------------------- #
# attach — renders a SkillToolset onto an LlmAgent
# --------------------------------------------------------------------------- #
def test_attach_renders_skill_toolset(tmp_path: Path) -> None:
    _make_agent_with_skills(tmp_path, "demo", "greeter", "summarizer")
    txt = _agent_py(tmp_path, "demo")
    assert "from google.adk.tools.skill_toolset import SkillToolset" in txt
    assert "from google.adk.skills import load_skill_from_dir" in txt
    assert '_ADK_SKILLS_DIR = Path(__file__).parent / "skills"' in txt
    assert 'load_skill_from_dir(_ADK_SKILLS_DIR / "greeter")' in txt
    assert 'load_skill_from_dir(_ADK_SKILLS_DIR / "summarizer")' in txt
    assert "assistant_skills = SkillToolset(" in txt
    # The toolset goes directly into the agent's tools list.
    assert "tools=[assistant_skills]" in txt


def test_attach_persists_in_sidecar(tmp_path: Path) -> None:
    _make_agent_with_skills(tmp_path, "demo", "greeter")
    sidecar = json.loads((tmp_path / "demo" / SIDECAR_PATH).read_text(encoding="utf-8"))
    tool = sidecar["agents"][0]["tools"][0]
    assert tool["kind"] == "skill_toolset"
    assert tool["name"] == "assistant_skills"
    assert tool["skill_names"] == ["greeter"]


def test_attach_custom_toolset_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "assistant", instruction="Help.")
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    res = attach(str(tmp_path), "demo", "assistant", ["greeter"], name="my_skills")
    assert res["ok"] is True, res["error"]
    assert "my_skills = SkillToolset(" in _agent_py(tmp_path, "demo")


def test_attach_is_idempotent_replace_by_name(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "assistant", instruction="Help.")
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    create(str(tmp_path), "demo", "summarizer", description="Summarizes.", instruction="# S")
    attach(str(tmp_path), "demo", "assistant", ["greeter"])
    res = attach(str(tmp_path), "demo", "assistant", ["greeter", "summarizer"])
    assert res["ok"] is True
    # Replace by name (same toolset var) → exactly one skill_toolset tool.
    assert res["data"]["tools"] == ["skill_toolset:assistant_skills"]
    txt = _agent_py(tmp_path, "demo")
    assert txt.count("assistant_skills = SkillToolset(") == 1
    assert 'load_skill_from_dir(_ADK_SKILLS_DIR / "summarizer")' in txt


def test_attach_rejects_missing_agent(tmp_path: Path) -> None:
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    res = attach(str(tmp_path), "demo", "ghost", ["greeter"])
    assert res["ok"] is False
    assert "Agent not found" in res["error"]


def test_attach_rejects_non_llm_agent(tmp_path: Path) -> None:
    """Only LlmAgent carries tools; attaching to a SequentialAgent is rejected."""
    from adk_toolkit_mcp.domains.agents import create_sequential

    create_sequential(str(tmp_path), "demo", "pipeline", sub_agents=[])
    create(str(tmp_path), "demo", "greeter", description="Greets.", instruction="# G")
    res = attach(str(tmp_path), "demo", "pipeline", ["greeter"])
    assert res["ok"] is False
    assert "only 'llm' agents" in res["error"]


def test_attach_rejects_missing_skill_on_disk(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "assistant", instruction="Help.")
    res = attach(str(tmp_path), "demo", "assistant", ["ghost"])
    assert res["ok"] is False
    assert "not found" in res["error"]


def test_attach_rejects_empty_skill_list(tmp_path: Path) -> None:
    create_llm(str(tmp_path), "demo", "assistant", instruction="Help.")
    res = attach(str(tmp_path), "demo", "assistant", [])
    assert res["ok"] is False
    assert "at least one skill" in res["error"]


def test_attach_rejects_unloadable_skill(tmp_path: Path) -> None:
    """A skill folder that exists but is malformed is rejected by the REAL loader at attach time."""
    create_llm(str(tmp_path), "demo", "assistant", instruction="Help.")
    skill_dir = tmp_path / "demo" / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("not valid", encoding="utf-8")
    res = attach(str(tmp_path), "demo", "assistant", ["broken"])
    assert res["ok"] is False
    assert "not loadable" in res["error"]


# --------------------------------------------------------------------------- #
# Codegen cleanliness
# --------------------------------------------------------------------------- #
def test_generated_agent_with_skills_is_codegen_clean(tmp_path: Path) -> None:
    _make_agent_with_skills(tmp_path, "clean", "greeter", "summarizer")
    _assert_codegen_clean(_agent_py(tmp_path, "clean"), tmp_path, "skill_agent")


def test_generated_agent_two_toolsets_shares_single_anchor(tmp_path: Path) -> None:
    """Two SkillToolsets on one agent share a single ``_ADK_SKILLS_DIR`` anchor (dedup)."""
    create_llm(str(tmp_path), "clean2", "assistant", instruction="Help.")
    for n in ("greeter", "summarizer", "translator"):
        create(str(tmp_path), "clean2", n, description=f"{n} skill.", instruction=f"# {n}")
    attach(str(tmp_path), "clean2", "assistant", ["greeter", "summarizer"], name="set_a")
    attach(str(tmp_path), "clean2", "assistant", ["translator"], name="set_b")
    txt = _agent_py(tmp_path, "clean2")
    assert txt.count('_ADK_SKILLS_DIR = Path(__file__).parent / "skills"') == 1
    _assert_codegen_clean(txt, tmp_path, "two_toolsets")


# --------------------------------------------------------------------------- #
# FUNCTIONAL PROOF — the generated SkillToolset loads real skills offline
# --------------------------------------------------------------------------- #
def test_functional_skill_toolset_loads_real_skills_offline(tmp_path: Path) -> None:
    """The generated agent.py imports; its SkillToolset loads the on-disk skills via the real
    ADK loader and exposes the real ADK skill tools — offline, no API key."""
    _make_agent_with_skills(tmp_path, "fn_run", "greeter", "summarizer")
    info = _probe_toolset(str(tmp_path), "fn_run", "assistant_skills")
    assert info["toolset_type"] == "SkillToolset"
    assert info["skills"] == ["greeter", "summarizer"]
    # The 4 core skill tools the real SkillToolset always exposes (no registry → no search_skills).
    assert info["tool_names"] == [
        "list_skills",
        "load_skill",
        "load_skill_resource",
        "run_skill_script",
    ]


# --------------------------------------------------------------------------- #
# Mount wiring — in-memory client + end-to-end functional proof
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_skills_mounted_names_and_functional(tmp_path: Path) -> None:
    mcp = build_server()
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        for expected in (
            "skills_create",
            "skills_list",
            "skills_load",
            "skills_attach",
            "skills_registry_info",
        ):
            assert expected in tool_names, f"missing: {expected}"
        # No double prefix (e.g. skills_skills_create).
        assert not any(n.startswith("skills_skills_") for n in tool_names)

        path = str(tmp_path)
        # Scaffold an agent (agents domain) then drive a full skills flow via the client.
        assert (
            await client.call_tool(
                "agents_create_llm",
                {"path": path, "app_name": "client_sk", "name": "assistant", "instruction": "Go."},
            )
        ).data["ok"] is True
        created = await client.call_tool(
            "skills_create",
            {
                "path": path,
                "app_name": "client_sk",
                "name": "greeter",
                "description": "Greets the user.",
                "instruction": "# Greeter\nGreet warmly.",
            },
        )
        assert created.data["ok"] is True
        listed = await client.call_tool("skills_list", {"path": path, "app_name": "client_sk"})
        assert [s["id"] for s in listed.data["data"]["skills"]] == ["greeter"]
        loaded = await client.call_tool(
            "skills_load", {"path": path, "app_name": "client_sk", "name": "greeter"}
        )
        assert loaded.data["data"]["name"] == "greeter"
        attached = await client.call_tool(
            "skills_attach",
            {
                "path": path,
                "app_name": "client_sk",
                "agent_name": "assistant",
                "skill_names": ["greeter"],
            },
        )
        assert attached.data["ok"] is True
        assert attached.data["data"]["tools"] == ["skill_toolset:assistant_skills"]

    # Outside the client: the generated module must import + load the real skill.
    info = _probe_toolset(str(tmp_path), "client_sk", "assistant_skills")
    assert info["toolset_type"] == "SkillToolset"
    assert info["skills"] == ["greeter"]
