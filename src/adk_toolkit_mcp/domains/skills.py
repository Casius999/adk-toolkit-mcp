"""`skills` domain: Agent Skill Registry (``google.adk.skills``, new in ADK 2.x).

A FastMCP sub-server mounted by the root server under the ``skills`` namespace (tools exposed as
``skills_<name>`` on the client side). Functions named with **BARE** names (``create``, ``list``,
``load``, ``attach``, ``registry_info``) — cf. ``docs/adk-api-notes/conventions.md``.

A **skill** is an on-disk folder (``<name>/SKILL.md`` + optional ``references/``/``assets/``/
``scripts/``) of model-facing instructions and resources (cf. ``docs/adk-api-notes/skills.md``).
This domain manages the project's skills directory ``<path>/<app_name>/skills/`` (code-first, via
:class:`~adk_toolkit_mcp.workspace.Workspace`) and wires skills onto an agent via a
``SkillToolset`` (the real ADK attach mechanism), regenerating ``agent.py`` exactly like the
``tools`` domain.

The list/load/registry tools round-trip skills through the **real** loaders
``google.adk.skills.list_skills_in_dir`` / ``load_skill_from_dir`` (proving on-disk schema
conformance, not a guess). ``SkillRegistry`` is an ABC with no concrete impl in google-adk 2.1.0,
so ``registry_info`` reports a directory-backed inventory (the faithful local equivalent) rather
than fabricating a subclass. Everything returns the ``{ok, data, error}`` envelope; invalid inputs
return ``err(...)`` (never an exception).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import (
    SKILLS_DIR,
    ProjectModel,
    ToolSpec,
    add_or_replace_tool,
    add_or_update_agent,
    is_identifier,
    is_skill_name,
    load_model,
    regenerate,
    save_model,
    validate_tool_spec,
)
from ..workspace import Workspace

skills_server: FastMCP = FastMCP("skills")

#: app_name = Python package identifier (both folder AND module name).
_APP_NAME_ERR = (
    "Invalid app_name: expected a Python identifier "
    "(letters, digits, underscore; not starting with a digit)."
)

#: Guidance reused across tools when a skill name is malformed.
_SKILL_NAME_ERR = (
    "A skill name must be lowercase kebab-case (a-z, 0-9, hyphens), <= 64 chars, with no "
    "leading/trailing/consecutive hyphens (it is also the on-disk directory name)."
)


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _skills_base(path: str, app_name: str) -> Path:
    """Absolute path of the project's skills directory (``<path>/<app_name>/skills``)."""
    return Path(path) / app_name / SKILLS_DIR


def _load(path: str, app_name: str) -> ProjectModel | dict[str, Any]:
    """Load the model; return an ``err(...)`` (dict) if the sidecar is corrupt."""
    ws = _app_ws(path, app_name)
    try:
        return load_model(ws, app_name)
    except ValueError as exc:
        return err(str(exc))


def _frontmatter(name: str, description: str) -> str:
    """Render the YAML frontmatter block for a ``SKILL.md`` (name + description only).

    Both values are single-line here (the MCP args are plain strings); a multi-line description
    would still be valid YAML but the toolkit keeps it on one line for determinism. The name is
    pre-validated as kebab-case == directory name, so it needs no quoting.
    """
    # Quote the description to survive ``:`` and other YAML-significant characters safely.
    safe_desc = description.replace("\\", "\\\\").replace('"', '\\"')
    return f'---\nname: {name}\ndescription: "{safe_desc}"\n---\n'


# --------------------------------------------------------------------------- #
# MCP tools — define a skill on disk
# --------------------------------------------------------------------------- #
@skills_server.tool(tags={"skills"})
def create(
    path: str,
    app_name: str,
    name: str,
    description: str,
    instruction: str,
) -> dict[str, Any]:
    """Define a skill on disk in the layout ADK expects, so the real loaders find it.

    Writes ``<path>/<app_name>/skills/<name>/SKILL.md`` with a valid YAML frontmatter
    (``name`` == ``<name>``, the required ``description``) and the markdown ``instruction`` as the
    body (the skill's L2 instructions). ``name`` must be lowercase **kebab-case** (it is also the
    directory name — ADK requires the two to match). Idempotent: re-creating the same ``name``
    overwrites its ``SKILL.md``. Attach it to an agent later with ``attach``.

    Conforms to ``google.adk.skills`` (cf. ``docs/adk-api-notes/skills.md``): the written folder is
    immediately loadable via ``load`` / ``list`` (which call the real ADK loaders).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_skill_name(name):
        return err(f"Invalid skill name: {name!r}. {_SKILL_NAME_ERR}")
    if not description.strip():
        return err(
            "description is required (what the skill does and when the model should use it)."
        )
    if len(description) > 1024:
        return err(f"description must be at most 1024 characters (got {len(description)}).")

    ws = _app_ws(path, app_name)
    rel = f"{SKILLS_DIR}/{name}/SKILL.md"
    body = instruction.strip("\n")
    content = _frontmatter(name, description) + (f"\n{body}\n" if body else "")
    changed = ws.write(rel, content)
    return ok(
        {
            "app_name": app_name,
            "skill": name,
            "skill_dir": str(_skills_base(path, app_name) / name),
            "skill_md": str(ws.path(rel)),
            "changed": changed,
        }
    )


# --------------------------------------------------------------------------- #
# MCP tools — read (round-trip through the REAL ADK loaders)
# --------------------------------------------------------------------------- #
@skills_server.tool(tags={"skills"}, name="list")
def list_skills(path: str, app_name: str) -> dict[str, Any]:
    """List the project's skills via the real ``list_skills_in_dir`` (frontmatter only). Read-only.

    Scans ``<path>/<app_name>/skills/`` with ``google.adk.skills.list_skills_in_dir`` and returns
    ``{id, name, description}`` per discovered skill (invalid skill folders are skipped by ADK). A
    missing skills dir yields an empty list (not an error).

    Named ``list_skills`` in Python (so as not to shadow the ``list`` builtin in this module), but
    **registered under the BARE tool name ``list``** -> exposed as ``skills_list``.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    from google.adk.skills import list_skills_in_dir

    base = _skills_base(path, app_name)
    listing = list_skills_in_dir(base)  # tolerant: empty dict if base is absent
    skills = [
        {"id": skill_id, "name": fm.name, "description": fm.description}
        for skill_id, fm in sorted(listing.items())
    ]
    return ok({"app_name": app_name, "skills_dir": str(base), "skills": skills})


@skills_server.tool(tags={"skills"})
def load(path: str, app_name: str, name: str) -> dict[str, Any]:
    """Load one skill fully via the real ``load_skill_from_dir`` and return its fields. Read-only.

    Loads ``<path>/<app_name>/skills/<name>/`` with ``google.adk.skills.load_skill_from_dir`` and
    returns the real ``Skill`` object's fields: ``name``/``description`` (frontmatter), the L2
    ``instructions`` (SKILL.md body), the frontmatter dict, and the names of any L3 resources
    (``references``/``assets``/``scripts``). A missing/invalid skill returns ``err(...)`` (the ADK
    loader's message, e.g. SKILL.md not found or name/dir mismatch).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_skill_name(name):
        return err(f"Invalid skill name: {name!r}. {_SKILL_NAME_ERR}")

    from google.adk.skills import load_skill_from_dir

    skill_dir = _skills_base(path, app_name) / name
    if not skill_dir.is_dir():
        return err(f"Skill not found: {name!r} (no directory at {skill_dir}). Create it first.")
    try:
        skill = load_skill_from_dir(skill_dir)
    except (FileNotFoundError, ValueError) as exc:
        return err(str(exc))

    return ok(
        {
            "app_name": app_name,
            "skill_dir": str(skill_dir),
            "name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
            "frontmatter": skill.frontmatter.model_dump(by_alias=True),
            "resources": {
                "references": skill.resources.list_references(),
                "assets": skill.resources.list_assets(),
                "scripts": skill.resources.list_scripts(),
            },
        }
    )


@skills_server.tool(tags={"skills"})
def registry_info(path: str, app_name: str) -> dict[str, Any]:
    """Report the skills registered over the project's skills dir (directory-backed inventory).

    ``google.adk.skills.SkillRegistry`` is an **abstract** interface (no concrete implementation
    ships in google-adk 2.1.0 — cf. ``docs/adk-api-notes/skills.md``). Rather than fabricate a
    subclass, this builds the faithful local equivalent: it enumerates the skills dir via the real
    ``list_skills_in_dir`` and reports each registered skill (``id``/``name``/``description``) plus
    the count. This mirrors what a directory-backed registry's ``search_skills`` would surface.
    Read-only.
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)

    from google.adk.skills import list_skills_in_dir

    base = _skills_base(path, app_name)
    listing = list_skills_in_dir(base)
    registered = [
        {"id": skill_id, "name": fm.name, "description": fm.description}
        for skill_id, fm in sorted(listing.items())
    ]
    return ok(
        {
            "app_name": app_name,
            "skills_dir": str(base),
            "registry_kind": "directory",
            "count": len(registered),
            "registered": registered,
        }
    )


# --------------------------------------------------------------------------- #
# MCP tool — attach skills onto an agent (via SkillToolset) + regenerate
# --------------------------------------------------------------------------- #
@skills_server.tool(tags={"skills"})
def attach(
    path: str,
    app_name: str,
    agent_name: str,
    skill_names: list[str],
    name: str | None = None,
) -> dict[str, Any]:
    """Wire skills onto ``agent_name`` via a ``SkillToolset`` and regenerate ``agent.py``.

    Attaches a ``skill_toolset`` tool to the (existing ``LlmAgent``) ``agent_name``: the generated
    code builds ``<var> = SkillToolset(skills=[load_skill_from_dir(_ADK_SKILLS_DIR / "<n>"), ...])``
    and places ``<var>`` in ``tools=[...]`` (a ``SkillToolset`` is a ``BaseToolset`` accepted
    directly — cf. ``docs/adk-api-notes/skills.md``). Skills are loaded **from disk at the agent's
    runtime** from ``<app>/skills/``.

    ``skill_names`` are skill **directory names** (each must already exist on disk via ``create``
    and be valid kebab-case). ``name`` is the toolset variable identifier (defaults to
    ``<agent_name>_skills``). "Append unique / replace by name" semantics: re-attaching the same
    ``name`` replaces the toolset. Each skill folder is verified on disk and loadable via the real
    ADK loader before wiring (so a typo fails fast with a clear message).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if not is_identifier(agent_name):
        return err(f"Invalid agent_name: {agent_name!r}. Expected a Python identifier.")
    if not skill_names:
        return err("skill_names is empty: provide at least one skill to attach (create it first).")

    toolset_name = name if name is not None else f"{agent_name}_skills"
    if not is_identifier(toolset_name):
        return err(f"Invalid toolset name: {toolset_name!r}. Expected a Python identifier.")

    model = _load(path, app_name)
    if isinstance(model, dict):  # err()
        return model

    agent = model.get(agent_name)
    if agent is None:
        return err(f"Agent not found: {agent_name!r}. Create it first (agents domain).")
    if agent.type != "llm":
        return err(
            f"The {agent_name!r} agent is of type {agent.type!r}; only 'llm' agents "
            "(LlmAgent) carry tools."
        )

    # Verify each skill exists on disk AND loads via the REAL ADK loader (schema conformance),
    # so wiring never references a missing/invalid skill.
    base = _skills_base(path, app_name)
    load_error = _verify_skills_loadable(base, skill_names)
    if load_error is not None:
        return err(load_error)

    tool = ToolSpec(
        kind="skill_toolset",
        name=toolset_name,
        skill_names=tuple(skill_names),
    )
    tool_error = validate_tool_spec(tool, model, agent_name)
    if tool_error is not None:
        return err(tool_error)

    updated = add_or_replace_tool(agent, tool)
    model = add_or_update_agent(model, updated)

    ws = _app_ws(path, app_name)
    try:
        regen = regenerate(ws, model)
    except ValueError as exc:  # cycle detected at render time (agents graph)
        return err(str(exc))
    sidecar_changed = save_model(ws, model)
    return ok(
        {
            "app_name": app_name,
            "agent": agent_name,
            "toolset": toolset_name,
            "skills": list(skill_names),
            "tools": [t.ref_key() for t in updated.tool_specs()],
            "sidecar": str(ws.path(".adk_toolkit/agents.json")),
            "regenerated": {"agent_py": regen["agent_py"], "init_py": regen["init_py"]},
            "changed": bool(regen["changed"]) or sidecar_changed,
        }
    )


def _verify_skills_loadable(base: Path, skill_names: list[str]) -> str | None:
    """Return an error if any of ``skill_names`` is invalid / missing / not ADK-loadable, else None.

    Uses the real ``google.adk.skills.load_skill_from_dir`` so attachment is gated on genuine
    on-disk schema conformance (not a guess): a malformed ``SKILL.md`` or a name/dir mismatch is
    surfaced with ADK's own message.
    """
    from google.adk.skills import load_skill_from_dir

    for sname in skill_names:
        if not is_skill_name(sname):
            return f"Invalid skill name: {sname!r}. {_SKILL_NAME_ERR}"
        skill_dir = base / sname
        if not skill_dir.is_dir():
            return (
                f"Skill not found: {sname!r} (no directory at {skill_dir}). "
                "Create it first (skills create)."
            )
        try:
            load_skill_from_dir(skill_dir)
        except (FileNotFoundError, ValueError) as exc:
            return f"Skill {sname!r} is not loadable: {exc}"
    return None
