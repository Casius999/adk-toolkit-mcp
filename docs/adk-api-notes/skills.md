# `google.adk.skills` — Agent Skill Registry (real API, confirmed by introspection)

Captured 2026-06-02. **google-adk 2.1.0**, fastmcp 3.3.1, Python 3.12.

Source of truth for the `skills` domain (`src/adk_toolkit_mcp/domains/skills.py`) and the
`skill_toolset` tool kind added to `project_model`. Everything below was confirmed by
introspecting the **installed** package (no guessing), and a minimal real example was built and
round-tripped through the real ADK loaders.

---

## 1. What a "Skill" is

A **skill** is a folder of model-facing instructions + optional resources, following the
[Agent Skills spec](https://agentskills.io). ADK exposes it as a 3-layer model
(`google/adk/skills/models.py`):

- **L1 `Frontmatter`** — discovery metadata parsed from `SKILL.md`'s YAML frontmatter.
- **L2 `instructions: str`** — the markdown **body** of `SKILL.md` (loaded when the skill is
  triggered).
- **L3 `Resources`** — `references/`, `assets/`, `scripts/` (loaded on demand).

### `Skill` (pydantic BaseModel)

```
Skill(*, frontmatter: Frontmatter, instructions: str, resources: Resources = Resources(...))
```

- `frontmatter: Frontmatter` (required)
- `instructions: str` (required) — the SKILL.md body
- `resources: Resources` (default empty)
- `.name` (property → `frontmatter.name`)
- `.description` (property → `frontmatter.description`)

### `Frontmatter` (pydantic BaseModel, `extra="allow"`, `populate_by_name=True`)

| field | type | required | notes |
|-------|------|----------|-------|
| `name` | `str` | **yes** | kebab-case (a-z, 0-9, `-`), ≤ 64 chars. **Must equal the skill's directory name.** snake_case also allowed *only* if the `SNAKE_CASE_SKILL_NAME` feature is on — it is **OFF by default in 2.1.0**, so kebab-case is the rule. |
| `description` | `str` | **yes** | non-empty, ≤ 1024 chars |
| `license` | `str?` | no | |
| `compatibility` | `str?` | no | ≤ 500 chars |
| `allowed_tools` | `str?` | no | YAML key `allowed-tools` (alias); space-delimited pre-approved tools (experimental) |
| `metadata` | `dict[str, Any]` | no | client-specific; `adk_additional_tools` (a **list** of tool names) lets an activated skill pull in extra agent tools |

Name validation regex (kebab, default): `^[a-z0-9]+(-[a-z0-9]+)*$`.

### `Resources` (pydantic BaseModel)

- `references: dict[str, str | bytes]` — markdown docs/examples
- `assets: dict[str, str | bytes]` — templates, schemas, etc.
- `scripts: dict[str, Script]` — executable scripts (`Script(src: str)`)
- helpers: `get_reference/get_asset/get_script`, `list_references/list_assets/list_scripts`.

---

## 2. On-disk layout (confirmed in `_utils.py`)

A skill is a **directory** whose name equals the frontmatter `name`:

```
<skills_base_path>/
  <skill-name>/
    SKILL.md            # REQUIRED. "SKILL.md" or "skill.md".
    references/         # optional → loaded recursively into Resources.references
    assets/             # optional → Resources.assets
    scripts/            # optional → Resources.scripts (each wrapped in Script)
```

`SKILL.md` format (parsed by `_parse_skill_md_content`):

```markdown
---
name: greeter
description: What the skill does and when to use it.
---
# Body becomes `instructions`

Markdown instructions for the model…
```

Hard rules enforced by the loader (errors raised → we surface as `err(...)`):

- File **must start** with `---` and have a properly **closed** `---` frontmatter (else
  `ValueError`).
- Frontmatter must be a YAML **mapping**.
- `skill_dir.name` **must equal** `frontmatter.name` (else `ValueError`:
  *"Skill name '…' does not match directory name '…'"*).
- `references/`/`assets/`/`scripts/` are loaded recursively; `__pycache__` skipped; non-UTF-8
  files skipped for text content.

---

## 3. Loader / lister functions (the public surface)

Re-exported from `google.adk.skills` (`__init__.py`):

```python
list_skills_in_dir(skills_base_path: str | Path) -> dict[str, Frontmatter]
load_skill_from_dir(skill_dir: str | Path) -> Skill
list_skills_in_gcs_dir(bucket_name, skills_base_path="", project_id=None, credentials=None) -> dict[str, Frontmatter]
load_skill_from_gcs_dir(bucket_name, skill_id, skills_base_path="", project_id=None, credentials=None) -> Skill
```

- `list_skills_in_dir(base)` iterates the **immediate subdirectories** of `base`, reads each
  `SKILL.md`'s frontmatter only (lightweight), and returns `{skill_id: Frontmatter}`. Invalid
  skills are **logged and skipped** (not raised). A non-directory `base` → empty dict + warning.
- `load_skill_from_dir(skill_dir)` loads the **full** skill (frontmatter + body + resources).
- The GCS variants require the `google-cloud-storage` extra (`pip install google-adk[gcp]`) and
  raise `ImportError` otherwise — the toolkit does **not** import them at runtime; the `skills`
  domain is **local-dir only** (matching the toolkit's code-first, no-cloud-deps stance).

Round-trip verified (real call):

```
base/greeter/SKILL.md (+ references/langs.md)
→ list_skills_in_dir(base) == {"greeter": Frontmatter(name="greeter", description="…")}
→ load_skill_from_dir(base/"greeter") == Skill(name="greeter", instructions="# Greeter…",
                                               resources.references == {"langs.md": "en: Hello\n…"})
```

---

## 4. `SkillRegistry` — an **abstract** interface (no concrete impl shipped)

`google.adk.skills.SkillRegistry` is an **ABC** (`skill_registry.py`):

```python
class SkillRegistry(ABC):
    @abstractmethod
    async def get_skill(self, *, name: str) -> Skill: ...
    @abstractmethod
    async def search_skills(self, *, query: str) -> list[Frontmatter]: ...
    def search_tool_description(self) -> str | None: ...   # optional, defaults to None
```

- Methods: `get_skill`, `search_skills`, `search_tool_description`.
- **There is no concrete `SkillRegistry` implementation in google-adk 2.1.0** (no
  `DirSkillRegistry`, etc.). `SkillToolset` accepts an optional `registry=` to enable dynamic
  remote lookup + a `search_skills` tool, but you must implement the ABC yourself.
- The `skill_registry` symbol re-exported from the package is the **module**, not an instance.

**Toolkit decision (honest):** the `registry_info` tool does **not** instantiate the ABC. It
builds a directory-backed inventory via the real `list_skills_in_dir` and reports the registered
skills (id + name + description) — the faithful local equivalent of "what a registry over this
dir would expose". We do not fabricate a `SkillRegistry` subclass.

---

## 5. Attach mechanism — `SkillToolset` (the real wiring onto an agent)

Skills attach to an agent via a **toolset** placed directly in `tools=[...]`. It lives at
`google.adk.tools.skill_toolset` (NOT in `google.adk.tools`'s top-level `dir()` — the module is
lowercase, which is why a naive `'Skill' in dir(tools)` returns `[]`).

```python
from google.adk.tools.skill_toolset import SkillToolset

SkillToolset(
    skills: list[Skill] | None = None,
    *,
    registry: SkillRegistry | None = None,
    code_executor: BaseCodeExecutor | None = None,
    script_timeout: int = 300,
    additional_tools: list[ToolUnion] | None = None,
)
```

- A `SkillToolset` is a `BaseToolset` → goes **directly** into `LlmAgent(tools=[skill_toolset])`
  (like `OpenAPIToolset`/`McpToolset`; confirmed by import probe).
- It exposes these **core tools** to the model (`get_tools()`): `list_skills`, `load_skill`,
  `load_skill_resource`, `run_skill_script` — **plus** `search_skills` **iff** a `registry=` is
  given.
- It injects a system instruction (`_DEFAULT_SKILL_SYSTEM_INSTRUCTION`) telling the model to
  `load_skill` before acting.
- `run_skill_script` needs a `code_executor` (toolset-level or on the agent); without one it
  returns a `NO_CODE_EXECUTOR` error (it does not crash construction).
- Duplicate skill names in `skills=[...]` → `ValueError("Duplicate skill name …")`.
- `from_config(...)` classmethod exists (declarative config path) — not used by the toolkit.

### Skill tool classes (all `@experimental(FeatureName.SKILL_TOOLSET)`)

`ListSkillsTool`, `LoadSkillTool`, `LoadSkillResourceTool`, `RunSkillScriptTool`,
`SearchSkillsTool` — instantiated internally by `SkillToolset`; not constructed directly by the
toolkit.

### EXPERIMENTAL warning (important for the test gate)

`SkillToolset` and all skill tool classes are decorated `@experimental(FeatureName.SKILL_TOOLSET)`.
The feature is **enabled** (`is_feature_enabled(...) == True`) but instantiation emits a
**`UserWarning`**:

```
[EXPERIMENTAL] feature FeatureName.SKILL_TOOLSET is enabled.
```

This is a `UserWarning`, **not** a `DeprecationWarning` — so it does **not** trip the
`pytest -W error::DeprecationWarning` gate. Generated `agent.py` emits it at import; the
`find_spec`/exec import probe tolerates it (it is not an error). Tests that build a `SkillToolset`
filter `UserWarning` to keep output clean.

---

## 6. How the toolkit renders this (the `skill_toolset` tool kind)

Code-first, regeneration-based (like every other `tools`-domain kind). The toolkit:

1. **`skills_create`** writes a skill folder under the project's skills dir
   `<path>/<app_name>/skills/<name>/SKILL.md` (kebab-case `name` == dir name), conforming to the
   layout in §2 so the real `load_skill_from_dir`/`list_skills_in_dir` find it.
2. **`skills_attach`** adds a `skill_toolset` `ToolSpec` to an `LlmAgent` and regenerates
   `agent.py`. Rendered code (stable for `ruff format` / `ruff check --select I`):

   ```python
   from pathlib import Path

   from google.adk.skills import load_skill_from_dir
   from google.adk.tools.skill_toolset import SkillToolset

   _ADK_SKILLS_DIR = Path(__file__).parent / "skills"
   <var> = SkillToolset(
       skills=[
           load_skill_from_dir(_ADK_SKILLS_DIR / "greeter"),
           load_skill_from_dir(_ADK_SKILLS_DIR / "summarizer"),
       ]
   )
   ```

   and `<var>` goes into `tools=[...]`. Skills are loaded **from disk at the agent's runtime**
   (the toolkit emits source; it imports `load_skill_from_dir` only inside its own domain tools to
   read fields, never bakes skill content into `agent.py`). When a single skill is attached the
   list folds inline; multiple skills fold one-per-line (ruff-stable). The
   `_ADK_SKILLS_DIR = Path(__file__).parent / "skills"` line is emitted **once** per module
   (deduplicated like an import) so multiple skill toolsets share it.

`ToolSpec` (new fields): `kind="skill_toolset"`, `name` (toolset variable identifier),
`skill_names: tuple[str, ...]` (the skill dir names to load), `skills_dir` (relative dir, default
`"skills"`). `ref_key()` → `skill_toolset:<name>`. No auth (a `SkillToolset` takes no
`auth_credential`).

---

## 7. Summary — what exists vs. what doesn't

| Capability | Real in 2.1.0? | Toolkit exposure |
|------------|----------------|------------------|
| `Skill` / `Frontmatter` / `Resources` / `Script` models | ✅ | `load` returns the fields |
| `load_skill_from_dir` / `list_skills_in_dir` | ✅ | `load` / `list` / `registry_info` |
| On-disk `SKILL.md` + `references/`/`assets/`/`scripts/` layout | ✅ | `create` writes it |
| `SkillToolset` (attach to agent) + 4 core tools | ✅ (experimental) | `attach` (renders into `agent.py`) |
| `search_skills` tool | ✅ (needs a `registry=`) | not wired (no concrete registry) |
| Concrete `SkillRegistry` impl | ❌ (ABC only) | `registry_info` = dir-backed inventory (honest) |
| GCS skill loaders | ✅ (needs `[gcp]` extra) | not wired (local-only, no cloud deps) |
| `SkillTool` / `ListSkillsTool` as standalone public tools | exist but internal to `SkillToolset` | not exposed individually |
