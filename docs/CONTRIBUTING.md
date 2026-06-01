# Contributing to adk-toolkit-mcp

This document covers the development environment, the project conventions, the quality bar,
and a step-by-step guide for adding a new domain.

---

## Dev setup

Requires Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
# Clone and enter the repo
git clone <repo-url>
cd adk-toolkit-mcp

# Create the venv and install all dev deps (includes eval, sqlalchemy, ruff, mypy, pytest)
uv venv && uv sync --extra dev
```

The `dev` extra pulls in `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`,
`sqlalchemy`, and `google-adk[eval]` (offline evaluation metrics).

### Run the quality suite

```bash
# Lint + format check
uv run ruff check .
uv run ruff format --check .

# Type check
uv run mypy src

# Tests (all, with coverage)
uv run pytest -q --cov=adk_toolkit_mcp --cov-report=term-missing

# Tests with strict DeprecationWarning (the CI baseline)
uv run pytest -q -W error::DeprecationWarning
```

All four must pass before committing.

---

## Project conventions

### Tool naming (mandatory)

Each domain is a `FastMCP("<domain>")` instance named `<domain>_server`, mounted in
`server.py` via `mcp.mount(<domain>_server, namespace="<domain>")`. Tool functions use
**bare names** (no domain prefix):

```python
# CORRECT — bare name, exposed as `project_create`
@project_server.tool(tags={"project"})
def create(...) -> dict[str, Any]: ...

# WRONG — double-prefixed, would expose as `project_project_create`
@project_server.tool(tags={"project"})
def project_create(...) -> dict[str, Any]: ...
```

Never use `prefix=` (deprecated in fastmcp 3.3.1; emits `DeprecationWarning`).

### Domain tags (mandatory)

Every `@<domain>_server.tool` decorator must carry `tags={"<domain>"}`. This enables Code
Mode discovery and client-side filtering.

### `{ok, data, error}` envelope (mandatory)

Every tool must return the uniform envelope:

```python
from ..envelope import ok, err

return ok({"result": value})      # success
return err("Descriptive message.") # failure — actionable hint, never raises
```

`err(...)` never raises. Never swallow exceptions silently; always convert to `err(...)`.

An eval failure (agent does not meet thresholds) is `ok=True, data={passed: False}` — a
normal result, not an error.

### Generated code quality bar

Code written to disk by the author domains (`project`, `agents`, `tools`, `models`, `safety`)
must pass:

1. `ast.parse(source)` — syntactically valid Python.
2. `ruff format --check` — ruff-formatted (no whitespace/line-length diffs).
3. `ruff check --select I` — isort-clean import order.

The `render.py` / `_codegen.py` helpers enforce this. Tests assert all three for every
generated module.

### Lazy optional dependencies

No optional dependency is imported at module load time.

- **Author domains**: never import `google-adk`. Only manipulate the sidecar.
- **Runtime domains**: import ADK inside the tool body. Wrap `ImportError` /
  `ModuleNotFoundError` for a missing extra into `err(...)` with an install hint:
  ```python
  try:
      from google.adk.evaluation import AgentEvaluator
  except ModuleNotFoundError:
      return err("Install 'eval' extra: uv add 'adk-toolkit-mcp[eval]'")
  ```
- **`TYPE_CHECKING` guards**: use for type hints that reference heavy types.

### File and function size

- Files: ≤ 800 lines.
- Functions: ≤ 50 lines.

### Test coverage

Minimum 80% (current: ~95%). Write tests in `tests/unit/test_<domain>.py`.

Tests must run offline (no Google API key). Use `FakeLlm` / `ScriptedLlm` from
`tests/unit/fake_llm.py` for any test that invokes an agent loop.

The full suite must stay green under `-W error::DeprecationWarning`. When an ADK-internal
call emits a `DeprecationWarning` you cannot avoid, wrap it with a **narrowly scoped**
`warnings.catch_warnings()` filter (see `eval.py` and `mcp_bridge.py` for the pattern).

---

## How to add a new domain

### 1. Introspect the ADK surface

Before writing any code, run the relevant ADK APIs against the installed package and record
your findings. Write an introspection note to `docs/adk-api-notes/<domain>.md`. Base all
implementation decisions on confirmed facts, not on guesses or docs. See any existing
`adk-api-notes/*.md` file for the format.

### 2. Create `domains/<domain>.py`

```python
from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..workspace import Workspace

<domain>_server: FastMCP = FastMCP("<domain>")


@<domain>_server.tool(tags={"<domain>"})
def <bare_name>(path: str, app_name: str, ...) -> dict[str, Any]:
    """One-line docstring."""
    ws = Workspace(...)
    # ... implementation ...
    return ok({"result": value})
```

### 3. Mount it in `server.py`

Add one line in `build_server()`, after the existing mounts:

```python
from .domains.<domain> import <domain>_server
...
mcp.mount(<domain>_server, namespace="<domain>")
```

### 4. Write tests first (TDD)

Create `tests/unit/test_<domain>.py`. Write tests before implementing the domain body.
Use the in-memory `fastmcp.Client` for mounted-server integration tests:

```python
import asyncio
import pytest
from fastmcp import Client
from adk_toolkit_mcp.server import build_server

@pytest.mark.asyncio
async def test_<domain>_<tool>_ok(tmp_path):
    async with Client(build_server()) as client:
        result = await client.call_tool("<domain>_<bare>", {"path": str(tmp_path), ...})
        assert result.data["ok"] is True
```

For offline agent execution tests use `FakeLlm` from `tests/unit/fake_llm.py`.

### 5. Update the skill

If the new domain adds tools, update:

- `skill/references/13-tool-catalog.md` — add a new section for the domain.
- The relevant thematic reference (or create `skill/references/<n>-<domain>.md`).
- Re-copy the skill to `~/.claude/skills/adk-toolkit/`.

### 6. Run the full quality suite

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q -W error::DeprecationWarning --cov=adk_toolkit_mcp
```

All must pass. Fix any regressions before committing.

### 7. Commit

Use conventional commits:

```
feat: add <domain> domain (<n> tools)

- tool_one: ...
- tool_two: ...
Covers <brief description of what the domain enables>.
```

Update `PROGRESS.md` with the new domain's tool count and any noteworthy implementation facts.

---

## Key files reference

| File | Role |
|---|---|
| `src/adk_toolkit_mcp/server.py` | Root server, `build_server()`, Code Mode |
| `src/adk_toolkit_mcp/envelope.py` | `ok()` / `err()` envelope helpers |
| `src/adk_toolkit_mcp/workspace.py` | `Workspace` — sidecar / file I/O helper |
| `src/adk_toolkit_mcp/runtime.py` | `RuntimeConfig`, service factories, singleton cache |
| `src/adk_toolkit_mcp/run_core.py` | `build_runner`, `collect_events`, `import_root_agent` |
| `src/adk_toolkit_mcp/adk_cli.py` | `adk_executable`, `run_adk`, process registry |
| `src/adk_toolkit_mcp/project_model/` | Sidecar + codegen engine |
| `docs/adk-api-notes/conventions.md` | Naming convention (authoritative) |
| `docs/adk-api-notes/<domain>.md` | Per-domain ADK introspection notes |
| `tests/unit/fake_llm.py` | `FakeLlm` / `ScriptedLlm` for offline tests |
