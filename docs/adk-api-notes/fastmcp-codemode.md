# FastMCP 3.3.1 — Code Mode & tool tags (P6a introspection)

Captured 2026-06-01 against the installed `fastmcp==3.3.1` (Python 3.12). Everything
below was confirmed by introspecting the installed package and by running an in-memory
`fastmcp.Client` — not guessed.

## TL;DR

- **Real Code Mode EXISTS** in fastmcp 3.3.1 as a *catalog transform*:
  `fastmcp.experimental.transforms.code_mode.CodeMode`. It is applied with
  `mcp.add_transform(CodeMode(...))` and collapses the **entire** tool catalog into a
  tiny discovery + execute surface (default: `search`, `get_schema`, `execute`).
- `@FastMCP.tool(...)` **accepts `tags: set[str] | None`** (and so does `@FastMCP.prompt`).
  Tags surface as `tool.tags` and are read by the Code Mode discovery tools for
  tag-filtered search — so tagging every tool by its domain is both hygiene AND the
  enabler for cheap discovery.
- **Caveat (documented honestly):** the `execute` meta-tool's *default* sandbox
  (`MontySandboxProvider`) needs the optional `pydantic-monty` package (the
  `fastmcp[code-mode]` extra), which is **NOT installed** in this project's env. The
  discovery tools (`search`, `get_schema`, `tags`, `list_tools`) work **without** monty;
  only calling `execute` requires it (lazily, at call time, with a clear `ImportError`).
  We therefore wire real Code Mode and gate it behind an opt-in flag, and document that
  `execute` needs the extra. We do NOT fake it.

## The relevant `FastMCP` surface (3.3.1)

`[m for m in dir(FastMCP) if any(k in m.lower() for k in ('transform','tag','enable','disable','tool','prompt','mount'))]`:

```
add_prompt, add_tool, add_tool_transformation, add_transform, call_tool, disable,
enable, get_app_tool, get_prompt, get_tool, get_tool_by_hash, list_prompts,
list_tools, mount, prompt, remove_tool, remove_tool_transformation, tool,
transforms, wrap_transform
```

- `FastMCP.add_transform(transform: Transform) -> None` — register a catalog transform.
- `@FastMCP.tool(... , tags: set[str] | None = None, ...)` — the decorator accepts `tags`.
- `@FastMCP.prompt(... , tags: set[str] | None = None, ...)` — same for prompts.
- `FastMCP.list_tools(*, run_middleware=True) -> Sequence[Tool]` — reflects transforms.

## Code Mode — exact API

```python
from fastmcp.experimental.transforms.code_mode import (
    CodeMode, Search, GetSchemas, GetTags, ListTools, MontySandboxProvider,
)
```

`CodeMode` is a `CatalogTransform` (a `Transform` subclass), constructor:

```python
CodeMode(
    *,
    sandbox_provider: SandboxProvider | None = None,   # default MontySandboxProvider()
    discovery_tools: list[DiscoveryToolFactory] | None = None,  # default [Search(), GetSchemas()]
    execute_tool_name: str = "execute",
    execute_description: str | None = None,
)
```

- `transform_tools(...)` returns `[*discovery_tools, execute]` — i.e. it **replaces** the
  whole catalog. Proven: a server with N tagged tools exposes exactly
  `['execute', 'get_schema', 'search']` after `add_transform(CodeMode())`.
- Discovery tool factories are composable. Each is a callable
  `(get_catalog) -> Tool`. Built-ins:
  - `Search(*, search_fn=None, name='search', default_detail='brief', default_limit=None)`
    — BM25 ranking by default; the synthetic `search(query, tags=None, detail, limit)`
    tool **filters by `tool.tags`** before searching (`tags=['agents']`, or
    `'untagged'`).
  - `GetSchemas(*, name='get_schema', default_detail='detailed')` — returns parameter
    schemas for named tools (`get_schema(tools=[...], detail=...)`).
  - `GetTags(*, name='tags', default_detail='brief')` — lists tags with tool counts
    (or full per-tag tool listing). **Directly motivates domain tags.**
  - `ListTools(*, name='list_tools', default_detail='brief')`.
- The `execute` tool exposes `call_tool(name, params)` inside a sandbox and runs
  LLM-authored Python; it is always present.

### Sandbox dependency (the honest caveat)

`MontySandboxProvider.run(...)` does `importlib.import_module("pydantic_monty")` and, if
missing, raises:

```
ImportError: CodeMode requires pydantic-monty for the Monty sandbox provider.
Install it with `fastmcp[code-mode]` or pass a custom SandboxProvider.
```

`pydantic_monty` is **not installed** here (verified: `ModuleNotFoundError`). Confirmed
by running an in-memory client against a `CodeMode`-transformed server:

- `client.list_tools()` → `['execute', 'get_schema', 'search']` ✓ (no monty)
- `client.call_tool('search', {...})` ✓ (no monty)
- `client.call_tool('get_schema', {...})` ✓ (no monty)
- `client.call_tool('execute', {...})` would raise the `ImportError` above unless
  `pydantic-monty` is installed or a custom `SandboxProvider` is supplied.

We do **not** add `pydantic-monty` to the locked deps (no `uv.lock` change). The discovery
surface — the actual token-efficiency win for an 81-tool catalog — is fully functional
without it.

## How P6a uses this

- **Tags (TASK 1):** every `@<domain>_server.tool` carries `tags={"<domain>"}`. Exposed
  tool **names are unchanged** (`tags` is metadata only). The 5 workflow prompts carry
  `tags={"workflow"}`.
- **Code Mode (TASK 2):** `build_server(code_mode: bool = False)`. Default = direct tools
  (all 81 exposed by name — existing read-through tests unchanged). When `code_mode=True`
  (or env `ADK_TOOLKIT_CODE_MODE` ∈ {1,true,yes,on}), `build_server` calls
  `mcp.add_transform(CodeMode(discovery_tools=[Search(), GetSchemas(), GetTags()]))` so the
  surface collapses to `search` / `get_schema` / `tags` / `execute`. `main()` reads the env
  flag so users launch either mode. Token-surface reduction proven qualitatively: 81
  top-level tools → 4 (a ~95% reduction in listed tools).
- We add `GetTags` to the discovery set specifically because we tag by domain — the model
  can browse the 15 domains, then `search(tags=[...])`, then `get_schema(...)`, then call
  through `execute`.

## Why opt-in (not default)

The toolkit's primary UX and its full read-through test suite call tools **by name**
(`project_create`, `run_agent`, …). Making Code Mode the default would erase those names
and require `pydantic-monty` for any real execution. Opt-in keeps the direct-tool UX and
the 81-name contract intact, while making the cheap catalog available to clients that want
it (and that install `fastmcp[code-mode]` for `execute`).
