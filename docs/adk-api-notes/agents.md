# ADK API notes — `agents` domain

Captured by introspection on 2026-06-01. google-adk **2.1.0**, fastmcp **3.3.1**, Python 3.12.
These are observed facts (run against the installed packages), not guesses.

## Canonical imports

All five agent classes import cleanly from the package root:

```python
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, BaseAgent
```

- `google.adk.agents.Agent is LlmAgent` -> `True` (alias; the `adk create` template uses
  `from google.adk.agents.llm_agent import Agent`).
- Submodule paths also exist (`google.adk.agents.llm_agent.LlmAgent`,
  `...sequential_agent.SequentialAgent`, etc.) but the package-root import is the canonical
  one and is what we emit in generated code.

## Constructor signatures

Every agent is a **Pydantic model**; `__init__` is `(self, /, **data: Any)`, so the public API
is the set of model fields, not positional parameters. Construct with keyword args only.

### `BaseAgent` (shared fields)

`name` (**required**), `description`, `sub_agents`, `parent_agent`, plus callbacks
(`before_agent_callback`, `after_agent_callback`), schemas, `timeout`, `retry_config`, etc.

`name` is the only required field. `sub_agents: list[BaseAgent]` defaults to empty.

Abstract method to override for custom agents:

```python
async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]: ...
```

It is an **async generator** (must `yield`, or be a no-op generator), NOT a plain coroutine.
A valid minimal stub:

```python
class MyAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        return
        yield  # unreachable; makes this an async generator
```

(Confirmed instantiable: `MyAgent(name="custom_one", description="d")`.)

### `LlmAgent` (adds, all optional)

`model`, `instruction`, `global_instruction`, `static_instruction`, `tools`, `output_key`,
`generate_content_config`, `planner`, `code_executor`, model/tool callbacks, transfer flags.

Minimal: `LlmAgent(name="root", model="gemini-2.5-flash", instruction="...")`.
`output_key` and `tools` are plain optional fields.

### `SequentialAgent` / `ParallelAgent`

Only `BaseAgent` fields. Orchestrate via `sub_agents`. No extra constructor params.

### `LoopAgent`

`BaseAgent` fields **plus** `max_iterations` (optional int).
`LoopAgent(name="lp", sub_agents=[...], max_iterations=3)`.

## CRITICAL — workflow agents emit `DeprecationWarning` in 2.1.0

Constructing `SequentialAgent`, `ParallelAgent`, or `LoopAgent` raises:

```
DeprecationWarning: <X> is deprecated and will be removed in future versions.
Please use Workflow instead.
```

`LlmAgent` and `BaseAgent` subclasses do **not** warn.

Implications:

- This is ADK's own deprecation, not a toolkit defect. The generated `agent.py` uses the
  documented, currently-functional API, so we keep emitting `SequentialAgent` / `ParallelAgent`
  / `LoopAgent`. (Migrating to an undocumented `Workflow` surface is out of scope for P1.)
- The task mandates running the suite once with `-W error::DeprecationWarning`. Therefore our
  **in-process** tests never construct workflow agents directly; `project_model` tests assert on
  the rendered **source string** only (no construction), which is warning-free.
- The **functional probe** imports the generated module in a **subprocess**. To keep that probe
  hermetic regardless of the warning, the subprocess is launched with
  `-W ignore::DeprecationWarning` so importing a module that builds a `SequentialAgent`/`LoopAgent`
  succeeds. The probe still proves the real ADK objects instantiate with correct types/attrs.

## CRITICAL — single-parent constraint

An agent instance may belong to **one** parent only. If two different parents list the same
child in `sub_agents`, ADK raises at construction:

```
ValidationError: Agent `a1` already has a parent agent, current parent: `pipeline`,
trying to add: `lp`
```

Because the generated module defines each agent as a module-level variable referenced by name,
a model whose graph assigns one child to two parents will fail at *import time* of the generated
module (surfaced clearly, not silently). Our codegen therefore does not attempt to share
instances; building a valid tree (each agent has at most one parent) is the caller's
responsibility, and the toolkit's cycle detection guards against the other structural failure.

## `AgentTool` (for `agents.as_tool`)

```python
from google.adk.tools import AgentTool
AgentTool.__init__(self, agent: BaseAgent, skip_summarization: bool = False, *,
                   include_plugins: bool = True, propagate_grounding_metadata: bool = False)
```

- Module: `google.adk.tools.agent_tool`.
- Wraps an existing agent so it can be used as a tool by another `LlmAgent`
  (`LlmAgent(..., tools=[AgentTool(agent=some_agent)])`).
- `agents.as_tool` returns the source snippet to do this (a helper; no file mutation), since the
  wrapping happens at the `tools`-domain layer in P3.

## Toolkit design consequences

- `project_model.render_agent_module` imports only the classes actually used by the model, in the
  fixed canonical order `LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, BaseAgent`.
- Agents are emitted **topologically ordered** (a referenced child before its parent); cycles
  raise `ValueError` which the domain tools convert to `err(...)`.
- Empty/None kwargs are omitted from generated calls (`output_key`, empty `tools`/`sub_agents`,
  empty `description`).
