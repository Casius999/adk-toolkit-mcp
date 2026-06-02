# ADK API notes — `planners` dimension (`google.adk.planners`)

Captured by introspection on 2026-06-02. google-adk **2.1.0**, google-genai bundled with it,
fastmcp **3.3.1**, Python 3.12. These are observed facts (run against the installed packages
**and verified by constructing the real planner objects and attaching them to an `LlmAgent`**),
not guesses.

`google.adk.planners` is the ADK dimension that lets an `LlmAgent` **plan before acting**: it
either turns on the model's native "thinking" feature (`BuiltInPlanner`) or injects a
Plan-Reason-Act prompt scaffold around any model (`PlanReActPlanner`).

## Module exports

```python
import google.adk.planners as p
# [n for n in dir(p) if not n.startswith('_')]:
#   ['BasePlanner', 'BuiltInPlanner', 'PlanReActPlanner',
#    'base_planner', 'built_in_planner', 'plan_re_act_planner']
```

The three classes (the public surface the toolkit wraps):

| Class | Module | Base |
|-------|--------|------|
| `BasePlanner` | `google.adk.planners.base_planner` | abstract base |
| `BuiltInPlanner` | `google.adk.planners.built_in_planner` | `BasePlanner` |
| `PlanReActPlanner` | `google.adk.planners.plan_re_act_planner` | `BasePlanner` |

Both `BuiltInPlanner` and `PlanReActPlanner` are `issubclass(..., BasePlanner) == True` (verified).

## Constructor signatures (verified)

```python
import inspect
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
inspect.signature(BuiltInPlanner)   # (*, thinking_config: 'types.ThinkingConfig')
inspect.signature(PlanReActPlanner) # ()
```

- **`BuiltInPlanner(*, thinking_config: types.ThinkingConfig)`** — `thinking_config` is a
  **required keyword-only** argument. It uses the model's native thinking feature (Gemini 2.5
  "thinking"); the `thinking_config` is forwarded into the request.
- **`PlanReActPlanner()`** — **no arguments**. It works with any model and injects a structured
  planning prompt (`/*PLANNING*/`, `/*REASONING*/`, `/*ACTION*/`, `/*FINAL_ANSWER*/` tags) and
  parses the response, so it needs no model-side thinking support.

### `thinking_config` is `google.genai.types.ThinkingConfig` (core, no extra)

`from google.genai import types` is part of `google-genai`, a **core** dependency pulled in by
`google-adk` (no optional extra). The dataclass:

```python
inspect.signature(types.ThinkingConfig)
# (*, includeThoughts: Optional[bool] = None,
#     thinkingBudget: Optional[int] = None,
#     thinkingLevel: Optional[ThinkingLevel] = None) -> None
# model_fields (snake_case): include_thoughts, thinking_budget, thinking_level
```

All three fields default to `None`, so **`types.ThinkingConfig()` (no args) is valid** and is the
safe "thinking on, no budget cap" form. The common knob is **`thinking_budget`** (an `int`: the
token budget allotted to thinking). Verified:

```python
from google.adk.planners import BuiltInPlanner
from google.genai import types
BuiltInPlanner(thinking_config=types.ThinkingConfig())                       # OK
BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=1024))   # OK; .thinking_config.thinking_budget == 1024
```

## How an `LlmAgent` takes a planner — `planner=` field (verified)

`LlmAgent` exposes a model field **`planner`**:

```python
from google.adk.agents import LlmAgent
LlmAgent.model_fields['planner'].annotation
# typing.Optional[google.adk.planners.base_planner.BasePlanner]
LlmAgent.model_fields['planner'].default   # None
```

So the planner is passed directly as a kwarg and round-trips:

```python
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
a = LlmAgent(
    name="x", model="gemini-2.5-flash",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=1024)),
)
type(a.planner).__name__   # 'BuiltInPlanner'  (verified)
```

A `PlanReActPlanner` attaches the same way: `LlmAgent(..., planner=PlanReActPlanner())`.

`planner` is only a kwarg on `LlmAgent` (the workflow agents `SequentialAgent` /
`ParallelAgent` / `LoopAgent` orchestrate sub-agents and have no `planner`; the planner belongs
to the LLM leaf). The toolkit therefore only renders `planner=` on `llm`-type agents.

## What the toolkit wraps

The two `kind` values exposed by the toolkit (`agents_set_planner`):

| `kind` | Renders | Import(s) emitted |
|--------|---------|-------------------|
| `built_in` | `planner=BuiltInPlanner(thinking_config=types.ThinkingConfig([thinking_budget=N]))` | `from google.adk.planners import BuiltInPlanner` + `from google.genai import types` |
| `plan_react` | `planner=PlanReActPlanner()` | `from google.adk.planners import PlanReActPlanner` |

- For `built_in`, an optional `thinking_budget` (int) is rendered as
  `types.ThinkingConfig(thinking_budget=N)`; omitted, it renders the **no-arg** `types.ThinkingConfig()`
  that is always valid (thinking on, no explicit budget).
- For `plan_react`, no config is needed; `thinking_budget` (if supplied) is ignored.

The planner is persisted in the sidecar (`.adk_toolkit/agents.json`) on the `AgentSpec` and fully
re-rendered into `agent.py` like every other agent attribute (no AST patching).

## Intentionally left out

- `include_thoughts` / `thinking_level` on `ThinkingConfig` — the budget is the common, portable
  knob; the others are model/preview-specific and add surface without clear payoff for a
  code-first sidecar. Users who need them can edit the (regenerated) `agent.py` or extend the
  spec later.
- A custom `BasePlanner` subclass — out of scope for a static sidecar (would require generating a
  user class). The two concrete ADK planners cover the documented surface.
