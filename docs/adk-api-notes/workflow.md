# ADK API notes — `workflow` domain

Captured by introspection on 2026-06-02. google-adk **2.1.0**, fastmcp **3.3.1**, Python 3.12.
These are observed facts (run against the installed packages **and verified by constructing +
running** real `Workflow` objects), not guesses.

The `google.adk.workflow` package is a **graph orchestration engine**: non-linear /
conditional / cyclical agent (and function) execution. This is a genuinely new ADK 2.0
dimension, distinct from the linear/static `SequentialAgent` / `ParallelAgent` / `LoopAgent`
(the latter are deprecated in 2.1.0 — cf. `agents.md`).

## Module exports

```python
import google.adk.workflow as w
# w.__all__-ish (public names):
#   BaseNode, Node, FunctionNode, JoinNode, Workflow,
#   Edge, START, DEFAULT_ROUTE, RetryConfig, NodeTimeoutError, node
```

- `Workflow`, `Node`, `FunctionNode`, `JoinNode`, `BaseNode` are all **Pydantic models**
  (subclasses of `BaseNode`). `__init__` is `(self, /, **data: Any)` — construct with keyword
  args only; the public API is the set of model fields.
- `node` is a **decorator** turning a plain callable into a `FunctionNode`.
- `START` is a sentinel `BaseNode(name='__START__')` — the graph entry marker.
- `DEFAULT_ROUTE` is the string `'__DEFAULT__'` (fallback route value).

## How a graph is built — `Workflow(name=..., edges=[...])`

A `Workflow` is constructed from a `name` and a list of **edges**. There is **no separate
`add_node` call in ADK** — nodes are introduced *implicitly* by referencing them as edge
endpoints. `model_post_init` compiles `edges` into a validated `graph` automatically.

```python
from google.adk.workflow import Workflow, START, node

@node
def step_a(ctx, node_input):
    return {"a": 1}

@node
def step_b(ctx, node_input):
    return {"b": 2}

wf = Workflow(
    name="my_flow",
    edges=[
        (START, step_a),     # entry edge: START -> step_a
        (step_a, step_b),    # step_a -> step_b
    ],
)
# wf.graph is auto-built and validated; wf.graph.nodes == [START, step_a, step_b]
# Terminal nodes (no outgoing edges) == {"step_b"}
```

### Edge forms (all verified)

`Workflow.edges: list[EdgeItem]` where an `EdgeItem` is **either**:

1. **A 2-tuple `(from_node, to_node)`** — an unconditional edge (`route=None`).
2. **A 2-tuple `(from_node, {route_value: to_node, ...})`** — fan-out / conditional routing.
   The producing node returns a route value (a `str | int | bool`); the engine follows the
   matching branch. This compiles into one `Edge` per dict entry, each with `route=<key>`.
3. **An explicit `Edge(from_node=..., to_node=..., route=<value|list|None>)`** object.

`route` on an `Edge` is `bool | int | str | list[...] | None`. A list means "follow this edge
when the emitted route matches **any** value in the list".

A node endpoint may be: another `BaseNode` (incl. an `LlmAgent` or a nested `Workflow`), a
`BaseTool`, a **bare callable** (auto-wrapped into a `FunctionNode`), the `START` literal, a
tuple of those, or a route dict.

### Node kinds

| Kind | Construction | Wraps |
|------|-------------|-------|
| `LlmAgent` (or any `BaseAgent`) | used directly as an edge endpoint | an LLM agent — `LlmAgent` **is** a `BaseNode` |
| `FunctionNode` | `@node` decorator, or a bare callable endpoint, or `FunctionNode(func=...)` | a Python callable `(ctx, node_input) -> output|route` |
| `JoinNode` | `JoinNode(name=...)` | a fan-in barrier (waits for **all** predecessors) |
| nested `Workflow` | a `Workflow` used as an edge endpoint | a sub-graph |

`FunctionNode.__init__` (the only non-`**data` signature):

```python
FunctionNode(*, func: Callable[..., Any], name: str | None = None,
             rerun_on_resume: bool = False, retry_config: RetryConfig | None = None,
             timeout: float | None = None, auth_config: AuthConfig | None = None,
             parameter_binding: Literal['state', 'node_input'] = 'state',
             state_schema: type[BaseModel] | None = None)
```

A bare callable endpoint gets `name = func.__name__`. `JoinNode._requires_all_predecessors`
is `True` (verified) — it is the canonical fan-in.

### `Workflow` model fields (verified)

`name` (**required**), `description`, `edges` (default `[]`), `max_concurrency`
(`int | None`), `rerun_on_resume` (default `True`), `wait_for_output`, `retry_config`,
`timeout`, `input_schema`, `output_schema`, `state_schema`, `graph` (compiled, auto).

## Graph validation rules (enforced at construction)

`Graph.validate_graph()` runs inside `model_post_init`. It raises `ValueError` (a Pydantic
`ValidationError`-wrapped one for field issues) on:

- **duplicate node names** / **duplicate edges**;
- **no path from START** — every node must be reachable from `START`; `START` must have **no
  incoming edge**;
- **unconditional cycle** — a cycle made entirely of `route=None` edges loops forever and is
  rejected. *Cycles are allowed only if at least one edge in the cycle is conditional
  (routed).* This is how ReAct-style loops are expressed (verified):
  ```python
  edges=[(START, reason), (reason, act),
         (act, {"continue": reason, "stop": finish})]   # conditional cycle: OK
  ```
- **multiple terminal nodes producing output** — at FINALIZE a workflow must have at most one
  terminal output (a node with no outgoing edges). Fan-in to a single `JoinNode` keeps a
  single terminal.

## How a Workflow roots and runs

### Rooting — `root_agent = <workflow>` works

The ADK `AgentLoader` (`google.adk.cli.utils.agent_loader`) accepts a module-level `root_agent`
that is **either a `BaseAgent` OR a `BaseNode`** (verified in source):

```python
if isinstance(module_candidate.root_agent, (BaseAgent, BaseNode)):
    return module_candidate.root_agent
```

Since `Workflow` is a `BaseNode`, the toolkit renders `root_agent = <workflow_name>` into
`agent.py` exactly like an agent root — **no new discovery hook is required**. `adk web` /
`adk run` / `adk api_server` discover it via the same `root_agent` attribute.

### Running — `InMemoryRunner(node=...)` / `Runner(... )` over a node

`InMemoryRunner.__init__` exposes a dedicated **`node=`** parameter (distinct from `agent=`):

```python
InMemoryRunner(agent: BaseAgent | None = None, *, node: Any = None,
               app_name: str | None = None, plugins=..., app=..., ...)
```

A `Workflow` is passed as `node=wf` (NOT `agent=`). **Verified offline end-to-end** with two
`LlmAgent` nodes backed by `FakeLlm` (`tests/unit/fake_llm.py`):

```python
writer = LlmAgent(name="writer",   model=FakeLlm(model="fake", answer="draft text"),    instruction="Write.")
reviewer = LlmAgent(name="reviewer", model=FakeLlm(model="fake", answer="reviewed text"), instruction="Review.")
wf = Workflow(name="editorial", edges=[(START, writer), (writer, reviewer)])

runner = InMemoryRunner(node=wf, app_name="editorial_app")
# run_async(...) emits, in order:  ('writer','draft text'), ('reviewer','reviewed text')
```

A **pure function-node** workflow with conditional routing also runs offline with **no LLM at
all** (verified): `(START, classify), (classify, {"urgent": handle_urgent, "normal":
handle_normal})` dispatches to the matching branch based on `classify`'s return value.

> Caveat for the `run` domain: the toolkit's existing `run` tools call
> `Runner(agent=root_agent)` and `import_root_agent` returns the module's `root_agent`. A
> `Workflow` is a `BaseNode`, **not** a `BaseAgent`, so it cannot be passed via `agent=`.
> Running a Workflow root therefore needs the `node=` entry point (what `adk web` and a direct
> `InMemoryRunner(node=...)` use). The `workflow` domain proves construction + an offline run
> via a subprocess probe using `InMemoryRunner(node=...)`; it does not change the `run` domain.

## `node` decorator signature

```python
node(node_like=None, *, name=None, rerun_on_resume=None, retry_config=None,
     timeout=None, parallel_worker=False, auth_config=None) -> FunctionNode
```

`@node`-decorated callables become `FunctionNode`s carrying the function name. In the
toolkit's generated code we render function nodes as plain `def`s decorated with `@node` (or
referenced bare in the edge list — both are equivalent; we use the bare-callable form so the
generated module stays minimal and `ast.parse`-clean).

## What the toolkit wraps (and what it leaves out)

**Wrapped** (the solid, verified surface):

- `Workflow(name=..., edges=[...])` construction;
- node kinds: **agent** node (wraps an existing model agent / `LlmAgent`), **function** node
  (generated `def`), **join** node (`JoinNode` fan-in);
- edges: unconditional `(from, to)` and **conditional** `(from, {route: to})` (incl.
  loop-back cycles via a routed edge);
- setting a workflow as the app **root** (`root_agent = <workflow>`).

**Intentionally left out** (experimental / out of a code-first sidecar's reliable scope):

- `input_schema` / `output_schema` / `state_schema` typed Pydantic schemas on nodes/workflow
  (would require generating user model classes; not needed for the core graph surface);
- dynamic node scheduling via `ctx.run_node()` (runtime-only, not a static graph feature);
- HITL resume / checkpoint replay (`rerun_on_resume`, `resume_inputs`) — the engine supports
  it but it is a runtime concern, not expressible in the static sidecar;
- `retry_config` / `timeout` / `max_concurrency` tuning (could be added later as optional
  node/workflow kwargs; omitted to keep the first cut focused and reliable).
