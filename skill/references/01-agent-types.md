# 01 — Agent types & composition (the `agents` domain)

Choose the right agent type and wire the graph. Maps to the `agents_*` tools. All five ADK agent
classes import from `google.adk.agents`; `Agent is LlmAgent` (alias).

## The agent types

| Type | Class | What it does | Toolkit tool |
|---|---|---|---|
| LLM | `LlmAgent` | A reasoning agent backed by a model; carries tools, instruction, callbacks, sub_agents. The workhorse. | `agents_create_llm` |
| Sequential | `SequentialAgent` | Runs `sub_agents` in order, passing context along. **Deprecated in 2.1.0** (still functional). | `agents_create_sequential` |
| Parallel | `ParallelAgent` | Runs `sub_agents` concurrently. **Deprecated in 2.1.0** (still functional). | `agents_create_parallel` |
| Loop | `LoopAgent` | Repeats `sub_agents` up to `max_iterations`. **Deprecated in 2.1.0** (still functional). | `agents_create_loop` |
| Custom | `BaseAgent` subclass | Arbitrary orchestration via an `async _run_async_impl` generator (toolkit emits a stub). | `agents_create_custom` |
| Remote A2A | `RemoteA2aAgent` | A proxy to another agent served over A2A (see `09-a2a.md`). | `a2a_consume` (not an `agents_*` tool) |

> **Deprecation note.** `SequentialAgent`/`ParallelAgent`/`LoopAgent` emit a `DeprecationWarning` in
> 2.1.0 ("use Workflow instead") but are fully functional and the toolkit keeps emitting them. The
> Workflow successor API is not yet stable. Prefer composing LLM agents + sub_agents where you can;
> use the workflow agents when you genuinely need fixed sequential/parallel/loop control flow.

## Decision tree — how should agent A use agent B?

```
Need agent B's behavior available to agent A?
├── B is a remote service (served over A2A, possibly another team/runtime)
│     → RemoteA2aAgent proxy:  a2a_consume(path, app_name, name=B, agent_card_url=...)
│       then compose B as a sub_agent of A (agents_compose).
├── A should DELEGATE control to B (B becomes a child in A's tree; ADK can transfer to it)
│     → sub_agents:  put B in A's sub_agents (create A with it, or agents_compose A [B]).
│       Use when B is a peer/sub-task handler and you want ADK's agent-transfer machinery.
└── A should CALL B like a function and get a result back (B does NOT take over the turn)
      → AgentTool:  tools_add_agent_tool(path, app_name, agent_name=A, target_agent=B).
        B is wrapped as a tool of A. B is NOT also a sub_agent (single-parent rule).
```

### sub_agents vs AgentTool — the crisp rule

- **`sub_agents`** = composition/delegation. ADK can **transfer** the conversation to a sub-agent;
  the sub-agent runs its own turn. Good for routers, planners, specialist hand-off.
- **`AgentTool`** = encapsulation. The parent **invokes** the wrapped agent as a tool and incorporates
  its result into its own turn. Good for "ask the summarizer agent and continue".
- An agent wrapped as an `AgentTool` is **not** also added as a `sub_agent` — ADK enforces a
  **single-parent** rule (an agent instance has at most one parent). The toolkit respects this.

### When LoopAgent vs ParallelAgent vs SequentialAgent

- **Sequential**: ordered pipeline (A → B → C), each stage sees prior output. e.g. draft → critique → revise.
- **Parallel**: independent sub-tasks run at once, results gathered. e.g. fan-out research across sources.
- **Loop**: iterate until a condition or `max_iterations` (the toolkit defaults `max_iterations=3`,
  must be > 0). e.g. refine-until-good. A sub-agent can emit `exit_loop` (a builtin tool) to stop early.

### Gemini-native vs LiteLlm

That's a **model** decision, not an agent-type decision — see `03-models.md`. Any `LlmAgent` can use
either a Gemini string model or a `LiteLlm(...)` wrapper (Anthropic/OpenAI/Ollama/LM Studio/etc.).

## The `agents` domain tools

All operate on `(path, app_name, …)`, mutate the sidecar, and regenerate `agent.py`.

| Tool | Signature (key args) | Notes |
|---|---|---|
| `agents_create_llm` | `name, model="gemini-2.5-flash", instruction="", description="", output_key=None` | Add/update an `LlmAgent`. `output_key` stores the agent's output in session state under that key. |
| `agents_create_sequential` | `name, sub_agents: list[str], description=""` | sub_agents must already exist in the model. |
| `agents_create_parallel` | `name, sub_agents: list[str], description=""` | Same existence rule. |
| `agents_create_loop` | `name, sub_agents: list[str], max_iterations=3, description=""` | `max_iterations` > 0. |
| `agents_create_custom` | `name, description=""` | Emits a `BaseAgent` subclass stub (`_run_async_impl` no-op generator) + instance. |
| `agents_compose` | `name, sub_agents: list[str]` | **Replace** an existing agent's sub_agents. Rejects self-reference, missing children, and custom agents. |
| `agents_set_root` | `name` | Designate which agent is the app's `root_agent`. The agent must exist. **Do this once your graph is built.** |
| `agents_as_tool` | `agent_name` | Returns the **source snippet** for `AgentTool(agent=…)` (read-only; no file change). To actually attach, use `tools_add_agent_tool`. |
| `agents_list` | `(path, app_name)` | List agents (name + type) and the current root. Read-only. |
| `agents_get` | `name` | Full serialized spec of one agent. Read-only. |

## Gotchas

- **Create children before parents.** `create_sequential/parallel/loop` and `compose` require the named
  sub_agents to already exist (creation order is otherwise free).
- **Single parent.** Don't assign the same agent as a child of two parents — ADK raises at import time.
  The toolkit's codegen never shares instances; cycle detection guards the other structural failure
  (a cycle returns a clean `err`).
- **Only LlmAgents carry tools.** `tools_add_*` reject non-LLM agents. Workflow/custom agents orchestrate
  via sub_agents, not tools.
- **Set the root.** A graph with no designated root has no entry point. Call `agents_set_root` (the
  scaffold's initial `root_agent` is already named after the app, but composing new agents may change
  intent).
