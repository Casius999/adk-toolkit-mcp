# 11 — Safety & guardrails (the `safety` domain)

Two guardrail mechanisms: **callbacks** (per-agent, can short-circuit) and **plugins** (global, across
the whole run). Plus **safety settings** (Gemini safety thresholds + LLM call budget). Maps to `safety_*`.

## Callbacks vs plugins — local vs global

```
Scope of the guardrail?
├── ONE agent, hook into its model/tool/agent lifecycle, can short-circuit
│     → safety_add_callback   (renders a real function on the LlmAgent callback kwarg)
└── EVERY agent/tool in the run (cross-cutting policy, logging, global denylist)
      → safety_add_plugin     (renders a BasePlugin subclass + runtime manifest)
```

- **Callbacks** attach to a single `LlmAgent` via its real callback kwargs and run **positionally**.
  Returning a non-None value **short-circuits** (the model/tool is never called). Best for
  agent-specific guardrails.
- **Plugins** are **global** (`BasePlugin`, keyword-only async hooks) wired through the **`App`** path
  (`Runner(app=App(plugins=[...]))`) — NOT the deprecated `Runner(plugins=)`. Best for cross-cutting
  concerns: audit logging, a global tool denylist.

## `safety_add_callback` — per-agent guardrail
```
safety_add_callback(path, app_name, agent_name, hook, policy)
```
- `hook` ∈ {`before_model`, `after_model`, `before_tool`, `after_tool`, `before_agent`, `after_agent`}.
- `policy` is `{"kind": "<policy>", ...params}`. Three concrete policies:
  - **`block_keywords`** (before_model): `{"kind": "block_keywords", "keywords": "bomb,hack",
    "refusal": "..."}` — refuses (short-circuits the LLM with a canned `LlmResponse`) if the user text
    contains a blocked term.
  - **`max_input_chars`** (before_model): `{"kind": "max_input_chars", "max_chars": "2000"}` — refuses
    if the input exceeds N characters.
  - **`block_tool`** (before_tool): `{"kind": "block_tool", "denylist": "delete_db", "message": "..."}`
    — short-circuits the tool (returns the message dict) if its name is in the denylist.
- Renders a **real generated function** attached via the real kwarg (e.g.
  `before_model_callback=_guard_before_model`). One callback per hook (a second replaces it). Targets an
  existing **LlmAgent**.

> Short-circuit semantics are proven offline: a `block_keywords` guard returns the refusal and the
> model is never invoked; a `block_tool` guard returns its dict and the tool body never runs.

## `safety_add_plugin` — global policy
```
safety_add_plugin(path, app_name, name, kind, config=None)
```
Generates/extends `<app_dir>/<app>/plugins.py` with a `BasePlugin` subclass and registers it in the
**runtime manifest** (`runtime.json` `plugins` key) so the Runner imports it. Two kinds:
- **`logging`** — records every event via `on_event_callback` into a module-level `<name>_events` list
  (inspectable offline) and logs via `logging`.
- **`tool_denylist`** — globally short-circuits any tool whose name is in `config={"denylist":
  "delete_db,drop_table"}` (via `before_tool_callback`).

The plugin is a module-level variable `<name>` in `plugins.py`; the manifest lists its var name so
`build_runner` wires it through `App(plugins=[...])`.

> **Plugins use the `App` path, not `Runner(plugins=)`** — the latter is deprecated in 2.1.0 and would
> raise under `-W error::DeprecationWarning`. The toolkit's `App(name, root_agent, plugins=[...])` path
> emits zero warnings.

## `safety_settings` — Gemini safety + call budget
```
safety_settings(path, app_name, agent_name, max_llm_calls=None, gemini_safety=None)
```
(Exposed as `safety_settings`.) Targets an existing **LlmAgent**:
- **`gemini_safety`** — a list of `{"category": "<HarmCategory>", "threshold": "<HarmBlockThreshold>"}`.
  **Routes through the SAME `generate_content_config` rendering** as `models_generate_config` (reuses
  `types.SafetySetting` — no duplication). Merges with an existing config (preserves temperature, etc.).
  Enum members are listed in `03-models.md`.
- **`max_llm_calls`** — stored on the agent spec and surfaced. It maps to `RunConfig.max_llm_calls`
  (validated by the run domain). It is **NOT** rendered into `agent.py` (it's a runtime setting). At run
  time, `run_agent`/`run_stream` use it as the default budget unless the caller passes an explicit
  `max_llm_calls` (explicit wins). See `06-runtime.md`.

## Choosing a guardrail (quick guide)

| Need | Use |
|---|---|
| Refuse certain prompts on one agent | `safety_add_callback(hook="before_model", policy={"kind":"block_keywords", ...})` |
| Cap input size on one agent | `safety_add_callback(hook="before_model", policy={"kind":"max_input_chars", ...})` |
| Block a dangerous tool on one agent | `safety_add_callback(hook="before_tool", policy={"kind":"block_tool", ...})` |
| Block dangerous tools **everywhere** | `safety_add_plugin(kind="tool_denylist", config={"denylist": ...})` |
| Audit every event | `safety_add_plugin(kind="logging")` |
| Tighten Gemini safety thresholds | `safety_settings(gemini_safety=[...])` |
| Cap total LLM calls per run | `safety_settings(max_llm_calls=N)` |

## Overlap honesty

- `safety_settings(gemini_safety=...)` reuses the `models` domain's safety rendering (one
  implementation). Set sampling + safety together via `models_generate_config`, or think in guardrail
  terms via `safety_settings` — both land in the same `generate_content_config`.
- `max_llm_calls` is the bridge to `RunConfig` — there's no separate "set call budget" tool; it lives on
  `safety_settings`.
