# 06 — Runtime execution (the `run` domain)

Execute a real ADK agent loop via a `Runner`. Maps to the `run_*` tools. This domain imports the
project's `root_agent` from `<path>/<app_name>/agent.py`, wires it into a `Runner` on the configured
session/memory/artifact services (`runtime.json`), and collects the `Event`s.

## How execution works (the Runner)

- **`Runner`** is wired with the toolkit's configured services (NOT `InMemoryRunner`, which would build
  its own services and bypass `runtime.json` + the singleton cache). `session_service` is required;
  memory/artifact services are passed only when a backend is configured.
- `Runner.run_async(*, user_id, session_id, new_message, run_config)` is an **async generator of
  `Event`**. The toolkit ensures the session exists (creates it if absent — `auto_create_session` is
  False by default), runs the loop, and serializes each event.
- **Offline-testable.** A fake LLM (`FakeLlm(BaseLlm)` overriding `generate_content_async`) drives the
  whole loop with **no API key** — LLM → tool call → tool execution → final answer. This is how the
  toolkit proves the wiring; you can think of `run_agent` as fully deterministic given a fake model.

## `RunConfig` — streaming + call budget

| `streaming_mode` | Enum value | Meaning |
|---|---|---|
| `NONE` (default) | `None` | one final `LlmResponse` per turn |
| `SSE` | `'sse'` | server-sent-event streaming (partial events) — `run_stream` reports progress |
| `BIDI` | `'bidi'` | bidirectional live (Gemini Live websocket) — `run_live`, experimental |

`max_llm_calls` bounds the number of LLM calls (ADK default **500**). Precedence for `run_agent` /
`run_stream`: an explicit `max_llm_calls` arg **wins**; else the value persisted by
`safety_settings(..., max_llm_calls=N)` on the **root** agent's spec; else the ADK default.

## The `run` domain tools

| Tool | Key args | Notes |
|---|---|---|
| `run_agent` | `user_id, session_id, message, max_llm_calls=None, streaming_mode="NONE"` | Run the root agent on `message`. Returns `{event_count, events, final_text, ...}`. Creates the session if needed. |
| `run_stream` | `user_id, session_id, message, max_llm_calls=None` | Like `run_agent` but forces `SSE` and reports per-event progress to the MCP client (`ctx.report_progress` + `ctx.info`). Same return shape. |
| `run_live` | `user_id, session_id, message, max_llm_calls=None` | **[EXPERIMENTAL]** BIDI/Gemini Live. Requires a live-capable model + creds (`GOOGLE_API_KEY` or Vertex). Detects capability and returns a clean `err` **before** any connection if absent — never hangs. Cannot run in CI. |
| `run_config_build` | `streaming_mode="NONE", max_llm_calls=None, response_modalities=None` | Validate + describe a `RunConfig` without executing. Returns the descriptor + valid `streaming_options`. |
| `run_inspect_events` | `events: list[dict]` | **Pure** (no I/O). Summarize a serialized event list: counts function calls, unique tool names, transfers, state_delta keys, and final text. Feed it the `events` from `run_agent`. |

## Serialized event shape

`run_agent`/`run_stream` return `events` as a list of:
```json
{
  "author": "agent_name",
  "text": "joined text parts or null",
  "function_calls": [{"name": "...", "args": {...}}],
  "function_responses": [{"name": "...", "response": ...}],
  "state_delta": {...},
  "transfer_to_agent": "name or null",
  "is_final": true,
  "partial": false
}
```
`final_text` in the result is the text of the last final event. Use `run_inspect_events` to analyze a
tool-call trajectory (e.g. which tools fired, in what order).

## Reload-after-edit

`run_*` re-imports `root_agent` with a fresh module name + compile/exec on each call (defeats the
Windows bytecode mtime cache), so editing the app via the authoring tools and immediately re-running
picks up the change. You never need to restart the server to see a regenerated `agent.py`.

## Failure modes (all clean `err`, never a hang)

- Corrupt `runtime.json` → `err`. Missing/broken `agent.py` or absent `root_agent` → `err`.
- Invalid `streaming_mode` → `err` (must be NONE/SSE/BIDI). A backend needing the `gcp` extra without
  it → `err`. `run_live` without creds/capability → actionable `err` before connecting.

## Next

- Evaluate the agent's quality → `07-eval.md`. Run an interactive dev UI / one-shot CLI → `08-deploy.md`
  (`dev_*`).
