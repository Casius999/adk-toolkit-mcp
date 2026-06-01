# ADK API notes — `run` (P3a runtime execution core)

Captured 2026-06-01 by introspection. `google-adk` **2.1.0**, `fastmcp` **3.3.1**, Python 3.12.

These notes back the shared `run_core.py` helpers and the `run` domain sub-server. Unlike the
P1 author domains (which *write* `agent.py` source), the `run` domain **executes a real ADK
agent loop** through a `Runner` wired to the configured session/memory/artifact services from
`runtime.py`. The load-bearing proof is that a `FakeLlm(BaseLlm)` lets the full loop run
**offline (no API key)** in tests.

## `Runner` — constructor and `run_async`

```text
Runner(*, app: Optional[App] = None,
          app_name: Optional[str] = None,
          agent: Optional[BaseAgent] = None,
          node: Any = None,
          plugins: Optional[List[BasePlugin]] = None,
          artifact_service: Optional[BaseArtifactService] = None,
          session_service: BaseSessionService,            # the ONLY required arg
          memory_service: Optional[BaseMemoryService] = None,
          credential_service: Optional[BaseCredentialService] = None,
          plugin_close_timeout: float = 5.0,
          auto_create_session: bool = False)

async run_async(*, user_id: str, session_id: str,
                invocation_id: Optional[str] = None,
                new_message: Optional[types.Content] = None,
                state_delta: Optional[dict[str, Any]] = None,
                run_config: Optional[RunConfig] = None,
                yield_user_message: bool = False) -> AsyncGenerator[Event, None]
```

- `Runner` is **all keyword-only**. `session_service` is required; `app_name` + `agent` are
  the normal wiring. `memory_service` / `artifact_service` are optional → we pass them from
  `runtime.py` only when a backend is configured (else omit, ADK tolerates `None`).
- `run_async` is an **async generator** yielding `Event` objects. `new_message` is an optional
  `google.genai.types.Content`. There is no positional form.
- `auto_create_session=False` by default → the session must already exist; `collect_events`
  therefore ensures the session exists (creates it if `get_session` returns `None`) before
  iterating `run_async`. (We do NOT rely on `auto_create_session` to keep behaviour explicit
  and backend-agnostic.)

### `InMemoryRunner` (not used by the toolkit core)

```text
InMemoryRunner(agent=None, *, node=None, app_name=None, plugins=None, app=None,
               plugin_close_timeout=5.0)
```

`InMemoryRunner` builds its OWN in-memory session/memory/artifact services internally, which
would bypass the toolkit's configured `runtime.json` backends (and its singleton cache). We
therefore use **`Runner`** wired with the toolkit's services, not `InMemoryRunner`.

## `RunConfig` and `StreamingMode`

```text
RunConfig(*, speech_config=None, response_modalities: Optional[list[str]] = None,
             ... , streaming_mode: StreamingMode = StreamingMode.NONE,
             ... , max_llm_calls: int = 500, custom_metadata=None,
             get_session_config=None)

StreamingMode = [NONE (value None), SSE (value 'sse'), BIDI (value 'bidi')]
```

- `streaming_mode` is the `StreamingMode` enum; default `NONE`. `SSE` drives
  server-sent-event streaming (partial events). `BIDI` is for the live/bidi path.
- We validate a string against the enum **by name** (`StreamingMode[name.upper()]`):
  `StreamingMode.NONE.name == "NONE"`, `.SSE.name == "SSE"`, `.BIDI.name == "BIDI"`. The enum
  *values* are `None` / `'sse'` / `'bidi'` (lowercase) — we expose names in tool descriptors.
- `max_llm_calls` defaults to **500**. `build_run_config(max_llm_calls=None)` leaves the ADK
  default in place (does not pass the kwarg); a provided int is forwarded as-is.
- `response_modalities` is `Optional[list[str]]` (e.g. `["TEXT"]`).

## `BaseLlm.generate_content_async` — how we build a FakeLlm

```text
@abstractmethod
async def generate_content_async(self, llm_request: LlmRequest,
                                 stream: bool = False) -> AsyncGenerator[LlmResponse, None]
```

- It is an **async generator** (`inspect.isasyncgenfunction` → True), NOT a coroutine. A
  `FakeLlm` overrides it with `async def ... yield LlmResponse(...)`.
- Non-streaming contract (`stream=False`): yield **exactly one** `LlmResponse` with the
  complete output and `partial=False`. Streaming (`stream=True`): yield partial chunks
  (`partial=True`) then a final aggregated chunk (`partial=False`).
- `BaseLlm` is a **pydantic model**; a `FakeLlm` subclass declares its scripting state as a
  pydantic field (e.g. `calls: int = 0`) and constructs with `FakeLlm(model="fake")`.

### `LlmResponse` (pydantic, camelCase aliases, snake_case fields)

```text
LlmResponse(*, model_version=None, content: Optional[types.Content] = None,
               grounding_metadata=None, partial: Optional[bool] = None,
               turn_complete=None, finish_reason=None, error_code=None,
               error_message=None, interrupted=None, custom_metadata=None,
               usage_metadata=None, ... )
```

A canned final answer: `LlmResponse(content=types.Content(role="model",
parts=[types.Part.from_text(text="...")]))`. A canned tool call: a `Content` whose part is
`types.Part.from_function_call(name=..., args={...})`.

## Building `new_message` (the user turn)

```python
from google.genai import types
new_message = types.Content(role="user", parts=[types.Part.from_text(text=message)])
```

`types.Part.from_text(text=...)` and `types.Part.from_function_call(name=, args=)` are both
keyword-only. `Content.role` is `"user"` for the human turn, `"model"` for LLM output.

## `Event` accessors (confirmed on an instance)

`dir(Event)` relevant members: `get_function_calls`, `get_function_responses`,
`is_final_response`, `has_trailing_code_execution_result`, plus pydantic machinery. Instance
attributes: `author` (str), `content` (`Optional[types.Content]`), `actions` (`EventActions`),
`partial` (`Optional[bool]`; `"partial" in Event.model_fields` is True).

- `event.get_function_calls()` → `list[types.FunctionCall]` (each has `.name`, `.args`).
- `event.get_function_responses()` → `list[types.FunctionResponse]` (each has `.name`,
  `.response`).
- `event.actions.state_delta` → `dict` (empty `{}` by default), `.transfer_to_agent` →
  `Optional[str]`.
- `event.is_final_response()` → bool. `event.content.parts[*].text` holds text (may be `None`
  for function-call / function-response parts).

### Serialized event shape (`serialize_event`)

```python
{
  "author": str,
  "text": str | None,                       # joined text parts
  "function_calls": [{"name": str, "args": dict}],
  "function_responses": [{"name": str, "response": Any}],
  "state_delta": dict,                        # event.actions.state_delta
  "transfer_to_agent": str | None,
  "is_final": bool,
  "partial": bool | None,
}
```

## PROVEN offline agent loop (the load-bearing result)

With a `FakeLlm`/`ScriptedLlm(BaseLlm)` and **no API key**, wiring an `LlmAgent` directly
(`LlmAgent(name=, model=FakeLlm(...), tools=[python_fn])`) through `Runner` +
`run_async` produced:

- **Final-text case:** 1 event — `author=<agent>`, `is_final_response()=True`,
  text == the canned answer.
- **Tool-call loop case** (scripted: turn 1 yields a `function_call`, turn 2 yields final
  text), with a registered plain python tool `add_numbers(a, b)`:
  - event[0]: `get_function_calls() == [add_numbers(a=2,b=3)]`, not final.
  - event[1]: `get_function_responses() == [add_numbers]` (ADK auto-executed the tool), not
    final.
  - event[2]: `is_final_response()=True`, text == `"The sum is 5."`.

This proves the toolkit's `Runner` wiring executes a full agent loop (LLM → tool call → tool
execution → final answer) entirely offline. A plain python function in `tools=[...]` resolves
to a `FunctionTool` automatically (emits a benign `UserWarning`
`FeatureName.JSON_SCHEMA_FOR_FUNC_DECL` — NOT a `DeprecationWarning`, so it does not trip
`-W error::DeprecationWarning`).

## `run_live` / BIDI — why it can't run in CI, and how it degrades

```text
async run_live(*, user_id=None, session_id=None, live_request_queue: LiveRequestQueue,
               run_config=None, session=None) -> AsyncGenerator[Event, None]
```

- Live uses **`BaseLlm.connect(llm_request) -> BaseLlmConnection`** (a websocket to the Gemini
  Live API), NOT `generate_content_async`. The **base** `BaseLlm.connect` raises
  `NotImplementedError(f"Live connection is not supported for {self.model}.")`; only `Gemini`
  overrides it (an `@asynccontextmanager`) and even then requires a live-capable model + a real
  `GOOGLE_API_KEY` (or Vertex creds) + outbound websocket. It therefore **cannot run in CI**.
- `LlmAgent` resolves a string model to `canonical_model` (a `Gemini` instance);
  `agent.canonical_model.model` is the model id. We detect live capability by:
  1. requiring `GOOGLE_API_KEY` (AI Studio) OR `GOOGLE_GENAI_USE_VERTEXAI=TRUE` + project; and
  2. checking the resolved model's class actually overrides `connect` (i.e. is not the base
     `NotImplementedError` stub).
- The `run_live` tool performs the faithful wiring (import `root_agent`, build the runner,
  build a `LiveRequestQueue`, a BIDI `RunConfig`) but **short-circuits with an actionable
  `err`** before opening any connection when capability/key is absent — so it never hangs. It
  is marked **experimental** in its docstring.

## `import_root_agent` — unique module name to defeat the import cache

`import_root_agent(path, app_name)` loads `<path>/<app_name>/agent.py`'s `root_agent` via
`importlib.util.spec_from_file_location(<unique_name>, <agent.py>)`. The module name MUST be
**unique per call** (e.g. a monotonic counter / uuid suffix) so a re-import after the user
edits `agent.py` is NOT served stale from `sys.modules`. Import/attribute errors are wrapped in
a clear exception the tool converts to `err(...)`. Confirmed: editing `agent.py` between two
`import_root_agent` calls yields the UPDATED `root_agent` (proven in tests).

## `fastmcp.Context` progress (for `run_stream`)

```text
Context.report_progress(self, progress: float, total: float | None = None,
                         message: str | None = None) -> None   # async
Context.info(self, message: str, logger_name=None, extra=None) -> None             # async
```

Both are awaited. `run_stream` builds an `SSE` `RunConfig` and, for each event, awaits
`ctx.report_progress(i+1, total=None, message=...)` and `ctx.info(...)`. In unit tests the
streaming progress is exercised via `collect_events(..., progress=<async callback>)` (asserting
the callback is awaited once per event) and via an in-memory `fastmcp.Client`.
