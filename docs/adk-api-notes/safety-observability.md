# ADK 2.1.0 facts — safety (callbacks + plugins) & observability (OpenTelemetry)

Captured 2026-06-01 by introspection (P4c). `google-adk` 2.1.0, Python 3.12.
Every fact below was verified against the installed package, not guessed.

## Agent callbacks (`LlmAgent` kwargs)

`LlmAgent` exposes these callback kwargs (introspected via `inspect.signature` +
`model_fields`):

```
before_agent_callback   after_agent_callback
before_model_callback   after_model_callback   on_model_error_callback
before_tool_callback    after_tool_callback    on_tool_error_callback
```

Each accepts **a single callable OR a list of callables OR None**. The toolkit attaches
**one generated function** per hook via the real kwarg (e.g.
`before_model_callback=_guard_before_model`).

### Confirmed signatures + short-circuit semantics

The agent-level callable signatures (POSITIONAL args — the agent invokes them positionally;
this differs from the plugin hooks which are keyword-only, see below):

- `before_model_callback(callback_context, llm_request) -> LlmResponse | None`
  Returning a non-`None` `LlmResponse` **short-circuits the LLM** (the model is never called).
  **PROVEN offline**: a guard returning a canned refusal `LlmResponse` made the Runner emit
  ONLY that refusal; the `FakeLlm.answer` never appeared.
- `before_tool_callback(tool, args, tool_context) -> dict | None`
  Returning a non-`None` `dict` **short-circuits the tool** (used as the tool result).
  **PROVEN offline**: a denylist guard returning `{"error": ...}` for `add_numbers` made the
  function-response carry that dict; the real tool body never ran. `tool.name` is the tool's
  name (instance attr `self.name`, set in `BaseTool.__init__(*, name, description, ...)`).
- `before_agent_callback(callback_context) -> types.Content | None`
  Returning a non-`None` `Content` short-circuits the agent (skips its run, uses that content).

Reading the user's text inside `before_model_callback`: `llm_request.contents` is a
`list[google.genai.types.Content]`; the latest user turn is the last `Content` with
`role == "user"`; concatenate its `parts[*].text`.

Constructing the refusal:
```python
from google.adk.models import LlmResponse
from google.genai import types
LlmResponse(content=types.Content(role="model", parts=[types.Part.from_text(text="<refusal>")]))
```

`LlmResponse` is importable from `google.adk.models` (alongside `BaseLlm`, `LlmRequest`).

## Plugins (`BasePlugin` + `App` + `Runner`)

`from google.adk.plugins import BasePlugin`. Public hook methods (introspected):

```
before_run_callback        after_run_callback
on_user_message_callback   on_event_callback
before_agent_callback      after_agent_callback
before_model_callback      after_model_callback   on_model_error_callback
before_tool_callback       after_tool_callback    on_tool_error_callback
close
```

Plugin hooks are **keyword-only** (unlike the agent callbacks):

- `async on_event_callback(self, *, invocation_context, event) -> Event | None`
- `async before_tool_callback(self, *, tool, tool_args, tool_context) -> dict | None`
- `async before_run_callback(self, *, invocation_context) -> types.Content | None`
- `async before_model_callback(self, *, callback_context, llm_request) -> LlmResponse | None`

`BasePlugin.__init__` takes `name=` (a string). Subclass + override the hooks you need; the
base hooks return `None` (no-op), so a subclass only implements its policy.

### Wiring plugins into the Runner — use `App`, NOT the deprecated `plugins=`

`Runner.__init__` params: `app, app_name, agent, node, plugins, artifact_service,
session_service, memory_service, credential_service, plugin_close_timeout,
auto_create_session`.

- ⚠️ **`Runner(plugins=[...])` is DEPRECATED in 2.1.0** — emits a real `DeprecationWarning`
  ("The `plugins` argument is deprecated. Please use the `app` argument to provide plugins").
  Under `-W error::DeprecationWarning` this would RAISE.
- ✅ **Non-deprecated path: `App`**. `from google.adk.apps import App`. Fields:
  `name, root_agent, plugins, events_compaction_config, context_cache_config,
  resumability_config`. Construct `App(name=<app_name>, root_agent=<agent>, plugins=[...])`
  and pass `Runner(app=app, session_service=...)`. **Zero warnings** (verified). `runner.app_name`
  is then derived from `App.name`, and `runner.plugin_manager` is populated.

So `run_core.build_runner`: **no plugins → unchanged** (`Runner(app_name=, agent=, ...)`); **with
plugins → `Runner(app=App(name, root_agent, plugins=[...]), session_service=..., [memory/artifact])`.**
This sidesteps the deprecation entirely (no scoped-suppression needed for the plugin path — the
narrow `warnings` filter that `eval.py` uses is reserved for ADK-internal `Runner(plugins=)` calls
we don't control).

**PROVEN offline**: a `RecPlugin(BasePlugin)` overriding `on_event_callback` recorded 3 events
when running a `ScriptedLlm` agent through `Runner(app=App(plugins=[RecPlugin(...)]))` — plugin
wiring works end-to-end with no key.

### Project plugins manifest

Generated `<app_dir>/<app>/plugins.py` declares plugin instances at module level (e.g.
`logging_plugin = LoggingPlugin(name="logging")`). A manifest in the **runtime sidecar**
(`runtime.json`, new top-level key `"plugins": [{"var": "logging_plugin", "name": ...,
"kind": ...}]`) lists the module-level VARIABLE names so `build_runner` knows which symbols to
import from `plugins.py`. Backward-compatible: a `runtime.json` with no `plugins` key → no
plugins → unchanged Runner. `build_runner` imports `<app_dir>/<app>/plugins.py` with the same
fresh-`compile()`/`exec()` trick `import_root_agent` uses (defeats Windows bytecode staleness),
then collects the listed variables.

## Observability — OpenTelemetry

ADK's telemetry uses **standard OpenTelemetry**. `from google.adk.telemetry import tracer` is an
`opentelemetry.trace.ProxyTracer` bound to the global provider; module exports:
`node_tracing, trace_call_llm, trace_merged_tool_calls, trace_send_data, trace_tool_call, tracer,
tracing`. ADK creates spans on the GLOBAL tracer provider — so **a user enables a custom exporter
by installing a `TracerProvider` with a span processor as the global provider** (standard OTel
setup), BEFORE the agent runs. No ADK-specific exporter API is needed.

### What's installed (core google-adk dep) vs. what isn't

- ✅ **CORE** (present in the base env): `opentelemetry.sdk.trace.TracerProvider`,
  `opentelemetry.sdk.trace.export.{BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter}`,
  `opentelemetry.sdk.resources.{Resource, SERVICE_NAME}` (`SERVICE_NAME == "service.name"`),
  `opentelemetry.trace.set_tracer_provider`.
- ❌ **NOT installed**: the OTLP exporter `opentelemetry.exporter.otlp.proto.{grpc,http}.trace_exporter`
  (`ModuleNotFoundError: No module named 'opentelemetry.exporter'`). It ships in the separate
  `opentelemetry-exporter-otlp` PyPI package. So generated OTLP setup must import it LAZILY and the
  generated module documents `pip install opentelemetry-exporter-otlp`. The toolkit never imports
  it itself (codegen-only) — `enable_otel` only emits AST-valid source.

### Generated `otel_setup.py`

`observability_enable_otel` writes `<app_dir>/<app>/otel_setup.py` defining a `setup_otel()` that:
console exporter → `ConsoleSpanExporter` (core, always works); OTLP → lazy
`from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter` guarded by a
clear `ImportError` message. Builds a `TracerProvider(resource=Resource.create({SERVICE_NAME: ...}))`,
adds a `BatchSpanProcessor(exporter)`, and calls `trace.set_tracer_provider(provider)`. AST-valid
(ast.parse), ruff-format + isort clean (rendered via the same `_Call` machinery / import merge).

### CLI flags (re-confirmed on this 2.1.0 — matches P4a deploy notes)

`--trace_to_cloud` (Cloud Trace) AND `--otel_to_cloud` (write OTel data to Cloud Trace + Logging)
are BOTH present on: `deploy cloud_run`, `deploy agent_engine`, `deploy gke`, `web`, `api_server`.
NEITHER is on `adk run`. So `observability_cloud_trace(target)` returns `--trace_to_cloud` and
references the existing `deploy_*`/`dev_*` tool that owns that target (no flag is emitted here —
the deploy/dev domains already validate every flag against the real `--help`).

## Overlap honesty (no duplication)

- `safety_settings(gemini_safety=...)` routes through the EXISTING
  `project_model.GenerateContentConfigSpec` + the models-domain rendering of
  `types.GenerateContentConfig(safety_settings=[types.SafetySetting(...)])`. It does NOT re-render
  safety settings — it reuses `_resolve_agent`-style logic and `add_or_update_agent` +
  `regenerate`, exactly like `models_generate_config`. `max_llm_calls` is stored on the agent spec
  and surfaced (it maps to `RunConfig.max_llm_calls`, validated by `run_core.build_run_config`).
- `observability_cloud_trace` does NOT shell out — it returns the flag + points at `deploy_cloud_run`
  / `dev_web` (which already apply flags). `observability_trace_view` does NOT reimplement a server
  — it **delegates to the same `adk_cli` process registry that `dev_web` uses** (the ADK dev Web UI
  hosts the trace view). Gated behind an env flag for any real boot, exactly like `dev`'s tests.
