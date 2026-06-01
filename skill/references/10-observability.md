# 10 — Observability (the `observability` domain)

ADK telemetry is **standard OpenTelemetry**. ADK creates spans on the **global** tracer provider, so
you enable tracing by installing a `TracerProvider` (with a span processor + exporter) as the global
provider **before the agent runs**. No ADK-specific exporter API exists. Maps to `observability_*`.

## The four tools

| Tool | Key args | Purpose |
|---|---|---|
| `observability_enable_otel` | `exporter="console", endpoint=None` | Generate `<app_dir>/<app>/otel_setup.py` with a `setup_otel()` that installs a global `TracerProvider`. |
| `observability_cloud_trace` | `target` | Tell you the real CLI flag + which deploy/dev tool applies it (does **not** execute). |
| `observability_third_party` | `provider, endpoint=None, headers=None` | Emit OTLP env vars + a setup snippet for a third-party backend. |
| `observability_trace_view` | `app_name=None, port=8000` | Launch the `adk web` UI (which hosts the Trace view) by **delegating to `dev_web`**. |

## `observability_enable_otel` — generate the OTel setup
```
observability_enable_otel(path, app_name, exporter="console", endpoint=None)
```
Writes `otel_setup.py` defining `setup_otel()` that builds a `TracerProvider(resource=Resource.create(
{SERVICE_NAME: <app>}))`, adds a `BatchSpanProcessor(exporter)`, and calls
`trace.set_tracer_provider(provider)`. Call `setup_otel()` at startup (before running the agent).

- **`exporter="console"`** → `ConsoleSpanExporter` — base OTel SDK, **always available** (core
  google-adk dep). Spans print to stdout. Great for dev.
- **`exporter="otlp"`** → `OTLPSpanExporter` (HTTP), imported **lazily**. Requires `endpoint` (e.g.
  `http://localhost:4318/v1/traces`) and the **separate** `opentelemetry-exporter-otlp` package (NOT a
  google-adk dep). The generated file documents `pip install opentelemetry-exporter-otlp`; the toolkit
  never imports it (codegen-only).

Generated code is `ast.parse` + ruff/isort clean. Returns `{otel_setup, usage, exporter, endpoint, notes}`.

## `observability_cloud_trace` — the Cloud Trace flag (no execution)
```
observability_cloud_trace(target)   # target ∈ {cloud_run, agent_engine, gke, web, api_server}
```
Returns the real flag and points at the tool that applies it — it **does not run anything** (avoids
duplicating deploy logic):
- **`flag` = `--trace_to_cloud`** — the flag the toolkit applies (via `deploy_cloud_run(...
  enable_cloud_trace=True)` / the deploy/dev tool for that target). `--trace_to_cloud` and
  `--otel_to_cloud` are on `deploy cloud_run`/`agent_engine`/`gke` + `web`/`api_server` (NOT `adk run`).
- **`otel_flag` = `--otel_to_cloud`** — ADK manual-only; **no toolkit tool auto-applies it** (pass it to
  `adk` yourself). The toolkit is honest that it doesn't emit this one.

## `observability_third_party` — Phoenix / Arize / Weave / SigNoz / OTLP
```
observability_third_party(provider, endpoint=None, headers=None)
```
`provider` ∈ {`phoenix`, `arize`, `weave`, `signoz`, `otlp`}. All ingest standard OTLP, so the tool
returns the canonical OTel env vars (`OTEL_EXPORTER_OTLP_ENDPOINT` + `OTEL_EXPORTER_OTLP_HEADERS` if
`headers`) and a snippet pointing at `observability_enable_otel(exporter='otlp', endpoint=...)`. A
default `endpoint` is supplied where the backend has one (phoenix/arize/signoz); `weave`/`otlp` require
`endpoint`. **Secrets via env** — `headers` (e.g. an API key) are emitted as env values to set (the
snippet reads `os.environ`), never hardcoded.

## `observability_trace_view` — open the Trace UI
```
observability_trace_view(path, app_name=None, port=8000)   # async
```
The `adk web` dev UI has a **Trace** tab visualizing a run's spans. Rather than reimplement a server,
this **delegates to `dev_web`** (same process registry). Returns `{key, pid, port, url, trace_url, ...}`;
drive it with `dev_status`/`dev_logs`/`dev_stop` (same keys as `dev`). Real boot only if a valid agents
dir exists.

## Typical flows

- **Local trace inspection:** `observability_trace_view(path, app_name)` → open `trace_url` → run the
  agent (`run_agent` or via the UI) → see spans.
- **Console spans in code:** `observability_enable_otel(path, app_name, exporter="console")` → call
  `setup_otel()` at startup.
- **Ship to a backend:** `observability_third_party("phoenix")` → set the env vars →
  `observability_enable_otel(exporter="otlp", endpoint=...)`.
- **Cloud Trace on deploy:** `observability_cloud_trace("cloud_run")` → it says use
  `deploy_cloud_run(... enable_cloud_trace=True)` (→ `--trace_to_cloud`).
