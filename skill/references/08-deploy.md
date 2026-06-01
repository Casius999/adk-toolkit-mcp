# 08 — Deploy & dev servers (the `deploy` and `dev` domains)

Ship the agent and run dev servers. Maps to `deploy_*` (build/run `adk deploy …`) and `dev_*` (managed
`adk web`/`api_server`/`run`). The toolkit shells to the real `adk` CLI and **validates every emitted
flag against the installed ADK's `--help`** — it cannot emit a flag this ADK lacks.

## Deploy target choice

```
Where to deploy?
├── Vertex AI Agent Engine — Google's managed agent runtime (serverless, Vertex-native)
│     → deploy_agent_engine   (needs project+region; Vertex backend)
├── Cloud Run — serverless containers, simplest general-purpose hosting, optional UI
│     → deploy_cloud_run      (needs project+region; --with_ui for the web UI)
├── GKE — Kubernetes, when you need cluster control / existing GKE infra
│     → deploy_gke            (needs project+region+cluster)
└── Just a container image — build your own / deploy anywhere
      → deploy_containerize   (writes a Dockerfile serving `adk api_server` on $PORT)
```

## `execute` flag (plan vs run) — important

`deploy_agent_engine` / `deploy_cloud_run` / `deploy_gke` take **`execute: bool = False`**:
- **`execute=False` (default)** — returns `{argv, plan, notes, executed: False}` and **never** runs the
  deploy. Use this to review the exact command first. (This is the safe default; real deploys touch GCP.)
- **`execute=True`** — runs the real `adk deploy …` (GCP credentials required).

The positional `AGENT` is always `<path>/<app_name>` (last). Required args are validated up front.

## Real 2.1.0 flags (these drift between versions — this is the verified truth)

| Tool | Required | Maps params → real flags | Notes |
|---|---|---|---|
| `deploy_agent_engine(path, app_name, project, region, staging_bucket=None, display_name=None, requirements_file=None, execute=False)` | `path, project, region` | `--project`, `--region`, `--display_name`, `--requirements_file` | **NO `--app_name`** for agent_engine. `app_name` maps to `--display_name` (a `display_name` arg wins). **`--staging_bucket` is DEPRECATED/no-op** — only noted, never emitted. |
| `deploy_cloud_run(path, app_name, project, region, service_name=None, with_ui=False, enable_cloud_trace=False, execute=False)` | `path, project, region` | `--project`, `--region`, `--service_name`, `--app_name`, `--with_ui`, **`--trace_to_cloud`** | `enable_cloud_trace` → **`--trace_to_cloud`** (NOT `--enable_cloud_trace`, which doesn't exist). `--with_ui` serves the web UI. |
| `deploy_gke(path, app_name, project, region, cluster, service_name=None, execute=False)` | `path, project, region, cluster` | `--project`, `--region`, **`--cluster_name`**, `--service_name`, `--app_name` | `cluster` → **`--cluster_name`** (NOT `--cluster`). |
| `deploy_containerize(path, app_name)` | `path, app_name` | — | Writes `<path>/Dockerfile` serving `adk api_server` on `$PORT` (Cloud Run injects `PORT`=8080). Idempotent. |

> If the toolkit ever detects an emitted flag the installed ADK doesn't have (version drift), it returns
> a clean `err` listing the unknowns instead of running a broken command.

## Preflight & status (best-effort, never hang)

- `deploy_preflight(target="cloud_run")` — checks `gcloud`/`adk`/`kubectl` on PATH and gives
  target-specific findings. Always `ok` (diagnostic, not a gate). Run it before deploying.
- `deploy_status(target, project=None, region=None, service_name=None, cluster=None)` — best-effort
  status (short timeout): `cloud_run` shells to `gcloud run services describe`; `gke` to
  `kubectl get service`; `agent_engine` returns guidance. Missing tool → `available: False` + guidance.

## Credentials needed

- **Cloud Run / GKE / Agent Engine** need `gcloud` authenticated + a GCP project. Agent Engine implies
  the **Vertex** backend (`project_create(... backend="vertex")` or `project_set_env`). Set
  `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` in `.env`.
- The toolkit does **not** support agent_engine express-mode `--api_key` (would mean a secret on argv).

## The `dev` domain — local CLI servers (the `dev_*` tools)

`adk web`/`api_server` **block while serving**, so the toolkit runs them as **managed background
processes** via a process registry (start detached, log to a file, stop the whole tree). `adk run` is
one-shot.

| Tool | Key args | Notes |
|---|---|---|
| `dev_web` | `app_name=None, port=8000, host="127.0.0.1"` | Start `adk web` (dev UI + API + Eval/Trace tabs) as a managed process. Returns `{key, pid, port, url, ...}`. **Dev/test only, not production.** |
| `dev_api_server` | `app_name=None, port=8000, host="127.0.0.1"` | Start `adk api_server` (FastAPI, no UI). OpenAPI at `<url>/docs`. |
| `dev_run` | `app_name, message=None` | One-shot `adk run <agent_dir> "<message>"`. **The message is a positional QUERY, not a flag** (there is no `--input`). No message → guidance (interactive mode would block). Bounded timeout. |
| `dev_stop` | `key` | Stop a managed server by key (terminates the process tree; Windows `taskkill /T`). Idempotent. |
| `dev_status` | `key` | `{found, running, pid, returncode, log_path, argv}`. |
| `dev_logs` | `key, tail=50` | Last `tail` lines of the server's log. |

### `app_name` for web/api_server

There is **no `--app_name`** for `web`/`api_server` — you point the positional `AGENTS_DIR`. If
`app_name` is given, the toolkit serves `<path>/<app_name>` (a single agent folder); else `<path>` (a
directory of agents, each subfolder an agent). `--host`/`--port` are real flags.

## Typical flow

1. `deploy_preflight(target="cloud_run")` — sanity-check tooling.
2. `deploy_cloud_run(path, app_name, project, region, with_ui=True)` (default `execute=False`) — review
   the `argv`/`plan`.
3. Re-call with `execute=True` to deploy for real.
4. Or locally: `dev_web(path, app_name)` → open the `url`; `dev_stop(key)` when done.

## Cloud Trace

To enable Cloud Trace on a deploy, pass `enable_cloud_trace=True` to `deploy_cloud_run` (→
`--trace_to_cloud`). See `10-observability.md` (`observability_cloud_trace` tells you which flag/tool).
