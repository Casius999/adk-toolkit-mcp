# ADK 2.1.0 CLI facts — `deploy` + `dev` (P4a)

Captured 2026-06-01 by direct introspection (`uv run adk ... --help`).
google-adk **2.1.0**, fastmcp 3.3.1, Python 3.12 (Windows). These flags **drift between
versions** — this file is the 2.1.0 truth that `domains/deploy.py` + `domains/dev.py` emit and
validate against (`adk_cli.available_flags`). Re-introspect on any ADK bump.

## How `adk` is invoked (`adk_cli.adk_executable`)

- The venv ships a real console script: `.venv/Scripts/adk.exe` (resolved via `shutil.which("adk")`
  on the venv `Scripts`/`bin` dir relative to `sys.executable`, then PATH). PREFERRED.
- Fallback: `[sys.executable, "-m", "google.adk.cli"]`. VERIFIED real module:
  `google.adk.cli.__main__` exists and `python -m google.adk.cli --version` → `..., version 2.1.0`.
- `run_adk(args)` runs `<adk> <args>` with `subprocess.run` (argv list, **no `shell=True`**),
  capturing rc/stdout/stderr. Used for `--help` introspection and (only when `execute=True`) for
  real deploys.

## `adk --help` — top-level commands

```
api_server  conformance  create  deploy  eval  eval_set  migrate  optimize  run  test  web
```

## `adk deploy --help` — subcommands

```
agent_engine   Deploys an agent to Agent Engine.
cloud_run      Deploys an agent to Cloud Run.
gke            Deploys an agent to GKE.
```

## `adk deploy agent_engine [OPTIONS] AGENT`

Positional: **`AGENT`** (path to agent source folder). Example shows
`adk deploy agent_engine --project=[p] --region=[r] --display_name=[name] my_agent`.

Click marks **nothing as strictly required** (project/region are `Optional` and fall back to
`.env`/gcloud/express-mode). Toolkit policy: require `path`; require either `project`+`region`
(Vertex) — we do NOT support express-mode `--api_key` (would mean passing a secret on argv).

Flags (2.1.0): `--api_key`, `--project`, `--region`, **`--staging_bucket` (DEPRECATED — "no
longer required or used")**, `--agent_engine_id`, `--trace_to_cloud / --no-trace_to_cloud`,
`--otel_to_cloud`, `--display_name` (default `""`), `--description` (default `""`), `--adk_app`,
`--temp_folder`, `--adk_app_object` (`root_agent`|`app`), `--env_file`, `--requirements_file`,
`--absolutize_imports` (DEPRECATED), `--agent_engine_config_file`,
`--validate-agent-import / --no-validate-agent-import`, `--skip-agent-import-validation`.

> ⚠️ Task spec named `--app_name` + `--staging_bucket` for agent_engine — **WRONG for 2.1.0**.
> agent_engine has NO `--app_name`; `--staging_bucket` exists but is deprecated/no-op. The
> toolkit `agent_engine(...)` accepts an `app_name` param and maps it to **`--display_name`**
> (the closest real concept) and accepts `staging_bucket` only to emit a deprecation note
> (NOT passed as a flag). `requirements_file` → `--requirements_file` (real).

## `adk deploy cloud_run [OPTIONS] AGENT`

Positional: **`AGENT`** (agent source folder). `--` separates trailing gcloud args.

Flags (2.1.0): `--project` (Required*), `--region` (Required*), `--service_name`
(default `adk-default-service-name`), `--app_name` (default = folder name), `--port`
(default 8000), **`--trace_to_cloud`** (NOT `--enable_cloud_trace`), `--otel_to_cloud`,
`--with_ui`, `--temp_folder`, `--log_level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`, `--adk_version`
(default 2.1.0), `--a2a`, `--trigger_sources`, `--allow_origins`, `--session_service_uri`,
`--artifact_service_uri`, `--use_local_storage / --no_use_local_storage`, `--memory_service_uri`.

\* "Required" in help text but Click does not enforce (absent → gcloud default project / prompt).
Toolkit requires `path`, `project`, `region` for a deterministic plan.

> ⚠️ Task spec named `enable_cloud_trace` — real flag is **`--trace_to_cloud`**. Toolkit param
> `enable_cloud_trace` maps to `--trace_to_cloud`.

## `adk deploy gke [OPTIONS] AGENT`

Positional: **`AGENT`**. Example:
`adk deploy gke --project=[p] --region=[r] --cluster_name=[c] path/to/my_agent`.

Flags (2.1.0): `--project` (Required*), `--region` (Required*), **`--cluster_name`** (Required —
NOT `--cluster`), `--service_name` (default `adk-default-service-name`), `--app_name`
(default = folder name), `--port` (default 8000), `--trace_to_cloud`, `--otel_to_cloud`,
`--with_ui`, `--log_level [...]`, `--service_type [ClusterIP|LoadBalancer]` (default ClusterIP),
`--temp_folder`, `--adk_version` (default 2.1.0), `--trigger_sources`, `--session_service_uri`,
`--artifact_service_uri`, `--use_local_storage / --no_use_local_storage`, `--memory_service_uri`.

> ⚠️ Task spec named `cluster` — real flag is **`--cluster_name`**. Toolkit param `cluster` maps
> to `--cluster_name`. gke requires `path`, `project`, `region`, `cluster`.

## `adk web [OPTIONS] [AGENTS_DIR]`

Positional: **`AGENTS_DIR`** (dir of agents OR a single agent folder). Starts FastAPI + Web UI.
Relevant flags: `--host` (default 127.0.0.1), **`--port`** (INTEGER, NO default → ADK picks/binds),
`--log_level`, `--reload / --no-reload`, `--a2a`, `--reload_agents`, `--allow_origins`,
`--session_service_uri`, `--artifact_service_uri`, `--use_local_storage` (default ON for web),
`--memory_service_uri`, `--default_llm_model`, `-v/--verbose`, `--trace_to_cloud`,
`--otel_to_cloud`, `--url_prefix`, `--trigger_sources`, `--logo-text`, `--logo-image-url`,
`--enable_features`, `--disable_features`, `--extra_plugins`, `--eval_storage_uri`.

> There is **no `--app_name`** for web/api_server — you point AGENTS_DIR at the agents dir (or a
> single agent folder). The dev domain accepts an optional `app_name`: if given, it points
> AGENTS_DIR at `<path>/<app_name>` (single-agent folder); else at `<path>` (a dir of agents).
> `--host` and `--port` are real flags. A managed long-running process is the right model
> (`adk web` blocks serving) → process registry, NOT `run_adk` (which waits for exit).

## `adk api_server [OPTIONS] [AGENTS_DIR]`

Positional: **`AGENTS_DIR`** (dir of agents OR single agent folder). Starts FastAPI (no UI).
Same core flags as `web` PLUS: `--auto_create_session` (create a session on `/run` if missing),
`--with_ui` (serve UI). Lacks web-only `--reload`, `--logo-*`, `--default_llm_model`,
`--eval_storage_uri`. `--host` default 127.0.0.1; `--port` INTEGER no default.
The HTTP health/probe surface: a FastAPI app — `GET /docs` (Swagger) returns 200 once up
(used by the functional dev test as a readiness probe; `/list-apps` also exists).

## `adk run [OPTIONS] AGENT [QUERY]`

Positional: **`AGENT`** (agent source folder) + **optional `QUERY`** (a single user message →
single-step non-interactive run). With no QUERY it enters **interactive** mode (would block).

So **non-interactive one-shot IS supported**: `adk run <agent_dir> "<message>"`. The dev domain's
`run(path, app_name, message)` runs `adk run <agent_folder> <message>` via `run_adk` with a short
timeout when `message` is provided (capturing stdout/stderr); with no `message` it returns
guidance (interactive mode would block — not scriptable). Useful structured flag: **`--jsonl`**
(structured JSONL output instead of human text); `--timeout 30s` bounds a single turn;
`--in_memory` avoids writing local `.adk` storage. Other flags: `--session_service_uri`,
`--artifact_service_uri`, `--memory_service_uri`, `--use_local_storage`, `--save_session`,
`--session_id`, `--replay FILE`, `--resume FILE`, `--state`, `--default_llm_model`,
`--enable_features`, `--disable_features`.

> ⚠️ Task spec hypothesised an `--input` flag — there is **none**. The message is a **positional
> QUERY**, not a flag. `adk run` still needs model creds to actually produce a response, so the
> dev `run` tool runs the real command but a no-creds environment yields a non-zero rc / error
> text in the captured output (returned as data, not a hang — short timeout enforced).

## `available_flags` parsing

`adk <subcommand> --help` text lists each option starting at column 2 as `--flag`/`-x, --flag`,
sometimes `--flag / --no-flag`. `available_flags` runs the help and regex-extracts every
`--[a-z0-9][a-z0-9_-]*` token (both sides of a `/`-pair). Confirmed non-empty for
`deploy cloud_run` incl. `--project`, `--region`, `--service_name`, `--app_name`, `--with_ui`,
`--trace_to_cloud`; for `deploy gke` incl. `--cluster_name`; for `web`/`api_server` incl.
`--host`, `--port`.
