"""MCP workflow prompts (P6a) — step-by-step guides toward the right ``<domain>_*`` tools.

Five ``@mcp.prompt`` prompts registered by :func:`register_prompts`. Each returns an
**actionable workflow string** that cites the REAL tool names exposed by this server
(``project_create``, ``agents_create_llm``, ``run_agent``, …) in the order you call them.
These are deterministic *templates* (no I/O, no ADK import): an MCP client renders them via
``get_prompt(<name>, {args})`` to frame a task before calling the tools.

Every tool name cited is guaranteed to exist in the catalog (cf. the cross-check in
``tests/unit/test_prompts.py``). The prompts carry ``tags={"workflow"}`` (consistent with the
per-domain tagging of the tools, and filterable on the Code Mode side).
"""

from __future__ import annotations

from fastmcp import FastMCP

#: Tag shared by all workflow prompts (parity with the per-domain tagging of the tools).
_WORKFLOW_TAGS = {"workflow"}


def register_prompts(mcp: FastMCP) -> None:
    """Register the 5 workflow prompts on the root MCP server.

    Called by ``build_server`` before mounting the sub-servers. Each prompt is a pure function
    returning a ``str``; its tool name = the function name, its description = the first line of
    its docstring.
    """

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def scaffold_multi_agent(goal: str) -> str:
        """Step-by-step plan to scaffold an ADK multi-agent system for a given goal."""
        return f"""# Scaffold an ADK multi-agent system
Goal: {goal}

Follow these steps, calling the toolkit tools in order. Choose a `<path>` (parent folder) and
an `app_name` (Python identifier: letters/digits/underscore, not starting with a digit) that
are consistent, and reuse them on every call.

1. **Scaffold the app.** `project_create(path, app_name, model="gemini-2.5-flash",
   backend="ai_studio")` — writes `agent.py` + `__init__.py` + `.env`. (Backend `vertex` if you
   go through Vertex AI.) Then fill in the keys via `project_set_env` if needed.

2. **Create the child agents (workers).** One `agents_create_llm(path, app_name, name=...,
   model=..., instruction=..., description=...)` per subtask of "{goal}". Give each one a
   precise `instruction` and a `description` (useful if a parent must route to it). To assign a
   non-Gemini model to a child: `models_configure_litellm(path, app_name, agent_name,
   provider=..., model=...)` (e.g. provider `lm_studio`/`openai`/`anthropic`).

3. **Compose an orchestration agent.** Depending on the flow:
   - sequential (step-by-step pipeline): `agents_create_sequential(path, app_name, name=...,
     sub_agents=[...])`;
   - parallel (simultaneous fan-out): `agents_create_parallel(path, app_name, name=...,
     sub_agents=[...])`;
   - loop (repeat until a criterion): `agents_create_loop(path, app_name, name=...,
     sub_agents=[...], max_iterations=N)`.
   The `sub_agents` must already exist (create them in step 2 first). To (re)wire the children of
   an existing agent: `agents_compose(path, app_name, name, sub_agents=[...])`.
   (Agent-as-tool delegation tip: `agents_as_tool` / `tools_add_agent_tool`.)

4. **Designate the root.** `agents_set_root(path, app_name, name=<orchestrator>)` — this is the
   `root_agent` that runs. Check the tree with `agents_list(path, app_name)` /
   `agents_get(path, app_name, name)`.

5. **Tune the models if needed.** `models_set(path, app_name, agent_name, model=...)` for a
   Gemini model by string; `models_generate_config(...)` for temperature/safety.

6. **Run.** `run_agent(path, app_name, user_id, session_id, message)` runs the `root_agent` and
   returns the events + the final text. (`run_stream` for SSE progress.) Requires model
   credentials in `.env` (GOOGLE_API_KEY or Vertex creds).

Reminder: after each `agents_*`/`models_*`/`tools_*` step, `agent.py` is fully regenerated from
the sidecar — do not edit `agent.py` by hand."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def add_guardrail(agent: str, concern: str) -> str:
        """Decide between a callback (per-agent) and a plugin (global), then attach a guardrail."""
        return f"""# Add a guardrail to an ADK agent
Target agent: {agent}
Concern: {concern}

## 1. Choose the scope: callback (per-agent) vs plugin (global)
- **Per-agent callback** → `safety_add_callback`. Prefer this when the guardrail concerns ONLY
  the "{agent}" agent. Rendered as a real function attached via the actual ADK kwarg
  (`before_model_callback` / `before_tool_callback`). Returning non-`None` short-circuits the
  LLM or the tool.
- **Global plugin** → `safety_add_plugin`. Prefer this when the policy must apply to ALL agents
  /tools of the app (a `BasePlugin` subclass wired onto the `Runner` via `App`).

## 2. Call the tool
### Filtering user INPUT (the most common for "{concern}") — before_model callback
- Block keywords. Call `safety_add_callback(path, app_name, agent_name="{agent}",
  hook="before_model", policy=...)` with
  `policy = {{"kind": "block_keywords", "keywords": "word1,word2", "refusal": "Sorry."}}`.
- Limit input size: same call `safety_add_callback(..., hook="before_model", ...)`
  with `policy = {{"kind": "max_input_chars", "max_chars": "2000"}}`.

### Block a TOOL's use — before_tool callback (per-agent)
- `safety_add_callback(path, app_name, agent_name="{agent}", hook="before_tool", policy=...)`
  with `policy = {{"kind": "block_tool", "denylist": "delete_db", "message": "Forbidden."}}`.

### GLOBAL policy (all agents) — plugin
- Global tool denylist:
  `safety_add_plugin(path, app_name, name="tool_guard", kind="tool_denylist",
   config={{"denylist": "delete_db,drop_table"}})`
- Logging of all events:
  `safety_add_plugin(path, app_name, name="event_log", kind="logging")`

## 3. (Optional) Model safety settings + call cap
- `safety_settings(path, app_name, agent_name="{agent}", gemini_safety=[...])` with an item
  `{{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}}`
  (reuses the `generate_content_config` rendering).
- `safety_settings(path, app_name, agent_name="{agent}", max_llm_calls=20)` caps the LLM calls
  per run (actually applied by `run_agent`/`run_stream` when no explicit value is passed).

After attaching, `agent.py` (callbacks) or `plugins.py` (plugins) is regenerated; check via
`agents_get(path, app_name, name="{agent}")`."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def write_evalset(agent: str) -> str:
        """Plan to create an eval set, set offline criteria, then run the evaluation."""
        return f"""# Write and run an ADK eval set
Agent to evaluate: {agent} (the app's `root_agent`, or a sub-agent via `agent_name`)

OFFLINE metrics (no judge model, no key) — prefer them in CI:
- `tool_trajectory_avg_score`: STRUCTURAL comparison of expected vs actual tool calls.
- `response_match_score`: ROUGE-1 between the final response and `expected_response`.
(The "LLM-judge" metrics like `response_evaluation_score` require a judge model + creds → no
offline run possible; do not use them for a deterministic check.)

1. **Create the eval set.** `eval_create_set(path, app_name, name="smoke",
   cases=[{{"query": "...", "expected_response": "...",
            "expected_tool_use": [{{"name": "<tool>", "args": {{...}}}}]}}])`
   — writes `<app>/eval/smoke.evalset.json` (schema-compliant `EvalSet`). Set `expected_tool_use`
   only if you evaluate the tool trajectory; otherwise an empty list is enough.

2. **Set the criteria (thresholds).** `eval_set_criteria(path, app_name,
   tool_trajectory_avg_score=1.0, response_match_score=0.8)` — writes `eval/test_config.json`
   (thresholds in [0, 1]). Read automatically by `eval_run`.

3. **Run the evaluation.** `eval_run(path, app_name, ...)`, passing it the path of the
   `.evalset.json` file created in step 1 (the eval set path parameter) and `num_runs=1`. This
   imports the agent, runs the offline eval, persists a report and returns
   `passed` + the per-metric scores. WARNING: a NON-conformance to the thresholds is a NORMAL
   result (`ok=True, passed=False`), not an error. Real failures (eval set missing, agent
   requiring a key, `eval` extra missing) return `err`. To evaluate a sub-agent, pass
   `agent_name=...`.

4. **Re-read a report.** `eval_report(path, app_name, report_id=<id returned by eval_run>)`.

Prerequisite: the evaluation extra (`uv add 'adk-toolkit-mcp[eval]'`) for the ROUGE metrics."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def deploy_checklist(target: str) -> str:
        """Deployment checklist: preflight, target choice, command, flags and creds."""
        return f"""# ADK deployment checklist
Requested target: {target}  (expected: agent_engine | cloud_run | gke)

1. **Preflight (best-effort, never blocks).** `deploy_preflight(target="{target}")` — checks
   `gcloud`/`adk` on the PATH (and `kubectl` for gke). Fix the reported gaps before deploying.

2. **(Optional) Containerize.** `deploy_containerize(path, app_name)` writes a `Dockerfile`
   serving `adk api_server` on `$PORT` (useful for Cloud Run / GKE with a custom image).

3. **Build the command (default `execute=False` → returns the argv + a plan, runs nothing).**
   Choose by target:
   - **Agent Engine (Vertex AI)**: `deploy_agent_engine(path, app_name, project=..., region=...,
     requirements_file=..., execute=False)`. Real 2.1.0 flags: `--project`, `--region`,
     `--display_name` (the `app_name` maps to it), `--requirements_file`. WARNING: NO `--app_name`;
     `--staging_bucket` is DEPRECATED (no-op, not emitted).
   - **Cloud Run**: `deploy_cloud_run(path, app_name, project=..., region=..., service_name=...,
     with_ui=False, enable_cloud_trace=False, execute=False)`. `enable_cloud_trace=True` emits the
     real `--trace_to_cloud` flag (NOT `--enable_cloud_trace`).
   - **GKE**: `deploy_gke(path, app_name, project=..., region=..., cluster=..., service_name=...,
     execute=False)`. The `cluster` parameter maps to `--cluster_name` (NOT `--cluster`).

4. **Check the plan, then execute.** Re-read `argv`/`plan`. The real deployment = re-calling the
   SAME tool with `execute=True` (requires GCP credentials: `gcloud auth login` +
   `gcloud config set project`, and for Vertex a valid project/region). Each emitted flag is
   validated against the real `adk <sub> --help` — an unknown flag returns `err`.

5. **Status.** `deploy_status(target="{target}", project=..., region=..., service_name=...,
   cluster=...)` queries Cloud Run (gcloud) / GKE (kubectl); Agent Engine returns guidance
   (no dedicated status CLI).

Creds reminders: NEVER hardcode a secret; use `.env`/environment variables. The Web UI
(`--with_ui`) is for dev/test, not production."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def debug_agent(symptom: str) -> str:
        """Troubleshooting route for an ADK agent: inspect the events + known pitfalls."""
        return f"""# Debug an ADK agent
Symptom: {symptom}

## 1. Reproduce and inspect the events
- Re-run the agent: `run_agent(path, app_name, user_id, session_id, message)` — returns the list
  of serialized events + `final_text`.
- Pass those events to `run_inspect_events(events=<returned events>)`: a PURE tool that summarizes
  the `function_calls`, the tools actually used (`tool_names`), the agent transfers
  (`transfers`), the `state_delta` keys and the final text. This is the starting point to see
  WHAT the agent did.
- To follow progress live (where it gets stuck): `run_stream(path, app_name, user_id,
  session_id, message)` (reports each event via the MCP context).

## 2. Check the agent structure
- `agents_list(path, app_name)`: is the root the expected one (`agents_set_root`)?
- `agents_get(path, app_name, name)`: is the spec (model, instruction, sub_agents) correct?
- `tools_list(path, app_name, agent_name)`: are the expected tools actually attached?

## 3. Known pitfalls (mapped to the symptom)
- **"no response / API key"** → missing credentials: fill in `.env`
  (GOOGLE_API_KEY for AI Studio, or GOOGLE_GENAI_USE_VERTEXAI=TRUE + GOOGLE_CLOUD_PROJECT for
  Vertex) via `project_set_env`. `run_live` ALSO requires a live-capable Gemini model.
- **"the agent loops / too many LLM calls"** → cap via `safety_settings(..., max_llm_calls=N)`
  (applied by `run_agent`), or pass `max_llm_calls` directly to `run_agent`.
- **"a tool is never called / wrong trajectory"** → check `tools_list` and the agent's
  `instruction` (`agents_get`), then formalize the expectation with `eval_create_set` +
  `eval_run` (`tool_trajectory_avg_score`).
- **"my edit to `agent.py` disappeared"** → NORMAL: `agent.py` is regenerated from the sidecar on
  every `agents_*`/`tools_*`/`models_*` mutation. Edit via the tools, not the file.
- **"state not persistent"** → `temp:` state is NOT persisted across `get_session` (ADK design);
  inspect via `sessions_state_get` / `sessions_get`. For a DB session, the URL must be async
  (`sqlite+aiosqlite:///...`).
- **unexpected guardrail** → a `safety_add_callback`/`safety_add_plugin` may be short-circuiting
  the LLM/the tool; check via `agents_get` (callbacks) and `plugins.py`.

## 4. Trace more finely (optional)
`observability_enable_otel(path, app_name, exporter="console")` generates `otel_setup.py`; or
`observability_trace_view(path, app_name)` launches ADK's Web UI ("Trace" tab)."""
