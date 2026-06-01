# ADK API notes — `eval` (P3b evaluation)

Captured 2026-06-01 by introspection. `google-adk` **2.1.0**, `fastmcp` **3.3.1**, Python 3.12.
The `eval` extra **is required** for the offline metrics (`rouge-score` / `pandas` / `tabulate`
ship with it). Core eval *models* (`EvalSet`/`EvalCase`/`Invocation`) import from base
`google-adk`, but `ResponseEvaluator`/`final_response_match_v1` need `rouge_score`, and
`AgentEvaluator._get_eval_results_by_eval_id` needs `pandas`+`tabulate` — all absent in core,
present in `eval`. See "Extra" below.

These notes back the `eval` domain sub-server. Unlike P1 (which *writes* `agent.py`), `eval`
runs a **real ADK evaluation** of a project's agent against an eval set. The load-bearing proof
is that `tool_trajectory_avg_score` + `response_match_score` pass **fully offline (no API key)**
against a `FakeLlm`-backed agent.

## `google.adk.evaluation` module surface

`dir(google.adk.evaluation)` (public): `AgentEvaluator`, `agent_evaluator`, `app_details`,
`common`, `constants`, `conversation_scenarios`, `eval_case`, `eval_config`, `eval_metrics`,
`eval_result`, `eval_rubrics`, `eval_set`, `eval_sets_manager`, `evaluator`,
`in_memory_eval_sets_manager`, `local_eval_service`, `local_eval_sets_manager`, `logger`,
`simulation`.

## Schema (the REAL pydantic models)

`*.evalset.json` is a serialized **`EvalSet`** (`google.adk.evaluation.eval_set`). Confirmed
fields (`*` = required):

```text
EvalSet(*eval_set_id: str, name: Optional[str], description: Optional[str],
        *eval_cases: list[EvalCase], creation_timestamp: float = 0.0)

EvalCase(*eval_id: str, conversation: Optional[list[Invocation]],
         conversation_scenario: Optional[ConversationScenario],
         session_input: Optional[SessionInput], creation_timestamp: float,
         rubrics: Optional[list[Rubric]], final_session_state: Optional[dict[str, Any]])

# google.adk.evaluation.eval_case
Invocation(invocation_id: str = <uuid>, *user_content: genai.types.Content,
           final_response: Optional[genai.types.Content],
           intermediate_data: Optional[IntermediateData | InvocationEvents],
           creation_timestamp: float, rubrics: Optional[list[Rubric]],
           app_details: Optional[AppDetails])

IntermediateData(tool_uses: list[genai.types.FunctionCall] = [],
                 tool_responses: list[genai.types.FunctionResponse] = [],
                 intermediate_responses: list[tuple[str, list[genai.types.Part]]] = [])

SessionInput(*app_name: str, *user_id: str, state: dict[str, Any] = {})
```

- A **query** is `Invocation.user_content` = `types.Content(role="user", parts=[Part.from_text])`.
- An **expected response** is `Invocation.final_response` (same `Content` shape, `role="model"`).
- An **expected tool trajectory** is `IntermediateData.tool_uses` = a list of
  `types.FunctionCall(name=..., args={...})`.
- Build with the real models and serialize via `eval_set.model_dump_json(indent=2,
  exclude_none=True)`. It round-trips: `EvalSet.model_validate_json(file_text)` succeeds — this
  is exactly what the test asserts (schema conformance, not a guess).

### `*.test.json` (older format) — still supported, auto-detected

`AgentEvaluator._load_eval_set_from_file` first tries `EvalSet.model_validate_json(content)`
(new schema). On `ValidationError` it logs a warning and falls back to the **old** format
(`_get_eval_set_from_old_format`) whose columns are `query` / `reference` / `expected_tool_use`
(`QUERY_COLUMN`/`REFERENCE_COLUMN`/`EXPECTED_TOOL_USE_COLUMN`). The toolkit emits the **new**
`EvalSet` schema (`*.evalset.json`) and does not write the old format. `migrate_eval_data_to_new_schema`
exists for converting old files.

## `test_config.json` — a serialized `EvalConfig`

`AgentEvaluator.find_config_for_test_file(test_file)` looks for `test_config.json` **in the same
folder** as the eval file and parses it with `EvalConfig.model_validate_json`. The classic flat
form is accepted and round-trips:

```json
{"criteria": {"tool_trajectory_avg_score": 1.0, "response_match_score": 0.8}}
```

```text
EvalConfig(criteria: dict[str, float | BaseCriterion] = {},
           custom_metrics: Optional[dict[str, CustomMetricConfig]],
           user_simulator_config: Optional[BaseUserSimulatorConfig])
BaseCriterion(*threshold: float, include_intermediate_responses_in_final: bool = False)
```

A bare `float` value is auto-wrapped into `BaseCriterion(threshold=float)` by
`get_eval_metrics_from_config`. The toolkit writes the **flat float** form (human-friendly, and
what classic `adk eval` `test_config.json` uses). If `test_config.json` is absent, ADK uses a
default criteria set.

## Metric keys (`PrebuiltMetrics` enum) and which run OFFLINE

```text
TOOL_TRAJECTORY_AVG_SCORE = 'tool_trajectory_avg_score'   # OFFLINE (structural)
RESPONSE_MATCH_SCORE      = 'response_match_score'        # OFFLINE (ROUGE-1; needs rouge_score)
RESPONSE_EVALUATION_SCORE = 'response_evaluation_score'   # LLM JUDGE — needs a model/creds
SAFETY_V1, FINAL_RESPONSE_MATCH_V2, HALLUCINATIONS_V1, RUBRIC_BASED_*_V1,
PER_TURN_USER_SIMULATOR_QUALITY_V1, MULTI_TURN_*_V1        # judge/simulator — need a model
```

- **`tool_trajectory_avg_score`** → `TrajectoryEvaluator` does a **pure structural** comparison
  of `intermediate_data.tool_uses` (exact / in-order / any-order match → 1.0 or 0.0 per
  invocation, averaged). No model, no rouge. Always offline.
- **`response_match_score`** → `ResponseEvaluator` / `final_response_match_v1.RougeEvaluator`
  computes **ROUGE-1** between the agent's actual `final_response` text and the expected
  `final_response` text. Needs the `rouge_score` package (in the `eval` extra) but **no model /
  no API key**. Offline.
- LLM-judge metrics (`response_evaluation_score`, `*_v1/_v2`, safety, hallucinations) require a
  judge model + credentials → NOT offline. The toolkit's offline path uses only the first two.

## `AgentEvaluator.evaluate` / `evaluate_eval_set` — async, assert-based

```text
@staticmethod
async def evaluate(agent_module: str, eval_dataset_file_path_or_dir: str, num_runs: int = 2,
                   agent_name: Optional[str] = None, initial_session_file: Optional[str] = None,
                   print_detailed_results: bool = True)

@staticmethod
async def evaluate_eval_set(agent_module: str, eval_set: EvalSet,
                            criteria: Optional[dict[str, float]] = None,        # DEPRECATED
                            eval_config: Optional[EvalConfig] = None,
                            num_runs: int = 2, agent_name: Optional[str] = None,
                            print_detailed_results: bool = True)
```

- **Both are `async`** (`inspect.iscoroutinefunction` → True). Await them.
- **Verdict is assert-based:** on success they return `None`; on **failure** they raise
  `AssertionError` whose message lists `"<metric> for <module> Failed. Expected <threshold>, but
  got <score>."` per failing metric. The toolkit catches `AssertionError` → `ok({passed: False,
  ...})` (an eval *failure* is a normal result, NOT a tool error). Other exceptions (missing
  model creds, import errors) → `err(...)`.
- `criteria` (flat dict) still works but is **deprecated** (a `logger.warning`, NOT a
  `DeprecationWarning` — does not trip `-W error::DeprecationWarning`); it is auto-mapped to an
  `EvalConfig`. The toolkit builds an `EvalConfig` and passes `eval_config=` (the non-deprecated
  path). If neither is given, `evaluate_eval_set` raises `ValueError("`eval_config` is required.")`.
- `evaluate(path_or_dir)`: if a **directory**, walks it for `*.test.json` files; if a **single
  file path**, uses it directly (no suffix requirement) — so a `*.evalset.json` file path is
  accepted directly, and `test_config.json` in the same folder is auto-picked up.
- `num_runs` defaults to **2**; the toolkit exposes it (default 1 for speed/determinism). With
  `num_runs>1` the per-invocation scores are averaged before thresholding.

## How the agent is loaded — `_get_agent_for_eval` (dotted module, importlib)

```text
@staticmethod
async def _get_agent_for_eval(module_name: str, agent_name: Optional[str] = None) -> BaseAgent:
    agent_module = importlib.import_module(module_name)        # DOTTED module path, NOT a file
    # requires hasattr(module, "agent") OR module_name endswith ".agent"
    holder = agent_module.agent if hasattr(agent_module, "agent") else agent_module
    root_agent = holder.root_agent  (or await holder.get_agent_async())
    if agent_name: root_agent = root_agent.find_agent(agent_name)
```

- `agent_module` is a **dotted importable module path**, resolved via `importlib.import_module`
  (NOT a filesystem path). The module must either expose a member named `agent` OR its dotted
  name must end with `.agent`; then `root_agent` (or `get_agent_async`) is read.
- A toolkit-scaffolded app `<app_name>/` has `__init__.py` (`from . import agent`) + `agent.py`
  → it is an importable package. The `eval` domain inserts `path` onto `sys.path` and imports
  **`<app_name>.agent`** (ends with `.agent`, satisfies the check directly, and `agent.py`
  defines `root_agent`). To pick up edits, the domain evicts any cached `<app_name>` /
  `<app_name>.agent` from `sys.modules` before importing (importlib caches).
- `agent_name` (optional) evaluates a **sub-agent** via `root_agent.find_agent(name)`.

## Per-metric scores (richer report) — `_get_eval_results_by_eval_id`

For the persisted report the toolkit also collects per-metric scores (the public assert API
only yields pass/fail). It reuses the same machinery `evaluate_eval_set` does internally:

```text
agent      = await AgentEvaluator._get_agent_for_eval(module_name="<app>.agent")
metrics    = get_eval_metrics_from_config(EvalConfig(criteria={...: BaseCriterion(threshold=...)}))
results    = await AgentEvaluator._get_eval_results_by_eval_id(
                 agent_for_eval=agent, eval_set=eval_set, eval_metrics=metrics,
                 num_runs=n, user_simulator_provider=UserSimulatorProvider())
# results: dict[eval_id -> list[EvalCaseResult]]
EvalCaseResult.final_eval_status: EvalStatus          # PASSED=1 / FAILED=2 / NOT_EVALUATED=3
EvalCaseResult.overall_eval_metric_results: list[EvalMetricResult]
EvalMetricResult(metric_name: str, threshold: Optional[float], score: Optional[float],
                 eval_status: EvalStatus, ...)
```

Verdict logic mirrors ADK: per metric, `mean(scores) >= threshold → PASSED` else `FAILED`
(`NOT_EVALUATED` if no score). The overall run passes iff every case's `final_eval_status ==
PASSED`. `UserSimulatorProvider` / `MetricEvaluatorRegistry` emit `[EXPERIMENTAL]` **UserWarning**
(NOT `DeprecationWarning` — safe under `-W error::DeprecationWarning`).

## PROVEN offline (the load-bearing result)

With a `FakeLlm` / `ScriptedLlm(BaseLlm)` agent (no API key) and criteria limited to
`tool_trajectory_avg_score` + `response_match_score`:

- **Text case** (`FakeLlm` returns a fixed answer == the eval case's `final_response`):
  `response_match_score = 1.0` → **PASSED** offline.
- **Tool case** (`ScriptedLlm` emits one `add_numbers(a=2,b=3)` call then final text; eval case's
  `intermediate_data.tool_uses == [add_numbers(a=2,b=3)]` and `final_response == "The sum is 5."`):
  `tool_trajectory_avg_score = 1.0` AND `response_match_score = 1.0` → **PASSED** offline.
- **Negative control** (a deliberately wrong expected answer, threshold 0.9): correctly raises
  `AssertionError` → **FAILED**. Proves the pipeline genuinely evaluates and does not fake a pass.

## Missing-credential path (no hang, actionable `err`)

A real Gemini model with no key, or an LLM-judge metric, fails during inference/scoring (not an
`AssertionError`). `eval_run` wraps the whole run in `try/except`: an `AssertionError` is the
eval-failure verdict (`ok`, `passed=False`); any other exception (model creds, import,
`ModuleNotFoundError` for the eval extra) is converted to a clean `err(...)` with an actionable
message — it never hangs and never lets an exception escape.

## Report storage + resource choice

Reports persist to `<app_dir>/eval/reports/<report_id>.json` (`report_id` = timestamp-based
slug). Reads use a **`eval_report(path, app_name, report_id)` tool**, NOT an `adk://eval/{id}`
resource template: a report is addressed by `(path, app_name, report_id)` (three coordinates),
and FastMCP 3.3.1 resource templates key on a single opaque id with no way to pass `path`/
`app_name`, so a two-/three-param lookup is awkward and ambiguous as a resource. A tool keeps the
addressing explicit and consistent with every other `eval_*` tool (all take `path, app_name`).
The in-memory `fastmcp.Client` read-through test exercises `eval_report`.

## Extra (`eval`) — required; uv.lock already resolved

The `eval` extra (`google-adk[eval]`) brings `rouge-score`, `pandas`, `tabulate`, `nltk`,
`scikit-learn`, `jinja2`, `gepa`, `google-cloud-aiplatform[evaluation]`. It was added to the
`dev` extra so CI installs it (offline metrics are testable). `uv.lock` already contained these
(the user-facing `eval`/`all` extras were locked in P0) → **no `uv.lock` change**; only
`pyproject.toml` `dev` gains `adk-toolkit-mcp[eval]`. Heavy import (`AgentEvaluator` et al.) is
kept **lazy** inside the tool body; a `ModuleNotFoundError` (extra absent in a slim env) is
converted to an actionable `err` like the `gcp`/`db` extras.
