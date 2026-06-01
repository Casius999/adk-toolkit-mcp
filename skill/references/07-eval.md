# 07 — Evaluation (the `eval` domain)

Run a **real ADK evaluation** of the project's agent against an evalset. Maps to the `eval_*` tools.
Files live under `<app_dir>/eval/` (`<name>.evalset.json`, `test_config.json`, `reports/<id>.json`).
**Requires the `eval` extra** for the offline metrics (`rouge-score`/`pandas`/`tabulate`).

## The evaluation discipline

1. **Create an evalset** — a set of cases, each a `{query, expected_response, expected_tool_use?}`.
2. **Set criteria** — thresholds for the metrics.
3. **Run** — import the agent, evaluate each case, capture verdict + per-metric scores, persist a report.
4. **Read the report** — by `(path, app_name, report_id)`.

An eval **failure is a normal result**: `ok=True, passed=False` (the agent didn't meet thresholds). A
clean `err` is reserved for real errors (missing evalset, import failure, model needing creds, LLM-judge
metric, `eval` extra absent) — never a hang, never an exception.

## Metrics: offline vs LLM-judge

| Metric | What it measures | Offline? |
|---|---|---|
| `tool_trajectory_avg_score` | **Structural** compare of the agent's tool calls vs `expected_tool_use` (in-order/any-order → 1.0 or 0.0 per case, averaged). No model, no ROUGE. | ✅ always |
| `response_match_score` | **ROUGE-1** overlap between the agent's final text and `expected_response`. Needs `rouge_score` (in the `eval` extra) but **no model**. | ✅ yes |
| `response_evaluation_score`, `*_v1/_v2`, safety, hallucinations | An **LLM judge** scores the response. | ❌ needs a judge model + creds |

The toolkit's offline path uses **only the first two** (the ones that need no API key). That's the
discipline: prove tool trajectories and response overlap deterministically; reach for LLM-judge metrics
only when you have a judge model.

## The `eval` domain tools

| Tool | Key args | Notes |
|---|---|---|
| `eval_create_set` | `name, cases: list[{query, expected_response, expected_tool_use?}]` | Writes a schema-conformant `<app_dir>/eval/<name>.evalset.json` (built from the real `EvalSet`/`EvalCase`/`Invocation` models). `expected_tool_use` is a list of `{name, args}`. |
| `eval_set_criteria` | `tool_trajectory_avg_score=1.0, response_match_score=0.8` | Writes `<app_dir>/eval/test_config.json` (an `EvalConfig`). Thresholds in `[0, 1]`. Auto-read by `eval_run` (and `adk eval`). |
| `eval_run` | `eval_set_file, config_file=None, num_runs=1, agent_name=None` | Imports `<app_name>.agent`, evaluates offline (the two metrics), captures verdict + scores, persists a report. `num_runs` averages repeated inference. `agent_name` evaluates a sub-agent. |
| `eval_report` | `report_id` | Re-read a stored report by `(path, app_name, report_id)`. |

## Case schema (what `eval_create_set` builds)

Each case in `cases`:
```json
{
  "query": "What is the capital of France?",
  "expected_response": "The capital of France is Paris.",
  "expected_tool_use": [{"name": "lookup_capital", "args": {"country": "France"}}]
}
```
- `query` → the user turn (`Invocation.user_content`).
- `expected_response` → the expected final text (`Invocation.final_response`).
- `expected_tool_use` (optional) → the expected tool trajectory (`IntermediateData.tool_uses`).

The written file round-trips `EvalSet.model_validate_json` (schema conformance is proven, not guessed).
The toolkit emits the **new** `*.evalset.json` schema (the legacy `*.test.json` `query`/`reference`/
`expected_tool_use` format is still auto-detected by ADK on read, but not written).

## Eval flow

1. `eval_create_set(path, app_name, name="smoke", cases=[...])` → writes the evalset file (note its path).
2. `eval_set_criteria(path, app_name, tool_trajectory_avg_score=1.0, response_match_score=0.8)`.
3. `eval_run(path, app_name, eval_set_file="<that path>")` → `{passed, cases, metrics, report_id}`.
4. `eval_report(path, app_name, report_id)` → the full persisted report.

## Notes & gotchas

- The agent is imported as the **dotted module** `<app_name>.agent` (a scaffolded app is a package:
  `__init__.py` + `agent.py`). `eval_run` inserts `path` on `sys.path` and evicts cached modules to pick
  up edits.
- The `eval` extra brings `rouge-score`, `pandas`, `tabulate`, `nltk`, `scikit-learn`, etc. A missing
  extra → actionable `err` (install `adk-toolkit-mcp[eval]`).
- A report is a tool (not an `adk://eval/{id}` resource) because it's addressed by three coordinates
  `(path, app_name, report_id)`.
- The **`adk web` UI** also has an Eval tab for interactive evaluation — start it via `dev_web` (see
  `08-deploy.md`).
