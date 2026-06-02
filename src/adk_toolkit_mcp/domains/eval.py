"""`eval` domain: EVALUATES an ADK agent via ``AgentEvaluator`` (P3b — evaluation).

Like the ``run`` domain (P3a), this domain **imports the ``root_agent``** of an app and runs a
REAL ADK evaluation against an *eval set*, but via the ``google.adk.evaluation`` pipeline rather
than a plain ``Runner``. The eval files live under ``<app_dir>/eval/`` (``<name>.evalset.json``,
``test_config.json``, ``reports/<id>.json``).

Exposed tools (under ``namespace="eval"`` → ``eval_<name>``):

- ``eval_create_set`` — writes a SCHEMA-COMPLIANT ``<name>.evalset.json`` (built from the REAL
  pydantic models ``EvalSet``/``EvalCase``/``Invocation``/``IntermediateData``).
- ``eval_set_criteria`` — writes a ``test_config.json`` (``EvalConfig`` form: ``{"criteria":
  {...}}``) with ``tool_trajectory_avg_score`` / ``response_match_score``.
- ``eval_run`` — imports the agent (``<app_name>.agent``), runs the offline evaluation (OFFLINE
  metrics: tool trajectory + ROUGE), captures the verdict + per-metric scores, persists a report,
  and returns the summary + the ``report_id``.
- ``eval_report`` — re-reads a report stored by ``(path, app_name, report_id)``.

"Tool rather than resource" choice for reading the report: a report is addressed by THREE
coordinates ``(path, app_name, report_id)``. A FastMCP 3.3.1 *resource template*
(``adk://eval/{report_id}``) only carries an opaque identifier and cannot carry
``path``/``app_name`` → ambiguous. The ``eval_report`` tool keeps the addressing explicit and
consistent with all the other ``eval_*`` (cf. ``docs/adk-api-notes/eval.md``).

Each tool returns the ``{ok, data, error}`` envelope. An eval NON-CONFORMANCE (the agent does not
meet the thresholds) is a NORMAL result (``ok=True, passed=False``), NOT an error. Real failures
(missing file, ``root_agent`` import, model requiring creds, missing ``eval`` extra) return
``err(...)`` — never an exception that propagates, never a hang.

All ``google.adk.evaluation`` imports are **lazy**: the ``eval`` extra may be absent from a slim
environment; a ``ModuleNotFoundError`` is converted into an actionable ``err``.
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - hints only, real imports are lazy
    from google.adk.evaluation.eval_set import EvalSet

eval_server: FastMCP = FastMCP("eval")

#: Subfolder (in the app) where the eval files live.
_EVAL_DIR = "eval"
#: Subfolder of the persisted reports.
_REPORTS_DIR = "eval/reports"
#: Name of the criteria file read by ADK (``EvalConfig``) in the eval set's folder.
_CONFIG_FILE = "test_config.json"
#: Canonical suffix of the new eval set schema.
_EVALSET_SUFFIX = ".evalset.json"

#: OFFLINE metric keys (cf. ``PrebuiltMetrics``; docs/adk-api-notes/eval.md).
_TOOL_TRAJECTORY_KEY = "tool_trajectory_avg_score"
_RESPONSE_MATCH_KEY = "response_match_score"

#: Actionable message if the ``eval`` extra is absent.
_EVAL_EXTRA_HINT = (
    "The ADK evaluation module is unavailable (the 'eval' extra is missing). "
    "Install it: uv add 'adk-toolkit-mcp[eval]' (or 'google-adk[eval]')."
)

#: Characters allowed in a filename slug (secures the eval set/report names).
_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Internal helpers (not exposed)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointing at the app folder (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _safe_slug(value: str) -> str:
    """Reduce ``value`` to a safe filename slug (no path separators)."""
    return _SAFE_SLUG_RE.sub("_", value.strip()).strip("._-") or "unnamed"


def _build_eval_set(name: str, cases: list[dict[str, Any]]) -> EvalSet | str:
    """Build an ``EvalSet`` (real pydantic models) from ``cases``, or an error message.

    Each case = ``{query, expected_response, expected_tool_use?}`` where ``expected_tool_use`` is a
    list of ``{name, args}``. Returns a **string** (validation error message) instead of raising,
    so the caller produces a clean ``err``.
    """
    from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation
    from google.adk.evaluation.eval_set import EvalSet
    from google.genai import types

    eval_cases: list[EvalCase] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return f"cases[{index}] must be an object {{query, expected_response, ...}}."
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            return f"cases[{index}]: 'query' (non-empty string) is required."
        expected = case.get("expected_response")
        if not isinstance(expected, str):
            return f"cases[{index}]: 'expected_response' (string) is required."

        tool_uses_raw = case.get("expected_tool_use") or []
        if not isinstance(tool_uses_raw, list):
            return f"cases[{index}]: 'expected_tool_use' must be a list of {{name, args}}."
        tool_uses: list[types.FunctionCall] = []
        for tindex, tool in enumerate(tool_uses_raw):
            if not isinstance(tool, dict):
                return f"cases[{index}].expected_tool_use[{tindex}] must be an object."
            tname = tool.get("name")
            if not isinstance(tname, str) or not tname.strip():
                return f"cases[{index}].expected_tool_use[{tindex}]: 'name' (string) is required."
            targs = tool.get("args") or {}
            if not isinstance(targs, dict):
                return f"cases[{index}].expected_tool_use[{tindex}]: 'args' must be an object."
            tool_uses.append(types.FunctionCall(name=tname, args=targs))

        invocation = Invocation(
            user_content=types.Content(role="user", parts=[types.Part.from_text(text=query)]),
            final_response=types.Content(role="model", parts=[types.Part.from_text(text=expected)]),
            intermediate_data=IntermediateData(tool_uses=tool_uses) if tool_uses else None,
        )
        eval_cases.append(EvalCase(eval_id=f"case-{index + 1}", conversation=[invocation]))

    slug = _safe_slug(name)
    return EvalSet(eval_set_id=slug, name=name.strip(), eval_cases=eval_cases)


def _load_eval_set_file(eval_set_file: str) -> EvalSet | str:
    """Load an ``EvalSet`` from a file (new schema), or an error message.

    Returns an error **string** (missing file / invalid JSON / non-conforming schema) so the
    caller produces a clean ``err``.
    """
    from google.adk.evaluation.eval_set import EvalSet
    from pydantic import ValidationError

    target = Path(eval_set_file)
    if not target.is_file():
        return f"Eval set not found: {eval_set_file}. Create it first (eval_create_set)."
    try:
        return EvalSet.model_validate_json(target.read_text(encoding="utf-8"))
    except ValidationError as exc:
        return f"Eval set does not conform to the EvalSet schema: {exc}"


def _build_eval_config(
    tool_trajectory_avg_score: float | None, response_match_score: float | None
) -> Any:
    """Build an ``EvalConfig`` with the provided OFFLINE thresholds (at least one required).

    Raises ``ValueError`` if no threshold is provided (nothing to evaluate).
    """
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.eval_metrics import BaseCriterion

    # Annotated with EvalConfig's field type (``float | BaseCriterion``) — invariant dict.
    criteria: dict[str, float | BaseCriterion] = {}
    if tool_trajectory_avg_score is not None:
        criteria[_TOOL_TRAJECTORY_KEY] = BaseCriterion(threshold=tool_trajectory_avg_score)
    if response_match_score is not None:
        criteria[_RESPONSE_MATCH_KEY] = BaseCriterion(threshold=response_match_score)
    if not criteria:
        raise ValueError("No evaluation criterion: provide at least one threshold.")
    return EvalConfig(criteria=criteria)


def _config_from_file_or_defaults(ws: Workspace, config_file: str | None) -> Any:
    """Load an ``EvalConfig`` from ``config_file`` (or the eval folder's ``test_config.json``),
    otherwise OFFLINE defaults.

    Priority: explicit ``config_file`` > ``<app_dir>/eval/test_config.json`` > defaults
    (``tool_trajectory_avg_score=1.0`` + ``response_match_score=0.8``). Raises ``ValueError`` if a
    provided file is unreadable.
    """
    from google.adk.evaluation.eval_config import EvalConfig

    candidate: Path | None = None
    if config_file:
        candidate = Path(config_file)
        if not candidate.is_file():
            raise ValueError(f"config_file not found: {config_file}.")
    else:
        default_path = ws.path(f"{_EVAL_DIR}/{_CONFIG_FILE}")
        if default_path.is_file():
            candidate = default_path

    if candidate is not None:
        try:
            return EvalConfig.model_validate_json(candidate.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - invalid JSON/schema → actionable ValueError
            raise ValueError(f"test_config.json unreadable: {exc}") from exc

    # OFFLINE defaults (cf. eval_set_criteria) if no criteria file.
    return _build_eval_config(tool_trajectory_avg_score=1.0, response_match_score=0.8)


def _evict_app_modules(app_name: str) -> None:
    """Evict ``<app_name>`` and its submodules from ``sys.modules`` (picks up an edit).

    ``AgentEvaluator._get_agent_for_eval`` uses ``importlib.import_module`` (cached). So that an
    eval re-run after an edit to ``agent.py`` is not served from a stale cache, we remove the
    entries matching the app's package.
    """
    prefix = f"{app_name}."
    for mod_name in [m for m in sys.modules if m == app_name or m.startswith(prefix)]:
        del sys.modules[mod_name]


def _metric_results_to_payload(case_results: list[Any]) -> list[dict[str, Any]]:
    """Flatten the aggregated ``EvalMetricResult`` of a list of ``EvalCaseResult`` into dicts.

    Aggregates by metric name across runs/cases: keeps the average score and the worst status.
    """
    from google.adk.evaluation.eval_metrics import EvalStatus

    by_metric: dict[str, dict[str, Any]] = {}
    for case_result in case_results:
        for metric in case_result.overall_eval_metric_results:
            entry = by_metric.setdefault(
                metric.metric_name,
                {
                    "metric_name": metric.metric_name,
                    "threshold": metric.threshold,
                    "_scores": [],
                    "statuses": [],
                },
            )
            if metric.score is not None:
                entry["_scores"].append(metric.score)
            entry["statuses"].append(
                metric.eval_status.name
                if isinstance(metric.eval_status, EvalStatus)
                else str(metric.eval_status)
            )

    payload: list[dict[str, Any]] = []
    for entry in by_metric.values():
        scores = entry.pop("_scores")
        avg = sum(scores) / len(scores) if scores else None
        statuses = entry["statuses"]
        # PASSED only if NO non-PASSED status.
        overall = "PASSED" if statuses and all(s == "PASSED" for s in statuses) else "FAILED"
        if not scores:
            overall = "NOT_EVALUATED"
        payload.append(
            {
                "metric_name": entry["metric_name"],
                "threshold": entry["threshold"],
                "score": avg,
                "eval_status": overall,
            }
        )
    return payload


async def _evaluate_offline(
    path: str,
    app_name: str,
    eval_set: EvalSet,
    eval_config: Any,
    num_runs: int,
    agent_name: str | None,
) -> dict[str, Any]:
    """Run the ADK evaluation and return a dict ``{passed, cases, metrics}``.

    Reuses exactly ``AgentEvaluator``'s public machinery (``_get_agent_for_eval`` +
    ``_get_eval_results_by_eval_id``) — the core of ``AgentEvaluator.evaluate`` — to both obtain
    the verdict (``final_eval_status``) AND capture the per-metric scores (the assert-only API does
    not expose them). The agent is imported via the dotted module ``<app_name>.agent`` (``path``
    injected into ``sys.path``).

    Any exception (model requiring creds, LLM-judge metric, import) propagates to the caller, which
    converts it into ``err``.
    """
    from google.adk.evaluation.agent_evaluator import (
        AgentEvaluator,
        get_eval_metrics_from_config,
    )
    from google.adk.evaluation.eval_metrics import EvalStatus
    from google.adk.evaluation.simulation.user_simulator_provider import UserSimulatorProvider

    if path not in sys.path:
        sys.path.insert(0, path)
    _evict_app_modules(app_name)

    # Dotted module ending in ``.agent`` (satisfies _get_agent_for_eval's convention).
    agent = await AgentEvaluator._get_agent_for_eval(
        module_name=f"{app_name}.agent", agent_name=agent_name
    )
    metrics = get_eval_metrics_from_config(eval_config)
    # ADK's eval pipeline builds a `Runner(plugins=...)` internally, emitting a
    # DeprecationWarning from `google.adk.runners`. Under `-W error::DeprecationWarning` that
    # warning would be RAISED inside ADK and abort the inference (it's caught there and recorded
    # as "Inference failed"), defeating a perfectly valid offline eval. We have no public API to
    # avoid the internal call, so we downgrade ONLY that specific ADK-internal warning, scoped to
    # this block — OUR code stays strict everywhere else.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The `plugins` argument is deprecated.*",
            category=DeprecationWarning,
            module=r"google\.adk\.runners",
        )
        results_by_id = await AgentEvaluator._get_eval_results_by_eval_id(
            agent_for_eval=agent,
            eval_set=eval_set,
            eval_metrics=metrics,
            num_runs=num_runs,
            user_simulator_provider=UserSimulatorProvider(),
        )

    cases: list[dict[str, Any]] = []
    all_metric_results: list[Any] = []
    passed = True
    for eval_id, case_results in results_by_id.items():
        all_metric_results.extend(case_results)
        # A case passes if ALL its runs are PASSED.
        case_passed = all(cr.final_eval_status == EvalStatus.PASSED for cr in case_results)
        passed = passed and case_passed
        cases.append({"eval_id": eval_id, "passed": case_passed, "runs": len(case_results)})

    return {
        "passed": passed,
        "cases": cases,
        "metrics": _metric_results_to_payload(all_metric_results),
    }


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #
@eval_server.tool(tags={"eval"})
async def create_set(
    path: str, app_name: str, name: str, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write a SCHEMA-COMPLIANT eval set ``<app_dir>/eval/<name>.evalset.json``; return the path.

    ``cases`` is a list of ``{query, expected_response, expected_tool_use?}`` where
    ``expected_tool_use`` is a list of ``{name, args}`` (expected tool trajectory). The file is
    built from the REAL pydantic models ``EvalSet``/``EvalCase``/``Invocation`` and is validated by
    ``EvalSet.model_validate_json`` (conformance proven).
    """
    if not name.strip():
        return err("name is empty.")
    if not isinstance(cases, list) or not cases:
        return err("cases is empty: provide at least one case {query, expected_response}.")

    try:
        built = _build_eval_set(name, cases)
    except ModuleNotFoundError:
        return err(_EVAL_EXTRA_HINT)
    if isinstance(built, str):
        return err(built)

    ws = _app_ws(path, app_name)
    relative = f"{_EVAL_DIR}/{_safe_slug(name)}{_EVALSET_SUFFIX}"
    content = built.model_dump_json(indent=2, exclude_none=True) + "\n"
    changed = ws.write(relative, content)

    return ok(
        {
            "app_name": app_name,
            "name": name.strip(),
            "eval_set_id": built.eval_set_id,
            "case_count": len(built.eval_cases),
            "eval_set_file": str(ws.path(relative)),
            "changed": changed,
        }
    )


@eval_server.tool(tags={"eval"})
async def set_criteria(
    path: str,
    app_name: str,
    tool_trajectory_avg_score: float = 1.0,
    response_match_score: float = 0.8,
) -> dict[str, Any]:
    """Write ``<app_dir>/eval/test_config.json`` (``EvalConfig`` form) with the OFFLINE thresholds.

    Both metrics are OFFLINE: ``tool_trajectory_avg_score`` (structural comparison of the tool
    calls) and ``response_match_score`` (ROUGE-1 on the final response). The thresholds must be in
    ``[0, 1]``. The file is read automatically by ``eval_run`` (and by ``adk eval``) from the eval
    set's folder.
    """
    for label, value in (
        ("tool_trajectory_avg_score", tool_trajectory_avg_score),
        ("response_match_score", response_match_score),
    ):
        if not (0.0 <= value <= 1.0):
            return err(f"{label} must be in [0, 1] (received {value}).")

    payload = {
        "criteria": {
            _TOOL_TRAJECTORY_KEY: tool_trajectory_avg_score,
            _RESPONSE_MATCH_KEY: response_match_score,
        }
    }
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ws = _app_ws(path, app_name)
    relative = f"{_EVAL_DIR}/{_CONFIG_FILE}"
    changed = ws.write(relative, content)

    return ok(
        {
            "app_name": app_name,
            "criteria": payload["criteria"],
            "config_file": str(ws.path(relative)),
            "changed": changed,
        }
    )


@eval_server.tool(tags={"eval"})
async def run(
    path: str,
    app_name: str,
    eval_set_file: str,
    config_file: str | None = None,
    num_runs: int = 1,
    agent_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate the app's ``root_agent`` against ``eval_set_file``; persist a report.

    Imports the agent (``<app_name>.agent``), loads the eval set + the criteria (``config_file`` or
    ``<app_dir>/eval/test_config.json``, otherwise OFFLINE defaults), runs the ADK evaluation and
    captures the verdict + the per-metric scores. The report is written under
    ``<app_dir>/eval/reports/<report_id>.json``.

    A NON-CONFORMANCE (thresholds not met) returns ``ok=True, passed=False`` (a normal result). A
    real failure — eval set absent/non-conforming, ``root_agent`` not found, model requiring creds
    (API key), LLM-judge metric, or missing ``eval`` extra — returns ``err(...)`` without a hang or
    a propagating exception. ``num_runs`` (default 1) repeats the inference and averages the
    scores. ``agent_name`` evaluates a sub-agent (otherwise the root).
    """
    if num_runs < 1:
        return err("num_runs must be >= 1.")

    loaded = _load_eval_set_file(eval_set_file)
    if isinstance(loaded, str):
        return err(loaded)
    eval_set = loaded

    ws = _app_ws(path, app_name)
    try:
        eval_config = _config_from_file_or_defaults(ws, config_file)
    except ValueError as exc:
        return err(str(exc))

    try:
        outcome = await _evaluate_offline(
            path, app_name, eval_set, eval_config, num_runs, agent_name
        )
    except ModuleNotFoundError as exc:
        # Missing ``eval`` extra (rouge_score/pandas/tabulate) OR agent module not found.
        if _looks_like_eval_extra_missing(exc):
            return err(_EVAL_EXTRA_HINT)
        return err(f"Import failed for the evaluation: {exc}")
    except Exception as exc:  # noqa: BLE001 - any inference/scoring failure → actionable err
        return err(_humanize_eval_failure(exc))

    report = _build_report(app_name, eval_set, eval_config, num_runs, outcome)
    relative = f"{_REPORTS_DIR}/{report['report_id']}.json"
    ws.write(relative, json.dumps(report, indent=2, sort_keys=True) + "\n")

    passed_count = sum(1 for c in outcome["cases"] if c["passed"])
    total = len(eval_set.eval_cases)
    verdict = "PASSED" if outcome["passed"] else "FAILED"
    metric_parts = ", ".join(
        f"{m['metric_name']}={m['score']:.3f}"
        for m in outcome["metrics"]
        if m.get("score") is not None
    )
    summary = f"{verdict} ({passed_count}/{total} cases passed)"
    if metric_parts:
        summary = f"{summary} — {metric_parts}"

    return ok(
        {
            "app_name": app_name,
            "report_id": report["report_id"],
            "report_path": str(ws.path(relative)),
            "passed": outcome["passed"],
            "summary": summary,
            "num_runs": num_runs,
            "case_count": len(eval_set.eval_cases),
            "cases": outcome["cases"],
            "metrics": outcome["metrics"],
        }
    )


@eval_server.tool(tags={"eval"})
async def report(path: str, app_name: str, report_id: str) -> dict[str, Any]:
    """Re-read an evaluation report stored by ``(path, app_name, report_id)``.

    Reads ``<app_dir>/eval/reports/<report_id>.json``. An unknown identifier returns ``err(...)``.
    Chosen as a TOOL (and not an ``adk://eval/{id}`` resource) because a report is addressed by
    three coordinates — a resource template only carries an opaque id (cf. the module docstring).
    """
    if not report_id.strip():
        return err("report_id is empty.")
    slug = _safe_slug(report_id)
    ws = _app_ws(path, app_name)
    relative = f"{_REPORTS_DIR}/{slug}.json"
    if not ws.exists(relative):
        return err(
            f"Report not found: {report_id!r} (app={app_name}). Run eval_run first to generate one."
        )
    try:
        data = json.loads(ws.read(relative))
    except json.JSONDecodeError as exc:
        return err(f"Report unreadable (invalid JSON): {exc}")
    return ok(data)


# --------------------------------------------------------------------------- #
# Report / error helpers (not exposed)
# --------------------------------------------------------------------------- #
def _build_report(
    app_name: str, eval_set: EvalSet, eval_config: Any, num_runs: int, outcome: dict[str, Any]
) -> dict[str, Any]:
    """Build the persistable report dict (id based on the timestamp + eval_set_id)."""
    timestamp = time.time()
    report_id = f"{int(timestamp * 1000)}-{_safe_slug(eval_set.eval_set_id)}"
    criteria = {name: crit_threshold(crit) for name, crit in (eval_config.criteria or {}).items()}
    return {
        "report_id": report_id,
        "app_name": app_name,
        "eval_set_id": eval_set.eval_set_id,
        "created_at": timestamp,
        "num_runs": num_runs,
        "criteria": criteria,
        "passed": outcome["passed"],
        "case_count": len(eval_set.eval_cases),
        "cases": outcome["cases"],
        "metrics": outcome["metrics"],
    }


def crit_threshold(criterion: Any) -> float | None:
    """Extract the threshold of an ``EvalConfig`` criterion (raw float or ``BaseCriterion``)."""
    if isinstance(criterion, (int, float)):
        return float(criterion)
    return getattr(criterion, "threshold", None)


def _looks_like_eval_extra_missing(exc: ModuleNotFoundError) -> bool:
    """Heuristic: does the exception concern a dependency of the ``eval`` extra?"""
    missing = (getattr(exc, "name", "") or "").lower()
    return any(pkg in missing for pkg in ("rouge", "pandas", "tabulate", "nltk", "sklearn"))


def _humanize_eval_failure(exc: Exception) -> str:
    """Turn an eval exception into an actionable ``err`` message (creds, etc.)."""
    text = str(exc)
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "api key",
            "api_key",
            "credential",
            "permission",
            "authenticat",
            "default credentials",
        )
    ):
        return (
            "Evaluation failed: the agent's model requires credentials. "
            "Set GOOGLE_API_KEY (AI Studio) or GOOGLE_GENAI_USE_VERTEXAI=TRUE + "
            f"GOOGLE_CLOUD_PROJECT (Vertex). Detail: {text}"
        )
    return f"Evaluation failed: {text}"
