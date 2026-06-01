"""Domaine `eval` : ÉVALUE un agent ADK via ``AgentEvaluator`` (P3b — évaluation).

Comme le domaine ``run`` (P3a), ce domaine **importe le ``root_agent``** d'une app et exécute
une VRAIE évaluation ADK contre un *eval set*, mais via le pipeline
``google.adk.evaluation`` plutôt qu'un simple ``Runner``. Les fichiers d'éval vivent sous
``<app_dir>/eval/`` (``<name>.evalset.json``, ``test_config.json``, ``reports/<id>.json``).

Outils exposés (sous ``namespace="eval"`` → ``eval_<nom>``) :

- ``eval_create_set`` — écrit un ``<name>.evalset.json`` SCHÉMA-CONFORME (construit à partir des
  VRAIS modèles pydantic ``EvalSet``/``EvalCase``/``Invocation``/``IntermediateData``).
- ``eval_set_criteria`` — écrit un ``test_config.json`` (forme ``EvalConfig`` : ``{"criteria":
  {...}}``) avec ``tool_trajectory_avg_score`` / ``response_match_score``.
- ``eval_run`` — importe l'agent (``<app_name>.agent``), lance l'évaluation hors-ligne (métriques
  OFFLINE : trajectoire d'outils + ROUGE), capture le verdict + scores par métrique, persiste un
  rapport, et renvoie le résumé + l'``report_id``.
- ``eval_report`` — relit un rapport stocké par ``(path, app_name, report_id)``.

Choix « tool plutôt que resource » pour la lecture du rapport : un rapport est adressé par TROIS
coordonnées ``(path, app_name, report_id)``. Un *resource template* FastMCP 3.3.1
(``adk://eval/{report_id}``) ne porte qu'un identifiant opaque et ne peut pas transporter
``path``/``app_name`` → ambigu. L'outil ``eval_report`` garde l'adressage explicite et cohérent
avec tous les autres ``eval_*`` (cf. ``docs/adk-api-notes/eval.md``).

Chaque outil renvoie l'enveloppe ``{ok, data, error}``. Une NON-CONFORMITÉ d'éval (l'agent ne
satisfait pas les seuils) est un résultat NORMAL (``ok=True, passed=False``), PAS une erreur. Les
échecs réels (fichier absent, import du ``root_agent``, modèle nécessitant des creds, extra
``eval`` absent) renvoient ``err(...)`` — jamais d'exception qui remonte, jamais de blocage.

Tous les imports ADK ``evaluation`` sont **paresseux** (lazy) : l'extra ``eval`` peut être absent
d'un environnement slim ; un ``ModuleNotFoundError`` est converti en ``err`` orienté action.
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

if TYPE_CHECKING:  # pragma: no cover - hints seulement, imports réels paresseux
    from google.adk.evaluation.eval_set import EvalSet

eval_server: FastMCP = FastMCP("eval")

#: Sous-dossier (dans l'app) où vivent les fichiers d'éval.
_EVAL_DIR = "eval"
#: Sous-dossier des rapports persistés.
_REPORTS_DIR = "eval/reports"
#: Nom du fichier de critères lu par ADK (``EvalConfig``) dans le dossier de l'eval set.
_CONFIG_FILE = "test_config.json"
#: Suffixe canonique du nouveau schéma d'eval set.
_EVALSET_SUFFIX = ".evalset.json"

#: Clés de métriques OFFLINE (cf. ``PrebuiltMetrics`` ; docs/adk-api-notes/eval.md).
_TOOL_TRAJECTORY_KEY = "tool_trajectory_avg_score"
_RESPONSE_MATCH_KEY = "response_match_score"

#: Message orienté action si l'extra ``eval`` est absent.
_EVAL_EXTRA_HINT = (
    "Le module d'évaluation ADK est indisponible (extra 'eval' manquant). "
    "Installe-le : uv add 'adk-toolkit-mcp[eval]' (ou 'google-adk[eval]')."
)

#: Caractères autorisés dans un slug de nom de fichier (sécurise les noms d'eval set/rapport).
_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- #
# Helpers internes (non exposés)
# --------------------------------------------------------------------------- #
def _app_ws(path: str, app_name: str) -> Workspace:
    """Workspace pointant sur le dossier de l'app (``<path>/<app_name>``)."""
    return Workspace(Path(path) / app_name)


def _safe_slug(value: str) -> str:
    """Réduit ``value`` à un slug de nom de fichier sûr (pas de séparateurs de chemin)."""
    return _SAFE_SLUG_RE.sub("_", value.strip()).strip("._-") or "unnamed"


def _build_eval_set(name: str, cases: list[dict[str, Any]]) -> EvalSet | str:
    """Construit un ``EvalSet`` (vrais modèles pydantic) depuis ``cases``, ou un message d'erreur.

    Chaque case = ``{query, expected_response, expected_tool_use?}`` où ``expected_tool_use`` est
    une liste de ``{name, args}``. Renvoie une **chaîne** (message d'erreur de validation) au lieu
    de lever, afin que l'appelant produise un ``err`` propre.
    """
    from google.adk.evaluation.eval_case import EvalCase, IntermediateData, Invocation
    from google.adk.evaluation.eval_set import EvalSet
    from google.genai import types

    eval_cases: list[EvalCase] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return f"cases[{index}] doit être un objet {{query, expected_response, ...}}."
        query = case.get("query")
        if not isinstance(query, str) or not query.strip():
            return f"cases[{index}] : 'query' (chaîne non vide) est requis."
        expected = case.get("expected_response")
        if not isinstance(expected, str):
            return f"cases[{index}] : 'expected_response' (chaîne) est requis."

        tool_uses_raw = case.get("expected_tool_use") or []
        if not isinstance(tool_uses_raw, list):
            return f"cases[{index}] : 'expected_tool_use' doit être une liste de {{name, args}}."
        tool_uses: list[types.FunctionCall] = []
        for tindex, tool in enumerate(tool_uses_raw):
            if not isinstance(tool, dict):
                return f"cases[{index}].expected_tool_use[{tindex}] doit être un objet."
            tname = tool.get("name")
            if not isinstance(tname, str) or not tname.strip():
                return f"cases[{index}].expected_tool_use[{tindex}] : 'name' (chaîne) est requis."
            targs = tool.get("args") or {}
            if not isinstance(targs, dict):
                return f"cases[{index}].expected_tool_use[{tindex}] : 'args' doit être un objet."
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
    """Charge un ``EvalSet`` depuis un fichier (nouveau schéma), ou un message d'erreur.

    Renvoie une **chaîne** d'erreur (fichier absent / JSON invalide / schéma non conforme) afin
    que l'appelant produise un ``err`` propre.
    """
    from google.adk.evaluation.eval_set import EvalSet
    from pydantic import ValidationError

    target = Path(eval_set_file)
    if not target.is_file():
        return f"Eval set introuvable : {eval_set_file}. Crée-le d'abord (eval_create_set)."
    try:
        return EvalSet.model_validate_json(target.read_text(encoding="utf-8"))
    except ValidationError as exc:
        return f"Eval set non conforme au schéma EvalSet : {exc}"


def _build_eval_config(
    tool_trajectory_avg_score: float | None, response_match_score: float | None
) -> Any:
    """Construit un ``EvalConfig`` avec les seuils OFFLINE fournis (au moins un requis).

    Lève ``ValueError`` si aucun seuil n'est fourni (rien à évaluer).
    """
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.eval_metrics import BaseCriterion

    # Annoté avec le type de champ d'EvalConfig (``float | BaseCriterion``) — dict invariant.
    criteria: dict[str, float | BaseCriterion] = {}
    if tool_trajectory_avg_score is not None:
        criteria[_TOOL_TRAJECTORY_KEY] = BaseCriterion(threshold=tool_trajectory_avg_score)
    if response_match_score is not None:
        criteria[_RESPONSE_MATCH_KEY] = BaseCriterion(threshold=response_match_score)
    if not criteria:
        raise ValueError("Aucun critère d'évaluation : fournis au moins un seuil.")
    return EvalConfig(criteria=criteria)


def _config_from_file_or_defaults(ws: Workspace, config_file: str | None) -> Any:
    """Charge un ``EvalConfig`` depuis ``config_file`` (ou le ``test_config.json`` du dossier eval),
    sinon des défauts OFFLINE.

    Priorité : ``config_file`` explicite > ``<app_dir>/eval/test_config.json`` > défauts
    (``tool_trajectory_avg_score=1.0`` + ``response_match_score=0.8``). Lève ``ValueError`` si un
    fichier fourni est illisible.
    """
    from google.adk.evaluation.eval_config import EvalConfig

    candidate: Path | None = None
    if config_file:
        candidate = Path(config_file)
        if not candidate.is_file():
            raise ValueError(f"config_file introuvable : {config_file}.")
    else:
        default_path = ws.path(f"{_EVAL_DIR}/{_CONFIG_FILE}")
        if default_path.is_file():
            candidate = default_path

    if candidate is not None:
        try:
            return EvalConfig.model_validate_json(candidate.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - JSON/schéma invalide → ValueError actionnable
            raise ValueError(f"test_config.json illisible : {exc}") from exc

    # Défauts OFFLINE (cf. eval_set_criteria) si aucun fichier de critères.
    return _build_eval_config(tool_trajectory_avg_score=1.0, response_match_score=0.8)


def _evict_app_modules(app_name: str) -> None:
    """Évince ``<app_name>`` et ses sous-modules de ``sys.modules`` (reprend une édition).

    ``AgentEvaluator._get_agent_for_eval`` utilise ``importlib.import_module`` (mis en cache). Pour
    qu'une éval relancée après une édition d'``agent.py`` ne soit pas servie depuis un cache
    périmé, on retire les entrées correspondant au paquet de l'app.
    """
    prefix = f"{app_name}."
    for mod_name in [m for m in sys.modules if m == app_name or m.startswith(prefix)]:
        del sys.modules[mod_name]


def _metric_results_to_payload(case_results: list[Any]) -> list[dict[str, Any]]:
    """Aplati les ``EvalMetricResult`` agrégés d'une liste d'``EvalCaseResult`` en dicts simples.

    Agrège par nom de métrique sur les runs/cases : conserve le score moyen et le pire statut.
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
        # PASSED seulement si AUCUN statut non-PASSED.
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
    """Exécute l'évaluation ADK et renvoie un dict ``{passed, cases, metrics}``.

    Réutilise exactement la machinerie publique d'``AgentEvaluator`` (``_get_agent_for_eval`` +
    ``_get_eval_results_by_eval_id``) — le cœur de ``AgentEvaluator.evaluate`` — pour à la fois
    obtenir le verdict (``final_eval_status``) ET capturer les scores par métrique (l'API
    assert-only ne les expose pas). L'agent est importé via le module dotté ``<app_name>.agent``
    (``path`` injecté dans ``sys.path``).

    Toute exception (modèle nécessitant des creds, métrique LLM-judge, import) remonte à
    l'appelant, qui la convertit en ``err``.
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

    # Module dotté terminant par ``.agent`` (satisfait la convention d'_get_agent_for_eval).
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
        # Un cas passe si TOUTES ses exécutions sont PASSED.
        case_passed = all(cr.final_eval_status == EvalStatus.PASSED for cr in case_results)
        passed = passed and case_passed
        cases.append({"eval_id": eval_id, "passed": case_passed, "runs": len(case_results)})

    return {
        "passed": passed,
        "cases": cases,
        "metrics": _metric_results_to_payload(all_metric_results),
    }


# --------------------------------------------------------------------------- #
# Outils MCP
# --------------------------------------------------------------------------- #
@eval_server.tool(tags={"eval"})
async def create_set(
    path: str, app_name: str, name: str, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Écrit un eval set SCHÉMA-CONFORME ``<app_dir>/eval/<name>.evalset.json`` ; renvoie le chemin.

    ``cases`` est une liste de ``{query, expected_response, expected_tool_use?}`` où
    ``expected_tool_use`` est une liste de ``{name, args}`` (trajectoire d'outils attendue). Le
    fichier est construit à partir des VRAIS modèles pydantic ``EvalSet``/``EvalCase``/
    ``Invocation`` et est validé par ``EvalSet.model_validate_json`` (conformité prouvée).
    """
    if not name.strip():
        return err("name est vide.")
    if not isinstance(cases, list) or not cases:
        return err("cases est vide : fournis au moins un cas {query, expected_response}.")

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
    """Écrit ``<app_dir>/eval/test_config.json`` (forme ``EvalConfig``) avec les seuils OFFLINE.

    Les deux métriques sont OFFLINE : ``tool_trajectory_avg_score`` (comparaison structurelle des
    appels d'outils) et ``response_match_score`` (ROUGE-1 sur la réponse finale). Les seuils
    doivent être dans ``[0, 1]``. Le fichier est lu automatiquement par ``eval_run`` (et par
    ``adk eval``) depuis le dossier de l'eval set.
    """
    for label, value in (
        ("tool_trajectory_avg_score", tool_trajectory_avg_score),
        ("response_match_score", response_match_score),
    ):
        if not (0.0 <= value <= 1.0):
            return err(f"{label} doit être dans [0, 1] (reçu {value}).")

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
    """Évalue le ``root_agent`` de l'app contre ``eval_set_file`` ; persiste un rapport.

    Importe l'agent (``<app_name>.agent``), charge l'eval set + les critères (``config_file`` ou
    ``<app_dir>/eval/test_config.json``, sinon défauts OFFLINE), exécute l'évaluation ADK et
    capture le verdict + les scores par métrique. Le rapport est écrit sous
    ``<app_dir>/eval/reports/<report_id>.json``.

    Une NON-CONFORMITÉ (seuils non atteints) renvoie ``ok=True, passed=False`` (résultat normal).
    Un échec réel — eval set absent/non conforme, ``root_agent`` introuvable, modèle nécessitant
    des creds (clé API), métrique LLM-judge, ou extra ``eval`` absent — renvoie ``err(...)`` sans
    blocage ni exception qui remonte. ``num_runs`` (défaut 1) répète l'inférence et moyenne les
    scores. ``agent_name`` évalue un sous-agent (sinon le root).
    """
    if num_runs < 1:
        return err("num_runs doit être >= 1.")

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
        # Extra ``eval`` absent (rouge_score/pandas/tabulate) OU module agent introuvable.
        if _looks_like_eval_extra_missing(exc):
            return err(_EVAL_EXTRA_HINT)
        return err(f"Échec de l'import pour l'évaluation : {exc}")
    except Exception as exc:  # noqa: BLE001 - tout échec d'inférence/scoring → err actionnable
        return err(_humanize_eval_failure(exc))

    report = _build_report(app_name, eval_set, eval_config, num_runs, outcome)
    relative = f"{_REPORTS_DIR}/{report['report_id']}.json"
    ws.write(relative, json.dumps(report, indent=2, sort_keys=True) + "\n")

    return ok(
        {
            "app_name": app_name,
            "report_id": report["report_id"],
            "report_path": str(ws.path(relative)),
            "passed": outcome["passed"],
            "num_runs": num_runs,
            "case_count": len(eval_set.eval_cases),
            "cases": outcome["cases"],
            "metrics": outcome["metrics"],
        }
    )


@eval_server.tool(tags={"eval"})
async def report(path: str, app_name: str, report_id: str) -> dict[str, Any]:
    """Relit un rapport d'évaluation stocké par ``(path, app_name, report_id)``.

    Lit ``<app_dir>/eval/reports/<report_id>.json``. Un identifiant inconnu renvoie ``err(...)``.
    Choisi comme OUTIL (et non resource ``adk://eval/{id}``) car un rapport est adressé par trois
    coordonnées — un resource template ne porte qu'un id opaque (cf. docstring du module).
    """
    if not report_id.strip():
        return err("report_id est vide.")
    slug = _safe_slug(report_id)
    ws = _app_ws(path, app_name)
    relative = f"{_REPORTS_DIR}/{slug}.json"
    if not ws.exists(relative):
        return err(
            f"Rapport introuvable : {report_id!r} (app={app_name}). "
            "Lance d'abord eval_run pour en générer un."
        )
    try:
        data = json.loads(ws.read(relative))
    except json.JSONDecodeError as exc:
        return err(f"Rapport illisible (JSON invalide) : {exc}")
    return ok(data)


# --------------------------------------------------------------------------- #
# Helpers de rapport / d'erreur (non exposés)
# --------------------------------------------------------------------------- #
def _build_report(
    app_name: str, eval_set: EvalSet, eval_config: Any, num_runs: int, outcome: dict[str, Any]
) -> dict[str, Any]:
    """Construit le dict de rapport persistable (id basé sur l'horodatage + eval_set_id)."""
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
    """Extrait le seuil d'un critère ``EvalConfig`` (float brut ou ``BaseCriterion``)."""
    if isinstance(criterion, (int, float)):
        return float(criterion)
    return getattr(criterion, "threshold", None)


def _looks_like_eval_extra_missing(exc: ModuleNotFoundError) -> bool:
    """Heuristique : l'exception concerne-t-elle une dépendance de l'extra ``eval`` ?"""
    missing = (getattr(exc, "name", "") or "").lower()
    return any(pkg in missing for pkg in ("rouge", "pandas", "tabulate", "nltk", "sklearn"))


def _humanize_eval_failure(exc: Exception) -> str:
    """Transforme une exception d'éval en message ``err`` actionnable (creds, etc.)."""
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
            "Échec de l'évaluation : le modèle de l'agent nécessite des identifiants. "
            "Définis GOOGLE_API_KEY (AI Studio) ou GOOGLE_GENAI_USE_VERTEXAI=TRUE + "
            f"GOOGLE_CLOUD_PROJECT (Vertex). Détail : {text}"
        )
    return f"Échec de l'évaluation : {text}"
