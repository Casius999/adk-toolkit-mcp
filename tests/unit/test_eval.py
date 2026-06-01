"""Tests unitaires du domaine ``eval`` (P3b — évaluation d'agents ADK).

Les outils ``eval_*`` opèrent sur un projet ``(path, app_name)`` et écrivent les fichiers
d'éval sous ``<app_dir>/eval/``. Les fonctions sont **async** (``asyncio_mode=auto``).

PREUVE FONCTIONNELLE (sans clé API) : on scaffolde une app dont ``agent.py`` importe un
``FakeLlm`` / ``ScriptedLlm`` (via ``sys.path``) et construit un ``LlmAgent``. On crée un
evalset dont ``expected_response`` == la réponse canned du fake (et, pour le cas outil, une
trajectoire d'outils que le ``ScriptedLlm`` satisfait). ``eval_run`` lance alors un VRAI
``AgentEvaluator`` hors-ligne avec des métriques OFFLINE (``tool_trajectory_avg_score`` +
``response_match_score``) et l'éval PASSE — prouvant le pipeline de bout en bout sans réseau.

Couverture complémentaire :
- ``create_set`` produit un fichier qui round-trip via le VRAI modèle pydantic ``EvalSet``
  (``EvalSet.model_validate_json`` réussit) → conformité de schéma prouvée.
- ``set_criteria`` écrit le ``test_config.json`` attendu (``EvalConfig`` chargeable).
- persistance du rapport + lecture via ``eval_report`` (et read-through ``fastmcp.Client``).
- ``eval_run`` sur un modèle nécessitant des creds / une métrique LLM-judge → ``err`` propre
  (pas de blocage).
- validations d'entrée (cases vide, fichier evalset absent, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import eval as E
from adk_toolkit_mcp.server import build_server

#: Dossier des fixtures (contient ``fake_llm.py``) — injecté dans le agent.py généré.
_FIXTURE_DIR = str(Path(__file__).parent)


# --------------------------------------------------------------------------- #
# Scaffolding d'apps offline (paquets importables : __init__.py + agent.py)
# --------------------------------------------------------------------------- #
def _scaffold_fake_agent(root: Path, app_name: str, answer: str) -> str:
    """App ``<app_name>`` dont ``root_agent`` est un LlmAgent + FakeLlm (réponse fixe)."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    body = (
        "import sys\n"
        f"sys.path.insert(0, r'{_FIXTURE_DIR}')\n"
        "from fake_llm import FakeLlm\n"
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(\n"
        f"    name='{app_name}', model=FakeLlm(model='fake', answer={answer!r})\n"
        ")\n"
    )
    (app_dir / "agent.py").write_text(body, encoding="utf-8")
    return str(root)


def _scaffold_tool_agent(root: Path, app_name: str) -> str:
    """App ``<app_name>`` dont ``root_agent`` utilise un ScriptedLlm + outil ``add_numbers``."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    body = (
        "import sys\n"
        f"sys.path.insert(0, r'{_FIXTURE_DIR}')\n"
        "from fake_llm import ScriptedLlm, add_numbers\n"
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(name='{app_name}', "
        "model=ScriptedLlm(model='scripted', tool_name='add_numbers', "
        "tool_args={'a': 2, 'b': 3}, final_text='The sum is 5.'), tools=[add_numbers])\n"
    )
    (app_dir / "agent.py").write_text(body, encoding="utf-8")
    return str(root)


def _scaffold_gemini_agent(root: Path, app_name: str) -> str:
    """App dont ``root_agent`` utilise un VRAI modèle Gemini (string) → nécessite des creds."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "__init__.py").write_text("from . import agent\n", encoding="utf-8")
    body = (
        "from google.adk.agents import LlmAgent\n"
        f"root_agent = LlmAgent(name='{app_name}', model='gemini-2.5-flash', "
        "instruction='Answer briefly.')\n"
    )
    (app_dir / "agent.py").write_text(body, encoding="utf-8")
    return str(root)


# --------------------------------------------------------------------------- #
# eval_create_set — conformité de schéma (round-trip via le VRAI EvalSet)
# --------------------------------------------------------------------------- #
async def test_create_set_roundtrips_through_real_evalset_model(tmp_path: Path) -> None:
    """Le fichier produit est validé par le VRAI modèle pydantic EvalSet (schéma conforme)."""
    path = _scaffold_fake_agent(tmp_path, "qa", answer="Paris.")
    result = await E.create_set(
        path=path,
        app_name="qa",
        name="basics",
        cases=[
            {"query": "Capital of France?", "expected_response": "Paris."},
            {
                "query": "Add them",
                "expected_response": "The sum is 5.",
                "expected_tool_use": [{"name": "add_numbers", "args": {"a": 2, "b": 3}}],
            },
        ],
    )
    assert result["ok"] is True, result
    eval_file = Path(result["data"]["eval_set_file"])
    assert eval_file.is_file()
    assert eval_file.name == "basics.evalset.json"

    # ROUND-TRIP via le VRAI modèle ADK : prouve la conformité (pas une supposition).
    from google.adk.evaluation.eval_set import EvalSet

    eval_set = EvalSet.model_validate_json(eval_file.read_text(encoding="utf-8"))
    assert eval_set.eval_set_id
    assert len(eval_set.eval_cases) == 2
    # 1er cas : pas d'outil ; 2e cas : trajectoire d'outils renseignée.
    case2 = eval_set.eval_cases[1]
    assert case2.conversation is not None
    inv = case2.conversation[0]
    assert inv.user_content.parts[0].text == "Add them"
    assert inv.final_response is not None
    assert inv.final_response.parts[0].text == "The sum is 5."
    assert inv.intermediate_data is not None
    assert inv.intermediate_data.tool_uses[0].name == "add_numbers"
    assert inv.intermediate_data.tool_uses[0].args == {"a": 2, "b": 3}


async def test_create_set_rejects_empty_cases(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(path=path, app_name="qa", name="empty", cases=[])
    assert result["ok"] is False
    assert "cases" in result["error"]


async def test_create_set_rejects_case_missing_query(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(
        path=path, app_name="qa", name="bad", cases=[{"expected_response": "x"}]
    )
    assert result["ok"] is False
    assert "query" in result["error"]


async def test_create_set_rejects_bad_tool_use_shape(tmp_path: Path) -> None:
    """Un expected_tool_use mal formé (pas de 'name') → err clair (validation d'entrée)."""
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(
        path=path,
        app_name="qa",
        name="bad",
        cases=[{"query": "q", "expected_response": "r", "expected_tool_use": [{"args": {}}]}],
    )
    assert result["ok"] is False
    assert "name" in result["error"]


async def test_create_set_rejects_empty_name(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(
        path=path, app_name="qa", name="  ", cases=[{"query": "q", "expected_response": "r"}]
    )
    assert result["ok"] is False
    assert "name" in result["error"]


# --------------------------------------------------------------------------- #
# eval_set_criteria — écrit un test_config.json (EvalConfig) attendu
# --------------------------------------------------------------------------- #
async def test_set_criteria_writes_expected_test_config(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.set_criteria(
        path=path, app_name="qa", tool_trajectory_avg_score=1.0, response_match_score=0.8
    )
    assert result["ok"] is True, result
    cfg_file = Path(result["data"]["config_file"])
    assert cfg_file.is_file()
    assert cfg_file.name == "test_config.json"

    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert raw["criteria"]["tool_trajectory_avg_score"] == 1.0
    assert raw["criteria"]["response_match_score"] == 0.8

    # Le fichier est un EvalConfig valide (ADK le chargera via model_validate_json).
    from google.adk.evaluation.eval_config import EvalConfig

    cfg = EvalConfig.model_validate_json(cfg_file.read_text(encoding="utf-8"))
    assert cfg.criteria["tool_trajectory_avg_score"] == 1.0


async def test_set_criteria_rejects_out_of_range(tmp_path: Path) -> None:
    """Un seuil hors [0, 1] → err (validation d'entrée)."""
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.set_criteria(path=path, app_name="qa", response_match_score=1.5)
    assert result["ok"] is False
    assert "response_match_score" in result["error"]


# --------------------------------------------------------------------------- #
# FUNCTIONAL — eval_run PASSE hors-ligne (métriques offline, sans clé)
# --------------------------------------------------------------------------- #
async def test_eval_run_passes_offline_response_match(tmp_path: Path) -> None:
    """eval_run : un agent FakeLlm dont la réponse == expected_response PASSE offline (ROUGE)."""
    answer = "Paris is the capital of France."
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer=answer)
    created = await E.create_set(
        path=path,
        app_name="qaapp",
        name="qa",
        cases=[{"query": "What is the capital of France?", "expected_response": answer}],
    )
    assert created["ok"] is True
    # Critères OFFLINE uniquement : response_match (ROUGE), pas de métrique LLM-judge.
    await E.set_criteria(path=path, app_name="qaapp", response_match_score=0.7)

    result = await E.run(
        path=path,
        app_name="qaapp",
        eval_set_file=created["data"]["eval_set_file"],
        config_file=None,  # auto-détecte test_config.json dans le dossier eval
        num_runs=1,
    )
    assert result["ok"] is True, result
    assert result["data"]["passed"] is True, result["data"]
    # Score par métrique capturé dans le rapport.
    metrics = {m["metric_name"]: m for m in result["data"]["metrics"]}
    assert "response_match_score" in metrics
    assert metrics["response_match_score"]["score"] >= 0.7
    # summary doit être non-vide et refléter la conformité.
    summary = result["data"]["summary"]
    assert summary and isinstance(summary, str)
    assert "PASSED" in summary
    assert "1/1" in summary  # 1 case, 1 passed


async def test_eval_run_passes_offline_tool_trajectory(tmp_path: Path) -> None:
    """eval_run : un ScriptedLlm satisfait la trajectoire d'outils + la réponse → PASSE offline."""
    path = _scaffold_tool_agent(tmp_path, "calcapp")
    created = await E.create_set(
        path=path,
        app_name="calcapp",
        name="calc",
        cases=[
            {
                "query": "2+3?",
                "expected_response": "The sum is 5.",
                "expected_tool_use": [{"name": "add_numbers", "args": {"a": 2, "b": 3}}],
            }
        ],
    )
    assert created["ok"] is True
    await E.set_criteria(
        path=path, app_name="calcapp", tool_trajectory_avg_score=1.0, response_match_score=0.7
    )

    result = await E.run(
        path=path,
        app_name="calcapp",
        eval_set_file=created["data"]["eval_set_file"],
        num_runs=1,
    )
    assert result["ok"] is True, result
    assert result["data"]["passed"] is True, result["data"]
    metrics = {m["metric_name"]: m for m in result["data"]["metrics"]}
    assert metrics["tool_trajectory_avg_score"]["score"] == 1.0
    assert metrics["response_match_score"]["score"] >= 0.7


async def test_eval_run_fails_offline_on_wrong_expected(tmp_path: Path) -> None:
    """Un expected_response délibérément faux → l'éval ÉCHOUE (passed False), mais ok=True.

    Prouve que le pipeline évalue VRAIMENT (ne fabrique pas un succès) : une non-conformité
    est un résultat d'éval normal (ok=True, passed=False), PAS une erreur d'outil.
    """
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="Paris.")
    created = await E.create_set(
        path=path,
        app_name="qaapp",
        name="wrong",
        cases=[{"query": "hi", "expected_response": "Totally unrelated zzzzz qqqqq wwww."}],
    )
    assert created["ok"] is True
    await E.set_criteria(path=path, app_name="qaapp", response_match_score=0.9)

    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"], num_runs=1
    )
    assert result["ok"] is True, result
    assert result["data"]["passed"] is False, result["data"]
    # summary doit indiquer FAILED pour une éval échouée.
    summary = result["data"]["summary"]
    assert summary and isinstance(summary, str)
    assert "FAILED" in summary


# --------------------------------------------------------------------------- #
# Persistance du rapport + lecture (eval_report) + read-through Client
# --------------------------------------------------------------------------- #
async def test_eval_run_persists_report_and_eval_report_reads_it(tmp_path: Path) -> None:
    answer = "Bonjour le monde."
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer=answer)
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": answer}]
    )
    await E.set_criteria(path=path, app_name="qaapp", response_match_score=0.7)
    run = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"], num_runs=1
    )
    assert run["ok"] is True, run
    report_id = run["data"]["report_id"]
    assert report_id

    # Le fichier rapport existe sous <app_dir>/eval/reports/.
    report_path = Path(run["data"]["report_path"])
    assert report_path.is_file()
    assert report_path.parent.name == "reports"

    # eval_report relit le rapport stocké.
    got = await E.report(path=path, app_name="qaapp", report_id=report_id)
    assert got["ok"] is True, got
    assert got["data"]["report_id"] == report_id
    assert got["data"]["passed"] is True
    assert got["data"]["metrics"]


async def test_eval_report_unknown_id_returns_err(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    result = await E.report(path=path, app_name="qaapp", report_id="does-not-exist")
    assert result["ok"] is False
    assert result["error"]


# --------------------------------------------------------------------------- #
# eval_run — chemins d'erreur propres (pas de blocage / pas d'exception)
# --------------------------------------------------------------------------- #
async def test_eval_run_missing_evalset_file_returns_err(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=str(tmp_path / "ghost.evalset.json")
    )
    assert result["ok"] is False
    assert result["error"]


async def test_eval_run_missing_agent_returns_err(tmp_path: Path) -> None:
    """App sans agent.py importable → err (import échoué), pas d'exception."""
    # Crée un evalset valide mais aucune app importable.
    (tmp_path / "ghost").mkdir(parents=True, exist_ok=True)
    es = tmp_path / "ghost" / "eval" / "x.evalset.json"
    es.parent.mkdir(parents=True, exist_ok=True)
    es.write_text(
        json.dumps(
            {
                "eval_set_id": "x",
                "eval_cases": [
                    {
                        "eval_id": "c1",
                        "conversation": [
                            {
                                "user_content": {"role": "user", "parts": [{"text": "hi"}]},
                                "final_response": {"role": "model", "parts": [{"text": "hi"}]},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = await E.run(path=str(tmp_path), app_name="ghost", eval_set_file=str(es))
    assert result["ok"] is False
    assert result["error"]


async def test_eval_run_credential_needing_model_returns_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un VRAI modèle Gemini sans clé → err actionnable (pas de hang, pas d'exception).

    On neutralise toute creds de l'environnement puis on lance une éval contre un agent dont
    le modèle est ``gemini-2.5-flash`` (string) : l'inférence échoue faute de clé → eval_run
    convertit l'exception en err propre. (Si une clé était présente l'appel réseau réel serait
    tenté ; en CI il n'y en a pas, ce qui est le cas couvert.)
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    path = _scaffold_gemini_agent(tmp_path, "realapp")
    created = await E.create_set(
        path=path, app_name="realapp", name="qa", cases=[{"query": "hi", "expected_response": "hi"}]
    )
    await E.set_criteria(path=path, app_name="realapp", response_match_score=0.7)
    result = await E.run(
        path=path, app_name="realapp", eval_set_file=created["data"]["eval_set_file"], num_runs=1
    )
    assert result["ok"] is False, result
    assert result["error"]


# --------------------------------------------------------------------------- #
# In-memory fastmcp.Client read-through (noms exposés + garde double-préfixe)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_eval_run(tmp_path: Path) -> None:
    """Les outils sont exposés eval_<bare> (pas de double-préfixe) et eval_run s'exécute offline."""
    answer = "client eval ok"
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer=answer)
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {"eval_create_set", "eval_set_criteria", "eval_run", "eval_report"}
        assert expected <= names
        assert not any(n.startswith("eval_eval_") for n in names)

        created = await client.call_tool(
            "eval_create_set",
            {
                "path": path,
                "app_name": "qaapp",
                "name": "qa",
                "cases": [{"query": "hi", "expected_response": answer}],
            },
        )
        assert created.data["ok"] is True
        await client.call_tool(
            "eval_set_criteria",
            {"path": path, "app_name": "qaapp", "response_match_score": 0.7},
        )
        run = await client.call_tool(
            "eval_run",
            {
                "path": path,
                "app_name": "qaapp",
                "eval_set_file": created.data["data"]["eval_set_file"],
                "num_runs": 1,
            },
        )
        assert run.data["ok"] is True, run.data
        assert run.data["data"]["passed"] is True

        # Read-through du rapport via eval_report.
        got = await client.call_tool(
            "eval_report",
            {"path": path, "app_name": "qaapp", "report_id": run.data["data"]["report_id"]},
        )
        assert got.data["ok"] is True
        assert got.data["data"]["report_id"] == run.data["data"]["report_id"]


# --------------------------------------------------------------------------- #
# create_set — branches de validation détaillées (cases malformés)
# --------------------------------------------------------------------------- #
async def test_create_set_rejects_non_dict_case(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(path=path, app_name="qa", name="bad", cases=["not a dict"])
    assert result["ok"] is False
    assert "cases[0]" in result["error"]


async def test_create_set_rejects_missing_expected_response(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(path=path, app_name="qa", name="bad", cases=[{"query": "q"}])
    assert result["ok"] is False
    assert "expected_response" in result["error"]


async def test_create_set_rejects_tool_use_not_list(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(
        path=path,
        app_name="qa",
        name="bad",
        cases=[{"query": "q", "expected_response": "r", "expected_tool_use": "nope"}],
    )
    assert result["ok"] is False
    assert "expected_tool_use" in result["error"]


async def test_create_set_rejects_tool_not_dict(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(
        path=path,
        app_name="qa",
        name="bad",
        cases=[{"query": "q", "expected_response": "r", "expected_tool_use": ["x"]}],
    )
    assert result["ok"] is False


async def test_create_set_rejects_tool_args_not_object(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qa", answer="x")
    result = await E.create_set(
        path=path,
        app_name="qa",
        name="bad",
        cases=[
            {
                "query": "q",
                "expected_response": "r",
                "expected_tool_use": [{"name": "t", "args": "nope"}],
            }
        ],
    )
    assert result["ok"] is False
    assert "args" in result["error"]


# --------------------------------------------------------------------------- #
# eval_run — branches config + validations
# --------------------------------------------------------------------------- #
async def test_eval_run_rejects_num_runs_zero(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": "x"}]
    )
    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"], num_runs=0
    )
    assert result["ok"] is False
    assert "num_runs" in result["error"]


async def test_eval_run_explicit_config_file_not_found_returns_err(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": "x"}]
    )
    result = await E.run(
        path=path,
        app_name="qaapp",
        eval_set_file=created["data"]["eval_set_file"],
        config_file=str(tmp_path / "nope.json"),
    )
    assert result["ok"] is False
    assert "config_file" in result["error"]


async def test_eval_run_malformed_config_file_returns_err(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": "x"}]
    )
    bad_cfg = tmp_path / "qaapp" / "eval" / "test_config.json"
    bad_cfg.write_text("{ not valid json", encoding="utf-8")
    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"]
    )
    assert result["ok"] is False
    assert "test_config.json" in result["error"]


async def test_eval_run_uses_default_criteria_when_no_config(tmp_path: Path) -> None:
    """Sans test_config.json, eval_run applique les défauts OFFLINE (1.0 / 0.8) et PASSE."""
    answer = "Default criteria path works."
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer=answer)
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": answer}]
    )
    # Pas d'appel à set_criteria → défauts.
    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"], num_runs=1
    )
    assert result["ok"] is True, result
    assert result["data"]["passed"] is True


async def test_eval_run_non_conformant_evalset_file_returns_err(tmp_path: Path) -> None:
    """Un fichier evalset présent mais non conforme au schéma EvalSet → err propre."""
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    bad = tmp_path / "qaapp" / "eval" / "bad.evalset.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(json.dumps({"not": "an evalset"}), encoding="utf-8")
    result = await E.run(path=path, app_name="qaapp", eval_set_file=str(bad))
    assert result["ok"] is False
    assert result["error"]


async def test_eval_report_rejects_empty_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    result = await E.report(path=path, app_name="qaapp", report_id="  ")
    assert result["ok"] is False
    assert "report_id" in result["error"]


async def test_eval_report_corrupt_json_returns_err(tmp_path: Path) -> None:
    """Un fichier rapport corrompu → err (JSON invalide), pas d'exception."""
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    report_file = tmp_path / "qaapp" / "eval" / "reports" / "corrupt.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("{ broken", encoding="utf-8")
    result = await E.report(path=path, app_name="qaapp", report_id="corrupt")
    assert result["ok"] is False
    assert result["error"]


# --------------------------------------------------------------------------- #
# Helpers purs (unitaires, sans I/O)
# --------------------------------------------------------------------------- #
def test_crit_threshold_handles_float_and_criterion() -> None:
    """crit_threshold extrait le seuil d'un float brut ET d'un BaseCriterion."""
    from google.adk.evaluation.eval_metrics import BaseCriterion

    assert E.crit_threshold(0.75) == 0.75
    assert E.crit_threshold(BaseCriterion(threshold=0.9)) == 0.9
    assert E.crit_threshold(object()) is None


def test_humanize_eval_failure_flags_credentials() -> None:
    """Un message d'erreur mentionnant des creds → message orienté action (GOOGLE_API_KEY)."""
    msg = E._humanize_eval_failure(RuntimeError("Missing API key for the model"))
    assert "GOOGLE_API_KEY" in msg
    # Un message générique reste générique.
    generic = E._humanize_eval_failure(RuntimeError("some other failure"))
    assert "some other failure" in generic
    assert "GOOGLE_API_KEY" not in generic


def test_looks_like_eval_extra_missing() -> None:
    """L'heuristique reconnaît une dépendance de l'extra eval (rouge_score) et l'ignore sinon."""
    assert E._looks_like_eval_extra_missing(ModuleNotFoundError(name="rouge_score")) is True
    assert E._looks_like_eval_extra_missing(ModuleNotFoundError(name="pandas")) is True
    assert E._looks_like_eval_extra_missing(ModuleNotFoundError(name="some_app")) is False


def test_safe_slug_sanitizes_paths() -> None:
    """_safe_slug retire les séparateurs de chemin et caractères dangereux."""
    assert "/" not in E._safe_slug("a/b/c")
    assert "\\" not in E._safe_slug("a\\b")
    assert E._safe_slug("  ...  ") == "unnamed"


async def test_eval_run_extra_missing_returns_actionable_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Si l'évaluation lève ModuleNotFoundError(rouge_score), eval_run renvoie l'astuce extra.

    On simule l'absence de l'extra ``eval`` en forçant ``_evaluate_offline`` à lever
    ``ModuleNotFoundError(name='rouge_score')`` (la vraie machinerie n'est pas appelée).
    """

    async def _boom(*_args: object, **_kwargs: object) -> dict:
        raise ModuleNotFoundError(name="rouge_score")

    monkeypatch.setattr(E, "_evaluate_offline", _boom)
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": "x"}]
    )
    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"]
    )
    assert result["ok"] is False
    assert "eval" in result["error"].lower()


async def test_eval_run_import_error_non_extra_returns_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Un ModuleNotFoundError SANS rapport avec l'extra eval → err d'import générique."""

    async def _boom(*_args: object, **_kwargs: object) -> dict:
        raise ModuleNotFoundError(name="totally_unrelated_pkg")

    monkeypatch.setattr(E, "_evaluate_offline", _boom)
    path = _scaffold_fake_agent(tmp_path, "qaapp", answer="x")
    created = await E.create_set(
        path=path, app_name="qaapp", name="qa", cases=[{"query": "hi", "expected_response": "x"}]
    )
    result = await E.run(
        path=path, app_name="qaapp", eval_set_file=created["data"]["eval_set_file"]
    )
    assert result["ok"] is False
    assert "import" in result["error"].lower()
