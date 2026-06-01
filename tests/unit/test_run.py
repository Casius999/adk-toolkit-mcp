"""Tests unitaires du domaine ``run`` (P3a — exécution d'agents ADK).

Les outils sont **async** (``asyncio_mode=auto``). On appelle les fonctions bare directement
et, pour le read-through, via un ``fastmcp.Client`` in-memory.

PREUVE FONCTIONNELLE (sans clé API) : on scaffolde une app dont ``agent.py`` importe un
``FakeLlm`` de la fixture (via ``sys.path``) et construit un ``LlmAgent``. ``run_agent`` exécute
alors une vraie boucle d'agent hors-ligne et renvoie le texte final canned — prouvant que
l'outil monté exécute un agent de bout en bout sans réseau.

Couverture complémentaire :
- validations (user_id/session_id/message vides) et erreurs propres (agent.py absent →
  RootAgentImportError convertie en err ; config corrompue → err).
- ``run_config_build`` : modes valides + invalide.
- ``run_inspect_events`` (PUR) : résumé d'événements synthétiques + invalides.
- ``run_stream`` : le callback de progression est invoqué par événement (prouvé via
  ``collect_events`` avec un callback, et via un ``fastmcp.Client`` qui capte ctx.report_progress).
- ``run_live`` : renvoie un err actionnable quand la capacité/clé Live est absente (pas de blocage).
- read-through ``fastmcp.Client`` pour ``run_agent`` contre un agent FakeLlm.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.domains import run as R
from adk_toolkit_mcp.runtime import reset_service_cache
from adk_toolkit_mcp.server import build_server

#: Dossier des fixtures de test (contient ``fake_llm.py``) — injecté dans le agent.py généré.
_FIXTURE_DIR = str(Path(__file__).parent)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Isole les tests : vide le cache singleton de services avant/après chacun."""
    reset_service_cache()
    yield
    reset_service_cache()


def _scaffold_fake_agent(
    root: Path, app_name: str = "myapp", answer: str = "Hello offline!"
) -> str:
    """Écrit une app dont ``agent.py`` construit un LlmAgent + FakeLlm (offline). Renvoie path."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
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


def _scaffold_tool_agent(root: Path, app_name: str = "calc") -> str:
    """Écrit une app dont ``agent.py`` construit un agent ScriptedLlm + outil (offline)."""
    app_dir = root / app_name
    app_dir.mkdir(parents=True, exist_ok=True)
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


def _persist_max_llm_calls(path: str, app_name: str, value: int) -> None:
    """Persiste ``max_llm_calls=value`` sur l'agent ROOT via le VRAI outil ``safety_settings``.

    On crée d'abord l'agent root dans le sidecar (``agents_create_llm`` + ``agents_set_root``),
    puis on appelle ``safety_settings(max_llm_calls=value)`` — exactement le chemin utilisateur.
    ``safety_settings`` régénère ``agent.py`` (modèle Gemini), donc l'appelant le RÉÉCRIT ensuite
    avec un FakeLlm pour rester exécutable hors-ligne (le sidecar ``agents.json`` — d'où la valeur
    persistée est relue — n'est pas affecté par cette réécriture d'``agent.py``).
    """
    from adk_toolkit_mcp.domains import agents as AGENTS
    from adk_toolkit_mcp.domains import safety as SAFETY

    assert AGENTS.create_llm(path=path, app_name=app_name, name=app_name)["ok"]
    assert AGENTS.set_root(path=path, app_name=app_name, name=app_name)["ok"]
    res = SAFETY.safety_settings(
        path=path, app_name=app_name, agent_name=app_name, max_llm_calls=value
    )
    assert res["ok"], res
    assert res["data"]["max_llm_calls"] == value


# --------------------------------------------------------------------------- #
# FUNCTIONAL — run_agent exécute un agent FakeLlm hors-ligne
# --------------------------------------------------------------------------- #
async def test_run_agent_functional_offline(tmp_path: Path) -> None:
    """run_agent (outil monté) exécute un agent FakeLlm chargé depuis agent.py → texte final."""
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="42 is the answer")
    result = await R.agent(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s1",
        message="what is the answer?",
    )
    assert result["ok"] is True, result
    assert result["data"]["final_text"] == "42 is the answer"
    assert result["data"]["event_count"] >= 1
    # Les événements sont sérialisés (clés attendues).
    ev = result["data"]["events"][0]
    assert {"author", "text", "is_final", "function_calls"} <= set(ev)


async def test_run_agent_functional_tool_loop_offline(tmp_path: Path) -> None:
    """run_agent prouve une boucle tool-call complète offline : call → response → final."""
    path = _scaffold_tool_agent(tmp_path, "calc")
    result = await R.agent(
        path=path, app_name="calc", user_id="u1", session_id="s1", message="2+3?"
    )
    assert result["ok"] is True, result
    events = result["data"]["events"]
    assert any(e["function_calls"] for e in events), events
    assert any(e["function_responses"] for e in events), events
    assert result["data"]["final_text"] == "The sum is 5."


async def test_run_agent_reuses_session_across_calls(tmp_path: Path) -> None:
    """Deux run_agent sur le même session_id : le second voit les événements du premier."""
    path = _scaffold_fake_agent(tmp_path, "myapp")
    first = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert first["ok"] is True
    first_count = first["data"]["event_count"]

    # Vérifie via le domaine sessions que la session accumule des événements.
    from adk_toolkit_mcp.domains import sessions as S

    got = await S.get(path=path, app_name="myapp", user_id="u1", session_id="s1")
    assert got["ok"] is True
    assert got["data"]["event_count"] >= first_count


# --------------------------------------------------------------------------- #
# FUNCTIONAL — persisted max_llm_calls (safety_settings) is honored by run_*
# --------------------------------------------------------------------------- #
async def test_run_agent_uses_persisted_max_llm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_agent SANS max_llm_calls explicite applique le plafond PERSISTÉ (=7) du root agent.

    On persiste ``safety_settings(..., max_llm_calls=7)`` sur le root (vrai chemin utilisateur),
    on réécrit ``agent.py`` en FakeLlm (offline), puis on appelle ``run_agent`` SANS passer
    ``max_llm_calls``. On capte la ``RunConfig`` réellement construite via un seam : on monkeypatch
    ``R.build_run_config`` pour enregistrer l'argument ``max_llm_calls`` reçu et le ``RunConfig``
    renvoyé (en déléguant à la vraie fabrique pour que le run aboutisse).

    Ce test ÉCHOUE avant le correctif (run_* ignorait la valeur persistée → build_run_config
    recevait ``None`` au lieu de 7).
    """
    from adk_toolkit_mcp import run_core

    path = _scaffold_fake_agent(tmp_path, "myapp", answer="capped")
    _persist_max_llm_calls(path, "myapp", 7)
    # safety_settings a régénéré agent.py (Gemini) : on le remet en FakeLlm (exécution offline).
    _scaffold_fake_agent(tmp_path, "myapp", answer="capped")

    seen: dict[str, object] = {}
    real_build = run_core.build_run_config

    def _spy_build_run_config(**kwargs: object) -> object:
        seen["max_llm_calls"] = kwargs.get("max_llm_calls")
        cfg = real_build(**kwargs)  # type: ignore[arg-type]
        seen["run_config"] = cfg
        return cfg

    monkeypatch.setattr(R, "build_run_config", _spy_build_run_config)

    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is True, result
    # Le plafond persisté (7) a été résolu et passé à build_run_config…
    assert seen["max_llm_calls"] == 7
    # …et le RunConfig réellement utilisé par le runner porte bien max_llm_calls == 7.
    assert seen["run_config"].max_llm_calls == 7  # type: ignore[attr-defined]


async def test_run_agent_explicit_max_llm_calls_overrides_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Une valeur d'appelant explicite PRIME sur le persisté (7 persisté, 3 explicite → 3)."""
    from adk_toolkit_mcp import run_core

    path = _scaffold_fake_agent(tmp_path, "myapp", answer="capped")
    _persist_max_llm_calls(path, "myapp", 7)
    _scaffold_fake_agent(tmp_path, "myapp", answer="capped")

    seen: dict[str, object] = {}
    real_build = run_core.build_run_config

    def _spy_build_run_config(**kwargs: object) -> object:
        seen["max_llm_calls"] = kwargs.get("max_llm_calls")
        return real_build(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(R, "build_run_config", _spy_build_run_config)

    result = await R.agent(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s1",
        message="hi",
        max_llm_calls=3,
    )
    assert result["ok"] is True, result
    # L'explicite (3) écrase le persisté (7).
    assert seen["max_llm_calls"] == 3


async def test_run_agent_without_sidecar_uses_adk_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sans sidecar ni valeur explicite, max_llm_calls reste None → défaut ADK (pas de régression).

    Garantit que l'enrichissement par valeur persistée est best-effort : une app scaffoldée sans
    sidecar ``agents.json`` (cas historique de ces tests) garde le comportement d'avant (``None``).
    """
    from adk_toolkit_mcp import run_core

    path = _scaffold_fake_agent(tmp_path, "myapp", answer="default")
    seen: dict[str, object] = {}
    real_build = run_core.build_run_config

    def _spy_build_run_config(**kwargs: object) -> object:
        seen["max_llm_calls"] = kwargs.get("max_llm_calls")
        return real_build(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(R, "build_run_config", _spy_build_run_config)

    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is True, result
    assert seen["max_llm_calls"] is None


# --------------------------------------------------------------------------- #
# Validation + error paths
# --------------------------------------------------------------------------- #
async def test_run_agent_rejects_empty_message(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="  ")
    assert result["ok"] is False
    assert "message" in result["error"]


async def test_run_agent_rejects_empty_user_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(path=path, app_name="myapp", user_id="  ", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "user_id" in result["error"]


async def test_run_agent_rejects_empty_session_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id=" ", message="hi")
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_run_agent_missing_agent_py_returns_err(tmp_path: Path) -> None:
    """Pas d'agent.py → RootAgentImportError convertie en err actionnable (pas d'exception)."""
    result = await R.agent(
        path=str(tmp_path), app_name="ghost", user_id="u1", session_id="s1", message="hi"
    )
    assert result["ok"] is False
    assert "introuvable" in result["error"].lower()


async def test_run_agent_broken_agent_py_returns_err(tmp_path: Path) -> None:
    app_dir = tmp_path / "myapp"
    app_dir.mkdir(parents=True)
    (app_dir / "agent.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    result = await R.agent(
        path=str(tmp_path), app_name="myapp", user_id="u1", session_id="s1", message="hi"
    )
    assert result["ok"] is False
    assert result["error"]


async def test_run_agent_invalid_streaming_mode_returns_err(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.agent(
        path=path,
        app_name="myapp",
        user_id="u1",
        session_id="s1",
        message="hi",
        streaming_mode="TURBO",
    )
    assert result["ok"] is False
    assert "streaming_mode" in result["error"]


async def test_run_agent_corrupt_config_returns_err(tmp_path: Path) -> None:
    """runtime.json corrompue → err propre (la config se charge avant l'import de l'agent)."""
    path = _scaffold_fake_agent(tmp_path, "myapp")
    cfg_dir = tmp_path / "myapp" / ".adk_toolkit"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "runtime.json").write_text("{ broken", encoding="utf-8")
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert result["error"]


# --------------------------------------------------------------------------- #
# run_stream — progression par événement
# --------------------------------------------------------------------------- #
async def test_run_stream_offline_no_ctx(tmp_path: Path) -> None:
    """run_stream fonctionne sans ctx (progression no-op) et renvoie le texte final."""
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="streamed!")
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id="s1", message="go", ctx=None
    )
    assert result["ok"] is True
    assert result["data"]["streaming_mode"] == "SSE"
    assert result["data"]["final_text"] == "streamed!"


async def test_run_stream_rejects_empty_message(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id="s1", message="", ctx=None
    )
    assert result["ok"] is False


async def test_run_stream_rejects_empty_user_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.stream(
        path=path, app_name="myapp", user_id=" ", session_id="s1", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert "user_id" in result["error"]


async def test_run_stream_rejects_empty_session_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id=" ", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_run_stream_missing_agent_returns_err(tmp_path: Path) -> None:
    """run_stream sur une app sans agent.py → err (import échoué), pas d'exception."""
    result = await R.stream(
        path=str(tmp_path), app_name="ghost", user_id="u1", session_id="s1", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert result["error"]


async def test_run_stream_invalid_backend_returns_err(tmp_path: Path) -> None:
    """Backend invalide (database sans db_url, édité à la main) → err propre via run_stream.

    Couvre la branche ValueError de run_stream (backend non instanciable). run_stream force
    streaming_mode='SSE', donc l'unique ValueError vient ici du backend.
    """
    path = _scaffold_fake_agent(tmp_path, "myapp")
    cfg_dir = tmp_path / "myapp" / ".adk_toolkit"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "runtime.json").write_text(
        '{"session": {"kind": "database", "db_url": null}}', encoding="utf-8"
    )
    result = await R.stream(
        path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi", ctx=None
    )
    assert result["ok"] is False
    assert "db_url" in result["error"]


async def test_run_agent_invalid_backend_returns_err(tmp_path: Path) -> None:
    """run_agent sur un backend non instanciable (database sans db_url) → err propre."""
    path = _scaffold_fake_agent(tmp_path, "myapp")
    cfg_dir = tmp_path / "myapp" / ".adk_toolkit"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "runtime.json").write_text(
        '{"session": {"kind": "database", "db_url": null}}', encoding="utf-8"
    )
    result = await R.agent(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "db_url" in result["error"]


async def test_run_stream_progress_via_client(tmp_path: Path) -> None:
    """Via un fastmcp.Client, run_stream rapporte la progression : le handler reçoit des appels.

    On capte ``progress`` côté client (FastMCP injecte le Context et relaie report_progress).
    Au moins un événement → au moins un appel de progression.
    """
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="progress proof")
    mcp = build_server()

    progress_calls: list[tuple[float, float | None, str | None]] = []

    async def _handler(progress: float, total: float | None, message: str | None) -> None:
        progress_calls.append((progress, total, message))

    async with Client(mcp, progress_handler=_handler) as client:
        res = await client.call_tool(
            "run_stream",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": "s1",
                "message": "hi",
            },
        )
        assert res.data["ok"] is True
        assert res.data["data"]["final_text"] == "progress proof"

    # Le handler de progression a été invoqué au moins une fois (un event final au minimum).
    assert progress_calls, "report_progress aurait dû être relayé au client"


# --------------------------------------------------------------------------- #
# run_live — dégradation actionnable sans clé/capacité (pas de blocage)
# --------------------------------------------------------------------------- #
async def test_run_live_without_credentials_returns_actionable_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sans clé Live, run_live renvoie un err actionnable immédiat (jamais de hang)."""
    # Neutralise toute creds Live éventuellement présente dans l'environnement.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "GOOGLE_API_KEY" in result["error"]


async def test_run_live_with_key_but_non_live_model_returns_err(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avec une clé mais un modèle non live-capable (FakeLlm), run_live renvoie un err clair.

    Prouve la seconde garde : même avec des creds, un FakeLlm (connect non surchargé) ne peut pas
    streamer en Live → err actionnable, toujours sans blocage réseau.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-not-used")
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    assert "Live" in result["error"]


async def test_run_live_vertex_credentials_recognized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Des creds Vertex (USE_VERTEXAI=TRUE + PROJECT) passent la 1re garde ; échec sur le modèle.

    Couvre la branche Vertex de _has_live_credentials : sans clé AI Studio mais avec Vertex
    configuré, la détection de creds réussit → on tombe sur la garde de capacité du modèle
    (FakeLlm non live-capable) → err ``Live``, toujours sans blocage.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")

    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="hi")
    assert result["ok"] is False
    # 1re garde franchie (creds Vertex reconnues) → on bute sur la garde modèle (Live).
    assert "Live" in result["error"]


def test_model_supports_live_handles_missing_canonical_model() -> None:
    """_model_supports_live renvoie False (sans lever) si l'agent n'a pas de canonical_model."""

    class _Bare:
        pass

    assert R._model_supports_live(_Bare()) is False  # type: ignore[arg-type]


def test_model_supports_live_swallows_exceptions() -> None:
    """Une erreur en accédant au modèle → False (détection défensive, jamais de raise)."""

    class _Exploding:
        @property
        def canonical_model(self) -> object:
            raise RuntimeError("boom on access")

    assert R._model_supports_live(_Exploding()) is False  # type: ignore[arg-type]


async def test_run_live_rejects_empty_message(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="s1", message="  ")
    assert result["ok"] is False
    assert "message" in result["error"]


async def test_run_live_rejects_empty_session_id(tmp_path: Path) -> None:
    path = _scaffold_fake_agent(tmp_path, "myapp")
    result = await R.live(path=path, app_name="myapp", user_id="u1", session_id="  ", message="hi")
    assert result["ok"] is False
    assert "session_id" in result["error"]


async def test_run_live_missing_agent_returns_err(tmp_path: Path) -> None:
    result = await R.live(
        path=str(tmp_path), app_name="ghost", user_id="u1", session_id="s1", message="hi"
    )
    assert result["ok"] is False


# --------------------------------------------------------------------------- #
# run_config_build (validation pure)
# --------------------------------------------------------------------------- #
def test_config_build_valid() -> None:
    result = R.config_build(streaming_mode="SSE", max_llm_calls=10)
    assert result["ok"] is True
    assert result["data"]["streaming_mode"] == "SSE"
    assert result["data"]["max_llm_calls"] == 10
    assert set(result["data"]["streaming_options"]) == {"NONE", "SSE", "BIDI"}


def test_config_build_default_max_llm_calls() -> None:
    """max_llm_calls None → défaut ADK (500) reflété dans le descripteur."""
    result = R.config_build(streaming_mode="NONE")
    assert result["ok"] is True
    assert result["data"]["max_llm_calls"] == 500


def test_config_build_invalid_mode() -> None:
    result = R.config_build(streaming_mode="WARP")
    assert result["ok"] is False
    assert "streaming_mode" in result["error"]


def test_config_build_response_modalities() -> None:
    result = R.config_build(streaming_mode="NONE", response_modalities=["TEXT"])
    assert result["ok"] is True
    assert result["data"]["response_modalities"] == ["TEXT"]


# --------------------------------------------------------------------------- #
# run_inspect_events (PUR)
# --------------------------------------------------------------------------- #
def _synthetic_events() -> list[dict]:
    """Liste d'événements sérialisés synthétiques pour tester le résumé."""
    return [
        {
            "author": "planner",
            "text": None,
            "function_calls": [{"name": "search", "args": {"q": "x"}}],
            "function_responses": [],
            "state_delta": {"app:hits": 1},
            "transfer_to_agent": "worker",
            "is_final": False,
            "partial": None,
        },
        {
            "author": "worker",
            "text": None,
            "function_calls": [{"name": "fetch", "args": {}}, {"name": "search", "args": {}}],
            "function_responses": [{"name": "search", "response": {"r": 1}}],
            "state_delta": {"user:seen": True},
            "transfer_to_agent": None,
            "is_final": False,
            "partial": None,
        },
        {
            "author": "worker",
            "text": "Done.",
            "function_calls": [],
            "function_responses": [],
            "state_delta": {},
            "transfer_to_agent": None,
            "is_final": True,
            "partial": None,
        },
    ]


def test_inspect_events_summary() -> None:
    result = R.inspect_events(_synthetic_events())
    assert result["ok"] is True
    data = result["data"]
    assert data["event_count"] == 3
    assert data["function_call_count"] == 3
    assert data["function_response_count"] == 1
    # Outils uniques, ordre de première apparition préservé.
    assert data["tool_names"] == ["search", "fetch"]
    assert data["transfers"] == ["worker"]
    assert data["state_delta_keys"] == ["app:hits", "user:seen"]
    assert data["final_text"] == "Done."


def test_inspect_events_empty_list() -> None:
    result = R.inspect_events([])
    assert result["ok"] is True
    assert result["data"]["event_count"] == 0
    assert result["data"]["final_text"] is None
    assert result["data"]["tool_names"] == []


def test_inspect_events_rejects_non_list() -> None:
    result = R.inspect_events({"not": "a list"})  # type: ignore[arg-type]
    assert result["ok"] is False


def test_inspect_events_rejects_non_dict_item() -> None:
    result = R.inspect_events(["not a dict"])  # type: ignore[list-item]
    assert result["ok"] is False


async def test_inspect_events_consumes_run_agent_output(tmp_path: Path) -> None:
    """Bout-en-bout : la sortie events de run_agent est résumable par run_inspect_events."""
    path = _scaffold_tool_agent(tmp_path, "calc")
    run = await R.agent(path=path, app_name="calc", user_id="u1", session_id="s1", message="2+3?")
    assert run["ok"] is True
    summary = R.inspect_events(run["data"]["events"])
    assert summary["ok"] is True
    assert "add_numbers" in summary["data"]["tool_names"]
    assert summary["data"]["final_text"] == "The sum is 5."


# --------------------------------------------------------------------------- #
# In-memory fastmcp.Client read-through (exposed names + double-prefix guard)
# --------------------------------------------------------------------------- #
async def test_client_exposed_names_and_run_agent(tmp_path: Path) -> None:
    """Les outils sont exposés run_<bare> (pas de double-préfixe) et run_agent s'exécute."""
    path = _scaffold_fake_agent(tmp_path, "myapp", answer="client says hi")
    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {"run_agent", "run_stream", "run_live", "run_config_build", "run_inspect_events"}
        assert expected <= names
        assert not any(n.startswith("run_run_") for n in names)

        res = await client.call_tool(
            "run_agent",
            {
                "path": path,
                "app_name": "myapp",
                "user_id": "u1",
                "session_id": "s1",
                "message": "hi",
            },
        )
        assert res.data["ok"] is True
        assert res.data["data"]["final_text"] == "client says hi"


async def test_client_run_config_build(tmp_path: Path) -> None:
    """run_config_build accessible via le client (validation pure)."""
    mcp = build_server()
    async with Client(mcp) as client:
        res = await client.call_tool("run_config_build", {"streaming_mode": "SSE"})
        assert res.data["ok"] is True
        assert res.data["data"]["streaming_mode"] == "SSE"
