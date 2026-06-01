"""Domaine `observability` : OpenTelemetry pour les agents ADK (P4c).

Sous-serveur FastMCP monté sous ``namespace="observability"`` → outils exposés
``observability_<nom>``. Noms BARE (``enable_otel``, ``cloud_trace``, ``third_party``,
``trace_view``). Chaque outil renvoie l'enveloppe ``{ok, data, error}``.

**Honnêteté sur les recouvrements** (cf. ``docs/adk-api-notes/safety-observability.md``) — ce
domaine NE duplique PAS la logique des domaines ``deploy``/``dev`` :

1. :func:`enable_otel` — génère un ``<app_dir>/<app>/otel_setup.py`` RÉEL (ast-valide,
   ruff/isort-clean) qui configure un exportateur OpenTelemetry (``console`` toujours dispo ;
   ``otlp`` importé paresseusement — paquet séparé) câblé sur le provider GLOBAL qu'ADK utilise.
2. :func:`cloud_trace` — renvoie le vrai flag CLI ``--trace_to_cloud`` (confirmé en P4a sur
   ``deploy cloud_run``/``agent_engine``/``gke`` + ``web``/``api_server``) et **référence** l'outil
   ``deploy``/``dev`` qui l'applique réellement (aucun flag émis ici). ``--otel_to_cloud`` est
   exposé pour information mais marqué **manuel uniquement** : aucun outil du toolkit ne l'applique
   (on ne prétend pas appliquer un flag qu'on n'émet pas).
3. :func:`third_party` — renvoie les variables d'env OTLP + un snippet de setup pour un backend
   tiers (phoenix/arize/weave/signoz/otlp).
4. :func:`trace_view` — **délègue** au même registre de process que ``dev_web`` (l'UI Web d'ADK
   héberge la vue des traces) ; un vrai boot est protégé par un flag d'env (comme les tests de
   ``dev``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..envelope import err, ok
from ..project_model import is_identifier
from . import dev, observability_setup

observability_server: FastMCP = FastMCP("observability")

#: app_name = identifiant de package Python (nom de dossier ET de module).
_APP_NAME_ERR = (
    "app_name invalide : attendu un identifiant Python "
    "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
)

#: Exportateurs OTel supportés par :func:`enable_otel`.
_EXPORTERS: frozenset[str] = frozenset({"console", "otlp"})

#: Nom du fichier de setup OTel généré (dans le dossier de l'app).
_OTEL_FILE = "otel_setup.py"

#: Le vrai flag CLI ADK 2.1.0 activant Cloud Trace (confirmé sur deploy/web/api_server).
_CLOUD_TRACE_FLAG = "--trace_to_cloud"

#: Cibles ``cloud_trace`` reconnues -> outil du toolkit qui applique réellement le flag.
_CLOUD_TRACE_TARGETS: dict[str, str] = {
    "cloud_run": "deploy_cloud_run(enable_cloud_trace=True)",
    "agent_engine": "deploy_agent_engine",
    "gke": "deploy_gke",
    "web": "dev_web",
    "api_server": "dev_api_server",
}

#: Backends tiers OTLP supportés par :func:`third_party` + leur endpoint OTLP par défaut (None si
#: aucun défaut universel : l'utilisateur DOIT fournir ``endpoint``). Tous parlent OTLP/HTTP.
_THIRD_PARTY: dict[str, str | None] = {
    "phoenix": "http://localhost:6006/v1/traces",
    "arize": "https://otlp.arize.com/v1/traces",
    "weave": None,  # W&B Weave : endpoint propre au projet (https://trace.wandb.ai/...).
    "signoz": "http://localhost:4318/v1/traces",
    "otlp": None,  # OTLP générique : endpoint requis.
}


# --------------------------------------------------------------------------- #
# Outil 1 — enable_otel (génère otel_setup.py)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
def enable_otel(
    path: str,
    app_name: str,
    exporter: str = "console",
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Génère ``<app_dir>/<app>/otel_setup.py`` configurant un exportateur OpenTelemetry.

    ``exporter`` ∈ {``console``, ``otlp``}. Le fichier définit ``setup_otel()`` qui construit un
    ``TracerProvider`` (avec un ``Resource`` ``service.name=<app>``), y ajoute un
    ``BatchSpanProcessor(exporter)`` et l'installe comme provider GLOBAL (``trace.set_tracer_
    provider``) — c'est CE provider que la télémétrie d'ADK utilise (cf. notes). L'utilisateur
    appelle ``setup_otel()`` au démarrage (avant de lancer l'agent).

    - ``console`` : ``ConsoleSpanExporter`` (paquet OTel SDK de base — toujours disponible).
    - ``otlp`` : ``OTLPSpanExporter`` (HTTP) importé **paresseusement** (paquet séparé
      ``opentelemetry-exporter-otlp`` — le fichier généré documente l'installation). ``endpoint``
      est alors requis (ex. ``http://localhost:4318/v1/traces``).

    Le code généré est ast-valide + ruff/isort-clean (le toolkit n'importe jamais OTLP lui-même).
    """
    if not is_identifier(app_name):
        return err(_APP_NAME_ERR)
    if exporter not in _EXPORTERS:
        return err(f"exporter inconnu : {exporter!r}. Connus : {', '.join(sorted(_EXPORTERS))}.")
    if exporter == "otlp" and not (endpoint or "").strip():
        return err(
            "exporter 'otlp' : 'endpoint' est requis (ex. 'http://localhost:4318/v1/traces')."
        )

    app_dir = Path(path) / app_name
    if not (app_dir / "agent.py").is_file():
        agent_py = app_dir / "agent.py"
        return err(f"Dossier d'app introuvable : {agent_py}. Scaffolde d'abord (project_create).")

    from ..workspace import Workspace

    ws = Workspace(app_dir)
    source = observability_setup.render_otel_setup(
        app_name=app_name, exporter=exporter, endpoint=endpoint
    )
    changed = ws.write(_OTEL_FILE, source)
    return ok(
        {
            "app_name": app_name,
            "exporter": exporter,
            "endpoint": endpoint,
            "otel_setup": str(ws.path(_OTEL_FILE)),
            "usage": "import otel_setup; otel_setup.setup_otel()  # au démarrage, avant l'agent",
            "changed": changed,
            "notes": _otel_notes(exporter),
        }
    )


def _otel_notes(exporter: str) -> list[str]:
    """Notes orientées action selon l'exportateur choisi."""
    notes = [
        "Appelle setup_otel() au démarrage (avant de lancer l'agent) : il installe le "
        "TracerProvider GLOBAL qu'ADK utilise pour ses spans.",
    ]
    if exporter == "otlp":
        notes.append(
            "OTLP nécessite le paquet séparé : pip install opentelemetry-exporter-otlp "
            "(non tiré par google-adk)."
        )
    return notes


# --------------------------------------------------------------------------- #
# Outil 2 — cloud_trace (renvoie le vrai flag + référence l'outil deploy/dev)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
def cloud_trace(target: str) -> dict[str, Any]:
    """Renvoie le flag CLI activant Cloud Trace pour ``target`` + l'outil qui l'applique.

    ``target`` ∈ {cloud_run, agent_engine, gke, web, api_server}. Le flag réellement APPLIQUÉ par
    les outils ``deploy_*`` / ``dev_*`` du toolkit (confirmé en P4a) est ``--trace_to_cloud``.
    **Cet outil n'exécute rien** : il référence l'outil ``deploy_*`` / ``dev_*`` qui émet ce flag
    (le domaine ``deploy`` valide chaque flag émis contre le vrai ``--help`` ; ``deploy_cloud_run``
    expose le paramètre ``enable_cloud_trace`` qui mappe sur ``--trace_to_cloud``). Évite toute
    duplication de la logique de déploiement.

    Le retour distingue clairement les deux flags :
    - ``flag`` = ``--trace_to_cloud`` : le flag que CE toolkit applique via ``apply_with``.
    - ``otel_flag`` = ``--otel_to_cloud`` : flag ADK **manuel uniquement** — AUCUN outil du toolkit
      ne l'applique automatiquement (il faut le passer soi-même à la CLI ``adk``). On ne prétend
      donc pas que le toolkit l'émet.
    """
    if target not in _CLOUD_TRACE_TARGETS:
        return err(
            f"target inconnu : {target!r}. Connus : {', '.join(sorted(_CLOUD_TRACE_TARGETS))}."
        )
    return ok(
        {
            "target": target,
            "flag": _CLOUD_TRACE_FLAG,
            "otel_flag": "--otel_to_cloud",
            "otel_flag_note": (
                "manuel uniquement — non appliqué automatiquement par ce toolkit ; à passer "
                "soi-même à la CLI 'adk' pour exporter aussi les données OTel vers Cloud Trace."
            ),
            "apply_with": _CLOUD_TRACE_TARGETS[target],
            "guidance": (
                f"Active Cloud Trace en passant {_CLOUD_TRACE_FLAG} à '{target}'. "
                f"Utilise l'outil du toolkit '{_CLOUD_TRACE_TARGETS[target]}' qui l'émet et le "
                f"valide (ne réimplémente pas la commande ici). Le flag '{_CLOUD_TRACE_FLAG}' est "
                "le SEUL appliqué par ce toolkit ; '--otel_to_cloud' existe côté ADK mais n'est "
                "PAS auto-appliqué par le toolkit (manuel uniquement)."
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Outil 3 — third_party (env OTLP + snippet pour un backend tiers)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
def third_party(
    provider: str,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Émet la config OTLP (variables d'env + snippet) pour un backend d'observabilité tiers.

    ``provider`` ∈ {phoenix, arize, weave, signoz, otlp}. Tous ces backends ingèrent l'OTLP
    standard : on renvoie les variables d'env OTel canoniques (``OTEL_EXPORTER_OTLP_ENDPOINT`` +
    ``OTEL_EXPORTER_OTLP_HEADERS`` si ``headers``) et un snippet de setup pointant vers
    ``observability_enable_otel(exporter='otlp', endpoint=...)``. Un ``endpoint`` par défaut est
    fourni quand le backend en a un (phoenix/arize/signoz) ; pour ``weave``/``otlp`` il est requis.

    Aucun secret n'est écrit en dur : les ``headers`` (ex. clé d'API) sont émis comme **valeurs
    d'environnement à définir** (le snippet lit ``os.environ``), pas figés dans le code.
    """
    if provider not in _THIRD_PARTY:
        return err(f"provider inconnu : {provider!r}. Connus : {', '.join(sorted(_THIRD_PARTY))}.")

    default_endpoint = _THIRD_PARTY[provider]
    resolved = (endpoint or "").strip() or default_endpoint
    if not resolved:
        return err(
            f"provider {provider!r} : 'endpoint' est requis (pas de défaut universel). "
            "Fournis l'URL OTLP/HTTP du collecteur (ex. 'https://host/v1/traces')."
        )

    env: dict[str, str] = {"OTEL_EXPORTER_OTLP_ENDPOINT": resolved}
    if headers:
        # Format OTLP : "k1=v1,k2=v2". On documente que les valeurs sensibles viennent de l'env.
        env["OTEL_EXPORTER_OTLP_HEADERS"] = ",".join(f"{k}={v}" for k, v in headers.items())

    snippet = (
        "# 1) Exporte les variables d'env ci-dessus (ne mets PAS de secret en dur).\n"
        "# 2) Génère le setup OTLP puis appelle-le au démarrage :\n"
        f"#    observability_enable_otel(path, app_name, exporter='otlp', endpoint='{resolved}')\n"
        "#    import otel_setup; otel_setup.setup_otel()"
    )
    return ok(
        {
            "provider": provider,
            "endpoint": resolved,
            "env": env,
            "exporter": "otlp",
            "setup_snippet": snippet,
            "note": (
                f"{provider} ingère l'OTLP standard. Installe 'opentelemetry-exporter-otlp' puis "
                "utilise observability_enable_otel(exporter='otlp', endpoint=...)."
            ),
        }
    )


# --------------------------------------------------------------------------- #
# Outil 4 — trace_view (délègue à dev_web : l'UI ADK héberge la vue des traces)
# --------------------------------------------------------------------------- #
@observability_server.tool(tags={"observability"})
async def trace_view(path: str, app_name: str | None = None, port: int = 8000) -> dict[str, Any]:
    """Lance l'UI Web d'ADK (qui héberge la **vue des traces**) en **déléguant** à ``dev_web``.

    L'UI de dev d'ADK (``adk web``) inclut un onglet « Trace » visualisant les spans d'une
    exécution. Plutôt que de réimplémenter un serveur, cet outil délègue au **même** chemin de
    registre de process que ``dev_web`` (``adk_cli`` ; cf. domaine ``dev``). Le serveur démarré
    est piloté via ``dev_status`` / ``dev_logs`` / ``dev_stop`` (mêmes clés).

    Renvoie ``{key, pid, port, url, trace_url, ...}`` (``trace_url`` = page d'accueil de l'UI où la
    vue Trace est accessible). Le boot réel n'a lieu que si un dossier d'agents valide existe (la
    délégation à ``dev.web`` valide tout) ; les tests gardent un vrai démarrage derrière un flag
    d'env (comme ``dev``).
    """
    result = await dev.web(path=path, app_name=app_name, port=port)
    if not result["ok"]:
        return result
    data = dict(result["data"])
    data["delegated_to"] = "dev_web"
    data["trace_url"] = data.get("url")
    data["guidance"] = (
        "L'UI Web d'ADK est démarrée ; ouvre l'URL et sélectionne l'onglet « Trace » d'une "
        "session pour visualiser les spans. Pilote via dev_status/dev_logs/dev_stop (même clé)."
    )
    return ok(data)
