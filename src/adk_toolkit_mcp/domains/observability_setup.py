"""Génération de ``otel_setup.py`` : configuration OpenTelemetry réelle (ast-valide, P4c).

Module **pur** (aucune dépendance à OpenTelemetry au chargement : on ne produit qu'une *chaîne
source*). Le domaine ``observability`` (``observability_enable_otel``) l'utilise pour écrire
``<app_dir>/<app>/otel_setup.py``. Le fichier généré définit ``setup_otel()`` qui installe un
``TracerProvider`` GLOBAL (celui que la télémétrie d'ADK utilise pour ses spans), avec :

- un ``Resource`` portant ``service.name=<app>`` ;
- un ``BatchSpanProcessor`` câblé sur l'exportateur choisi :
  - ``console`` : ``ConsoleSpanExporter`` (paquet OTel SDK de base — toujours disponible) ;
  - ``otlp`` : ``OTLPSpanExporter`` (HTTP) importé **paresseusement** DANS ``setup_otel`` (paquet
    séparé ``opentelemetry-exporter-otlp`` — un ``ImportError`` clair guide l'installation).

Les imports SDK de base (``trace``, ``TracerProvider``, ``BatchSpanProcessor``,
``ConsoleSpanExporter``, ``Resource``, ``SERVICE_NAME``) sont confirmés présents avec
``google-adk`` (cf. ``docs/adk-api-notes/safety-observability.md``). Le code généré est
**ast.parse + ruff format + isort clean** (vérifié en test).
"""

from __future__ import annotations

#: En-tête du module généré.
_HEADER = (
    '"""Configuration OpenTelemetry générée par adk-toolkit-mcp (observability_enable_otel).\n\n'
    "Appelle ``setup_otel()`` au démarrage (avant de lancer l'agent) : il installe le\n"
    "``TracerProvider`` GLOBAL qu'ADK utilise pour émettre ses spans.\n"
    '"""\n\n'
)

#: Imports communs (SDK OTel de base — disponibles avec google-adk).
_BASE_IMPORTS = (
    "from opentelemetry import trace\n"
    "from opentelemetry.sdk.resources import SERVICE_NAME, Resource\n"
    "from opentelemetry.sdk.trace import TracerProvider\n"
    "from opentelemetry.sdk.trace.export import BatchSpanProcessor\n"
)


def _py_str(value: str) -> str:
    """Littéral chaîne Python stable pour ``ruff format`` (guillemets doubles par défaut)."""
    has_double = '"' in value
    has_single = "'" in value
    if has_double and not has_single:
        return "'" + value.replace("\\", "\\\\") + "'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _console_exporter_lines() -> tuple[str, str]:
    """Renvoie ``(import_supplémentaire, lignes_corps)`` pour l'exportateur console.

    L'import ``ConsoleSpanExporter`` est groupé avec les autres imports SDK (top-level) ; le corps
    instancie simplement l'exportateur.
    """
    extra_import = "from opentelemetry.sdk.trace.export import ConsoleSpanExporter\n"
    body = "    exporter = ConsoleSpanExporter()\n"
    return extra_import, body


def _otlp_exporter_lines(endpoint: str) -> tuple[str, str]:
    """Renvoie ``(import_supplémentaire, lignes_corps)`` pour l'exportateur OTLP/HTTP.

    L'import est fait **paresseusement DANS** ``setup_otel`` (paquet séparé) avec un message
    d'erreur actionnable ; ``endpoint`` est injecté comme littéral. Aucun import top-level (le
    paquet OTLP peut être absent — le fichier doit rester importable).
    """
    extra_import = ""  # import paresseux dans le corps (paquet séparé, peut être absent)
    body = (
        "    try:\n"
        "        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (\n"
        "            OTLPSpanExporter,\n"
        "        )\n"
        "    except ImportError as exc:  # paquet séparé\n"
        "        raise ImportError(\n"
        '            "OTLP exporter manquant : pip install opentelemetry-exporter-otlp"\n'
        "        ) from exc\n"
        f"    exporter = OTLPSpanExporter(endpoint={_py_str(endpoint)})\n"
    )
    return extra_import, body


def render_otel_setup(*, app_name: str, exporter: str, endpoint: str | None) -> str:
    """Produit la source complète de ``otel_setup.py`` pour l'exportateur choisi.

    ``exporter`` ∈ {``console``, ``otlp``}. Pour ``otlp``, ``endpoint`` est requis (validé en
    amont par le domaine). Sortie ast-valide + ruff/isort-clean.
    """
    if exporter == "otlp":
        extra_import, exporter_body = _otlp_exporter_lines(endpoint or "")
    else:
        extra_import, exporter_body = _console_exporter_lines()

    # Section d'imports : base + éventuel import console (groupés/triés façon isort). Tous sous
    # ``opentelemetry.*`` -> un seul bloc third-party ; les noms d'un même module triés à la main
    # ici (SERVICE_NAME avant Resource ; voir _BASE_IMPORTS). On insère ConsoleSpanExporter dans
    # la ligne ``...export import BatchSpanProcessor`` pour rester isort-clean.
    imports = _merge_imports(extra_import)

    func = (
        "def setup_otel() -> TracerProvider:\n"
        '    """Installe un TracerProvider global exportant les spans ADK ; le renvoie."""\n'
        f"    resource = Resource.create({{SERVICE_NAME: {_py_str(app_name)}}})\n"
        "    provider = TracerProvider(resource=resource)\n"
        f"{exporter_body}"
        "    provider.add_span_processor(BatchSpanProcessor(exporter))\n"
        "    trace.set_tracer_provider(provider)\n"
        "    return provider\n"
    )
    return _HEADER + imports + "\n\n" + func


def _merge_imports(extra_console_import: str) -> str:
    """Fusionne les imports SDK (+ éventuel ConsoleSpanExporter) façon isort (stable pour ruff).

    ``BatchSpanProcessor`` et ``ConsoleSpanExporter`` partagent le module
    ``opentelemetry.sdk.trace.export`` : on les met sur une seule ligne, noms triés
    (``BatchSpanProcessor`` < ``ConsoleSpanExporter``). Les autres lignes de :data:`_BASE_IMPORTS`
    sont conservées telles quelles (déjà triées par module + par nom).
    """
    if not extra_console_import:
        return _BASE_IMPORTS
    # Remplace la ligne export de base par une ligne fusionnée triée.
    base = _BASE_IMPORTS.replace(
        "from opentelemetry.sdk.trace.export import BatchSpanProcessor\n",
        "from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter\n",
    )
    return base
