"""Génération de ``plugins.py`` : sous-classes ``BasePlugin`` réelles (politiques globales, P4c).

Module **pur** (aucune dépendance à google-adk : on ne produit qu'une *chaîne source*). Le
domaine ``safety`` (``safety_add_plugin``) l'utilise pour (ré)générer
``<app_dir>/<app>/plugins.py`` à partir d'un manifeste de plugins, puis ``run_core.build_runner``
importe ces instances et les passe au ``Runner`` (via ``App``).

Deux politiques globales concrètes + fonctionnelles (cf.
``docs/adk-api-notes/safety-observability.md`` pour les signatures de hooks ``BasePlugin``
confirmées — hooks **keyword-only**, async) :

- ``logging`` : ``on_event_callback`` enregistre l'auteur de chaque évènement dans une liste
  module-level ``<var>_events`` (inspectable hors-ligne, ce qui permet la PREUVE fonctionnelle)
  et journalise via le module ``logging``.
- ``tool_denylist`` : ``before_tool_callback`` court-circuite tout appel d'outil dont le nom est
  dans une denylist (renvoie un ``dict`` → l'outil n'est jamais exécuté).

Le code généré est **ast.parse + ruff format + isort clean** (vérifié en test).
"""

from __future__ import annotations

import re
from typing import Any

#: En-tête du module généré (ne pas éditer à la main : régénéré depuis le manifeste runtime).
_HEADER = (
    '"""Généré par adk-toolkit-mcp (safety_add_plugin). NE PAS éditer à la main.\n\n'
    "Plugins BasePlugin déclarés au niveau module ; le manifeste vit dans "
    "`.adk_toolkit/runtime.json`\n"
    "(clé `plugins`). `run_core.build_runner` importe ces instances et les passe au Runner.\n"
    '"""\n\n'
)

#: Classe PascalCase de plugin pour une variable (``logging_plugin`` -> ``LoggingPluginPlugin``).
_CLASS_SUFFIX = "Plugin"


def _py_str(value: str) -> str:
    """Littéral chaîne Python stable pour ``ruff format`` (guillemets doubles par défaut)."""
    has_double = '"' in value
    has_single = "'" in value
    if has_double and not has_single:
        return "'" + value.replace("\\", "\\\\") + "'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _py_str_list(values: list[str]) -> str:
    """Rend ``["a", "b"]`` (littéral liste de chaînes) inline et ruff-stable."""
    return "[" + ", ".join(_py_str(v) for v in values) + "]"


def _class_name(var: str) -> str:
    """Nom de classe PascalCase dérivé du nom de variable (``my_plugin`` -> ``MyPluginPlugin``)."""
    pascal = "".join(part.capitalize() for part in var.split("_") if part)
    if not pascal:
        pascal = "Custom"
    return f"{pascal}{_CLASS_SUFFIX}"


def _render_logging_plugin(var: str, name: str) -> list[str]:
    """Rend les blocs top-level d'un plugin ``logging`` (``on_event_callback``).

    Renvoie ``[events_assignment, class_block, instance_block]``. Une liste module-level
    ``<var>_events`` accumule l'auteur de chaque évènement (preuve fonctionnelle hors-ligne) ; le
    hook journalise aussi via ``logging.getLogger(__name__)``.
    """
    cls = _class_name(var)
    events_var = f"{var}_events"
    events_block = f"{events_var}: list[str] = []\n"
    class_block = (
        f"class {cls}(BasePlugin):\n"
        '    """Plugin de journalisation : enregistre chaque évènement (auteur) et le logge."""\n'
        "\n"
        "    async def on_event_callback(self, *, invocation_context, event):\n"
        f"        {events_var}.append(event.author)\n"
        '        logging.getLogger(__name__).info("event from %s", event.author)\n'
        "        return None\n"
    )
    instance_block = f"{var} = {cls}(name={_py_str(name)})\n"
    return [events_block, class_block, instance_block]


def _render_tool_denylist_plugin(var: str, name: str, denylist: list[str]) -> list[str]:
    """Rend les blocs top-level d'un plugin ``tool_denylist`` (``before_tool_callback`` global).

    Court-circuite tout appel d'outil dont le nom est dans ``denylist`` (renvoie un ``dict``).
    Renvoie ``[denylist_assignment, class_block, instance_block]``.
    """
    cls = _class_name(var)
    denylist_var = f"{var}_denylist"
    denylist_block = f"{denylist_var} = {_py_str_list(denylist)}\n"
    blocked = _py_str("Tool blocked by global safety plugin.")
    class_block = (
        f"class {cls}(BasePlugin):\n"
        '    """Plugin denylist global : court-circuite tout outil dont le nom est interdit."""\n'
        "\n"
        "    async def before_tool_callback(self, *, tool, tool_args, tool_context):\n"
        f"        if tool.name in {denylist_var}:\n"
        f"            return {{{_py_str('error')}: {blocked}}}\n"
        "        return None\n"
    )
    instance_block = f"{var} = {cls}(name={_py_str(name)})\n"
    return [denylist_block, class_block, instance_block]


def _render_one(payload: dict[str, Any]) -> list[str]:
    """Aiguille vers le renderer du bon ``kind`` ; renvoie la liste de blocs top-level."""
    var = str(payload["var"])
    name = str(payload.get("name") or var)
    kind = str(payload["kind"])
    if kind == "logging":
        return _render_logging_plugin(var, name)
    if kind == "tool_denylist":
        denylist = [str(x) for x in payload.get("denylist") or []]
        return _render_tool_denylist_plugin(var, name, denylist)
    raise ValueError(f"Genre de plugin non rendu : {kind!r}")  # pragma: no cover


def _starts_class_or_def(block: str) -> bool:
    """Vrai si le bloc top-level débute par ``class`` ou ``def`` (règle d'espacement E302/E305)."""
    return block.startswith("class ") or block.startswith("def ")


def _space_blocks(blocks: list[str]) -> str:
    """Assemble des blocs top-level avec l'espacement exact de ``ruff format`` (E302/E303/E305).

    2 lignes vides autour d'un bloc ``class``/``def`` ; 1 ligne vide entre deux assignations.
    Chaque bloc finit déjà par exactement un ``\\n``.
    """
    parts: list[str] = []
    for i, block in enumerate(blocks):
        parts.append(block)
        if i < len(blocks) - 1:
            nxt = blocks[i + 1]
            parts.append(
                "\n\n" if _starts_class_or_def(block) or _starts_class_or_def(nxt) else "\n"
            )
    return "".join(parts)


def render_plugins_module(payloads: list[dict[str, Any]]) -> str:
    """Produit la source complète de ``plugins.py`` à partir des payloads de plugins.

    Chaque payload : ``{"var", "name", "kind", "denylist"?}``. Les imports (``logging`` stdlib +
    ``BasePlugin``) sont émis selon les genres présents (isort : stdlib avant third-party). Les
    blocs sont espacés façon ``ruff format`` (2 lignes vides autour des classes top-level).
    """
    if not payloads:
        # Aucun plugin : module minimal valide (cas dégénéré ; le domaine n'appelle pas ainsi).
        return _HEADER + "# Aucun plugin déclaré.\n"

    kinds = {str(p["kind"]) for p in payloads}
    needs_logging = "logging" in kinds

    # Imports : stdlib (``import logging``) avant third-party (``from google.adk.plugins ...``).
    import_lines: list[str] = []
    if needs_logging:
        import_lines.append("import logging")
        import_lines.append("")  # ligne vide entre stdlib et third-party (isort)
    import_lines.append("from google.adk.plugins import BasePlugin")
    import_block = "\n".join(import_lines) + "\n"

    # Aplati tous les blocs (assignation + classe + instance par plugin) puis espace façon ruff.
    all_blocks: list[str] = [b for payload in payloads for b in _render_one(payload)]
    body = _space_blocks(all_blocks)

    # Le 1er bloc top-level débute toujours par une ASSIGNATION (events/denylist) : isort/ruff
    # n'exige qu'UNE ligne vide après le bloc d'imports avant une assignation (2 ne seraient
    # requises que devant une classe/def). On émet donc une seule ligne vide ici.
    return _HEADER + import_block + "\n" + body


#: Regex pour relire le ``denylist`` d'un plugin existant (best-effort, lors d'une régénération).
_DENYLIST_RE = re.compile(r"^(\w+)_denylist = (\[.*\])$", re.MULTILINE)


def parse_existing_denylists(source: str) -> dict[str, list[str]]:
    """Relit les denylists des plugins ``tool_denylist`` d'un ``plugins.py`` existant (best-effort).

    Renvoie ``{var: [tool, ...]}``. Utilisé pour préserver la config des plugins déjà présents
    lors d'une régénération complète du fichier. Tolérant : une ligne non reconnue est ignorée.
    """
    out: dict[str, list[str]] = {}
    for match in _DENYLIST_RE.finditer(source):
        var = match.group(1)
        raw_list = match.group(2)
        items = re.findall(r'"([^"]*)"|\'([^\']*)\'', raw_list)
        out[var] = [a or b for a, b in items]
    return out
