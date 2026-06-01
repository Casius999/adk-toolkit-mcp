"""Tests du skill compagnon ``adk-toolkit`` (P5).

Vérifie que le skill livré dans ``skill/`` est bien formé :

- ``skill/SKILL.md`` existe et porte un frontmatter YAML valide avec
  ``name == "adk-toolkit"`` et une ``description`` non vide ;
- chaque fichier de référence mentionné par ``SKILL.md`` (``references/NN-*.md``)
  existe réellement sur disque.

Le frontmatter est parsé de façon **minimale** (sans dépendance YAML) : on lit le bloc
entre les deux délimiteurs ``---`` en tête de fichier et on en extrait les clés simples
``clé: valeur`` (suffisant pour ``name`` ; ``description`` peut être un scalaire de bloc
``>-`` multi-lignes, géré explicitement).
"""

from __future__ import annotations

import re
from pathlib import Path

#: Racine du dépôt = deux niveaux au-dessus de ``tests/unit/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Dossier du skill livré dans le dépôt.
_SKILL_DIR = _REPO_ROOT / "skill"
_SKILL_MD = _SKILL_DIR / "SKILL.md"


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Sépare le frontmatter YAML (entre les deux ``---``) du corps.

    Renvoie ``(frontmatter, body)``. Lève ``AssertionError`` si le fichier ne commence
    pas par un bloc ``---`` ... ``---``.
    """
    assert text.startswith("---"), "SKILL.md doit commencer par un frontmatter '---'."
    # Le corps suit le second '---' en début de ligne.
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    assert match is not None, "Frontmatter YAML mal délimité (attendu '---' ... '---')."
    return match.group(1), match.group(2)


def _parse_frontmatter(front: str) -> dict[str, str]:
    """Parse minimal du frontmatter : clés ``clé: valeur`` + scalaire de bloc ``>-``.

    Suffisant pour ce skill : ``name`` est une valeur simple ; ``description`` est un
    scalaire de bloc ``>-`` (lignes indentées suivantes jointes par des espaces). On
    n'a PAS besoin d'un parseur YAML complet (pas de dépendance ajoutée).
    """
    result: dict[str, str] = {}
    lines = front.splitlines()
    index = 0
    key_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
    while index < len(lines):
        line = lines[index]
        key_match = key_re.match(line)
        if key_match is None:
            index += 1
            continue
        key, value = key_match.group(1), key_match.group(2).strip()
        if value in (">-", ">", "|", "|-"):
            # Scalaire de bloc : agrège les lignes indentées suivantes.
            block: list[str] = []
            index += 1
            while index < len(lines):
                nxt = lines[index]
                if not (nxt.startswith((" ", "\t")) or not nxt.strip()):
                    break
                block.append(nxt.strip())
                index += 1
            result[key] = " ".join(part for part in block if part).strip()
            continue
        result[key] = value
        index += 1
    return result


def test_skill_md_exists() -> None:
    """``skill/SKILL.md`` doit exister."""
    assert _SKILL_MD.is_file(), f"SKILL.md introuvable : {_SKILL_MD}"


def test_skill_frontmatter_name_and_description() -> None:
    """Le frontmatter porte ``name == 'adk-toolkit'`` et une ``description`` non vide."""
    front, _ = _split_frontmatter(_SKILL_MD.read_text(encoding="utf-8"))
    meta = _parse_frontmatter(front)

    name = meta.get("name")
    assert name == "adk-toolkit", f"name attendu 'adk-toolkit', obtenu {name!r}."
    description = meta.get("description", "")
    assert description.strip(), "La description du frontmatter ne doit pas être vide."
    # La description est la surface de déclenchement : on exige un minimum de substance.
    assert len(description) >= 50, "La description doit être riche en déclencheurs (>= 50 car.)."


def test_referenced_reference_files_exist() -> None:
    """Chaque ``references/NN-*.md`` cité par SKILL.md existe sur disque."""
    text = _SKILL_MD.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"references/[0-9A-Za-z._-]+\.md", text)))
    assert referenced, "SKILL.md devrait référencer au moins un fichier references/*.md."

    missing = [rel for rel in referenced if not (_SKILL_DIR / rel).is_file()]
    assert not missing, f"Fichiers de référence cités mais absents : {missing}"


def test_expected_reference_set_present() -> None:
    """Les 14 références canoniques (00..13) sont présentes dans ``skill/references/``.

    Garde-fou contre une livraison partielle : la table de routage du SKILL.md route vers
    ces 14 fichiers (cf. spec §6 ; ``13-tool-catalog.md`` est le pont tâche→outil).
    """
    refs_dir = _SKILL_DIR / "references"
    expected = {
        "00-mental-model.md",
        "01-agent-types.md",
        "02-tools.md",
        "03-models.md",
        "04-sessions-state.md",
        "05-memory-artifacts.md",
        "06-runtime.md",
        "07-eval.md",
        "08-deploy.md",
        "09-a2a.md",
        "10-observability.md",
        "11-safety.md",
        "12-troubleshooting.md",
        "13-tool-catalog.md",
    }
    present = {p.name for p in refs_dir.glob("*.md")}
    missing = expected - present
    assert not missing, f"Références manquantes : {sorted(missing)}"
