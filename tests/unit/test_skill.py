"""Tests for the companion skill ``adk-toolkit`` (P5).

Verifies that the skill shipped in ``skill/`` is well-formed:

- ``skill/SKILL.md`` exists and carries a valid YAML frontmatter with
  ``name == "adk-toolkit"`` and a non-empty ``description``;
- each reference file mentioned by ``SKILL.md`` (``references/NN-*.md``) actually exists on disk.

The frontmatter is parsed **minimally** (no YAML dependency): we read the block between the two
``---`` delimiters at the top of the file and extract the simple ``key: value`` keys (enough for
``name``; ``description`` can be a multi-line ``>-`` block scalar, handled explicitly).
"""

from __future__ import annotations

import re
from pathlib import Path

#: Repo root = two levels above ``tests/unit/``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Folder of the skill shipped in the repo.
_SKILL_DIR = _REPO_ROOT / "skill"
_SKILL_MD = _SKILL_DIR / "SKILL.md"


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split the YAML frontmatter (between the two ``---``) from the body.

    Returns ``(frontmatter, body)``. Raises ``AssertionError`` if the file does not start with a
    ``---`` ... ``---`` block.
    """
    assert text.startswith("---"), "SKILL.md must start with a '---' frontmatter."
    # The body follows the second '---' at the start of a line.
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    assert match is not None, "YAML frontmatter badly delimited (expected '---' ... '---')."
    return match.group(1), match.group(2)


def _parse_frontmatter(front: str) -> dict[str, str]:
    """Minimal parse of the frontmatter: ``key: value`` keys + ``>-`` block scalar.

    Enough for this skill: ``name`` is a simple value; ``description`` is a ``>-`` block scalar
    (following indented lines joined by spaces). We do NOT need a full YAML parser (no added
    dependency).
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
            # Block scalar: aggregate the following indented lines.
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
    """``skill/SKILL.md`` must exist."""
    assert _SKILL_MD.is_file(), f"SKILL.md not found: {_SKILL_MD}"


def test_skill_frontmatter_name_and_description() -> None:
    """The frontmatter carries ``name == 'adk-toolkit'`` and a non-empty ``description``."""
    front, _ = _split_frontmatter(_SKILL_MD.read_text(encoding="utf-8"))
    meta = _parse_frontmatter(front)

    name = meta.get("name")
    assert name == "adk-toolkit", f"expected name 'adk-toolkit', got {name!r}."
    description = meta.get("description", "")
    assert description.strip(), "The frontmatter description must not be empty."
    # The description is the triggering surface: we require a minimum of substance.
    assert len(description) >= 50, "The description must be rich in triggers (>= 50 chars)."


def test_referenced_reference_files_exist() -> None:
    """Each ``references/NN-*.md`` cited by SKILL.md exists on disk."""
    text = _SKILL_MD.read_text(encoding="utf-8")
    referenced = sorted(set(re.findall(r"references/[0-9A-Za-z._-]+\.md", text)))
    assert referenced, "SKILL.md should reference at least one references/*.md file."

    missing = [rel for rel in referenced if not (_SKILL_DIR / rel).is_file()]
    assert not missing, f"Reference files cited but absent: {missing}"


def test_expected_reference_set_present() -> None:
    """The 14 canonical references (00..13) are present in ``skill/references/``.

    Guardrail against a partial delivery: the SKILL.md routing table routes to these 14 files
    (cf. spec §6; ``13-tool-catalog.md`` is the task→tool bridge).
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
    assert not missing, f"Missing references: {sorted(missing)}"
