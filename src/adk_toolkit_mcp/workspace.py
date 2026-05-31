from __future__ import annotations

import re
from pathlib import Path

_ROOT_AGENT_RE = re.compile(r"^\s*root_agent\s*=", re.MULTILINE)


class Workspace:
    """Accès idempotent aux fichiers d'un projet ADK (source de vérité code-first)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, relative: str) -> Path:
        return self.root / relative

    def write(self, relative: str, content: str) -> bool:
        """Écrit `content`. Renvoie True si créé/modifié, False si inchangé."""
        target = self.path(relative)
        if target.exists() and target.read_text(encoding="utf-8") == content:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return True

    def read(self, relative: str) -> str:
        return self.path(relative).read_text(encoding="utf-8")

    def exists(self, relative: str) -> bool:
        return self.path(relative).exists()

    def has_root_agent(self, relative: str = "agent.py") -> bool:
        """Détecte l'assignation `root_agent = ...` générée par le toolkit (garde d'idempotence).

        HEURISTIQUE ancrée en début de ligne (regex `^\\s*root_agent\\s*=`), PAS un parseur
        Python. Elle peut donner un faux positif sur une assignation commentée
        (`# root_agent = ...`) et un faux négatif sur des formes annotées ou dynamiques
        (`root_agent: Agent = ...`, `setattr(mod, "root_agent", ...)`).
        """
        if not self.exists(relative):
            return False
        return bool(_ROOT_AGENT_RE.search(self.read(relative)))
