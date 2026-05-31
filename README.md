# adk-toolkit-mcp

Serveur MCP exhaustif enveloppant **Google ADK** (Agent Development Kit, `google-adk` 2.x) :
15 sous-serveurs / ~65 outils couvrant agents, outils, modèles, sessions, mémoire,
artefacts, runtime, évaluation, déploiement, A2A, observabilité, sécurité.

> Dépôt 100% autonome. Aucun lien avec d'autres projets locaux.

## Installation
```bash
uv venv && uv sync --extra dev
uv sync --extra all
```

## Lancer (stdio)
```bash
uv run adk-toolkit-mcp
```

## Config client MCP (Claude Code)
```json
{ "mcpServers": { "adk-toolkit": { "command": "uv", "args": ["run", "adk-toolkit-mcp"] } } }
```

## Skill compagnon
Voir `skill/` (installé dans `~/.claude/skills/adk-toolkit/`).
