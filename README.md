# adk-toolkit-mcp

Serveur MCP exhaustif enveloppant **Google ADK** (Agent Development Kit, `google-adk` 2.x) :
15 sous-serveurs / **81 outils** couvrant agents, outils, modèles, sessions, mémoire,
artefacts, runtime, évaluation, déploiement, A2A, observabilité, sécurité — plus **5 prompts
de workflow** et un **Code Mode** opt-in.

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

## Prompts de workflow
Cinq prompts MCP (`get_prompt`) cadrent les tâches courantes en citant les vrais outils, dans
l'ordre d'appel :

| Prompt | Arguments | Couvre |
|--------|-----------|--------|
| `scaffold_multi_agent` | `goal` | `project_create` → `agents_create_llm` → `agents_create_sequential`/`parallel`/`loop` → `agents_compose`/`set_root` → `models_*` → `run_agent` |
| `add_guardrail` | `agent`, `concern` | callback par-agent (`safety_add_callback`) vs plugin global (`safety_add_plugin`) + `safety_settings` |
| `write_evalset` | `agent` | `eval_create_set` → `eval_set_criteria` → `eval_run` → `eval_report` (métriques offline) |
| `deploy_checklist` | `target` | `deploy_preflight` → `deploy_containerize` → `deploy_agent_engine`/`cloud_run`/`gke` → `deploy_status` |
| `debug_agent` | `symptom` | `run_inspect_events`, `run_stream`, `agents_get`/`list`, `tools_list` + pièges connus |

## Code Mode (opt-in — efficacité de tokens)
Les 81 outils sont **tagués par domaine** (`agents`, `deploy`, …). Par défaut, tous sont exposés
par leur nom (`project_create`, `run_agent`, …). Pour effondrer le catalogue en une petite
surface de découverte + exécution (gros gain de tokens), active le **Code Mode** — le vrai
transform `CodeMode` de FastMCP ≥3.1 (présent en 3.3.1) :

```bash
ADK_TOOLKIT_CODE_MODE=1 uv run adk-toolkit-mcp
```

La surface passe alors des 81 outils nommés à `search` / `get_schema` / `tags` / `execute` : le
client cherche les outils par mot-clé ou par tag de domaine (`tags`/`search(tags=[...])`),
récupère leurs schémas (`get_schema`), puis enchaîne les appels via du code dans `execute`.
En API : `build_server(code_mode=True)`.

> **Note honnête sur `execute`.** Les outils de découverte (`search`/`get_schema`/`tags`)
> fonctionnent tels quels. L'outil `execute` exécute du code dans un bac à sable
> (`MontySandboxProvider`) qui nécessite le paquet optionnel `pydantic-monty` (extra
> `fastmcp[code-mode]`), **non installé par défaut** ici : sans lui, `execute` lève un
> `ImportError` explicite (les autres outils Code Mode restent utilisables). Installe-le si tu
> veux l'exécution de code côté serveur :
> ```bash
> uv pip install 'fastmcp[code-mode]'
> ```
> Détails et introspection : `docs/adk-api-notes/fastmcp-codemode.md`.

## Skill compagnon
Voir `skill/` (installé dans `~/.claude/skills/adk-toolkit/`).
