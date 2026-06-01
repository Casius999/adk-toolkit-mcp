"""Prompts de workflow MCP (P6a) — guides pas-à-pas vers les bons outils ``<domaine>_*``.

Cinq prompts ``@mcp.prompt`` enregistrés par :func:`register_prompts`. Chacun renvoie une
**chaîne de workflow actionnable** qui cite les VRAIS noms d'outils exposés par ce serveur
(``project_create``, ``agents_create_llm``, ``run_agent``, …) dans l'ordre où on les appelle.
Ce sont des *templates* déterministes (aucune I/O, aucun import ADK) : un client MCP les rend
via ``get_prompt(<name>, {args})`` pour cadrer une tâche avant d'appeler les outils.

Tous les noms d'outils cités sont garantis exister dans le catalogue (cf. le cross-check de
``tests/unit/test_prompts.py``). Les prompts portent ``tags={"workflow"}`` (cohérent avec le
tagging par domaine des outils, et filtrable côté Code Mode).
"""

from __future__ import annotations

from fastmcp import FastMCP

#: Tag commun aux prompts de workflow (parité avec le tagging par domaine des outils).
_WORKFLOW_TAGS = {"workflow"}


def register_prompts(mcp: FastMCP) -> None:
    """Enregistre les 5 prompts de workflow sur le serveur MCP racine.

    Appelée par ``build_server`` avant le montage des sous-serveurs. Chaque prompt est une
    fonction pure renvoyant un ``str`` ; son nom d'outil = le nom de la fonction, sa
    description = la première ligne de sa docstring.
    """

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def scaffold_multi_agent(goal: str) -> str:
        """Plan pas-à-pas pour scaffolder un système multi-agents ADK pour un objectif donné."""
        return f"""# Scaffolder un système multi-agents ADK
Objectif : {goal}

Suis ces étapes en appelant les outils du toolkit dans l'ordre. Choisis `<path>` (dossier
parent) et un `app_name` (identifiant Python : lettres/chiffres/underscore, ne commence pas
par un chiffre) cohérents et réutilise-les à chaque appel.

1. **Scaffolder l'app.** `project_create(path, app_name, model="gemini-2.5-flash",
   backend="ai_studio")` — écrit `agent.py` + `__init__.py` + `.env`. (Backend `vertex` si tu
   passes par Vertex AI.) Renseigne ensuite les clés via `project_set_env` si besoin.

2. **Créer les agents enfants (workers).** Un `agents_create_llm(path, app_name, name=...,
   model=..., instruction=..., description=...)` par sous-tâche de « {goal} ». Donne à chacun
   une `instruction` précise et une `description` (utile si un parent doit router vers lui).
   Pour confier un modèle non-Gemini à un enfant : `models_configure_litellm(path, app_name,
   agent_name, provider=..., model=...)` (ex. provider `lm_studio`/`openai`/`anthropic`).

3. **Composer un agent d'orchestration.** Selon le flux :
   - séquentiel (pipeline étape→étape) : `agents_create_sequential(path, app_name, name=...,
     sub_agents=[...])` ;
   - parallèle (fan-out simultané) : `agents_create_parallel(path, app_name, name=...,
     sub_agents=[...])` ;
   - boucle (répéter jusqu'à critère) : `agents_create_loop(path, app_name, name=...,
     sub_agents=[...], max_iterations=N)`.
   Les `sub_agents` doivent déjà exister (crée-les à l'étape 2 d'abord). Pour (re)brancher les
   enfants d'un agent existant : `agents_compose(path, app_name, name, sub_agents=[...])`.
   (Astuce délégation agent-comme-outil : `agents_as_tool` / `tools_add_agent_tool`.)

4. **Désigner la racine.** `agents_set_root(path, app_name, name=<orchestrateur>)` — c'est le
   `root_agent` exécuté. Vérifie l'arbre avec `agents_list(path, app_name)` /
   `agents_get(path, app_name, name)`.

5. **Régler les modèles si besoin.** `models_set(path, app_name, agent_name, model=...)` pour
   un modèle Gemini par chaîne ; `models_generate_config(...)` pour temperature/safety.

6. **Exécuter.** `run_agent(path, app_name, user_id, session_id, message)` lance le
   `root_agent` et renvoie les événements + le texte final. (`run_stream` pour la progression
   SSE.) Nécessite des identifiants modèle dans `.env` (GOOGLE_API_KEY ou creds Vertex).

Rappel : après chaque étape `agents_*`/`models_*`/`tools_*`, `agent.py` est intégralement
régénéré depuis le sidecar — n'édite pas `agent.py` à la main."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def add_guardrail(agent: str, concern: str) -> str:
        """Décide entre callback (par-agent) et plugin (global) puis attache le garde-fou."""
        return f"""# Ajouter un garde-fou à un agent ADK
Agent ciblé : {agent}
Préoccupation : {concern}

## 1. Choisir la portée : callback (par-agent) vs plugin (global)
- **Callback par-agent** → `safety_add_callback`. À privilégier quand le garde-fou ne concerne
  QUE l'agent « {agent} ». Rendu comme une vraie fonction attachée via le kwarg ADK réel
  (`before_model_callback` / `before_tool_callback`). Renvoyer non-`None` court-circuite le
  LLM ou l'outil.
- **Plugin global** → `safety_add_plugin`. À privilégier quand la politique doit s'appliquer à
  TOUS les agents/outils de l'app (sous-classe `BasePlugin` câblée sur le `Runner` via `App`).

## 2. Appeler l'outil
### Filtrage d'ENTRÉE utilisateur (le plus courant pour « {concern} ») — callback before_model
- Bloquer des mots-clés. Appelle `safety_add_callback(path, app_name, agent_name="{agent}",
  hook="before_model", policy=...)` avec
  `policy = {{"kind": "block_keywords", "keywords": "mot1,mot2", "refusal": "Désolé."}}`.
- Limiter la taille d'entrée : même appel `safety_add_callback(..., hook="before_model", ...)`
  avec `policy = {{"kind": "max_input_chars", "max_chars": "2000"}}`.

### Bloquer l'usage d'un OUTIL — callback before_tool (par-agent)
- `safety_add_callback(path, app_name, agent_name="{agent}", hook="before_tool", policy=...)`
  avec `policy = {{"kind": "block_tool", "denylist": "delete_db", "message": "Interdit."}}`.

### Politique GLOBALE (tous les agents) — plugin
- Denylist d'outils globale :
  `safety_add_plugin(path, app_name, name="tool_guard", kind="tool_denylist",
   config={{"denylist": "delete_db,drop_table"}})`
- Journalisation de tous les événements :
  `safety_add_plugin(path, app_name, name="event_log", kind="logging")`

## 3. (Optionnel) Réglages de sûreté du modèle + plafond d'appels
- `safety_settings(path, app_name, agent_name="{agent}", gemini_safety=[...])` avec un item
  `{{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}}`
  (réutilise le rendu `generate_content_config`).
- `safety_settings(path, app_name, agent_name="{agent}", max_llm_calls=20)` borne les appels
  LLM par exécution (réellement appliqué par `run_agent`/`run_stream` quand aucune valeur
  explicite n'est passée).

Après l'attache, `agent.py` (callbacks) ou `plugins.py` (plugins) est régénéré ; vérifie via
`agents_get(path, app_name, name="{agent}")`."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def write_evalset(agent: str) -> str:
        """Plan pour créer un eval set, fixer les critères offline, puis lancer l'évaluation."""
        return f"""# Écrire et lancer un eval set ADK
Agent à évaluer : {agent} (le `root_agent` de l'app, ou un sous-agent via `agent_name`)

Métriques OFFLINE (aucun modèle juge, aucune clé) — privilégie-les en CI :
- `tool_trajectory_avg_score` : comparaison STRUCTURELLE des appels d'outils attendus vs réels.
- `response_match_score` : ROUGE-1 entre la réponse finale et `expected_response`.
(Les métriques « LLM-judge » comme `response_evaluation_score` nécessitent un modèle juge + des
creds → hors-ligne impossible ; ne les utilise pas pour un check déterministe.)

1. **Créer l'eval set.** `eval_create_set(path, app_name, name="smoke",
   cases=[{{"query": "...", "expected_response": "...",
            "expected_tool_use": [{{"name": "<outil>", "args": {{...}}}}]}}])`
   — écrit `<app>/eval/smoke.evalset.json` (schéma `EvalSet` conforme). Mets `expected_tool_use`
   seulement si tu évalues la trajectoire d'outils ; sinon une liste vide suffit.

2. **Fixer les critères (seuils).** `eval_set_criteria(path, app_name,
   tool_trajectory_avg_score=1.0, response_match_score=0.8)` — écrit `eval/test_config.json`
   (seuils dans [0, 1]). Lu automatiquement par `eval_run`.

3. **Lancer l'évaluation.** `eval_run(path, app_name, ...)` en lui passant le chemin du fichier
   `.evalset.json` créé à l'étape 1 (paramètre du chemin de l'eval set) et `num_runs=1`. Cela
   importe l'agent, exécute l'éval offline, persiste un rapport et renvoie
   `passed` + les scores par métrique. ⚠️ Une NON-conformité aux seuils est un résultat NORMAL
   (`ok=True, passed=False`), pas une erreur. Les vrais échecs (eval set absent, agent
   nécessitant une clé, extra `eval` manquant) renvoient `err`. Pour évaluer un sous-agent,
   passe `agent_name=...`.

4. **Relire un rapport.** `eval_report(path, app_name, report_id=<id renvoyé par eval_run>)`.

Pré-requis : l'extra d'évaluation (`uv add 'adk-toolkit-mcp[eval]'`) pour les métriques ROUGE."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def deploy_checklist(target: str) -> str:
        """Checklist de déploiement : preflight, choix de cible, commande, flags et creds."""
        return f"""# Checklist de déploiement ADK
Cible demandée : {target}  (attendu : agent_engine | cloud_run | gke)

1. **Preflight (best-effort, ne bloque jamais).** `deploy_preflight(target="{target}")` — vérifie
   `gcloud`/`adk` sur le PATH (et `kubectl` pour gke). Corrige les manques signalés avant de
   déployer.

2. **(Optionnel) Conteneuriser.** `deploy_containerize(path, app_name)` écrit un `Dockerfile`
   servant `adk api_server` sur `$PORT` (utile pour Cloud Run / GKE en image custom).

3. **Construire la commande (par défaut `execute=False` → renvoie l'argv + un plan, n'exécute
   rien).** Choisis selon la cible :
   - **Agent Engine (Vertex AI)** : `deploy_agent_engine(path, app_name, project=..., region=...,
     requirements_file=..., execute=False)`. Flags réels 2.1.0 : `--project`, `--region`,
     `--display_name` (l'`app_name` y est mappé), `--requirements_file`. ⚠️ PAS de `--app_name` ;
     `--staging_bucket` est DÉPRÉCIÉ (no-op, non émis).
   - **Cloud Run** : `deploy_cloud_run(path, app_name, project=..., region=..., service_name=...,
     with_ui=False, enable_cloud_trace=False, execute=False)`. `enable_cloud_trace=True` émet le
     vrai flag `--trace_to_cloud` (PAS `--enable_cloud_trace`).
   - **GKE** : `deploy_gke(path, app_name, project=..., region=..., cluster=..., service_name=...,
     execute=False)`. Le paramètre `cluster` mappe sur `--cluster_name` (PAS `--cluster`).

4. **Vérifier le plan, puis exécuter.** Relis `argv`/`plan`. Le déploiement réel = ré-appeler le
   MÊME outil avec `execute=True` (nécessite des identifiants GCP : `gcloud auth login` +
   `gcloud config set project`, et pour Vertex un projet/région valides). Chaque flag émis est
   validé contre le vrai `adk <sub> --help` — un flag inconnu renvoie `err`.

5. **Statut.** `deploy_status(target="{target}", project=..., region=..., service_name=...,
   cluster=...)` interroge Cloud Run (gcloud) / GKE (kubectl) ; Agent Engine renvoie une guidance
   (pas de CLI de statut dédiée).

Rappels creds : ne mets JAMAIS de secret en dur ; utilise `.env`/variables d'environnement. Le
Web UI (`--with_ui`) est pour le dev/test, pas la production."""

    @mcp.prompt(tags=_WORKFLOW_TAGS)
    def debug_agent(symptom: str) -> str:
        """Itinéraire de dépannage d'un agent ADK : inspecter les événements + pièges connus."""
        return f"""# Déboguer un agent ADK
Symptôme : {symptom}

## 1. Reproduire et inspecter les événements
- Relance l'agent : `run_agent(path, app_name, user_id, session_id, message)` — renvoie la liste
  des événements sérialisés + `final_text`.
- Passe ces événements à `run_inspect_events(events=<events renvoyés>)` : outil PUR qui résume
  les `function_calls`, les outils réellement utilisés (`tool_names`), les transferts d'agents
  (`transfers`), les clés de `state_delta` et le texte final. C'est le point de départ pour voir
  CE QUE l'agent a fait.
- Pour suivre la progression en direct (où ça bloque) : `run_stream(path, app_name, user_id,
  session_id, message)` (rapporte chaque événement via le contexte MCP).

## 2. Vérifier la structure de l'agent
- `agents_list(path, app_name)` : la racine est-elle celle attendue (`agents_set_root`) ?
- `agents_get(path, app_name, name)` : la spec (modèle, instruction, sub_agents) est-elle correcte ?
- `tools_list(path, app_name, agent_name)` : les outils attendus sont-ils bien attachés ?

## 3. Pièges connus (mappés au symptôme)
- **« pas de réponse / clé API »** → identifiants manquants : renseigne `.env`
  (GOOGLE_API_KEY pour AI Studio, ou GOOGLE_GENAI_USE_VERTEXAI=TRUE + GOOGLE_CLOUD_PROJECT pour
  Vertex) via `project_set_env`. `run_live` exige EN PLUS un modèle Gemini live-capable.
- **« l'agent boucle / trop d'appels LLM »** → borne via `safety_settings(..., max_llm_calls=N)`
  (appliqué par `run_agent`), ou passe `max_llm_calls` directement à `run_agent`.
- **« un outil n'est jamais appelé / mauvaise trajectoire »** → vérifie `tools_list` et
  l'`instruction` de l'agent (`agents_get`), puis formalise l'attendu avec `eval_create_set` +
  `eval_run` (`tool_trajectory_avg_score`).
- **« mon edit d'`agent.py` a disparu »** → NORMAL : `agent.py` est régénéré depuis le sidecar à
  chaque mutation `agents_*`/`tools_*`/`models_*`. Modifie via les outils, pas le fichier.
- **« état non persistant »** → l'état `temp:` n'est PAS persisté entre `get_session` (design ADK) ;
  inspecte via `sessions_state_get` / `sessions_get`. Pour une session DB, l'URL doit être async
  (`sqlite+aiosqlite:///...`).
- **garde-fou inattendu** → un `safety_add_callback`/`safety_add_plugin` court-circuite peut-être
  le LLM/l'outil ; vérifie via `agents_get` (callbacks) et `plugins.py`.

## 4. Tracer plus finement (optionnel)
`observability_enable_otel(path, app_name, exporter="console")` génère `otel_setup.py` ; ou
`observability_trace_view(path, app_name)` lance l'UI Web d'ADK (onglet « Trace »)."""
