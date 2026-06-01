"""Modèle de projet ADK code-first : sidecar JSON + régénération complète de ``agent.py``.

Le toolkit décrit la composition multi-agents dans un **fichier sidecar**
``<app_dir>/.adk_toolkit/agents.json`` (où ``<app_dir> = <path>/<app_name>``), puis
**régénère intégralement** ``agent.py`` à partir de ce modèle. Régénérer plutôt que
patcher du Python est plus robuste (pas de parsing/round-trip d'AST, sortie déterministe).

Ce module est **pur et testable unitairement** (aucune dépendance à google-adk : on ne
fait que produire une *chaîne source* qui importera l'ADK à son propre runtime). Il fournit :

- des dataclasses (`ProjectModel`, `AgentSpec`) figées (`frozen=True`) ;
- `load_model` / `save_model` (lecture/écriture du sidecar, création si absent) ;
- `add_or_update_agent` (mise à jour **immuable** : renvoie un nouveau `ProjectModel`) ;
- `render_agent_module` (génère un ``agent.py`` valide, agents **triés topologiquement**,
  détection de cycle -> `ValueError`) ;
- `regenerate` (écrit ``agent.py`` + assure ``__init__.py`` via `Workspace`, idempotent).

Voir ``docs/adk-api-notes/agents.md`` pour les signatures ADK réelles confirmées par
introspection (et la note sur la dépréciation des agents workflow en google-adk 2.1.0).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .workspace import Workspace

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
#: Dossier du sidecar, relatif au dossier de l'app (`<path>/<app_name>`).
SIDECAR_DIR = ".adk_toolkit"

#: Nom du fichier sidecar dans `SIDECAR_DIR`.
SIDECAR_FILE = "agents.json"

#: Chemin relatif complet du sidecar (depuis le dossier de l'app).
SIDECAR_PATH = f"{SIDECAR_DIR}/{SIDECAR_FILE}"

#: Types d'agents supportés.
AgentType = Literal["llm", "sequential", "parallel", "loop", "custom"]

_AGENT_TYPES: frozenset[str] = frozenset({"llm", "sequential", "parallel", "loop", "custom"})

#: Un nom d'agent doit être un identifiant Python (sert de nom de variable de module).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Longueur de ligne cible (doit refléter ``[tool.ruff] line-length`` du pyproject) afin que
#: le code généré soit déjà dans la forme produite par ``ruff format`` (idempotence).
LINE_LENGTH = 100

# --------------------------------------------------------------------------- #
# Outils (domaine `tools`, passes 3a + 3b)
# --------------------------------------------------------------------------- #
#: Genres d'outils supportés. 3a (sans dépendance) : ``function``, ``long_running``,
#: ``builtin``, ``agent_tool``, ``openapi``. 3b (dépendance optionnelle / extras
#: ``google-adk[...]``, codegen-only) : ``bigquery``, ``spanner``, ``mcp_toolset``,
#: ``apihub``, ``langchain``, ``crewai``.
ToolKind = Literal[
    "function",
    "long_running",
    "builtin",
    "agent_tool",
    "openapi",
    "bigquery",
    "spanner",
    "mcp_toolset",
    "apihub",
    "langchain",
    "crewai",
]

_TOOL_KINDS: frozenset[str] = frozenset(
    {
        "function",
        "long_running",
        "builtin",
        "agent_tool",
        "openapi",
        "bigquery",
        "spanner",
        "mcp_toolset",
        "apihub",
        "langchain",
        "crewai",
    }
)

#: Genres « toolset » dont la ``ref`` est une variable module-level (``<id>`` dans ``tools=[...]``)
#: construite par un bloc helper. Le ``name`` du :class:`ToolSpec` sert d'identifiant de variable.
_TOOLSET_VAR_KINDS: frozenset[str] = frozenset(
    {"openapi", "bigquery", "spanner", "mcp_toolset", "apihub"}
)

#: Genres « toolset » qui acceptent nativement ``auth_scheme=`` / ``auth_credential=`` (confirmé
#: par introspection : ``OpenAPIToolset``, ``McpToolset``, ``APIHubToolset``). ``BigQueryToolset`` /
#: ``SpannerToolset`` n'en ont pas (ils prennent un ``credentials_config``) -> auth rejeté.
_AUTH_CAPABLE_KINDS: frozenset[str] = frozenset({"openapi", "apihub", "mcp_toolset"})

#: Transports MCP supportés -> classe de connection-params ADK (confirmée par introspection).
_MCP_TRANSPORTS: dict[str, str] = {
    "stdio": "StdioConnectionParams",
    "sse": "SseConnectionParams",
    "http": "StreamableHTTPConnectionParams",
}

#: Schémas d'auth supportés par :func:`set_auth` -> membre de ``AuthCredentialTypes`` (confirmé).
_AUTH_SCHEMES: frozenset[str] = frozenset({"apikey", "oauth2", "service_account", "bearer"})

_AUTH_TYPE_FOR_SCHEME: dict[str, str] = {
    "apikey": "API_KEY",
    "bearer": "HTTP",
    "oauth2": "OAUTH2",
    "service_account": "SERVICE_ACCOUNT",
}

#: Builtins ADK "core" : instances d'outils déjà exportées (aucun argument requis).
#: Confirmés par introspection en google-adk 2.1.0 (cf. ``docs/adk-api-notes/tools.md``).
#: Ce sont des **instances** (ex. ``google_search`` = ``GoogleSearchTool()``) ou des
#: fonctions (``exit_loop``, ``transfer_to_agent``) — elles entrent telles quelles dans
#: ``tools=[...]`` et s'importent depuis ``google.adk.tools``.
CORE_BUILTINS: frozenset[str] = frozenset(
    {
        "google_search",
        "url_context",
        "load_memory",
        "preload_memory",
        "load_artifacts",
        "get_user_choice",
        "exit_loop",
        "transfer_to_agent",
        "enterprise_web_search",
        "google_maps_grounding",
    }
)

#: Builtins nécessitant un argument (rendus comme un appel de constructeur).
#: ``vertex_ai_search`` -> ``VertexAiSearchTool(data_store_id=... | search_engine_id=...)``.
ARG_BUILTINS: frozenset[str] = frozenset({"vertex_ai_search"})

#: Ensemble complet des ``kind`` builtin reconnus.
BUILTIN_TOOLS: frozenset[str] = CORE_BUILTINS | ARG_BUILTINS

#: Mapping builtin nécessitant un arg -> nom de classe ADK importée.
_BUILTIN_CLASS: dict[str, str] = {"vertex_ai_search": "VertexAiSearchTool"}

#: Types Python autorisés pour les paramètres d'une function-tool (validation légère).
_ALLOWED_PARAM_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "list", "dict", "tuple", "set", "bytes", "Any", "None"}
)

#: Import depuis lequel les classes/builtins d'outils sont tirés (package root).
_TOOLS_IMPORT_MODULE = "google.adk.tools"

#: Import (chemin réel confirmé) pour ``OpenAPIToolset``.
_OPENAPI_IMPORT = "from google.adk.tools.openapi_tool import OpenAPIToolset"

#: Imports (chemins réels confirmés par introspection en 2.1.0) des toolsets 3b.
_BIGQUERY_IMPORT = "from google.adk.tools.bigquery import BigQueryToolset"
_SPANNER_IMPORT = "from google.adk.tools.spanner import SpannerToolset"
_APIHUB_IMPORT = "from google.adk.tools.apihub_tool import APIHubToolset"
#: Note (cf. docs/adk-api-notes/tools.md) : ces deux chemins re-exportent depuis
#: ``google.adk.integrations.*`` et émettent une ``DeprecationWarning`` au runtime utilisateur ;
#: on conserve le chemin demandé par la tâche (toujours fonctionnel, codegen-only).
_LANGCHAIN_IMPORT = "from google.adk.tools.langchain_tool import LangchainTool"
_CREWAI_IMPORT = "from google.adk.tools.crewai_tool import CrewaiTool"

#: Module des classes d'auth (re-export confirmé).
_AUTH_IMPORT_MODULE = "google.adk.auth"
#: Module des sous-objets d'auth (HttpAuth/OAuth2Auth/ServiceAccount/HttpCredentials).
_AUTH_CRED_IMPORT_MODULE = "google.adk.auth.auth_credential"
#: Imports MCP (toolset + StdioServerParameters depuis le paquet ``mcp``).
_MCP_TOOLSET_IMPORT_MODULE = "google.adk.tools.mcp_tool"
_MCP_STDIO_PARAMS_IMPORT = "from mcp import StdioServerParameters"

#: Mapping type d'agent -> nom de classe ADK à importer.
_CLASS_FOR_TYPE: dict[str, str] = {
    "llm": "LlmAgent",
    "sequential": "SequentialAgent",
    "parallel": "ParallelAgent",
    "loop": "LoopAgent",
    # `custom` produit une sous-classe de BaseAgent.
}

#: Ordre canonique d'import (sous-ensemble effectivement utilisé est conservé).
_IMPORT_ORDER: tuple[str, ...] = (
    "LlmAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "BaseAgent",
)


# --------------------------------------------------------------------------- #
# Dataclasses du modèle (immuables)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuthSpec:
    """Sous-spécification d'auth attachée à un toolset (3b).

    ``scheme`` ∈ :data:`_AUTH_SCHEMES` (``apikey``/``oauth2``/``service_account``/``bearer``).
    ``credential`` est une liste de paires ``(clé, valeur-littérale)`` (gelée en tuple pour
    rester hashable/immutable) rendue dans un ``AuthCredential(...)`` selon le schéma — voir
    :func:`_render_auth_credential` et ``docs/adk-api-notes/tools.md``.
    """

    scheme: str
    credential: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "credential": {k: v for k, v in self.credential}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthSpec:
        cred_raw = data.get("credential") or {}
        return cls(
            scheme=str(data.get("scheme", "")),
            credential=tuple((str(k), str(v)) for k, v in cred_raw.items()),
        )


@dataclass(frozen=True)
class ToolSpec:
    """Spécification immuable d'un outil attaché à un agent (domaine `tools`, 3a + 3b).

    Le champ ``kind`` discrimine ; seuls les champs pertinents sont renseignés/sérialisés :

    - ``function`` / ``long_running`` : ``name`` (identifiant), ``params`` (tuple de
      ``(name, type, default|None)``), ``docstring``, ``returns``, ``body``.
    - ``builtin`` : ``builtin_kind`` (membre de :data:`BUILTIN_TOOLS`), ``args`` (pour
      ``vertex_ai_search`` : ``{"data_store_id": ...}`` ou ``{"search_engine_id": ...}``).
    - ``agent_tool`` : ``target_agent`` (nom d'un agent **existant** du modèle).
    - ``openapi`` : ``name`` (identifiant de la variable toolset), ``spec`` (chaîne OpenAPI).
    - ``bigquery`` / ``spanner`` : ``name`` (var toolset), ``args`` (kwargs *expressions* source,
      ex. ``{"bigquery_tool_config": "my_cfg"}``).
    - ``mcp_toolset`` : ``name`` (var), ``transport`` ∈ {stdio,sse,http}, ``command``+``mcp_args``
      (stdio) ou ``url``+``headers`` (sse/http), ``tool_filter``.
    - ``apihub`` : ``name`` (var), ``apihub_resource_name``.
    - ``langchain`` / ``crewai`` : ``import_line`` (rendu verbatim), ``tool_expr`` (expression de
      construction), + ``name``/``description`` (crewai : ``name`` requis).
    - ``auth`` (optionnel, openapi/apihub/mcp_toolset) : :class:`AuthSpec` rendu en
      ``auth_credential=``.

    ``ref_key`` renvoie une clé d'identité stable utilisée pour le "remplacement par nom"
    (append unique / replace) côté domaine.
    """

    kind: ToolKind
    name: str = ""
    params: tuple[tuple[str, str, str | None], ...] = ()
    docstring: str = ""
    returns: str = "dict"
    body: str = "return {}"
    builtin_kind: str = ""
    args: tuple[tuple[str, str], ...] = ()
    target_agent: str = ""
    spec: str = ""
    # --- 3b : champs des toolsets à dépendance optionnelle ---
    transport: str = ""
    command: str = ""
    mcp_args: tuple[str, ...] = ()
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    tool_filter: tuple[str, ...] = ()
    apihub_resource_name: str = ""
    import_line: str = ""
    tool_expr: str = ""
    description: str = ""
    auth: AuthSpec | None = None

    def ref_key(self) -> str:
        """Clé d'unicité (utilisée pour append-unique / replace-by-name côté domaine).

        - genres « variable de toolset » (``openapi``/``bigquery``/``spanner``/``mcp_toolset``/
          ``apihub``) + ``function``/``long_running`` -> ``<kind>:<name>`` ;
        - ``builtin`` -> ``builtin:<builtin_kind>`` ; ``agent_tool`` -> ``agent_tool:<target>`` ;
        - ``langchain``/``crewai`` -> ``<kind>:<tool_expr>`` (l'expression identifie l'outil ;
          ``crewai`` peut aussi renommer via ``name`` mais l'expression reste l'identité).
        """
        if self.kind in ("function", "long_running") or self.kind in _TOOLSET_VAR_KINDS:
            return f"{self.kind}:{self.name}"
        if self.kind == "builtin":
            return f"builtin:{self.builtin_kind}"
        if self.kind == "agent_tool":
            return f"agent_tool:{self.target_agent}"
        if self.kind in ("langchain", "crewai"):
            return f"{self.kind}:{self.tool_expr}"
        return self.kind  # pragma: no cover (kind validé en amont)

    def to_dict(self) -> dict[str, Any]:
        """Sérialise vers la forme JSON du sidecar (champs pertinents selon ``kind``)."""
        base: dict[str, Any] = {"kind": self.kind}
        if self.kind in ("function", "long_running"):
            base.update(
                {
                    "name": self.name,
                    "params": [list(p) for p in self.params],
                    "docstring": self.docstring,
                    "returns": self.returns,
                    "body": self.body,
                }
            )
        elif self.kind == "builtin":
            base["builtin_kind"] = self.builtin_kind
            if self.args:
                base["args"] = {k: v for k, v in self.args}
        elif self.kind == "agent_tool":
            base["target_agent"] = self.target_agent
        elif self.kind == "openapi":
            base.update({"name": self.name, "spec": self.spec})
        elif self.kind in ("bigquery", "spanner"):
            base["name"] = self.name
            if self.args:
                base["args"] = {k: v for k, v in self.args}
        elif self.kind == "mcp_toolset":
            base.update({"name": self.name, "transport": self.transport})
            if self.command:
                base["command"] = self.command
            if self.mcp_args:
                base["mcp_args"] = list(self.mcp_args)
            if self.url:
                base["url"] = self.url
            if self.headers:
                base["headers"] = {k: v for k, v in self.headers}
            if self.tool_filter:
                base["tool_filter"] = list(self.tool_filter)
        elif self.kind == "apihub":
            base.update({"name": self.name, "apihub_resource_name": self.apihub_resource_name})
        elif self.kind in ("langchain", "crewai"):
            base.update({"import_line": self.import_line, "tool_expr": self.tool_expr})
            if self.name:
                base["name"] = self.name
            if self.description:
                base["description"] = self.description
        if self.auth is not None:
            base["auth"] = self.auth.to_dict()
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> ToolSpec:
        """Désérialise une entrée ``tools`` du sidecar.

        Tolérant à la **forme héritée** (P1) où une entrée d'outil était une simple chaîne
        (nom déjà importé dans le module) : on la mappe vers un ``builtin`` (rendu bare).
        """
        if isinstance(data, str):
            return cls(kind="builtin", builtin_kind=data)
        kind: ToolKind = data.get("kind", "builtin")
        params = tuple(
            (str(p[0]), str(p[1]), (None if len(p) < 3 or p[2] is None else str(p[2])))
            for p in (data.get("params") or [])
        )
        args_raw = data.get("args") or {}
        args = tuple((str(k), str(v)) for k, v in args_raw.items())
        headers_raw = data.get("headers") or {}
        headers = tuple((str(k), str(v)) for k, v in headers_raw.items())
        auth_raw = data.get("auth")
        auth = AuthSpec.from_dict(auth_raw) if isinstance(auth_raw, dict) else None
        return cls(
            kind=kind,
            name=str(data.get("name", "")),
            params=params,
            docstring=str(data.get("docstring", "")),
            returns=str(data.get("returns", "dict")),
            body=str(data.get("body", "return {}")),
            builtin_kind=str(data.get("builtin_kind", "")),
            args=args,
            target_agent=str(data.get("target_agent", "")),
            spec=str(data.get("spec", "")),
            transport=str(data.get("transport", "")),
            command=str(data.get("command", "")),
            mcp_args=tuple(str(a) for a in (data.get("mcp_args") or [])),
            url=str(data.get("url", "")),
            headers=headers,
            tool_filter=tuple(str(t) for t in (data.get("tool_filter") or [])),
            apihub_resource_name=str(data.get("apihub_resource_name", "")),
            import_line=str(data.get("import_line", "")),
            tool_expr=str(data.get("tool_expr", "")),
            description=str(data.get("description", "")),
            auth=auth,
        )


@dataclass(frozen=True)
class ToolRender:
    """Résultat du rendu d'un outil : imports requis, blocs helper top-level, et la
    référence à placer dans ``tools=[...]`` de l'agent propriétaire."""

    imports: tuple[str, ...]
    helpers: tuple[str, ...]
    ref: str


@dataclass(frozen=True)
class AgentSpec:
    """Spécification immuable d'un agent dans le modèle de projet.

    Les champs non pertinents pour un type donné restent à leur valeur par défaut
    (ex. ``model``/``instruction`` ignorés pour un agent ``sequential``).
    """

    name: str
    type: AgentType
    model: str = "gemini-2.5-flash"
    instruction: str = ""
    description: str = ""
    output_key: str | None = None
    #: Outils attachés. ``ToolSpec`` (codegen riche) ; la forme ``str`` héritée (P1) reste
    #: tolérée et rendue comme une référence bare (nom déjà importé). Voir ``render_tool_ref``.
    tools: tuple[ToolSpec | str, ...] = ()
    sub_agents: tuple[str, ...] = ()
    max_iterations: int = 3

    def tool_specs(self) -> tuple[ToolSpec, ...]:
        """Normalise ``tools`` en ``ToolSpec`` (les chaînes héritées -> ``builtin``)."""
        return tuple(t if isinstance(t, ToolSpec) else ToolSpec.from_dict(t) for t in self.tools)

    def to_dict(self) -> dict[str, Any]:
        """Sérialise vers la forme JSON du sidecar (champs pertinents selon le type)."""
        base: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
        if self.type == "llm":
            base.update(
                {
                    "model": self.model,
                    "instruction": self.instruction,
                    "output_key": self.output_key,
                    "tools": [t.to_dict() if isinstance(t, ToolSpec) else t for t in self.tools],
                    "sub_agents": list(self.sub_agents),
                }
            )
        elif self.type in ("sequential", "parallel"):
            base["sub_agents"] = list(self.sub_agents)
        elif self.type == "loop":
            base["sub_agents"] = list(self.sub_agents)
            base["max_iterations"] = self.max_iterations
        # `custom` : seulement name/type/description.
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        """Désérialise une entrée du sidecar (tolérant aux champs absents)."""
        atype = data.get("type", "llm")
        raw_tools = data.get("tools", []) or []
        # Forme héritée (P1) : une entrée chaîne reste une chaîne (passthrough, rendue bare).
        # Forme riche (3a) : un dict est désérialisé en ``ToolSpec``.
        tools: tuple[ToolSpec | str, ...] = tuple(
            t if isinstance(t, str) else ToolSpec.from_dict(t) for t in raw_tools
        )
        return cls(
            name=str(data["name"]),
            type=atype,
            model=str(data.get("model", "gemini-2.5-flash")),
            instruction=str(data.get("instruction", "")),
            description=str(data.get("description", "")),
            output_key=data.get("output_key"),
            tools=tools,
            sub_agents=tuple(data.get("sub_agents", []) or []),
            max_iterations=int(data.get("max_iterations", 3)),
        )


@dataclass(frozen=True)
class ProjectModel:
    """Modèle complet d'une app ADK : liste d'agents + racine désignée."""

    app_name: str
    root: str | None = None
    agents: tuple[AgentSpec, ...] = field(default_factory=tuple)

    def agent_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.agents)

    def get(self, name: str) -> AgentSpec | None:
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "root": self.root,
            "agents": [a.to_dict() for a in self.agents],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectModel:
        agents = tuple(AgentSpec.from_dict(a) for a in data.get("agents", []) or [])
        return cls(
            app_name=str(data.get("app_name", "")),
            root=data.get("root"),
            agents=agents,
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def is_identifier(name: str) -> bool:
    """True si ``name`` est un identifiant Python valide (nom de variable de module)."""
    return bool(_IDENT_RE.match(name))


def validate_spec(spec: AgentSpec) -> str | None:
    """Renvoie un message d'erreur si la spec est invalide, sinon None."""
    if not is_identifier(spec.name):
        return (
            f"Nom d'agent invalide : {spec.name!r}. Attendu un identifiant Python "
            "(lettres, chiffres, underscore ; ne commence pas par un chiffre)."
        )
    if spec.type not in _AGENT_TYPES:
        return f"Type d'agent inconnu : {spec.type!r}. Connus : {', '.join(sorted(_AGENT_TYPES))}."
    if spec.type == "loop" and spec.max_iterations <= 0:
        return f"max_iterations doit être > 0 (reçu {spec.max_iterations})."
    for sub in spec.sub_agents:
        if not is_identifier(sub):
            return f"sub_agent invalide : {sub!r}. Attendu un identifiant Python."
    return None


# --------------------------------------------------------------------------- #
# Mutations immuables
# --------------------------------------------------------------------------- #
def add_or_update_agent(model: ProjectModel, spec: AgentSpec) -> ProjectModel:
    """Ajoute ``spec`` ou remplace l'agent existant de même nom. **Renvoie un nouveau modèle.**

    L'ordre est préservé : un remplacement reste à sa position ; un ajout est appended.
    """
    found = False
    new_agents: list[AgentSpec] = []
    for a in model.agents:
        if a.name == spec.name:
            new_agents.append(spec)
            found = True
        else:
            new_agents.append(a)
    if not found:
        new_agents.append(spec)
    return replace(model, agents=tuple(new_agents))


def set_root(model: ProjectModel, name: str) -> ProjectModel:
    """Renvoie un nouveau modèle dont la racine est ``name`` (sans valider l'existence ici)."""
    return replace(model, root=name)


def add_or_replace_tool(spec: AgentSpec, tool: ToolSpec) -> AgentSpec:
    """Attache ``tool`` à ``spec`` selon « **append unique, replace by name** ».

    Si un outil de même :meth:`ToolSpec.ref_key` existe déjà, il est **remplacé en place**
    (position préservée) ; sinon ``tool`` est **ajouté** en fin de liste. **Renvoie un nouvel
    ``AgentSpec``** (immuable). Les entrées héritées (chaîne) sont normalisées en ``ToolSpec``.
    """
    key = tool.ref_key()
    found = False
    new_tools: list[ToolSpec] = []
    for existing in spec.tool_specs():
        if existing.ref_key() == key:
            new_tools.append(tool)
            found = True
        else:
            new_tools.append(existing)
    if not found:
        new_tools.append(tool)
    return replace(spec, tools=tuple(new_tools))


# --------------------------------------------------------------------------- #
# Sidecar I/O
# --------------------------------------------------------------------------- #
def load_model(ws: Workspace, app_name: str) -> ProjectModel:
    """Charge le sidecar ``.adk_toolkit/agents.json`` ; renvoie un modèle vide si absent.

    ``ws`` doit pointer sur le **dossier de l'app** (``<path>/<app_name>``).
    """
    if not ws.exists(SIDECAR_PATH):
        return ProjectModel(app_name=app_name)
    raw = ws.read(SIDECAR_PATH)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # sidecar corrompu -> erreur claire
        raise ValueError(f"Sidecar JSON invalide ({SIDECAR_PATH}) : {exc}") from exc
    model = ProjectModel.from_dict(data)
    # On force app_name fourni (source de vérité = dossier).
    return replace(model, app_name=app_name)


def save_model(ws: Workspace, model: ProjectModel) -> bool:
    """Écrit le sidecar (JSON indenté, déterministe). Renvoie True si modifié."""
    payload = json.dumps(model.to_dict(), indent=2, sort_keys=False) + "\n"
    return ws.write(SIDECAR_PATH, payload)


# --------------------------------------------------------------------------- #
# Tri topologique + détection de cycle
# --------------------------------------------------------------------------- #
def _agent_dependencies(spec: AgentSpec) -> tuple[str, ...]:
    """Noms d'agents dont ``spec`` dépend pour être défini après eux dans ``agent.py``.

    Deux sources de dépendance vers un autre agent :
    - ``sub_agents`` (composition : l'enfant doit précéder le parent) ;
    - un outil ``agent_tool`` ciblant un agent (la cible doit précéder l'agent enveloppant,
      sinon ``AgentTool(agent=<cible>)`` référencerait une variable non définie).
    """
    deps: list[str] = list(spec.sub_agents)
    for tool in spec.tool_specs():
        if tool.kind == "agent_tool" and tool.target_agent:
            deps.append(tool.target_agent)
    return tuple(deps)


def topological_order(model: ProjectModel) -> list[AgentSpec]:
    """Trie les agents pour qu'une dépendance soit définie avant son dépendant.

    Une dépendance = un ``sub_agent`` **ou** la cible d'un outil ``agent_tool`` (cf.
    :func:`_agent_dependencies`). Lève ``ValueError`` si un cycle est détecté (les outils
    convertissent en ``err``). Les références à un nom absent sont ignorées pour
    l'ordonnancement (la validation d'existence est faite en amont par les outils du domaine).
    """
    by_name: dict[str, AgentSpec] = {a.name: a for a in model.agents}
    order: list[AgentSpec] = []
    # États : 0 = non visité, 1 = en cours (gris), 2 = terminé (noir).
    state: dict[str, int] = {a.name: 0 for a in model.agents}

    def visit(name: str, path: tuple[str, ...]) -> None:
        st = state.get(name, 2)
        if st == 2:
            return
        if st == 1:
            cycle = " -> ".join((*path, name))
            raise ValueError(f"Cycle détecté dans les dépendances d'agents : {cycle}")
        state[name] = 1
        spec = by_name[name]
        for dep in _agent_dependencies(spec):
            if dep in by_name:  # n'ordonne que les références internes connues
                visit(dep, (*path, name))
        state[name] = 2
        order.append(spec)

    # Ordre stable : on itère dans l'ordre d'insertion du modèle.
    for a in model.agents:
        visit(a.name, ())
    return order


# --------------------------------------------------------------------------- #
# Rendu de source — helpers
# --------------------------------------------------------------------------- #
def _py_str(value: str) -> str:
    """Littéral chaîne Python **stable pour ``ruff format``**.

    ``ruff format`` (comme Black) préfère les guillemets doubles, **sauf** si la valeur
    contient un ``"`` mais pas de ``'`` — auquel cas il bascule sur les guillemets simples
    pour éviter d'échapper. On reproduit exactement ce choix pour que la sortie générée soit
    déjà dans la forme que ruff écrirait (idempotence de ``format --check``).
    """
    has_double = '"' in value
    has_single = "'" in value
    if has_double and not has_single:
        # Guillemets simples : seul le backslash doit être échappé.
        escaped = value.replace("\\", "\\\\")
        return f"'{escaped}'"
    # Guillemets doubles par défaut : échapper backslash puis guillemet double.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_param(name: str, ptype: str, default: str | None) -> str:
    """Rend un paramètre de signature : ``name: type`` ou ``name: type = default``.

    ``default`` est un **littéral source déjà rendu** (ex. ``"x"``, ``0``, ``None``).
    Quand ``default`` est ``None`` (au sens Python), le paramètre n'a pas de défaut.
    """
    base = f"{name}: {ptype}"
    return base if default is None else f"{base} = {default}"


def _render_function_def(spec: ToolSpec) -> str:
    """Rend un bloc ``def`` top-level : signature typée, docstring 1-ligne, puis le corps.

    Le corps et la docstring sont indentés de 4 espaces ; le bloc se termine par un seul
    ``\\n`` (le renderer de module gère l'espacement inter-blocs façon ruff).
    """
    params = ", ".join(_render_param(n, t, d) for (n, t, d) in spec.params)
    doc = (spec.docstring or spec.name).replace("\\", "\\\\").replace('"', '\\"')
    # docstring sur une ligne, échappée (guillemets triples).
    doc_line = f'    """{doc}"""\n'
    body_lines = spec.body.splitlines() or ["return {}"]
    body = "".join(f"    {line}\n" for line in body_lines)
    return f"def {spec.name}({params}) -> {spec.returns}:\n{doc_line}{body}"


def _render_builtin_ref(spec: ToolSpec) -> ToolRender:
    """Rend la référence d'un builtin (core -> nom bare ; ``vertex_ai_search`` -> appel)."""
    if spec.builtin_kind in CORE_BUILTINS:
        imp = f"from {_TOOLS_IMPORT_MODULE} import {spec.builtin_kind}"
        return ToolRender(imports=(imp,), helpers=(), ref=spec.builtin_kind)
    if spec.builtin_kind in ARG_BUILTINS:
        class_name = _BUILTIN_CLASS[spec.builtin_kind]
        imp = f"from {_TOOLS_IMPORT_MODULE} import {class_name}"
        kwargs = ", ".join(f"{k}={_py_str(v)}" for k, v in spec.args)
        return ToolRender(imports=(imp,), helpers=(), ref=f"{class_name}({kwargs})")
    # builtin_kind inconnu : on rend tel quel (la validation amont l'aura rejeté).
    return ToolRender(imports=(), helpers=(), ref=spec.builtin_kind)  # pragma: no cover


def render_tool_ref(tool: ToolSpec | str) -> ToolRender:
    """Rendu d'une entrée ``tools`` -> :class:`ToolRender` (imports, helpers, ref).

    POINT D'EXTENSION implémenté en passes 3a + 3b. Genres gérés :

    Passe 3a (sans dépendance) :

    - ``function`` : helper = un ``def`` rendu ; ``ref`` = ``<name>`` (ADK auto-wrappe la
      fonction en ``FunctionTool`` via ``canonical_tools`` — cf. ``docs/adk-api-notes/tools.md``).
    - ``long_running`` : même helper ; import ``LongRunningFunctionTool`` ;
      ``ref`` = ``LongRunningFunctionTool(func=<name>)``.
    - ``builtin`` : ``ref`` = nom du builtin (ex. ``google_search``) importé ;
      ``vertex_ai_search`` -> ``VertexAiSearchTool(data_store_id="...")``.
    - ``agent_tool`` : import ``AgentTool`` ; ``ref`` = ``AgentTool(agent=<target>)``.
    - ``openapi`` : import ``OpenAPIToolset`` ; helper = ``<id> = OpenAPIToolset(spec_str=..., \
      spec_str_type="json")`` ; ``ref`` = ``<id>`` (le toolset entre **directement** dans
      ``tools=[...]`` — confirmé par introspection, pas de ``.get_tools()``).

    Passe 3b (dépendance optionnelle ; **codegen-only** — le toolkit n'importe jamais ces extras) :

    - ``bigquery`` / ``spanner`` : import du toolset ; helper ``<id> = BigQueryToolset(<args>)`` /
      ``SpannerToolset(<args>)`` ; ``ref`` = ``<id>``.
    - ``mcp_toolset`` : import ``McpToolset`` + classe de connection-params du transport
      (+ ``StdioServerParameters`` pour stdio) ; helper
      ``<id> = McpToolset(connection_params=..., tool_filter=[...])`` ; ``ref`` = ``<id>``.
    - ``apihub`` : import ``APIHubToolset`` ; helper
      ``<id> = APIHubToolset(apihub_resource_name="...")`` ; ``ref`` = ``<id>``.
    - ``langchain`` : import ``LangchainTool`` + la ligne d'import utilisateur (verbatim) ;
      ``ref`` = ``LangchainTool(tool=<tool_expr>)`` (pas de helper).
    - ``crewai`` : import ``CrewaiTool`` + ligne d'import utilisateur ;
      ``ref`` = ``CrewaiTool(tool=<tool_expr>, name=..., description=...)``.

    Auth (openapi/apihub/mcp_toolset) : si ``tool.auth`` est défini, ``auth_credential=\
    AuthCredential(...)`` est ajouté aux kwargs du helper + imports ``google.adk.auth``.

    Forme héritée (``str``) : rendue **telle quelle** (référence bare déjà importée), sans
    import ni helper, pour compat ascendante avec le modèle P1.
    """
    if isinstance(tool, str):
        return ToolRender(imports=(), helpers=(), ref=tool)

    if tool.kind == "function":
        return ToolRender(imports=(), helpers=(_render_function_def(tool),), ref=tool.name)

    if tool.kind == "long_running":
        imp = f"from {_TOOLS_IMPORT_MODULE} import LongRunningFunctionTool"
        return ToolRender(
            imports=(imp,),
            helpers=(_render_function_def(tool),),
            ref=f"LongRunningFunctionTool(func={tool.name})",
        )

    if tool.kind == "builtin":
        return _render_builtin_ref(tool)

    if tool.kind == "agent_tool":
        imp = f"from {_TOOLS_IMPORT_MODULE} import AgentTool"
        return ToolRender(imports=(imp,), helpers=(), ref=f"AgentTool(agent={tool.target_agent})")

    if tool.kind == "openapi":
        return _render_openapi(tool)

    if tool.kind == "bigquery":
        return _render_gcp_toolset(tool, "BigQueryToolset", _BIGQUERY_IMPORT)

    if tool.kind == "spanner":
        return _render_gcp_toolset(tool, "SpannerToolset", _SPANNER_IMPORT)

    if tool.kind == "mcp_toolset":
        return _render_mcp_toolset(tool)

    if tool.kind == "apihub":
        return _render_apihub(tool)

    if tool.kind == "langchain":
        imports = (_LANGCHAIN_IMPORT, tool.import_line)
        return ToolRender(imports=imports, helpers=(), ref=f"LangchainTool(tool={tool.tool_expr})")

    if tool.kind == "crewai":
        imports = (_CREWAI_IMPORT, tool.import_line)
        ref = (
            f"CrewaiTool(tool={tool.tool_expr}, name={_py_str(tool.name)}, "
            f"description={_py_str(tool.description)})"
        )
        return ToolRender(imports=imports, helpers=(), ref=ref)

    raise ValueError(f"Genre d'outil non rendu : {tool.kind!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Rendu de l'auth (set_auth) — ``auth_credential=AuthCredential(...)`` + imports
# --------------------------------------------------------------------------- #
def _auth_credential_call(auth: AuthSpec) -> tuple[_Call, tuple[str, ...]]:
    """Construit le :class:`_Call` ``AuthCredential(...)`` + les imports ``google.adk.auth`` requis.

    Le schéma dicte ``auth_type`` et le sous-objet porté :

    - ``apikey`` -> ``api_key="..."`` ;
    - ``bearer`` -> ``http=HttpAuth(scheme="bearer", credentials=HttpCredentials(token="..."))`` ;
    - ``oauth2`` -> ``oauth2=OAuth2Auth(client_id=..., client_secret=..., [access_token=...])`` ;
    - ``service_account`` -> ``service_account=ServiceAccount(use_default_credential=True |
      scopes=[...])``.
    """
    cred = dict(auth.credential)
    auth_type = _AUTH_TYPE_FOR_SCHEME[auth.scheme]
    imports: list[str] = [f"from {_AUTH_IMPORT_MODULE} import AuthCredential, AuthCredentialTypes"]
    inner: str | _Call

    if auth.scheme == "apikey":
        inner = f"api_key={_py_str(cred['api_key'])}"
    elif auth.scheme == "bearer":
        imports.append(f"from {_AUTH_CRED_IMPORT_MODULE} import HttpAuth, HttpCredentials")
        creds = _Call("HttpCredentials", (f"token={_py_str(cred['token'])}",))
        http = _Call("HttpAuth", ('scheme="bearer"', _kwarg_call("credentials", creds)))
        inner = _kwarg_call("http", http)
    elif auth.scheme == "oauth2":
        imports.append(f"from {_AUTH_CRED_IMPORT_MODULE} import OAuth2Auth")
        oauth = _Call("OAuth2Auth", tuple(f"{k}={_py_str(v)}" for k, v in auth.credential))
        inner = _kwarg_call("oauth2", oauth)
    else:  # service_account
        imports.append(f"from {_AUTH_CRED_IMPORT_MODULE} import ServiceAccount")
        sa = _Call("ServiceAccount", tuple(_service_account_kwargs(cred)))
        inner = _kwarg_call("service_account", sa)

    call = _Call("AuthCredential", (f"auth_type=AuthCredentialTypes.{auth_type}", inner))
    return call, tuple(imports)


def _service_account_kwargs(cred: dict[str, str]) -> list[str]:
    """Liste des kwargs d'un ``ServiceAccount`` depuis le dict credential (booléens/listes gérés).

    ``use_default_credential`` : valeur ``"true"``/``"false"`` -> littéral booléen Python.
    ``scopes`` : valeur séparée par des virgules -> liste de chaînes.
    """
    parts: list[str] = []
    for key, value in cred.items():
        if key == "use_default_credential":
            parts.append(f"use_default_credential={_py_bool(value)}")
        elif key == "scopes":
            scopes = [s.strip() for s in value.split(",") if s.strip()]
            parts.append(f"scopes=[{', '.join(_py_str(s) for s in scopes)}]")
        else:
            parts.append(f"{key}={_py_str(value)}")
    return parts


def _py_bool(value: str) -> str:
    """``"true"``/``"1"``/``"yes"`` -> ``True`` (sinon ``False``) — littéral source Python."""
    return "True" if value.strip().lower() in ("true", "1", "yes") else "False"


@dataclass(frozen=True)
class _Call:
    """Représentation structurée d'un appel ``Callee(arg1, arg2, ...)`` pour le rendu ruff-stable.

    Chaque argument est soit une **chaîne atomique** déjà rendue (``"key=value"``, un littéral,
    une liste/dict inline), soit un :class:`_Call` imbriqué (rendu récursivement). On ne replie
    jamais l'intérieur d'un littéral atomique — seuls les ``_Call`` sont éclatés récursivement,
    ce qui suffit pour reproduire la sortie ``ruff format`` de nos constructions.
    """

    callee: str
    args: tuple[str | _Call, ...]


def _render_call(call: _Call, col: int, base_indent: int) -> str:
    """Rend un :class:`_Call` **stable pour ``ruff format``**.

    ``col`` = colonne où débute ce rendu (budget de largeur inline) ; ``base_indent`` =
    indentation de la **ligne logique** propriétaire (le corps éclaté est indenté de
    ``base_indent + 4``, comme ``ruff format``). Algorithme reproduit :

    - forme inline si elle tient dans :data:`LINE_LENGTH` à partir de ``col`` ;
    - sinon, éclatement **un argument par ligne** (indent ``base_indent+4``). La virgule finale
      (« magic trailing comma ») n'est ajoutée **que** si le call a **≥ 2 arguments** : un call à
      argument unique qui doit être replié met cet argument seul sur sa ligne **sans** virgule
      finale (comportement exact de ``ruff format`` — vérifié par introspection).

    Ne termine **pas** par ``\\n`` (l'appelant gère sauts de ligne / suffixe ``= var``).
    """
    inline = _call_inline(call)
    if col + len(inline) <= LINE_LENGTH:
        return inline
    inner_indent = base_indent + 4
    pad = " " * inner_indent
    multi = len(call.args) >= 2
    trailing = "," if multi else ""
    lines: list[str] = []
    for arg in call.args:
        if isinstance(arg, _Call):
            rendered = _render_call(arg, col=inner_indent, base_indent=inner_indent)
            lines.append(f"{pad}{rendered}{trailing}")
        else:
            lines.append(f"{pad}{arg}{trailing}")
    body = "\n".join(lines)
    return f"{call.callee}(\n{body}\n{' ' * base_indent})"


def _call_inline(call: _Call) -> str:
    """Forme inline complète d'un :class:`_Call` (récursive, sans sauts de ligne)."""
    parts = [a if isinstance(a, str) else _call_inline(a) for a in call.args]
    return f"{call.callee}({', '.join(parts)})"


def _kwarg_call(key: str, call: _Call) -> _Call:
    """Combine ``key=`` + un :class:`_Call` en un :class:`_Call` repliable (``callee=key=Callee``).

    :func:`_render_call` choisit ensuite inline (``key=Callee(...)``) ou éclaté
    (``key=Callee(\\n ... \\n)``) selon la largeur — exactement la forme ``ruff format``.
    """
    return _Call(callee=f"{key}={call.callee}", args=call.args)


def _render_toolset_helper(var: str, call: _Call) -> str:
    """Rend ``<var> = <Call>`` (récursivement replié) terminé par un seul ``\\n``.

    Le call débute à la colonne ``len(var) + 3`` (``"<var> = "``) ; le corps éclaté est indenté
    depuis ``base_indent=0`` (statement top-level) -> +4, conforme à ``ruff format``.
    """
    return f"{var} = {_render_call(call, col=len(var) + 3, base_indent=0)}\n"


def _maybe_auth_arg(tool: ToolSpec) -> tuple[list[str | _Call], tuple[str, ...]]:
    """Renvoie ``([auth_credential=...] | [], imports)`` pour un toolset auth-capable.

    Si ``tool.auth`` est défini, rend ``auth_credential=AuthCredential(...)`` (repliable) + les
    imports d'auth requis ; sinon, listes vides. (La validation garantit que seuls les genres
    auth-capables portent un ``auth``.)
    """
    if tool.auth is None:
        return [], ()
    call, imports = _auth_credential_call(tool.auth)
    return [_kwarg_call("auth_credential", call)], imports


def _render_openapi(tool: ToolSpec) -> ToolRender:
    """``<id> = OpenAPIToolset(spec_str=..., spec_str_type="json"[, auth_credential=...])``."""
    args: list[str | _Call] = [f"spec_str={_py_str(tool.spec)}", 'spec_str_type="json"']
    auth_args, auth_imports = _maybe_auth_arg(tool)
    args += auth_args
    helper = _render_toolset_helper(tool.name, _Call("OpenAPIToolset", tuple(args)))
    return ToolRender(imports=(_OPENAPI_IMPORT, *auth_imports), helpers=(helper,), ref=tool.name)


def _render_gcp_toolset(tool: ToolSpec, class_name: str, import_stmt: str) -> ToolRender:
    """``<id> = BigQueryToolset(<args>)`` / ``SpannerToolset(<args>)``.

    Les ``args`` sont des **expressions source** (pas des littéraux chaîne) : un utilisateur
    fournit p.ex. ``{"bigquery_tool_config": "my_cfg"}`` pour référencer une variable/objet
    construit ailleurs. Pas d'auth ici (ces toolsets utilisent ``credentials_config``).
    """
    args: tuple[str | _Call, ...] = tuple(f"{k}={v}" for k, v in tool.args)
    helper = _render_toolset_helper(tool.name, _Call(class_name, args))
    return ToolRender(imports=(import_stmt,), helpers=(helper,), ref=tool.name)


def _render_apihub(tool: ToolSpec) -> ToolRender:
    """``<id> = APIHubToolset(apihub_resource_name="..."[, auth_credential=...])``."""
    args: list[str | _Call] = [f"apihub_resource_name={_py_str(tool.apihub_resource_name)}"]
    auth_args, auth_imports = _maybe_auth_arg(tool)
    args += auth_args
    helper = _render_toolset_helper(tool.name, _Call("APIHubToolset", tuple(args)))
    return ToolRender(imports=(_APIHUB_IMPORT, *auth_imports), helpers=(helper,), ref=tool.name)


def _mcp_connection_params_call(tool: ToolSpec) -> tuple[_Call, tuple[str, ...]]:
    """Construit le :class:`_Call` ``connection_params=...`` selon le transport + imports requis.

    - ``stdio`` -> ``StdioConnectionParams(server_params=StdioServerParameters(command=...,
      args=[...]))`` (importe aussi ``StdioServerParameters`` depuis ``mcp``) ;
    - ``sse`` -> ``SseConnectionParams(url="..."[, headers={...}])`` ;
    - ``http`` -> ``StreamableHTTPConnectionParams(url="..."[, headers={...}])``.
    """
    params_cls = _MCP_TRANSPORTS[tool.transport]
    imports: list[str] = [f"from {_MCP_TOOLSET_IMPORT_MODULE} import McpToolset, {params_cls}"]
    if tool.transport == "stdio":
        imports.append(_MCP_STDIO_PARAMS_IMPORT)
        args_list = f"[{', '.join(_py_str(a) for a in tool.mcp_args)}]"
        server = _Call(
            "StdioServerParameters", (f"command={_py_str(tool.command)}", f"args={args_list}")
        )
        conn = _Call(params_cls, (_kwarg_call("server_params", server),))
        return conn, tuple(imports)
    # sse / http : url + headers optionnels.
    inner: list[str] = [f"url={_py_str(tool.url)}"]
    if tool.headers:
        headers = ", ".join(f"{_py_str(k)}: {_py_str(v)}" for k, v in tool.headers)
        inner.append(f"headers={{{headers}}}")
    conn = _Call(params_cls, tuple(inner))
    return conn, tuple(imports)


def _render_mcp_toolset(tool: ToolSpec) -> ToolRender:
    """``<id> = McpToolset(connection_params=...[, tool_filter=[...]][, auth_credential=...])``."""
    conn_call, conn_imports = _mcp_connection_params_call(tool)
    args: list[str | _Call] = [_kwarg_call("connection_params", conn_call)]
    if tool.tool_filter:
        flt = ", ".join(_py_str(f) for f in tool.tool_filter)
        args.append(f"tool_filter=[{flt}]")
    auth_args, auth_imports = _maybe_auth_arg(tool)
    args += auth_args
    helper = _render_toolset_helper(tool.name, _Call("McpToolset", tuple(args)))
    return ToolRender(imports=(*conn_imports, *auth_imports), helpers=(helper,), ref=tool.name)


# --------------------------------------------------------------------------- #
# Validation d'outils
# --------------------------------------------------------------------------- #
def validate_tool_spec(tool: ToolSpec, model: ProjectModel, owner: str) -> str | None:
    """Renvoie un message d'erreur si ``tool`` est invalide, sinon None.

    ``model``/``owner`` servent à valider ``agent_tool`` (cible existante et != propriétaire).
    """
    if tool.kind not in _TOOL_KINDS:
        return f"Genre d'outil inconnu : {tool.kind!r}. Connus : {', '.join(sorted(_TOOL_KINDS))}."

    # Auth : seuls openapi/apihub/mcp_toolset acceptent auth_scheme/auth_credential (confirmé).
    if tool.auth is not None:
        auth_error = _validate_auth(tool)
        if auth_error is not None:
            return auth_error

    if tool.kind in ("function", "long_running"):
        if not is_identifier(tool.name):
            return f"Nom de fonction invalide : {tool.name!r}. Attendu un identifiant Python."
        for pname, ptype, _default in tool.params:
            if not is_identifier(pname):
                return f"Nom de paramètre invalide : {pname!r}. Attendu un identifiant Python."
            if not _is_allowed_type(ptype):
                return (
                    f"Type de paramètre non supporté : {ptype!r} (param {pname!r}). "
                    f"Types autorisés : {', '.join(sorted(_ALLOWED_PARAM_TYPES))} "
                    "(ou ``X | None`` / ``list[X]`` de ceux-ci)."
                )
        if not _is_allowed_type(tool.returns):
            return f"Type de retour non supporté : {tool.returns!r}."
        return None

    if tool.kind == "builtin":
        if tool.builtin_kind not in BUILTIN_TOOLS:
            return (
                f"Builtin inconnu : {tool.builtin_kind!r}. "
                f"Connus : {', '.join(sorted(BUILTIN_TOOLS))}."
            )
        if tool.builtin_kind in ARG_BUILTINS:
            keys = {k for k, _ in tool.args}
            if not ({"data_store_id", "search_engine_id"} & keys):
                return (
                    f"{tool.builtin_kind!r} requiert un argument 'data_store_id' "
                    "(ou 'search_engine_id')."
                )
        return None

    if tool.kind == "agent_tool":
        if not is_identifier(tool.target_agent):
            return f"target_agent invalide : {tool.target_agent!r}. Attendu un identifiant Python."
        if tool.target_agent == owner:
            return f"Un agent ne peut pas s'envelopper lui-même comme AgentTool : {owner!r}."
        if model.get(tool.target_agent) is None:
            return f"Agent cible introuvable : {tool.target_agent!r}. Créez-le d'abord."
        return None

    if tool.kind == "openapi":
        if not is_identifier(tool.name):
            return f"Nom de toolset OpenAPI invalide : {tool.name!r} (identifiant Python attendu)."
        if not tool.spec.strip():
            return "La spec OpenAPI est vide."
        return None

    if tool.kind in ("bigquery", "spanner"):
        if not is_identifier(tool.name):
            return (
                f"Nom de toolset {tool.kind} invalide : {tool.name!r} (identifiant Python attendu)."
            )
        return None

    if tool.kind == "mcp_toolset":
        return _validate_mcp(tool)

    if tool.kind == "apihub":
        if not is_identifier(tool.name):
            return f"Nom de toolset APIHub invalide : {tool.name!r} (identifiant Python attendu)."
        if not tool.apihub_resource_name.strip():
            return "apihub_resource_name est vide (ex. 'projects/<p>/locations/<l>/apis/<a>')."
        return None

    if tool.kind in ("langchain", "crewai"):
        if not tool.import_line.strip():
            return f"{tool.kind} : import_line est vide (ex. 'from x.tools import MyTool')."
        if not tool.tool_expr.strip():
            return f"{tool.kind} : tool_expr est vide (ex. 'MyTool(arg=...)')."
        if tool.kind == "crewai" and not tool.name.strip():
            return "crewai : 'name' est requis (CrewaiTool exige un nom, keyword-only)."
        return None

    return None  # pragma: no cover


def _validate_mcp(tool: ToolSpec) -> str | None:
    """Valide un ``mcp_toolset`` : nom identifiant, transport connu, et champs requis."""
    if not is_identifier(tool.name):
        return f"Nom de toolset MCP invalide : {tool.name!r} (identifiant Python attendu)."
    if tool.transport not in _MCP_TRANSPORTS:
        return (
            f"Transport MCP inconnu : {tool.transport!r}. "
            f"Connus : {', '.join(sorted(_MCP_TRANSPORTS))}."
        )
    if tool.transport == "stdio":
        if not tool.command.strip():
            return "Transport 'stdio' : 'command' est requis (ex. 'npx')."
    elif not tool.url.strip():
        return f"Transport {tool.transport!r} : 'url' est requis."
    return None


def _validate_auth(tool: ToolSpec) -> str | None:
    """Valide une sous-spec ``auth`` : genre auth-capable, schéma connu, champs requis du schéma."""
    if tool.kind not in _AUTH_CAPABLE_KINDS:
        return (
            f"Le genre {tool.kind!r} n'accepte pas d'auth (auth_scheme/auth_credential). "
            f"Genres compatibles : {', '.join(sorted(_AUTH_CAPABLE_KINDS))} "
            "(bigquery/spanner utilisent plutôt un credentials_config)."
        )
    auth = tool.auth
    assert auth is not None  # garanti par l'appelant
    if auth.scheme not in _AUTH_SCHEMES:
        return (
            f"Schéma d'auth inconnu : {auth.scheme!r}. Connus : {', '.join(sorted(_AUTH_SCHEMES))}."
        )
    keys = {k for k, _ in auth.credential}
    required: dict[str, set[str]] = {
        "apikey": {"api_key"},
        "bearer": {"token"},
        "oauth2": {"client_id"},
        "service_account": set(),  # use_default_credential OU scopes : au moins une clé
    }
    missing = required[auth.scheme] - keys
    if missing:
        fields = ", ".join(sorted(missing))
        return f"Auth {auth.scheme!r} : champ(s) credential manquant(s) : {fields}."
    if auth.scheme == "service_account" and not keys:
        return "Auth 'service_account' : fournir 'use_default_credential' ou 'scopes'."
    return None


def _is_allowed_type(t: str) -> bool:
    """Type de param/retour autorisé : un type de base, ou une composition simple
    (``X | None``, ``list[X]``, ``dict[X, Y]``, ``Optional[X]``) de types de base."""
    t = t.strip()
    if t in _ALLOWED_PARAM_TYPES:
        return True
    # Union avec None : ``X | None`` ou ``None | X``.
    if "|" in t:
        return all(_is_allowed_type(part) for part in t.split("|"))
    # Génériques simples : list[...], dict[...], tuple[...], set[...], Optional[...].
    m = re.fullmatch(r"(list|dict|tuple|set|Optional)\[(.+)\]", t)
    if m is not None:
        inner = m.group(2)
        return all(_is_allowed_type(part) for part in inner.split(","))
    return False


def _render_kwargs(pairs: list[tuple[str, str]]) -> str:
    """Assemble des ``k=v`` déjà rendus en une liste d'arguments multi-lignes."""
    return "".join(f"    {key}={value},\n" for key, value in pairs)


def _render_list_kwarg(key: str, refs: list[str]) -> str:
    """Rend la **valeur** d'un kwarg liste (``tools``/``sub_agents``) façon ``ruff format``.

    Inline ``[a, b]`` si la ligne ``    {key}={value},`` tient dans :data:`LINE_LENGTH` ;
    sinon, liste multi-lignes (un élément par ligne, indent 8, virgule finale) — exactement
    ce que produirait ``ruff format`` au-delà de la limite. Ainsi le ``agent.py`` généré est
    déjà stable (``format --check`` ne reformatte rien).
    """
    inline = f"[{', '.join(refs)}]"
    # 4 (indent kwarg) + len("key=") + len(inline) + 1 (virgule finale).
    if 4 + len(key) + 1 + len(inline) + 1 <= LINE_LENGTH:
        return inline
    items = "".join(f"        {ref},\n" for ref in refs)
    return f"[\n{items}    ]"


def _render_llm(spec: AgentSpec) -> str:
    """Rend un ``LlmAgent(...)`` en omettant les kwargs vides/None."""
    pairs: list[tuple[str, str]] = [
        ("name", _py_str(spec.name)),
        ("model", _py_str(spec.model)),
        ("instruction", _py_str(spec.instruction)),
    ]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    if spec.output_key is not None:
        pairs.append(("output_key", _py_str(spec.output_key)))
    if spec.tools:
        refs = [render_tool_ref(t).ref for t in spec.tools]
        pairs.append(("tools", _render_list_kwarg("tools", refs)))
    if spec.sub_agents:
        pairs.append(("sub_agents", _render_list_kwarg("sub_agents", list(spec.sub_agents))))
    return f"{spec.name} = LlmAgent(\n{_render_kwargs(pairs)})\n"


def _render_workflow(spec: AgentSpec, class_name: str) -> str:
    """Rend un ``SequentialAgent``/``ParallelAgent`` (name + sub_agents + description?)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("sub_agents", _render_list_kwarg("sub_agents", list(spec.sub_agents))))
    return f"{spec.name} = {class_name}(\n{_render_kwargs(pairs)})\n"


def _render_loop(spec: AgentSpec) -> str:
    """Rend un ``LoopAgent`` (name + sub_agents + max_iterations + description?)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("sub_agents", _render_list_kwarg("sub_agents", list(spec.sub_agents))))
    pairs.append(("max_iterations", str(spec.max_iterations)))
    return f"{spec.name} = LoopAgent(\n{_render_kwargs(pairs)})\n"


def _custom_class_name(name: str) -> str:
    """Nom de classe PascalCase pour un agent custom (``my_agent`` -> ``MyAgentAgent``)."""
    pascal = "".join(part.capitalize() for part in name.split("_") if part)
    if not pascal:
        pascal = "Custom"
    return f"{pascal}Agent"


def _render_custom(spec: AgentSpec) -> tuple[str, str]:
    """Rend une sous-classe ``BaseAgent`` (stub) + une instance module-level.

    Retourne un tuple ``(class_block, instance_block)`` pour permettre au renderer
    de module d'insérer exactement 2 lignes vides entre les deux (PEP 8 E305).

    Le ``_run_async_impl`` est un **async generator** no-op (``return`` puis ``yield``
    inatteignable) — c'est la forme valide attendue par ADK (cf. agents.md).
    """
    class_name = _custom_class_name(spec.name)
    desc = _py_str(spec.description) if spec.description else _py_str("")
    class_block = (
        f"class {class_name}(BaseAgent):\n"
        f'    """Agent custom généré (stub). Complétez `_run_async_impl`."""\n'
        "\n"
        "    async def _run_async_impl(self, ctx):\n"
        "        # TODO: implémenter la logique de l'agent.\n"
        "        return\n"
        "        yield  # rend cette méthode un async generator (inatteignable)\n"
    )
    instance_block = f"{spec.name} = {class_name}(name={_py_str(spec.name)}, description={desc})\n"
    return class_block, instance_block


def _render_agent_blocks(spec: AgentSpec) -> list[str]:
    """Retourne la liste de blocs de code (1 ou 2) pour un agent donné.

    Un agent ``custom`` émet deux blocs distincts (classe + instance) afin que le
    renderer de module puisse insérer le bon nombre de lignes vides entre eux.
    Tous les autres types émettent un seul bloc d'assignation.
    """
    if spec.type == "llm":
        return [_render_llm(spec)]
    if spec.type in ("sequential", "parallel"):
        return [_render_workflow(spec, _CLASS_FOR_TYPE[spec.type])]
    if spec.type == "loop":
        return [_render_loop(spec)]
    if spec.type == "custom":
        class_block, instance_block = _render_custom(spec)
        return [class_block, instance_block]
    raise ValueError(f"Type d'agent non rendu : {spec.type!r}")  # pragma: no cover


def _render_agent(spec: AgentSpec) -> str:
    """Aiguille vers le renderer du bon type — retourne un seul bloc de texte.

    Note: pour un agent ``custom``, le bloc unique inclut la classe *et* l'instance
    séparées par une ligne vide interne. Utiliser ``_render_agent_blocks`` (liste) quand
    on a besoin du contrôle fin des espacements inter-blocs dans le module complet.
    """
    if spec.type == "custom":
        class_block, instance_block = _render_custom(spec)
        return class_block + "\n" + instance_block
    blocks = _render_agent_blocks(spec)
    return blocks[0]


def _needed_agent_imports(model: ProjectModel) -> list[str]:
    """Classes d'agents ADK réellement utilisées, dans l'ordre canonique."""
    used: set[str] = set()
    for a in model.agents:
        if a.type == "custom":
            used.add("BaseAgent")
        else:
            used.add(_CLASS_FOR_TYPE[a.type])
    return [name for name in _IMPORT_ORDER if name in used]


def _collect_tool_renders(ordered: list[AgentSpec]) -> list[ToolRender]:
    """Rend tous les outils des agents (dans l'ordre topo fourni) en une liste de ``ToolRender``.

    L'ordre topologique garantit qu'un ``agent_tool`` ciblant un agent voit cet agent défini
    avant l'agent enveloppant (les helpers d'outils sont émis avant *tous* les agents, mais la
    cible étant elle-même un agent, son instance précède l'enveloppant dans la section agents).
    """
    renders: list[ToolRender] = []
    for spec in ordered:
        for tool in spec.tools:
            renders.append(render_tool_ref(tool))
    return renders


def _dedup_preserve(items: list[str]) -> list[str]:
    """Déduplique en préservant l'ordre de première apparition."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _agent_import_line(model: ProjectModel) -> str:
    """Ligne d'import des classes d'agents (vide si aucune classe agent utilisée)."""
    imports = _needed_agent_imports(model)
    if not imports:
        return ""
    return f"from google.adk.agents import {', '.join(imports)}\n"


def _merge_tool_imports(import_stmts: list[str]) -> list[str]:
    """Fusionne/trie des ``from <module> import <name>`` façon isort (stable pour ruff ``I``).

    - Regroupe par module ; fusionne les noms (dédupliqués, triés) sur une seule ligne.
    - Trie les modules par ordre alphabétique.
    Toute ligne non reconnue (improbable ici) est conservée telle quelle, en tête.
    """
    by_module: dict[str, set[str]] = {}
    passthrough: list[str] = []
    for stmt in import_stmts:
        m = re.fullmatch(r"from (\S+) import (.+)", stmt.strip())
        if m is None:
            passthrough.append(stmt)
            continue
        module, names = m.group(1), m.group(2)
        bucket = by_module.setdefault(module, set())
        for name in names.split(","):
            bucket.add(name.strip())
    merged = [
        _render_import_line(module, sorted(by_module[module])) for module in sorted(by_module)
    ]
    return _dedup_preserve(passthrough) + merged


def _render_import_line(module: str, names: list[str]) -> str:
    """Rend ``from <module> import a, b`` **stable pour ``ruff format``**.

    Inline si la ligne tient dans :data:`LINE_LENGTH` ; sinon, forme parenthésée multi-lignes
    (un nom par ligne, indent 4, virgule finale) — exactement ce que ``ruff format`` produit
    au-delà de la limite pour un import à noms multiples.
    """
    inline = f"from {module} import {', '.join(names)}"
    if len(inline) <= LINE_LENGTH:
        return inline
    body = "".join(f"    {name},\n" for name in names)
    return f"from {module} import (\n{body})"


# --------------------------------------------------------------------------- #
# Rendu de source — module complet
# --------------------------------------------------------------------------- #
def render_agent_module(model: ProjectModel) -> str:
    """Produit une source ``agent.py`` valide à partir du modèle.

    - Importe uniquement les classes utilisées (ordre canonique).
    - Définit chaque agent comme variable de module, **triées topologiquement** (un
      enfant avant son parent). Cycle -> ``ValueError``.
    - Omet les kwargs vides/None.
    - Termine par ``root_agent = <root>`` (ou un commentaire clair si racine non définie).
    """
    header = (
        '"""Généré par adk-toolkit-mcp. NE PAS éditer à la main : '
        "régénéré depuis le sidecar.\n\n"
        "Source de vérité : `.adk_toolkit/agents.json`.\n"
        '"""\n\n'
    )

    if not model.agents:
        body = "# Aucun agent défini dans le modèle.\n"
        root_line = "# root_agent non défini : ajoutez un agent puis appelez set_root.\n"
        return header + body + "\n" + root_line

    ordered = topological_order(model)  # peut lever ValueError (cycle)

    # Rendu des outils (imports + helpers + refs) dans l'ordre topo des agents propriétaires.
    tool_renders = _collect_tool_renders(ordered)
    tool_helpers = [helper for tr in tool_renders for helper in tr.helpers]

    # Section d'imports. La ligne des classes d'agents garde l'**ordre canonique** ADK
    # (LlmAgent, Sequential, Parallel, Loop, BaseAgent) — pas un tri alphabétique. Les imports
    # d'outils sont fusionnés par module (noms dédupliqués + triés, un module par ligne) et
    # placés après. ``ruff format`` ne réordonne pas les imports : la stabilité de format est
    # préservée (le tri isort n'est pas requis pour le fichier généré, jamais linté en repo).
    import_stmts: list[str] = []
    agent_imports = _agent_import_line(model)
    if agent_imports:
        import_stmts.append(agent_imports.rstrip("\n"))
    import_stmts.extend(_merge_tool_imports([imp for tr in tool_renders for imp in tr.imports]))
    # Bloc d'imports terminé par une ligne vide (séparation avec le corps).
    import_block = ("\n".join(import_stmts) + "\n\n") if import_stmts else ""

    # Blocs top-level : d'abord les helpers d'outils (defs/toolsets), puis les agents.
    # Chaque agent émet 1 bloc (llm/workflow/loop) ou 2 (custom : classe + instance).
    agent_blocks: list[str] = []
    for spec in ordered:
        agent_blocks.extend(_render_agent_blocks(spec))
    all_blocks: list[str] = tool_helpers + agent_blocks

    # PEP 8 / ruff-format spacing rules (E302, E303, E305):
    #   - Exactly 2 blank lines before a top-level class/def block.
    #   - Exactly 2 blank lines after a top-level class/def block.
    #   - 1 blank line between plain assignment blocks.
    #
    # Each block already ends with exactly one '\n'.
    # Separator '\n'  between two blocks → 1 blank line total (last \n + sep \n).
    # Separator '\n\n' between two blocks → 2 blank lines total.
    def _starts_class_or_def(block: str) -> bool:
        return block.startswith("class ") or block.startswith("def ")

    parts: list[str] = []
    for i, block in enumerate(all_blocks):
        parts.append(block)
        if i < len(all_blocks) - 1:
            next_block = all_blocks[i + 1]
            # 2 blank lines when leaving or entering a class/def block.
            if _starts_class_or_def(block) or _starts_class_or_def(next_block):
                parts.append("\n\n")
            else:
                parts.append("\n")
    blocks = "".join(parts)

    # The import block ends with '\n' (1 blank line).  If the first rendered block is a
    # class/def we need one more blank line to satisfy E302 (2 blank lines before class/def).
    if import_block and all_blocks and _starts_class_or_def(all_blocks[0]):
        import_block = import_block + "\n"

    if model.root is not None and model.get(model.root) is not None:
        root_line = f"\nroot_agent = {model.root}\n"
    elif model.root is not None:
        root_line = (
            f"\n# root '{model.root}' introuvable parmi les agents ; root_agent non défini.\n"
        )
    else:
        root_line = "\n# root_agent non défini : appelez set_root pour désigner la racine.\n"

    return header + import_block + blocks + root_line


# --------------------------------------------------------------------------- #
# Régénération sur disque
# --------------------------------------------------------------------------- #
def regenerate(ws: Workspace, model: ProjectModel) -> dict[str, Any]:
    """Écrit ``agent.py`` (rendu) + assure ``__init__.py``. Idempotent.

    Renvoie ``{"agent_py", "init_py", "changed"}`` (chemins absolus, drapeau global).
    Peut lever ``ValueError`` (cycle) — l'outil appelant le convertit en ``err``.
    """
    source = render_agent_module(model)
    agent_changed = ws.write("agent.py", source)
    init_changed = ws.write("__init__.py", "from . import agent\n")
    return {
        "agent_py": str(ws.path("agent.py")),
        "init_py": str(ws.path("__init__.py")),
        "changed": agent_changed or init_changed,
    }
