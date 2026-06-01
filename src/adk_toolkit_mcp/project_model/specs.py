"""Dataclasses, constantes et alias ``Literal`` du modèle de projet ADK.

Ce module regroupe la **surface de données pure** du modèle (aucune dépendance à
``google-adk`` ni I/O) : les constantes de domaine (providers LiteLLM, catégories/seuils
``Harm*``, genres d'outils/builtins, chemins du sidecar, ordre d'import canonique), les
alias ``Literal`` (:data:`AgentType`, :data:`ToolKind`) et les dataclasses **immuables**
(:class:`AuthSpec`, :class:`ToolSpec`, :class:`ToolRender`, :class:`LiteLlmSpec`,
:class:`SafetySettingSpec`, :class:`GenerateContentConfigSpec`, :class:`AgentSpec`,
:class:`ProjectModel`), plus le petit validateur d'identifiant :func:`is_identifier`.

Importé tel quel par :mod:`adk_toolkit_mcp.project_model.sidecar` (I/O + mutations) et
:mod:`adk_toolkit_mcp.project_model.render` (génération de ``agent.py``). La surface publique
historique reste ré-exportée depuis ``adk_toolkit_mcp.project_model``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Constantes modèles
# --------------------------------------------------------------------------- #
#: Providers LiteLLM supportés (validation domaine models).
LITELLM_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "anthropic",
        "ollama",
        "ollama_chat",
        "openrouter",
        "vllm",
        "lm_studio",
        "gemini",
    }
)

#: Membres valides de ``HarmCategory`` (confirmés par introspection google-genai).
HARM_CATEGORIES: frozenset[str] = frozenset(
    {
        "HARM_CATEGORY_UNSPECIFIED",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
        "HARM_CATEGORY_IMAGE_HATE",
        "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
        "HARM_CATEGORY_IMAGE_HARASSMENT",
        "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_JAILBREAK",
    }
)

#: Membres valides de ``HarmBlockThreshold`` (confirmés par introspection google-genai).
HARM_BLOCK_THRESHOLDS: frozenset[str] = frozenset(
    {
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE",
        "OFF",
    }
)

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
#: Dossier du sidecar, relatif au dossier de l'app (`<path>/<app_name>`).
SIDECAR_DIR = ".adk_toolkit"

#: Nom du fichier sidecar dans `SIDECAR_DIR`.
SIDECAR_FILE = "agents.json"

#: Chemin relatif complet du sidecar (depuis le dossier de l'app).
SIDECAR_PATH = f"{SIDECAR_DIR}/{SIDECAR_FILE}"

#: Types d'agents supportés. ``remote_a2a`` (P4b) = un proxy ``RemoteA2aAgent`` consommant un
#: agent distant via son agent-card (URL ou chemin JSON) ; il n'a pas d'enfants mais peut être
#: membre de ``sub_agents`` d'un autre agent.
AgentType = Literal["llm", "sequential", "parallel", "loop", "custom", "remote_a2a"]

_AGENT_TYPES: frozenset[str] = frozenset(
    {"llm", "sequential", "parallel", "loop", "custom", "remote_a2a"}
)

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
    "remote_a2a": "RemoteA2aAgent",
    # `custom` produit une sous-classe de BaseAgent.
}

#: Ordre canonique d'import (sous-ensemble effectivement utilisé est conservé). Les classes
#: importées depuis ``google.adk.agents`` UNIQUEMENT (RemoteA2aAgent vit dans un autre module —
#: cf. :data:`_REMOTE_A2A_IMPORT` — et n'apparaît donc PAS ici).
_IMPORT_ORDER: tuple[str, ...] = (
    "LlmAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "BaseAgent",
)

#: Import (chemin réel confirmé en 2.1.0 par introspection) de ``RemoteA2aAgent``. ⚠️ Cette
#: classe N'EST PAS dans ``google.adk.agents`` (ni dans son ``__all__``, ni en lazy getattr) :
#: le seul chemin valide est ce sous-module — qui requiert l'extra ``a2a`` au runtime utilisateur.
#: Codegen-only : le toolkit ne l'importe jamais lui-même.
_REMOTE_A2A_IMPORT = "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent"


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
class LiteLlmSpec:
    """Spécification immuable d'un modèle LiteLLM.

    ``provider`` ∈ :data:`LITELLM_PROVIDERS`. Pour ``lm_studio``, le provider est rendu
    comme ``openai`` dans le code généré, et ``api_base`` vaut par défaut
    ``http://127.0.0.1:1234/v1``.

    ``api_key_env`` : si fourni, le code généré inclut ``api_key=os.getenv("<ENV>")`` ; sinon
    ``api_key`` est omis (LiteLLM lit les variables d'env du provider automatiquement).
    **La clé n'est jamais écrite en dur.**
    """

    provider: str
    model: str
    api_base: str = ""
    api_key_env: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.api_base:
            d["api_base"] = self.api_base
        if self.api_key_env:
            d["api_key_env"] = self.api_key_env
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiteLlmSpec:
        return cls(
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            api_base=str(data.get("api_base", "")),
            api_key_env=str(data.get("api_key_env", "")),
        )


@dataclass(frozen=True)
class SafetySettingSpec:
    """Spécification immuable d'un SafetySetting (``category`` + ``threshold``).

    Les valeurs sont des **noms de membres enum** (ex. ``"HARM_CATEGORY_HARASSMENT"``,
    ``"BLOCK_MEDIUM_AND_ABOVE"``) — validées contre :data:`HARM_CATEGORIES` /
    :data:`HARM_BLOCK_THRESHOLDS`.
    """

    category: str
    threshold: str

    def to_dict(self) -> dict[str, str]:
        return {"category": self.category, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetySettingSpec:
        return cls(category=str(data.get("category", "")), threshold=str(data.get("threshold", "")))


@dataclass(frozen=True)
class GenerateContentConfigSpec:
    """Spécification immuable d'un ``types.GenerateContentConfig``.

    Seuls les champs non-None sont rendus dans le code généré.
    ``safety_settings`` est un tuple de :class:`SafetySettingSpec` (gelé pour l'immuabilité).
    """

    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: float | None = None
    safety_settings: tuple[SafetySettingSpec, ...] = ()
    response_modalities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            d["max_output_tokens"] = self.max_output_tokens
        if self.top_p is not None:
            d["top_p"] = self.top_p
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.safety_settings:
            d["safety_settings"] = [s.to_dict() for s in self.safety_settings]
        if self.response_modalities:
            d["response_modalities"] = list(self.response_modalities)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerateContentConfigSpec:
        raw_ss = data.get("safety_settings") or []
        safety = tuple(SafetySettingSpec.from_dict(s) for s in raw_ss)
        raw_rm = data.get("response_modalities") or []
        return cls(
            temperature=data.get("temperature"),
            max_output_tokens=data.get("max_output_tokens"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            safety_settings=safety,
            response_modalities=tuple(str(m) for m in raw_rm),
        )


@dataclass(frozen=True)
class AgentSpec:
    """Spécification immuable d'un agent dans le modèle de projet.

    Les champs non pertinents pour un type donné restent à leur valeur par défaut
    (ex. ``model``/``instruction`` ignorés pour un agent ``sequential``).

    ``model`` : chaîne Gemini (compat ascendante, ex. ``"gemini-2.5-flash"``).
    ``model_spec`` : si non-None, un :class:`LiteLlmSpec` ; prend la priorité sur ``model``
    pour le rendu du champ ``model=`` de ``LlmAgent``.
    ``generate_content_config`` : si non-None, un :class:`GenerateContentConfigSpec` ; rendu
    comme ``generate_content_config=types.GenerateContentConfig(...)`` sur ``LlmAgent``.
    """

    name: str
    type: AgentType
    model: str = "gemini-2.5-flash"
    instruction: str = ""
    description: str = ""
    output_key: str | None = None
    #: URL (ou chemin JSON local) de l'agent-card distant, pour le type ``remote_a2a`` uniquement.
    #: Rendu comme ``RemoteA2aAgent(name=..., agent_card="<url>")``. Ignoré pour les autres types.
    agent_card: str = ""
    #: Outils attachés. ``ToolSpec`` (codegen riche) ; la forme ``str`` héritée (P1) reste
    #: tolérée et rendue comme une référence bare (nom déjà importé). Voir ``render_tool_ref``.
    tools: tuple[ToolSpec | str, ...] = ()
    sub_agents: tuple[str, ...] = ()
    max_iterations: int = 3
    #: Spec LiteLLM (P4). Si non-None, prend la priorité sur ``model`` pour le rendu.
    model_spec: LiteLlmSpec | None = None
    #: Config generate_content (P4). Rendu comme ``generate_content_config=...``.
    generate_content_config: GenerateContentConfigSpec | None = None

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
            if self.model_spec is not None:
                base["model_spec"] = self.model_spec.to_dict()
            if self.generate_content_config is not None:
                base["generate_content_config"] = self.generate_content_config.to_dict()
        elif self.type in ("sequential", "parallel"):
            base["sub_agents"] = list(self.sub_agents)
        elif self.type == "loop":
            base["sub_agents"] = list(self.sub_agents)
            base["max_iterations"] = self.max_iterations
        elif self.type == "remote_a2a":
            base["agent_card"] = self.agent_card
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
        raw_ms = data.get("model_spec")
        model_spec = LiteLlmSpec.from_dict(raw_ms) if isinstance(raw_ms, dict) else None
        raw_gcc = data.get("generate_content_config")
        generate_content_config = (
            GenerateContentConfigSpec.from_dict(raw_gcc) if isinstance(raw_gcc, dict) else None
        )
        return cls(
            name=str(data["name"]),
            type=atype,
            model=str(data.get("model", "gemini-2.5-flash")),
            instruction=str(data.get("instruction", "")),
            description=str(data.get("description", "")),
            output_key=data.get("output_key"),
            agent_card=str(data.get("agent_card", "")),
            tools=tools,
            sub_agents=tuple(data.get("sub_agents", []) or []),
            max_iterations=int(data.get("max_iterations", 3)),
            model_spec=model_spec,
            generate_content_config=generate_content_config,
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
# Validation — identifiant Python
# --------------------------------------------------------------------------- #
def is_identifier(name: str) -> bool:
    """True si ``name`` est un identifiant Python valide (nom de variable de module)."""
    return bool(_IDENT_RE.match(name))
