"""Dataclasses, constants and ``Literal`` aliases of the ADK project model.

This module groups the model's **pure data surface** (no dependency on ``google-adk`` and no
I/O): the domain constants (LiteLLM providers, ``Harm*`` categories/thresholds, tool/builtin
kinds, sidecar paths, canonical import order), the ``Literal`` aliases (:data:`AgentType`,
:data:`ToolKind`) and the **immutable** dataclasses (:class:`AuthSpec`, :class:`ToolSpec`,
:class:`ToolRender`, :class:`LiteLlmSpec`, :class:`SafetySettingSpec`,
:class:`GenerateContentConfigSpec`, :class:`AgentSpec`, :class:`ProjectModel`), plus the small
identifier validator :func:`is_identifier`.

Imported as-is by :mod:`adk_toolkit_mcp.project_model.sidecar` (I/O + mutations) and
:mod:`adk_toolkit_mcp.project_model.render` (generation of ``agent.py``). The historical public
surface stays re-exported from ``adk_toolkit_mcp.project_model``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- #
# Model constants
# --------------------------------------------------------------------------- #
#: Supported LiteLLM providers (models domain validation).
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

#: Valid ``HarmCategory`` members (confirmed by google-genai introspection).
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

#: Valid ``HarmBlockThreshold`` members (confirmed by google-genai introspection).
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
# Constants
# --------------------------------------------------------------------------- #
#: Sidecar folder, relative to the app folder (`<path>/<app_name>`).
SIDECAR_DIR = ".adk_toolkit"

#: Sidecar file name inside `SIDECAR_DIR`.
SIDECAR_FILE = "agents.json"

#: Full relative path of the sidecar (from the app folder).
SIDECAR_PATH = f"{SIDECAR_DIR}/{SIDECAR_FILE}"

#: Supported agent types. ``remote_a2a`` (P4b) = a ``RemoteA2aAgent`` proxy consuming a remote
#: agent via its agent-card (URL or JSON path); it has no children but can be a member of
#: another agent's ``sub_agents``.
AgentType = Literal["llm", "sequential", "parallel", "loop", "custom", "remote_a2a"]

_AGENT_TYPES: frozenset[str] = frozenset(
    {"llm", "sequential", "parallel", "loop", "custom", "remote_a2a"}
)

# --------------------------------------------------------------------------- #
# Callbacks (guardrails) — `safety` domain, P4c
# --------------------------------------------------------------------------- #
#: Supported callback hooks on an ``LlmAgent`` (real kwargs confirmed by introspection in 2.1.0
#: — cf. ``docs/adk-api-notes/safety-observability.md``). The toolkit attaches ONE generated
#: function per hook via the real kwarg (e.g. ``before_model_callback=...``).
CallbackHook = Literal[
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "before_agent",
    "after_agent",
]

_CALLBACK_HOOKS: frozenset[str] = frozenset(
    {"before_model", "after_model", "before_tool", "after_tool", "before_agent", "after_agent"}
)

#: Mapping hook -> real kwarg name on ``LlmAgent`` (adds the ``_callback`` suffix).
_CALLBACK_KWARG: dict[str, str] = {h: f"{h}_callback" for h in _CALLBACK_HOOKS}

#: Supported guardrail policies (concrete + functional). Each is only valid for certain hooks
#: (cf. :data:`_POLICY_HOOKS`). See ``_codegen._render_callback_def`` for the rendering.
#: - ``block_keywords`` (before_model): refuses if the user text contains a blocked term.
#: - ``max_input_chars`` (before_model): refuses if the input exceeds N characters.
#: - ``block_tool`` (before_tool): short-circuits the tool if its name is in a denylist.
PolicyKind = Literal["block_keywords", "max_input_chars", "block_tool"]

_POLICY_KINDS: frozenset[str] = frozenset({"block_keywords", "max_input_chars", "block_tool"})

#: Public aliases (without underscore) re-exported for validation on the ``safety`` domain side.
CALLBACK_HOOKS: frozenset[str] = _CALLBACK_HOOKS
POLICY_KINDS: frozenset[str] = _POLICY_KINDS

#: Hooks compatible with each policy (validation: a policy can only attach to a hook whose
#: signature suits it).
_POLICY_HOOKS: dict[str, frozenset[str]] = {
    "block_keywords": frozenset({"before_model"}),
    "max_input_chars": frozenset({"before_model"}),
    "block_tool": frozenset({"before_tool"}),
}

#: Default refusal message rendered by a ``before_model`` guardrail that short-circuits the LLM.
_DEFAULT_REFUSAL = "I can't help with that request."

#: Prefix of the generated guardrail function names (e.g. ``_guard_before_model_0``).
_GUARD_FN_PREFIX = "_guard"

#: An agent name must be a Python identifier (it serves as a module variable name).
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Target line length (must mirror ``[tool.ruff] line-length`` from pyproject) so that the
#: generated code is already in the form produced by ``ruff format`` (idempotence).
LINE_LENGTH = 100

# --------------------------------------------------------------------------- #
# Tools (`tools` domain, passes 3a + 3b)
# --------------------------------------------------------------------------- #
#: Supported tool kinds. 3a (no dependency): ``function``, ``long_running``, ``builtin``,
#: ``agent_tool``, ``openapi``. 3b (optional dependency / ``google-adk[...]`` extras,
#: codegen-only): ``bigquery``, ``spanner``, ``mcp_toolset``, ``apihub``, ``langchain``,
#: ``crewai``.
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

#: "Toolset" kinds whose ``ref`` is a module-level variable (``<id>`` in ``tools=[...]``) built
#: by a helper block. The :class:`ToolSpec` ``name`` serves as the variable identifier.
_TOOLSET_VAR_KINDS: frozenset[str] = frozenset(
    {"openapi", "bigquery", "spanner", "mcp_toolset", "apihub"}
)

#: "Toolset" kinds that natively accept ``auth_scheme=`` / ``auth_credential=`` (confirmed by
#: introspection: ``OpenAPIToolset``, ``McpToolset``, ``APIHubToolset``). ``BigQueryToolset`` /
#: ``SpannerToolset`` do not (they take a ``credentials_config``) -> auth rejected.
_AUTH_CAPABLE_KINDS: frozenset[str] = frozenset({"openapi", "apihub", "mcp_toolset"})

#: Supported MCP transports -> ADK connection-params class (confirmed by introspection).
_MCP_TRANSPORTS: dict[str, str] = {
    "stdio": "StdioConnectionParams",
    "sse": "SseConnectionParams",
    "http": "StreamableHTTPConnectionParams",
}

#: Auth schemes supported by :func:`set_auth` -> ``AuthCredentialTypes`` member (confirmed).
_AUTH_SCHEMES: frozenset[str] = frozenset({"apikey", "oauth2", "service_account", "bearer"})

_AUTH_TYPE_FOR_SCHEME: dict[str, str] = {
    "apikey": "API_KEY",
    "bearer": "HTTP",
    "oauth2": "OAUTH2",
    "service_account": "SERVICE_ACCOUNT",
}

#: ADK "core" builtins: already-exported tool instances (no argument required).
#: Confirmed by introspection in google-adk 2.1.0 (cf. ``docs/adk-api-notes/tools.md``).
#: These are **instances** (e.g. ``google_search`` = ``GoogleSearchTool()``) or functions
#: (``exit_loop``, ``transfer_to_agent``) — they go as-is into ``tools=[...]`` and are imported
#: from ``google.adk.tools``.
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

#: Builtins requiring an argument (rendered as a constructor call).
#: ``vertex_ai_search`` -> ``VertexAiSearchTool(data_store_id=... | search_engine_id=...)``.
ARG_BUILTINS: frozenset[str] = frozenset({"vertex_ai_search"})

#: Complete set of recognized builtin ``kind`` values.
BUILTIN_TOOLS: frozenset[str] = CORE_BUILTINS | ARG_BUILTINS

#: Mapping of an arg-requiring builtin -> imported ADK class name.
_BUILTIN_CLASS: dict[str, str] = {"vertex_ai_search": "VertexAiSearchTool"}

#: Python types allowed for a function-tool's parameters (lightweight validation).
_ALLOWED_PARAM_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "bool", "list", "dict", "tuple", "set", "bytes", "Any", "None"}
)

#: Import from which the tool classes/builtins are pulled (package root).
_TOOLS_IMPORT_MODULE = "google.adk.tools"

#: Import (confirmed real path) for ``OpenAPIToolset``.
_OPENAPI_IMPORT = "from google.adk.tools.openapi_tool import OpenAPIToolset"

#: Imports (real paths confirmed by introspection in 2.1.0) of the 3b toolsets.
_BIGQUERY_IMPORT = "from google.adk.tools.bigquery import BigQueryToolset"
_SPANNER_IMPORT = "from google.adk.tools.spanner import SpannerToolset"
_APIHUB_IMPORT = "from google.adk.tools.apihub_tool import APIHubToolset"
#: Note (cf. docs/adk-api-notes/tools.md): these two paths re-export from
#: ``google.adk.integrations.*`` and emit a ``DeprecationWarning`` at the user's runtime; we keep
#: the path requested by the task (still functional, codegen-only).
_LANGCHAIN_IMPORT = "from google.adk.tools.langchain_tool import LangchainTool"
_CREWAI_IMPORT = "from google.adk.tools.crewai_tool import CrewaiTool"

#: Module of the auth classes (confirmed re-export).
_AUTH_IMPORT_MODULE = "google.adk.auth"
#: Module of the auth sub-objects (HttpAuth/OAuth2Auth/ServiceAccount/HttpCredentials).
_AUTH_CRED_IMPORT_MODULE = "google.adk.auth.auth_credential"
#: MCP imports (toolset + StdioServerParameters from the ``mcp`` package).
_MCP_TOOLSET_IMPORT_MODULE = "google.adk.tools.mcp_tool"
_MCP_STDIO_PARAMS_IMPORT = "from mcp import StdioServerParameters"

#: Mapping of agent type -> ADK class name to import.
_CLASS_FOR_TYPE: dict[str, str] = {
    "llm": "LlmAgent",
    "sequential": "SequentialAgent",
    "parallel": "ParallelAgent",
    "loop": "LoopAgent",
    "remote_a2a": "RemoteA2aAgent",
    # `custom` produces a BaseAgent subclass.
}

#: Canonical import order (only the subset actually used is kept). The classes imported from
#: ``google.adk.agents`` ONLY (RemoteA2aAgent lives in another module — cf.
#: :data:`_REMOTE_A2A_IMPORT` — and so does NOT appear here).
_IMPORT_ORDER: tuple[str, ...] = (
    "LlmAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "BaseAgent",
)

#: Import (real path confirmed in 2.1.0 by introspection) of ``RemoteA2aAgent``. WARNING: this
#: class is NOT in ``google.adk.agents`` (neither in its ``__all__`` nor via lazy getattr): the
#: only valid path is this submodule — which requires the ``a2a`` extra at the user's runtime.
#: Codegen-only: the toolkit never imports it itself.
_REMOTE_A2A_IMPORT = "from google.adk.agents.remote_a2a_agent import RemoteA2aAgent"


# --------------------------------------------------------------------------- #
# Model dataclasses (immutable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuthSpec:
    """Auth sub-spec attached to a toolset (3b).

    ``scheme`` ∈ :data:`_AUTH_SCHEMES` (``apikey``/``oauth2``/``service_account``/``bearer``).
    ``credential`` is a list of ``(key, literal-value)`` pairs (frozen into a tuple to stay
    hashable/immutable) rendered into an ``AuthCredential(...)`` according to the scheme — see
    :func:`_render_auth_credential` and ``docs/adk-api-notes/tools.md``.
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
    """Immutable spec of a tool attached to an agent (`tools` domain, 3a + 3b).

    The ``kind`` field discriminates; only the relevant fields are populated/serialized:

    - ``function`` / ``long_running``: ``name`` (identifier), ``params`` (tuple of
      ``(name, type, default|None)``), ``docstring``, ``returns``, ``body``.
    - ``builtin``: ``builtin_kind`` (member of :data:`BUILTIN_TOOLS`), ``args`` (for
      ``vertex_ai_search``: ``{"data_store_id": ...}`` or ``{"search_engine_id": ...}``).
    - ``agent_tool``: ``target_agent`` (name of an **existing** agent in the model).
    - ``openapi``: ``name`` (toolset variable identifier), ``spec`` (OpenAPI string).
    - ``bigquery`` / ``spanner``: ``name`` (toolset var), ``args`` (kwargs that are source
      *expressions*, e.g. ``{"bigquery_tool_config": "my_cfg"}``).
    - ``mcp_toolset``: ``name`` (var), ``transport`` ∈ {stdio,sse,http}, ``command``+``mcp_args``
      (stdio) or ``url``+``headers`` (sse/http), ``tool_filter``.
    - ``apihub``: ``name`` (var), ``apihub_resource_name``.
    - ``langchain`` / ``crewai``: ``import_line`` (rendered verbatim), ``tool_expr`` (construction
      expression), + ``name``/``description`` (crewai: ``name`` required).
    - ``auth`` (optional, openapi/apihub/mcp_toolset): :class:`AuthSpec` rendered as
      ``auth_credential=``.

    ``ref_key`` returns a stable identity key used for "replace by name" (append unique /
    replace) on the domain side.
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
    # --- 3b: fields of the optional-dependency toolsets ---
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
        """Uniqueness key (used for append-unique / replace-by-name on the domain side).

        - "toolset variable" kinds (``openapi``/``bigquery``/``spanner``/``mcp_toolset``/
          ``apihub``) + ``function``/``long_running`` -> ``<kind>:<name>``;
        - ``builtin`` -> ``builtin:<builtin_kind>``; ``agent_tool`` -> ``agent_tool:<target>``;
        - ``langchain``/``crewai`` -> ``<kind>:<tool_expr>`` (the expression identifies the tool;
          ``crewai`` can also rename via ``name`` but the expression stays the identity).
        """
        if self.kind in ("function", "long_running") or self.kind in _TOOLSET_VAR_KINDS:
            return f"{self.kind}:{self.name}"
        if self.kind == "builtin":
            return f"builtin:{self.builtin_kind}"
        if self.kind == "agent_tool":
            return f"agent_tool:{self.target_agent}"
        if self.kind in ("langchain", "crewai"):
            return f"{self.kind}:{self.tool_expr}"
        return self.kind  # pragma: no cover (kind validated upstream)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the sidecar's JSON form (relevant fields per ``kind``)."""
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
        """Deserialize a ``tools`` entry from the sidecar.

        Tolerant of the **legacy form** (P1) where a tool entry was a plain string (a name
        already imported in the module): we map it to a ``builtin`` (rendered bare).
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
    """Result of rendering a tool: required imports, top-level helper blocks, and the
    reference to place in the owning agent's ``tools=[...]``."""

    imports: tuple[str, ...]
    helpers: tuple[str, ...]
    ref: str


@dataclass(frozen=True)
class LiteLlmSpec:
    """Immutable spec of a LiteLLM model.

    ``provider`` ∈ :data:`LITELLM_PROVIDERS`. For ``lm_studio``, the provider is rendered as
    ``openai`` in the generated code, and ``api_base`` defaults to ``http://127.0.0.1:1234/v1``.

    ``api_key_env``: if provided, the generated code includes ``api_key=os.getenv("<ENV>")``;
    otherwise ``api_key`` is omitted (LiteLLM reads the provider's env variables automatically).
    **The key is never hardcoded.**
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
    """Immutable spec of a SafetySetting (``category`` + ``threshold``).

    The values are **enum member names** (e.g. ``"HARM_CATEGORY_HARASSMENT"``,
    ``"BLOCK_MEDIUM_AND_ABOVE"``) — validated against :data:`HARM_CATEGORIES` /
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
    """Immutable spec of a ``types.GenerateContentConfig``.

    Only the non-None fields are rendered in the generated code.
    ``safety_settings`` is a tuple of :class:`SafetySettingSpec` (frozen for immutability).
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
class CallbackSpec:
    """Immutable spec of a guardrail (callback) attached to an ``LlmAgent`` agent (P4c).

    ``hook`` ∈ :data:`_CALLBACK_HOOKS` designates the real kwarg (``before_model`` ->
    ``before_model_callback=``, etc.). ``policy`` ∈ :data:`_POLICY_KINDS` designates the policy
    rendered into an importable Python function. ``params`` is a **frozen** list of
    ``(key, value)`` pairs (strings) carrying the policy configuration:

    - ``block_keywords``: ``keywords`` = ``,``-separated list; ``refusal`` (optional).
    - ``max_input_chars``: ``max_chars`` = integer (as a string); ``refusal`` (optional).
    - ``block_tool``: ``denylist`` = ``,``-separated list of tool names; ``message`` (opt).

    The rendering produces a **real functional function** (cf. ``_codegen._render_callback_def``),
    attached to the agent via the real kwarg. Returning non-``None`` short-circuits (LLM/tool).
    """

    hook: CallbackHook
    policy: PolicyKind
    params: tuple[tuple[str, str], ...] = ()

    def param(self, key: str, default: str = "") -> str:
        """Return the value of parameter ``key`` (or ``default`` if absent)."""
        for k, v in self.params:
            if k == key:
                return v
        return default

    def kwarg_name(self) -> str:
        """Name of the real kwarg on ``LlmAgent`` (e.g. ``before_model_callback``)."""
        return _CALLBACK_KWARG[self.hook]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "policy": {"kind": self.policy, **{k: v for k, v in self.params}},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallbackSpec:
        policy_raw = data.get("policy") or {}
        # Tolerant deserialization: hook/policy are annotated Literal for mypy, but the actual
        # validity is guaranteed upstream by ``validate_callback_spec`` (on the domain side).
        hook: CallbackHook = data.get("hook", "before_model")
        policy: PolicyKind = policy_raw.get("kind", "")
        params = tuple((str(k), str(v)) for k, v in policy_raw.items() if k != "kind")
        return cls(hook=hook, policy=policy, params=params)


@dataclass(frozen=True)
class AgentSpec:
    """Immutable spec of an agent in the project model.

    Fields not relevant for a given type keep their default value (e.g. ``model``/``instruction``
    ignored for a ``sequential`` agent).

    ``model``: Gemini string (backward compatible, e.g. ``"gemini-2.5-flash"``).
    ``model_spec``: if not None, a :class:`LiteLlmSpec`; takes priority over ``model`` for
    rendering ``LlmAgent``'s ``model=`` field.
    ``generate_content_config``: if not None, a :class:`GenerateContentConfigSpec`; rendered as
    ``generate_content_config=types.GenerateContentConfig(...)`` on ``LlmAgent``.
    """

    name: str
    type: AgentType
    model: str = "gemini-2.5-flash"
    instruction: str = ""
    description: str = ""
    output_key: str | None = None
    #: URL (or local JSON path) of the remote agent-card, for the ``remote_a2a`` type only.
    #: Rendered as ``RemoteA2aAgent(name=..., agent_card="<url>")``. Ignored for other types.
    agent_card: str = ""
    #: Attached tools. ``ToolSpec`` (rich codegen); the legacy ``str`` form (P1) is still
    #: tolerated and rendered as a bare reference (name already imported). See ``render_tool_ref``.
    tools: tuple[ToolSpec | str, ...] = ()
    sub_agents: tuple[str, ...] = ()
    max_iterations: int = 3
    #: LiteLLM spec (P4). If not None, takes priority over ``model`` for rendering.
    model_spec: LiteLlmSpec | None = None
    #: generate_content config (P4). Rendered as ``generate_content_config=...``.
    generate_content_config: GenerateContentConfigSpec | None = None
    #: Guardrails (P4c): one :class:`CallbackSpec` per hook. Rendered as a generated function
    #: attached via the real kwarg (``before_model_callback=...``). LlmAgent only.
    callbacks: tuple[CallbackSpec, ...] = ()
    #: Default LLM call cap (P4c). Persisted in the sidecar but **NOT rendered** in ``agent.py``
    #: (it is not an ``LlmAgent`` kwarg but a ``RunConfig`` setting exposed by the ``run``
    #: domain). ``None`` = ADK default (500).
    max_llm_calls: int | None = None

    def tool_specs(self) -> tuple[ToolSpec, ...]:
        """Normalize ``tools`` to ``ToolSpec`` (legacy strings -> ``builtin``)."""
        return tuple(t if isinstance(t, ToolSpec) else ToolSpec.from_dict(t) for t in self.tools)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the sidecar's JSON form (relevant fields per type)."""
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
            if self.callbacks:
                base["callbacks"] = [c.to_dict() for c in self.callbacks]
            if self.max_llm_calls is not None:
                base["max_llm_calls"] = self.max_llm_calls
        elif self.type in ("sequential", "parallel"):
            base["sub_agents"] = list(self.sub_agents)
        elif self.type == "loop":
            base["sub_agents"] = list(self.sub_agents)
            base["max_iterations"] = self.max_iterations
        elif self.type == "remote_a2a":
            base["agent_card"] = self.agent_card
        # `custom`: only name/type/description.
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        """Deserialize a sidecar entry (tolerant of absent fields)."""
        atype = data.get("type", "llm")
        raw_tools = data.get("tools", []) or []
        # Legacy form (P1): a string entry stays a string (passthrough, rendered bare).
        # Rich form (3a): a dict is deserialized into a ``ToolSpec``.
        tools: tuple[ToolSpec | str, ...] = tuple(
            t if isinstance(t, str) else ToolSpec.from_dict(t) for t in raw_tools
        )
        raw_ms = data.get("model_spec")
        model_spec = LiteLlmSpec.from_dict(raw_ms) if isinstance(raw_ms, dict) else None
        raw_gcc = data.get("generate_content_config")
        generate_content_config = (
            GenerateContentConfigSpec.from_dict(raw_gcc) if isinstance(raw_gcc, dict) else None
        )
        raw_cbs = data.get("callbacks") or []
        callbacks = tuple(CallbackSpec.from_dict(c) for c in raw_cbs if isinstance(c, dict))
        raw_max = data.get("max_llm_calls")
        max_llm_calls = int(raw_max) if isinstance(raw_max, int) else None
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
            callbacks=callbacks,
            max_llm_calls=max_llm_calls,
        )


@dataclass(frozen=True)
class ProjectModel:
    """Full model of an ADK app: list of agents + designated root."""

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
# Validation — Python identifier
# --------------------------------------------------------------------------- #
def is_identifier(name: str) -> bool:
    """True if ``name`` is a valid Python identifier (module variable name)."""
    return bool(_IDENT_RE.match(name))
