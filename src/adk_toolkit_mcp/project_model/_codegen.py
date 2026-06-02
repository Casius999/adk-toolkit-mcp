"""Ruff-stable code generation primitives + tool rendering (internal).

A **private** module (``_`` prefix): it exposes no stable public API except
:func:`render_tool_ref`, itself re-exported via :mod:`adk_toolkit_mcp.project_model`. It groups:

- the low-level primitives **stable for ``ruff format``**: :class:`_Call` + ``_render_call`` /
  ``_call_inline`` / ``_kwarg_call`` (the "one argument per line" splitting that exactly
  reproduces ``ruff format``'s output), ``_py_str`` / ``_py_bool`` (literals), and the rendering
  of function-tool ``def`` blocks;
- the rendering of each tool kind (:func:`render_tool_ref`) and of the associated auth
  (``AuthCredential(...)``), including the 3b toolsets (openapi/bigquery/spanner/mcp/apihub/
  langchain/crewai).

Consumed by :mod:`adk_toolkit_mcp.project_model.render`, which assembles the complete
``agent.py`` module (agents, import order, PEP 8 spacing).
"""

from __future__ import annotations

from dataclasses import dataclass

from .specs import (
    _APIHUB_IMPORT,
    _AUTH_CRED_IMPORT_MODULE,
    _AUTH_IMPORT_MODULE,
    _AUTH_TYPE_FOR_SCHEME,
    _BIGQUERY_IMPORT,
    _BUILTIN_CLASS,
    _CREWAI_IMPORT,
    _DEFAULT_REFUSAL,
    _GUARD_FN_PREFIX,
    _LANGCHAIN_IMPORT,
    _MCP_STDIO_PARAMS_IMPORT,
    _MCP_TOOLSET_IMPORT_MODULE,
    _MCP_TRANSPORTS,
    _OPENAPI_IMPORT,
    _SPANNER_IMPORT,
    _TOOLS_IMPORT_MODULE,
    ARG_BUILTINS,
    CORE_BUILTINS,
    LINE_LENGTH,
    AuthSpec,
    CallbackSpec,
    ToolRender,
    ToolSpec,
)

#: Imports required by ``before_model`` guardrail bodies (refusal = ``LlmResponse``/``Content``).
_LLM_RESPONSE_IMPORT = "from google.adk.models import LlmResponse"
_GENAI_TYPES_IMPORT = "from google.genai import types"


# --------------------------------------------------------------------------- #
# Source rendering — low-level helpers
# --------------------------------------------------------------------------- #
def _py_str(value: str) -> str:
    """Python string literal **stable for ``ruff format``**.

    ``ruff format`` (like Black) prefers double quotes, **except** if the value contains a ``"``
    but no ``'`` — in which case it switches to single quotes to avoid escaping. We reproduce that
    choice exactly so the generated output is already in the form ruff would write (idempotence of
    ``format --check``).
    """
    has_double = '"' in value
    has_single = "'" in value
    if has_double and not has_single:
        # Single quotes: only the backslash needs escaping.
        escaped = value.replace("\\", "\\\\")
        return f"'{escaped}'"
    # Double quotes by default: escape backslash then double quote.
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render_param(name: str, ptype: str, default: str | None) -> str:
    """Render a signature parameter: ``name: type`` or ``name: type = default``.

    ``default`` is an **already-rendered source literal** (e.g. ``"x"``, ``0``, ``None``). When
    ``default`` is ``None`` (in the Python sense), the parameter has no default.
    """
    base = f"{name}: {ptype}"
    return base if default is None else f"{base} = {default}"


def _render_function_def(spec: ToolSpec) -> str:
    """Render a top-level ``def`` block: typed signature, 1-line docstring, then the body.

    The body and the docstring are indented by 4 spaces; the block ends with a single ``\\n`` (the
    module renderer handles the inter-block spacing ruff-style).
    """
    params = ", ".join(_render_param(n, t, d) for (n, t, d) in spec.params)
    doc = (spec.docstring or spec.name).replace("\\", "\\\\").replace('"', '\\"')
    # single-line docstring, escaped (triple quotes).
    doc_line = f'    """{doc}"""\n'
    body_lines = spec.body.splitlines() or ["return {}"]
    body = "".join(f"    {line}\n" for line in body_lines)
    return f"def {spec.name}({params}) -> {spec.returns}:\n{doc_line}{body}"


def _render_builtin_ref(spec: ToolSpec) -> ToolRender:
    """Render a builtin's reference (core -> bare name; ``vertex_ai_search`` -> call)."""
    if spec.builtin_kind in CORE_BUILTINS:
        imp = f"from {_TOOLS_IMPORT_MODULE} import {spec.builtin_kind}"
        return ToolRender(imports=(imp,), helpers=(), ref=spec.builtin_kind)
    if spec.builtin_kind in ARG_BUILTINS:
        class_name = _BUILTIN_CLASS[spec.builtin_kind]
        imp = f"from {_TOOLS_IMPORT_MODULE} import {class_name}"
        kwargs = ", ".join(f"{k}={_py_str(v)}" for k, v in spec.args)
        return ToolRender(imports=(imp,), helpers=(), ref=f"{class_name}({kwargs})")
    # unknown builtin_kind: render as-is (upstream validation will have rejected it).
    return ToolRender(imports=(), helpers=(), ref=spec.builtin_kind)  # pragma: no cover


def render_tool_ref(tool: ToolSpec | str) -> ToolRender:
    """Render a ``tools`` entry -> :class:`ToolRender` (imports, helpers, ref).

    EXTENSION POINT implemented in passes 3a + 3b. Handled kinds:

    Pass 3a (no dependency):

    - ``function``: helper = a rendered ``def``; ``ref`` = ``<name>`` (ADK auto-wraps the function
      in a ``FunctionTool`` via ``canonical_tools`` — cf. ``docs/adk-api-notes/tools.md``).
    - ``long_running``: same helper; import ``LongRunningFunctionTool``;
      ``ref`` = ``LongRunningFunctionTool(func=<name>)``.
    - ``builtin``: ``ref`` = the builtin's name (e.g. ``google_search``) imported;
      ``vertex_ai_search`` -> ``VertexAiSearchTool(data_store_id="...")``.
    - ``agent_tool``: import ``AgentTool``; ``ref`` = ``AgentTool(agent=<target>)``.
    - ``openapi``: import ``OpenAPIToolset``; helper = ``<id> = OpenAPIToolset(spec_str=..., \
      spec_str_type="json")``; ``ref`` = ``<id>`` (the toolset goes **directly** into
      ``tools=[...]`` — confirmed by introspection, no ``.get_tools()``).

    Pass 3b (optional dependency; **codegen-only** — the toolkit never imports these extras):

    - ``bigquery`` / ``spanner``: import the toolset; helper ``<id> = BigQueryToolset(<args>)`` /
      ``SpannerToolset(<args>)``; ``ref`` = ``<id>``.
    - ``mcp_toolset``: import ``McpToolset`` + the transport's connection-params class
      (+ ``StdioServerParameters`` for stdio); helper
      ``<id> = McpToolset(connection_params=..., tool_filter=[...])``; ``ref`` = ``<id>``.
    - ``apihub``: import ``APIHubToolset``; helper
      ``<id> = APIHubToolset(apihub_resource_name="...")``; ``ref`` = ``<id>``.
    - ``langchain``: import ``LangchainTool`` + the user's import line (verbatim);
      ``ref`` = ``LangchainTool(tool=<tool_expr>)`` (no helper).
    - ``crewai``: import ``CrewaiTool`` + the user's import line;
      ``ref`` = ``CrewaiTool(tool=<tool_expr>, name=..., description=...)``.

    Auth (openapi/apihub/mcp_toolset): if ``tool.auth`` is set, ``auth_credential=\
    AuthCredential(...)`` is added to the helper's kwargs + the ``google.adk.auth`` imports.

    Legacy form (``str``): rendered **as-is** (a bare reference already imported), with no import
    or helper, for backward compatibility with the P1 model.
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

    raise ValueError(f"Unrendered tool kind: {tool.kind!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Auth rendering (set_auth) — ``auth_credential=AuthCredential(...)`` + imports
# --------------------------------------------------------------------------- #
def _auth_credential_call(auth: AuthSpec) -> tuple[_Call, tuple[str, ...]]:
    """Build the ``AuthCredential(...)`` :class:`_Call` + the required ``google.adk.auth`` imports.

    The scheme dictates ``auth_type`` and the carried sub-object:

    - ``apikey`` -> ``api_key="..."``;
    - ``bearer`` -> ``http=HttpAuth(scheme="bearer", credentials=HttpCredentials(token="..."))``;
    - ``oauth2`` -> ``oauth2=OAuth2Auth(client_id=..., client_secret=..., [access_token=...])``;
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
    """List of a ``ServiceAccount``'s kwargs from the credential dict (booleans/lists handled).

    ``use_default_credential``: value ``"true"``/``"false"`` -> Python boolean literal.
    ``scopes``: comma-separated value -> list of strings.
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
    """``"true"``/``"1"``/``"yes"`` -> ``True`` (otherwise ``False``) — Python source literal."""
    return "True" if value.strip().lower() in ("true", "1", "yes") else "False"


@dataclass(frozen=True)
class _Call:
    """Structured representation of a ``Callee(arg1, arg2, ...)`` call for ruff-stable rendering.

    Each argument is either an already-rendered **atomic string** (``"key=value"``, a literal, an
    inline list/dict) or a nested :class:`_Call` (rendered recursively). We never fold the inside
    of an atomic literal — only ``_Call`` objects are split recursively, which is enough to
    reproduce the ``ruff format`` output of our constructs.
    """

    callee: str
    args: tuple[str | _Call, ...]


def _render_call(call: _Call, col: int, base_indent: int) -> str:
    """Render a :class:`_Call` **stable for ``ruff format``**.

    ``col`` = the column where this rendering starts (inline width budget); ``base_indent`` = the
    indentation of the owning **logical line** (the split body is indented by ``base_indent + 4``,
    like ``ruff format``). Reproduced algorithm:

    - inline form if it fits in :data:`LINE_LENGTH` starting from ``col``;
    - otherwise, split **one argument per line** (indent ``base_indent+4``). The trailing comma
      ("magic trailing comma") is added **only** if the call has **≥ 2 arguments**: a
      single-argument call that must be folded puts that argument alone on its line **without** a
      trailing comma (exact ``ruff format`` behavior — verified by introspection).

    Does **not** end with ``\\n`` (the caller handles line breaks / the ``= var`` suffix).
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
    """Full inline form of a :class:`_Call` (recursive, no line breaks)."""
    parts = [a if isinstance(a, str) else _call_inline(a) for a in call.args]
    return f"{call.callee}({', '.join(parts)})"


def _kwarg_call(key: str, call: _Call) -> _Call:
    """Combine ``key=`` + a :class:`_Call` into a foldable :class:`_Call` (``callee=key=Callee``).

    :func:`_render_call` then chooses inline (``key=Callee(...)``) or split
    (``key=Callee(\\n ... \\n)``) depending on the width — exactly the ``ruff format`` form.
    """
    return _Call(callee=f"{key}={call.callee}", args=call.args)


def _render_toolset_helper(var: str, call: _Call) -> str:
    """Render ``<var> = <Call>`` (recursively folded) ending with a single ``\\n``.

    The call starts at column ``len(var) + 3`` (``"<var> = "``); the split body is indented from
    ``base_indent=0`` (top-level statement) -> +4, matching ``ruff format``.
    """
    return f"{var} = {_render_call(call, col=len(var) + 3, base_indent=0)}\n"


def _maybe_auth_arg(tool: ToolSpec) -> tuple[list[str | _Call], tuple[str, ...]]:
    """Return ``([auth_credential=...] | [], imports)`` for an auth-capable toolset.

    If ``tool.auth`` is set, render ``auth_credential=AuthCredential(...)`` (foldable) + the
    required auth imports; otherwise, empty lists. (Validation guarantees that only auth-capable
    kinds carry an ``auth``.)
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

    The ``args`` are **source expressions** (not string literals): a user provides e.g.
    ``{"bigquery_tool_config": "my_cfg"}`` to reference a variable/object built elsewhere. No auth
    here (these toolsets use ``credentials_config``).
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
    """Build the ``connection_params=...`` :class:`_Call` per transport + required imports.

    - ``stdio`` -> ``StdioConnectionParams(server_params=StdioServerParameters(command=...,
      args=[...]))`` (also imports ``StdioServerParameters`` from ``mcp``);
    - ``sse`` -> ``SseConnectionParams(url="..."[, headers={...}])``;
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
    # sse / http: url + optional headers.
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
# Callback rendering (guardrails) — `safety` domain, P4c
# --------------------------------------------------------------------------- #
def _guard_fn_name(agent_name: str, hook: str) -> str:
    """Stable name of the generated guardrail function (unique per agent + hook).

    E.g. ``_guard_before_model_my_agent``. Stable (deterministic) so that regeneration is
    idempotent and the agent's kwarg references exactly this function.
    """
    return f"{_GUARD_FN_PREFIX}_{hook}_{agent_name}"


def _py_str_list(values: list[str]) -> str:
    """Render ``[ "a", "b" ]`` (string list literal) inline and ruff-stable."""
    return "[" + ", ".join(_py_str(v) for v in values) + "]"


def _split_csv(raw: str) -> list[str]:
    """Split a value ``"a, b ,c"`` into ``["a", "b", "c"]`` (empties ignored)."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _refusal_response_lines(refusal: str, indent: str) -> list[str]:
    """Line ``return _refuse("<refusal>")`` (before_model) — via the shared helper ``_refuse``.

    We delegate the construction of the ``LlmResponse`` to the top-level helper
    :data:`_REFUSE_HELPER` (emitted once). The guardrail body therefore only carries a
    single-argument call (a string): stable for ``ruff format`` whatever the message length (a
    long literal stays alone on its line, without a trailing comma — exact ruff behavior).
    """
    call = _Call("_refuse", (_py_str(refusal),))
    rendered = _render_call(call, col=len(indent) + len("return "), base_indent=len(indent))
    return [f"{indent}return {rendered}"]


def _block_keywords_body(spec: CallbackSpec) -> tuple[list[str], tuple[str, ...]]:
    """Body of ``before_model``: refuses if a blocked term appears in the user text.

    Reads ``llm_request.contents`` (list[Content]), concatenates the text of the LAST user turn's
    parts, compares it in lowercase against the list of blocked words; on a match -> returns a
    refusal ``LlmResponse`` (short-circuits the LLM). Otherwise ``return None`` (normal flow).
    """
    keywords = _split_csv(spec.param("keywords"))
    refusal = spec.param("refusal") or _DEFAULT_REFUSAL
    lines = [
        f"    blocked = {_py_str_list(keywords)}",
        "    text = _user_text(llm_request).lower()",
        "    if any(term.lower() in text for term in blocked):",
        *_refusal_response_lines(refusal, indent="        "),
        "    return None",
    ]
    # Imports (LlmResponse/types) carried by the shared helper ``_refuse`` -> no import here.
    return lines, ()


def _max_input_chars_body(spec: CallbackSpec) -> tuple[list[str], tuple[str, ...]]:
    """Body of ``before_model``: refuses if the user text exceeds ``max_chars``."""
    try:
        max_chars = int(spec.param("max_chars", "0"))
    except ValueError:  # pragma: no cover - validated upstream by the safety domain
        max_chars = 0
    refusal = spec.param("refusal") or _DEFAULT_REFUSAL
    lines = [
        f"    max_chars = {max_chars}",
        "    if len(_user_text(llm_request)) > max_chars:",
        *_refusal_response_lines(refusal, indent="        "),
        "    return None",
    ]
    # Imports (LlmResponse/types) carried by the shared helper ``_refuse`` -> no import here.
    return lines, ()


def _block_tool_body(spec: CallbackSpec) -> tuple[list[str], tuple[str, ...]]:
    """Body of ``before_tool``: short-circuits the tool if its name is in the denylist.

    Returns a ``dict`` (used as the tool's result), which prevents its execution.
    """
    denylist = _split_csv(spec.param("denylist"))
    message = spec.param("message") or "Tool call blocked by safety policy."
    lines = [
        f"    denylist = {_py_str_list(denylist)}",
        "    if tool.name in denylist:",
        f"        return {{{_py_str('error')}: {_py_str(message)}}}",
        "    return None",
    ]
    return lines, ()


#: Body of each policy -> (body lines, required imports).
_POLICY_BODY = {
    "block_keywords": _block_keywords_body,
    "max_input_chars": _max_input_chars_body,
    "block_tool": _block_tool_body,
}

#: Signature (positional parameters) of each hook (cf. 2.1.0 introspection).
_HOOK_SIGNATURE: dict[str, str] = {
    "before_model": "callback_context, llm_request",
    "before_tool": "tool, args, tool_context",
}

#: Shared top-level helper: extracts the text of the last user turn from an ``LlmRequest``.
#: Emitted ONCE if at least one ``before_model`` guardrail (keywords / max_chars) uses it.
_USER_TEXT_HELPER = (
    "def _user_text(llm_request) -> str:\n"
    '    """Concatenate the text of an LlmRequest\'s last user turn parts."""\n'
    "    for content in reversed(llm_request.contents or []):\n"
    '        if getattr(content, "role", None) == "user":\n'
    '            return "".join(p.text for p in (content.parts or []) if p.text)\n'
    '    return ""\n'
)

#: Shared top-level helper: builds a refusal ``LlmResponse`` from a text message.
#: Emitted ONCE if at least one ``before_model`` guardrail short-circuits the LLM. Centralizes the
#: construction (a single-argument call on the guardrail side -> ruff-stable rendering, even for a
#: long message) and the ``LlmResponse``/``types`` imports.
_REFUSE_HELPER = (
    "def _refuse(message: str) -> LlmResponse:\n"
    '    """Build a refusal response (short-circuits the LLM) carrying ``message``."""\n'
    "    return LlmResponse(\n"
    '        content=types.Content(role="model", parts=[types.Part.from_text(text=message)])\n'
    "    )\n"
)

#: Policies that require the ``_user_text`` helper (``before_model`` guardrails).
_NEEDS_USER_TEXT: frozenset[str] = frozenset({"block_keywords", "max_input_chars"})

#: Policies that short-circuit the LLM via ``_refuse`` (``before_model`` guardrails).
_NEEDS_REFUSE: frozenset[str] = frozenset({"block_keywords", "max_input_chars"})


def render_callback(spec: CallbackSpec, agent_name: str) -> ToolRender:
    """Render a guardrail -> :class:`ToolRender` (imports, helper def, ``ref`` = function name).

    The ``ref`` is the name of the generated function (to place as the value of the agent's real
    kwarg, e.g. ``before_model_callback=_guard_before_model_<agent>``). The helper is a
    **functional** top-level ``def`` (real body per policy). The required imports (``LlmResponse``
    / ``types``) are lifted to the module's import section by the renderer.
    """
    fn_name = _guard_fn_name(agent_name, spec.hook)
    params = _HOOK_SIGNATURE[spec.hook]
    body_lines, imports = _POLICY_BODY[spec.policy](spec)
    doc = f'    """{spec.policy} guardrail ({spec.hook}) generated by adk-toolkit-mcp."""'
    body = "\n".join(body_lines)
    helper = f"def {fn_name}({params}):\n{doc}\n{body}\n"
    return ToolRender(imports=imports, helpers=(helper,), ref=fn_name)


def callback_needs_user_text(spec: CallbackSpec) -> bool:
    """True if the callback's policy requires the top-level helper ``_user_text``."""
    return spec.policy in _NEEDS_USER_TEXT


def callback_needs_refuse(spec: CallbackSpec) -> bool:
    """True if the callback's policy short-circuits the LLM via the ``_refuse`` helper."""
    return spec.policy in _NEEDS_REFUSE


def refuse_helper_render() -> ToolRender:
    """``ToolRender`` of the shared ``_refuse`` helper (def + ``LlmResponse``/``types`` imports)."""
    return ToolRender(
        imports=(_LLM_RESPONSE_IMPORT, _GENAI_TYPES_IMPORT),
        helpers=(_REFUSE_HELPER,),
        ref="_refuse",
    )


def user_text_helper_render() -> ToolRender:
    """``ToolRender`` of the shared ``_user_text`` helper (def, no import)."""
    return ToolRender(imports=(), helpers=(_USER_TEXT_HELPER,), ref="_user_text")
