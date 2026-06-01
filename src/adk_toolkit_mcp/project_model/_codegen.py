"""Primitives de génération de code ruff-stable + rendu des outils (interne).

Module **privé** (préfixe ``_``) : il n'expose aucune API publique stable hormis
:func:`render_tool_ref`, lui-même ré-exporté via :mod:`adk_toolkit_mcp.project_model`. Il
regroupe :

- les primitives bas-niveau **stables pour ``ruff format``** : :class:`_Call` + ``_render_call``
  / ``_call_inline`` / ``_kwarg_call`` (éclatement « un argument par ligne » reproduisant
  exactement la sortie de ``ruff format``), ``_py_str`` / ``_py_bool`` (littéraux), et le rendu
  des ``def`` de function-tools ;
- le rendu de chaque genre d'outil (:func:`render_tool_ref`) et de l'auth associée
  (``AuthCredential(...)``), y compris les toolsets 3b (openapi/bigquery/spanner/mcp/apihub/
  langchain/crewai).

Consommé par :mod:`adk_toolkit_mcp.project_model.render`, qui assemble le module ``agent.py``
complet (agents, ordre d'import, espacement PEP 8).
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
    ToolRender,
    ToolSpec,
)


# --------------------------------------------------------------------------- #
# Rendu de source — helpers bas-niveau
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
