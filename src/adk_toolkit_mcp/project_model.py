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
# Outils (domaine `tools`, passe 3a — outils sans dépendance)
# --------------------------------------------------------------------------- #
#: Genres d'outils supportés en 3a.
ToolKind = Literal["function", "long_running", "builtin", "agent_tool", "openapi"]

_TOOL_KINDS: frozenset[str] = frozenset(
    {"function", "long_running", "builtin", "agent_tool", "openapi"}
)

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
class ToolSpec:
    """Spécification immuable d'un outil attaché à un agent (domaine `tools`, 3a).

    Le champ ``kind`` discrimine ; seuls les champs pertinents sont renseignés/sérialisés :

    - ``function`` / ``long_running`` : ``name`` (identifiant), ``params`` (tuple de
      ``(name, type, default|None)``), ``docstring``, ``returns``, ``body``.
    - ``builtin`` : ``builtin_kind`` (membre de :data:`BUILTIN_TOOLS`), ``args`` (pour
      ``vertex_ai_search`` : ``{"data_store_id": ...}`` ou ``{"search_engine_id": ...}``).
    - ``agent_tool`` : ``target_agent`` (nom d'un agent **existant** du modèle).
    - ``openapi`` : ``name`` (identifiant de la variable toolset), ``spec`` (chaîne OpenAPI).

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

    def ref_key(self) -> str:
        """Clé d'unicité : ``function``/``long_running``/``openapi`` -> nom ; ``builtin`` ->
        ``builtin:<kind>`` ; ``agent_tool`` -> ``agent_tool:<target>``."""
        if self.kind in ("function", "long_running", "openapi"):
            return f"{self.kind}:{self.name}"
        if self.kind == "builtin":
            return f"builtin:{self.builtin_kind}"
        if self.kind == "agent_tool":
            return f"agent_tool:{self.target_agent}"
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

    POINT D'EXTENSION implémenté en passe 3a (outils sans dépendance). Genres gérés :

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
        helper = _render_openapi_helper(tool)
        return ToolRender(imports=(_OPENAPI_IMPORT,), helpers=(helper,), ref=tool.name)

    raise ValueError(f"Genre d'outil non rendu : {tool.kind!r}")  # pragma: no cover


def _render_openapi_helper(tool: ToolSpec) -> str:
    """Rend ``<id> = OpenAPIToolset(spec_str=..., spec_str_type="json")`` (stable ruff).

    Inline si la ligne tient dans :data:`LINE_LENGTH` ; sinon, ruff replie l'appel avec les
    deux arguments sur **une** ligne indentée (4 espaces) tant qu'ils y tiennent — on reproduit
    exactement cette forme (l'éclatement un-arg-par-ligne n'est pas nécessaire ici).
    """
    spec_lit = _py_str(tool.spec)
    args = f'spec_str={spec_lit}, spec_str_type="json"'
    inline = f"{tool.name} = OpenAPIToolset({args})"
    if len(inline) <= LINE_LENGTH:
        return inline + "\n"
    # Repli niveau 1 : les deux args sur une ligne indentée (4 espaces), si elle tient.
    if 4 + len(args) <= LINE_LENGTH:
        return f"{tool.name} = OpenAPIToolset(\n    {args}\n)\n"
    # Repli niveau 2 : un argument par ligne (indent 4, virgule finale) — forme ruff au-delà.
    return (
        f'{tool.name} = OpenAPIToolset(\n    spec_str={spec_lit},\n    spec_str_type="json",\n)\n'
    )


# --------------------------------------------------------------------------- #
# Validation d'outils
# --------------------------------------------------------------------------- #
def validate_tool_spec(tool: ToolSpec, model: ProjectModel, owner: str) -> str | None:
    """Renvoie un message d'erreur si ``tool`` est invalide, sinon None.

    ``model``/``owner`` servent à valider ``agent_tool`` (cible existante et != propriétaire).
    """
    if tool.kind not in _TOOL_KINDS:
        return f"Genre d'outil inconnu : {tool.kind!r}. Connus : {', '.join(sorted(_TOOL_KINDS))}."

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

    return None  # pragma: no cover


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
        f"from {module} import {', '.join(sorted(by_module[module]))}"
        for module in sorted(by_module)
    ]
    return _dedup_preserve(passthrough) + merged


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
