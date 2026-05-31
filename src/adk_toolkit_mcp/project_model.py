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
    tools: tuple[str, ...] = ()
    sub_agents: tuple[str, ...] = ()
    max_iterations: int = 3

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
                    "tools": list(self.tools),
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
        return cls(
            name=str(data["name"]),
            type=atype,
            model=str(data.get("model", "gemini-2.5-flash")),
            instruction=str(data.get("instruction", "")),
            description=str(data.get("description", "")),
            output_key=data.get("output_key"),
            tools=tuple(data.get("tools", []) or []),
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
def topological_order(model: ProjectModel) -> list[AgentSpec]:
    """Trie les agents pour qu'un enfant soit défini avant son parent.

    Lève ``ValueError`` si un cycle est détecté (les outils convertissent en ``err``).
    Les ``sub_agents`` référençant un nom absent sont ignorés pour l'ordonnancement
    (la validation d'existence est faite en amont par les outils du domaine).
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
            raise ValueError(f"Cycle détecté dans sub_agents : {cycle}")
        state[name] = 1
        spec = by_name[name]
        for sub in spec.sub_agents:
            if sub in by_name:  # n'ordonne que les références internes connues
                visit(sub, (*path, name))
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
    """Littéral chaîne Python sûr (gère quotes/échappements via repr)."""
    return repr(value)


def render_tool_ref(tool: str) -> str:
    """Rendu d'une entrée ``tools`` (POINT D'EXTENSION pour le domaine `tools` en P3).

    En P1, on rend la référence **telle quelle** : un nom de variable d'outil ou un
    builtin déjà importé dans le module (ex. ``google_search``). Le domaine `tools`
    étendra ce helper (codegen de function-tools, imports, etc.). On ne fabrique
    volontairement aucun code de function-tool ici.
    """
    return tool


def _render_kwargs(pairs: list[tuple[str, str]]) -> str:
    """Assemble des ``k=v`` déjà rendus en une liste d'arguments multi-lignes."""
    return "".join(f"    {key}={value},\n" for key, value in pairs)


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
        rendered = ", ".join(render_tool_ref(t) for t in spec.tools)
        pairs.append(("tools", f"[{rendered}]"))
    if spec.sub_agents:
        pairs.append(("sub_agents", "[" + ", ".join(spec.sub_agents) + "]"))
    return f"{spec.name} = LlmAgent(\n{_render_kwargs(pairs)})\n"


def _render_workflow(spec: AgentSpec, class_name: str) -> str:
    """Rend un ``SequentialAgent``/``ParallelAgent`` (name + sub_agents + description?)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("sub_agents", "[" + ", ".join(spec.sub_agents) + "]"))
    return f"{spec.name} = {class_name}(\n{_render_kwargs(pairs)})\n"


def _render_loop(spec: AgentSpec) -> str:
    """Rend un ``LoopAgent`` (name + sub_agents + max_iterations + description?)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("sub_agents", "[" + ", ".join(spec.sub_agents) + "]"))
    pairs.append(("max_iterations", str(spec.max_iterations)))
    return f"{spec.name} = LoopAgent(\n{_render_kwargs(pairs)})\n"


def _custom_class_name(name: str) -> str:
    """Nom de classe PascalCase pour un agent custom (``my_agent`` -> ``MyAgentAgent``)."""
    pascal = "".join(part.capitalize() for part in name.split("_") if part)
    if not pascal:
        pascal = "Custom"
    return f"{pascal}Agent"


def _render_custom(spec: AgentSpec) -> str:
    """Rend une sous-classe ``BaseAgent`` (stub) + une instance module-level.

    Le ``_run_async_impl`` est un **async generator** no-op (``return`` puis ``yield``
    inatteignable) — c'est la forme valide attendue par ADK (cf. agents.md).
    """
    class_name = _custom_class_name(spec.name)
    desc = _py_str(spec.description) if spec.description else _py_str("")
    return (
        f"class {class_name}(BaseAgent):\n"
        f'    """Agent custom généré (stub). Complétez `_run_async_impl`."""\n'
        "\n"
        "    async def _run_async_impl(self, ctx):\n"
        "        # TODO: implémenter la logique de l'agent.\n"
        "        return\n"
        "        yield  # rend cette méthode un async generator (inatteignable)\n"
        "\n"
        f"{spec.name} = {class_name}(name={_py_str(spec.name)}, description={desc})\n"
    )


def _render_agent(spec: AgentSpec) -> str:
    """Aiguille vers le renderer du bon type."""
    if spec.type == "llm":
        return _render_llm(spec)
    if spec.type in ("sequential", "parallel"):
        return _render_workflow(spec, _CLASS_FOR_TYPE[spec.type])
    if spec.type == "loop":
        return _render_loop(spec)
    if spec.type == "custom":
        return _render_custom(spec)
    raise ValueError(f"Type d'agent non rendu : {spec.type!r}")  # pragma: no cover


def _needed_imports(model: ProjectModel) -> list[str]:
    """Classes ADK réellement utilisées, dans l'ordre canonique."""
    used: set[str] = set()
    for a in model.agents:
        if a.type == "custom":
            used.add("BaseAgent")
        else:
            used.add(_CLASS_FOR_TYPE[a.type])
    return [name for name in _IMPORT_ORDER if name in used]


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

    imports = _needed_imports(model)
    import_line = f"from google.adk.agents import {', '.join(imports)}\n\n"

    ordered = topological_order(model)  # peut lever ValueError (cycle)
    blocks = "\n".join(_render_agent(spec) for spec in ordered)

    if model.root is not None and model.get(model.root) is not None:
        root_line = f"\nroot_agent = {model.root}\n"
    elif model.root is not None:
        root_line = (
            f"\n# root '{model.root}' introuvable parmi les agents ; root_agent non défini.\n"
        )
    else:
        root_line = "\n# root_agent non défini : appelez set_root pour désigner la racine.\n"

    return header + import_line + blocks + root_line


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
