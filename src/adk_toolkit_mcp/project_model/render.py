"""Génération complète de ``agent.py`` à partir du :class:`ProjectModel`.

Cœur du renderer : tri topologique des agents (détection de cycle), rendu de chaque type
d'agent (``LlmAgent`` / workflow / loop / ``custom``), du ``model=`` LiteLLM et du
``generate_content_config=``, puis assemblage du module complet — section d'imports ordonnée
façon isort (stdlib, ``google.adk.agents``, autres third-party ; noms triés, dédupliqués) et
espacement PEP 8 (E302/E303/E305) stable pour ``ruff format``.

S'appuie sur :mod:`adk_toolkit_mcp.project_model._codegen` pour les primitives ruff-stables et
le rendu des outils, et sur :mod:`adk_toolkit_mcp.project_model.specs` pour les dataclasses et
constantes. :func:`render_tool_ref` est ré-importé ici (utilisé par le rendu de ``LlmAgent``)
et reste exposé via ``adk_toolkit_mcp.project_model``.
"""

from __future__ import annotations

import re
from typing import Any

from ..workspace import Workspace
from ._codegen import _Call, _py_str, _render_call, render_tool_ref
from .specs import (
    _CLASS_FOR_TYPE,
    _IMPORT_ORDER,
    LINE_LENGTH,
    AgentSpec,
    GenerateContentConfigSpec,
    LiteLlmSpec,
    ProjectModel,
    SafetySettingSpec,
    ToolRender,
)


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
# Rendu de source — agents, modèle, imports
# --------------------------------------------------------------------------- #
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


def _render_litellm_model(spec: LiteLlmSpec) -> tuple[str, tuple[str, ...]]:
    """Rend ``LiteLlm(model="<provider>/<model>"[, api_base=...][, api_key=...])`` + imports.

    - Pour ``lm_studio``, le provider est rendu comme ``openai`` et ``api_base`` vaut
      ``http://127.0.0.1:1234/v1`` si non fourni.
    - ``api_key`` est rendu comme ``os.getenv("<ENV>")`` (+ ``import os``) uniquement si
      ``api_key_env`` est défini. **La clé n'est jamais écrite en dur.**
    """
    provider = spec.provider
    api_base = spec.api_base

    # lm_studio : provider rendu comme openai, api_base par défaut.
    if provider == "lm_studio":
        provider = "openai"
        if not api_base:
            api_base = "http://127.0.0.1:1234/v1"

    model_str = f"{provider}/{spec.model}"
    args: list[str | _Call] = [f"model={_py_str(model_str)}"]
    if api_base:
        args.append(f"api_base={_py_str(api_base)}")

    imports: list[str] = ["from google.adk.models.lite_llm import LiteLlm"]
    if spec.api_key_env:
        args.append(f"api_key=os.getenv({_py_str(spec.api_key_env)})")
        imports.append("import os")

    call = _Call("LiteLlm", tuple(args))
    rendered = _render_call(call, col=len("    model="), base_indent=4)
    return rendered, tuple(imports)


def _render_safety_settings_arg(safety_settings: tuple[SafetySettingSpec, ...]) -> str:
    """Rend l'argument ``safety_settings=[...]`` pour ``GenerateContentConfig``.

    Les ``SafetySetting`` items sont dans la liste à ``base_indent=8`` (dans le corps
    de ``GenerateContentConfig`` qui est lui-même à base_indent=4 dans ``LlmAgent``).
    Chaque ``SafetySetting(...)`` est rendu avec ``_render_call(col=12, base_indent=12)``
    pour que le repli soit stable pour ``ruff format``.

    Ruff rend les items d'une liste multi-lignes avec **12 espaces** (8 + 4) — c'est la forme
    standard quand la liste est un argument d'un call replié à ``base_indent=8``.
    """
    # inner_indent = base_indent des items dans le corps du call GenerateContentConfig
    # = 8 (base_indent=4 pour les kwargs de GCC + 4 pour le repli).
    item_indent = 12  # 8 (inner du GCC replié) + 4 (un niveau de liste supplémentaire)
    pad = " " * item_indent
    closing_pad = " " * 8  # même niveau que les args de GenerateContentConfig

    rendered_items: list[str] = []
    for ss in safety_settings:
        ss_call = _Call(
            "types.SafetySetting",
            (
                f"category=types.HarmCategory.{ss.category}",
                f"threshold=types.HarmBlockThreshold.{ss.threshold}",
            ),
        )
        # col = item_indent (on est à 12 col dans le source), base_indent = item_indent
        r = _render_call(ss_call, col=item_indent, base_indent=item_indent)
        rendered_items.append(f"{pad}{r},")

    items_str = "\n".join(rendered_items)
    return f"safety_settings=[\n{items_str}\n{closing_pad}]"


def _render_generate_content_config(gcc: GenerateContentConfigSpec) -> tuple[str, tuple[str, ...]]:
    """Rend ``types.GenerateContentConfig(...)`` + imports.

    Seuls les champs non-None/non-vides sont inclus. La structure est rendue via
    :class:`_Call` pour être stable pour ``ruff format``.
    """
    imports: list[str] = ["from google.genai import types"]
    args: list[str | _Call] = []

    if gcc.temperature is not None:
        args.append(f"temperature={gcc.temperature!r}")
    if gcc.max_output_tokens is not None:
        args.append(f"max_output_tokens={gcc.max_output_tokens!r}")
    if gcc.top_p is not None:
        args.append(f"top_p={gcc.top_p!r}")
    if gcc.top_k is not None:
        args.append(f"top_k={gcc.top_k!r}")

    if gcc.safety_settings:
        args.append(_render_safety_settings_arg(gcc.safety_settings))

    if gcc.response_modalities:
        mods = ", ".join(_py_str(m) for m in gcc.response_modalities)
        args.append(f"response_modalities=[{mods}]")

    call = _Call("types.GenerateContentConfig", tuple(args))
    # col = len("    generate_content_config=") to match how it's embedded in LlmAgent kwargs
    rendered = _render_call(call, col=len("    generate_content_config="), base_indent=4)
    return rendered, tuple(imports)


def _render_llm_with_imports(spec: AgentSpec) -> tuple[str, tuple[str, ...]]:
    """Rend un ``LlmAgent(...)`` en omettant les kwargs vides/None + renvoie les imports modèle.

    Si ``model_spec`` est défini, rend ``model=LiteLlm(...)`` ; sinon ``model="<gemini>"``.
    Si ``generate_content_config`` est défini, rend le kwarg correspondant.
    Renvoie ``(bloc_source, imports_supplémentaires)``.
    """
    extra_imports: list[str] = []

    # Rendu du model=
    if spec.model_spec is not None:
        model_rendered, model_imports = _render_litellm_model(spec.model_spec)
        extra_imports.extend(model_imports)
        model_value = model_rendered
    else:
        model_value = _py_str(spec.model)

    pairs: list[tuple[str, str]] = [
        ("name", _py_str(spec.name)),
        ("model", model_value),
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
    if spec.generate_content_config is not None:
        gcc_rendered, gcc_imports = _render_generate_content_config(spec.generate_content_config)
        extra_imports.extend(gcc_imports)
        pairs.append(("generate_content_config", gcc_rendered))

    block = f"{spec.name} = LlmAgent(\n{_render_kwargs(pairs)})\n"
    return block, tuple(extra_imports)


def _render_llm(spec: AgentSpec) -> str:
    """Wrapper de :func:`_render_llm_with_imports` — les imports supplémentaires sont collectés
    séparément via :func:`_collect_model_imports` lors du rendu du module complet."""
    block, _ = _render_llm_with_imports(spec)
    return block


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


def _collect_model_imports(ordered: list[AgentSpec]) -> list[str]:
    """Collecte les imports supplémentaires liés au rendu du modèle (LiteLlm, types, os).

    Appelle :func:`_render_litellm_model` / :func:`_render_generate_content_config` directement
    (sans re-rendre le bloc LlmAgent entier) pour éviter la duplication.
    """
    imports: list[str] = []
    for spec in ordered:
        if spec.type == "llm":
            if spec.model_spec is not None:
                _, model_imps = _render_litellm_model(spec.model_spec)
                imports.extend(model_imps)
            if spec.generate_content_config is not None:
                _, gcc_imps = _render_generate_content_config(spec.generate_content_config)
                imports.extend(gcc_imps)
    return imports


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

    # Imports supplémentaires du rendu modèle (LiteLlm, types, os).
    model_imports = _collect_model_imports(ordered)

    # Section d'imports. La ligne des classes d'agents garde l'**ordre canonique** ADK
    # (LlmAgent, Sequential, Parallel, Loop, BaseAgent) — pas un tri alphabétique. Les imports
    # d'outils + modèle sont fusionnés par module (noms dédupliqués + triés). Les imports stdlib
    # (ex. ``import os``) sont extraits et placés **avant** les imports third-party (ruff isort).
    all_tool_and_model_imports = [imp for tr in tool_renders for imp in tr.imports] + model_imports
    merged = _merge_tool_imports(all_tool_and_model_imports)

    # Sépare stdlib plain-imports (ex. ``import os``) des ``from <module> import ...`` third-party.
    stdlib_imports: list[str] = []
    thirdparty_imports: list[str] = []
    for stmt in merged:
        if stmt.startswith("import ") and not stmt.startswith("import google"):
            stdlib_imports.append(stmt)
        else:
            thirdparty_imports.append(stmt)

    # Ordre final : stdlib seul si présent, puis agents ADK (from google.adk.agents),
    # puis autres third-party. Un saut de ligne entre les groupes (isort).
    import_lines: list[str] = []
    if stdlib_imports:
        import_lines.extend(stdlib_imports)
        import_lines.append("")  # blank line between stdlib and third-party
    agent_import_stmt = _agent_import_line(model).rstrip("\n")
    if agent_import_stmt:
        import_lines.append(agent_import_stmt)
    import_lines.extend(thirdparty_imports)

    import_block = ("\n".join(import_lines) + "\n\n") if import_lines else ""

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
