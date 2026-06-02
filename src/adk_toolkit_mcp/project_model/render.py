"""Full generation of ``agent.py`` from the :class:`ProjectModel`.

Core of the renderer: topological sort of the agents (cycle detection), rendering of each agent
type (``LlmAgent`` / workflow / loop / ``custom``), of the LiteLLM ``model=`` and of the
``generate_content_config=``, then assembly of the complete module — an import section ordered
isort-style (stdlib, ``google.adk.agents``, other third-party; sorted, deduplicated names) and
PEP 8 spacing (E302/E303/E305) stable for ``ruff format``.

Relies on :mod:`adk_toolkit_mcp.project_model._codegen` for the ruff-stable primitives and the
tool rendering, and on :mod:`adk_toolkit_mcp.project_model.specs` for the dataclasses and
constants. :func:`render_tool_ref` is re-imported here (used by the ``LlmAgent`` rendering) and
stays exposed via ``adk_toolkit_mcp.project_model``.
"""

from __future__ import annotations

import re
from typing import Any

from ..workspace import Workspace
from ._codegen import (
    _Call,
    _py_str,
    _render_call,
    callback_needs_refuse,
    callback_needs_user_text,
    refuse_helper_render,
    render_callback,
    render_tool_ref,
    user_text_helper_render,
)
from ._workflow_codegen import render_workflow_blocks, workflow_imports
from .sidecar import validate_workflow_graph as _validate_workflow_graph
from .specs import (
    _CLASS_FOR_TYPE,
    _IMPORT_ORDER,
    _REMOTE_A2A_IMPORT,
    _STDLIB_IMPORT_MODULES,
    LINE_LENGTH,
    AgentSpec,
    GenerateContentConfigSpec,
    LiteLlmSpec,
    ProjectModel,
    SafetySettingSpec,
    ToolRender,
)


# --------------------------------------------------------------------------- #
# Topological sort + cycle detection
# --------------------------------------------------------------------------- #
def _agent_dependencies(spec: AgentSpec) -> tuple[str, ...]:
    """Names of agents that ``spec`` depends on, to be defined after them in ``agent.py``.

    Two sources of dependency on another agent:
    - ``sub_agents`` (composition: the child must precede the parent);
    - an ``agent_tool`` tool targeting an agent (the target must precede the wrapping agent,
      otherwise ``AgentTool(agent=<target>)`` would reference an undefined variable).
    """
    deps: list[str] = list(spec.sub_agents)
    for tool in spec.tool_specs():
        if tool.kind == "agent_tool" and tool.target_agent:
            deps.append(tool.target_agent)
    return tuple(deps)


def topological_order(model: ProjectModel) -> list[AgentSpec]:
    """Sort the agents so that a dependency is defined before its dependent.

    A dependency = a ``sub_agent`` **or** the target of an ``agent_tool`` tool (cf.
    :func:`_agent_dependencies`). Raises ``ValueError`` if a cycle is detected (the tools convert
    it to ``err``). References to an absent name are ignored for ordering (existence validation is
    done upstream by the domain tools).
    """
    by_name: dict[str, AgentSpec] = {a.name: a for a in model.agents}
    order: list[AgentSpec] = []
    # States: 0 = unvisited, 1 = in progress (gray), 2 = done (black).
    state: dict[str, int] = {a.name: 0 for a in model.agents}

    def visit(name: str, path: tuple[str, ...]) -> None:
        st = state.get(name, 2)
        if st == 2:
            return
        if st == 1:
            cycle = " -> ".join((*path, name))
            raise ValueError(f"Cycle detected in agent dependencies: {cycle}")
        state[name] = 1
        spec = by_name[name]
        for dep in _agent_dependencies(spec):
            if dep in by_name:  # only orders known internal references
                visit(dep, (*path, name))
        state[name] = 2
        order.append(spec)

    # Stable order: we iterate in the model's insertion order.
    for a in model.agents:
        visit(a.name, ())
    return order


# --------------------------------------------------------------------------- #
# Source rendering — agents, model, imports
# --------------------------------------------------------------------------- #
def _render_kwargs(pairs: list[tuple[str, str]]) -> str:
    """Assemble already-rendered ``k=v`` pairs into a multi-line argument list."""
    return "".join(f"    {key}={value},\n" for key, value in pairs)


def _render_list_kwarg(key: str, refs: list[str]) -> str:
    """Render the **value** of a list kwarg (``tools``/``sub_agents``) ``ruff format``-style.

    Inline ``[a, b]`` if the line ``    {key}={value},`` fits in :data:`LINE_LENGTH`; otherwise,
    a multi-line list (one element per line, indent 8, trailing comma) — exactly what
    ``ruff format`` would produce beyond the limit. This way the generated ``agent.py`` is already
    stable (``format --check`` reformats nothing).
    """
    inline = f"[{', '.join(refs)}]"
    # 4 (kwarg indent) + len("key=") + len(inline) + 1 (trailing comma).
    if 4 + len(key) + 1 + len(inline) + 1 <= LINE_LENGTH:
        return inline
    items = "".join(f"        {ref},\n" for ref in refs)
    return f"[\n{items}    ]"


def _render_litellm_model(spec: LiteLlmSpec) -> tuple[str, tuple[str, ...]]:
    """Render ``LiteLlm(model="<provider>/<model>"[, api_base=...][, api_key=...])`` + imports.

    - For ``lm_studio``, the provider is rendered as ``openai`` and ``api_base`` defaults to
      ``http://127.0.0.1:1234/v1`` if not provided.
    - ``api_key`` is rendered as ``os.getenv("<ENV>")`` (+ ``import os``) only if ``api_key_env``
      is set. **The key is never hardcoded.**
    """
    provider = spec.provider
    api_base = spec.api_base

    # lm_studio: provider rendered as openai, default api_base.
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
    """Render the ``safety_settings=[...]`` argument for ``GenerateContentConfig``.

    The ``SafetySetting`` items are in the list at ``base_indent=8`` (within the body of
    ``GenerateContentConfig``, which is itself at base_indent=4 in ``LlmAgent``). Each
    ``SafetySetting(...)`` is rendered with ``_render_call(col=12, base_indent=12)`` so that the
    folding is stable for ``ruff format``.

    Ruff renders the items of a multi-line list with **12 spaces** (8 + 4) — this is the standard
    form when the list is an argument of a call folded at ``base_indent=8``.
    """
    # inner_indent = base_indent of the items within the GenerateContentConfig call body
    # = 8 (base_indent=4 for the GCC kwargs + 4 for the fold).
    item_indent = 12  # 8 (inner of the folded GCC) + 4 (one extra list level)
    pad = " " * item_indent
    closing_pad = " " * 8  # same level as the GenerateContentConfig args

    rendered_items: list[str] = []
    for ss in safety_settings:
        ss_call = _Call(
            "types.SafetySetting",
            (
                f"category=types.HarmCategory.{ss.category}",
                f"threshold=types.HarmBlockThreshold.{ss.threshold}",
            ),
        )
        # col = item_indent (we are at 12 col in the source), base_indent = item_indent
        r = _render_call(ss_call, col=item_indent, base_indent=item_indent)
        rendered_items.append(f"{pad}{r},")

    items_str = "\n".join(rendered_items)
    return f"safety_settings=[\n{items_str}\n{closing_pad}]"


def _render_generate_content_config(gcc: GenerateContentConfigSpec) -> tuple[str, tuple[str, ...]]:
    """Render ``types.GenerateContentConfig(...)`` + imports.

    Only the non-None/non-empty fields are included. The structure is rendered via :class:`_Call`
    to be stable for ``ruff format``.
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
    """Render an ``LlmAgent(...)`` omitting empty/None kwargs + return the model imports.

    If ``model_spec`` is set, renders ``model=LiteLlm(...)``; otherwise ``model="<gemini>"``.
    If ``generate_content_config`` is set, renders the corresponding kwarg.
    Returns ``(source_block, extra_imports)``.
    """
    extra_imports: list[str] = []

    # Rendering of model=
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
    # Guardrails (P4c): each callback is a generated top-level function; we attach its name via
    # the real kwarg (``before_model_callback=_guard_before_model_<agent>``). The body imports
    # (LlmResponse/types) are collected separately (cf. ``_collect_model_imports``).
    for cb in spec.callbacks:
        ref = render_callback(cb, spec.name).ref
        pairs.append((cb.kwarg_name(), ref))

    block = f"{spec.name} = LlmAgent(\n{_render_kwargs(pairs)})\n"
    return block, tuple(extra_imports)


def _render_llm(spec: AgentSpec) -> str:
    """Wrapper around :func:`_render_llm_with_imports` — the extra imports are collected
    separately via :func:`_collect_model_imports` when rendering the complete module."""
    block, _ = _render_llm_with_imports(spec)
    return block


def _render_workflow(spec: AgentSpec, class_name: str) -> str:
    """Render a ``SequentialAgent``/``ParallelAgent`` (name + sub_agents + description?)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("sub_agents", _render_list_kwarg("sub_agents", list(spec.sub_agents))))
    return f"{spec.name} = {class_name}(\n{_render_kwargs(pairs)})\n"


def _render_loop(spec: AgentSpec) -> str:
    """Render a ``LoopAgent`` (name + sub_agents + max_iterations + description?)."""
    pairs: list[tuple[str, str]] = [("name", _py_str(spec.name))]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    pairs.append(("sub_agents", _render_list_kwarg("sub_agents", list(spec.sub_agents))))
    pairs.append(("max_iterations", str(spec.max_iterations)))
    return f"{spec.name} = LoopAgent(\n{_render_kwargs(pairs)})\n"


def _render_remote_a2a(spec: AgentSpec) -> str:
    """Render a ``RemoteA2aAgent`` (name + agent_card + description?).

    ``agent_card`` is the URL (or JSON path) of the remote agent-card. The proxy has no children;
    it goes directly into other agents' ``tools=[...]``/``sub_agents=[...]`` like any agent
    variable. The import lives in a dedicated submodule — cf.
    :data:`~adk_toolkit_mcp.project_model.specs._REMOTE_A2A_IMPORT`.
    """
    pairs: list[tuple[str, str]] = [
        ("name", _py_str(spec.name)),
        ("agent_card", _py_str(spec.agent_card)),
    ]
    if spec.description:
        pairs.append(("description", _py_str(spec.description)))
    return f"{spec.name} = RemoteA2aAgent(\n{_render_kwargs(pairs)})\n"


def _custom_class_name(name: str) -> str:
    """PascalCase class name for a custom agent (``my_agent`` -> ``MyAgentAgent``)."""
    pascal = "".join(part.capitalize() for part in name.split("_") if part)
    if not pascal:
        pascal = "Custom"
    return f"{pascal}Agent"


def _render_custom(spec: AgentSpec) -> tuple[str, str]:
    """Render a ``BaseAgent`` subclass (stub) + a module-level instance.

    Returns a tuple ``(class_block, instance_block)`` to let the module renderer insert exactly 2
    blank lines between the two (PEP 8 E305).

    The ``_run_async_impl`` is a no-op **async generator** (``return`` then an unreachable
    ``yield``) — this is the valid form expected by ADK (cf. agents.md).
    """
    class_name = _custom_class_name(spec.name)
    desc = _py_str(spec.description) if spec.description else _py_str("")
    class_block = (
        f"class {class_name}(BaseAgent):\n"
        f'    """Generated custom agent (stub). Fill in `_run_async_impl`."""\n'
        "\n"
        "    async def _run_async_impl(self, ctx):\n"
        "        # TODO: implement the agent's logic.\n"
        "        return\n"
        "        yield  # makes this method an async generator (unreachable)\n"
    )
    instance_block = f"{spec.name} = {class_name}(name={_py_str(spec.name)}, description={desc})\n"
    return class_block, instance_block


def _render_agent_blocks(spec: AgentSpec) -> list[str]:
    """Return the list of code blocks (1 or 2) for a given agent.

    A ``custom`` agent emits two distinct blocks (class + instance) so the module renderer can
    insert the right number of blank lines between them. All other types emit a single assignment
    block.
    """
    if spec.type == "llm":
        return [_render_llm(spec)]
    if spec.type in ("sequential", "parallel"):
        return [_render_workflow(spec, _CLASS_FOR_TYPE[spec.type])]
    if spec.type == "loop":
        return [_render_loop(spec)]
    if spec.type == "remote_a2a":
        return [_render_remote_a2a(spec)]
    if spec.type == "custom":
        class_block, instance_block = _render_custom(spec)
        return [class_block, instance_block]
    raise ValueError(f"Unrendered agent type: {spec.type!r}")  # pragma: no cover


def _render_agent(spec: AgentSpec) -> str:
    """Dispatch to the renderer for the right type — returns a single block of text.

    Note: for a ``custom`` agent, the single block includes the class *and* the instance
    separated by an internal blank line. Use ``_render_agent_blocks`` (list) when fine-grained
    control of inter-block spacing in the complete module is needed.
    """
    if spec.type == "custom":
        class_block, instance_block = _render_custom(spec)
        return class_block + "\n" + instance_block
    blocks = _render_agent_blocks(spec)
    return blocks[0]


def _needed_agent_imports(model: ProjectModel) -> list[str]:
    """ADK agent classes imported **from ``google.adk.agents``**, in canonical order.

    ``remote_a2a`` is EXCLUDED: ``RemoteA2aAgent`` lives in a separate submodule
    (:data:`~adk_toolkit_mcp.project_model.specs._REMOTE_A2A_IMPORT`) and its import is added
    separately by :func:`render_agent_module`.
    """
    used: set[str] = set()
    for a in model.agents:
        if a.type == "custom":
            used.add("BaseAgent")
        elif a.type != "remote_a2a":
            used.add(_CLASS_FOR_TYPE[a.type])
    return [name for name in _IMPORT_ORDER if name in used]


def _uses_remote_a2a(model: ProjectModel) -> bool:
    """True if at least one agent in the model is of type ``remote_a2a`` (A2A proxy)."""
    return any(a.type == "remote_a2a" for a in model.agents)


def _collect_tool_renders(ordered: list[AgentSpec]) -> list[ToolRender]:
    """Render all the agents' tools (in the provided topo order) into a list of ``ToolRender``.

    The topological order guarantees that an ``agent_tool`` targeting an agent sees that agent
    defined before the wrapping agent (the tool helpers are emitted before *all* the agents, but
    the target being itself an agent, its instance precedes the wrapper in the agents section).
    """
    renders: list[ToolRender] = []
    for spec in ordered:
        for tool in spec.tools:
            renders.append(render_tool_ref(tool))
    return renders


def _collect_callback_renders(ordered: list[AgentSpec]) -> list[ToolRender]:
    """Render all the agents' guardrails (callbacks) into a list of ``ToolRender``.

    The guardrail functions are top-level ``def`` emitted BEFORE the agents (the agent references
    them by name via its kwarg). If at least one policy requires the shared helper ``_user_text``,
    it is inserted at the top (only once).
    """
    renders: list[ToolRender] = []
    needs_user_text = False
    needs_refuse = False
    for spec in ordered:
        for cb in spec.callbacks:
            renders.append(render_callback(cb, spec.name))
            needs_user_text = needs_user_text or callback_needs_user_text(cb)
            needs_refuse = needs_refuse or callback_needs_refuse(cb)
    # Shared helpers emitted ONCE, at the top (order: _user_text then _refuse). The
    # ``before_model`` guardrails call them by name; ``_refuse`` carries the ``LlmResponse``/
    # ``types`` imports (lifted to the module's import section).
    prelude: list[ToolRender] = []
    if needs_user_text:
        prelude.append(user_text_helper_render())
    if needs_refuse:
        prelude.append(refuse_helper_render())
    return prelude + renders


def _collect_model_imports(ordered: list[AgentSpec]) -> list[str]:
    """Collect the extra imports related to model rendering (LiteLlm, types, os).

    Calls :func:`_render_litellm_model` / :func:`_render_generate_content_config` directly
    (without re-rendering the whole LlmAgent block) to avoid duplication.
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
    """Deduplicate while preserving first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _agent_import_line(model: ProjectModel) -> str:
    """Import line for the agent classes (empty if no agent class is used).

    The **imported names are sorted** as ruff's isort rule (``I001``) requires: a standard
    ``sorted()`` (case-sensitive / ordinal — verified against ``ruff check --select I``),
    identical to the sorting applied to the tool/model imports by :func:`_render_import_line`.
    ``_needed_agent_imports`` keeps the ADK canonical order for the internal semantics; only the
    **emission** is sorted so the generated ``agent.py`` is isort-clean (and not just
    format-clean).
    """
    imports = _needed_agent_imports(model)
    if not imports:
        return ""
    return _render_import_line("google.adk.agents", sorted(imports)) + "\n"


def _merge_tool_imports(import_stmts: list[str]) -> list[str]:
    """Merge/sort ``from <module> import <name>`` lines isort-style (stable for ruff ``I``).

    - Groups by module; merges the names (deduplicated, sorted) onto a single line.
    - Sorts the modules alphabetically.
    Any unrecognized line (unlikely here) is kept as-is, at the top.
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


def _is_stdlib_import(stmt: str) -> bool:
    """True if ``stmt`` imports a **stdlib** module (isort places these before third-party).

    Recognizes both ``import <mod>[...]`` and ``from <mod>[...] import ...`` and checks the FIRST
    dotted segment of ``<mod>`` against :data:`_STDLIB_IMPORT_MODULES`. Multi-line parenthesized
    ``from`` imports (``from x import (``) are matched on their first line — fine here since the
    only stdlib imports the renderer emits (``os``, ``pathlib``) are always single-name inline.
    """
    head = stmt.lstrip()
    if head.startswith("import "):
        module = head[len("import ") :].strip()
    elif head.startswith("from "):
        module = head[len("from ") :].split(" import", 1)[0].strip()
    else:
        return False
    return module.split(".", 1)[0] in _STDLIB_IMPORT_MODULES


def _render_import_line(module: str, names: list[str]) -> str:
    """Render ``from <module> import a, b`` **stable for ``ruff format``**.

    Inline if the line fits in :data:`LINE_LENGTH`; otherwise, a multi-line parenthesized form
    (one name per line, indent 4, trailing comma) — exactly what ``ruff format`` produces beyond
    the limit for a multi-name import.
    """
    inline = f"from {module} import {', '.join(names)}"
    if len(inline) <= LINE_LENGTH:
        return inline
    body = "".join(f"    {name},\n" for name in names)
    return f"from {module} import (\n{body})"


# --------------------------------------------------------------------------- #
# Source rendering — complete module
# --------------------------------------------------------------------------- #
def _root_line(model: ProjectModel) -> str:
    """Render the trailing ``root_agent = <root>`` line (or a clear comment).

    Honors :attr:`ProjectModel.root_kind`: a ``"workflow"`` root must resolve to a defined
    workflow (a ``Workflow`` is a ``BaseNode``, which the ADK ``AgentLoader`` accepts as
    ``root_agent`` — cf. ``docs/adk-api-notes/workflow.md``); an ``"agent"`` root resolves to a
    defined agent.
    """
    if model.root is None:
        return "\n# root_agent undefined: call set_root (agent) or set_root (workflow).\n"
    if model.root_kind == "workflow":
        if model.get_workflow(model.root) is not None:
            return f"\nroot_agent = {model.root}\n"
        return f"\n# root workflow '{model.root}' not found; root_agent undefined.\n"
    if model.get(model.root) is not None:
        return f"\nroot_agent = {model.root}\n"
    return f"\n# root '{model.root}' not found among the agents; root_agent undefined.\n"


def render_agent_module(model: ProjectModel) -> str:
    """Produce valid ``agent.py`` source from the model.

    - Imports only the used classes (canonical order).
    - Defines each agent as a module variable, **topologically sorted** (a child before its
      parent). Cycle -> ``ValueError``.
    - Defines each workflow (function-node ``@node`` defs before the agents; join-node + the
      ``Workflow(...)`` assignment after the agents, so edges reference existing agent variables).
    - Omits empty/None kwargs.
    - Ends with ``root_agent = <root>`` (or a clear comment if the root is undefined).
    """
    header = (
        '"""Generated by adk-toolkit-mcp. DO NOT edit by hand: '
        "regenerated from the sidecar.\n\n"
        "Source of truth: `.adk_toolkit/agents.json`.\n"
        '"""\n\n'
    )

    if not model.agents and not model.workflows:
        body = "# No agent or workflow defined in the model.\n"
        root_line = "# root_agent undefined: add an agent/workflow then call set_root.\n"
        return header + body + "\n" + root_line

    ordered = topological_order(model)  # may raise ValueError (cycle)

    # Render the tools (imports + helpers + refs) in the topo order of the owning agents.
    tool_renders = _collect_tool_renders(ordered)
    # Deduplicate helpers while preserving first-appearance order: most helpers are unique
    # (function-tool defs, per-toolset assignments keyed by their variable name), but a *shared*
    # module-level anchor — e.g. ``_ADK_SKILLS_DIR = Path(__file__).parent / "skills"`` emitted by
    # every ``skill_toolset`` — must appear exactly once (E305/idempotence). Identical strings
    # collapse; distinct blocks never collide.
    tool_helpers = _dedup_preserve([helper for tr in tool_renders for helper in tr.helpers])

    # Render the guardrails (callbacks, P4c): top-level defs emitted before the agents + imports.
    callback_renders = _collect_callback_renders(ordered)
    callback_helpers = [helper for cr in callback_renders for helper in cr.helpers]

    # Extra imports from model rendering (LiteLlm, types, os).
    model_imports = _collect_model_imports(ordered)

    # Import section — fully **isort-clean** (ruff ``I001``), not just format-clean. The agent
    # classes line (``from google.adk.agents import ...``) is merged with the tool + model
    # imports then sorted **by module** like everything else (isort puts all third-party in a
    # single alphabetical block: ``crewai_tools`` < ``google.adk.*`` < ``langchain_community`` <
    # ``mcp`` …). The names inside each ``from X import a, b`` are sorted via
    # :func:`_render_import_line`. The stdlib imports (e.g. ``import os``) form a separate section
    # placed **before** the third-party (isort).
    agent_import_stmt = _agent_import_line(model).rstrip("\n")
    all_tool_and_model_imports = (
        [imp for tr in tool_renders for imp in tr.imports]
        + [imp for cr in callback_renders for imp in cr.imports]
        + model_imports
    )
    if agent_import_stmt:
        all_tool_and_model_imports.append(agent_import_stmt)
    # RemoteA2aAgent (P4b) lives in a dedicated submodule: its import is added here then sorted by
    # module with everything else by ``_merge_tool_imports`` (isort-clean).
    if _uses_remote_a2a(model):
        all_tool_and_model_imports.append(_REMOTE_A2A_IMPORT)
    # Workflow engine imports (``from google.adk.workflow import ...``): merged + sorted by module
    # with everything else (isort-clean). One ``from`` line per workflow; the merger dedups names.
    for wf in model.workflows:
        all_tool_and_model_imports.extend(workflow_imports(wf))
    merged = _merge_tool_imports(all_tool_and_model_imports)

    # Separate stdlib imports (``import os``, ``from pathlib import Path``) from third-party ones,
    # so isort's stdlib group precedes the third-party block (cf. :func:`_is_stdlib_import`).
    stdlib_imports: list[str] = []
    thirdparty_imports: list[str] = []
    for stmt in merged:
        if _is_stdlib_import(stmt):
            stdlib_imports.append(stmt)
        else:
            thirdparty_imports.append(stmt)

    # Final order: stdlib section (sorted) if present, a blank line, then the third-party already
    # sorted by module by :func:`_merge_tool_imports` (which includes the agents line).
    import_lines: list[str] = []
    if stdlib_imports:
        import_lines.extend(sorted(stdlib_imports))
        import_lines.append("")  # blank line between stdlib and third-party
    import_lines.extend(thirdparty_imports)

    import_block = ("\n".join(import_lines) + "\n\n") if import_lines else ""

    # Workflows: function-node ``@node`` defs are emitted with the helpers (before agents); join
    # + ``Workflow(...)`` assignments are emitted AFTER the agents (their edges reference agent
    # variables, which must already be defined).
    workflow_helpers: list[str] = []
    workflow_assigns: list[str] = []
    for wf in model.workflows:
        # A workflow whose graph does not yet validate renders as a placeholder (its
        # ``Workflow(...)`` call would raise at construction). ``set_root`` requires a valid
        # graph, so a rooted workflow is always materialized.
        complete = _validate_workflow_graph(wf) is None
        helpers, assigns = render_workflow_blocks(wf, complete=complete)
        workflow_helpers.extend(helpers)
        workflow_assigns.extend(assigns)

    # Top-level blocks: tool helpers, THEN guardrails (callbacks), THEN workflow function-node
    # defs, THEN the agents (which reference the guardrail functions by name), THEN the workflow
    # join/assign blocks (which reference agent variables). Each agent emits 1 block
    # (llm/workflow/loop) or 2 (custom: class + instance).
    agent_blocks: list[str] = []
    for spec in ordered:
        agent_blocks.extend(_render_agent_blocks(spec))
    all_blocks: list[str] = (
        tool_helpers + callback_helpers + workflow_helpers + agent_blocks + workflow_assigns
    )

    # PEP 8 / ruff-format spacing rules (E302, E303, E305):
    #   - Exactly 2 blank lines before a top-level class/def block.
    #   - Exactly 2 blank lines after a top-level class/def block.
    #   - 1 blank line between plain assignment blocks.
    #
    # Each block already ends with exactly one '\n'.
    # Separator '\n'  between two blocks → 1 blank line total (last \n + sep \n).
    # Separator '\n\n' between two blocks → 2 blank lines total.
    # A decorated function block starts with its decorator line (``@node``) — treat it like a def.
    def _starts_class_or_def(block: str) -> bool:
        return block.startswith(("class ", "def ", "@"))

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

    return header + import_block + blocks + _root_line(model)


# --------------------------------------------------------------------------- #
# On-disk regeneration
# --------------------------------------------------------------------------- #
def regenerate(ws: Workspace, model: ProjectModel) -> dict[str, Any]:
    """Write ``agent.py`` (rendered) + ensure ``__init__.py``. Idempotent.

    Returns ``{"agent_py", "init_py", "changed"}`` (absolute paths, global flag). May raise
    ``ValueError`` (cycle) — the calling tool converts it to ``err``.
    """
    source = render_agent_module(model)
    agent_changed = ws.write("agent.py", source)
    init_changed = ws.write("__init__.py", "from . import agent\n")
    return {
        "agent_py": str(ws.path("agent.py")),
        "init_py": str(ws.path("__init__.py")),
        "changed": agent_changed or init_changed,
    }
