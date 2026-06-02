from __future__ import annotations

import os

from fastmcp import FastMCP

from .domains.a2a import a2a_server
from .domains.agents import agents_server
from .domains.artifacts import artifacts_server
from .domains.deploy import deploy_server
from .domains.dev import dev_server
from .domains.eval import eval_server
from .domains.mcp_bridge import mcp_bridge_server
from .domains.memory import memory_server
from .domains.models import models_server
from .domains.observability import observability_server
from .domains.project import project_server
from .domains.run import run_server
from .domains.safety import safety_server
from .domains.sessions import sessions_server
from .domains.skills import skills_server
from .domains.tools import tools_server
from .domains.workflow import workflow_server
from .prompts import register_prompts
from .resources import register_resources

SERVER_NAME = "adk-toolkit-mcp"

#: Env values recognized as "truthy" to enable Code Mode (case-insensitive).
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})

#: Env variable enabling Code Mode at launch (``main()``).
_CODE_MODE_ENV = "ADK_TOOLKIT_CODE_MODE"


def code_mode_enabled() -> bool:
    """True if the ``ADK_TOOLKIT_CODE_MODE`` env variable requests Code Mode.

    Recognizes ``1``/``true``/``yes``/``on`` (case-insensitive). Any other value (or the absence
    of the variable) → ``False`` (direct-tools mode by default).
    """
    return (os.getenv(_CODE_MODE_ENV) or "").strip().lower() in _TRUTHY


def _apply_code_mode(mcp: FastMCP) -> None:
    """Collapse the tool catalog into a small discovery + execute surface (Code Mode).

    Applies the REAL FastMCP 3.3.1 transform
    (:class:`fastmcp.experimental.transforms.code_mode.CodeMode`) via
    :meth:`FastMCP.add_transform`. The exposed surface then goes from the 93 named tools to just
    ``search`` / ``get_schema`` / ``tags`` / ``execute`` (a big token saving for a large
    catalog). The discovery tools read ``tool.tags`` — hence the value of having tagged each tool
    by domain (TASK 1): ``GetTags`` lists the 17 domains, then ``search(tags=[...])`` filters by
    domain.

    NB (honesty, cf. ``docs/adk-api-notes/fastmcp-codemode.md``): the discovery tools
    (``search``/``get_schema``/``tags``) work WITHOUT any extra dependency; only the ``execute``
    tool (``MontySandboxProvider`` sandbox by default) requires the optional ``pydantic-monty``
    package (extra ``fastmcp[code-mode]``), imported lazily at call time. The transform is thus
    "wired" here, but executing code requires the extra. The import is local so it costs nothing
    for direct-tools mode (the default).
    """
    from fastmcp.experimental.transforms.code_mode import CodeMode, GetSchemas, GetTags, Search

    # GetTags is added to the default list (Search + GetSchemas) because we tag by domain: the
    # model can browse the domains, then search(tags=[...]), then get_schema, then execute.
    mcp.add_transform(CodeMode(discovery_tools=[Search(), GetSchemas(), GetTags()]))


def build_server(code_mode: bool = False) -> FastMCP:
    """Build the root MCP server (17 sub-servers, 93 tools).

    By default (``code_mode=False``), all tools are exposed by their ``<domain>_<bare>`` name
    (direct-tools UX; the read-through tests call them by name).

    If ``code_mode=True``, we apply the FastMCP 3.3.1 Code Mode transform AFTER mounting all the
    sub-servers: the catalog is collapsed into a discovery+execute surface
    (``search``/``get_schema``/``tags``/``execute``) — token saving for the 93 tools. See
    :func:`_apply_code_mode` and ``docs/adk-api-notes/fastmcp-codemode.md`` (the ``execute`` tool
    requires the ``fastmcp[code-mode]`` extra; discovery works without it).
    """
    mcp = FastMCP(SERVER_NAME)
    register_resources(mcp)
    register_prompts(mcp)
    # P1 domain 1/4: project. namespace -> tools exposed as `project_<name>`.
    # (`prefix=` is deprecated in fastmcp 3.3.1; `namespace=` is the current API.)
    mcp.mount(project_server, namespace="project")
    # P1 domain 2/4: agents. Tools exposed as `agents_<name>`.
    mcp.mount(agents_server, namespace="agents")
    # P3 domain 3/4: tools. Tools exposed as `tools_<name>`.
    mcp.mount(tools_server, namespace="tools")
    # P1 domain 4/4: models. Tools exposed as `models_<name>`.
    mcp.mount(models_server, namespace="models")
    # P2 domain a: sessions (runtime). Tools exposed as `sessions_<name>`.
    mcp.mount(sessions_server, namespace="sessions")
    # P2 domain b: memory (runtime). Tools exposed as `memory_<name>`.
    mcp.mount(memory_server, namespace="memory")
    # P2 domain b: artifacts (runtime). Tools exposed as `artifacts_<name>`.
    mcp.mount(artifacts_server, namespace="artifacts")
    # P3 domain a: run (agent execution). Tools exposed as `run_<name>`.
    mcp.mount(run_server, namespace="run")
    # P3 domain b: eval (agent evaluation). Tools exposed as `eval_<name>`.
    mcp.mount(eval_server, namespace="eval")
    # P4 domain a: deploy (building adk deploy commands). Exposed as `deploy_<name>`.
    mcp.mount(deploy_server, namespace="deploy")
    # P4 domain a: dev (long-running dev servers + one-shot run). Exposed as `dev_<name>`.
    mcp.mount(dev_server, namespace="dev")
    # P4 domain b: mcp_bridge (exposing ADK tools as MCP). Exposed as `mcp_bridge_<name>`.
    mcp.mount(mcp_bridge_server, namespace="mcp_bridge")
    # P4 domain b: a2a (consume/expose/agent_card Agent-to-Agent). Exposed as `a2a_<name>`.
    mcp.mount(a2a_server, namespace="a2a")
    # P4 domain c: safety (callbacks/plugins/safety settings). Exposed as `safety_<name>`.
    mcp.mount(safety_server, namespace="safety")
    # P4 domain c: observability (OpenTelemetry/Cloud Trace). Exposed as `observability_<name>`.
    mcp.mount(observability_server, namespace="observability")
    # P5: workflow (google.adk.workflow graph engine: conditional/cyclical orchestration).
    # Tools exposed as `workflow_<name>`.
    mcp.mount(workflow_server, namespace="workflow")
    # P7: skills (google.adk.skills Agent Skill Registry: create/list/load/attach/registry_info).
    # Tools exposed as `skills_<name>`.
    mcp.mount(skills_server, namespace="skills")
    # P6: opt-in Code Mode — AFTER all mounts (the transform acts on the complete catalog).
    if code_mode:
        _apply_code_mode(mcp)
    return mcp


def main() -> None:
    """CLI entry point: launch the server (Code Mode if ``ADK_TOOLKIT_CODE_MODE`` is truthy)."""
    build_server(code_mode=code_mode_enabled()).run()


if __name__ == "__main__":
    main()
