"""Example 02 — a multi-agent pipeline (SequentialAgent) with a function tool.

Fully offline: it scaffolds the agents and prints the generated, deploy-ready
`agent.py` (no model or API key needed). Add a model as in example 01 to run it.

    uv run python examples/02_multi_agent.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from fastmcp import Client

from adk_toolkit_mcp.server import build_server


async def main() -> None:
    server = build_server()
    path = tempfile.mkdtemp(prefix="adk_example_")
    app = "pipeline"

    async with Client(server) as client:
        await client.call_tool("project_create", {"path": path, "app_name": app})
        await client.call_tool(
            "agents_create_llm",
            {
                "path": path,
                "app_name": app,
                "name": "researcher",
                "instruction": "Gather concise facts about the topic.",
            },
        )
        await client.call_tool(
            "agents_create_llm",
            {
                "path": path,
                "app_name": app,
                "name": "writer",
                "instruction": "Write a one-paragraph summary from the gathered facts.",
            },
        )
        await client.call_tool(
            "tools_add_function",
            {
                "path": path,
                "app_name": app,
                "agent_name": "researcher",
                "func_name": "web_lookup",
                "params": [{"name": "query", "type": "str"}],
                "docstring": "Look up a query and return notes.",
                "returns": "dict",
                "body": 'return {"notes": f"notes about {query}"}',
            },
        )
        await client.call_tool(
            "agents_create_sequential",
            {
                "path": path,
                "app_name": app,
                "name": "pipeline",
                "sub_agents": ["researcher", "writer"],
            },
        )
        await client.call_tool(
            "agents_set_root", {"path": path, "app_name": app, "name": "pipeline"}
        )

    print((Path(path) / app / "agent.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
