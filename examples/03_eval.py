"""Example 03 — build an evaluation set and criteria for an agent.

Offline: creates a `.evalset.json` + `test_config.json` and prints the evalset.
Running the eval (the `eval_run` tool) additionally needs a configured model
(see example 01); ADK's offline metrics — tool_trajectory and response_match
(ROUGE) — need no judge model.

    uv run python examples/03_eval.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile

from fastmcp import Client

from adk_toolkit_mcp.server import build_server


async def main() -> None:
    server = build_server()
    path = tempfile.mkdtemp(prefix="adk_example_")
    app = "graded"

    async with Client(server) as client:
        await client.call_tool("project_create", {"path": path, "app_name": app})
        await client.call_tool(
            "agents_create_llm",
            {
                "path": path,
                "app_name": app,
                "name": "assistant",
                "instruction": "Answer factual questions in one sentence.",
            },
        )
        await client.call_tool(
            "agents_set_root", {"path": path, "app_name": app, "name": "assistant"}
        )
        created = await client.call_tool(
            "eval_create_set",
            {
                "path": path,
                "app_name": app,
                "name": "capitals",
                "cases": [
                    {"query": "What is the capital of France?", "expected_response": "Paris"},
                    {"query": "What is the capital of Japan?", "expected_response": "Tokyo"},
                ],
            },
        )
        await client.call_tool(
            "eval_set_criteria",
            {
                "path": path,
                "app_name": app,
                "tool_trajectory_avg_score": 1.0,
                "response_match_score": 0.8,
            },
        )

    print(json.dumps(created.data["data"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
