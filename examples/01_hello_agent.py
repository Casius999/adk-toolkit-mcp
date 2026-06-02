"""Example 01 — scaffold an agent, wire any OpenAI-compatible model, run it live.

Drives the mounted adk-toolkit-mcp server through an in-memory fastmcp Client
(the same surface a real MCP client uses). Configure a model via env vars (or a
gitignored .env), then run:

    uv run python examples/01_hello_agent.py

Env vars (with NVIDIA NIM defaults):
    ADK_EXAMPLE_MODEL        model id, e.g. moonshotai/kimi-k2.6 / gpt-4o-mini / llama3.1
    ADK_EXAMPLE_API_BASE     OpenAI-compatible base URL, e.g.
                               https://integrate.api.nvidia.com/v1   (NVIDIA NIM)
                               http://localhost:1234/v1              (LM Studio)
                               http://localhost:11434/v1             (Ollama)
    ADK_EXAMPLE_API_KEY_ENV  NAME of the env var holding the key (default NVIDIA_API_KEY)

The key value is read from the environment at run time and never written to disk.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from fastmcp import Client

from adk_toolkit_mcp.server import build_server

MODEL = os.getenv("ADK_EXAMPLE_MODEL", "moonshotai/kimi-k2.6")
API_BASE = os.getenv("ADK_EXAMPLE_API_BASE", "https://integrate.api.nvidia.com/v1")
API_KEY_ENV = os.getenv("ADK_EXAMPLE_API_KEY_ENV", "NVIDIA_API_KEY")


async def main() -> None:
    if not os.getenv(API_KEY_ENV):
        print(
            f"No key in ${API_KEY_ENV}. Set {API_KEY_ENV} (and optionally "
            "ADK_EXAMPLE_MODEL / ADK_EXAMPLE_API_BASE) to run a live model."
        )
        return

    server = build_server()
    path = tempfile.mkdtemp(prefix="adk_example_")
    app = "hello"

    async with Client(server) as client:
        await client.call_tool("project_create", {"path": path, "app_name": app})
        await client.call_tool(
            "agents_create_llm",
            {
                "path": path,
                "app_name": app,
                "name": "assistant",
                "instruction": "You are a concise assistant. Answer in one sentence.",
            },
        )
        await client.call_tool(
            "agents_set_root", {"path": path, "app_name": app, "name": "assistant"}
        )
        await client.call_tool(
            "models_configure_litellm",
            {
                "path": path,
                "app_name": app,
                "agent_name": "assistant",
                "provider": "openai",
                "model": MODEL,
                "api_base": API_BASE,
                "api_key_env": API_KEY_ENV,
            },
        )
        run = await client.call_tool(
            "run_agent",
            {
                "path": path,
                "app_name": app,
                "user_id": "u1",
                "session_id": "s1",
                "message": "What is the capital of France? Answer in one sentence.",
            },
        )

    print(run.data["data"]["final_text"])


if __name__ == "__main__":
    asyncio.run(main())
