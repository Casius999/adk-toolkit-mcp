import json

import pytest
from fastmcp import Client, FastMCP

from adk_toolkit_mcp.prompts import register_prompts
from adk_toolkit_mcp.resources import register_resources
from adk_toolkit_mcp.versions import adk_versions


def test_register_resources_runs_without_error():
    mcp = FastMCP("t")
    register_resources(mcp)


def test_register_prompts_runs_without_error():
    mcp = FastMCP("t")
    register_prompts(mcp)


@pytest.mark.asyncio
async def test_version_resource_reads_back_adk_versions():
    mcp = FastMCP("t")
    register_resources(mcp)
    async with Client(mcp) as client:
        result = await client.read_resource("adk://version")
    # read_resource returns a list of content blocks (TextResourceContents);
    # the JSON payload is on the first block's `.text`.
    payload = json.loads(result[0].text)
    assert payload == adk_versions()
