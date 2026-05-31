from fastmcp import FastMCP

from adk_toolkit_mcp.prompts import register_prompts
from adk_toolkit_mcp.resources import register_resources


def test_register_resources_runs_without_error():
    mcp = FastMCP("t")
    register_resources(mcp)


def test_register_prompts_runs_without_error():
    mcp = FastMCP("t")
    register_prompts(mcp)
