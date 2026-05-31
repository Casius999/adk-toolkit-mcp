from fastmcp import FastMCP

from adk_toolkit_mcp.server import build_server


def test_build_server_returns_fastmcp():
    mcp = build_server()
    assert isinstance(mcp, FastMCP)


def test_main_is_callable():
    from adk_toolkit_mcp.server import main
    assert callable(main)
