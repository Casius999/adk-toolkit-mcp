from adk_toolkit_mcp.versions import adk_versions


def test_adk_versions_reports_self_and_keys():
    v = adk_versions()
    assert v["adk_toolkit_mcp"]
    assert "google_adk" in v
    assert "fastmcp" in v
    assert "python" in v
