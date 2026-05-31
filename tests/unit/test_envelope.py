from adk_toolkit_mcp.envelope import err, ok


def test_ok_wraps_data():
    assert ok({"x": 1}) == {"ok": True, "data": {"x": 1}, "error": None}


def test_ok_defaults_none():
    assert ok() == {"ok": True, "data": None, "error": None}


def test_err_wraps_message():
    assert err("boom") == {"ok": False, "data": None, "error": "boom"}
