import pytest

from adk_toolkit_mcp.deps import MissingDependency, require


def test_require_returns_present_module():
    mod = require("json", extra="dev")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_missing_raises_with_extra_hint():
    with pytest.raises(MissingDependency) as exc:
        require("nonexistent_pkg_zzz", extra="bigquery")
    msg = str(exc.value)
    assert "bigquery" in msg
    assert "nonexistent_pkg_zzz" in msg
