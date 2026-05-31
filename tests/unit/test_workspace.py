from pathlib import Path

from adk_toolkit_mcp.workspace import Workspace


def test_write_then_read_roundtrip(tmp_path: Path):
    ws = Workspace(tmp_path)
    ws.write("agent.py", "root_agent = 1\n")
    assert ws.read("agent.py") == "root_agent = 1\n"


def test_write_is_idempotent(tmp_path: Path):
    ws = Workspace(tmp_path)
    assert ws.write("a.txt", "x") is True
    assert ws.write("a.txt", "x") is False
    assert ws.write("a.txt", "y") is True


def test_has_root_agent_detects_assignment(tmp_path: Path):
    ws = Workspace(tmp_path)
    ws.write("agent.py", "root_agent = object()\n")
    assert ws.has_root_agent() is True


def test_has_root_agent_false_when_absent(tmp_path: Path):
    ws = Workspace(tmp_path)
    ws.write("agent.py", "x = 1\n")
    assert ws.has_root_agent() is False
