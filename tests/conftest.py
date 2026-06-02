from __future__ import annotations

from pathlib import Path

import pytest

from adk_toolkit_mcp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """ADK workspace isolated in a tmpdir."""
    return Workspace(tmp_path)
