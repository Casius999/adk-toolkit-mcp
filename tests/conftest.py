from __future__ import annotations

from pathlib import Path

import pytest

from adk_toolkit_mcp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Workspace ADK isolé dans un tmpdir."""
    return Workspace(tmp_path)
