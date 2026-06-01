"""Gated live end-to-end test: a REAL model (Kimi K2.6 via NVIDIA NIM) driven through
the MOUNTED MCP server (in-memory ``fastmcp.Client``), exercising the real tools end to
end — ``project_create`` → ``agents_create_llm`` → ``agents_set_root`` →
``models_configure_litellm`` → ``run_agent``.

CI-safe by construction: the NVIDIA ``nvapi-`` key is discovered from the repo-root
``.env`` (gitignored) using **stdlib only**, WITHOUT ever reading its value into output.
When no ``.env`` exists, or no entry whose value starts with ``nvapi-`` is found, the test
is **skipped cleanly** (``pytest.mark.skipif``) so CI without the key never fails. Locally,
with the key present, it runs the live flow and asserts a non-empty real answer (Paris).

Security: the key value is never printed, logged, or committed. Only the discovered env-var
NAME (safe) is used; the value is injected into ``os.environ[NAME]`` so the generated
``agent.py``'s ``api_key=os.getenv("<NAME>")`` resolves it at runtime.

Marked ``@pytest.mark.integration`` (registered in ``pyproject.toml`` to avoid unknown-marker
warnings under ``-W error``).
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import pytest
from fastmcp import Client

from adk_toolkit_mcp.server import build_server

# --- Live model parameters (NVIDIA NIM, OpenAI-compatible) ------------------- #
_API_BASE = "https://integrate.api.nvidia.com/v1"
_MODEL = "moonshotai/kimi-k2.6"
_PROVIDER = "openai"  # the OpenAI-compatible litellm client, pointed at the custom base
_QUESTION = "Quelle est la capitale de la France ? Réponds en une phrase."

#: Repo root = three levels up from this file (tests/integration/test_e2e_kimi.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


def _discover_nvapi_env() -> tuple[str, str] | None:
    """Parse the repo-root ``.env`` (stdlib only) and return ``(NAME, VALUE)`` for the first
    entry whose VALUE starts with ``nvapi-``; ``None`` if no ``.env`` or no such entry.

    Simple ``KEY=VALUE`` lines; ignores blanks/comments; strips matching surrounding quotes.
    The VALUE is returned for injection into ``os.environ`` only — never printed/logged.
    """
    if not _ENV_FILE.is_file():
        return None
    for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if value.startswith("nvapi-"):
            return name, value
    return None


_DISCOVERED = _discover_nvapi_env()
_SKIP_REASON = (
    "No NVIDIA 'nvapi-' key found in repo-root .env (file absent or no nvapi- value); "
    "skipping the live Kimi E2E test (expected in CI)."
)


@pytest.mark.integration
@pytest.mark.skipif(_DISCOVERED is None, reason=_SKIP_REASON)
async def test_kimi_k26_via_nvidia_nim_through_run_agent() -> None:
    """Drive a real Kimi K2.6 response through the mounted ``run_agent`` tool.

    Builds the server, then via an in-memory ``fastmcp.Client`` exercises the REAL mounted
    tools (not raw functions): scaffold → create LlmAgent → set root → configure LiteLlm
    against NVIDIA NIM → run. Asserts the run is ``ok`` and the final text is a non-empty
    real answer mentioning Paris.
    """
    assert _DISCOVERED is not None  # guarded by skipif; narrows type for mypy
    name, value = _DISCOVERED
    # Inject the key so the generated agent.py's os.getenv("<NAME>") resolves it. Never print it.
    os.environ[name] = value

    server = build_server()
    tmp = Path(tempfile.mkdtemp(prefix="adk_e2e_kimi_"))
    path = str(tmp)
    app = "demo"

    async with Client(server) as client:
        r = await client.call_tool("project_create", {"path": path, "app_name": app})
        assert r.data["ok"] is True, f"project_create failed: {r.data}"

        r = await client.call_tool(
            "agents_create_llm",
            {
                "path": path,
                "app_name": app,
                "name": "assistant",
                "instruction": "Tu es un assistant concis. Réponds en une seule phrase.",
            },
        )
        assert r.data["ok"] is True, f"agents_create_llm failed: {r.data}"

        r = await client.call_tool(
            "agents_set_root", {"path": path, "app_name": app, "name": "assistant"}
        )
        assert r.data["ok"] is True, f"agents_set_root failed: {r.data}"

        r = await client.call_tool(
            "models_configure_litellm",
            {
                "path": path,
                "app_name": app,
                "agent_name": "assistant",
                "provider": _PROVIDER,
                "model": _MODEL,
                "api_base": _API_BASE,
                "api_key_env": name,
            },
        )
        assert r.data["ok"] is True, f"models_configure_litellm failed: {r.data}"

        # Sanity: the generated agent.py wires LiteLlm correctly and never leaks the key.
        agent_py = (tmp / app / "agent.py").read_text(encoding="utf-8")
        assert f'model="{_PROVIDER}/{_MODEL}"' in agent_py
        assert f'api_base="{_API_BASE}"' in agent_py
        assert f'api_key=os.getenv("{name}")' in agent_py
        assert "nvapi-" not in agent_py, "SECURITY: raw key must never appear in generated code"

        # Live run through the mounted tool. litellm (an OpenAI-compatible client pointed at
        # NVIDIA NIM) may emit benign DeprecationWarnings from its deep deps; under
        # `-W error::DeprecationWarning` those would abort the call. Downgrade ONLY warnings
        # originating from litellm/httpx/pydantic-internal here, scoped to this call.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"litellm.*")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"httpx.*")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pydantic.*")
            run = await client.call_tool(
                "run_agent",
                {
                    "path": path,
                    "app_name": app,
                    "user_id": "u1",
                    "session_id": "s1",
                    "message": _QUESTION,
                },
            )

    data = run.data
    assert data["ok"] is True, f"run_agent failed: {data}"
    result = data["data"]
    final_text = result.get("final_text")

    # The proof: a non-empty, real answer that mentions Paris.
    assert final_text, f"empty final_text; events={result.get('events')}"
    assert "paris" in final_text.lower(), f"unexpected answer (no 'Paris'): {final_text!r}"

    # Print the verbatim Kimi response (the proof) + a short event summary (with -s).
    print("\n" + "=" * 70)
    print(f"KIMI K2.6 (via NVIDIA NIM) verbatim response through run_agent:\n{final_text}")
    print("-" * 70)
    print(f"event_count={result['event_count']} streaming_mode={result['streaming_mode']}")
    for i, ev in enumerate(result["events"]):
        print(
            f"  event[{i}] author={ev['author']} final={ev['is_final']} "
            f"text={(ev['text'] or '')[:80]!r}"
        )
    print("=" * 70)
