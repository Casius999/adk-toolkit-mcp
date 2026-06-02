# Examples

Runnable examples that drive the **mounted** `adk-toolkit-mcp` server through an
in-memory `fastmcp.Client` — the same tool surface a real MCP client (e.g. Claude
Code) uses.

```bash
uv sync --extra dev --extra litellm   # litellm only needed for live model runs
uv run python examples/02_multi_agent.py   # offline, prints generated agent.py
uv run python examples/03_eval.py          # offline, prints an evalset
uv run python examples/01_hello_agent.py   # live: needs a model + key (see below)
```

| File | What it shows | Needs a model? |
|------|---------------|----------------|
| `01_hello_agent.py` | Scaffold → wire any OpenAI-compatible model → `run_agent` | Yes |
| `02_multi_agent.py` | A `SequentialAgent` pipeline + a function tool (prints generated code) | No |
| `03_eval.py` | Create an evalset + criteria (offline ADK metrics) | No |

## Running a live model (example 01)

`01_hello_agent.py` reads the model from the environment so it works with **any**
OpenAI-compatible endpoint via LiteLLM:

```bash
# NVIDIA NIM (default)
export ADK_EXAMPLE_MODEL="moonshotai/kimi-k2.6"
export ADK_EXAMPLE_API_BASE="https://integrate.api.nvidia.com/v1"
export ADK_EXAMPLE_API_KEY_ENV="NVIDIA_API_KEY"
export NVIDIA_API_KEY="nvapi-..."        # or put it in a gitignored .env

# LM Studio (local):  ADK_EXAMPLE_API_BASE=http://localhost:1234/v1
# Ollama (local):     ADK_EXAMPLE_API_BASE=http://localhost:11434/v1
```

The key is read from the environment at run time and is never written to disk or
committed. Without a key, example 01 prints a hint and exits cleanly.
