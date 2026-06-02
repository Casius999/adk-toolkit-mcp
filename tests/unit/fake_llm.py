"""Fake ``BaseLlm`` models to prove ADK agent execution **offline**.

No API key required: ``generate_content_async`` (an async generator on the ADK side) returns
canned ``LlmResponse`` objects. Two models:

- :class:`FakeLlm` — always returns a single final text response (``answer``).
- :class:`ScriptedLlm` — first turn: a ``function_call`` to ``tool_name`` with ``tool_args``;
  subsequent turns: the final text response (``final_text``).

``BaseLlm`` is a pydantic model: the scenario state is declared in pydantic fields. These classes
are importable from a generated ``agent.py`` (via ``sys.path``) to prove the mounted ``run_agent``
tool end to end, without a key.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types


class FakeLlm(BaseLlm):
    """Returns a single final text response, regardless of the input (offline)."""

    answer: str = "Hello from FakeLlm."

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Emit exactly one final ``LlmResponse`` carrying ``answer`` (partial=False)."""
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part.from_text(text=self.answer)]),
            partial=False,
        )


class ScriptedLlm(BaseLlm):
    """Turn 1 → a ``function_call``; subsequent turns → final text response (offline)."""

    tool_name: str = "add_numbers"
    tool_args: dict[str, Any] = {"a": 2, "b": 3}
    final_text: str = "The sum is 5."
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """First call: function_call; then: final text (a complete agent loop)."""
        self.calls += 1
        if self.calls == 1:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_function_call(name=self.tool_name, args=self.tool_args)],
                ),
                partial=False,
            )
        else:
            yield LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text=self.final_text)]
                ),
                partial=False,
            )


def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum (ADK tool for the tool-call proof)."""
    return a + b
