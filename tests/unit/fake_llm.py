"""Modèles ``BaseLlm`` factices pour prouver l'exécution d'agents ADK **hors-ligne**.

Aucune clé API n'est requise : ``generate_content_async`` (un async generator côté ADK) renvoie
des ``LlmResponse`` canned. Deux modèles :

- :class:`FakeLlm` — renvoie toujours une unique réponse texte finale (``answer``).
- :class:`ScriptedLlm` — premier tour : un ``function_call`` vers ``tool_name`` avec
  ``tool_args`` ; tours suivants : la réponse texte finale (``final_text``).

``BaseLlm`` est un modèle pydantic : l'état de scénario est déclaré en champs pydantic. Ces
classes sont importables depuis un ``agent.py`` généré (via ``sys.path``) pour prouver l'outil
``run_agent`` monté de bout en bout, sans clé.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types


class FakeLlm(BaseLlm):
    """Renvoie une unique réponse texte finale, quel que soit l'input (offline)."""

    answer: str = "Hello from FakeLlm."

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Émet exactement une ``LlmResponse`` finale portant ``answer`` (partial=False)."""
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part.from_text(text=self.answer)]),
            partial=False,
        )


class ScriptedLlm(BaseLlm):
    """Tour 1 → un ``function_call`` ; tours suivants → réponse texte finale (offline)."""

    tool_name: str = "add_numbers"
    tool_args: dict[str, Any] = {"a": 2, "b": 3}
    final_text: str = "The sum is 5."
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Premier appel : function_call ; ensuite : texte final (boucle d'agent complète)."""
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
    """Additionne deux entiers et renvoie la somme (outil ADK pour la preuve tool-call)."""
    return a + b
