from __future__ import annotations

from fastmcp import FastMCP

from .versions import adk_versions

#: Static catalog of known Gemini models and supported LiteLLM providers.
_MODELS_CATALOG: dict[str, object] = {
    "gemini_models": {
        "recommended": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "note": (
            "Pass the model name directly to models_set (e.g. 'gemini-2.5-flash'). "
            "See https://ai.google.dev/gemini-api/docs/models for the full list."
        ),
    },
    "litellm_providers": {
        "supported": sorted(
            [
                "openai",
                "anthropic",
                "ollama",
                "ollama_chat",
                "openrouter",
                "vllm",
                "lm_studio",
                "gemini",
            ]
        ),
        "notes": {
            "lm_studio": (
                "Rendered as the 'openai' provider with api_base='http://127.0.0.1:1234/v1' "
                "by default. Use api_key_env if LM Studio requires a key."
            ),
            "ollama": "Use 'ollama_chat' for conversational models.",
            "openrouter": (
                "Set OPENROUTER_API_KEY in your env and pass api_key_env='OPENROUTER_API_KEY'."
            ),
        },
    },
    "harm_categories": sorted(
        [
            "HARM_CATEGORY_UNSPECIFIED",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
            "HARM_CATEGORY_IMAGE_HATE",
            "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
            "HARM_CATEGORY_IMAGE_HARASSMENT",
            "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_JAILBREAK",
        ]
    ),
    "harm_block_thresholds": sorted(
        [
            "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
            "BLOCK_LOW_AND_ABOVE",
            "BLOCK_MEDIUM_AND_ABOVE",
            "BLOCK_ONLY_HIGH",
            "BLOCK_NONE",
            "OFF",
        ]
    ),
}


def register_resources(mcp: FastMCP) -> None:
    """Register the read-only resources. Extended in later phases."""

    @mcp.resource("adk://version", mime_type="application/json")
    def version_resource() -> dict[str, str]:
        return adk_versions()

    @mcp.resource("adk://models", mime_type="application/json")
    def models_resource() -> dict[str, object]:
        """Static catalog: known Gemini models, supported LiteLLM providers,
        HarmCategory / HarmBlockThreshold enum members."""
        return _MODELS_CATALOG
