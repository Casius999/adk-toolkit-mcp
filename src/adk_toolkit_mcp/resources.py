from __future__ import annotations

from fastmcp import FastMCP

from .versions import adk_versions

#: Catalogue statique des modèles Gemini connus et des providers LiteLLM supportés.
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
            "Passez le nom de modèle directement à models_set (ex. 'gemini-2.5-flash'). "
            "Consultez https://ai.google.dev/gemini-api/docs/models pour la liste complète."
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
                "Rendu comme provider 'openai' avec api_base='http://127.0.0.1:1234/v1' "
                "par défaut. Utilisez api_key_env si LM Studio requiert une clé."
            ),
            "ollama": "Utilisez 'ollama_chat' pour les modèles conversationnels.",
            "openrouter": (
                "Définissez OPENROUTER_API_KEY dans votre env et passez "
                "api_key_env='OPENROUTER_API_KEY'."
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
    """Enregistre les resources lecture-seule. Étendu aux phases suivantes."""

    @mcp.resource("adk://version", mime_type="application/json")
    def version_resource() -> dict[str, str]:
        return adk_versions()

    @mcp.resource("adk://models", mime_type="application/json")
    def models_resource() -> dict[str, object]:
        """Catalogue statique : modèles Gemini connus, providers LiteLLM supportés,
        membres des enums HarmCategory / HarmBlockThreshold."""
        return _MODELS_CATALOG
