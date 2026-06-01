# adk-toolkit-mcp — Models domain API notes

Captured 2026-06-01. google-adk 2.1.0, google-genai (core), Python 3.12.

## LiteLlm (`google.adk.models.lite_llm`)

**Import guard:** `LiteLlm` is importable only if `litellm` is installed (optional extra
`google-adk[extensions]`). The module raises `ImportError` at import time otherwise.

**Confirmed signature** (source inspection, `google.adk.models.lite_llm:2189`):

```python
class LiteLlm(BaseLlm):
    def __init__(self, model: str, **kwargs):
        ...
```

`model` is the LiteLLM model string including the provider prefix, e.g. `"openai/gpt-4o"`,
`"anthropic/claude-opus-4-5"`, `"ollama/llama3"`.  Additional kwargs are forwarded to the
`litellm.completion` / `litellm.acompletion` API.

Documented kwargs of interest:
- `api_base` — override the endpoint URL (used for LM Studio / local OpenAI-compat servers).
- `api_key` — override the API key (auth). **Never hardcode**; use `os.getenv("<ENV>")`.

LiteLLM reads provider-specific env vars automatically (e.g. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, etc.) when `api_key` is not provided.

## GenerateContentConfig (`google.genai.types.GenerateContentConfig`)

**Confirmed constructor** (all keyword-only, all optional):

```python
types.GenerateContentConfig(
    temperature=None,          # float
    top_p=None,                # float  (camelCase alias: topP)
    top_k=None,                # float  (camelCase alias: topK)
    max_output_tokens=None,    # int    (camelCase alias: maxOutputTokens)
    safety_settings=None,      # list[types.SafetySetting]
    response_modalities=None,  # list[str]  e.g. ["TEXT", "IMAGE"]
    # ... many more (system_instruction, stop_sequences, etc.)
)
```

Snake_case parameter names work in the constructor (pydantic aliases).

## SafetySetting

```python
types.SafetySetting(
    category=types.HarmCategory.<MEMBER>,
    threshold=types.HarmBlockThreshold.<MEMBER>,
)
```

## HarmCategory enum members (confirmed by introspection)

```
HARM_CATEGORY_UNSPECIFIED
HARM_CATEGORY_HARASSMENT
HARM_CATEGORY_HATE_SPEECH
HARM_CATEGORY_SEXUALLY_EXPLICIT
HARM_CATEGORY_DANGEROUS_CONTENT
HARM_CATEGORY_CIVIC_INTEGRITY
HARM_CATEGORY_IMAGE_HATE
HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT
HARM_CATEGORY_IMAGE_HARASSMENT
HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT
HARM_CATEGORY_JAILBREAK
```

## HarmBlockThreshold enum members (confirmed by introspection)

```
HARM_BLOCK_THRESHOLD_UNSPECIFIED
BLOCK_LOW_AND_ABOVE
BLOCK_MEDIUM_AND_ABOVE
BLOCK_ONLY_HIGH
BLOCK_NONE
OFF
```

## Codegen notes

- Gemini string model → `model="gemini-2.5-flash"` (unchanged, no import needed).
- LiteLlm model → `from google.adk.models.lite_llm import LiteLlm` + `model=LiteLlm(...)`.
- GenerateContentConfig → `from google.genai import types` + `generate_content_config=types.GenerateContentConfig(...)`.
- `api_key` is NEVER hardcoded in generated code: render `api_key=os.getenv("<ENV>")` only
  when `api_key_env` is provided; otherwise omit (LiteLLM reads provider env vars).
- For `lm_studio` provider: provider rendered as `openai`, `api_base` defaults to
  `http://127.0.0.1:1234/v1`, and `api_key=os.getenv("OPENAI_API_KEY", "not-needed")` is
  rendered only when `api_key_env` is given (LM Studio usually accepts any non-empty key).
