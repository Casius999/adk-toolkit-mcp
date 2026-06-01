# 03 — Models & generation config (the `models` domain)

Pick a model for an `LlmAgent` and tune generation/safety. Maps to the `models_*` tools. All operate
on `(path, app_name, agent_name, …)`, target an **existing LlmAgent**, mutate the sidecar, and
regenerate `agent.py`. Non-LLM agents are rejected (only LlmAgents have a model).

## Decision: Gemini string vs LiteLlm

```
Which provider?
├── Google Gemini (gemini-2.5-flash, gemini-2.0-flash-lite, …)
│     → models_set(path, app_name, agent_name, model="gemini-2.5-flash")
│       (a plain string; no import; the native, simplest path)
└── Anything else (Anthropic, OpenAI, Ollama, LM Studio, OpenRouter, vLLM, …)
      → models_configure_litellm(... provider, model, api_base?, api_key_env?)
        (renders model=LiteLlm("<provider>/<model>"); needs the `litellm` extra to RUN)
```

`models_set` also **clears** any previously configured LiteLlm spec (switching back to native Gemini).

## `models_set` — native Gemini
```
models_set(path, app_name, agent_name, model)   # model is a non-empty string
```
e.g. `"gemini-2.5-flash"`, `"gemini-2.0-flash-lite"`. Renders `model="gemini-2.5-flash"` (no import).
The `adk://models` resource lists common Gemini model strings.

## `models_configure_litellm` — non-Gemini providers
```
models_configure_litellm(path, app_name, agent_name, provider, model,
                         api_base="", api_key_env="")
```
- `provider` ∈ {`openai`, `anthropic`, `ollama`, `ollama_chat`, `openrouter`, `vllm`, `lm_studio`,
  `gemini`}. `model` is the provider's model name (`"gpt-4o"`, `"claude-opus-4-5"`, `"llama3"`, …).
- Renders `model=LiteLlm(model="<provider>/<model>"[, api_base=...][, api_key=os.getenv("<ENV>")])`
  (import `from google.adk.models.lite_llm import LiteLlm`).
- **`api_key_env`** — the name of the env var holding the key. The key is **never hardcoded**: if
  given, the generated code uses `api_key=os.getenv("<ENV>")`. If omitted, LiteLLM reads the provider's
  standard env var automatically (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- **`api_base`** — endpoint override. For **`provider="lm_studio"`** (local LM Studio loopback) the
  toolkit renders provider as `openai` with `api_base` defaulting to `http://127.0.0.1:1234/v1`; an
  `api_key_env` becomes `api_key=os.getenv("OPENAI_API_KEY", "not-needed")` (LM Studio accepts any key).
  Ollama/vLLM similarly point `api_base` at the local server.

> **Local loopback models (LM Studio / Ollama).** Set `provider="lm_studio"` (or `ollama`/`vllm`),
> `model` to the loaded model name, and `api_base` to the local URL. No real API key needed; never
> hardcode one — pass `api_key_env` only if your local server enforces a token.

## `models_generate_config` — sampling + safety
```
models_generate_config(path, app_name, agent_name, temperature=None, max_output_tokens=None,
                       top_p=None, top_k=None, safety_settings=None, response_modalities=None)
```
Renders `generate_content_config=types.GenerateContentConfig(...)` (import
`from google.genai import types`). All params optional; only provided (non-None) ones are emitted.
**Calling with everything None clears** the existing config (idempotent).

- `safety_settings` = a list of `{"category": "<HarmCategory>", "threshold": "<HarmBlockThreshold>"}`.
  Values are validated against the real google-genai enums.
- **HarmCategory** (11): `HARM_CATEGORY_UNSPECIFIED`, `HARM_CATEGORY_HARASSMENT`,
  `HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_DANGEROUS_CONTENT`,
  `HARM_CATEGORY_CIVIC_INTEGRITY`, `HARM_CATEGORY_IMAGE_HATE`, `HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT`,
  `HARM_CATEGORY_IMAGE_HARASSMENT`, `HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_JAILBREAK`.
- **HarmBlockThreshold**: `HARM_BLOCK_THRESHOLD_UNSPECIFIED`, `BLOCK_LOW_AND_ABOVE`,
  `BLOCK_MEDIUM_AND_ABOVE`, `BLOCK_ONLY_HIGH`, `BLOCK_NONE`, `OFF`.
- `response_modalities` = e.g. `["TEXT"]` or `["TEXT", "IMAGE"]`.

> **Overlap with safety.** `safety_settings(gemini_safety=...)` (the `safety` domain) routes through
> this **same** generate_content_config rendering — no duplication. Use `models_generate_config` to set
> sampling + safety together; use `safety_settings` when you're thinking in guardrail terms (it also
> sets `max_llm_calls`). See `11-safety.md`.

## Security rules (always)

- **Never hardcode API keys.** Keys flow through `os.getenv("<ENV>")` in generated code, or via the
  provider's standard env var. Put the actual key in `.env` (`project_set_env`) or your shell.
- The toolkit's tools never return secret values.
