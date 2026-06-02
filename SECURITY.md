# Security Policy

## Supported versions

This project is pre-1.0. Security fixes are applied to the latest released version on the
`main` branch.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately via [GitHub Security Advisories](https://github.com/Casius999/adk-toolkit-mcp/security/advisories/new)
("Report a vulnerability"), or email the maintainer (see the `[project.urls]` /
`authors` entries in `pyproject.toml`).

When reporting, please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof-of-concept is ideal).
- The affected version / commit.

You can expect an acknowledgement within a few business days. We will work with you on a
fix and coordinate disclosure once a patch is available.

## Secret-handling posture

The toolkit is designed so that **secrets never enter source control or generated code**:

- **API keys come from the environment, never from arguments written to disk.** When you
  configure a non-Gemini model, `models_configure_litellm(..., api_key_env="MY_KEY")`
  emits `api_key=os.getenv("MY_KEY")` into the generated `agent.py` — the literal key value
  is never rendered. If `api_key_env` is omitted, LiteLLM reads the provider's standard
  environment variable directly.
- **`.env` is gitignored** and is the recommended place for local credentials. The
  `project_set_env` / `project_inspect` tools **redact every value to `***`** when reporting
  `.env` contents back to the client — values are written but never echoed.
- **Database URLs are credential-redacted.** `sessions_service_set(kind="database",
  db_url=...)` stores the URL but redacts the userinfo (`user:secret@host` → `***@host`)
  in every envelope it returns, so a connection string with an embedded password is never
  surfaced.
- **No shell injection surface.** Every call into the `adk` CLI (the `deploy` / `dev`
  domains) passes an explicit argument vector to `subprocess`; **`shell=True` is never
  used**, and every emitted CLI flag is validated against the installed `adk --help` so the
  toolkit cannot emit a flag the local ADK lacks.
- **Optional heavy dependencies are codegen-only or lazily imported.** Author domains never
  import `google-adk`; runtime domains import ADK inside the tool body and convert a missing
  extra into an actionable error. The toolkit emits code that *imports* an integration
  (BigQuery, Spanner, GCS, A2A, …) without importing it itself.
- **Generated code is validated before it is written** (`ast.parse` + `ruff format` +
  isort), reducing the chance of malformed or surprising output landing on disk.

## Scope and trust model

This is a **local developer tool** invoked over the MCP stdio transport by a trusted client
(e.g. Claude Code). It executes `adk` CLI subcommands and imports the agent code it
generates. Treat the directories you point it at, and the agent code you ask it to run, with
the same trust you would give any code you execute locally. Do not expose the server to
untrusted remote callers.
