## Summary

What this PR changes and why.

## Checklist

- [ ] Follows the conventions in `docs/CONTRIBUTING.md` (bare tool names, `namespace=` mount, `{ok, data, error}` envelope).
- [ ] New/changed tools are covered by tests (TDD); generated code stays `ast.parse` + `ruff format` + `ruff check --select I` clean.
- [ ] `uv run ruff check .` and `uv run mypy src` pass.
- [ ] `uv run pytest -W error::DeprecationWarning --cov --cov-fail-under=80` passes.
- [ ] No secrets committed; optional deps imported lazily / codegen-only.
- [ ] If a tool was added/renamed: `skill/references/13-tool-catalog.md`, `docs/TOOL_CATALOG.md`, and the README surface table are updated (and the skill re-copied to `~/.claude/skills/adk-toolkit/`).

## Notes for reviewers

Anything that needs context (ADK API quirks, introspection notes, trade-offs).
