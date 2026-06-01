# ADK API notes — `memory` and `artifacts` (P2b runtime services)

Captured 2026-06-01 by introspection. `google-adk` **2.1.0**, Python 3.12.

These notes back the shared `runtime.py` factory (`MemoryBackend` / `ArtifactBackend`
+ `get_memory_service` / `get_artifact_service`) and the `memory` / `artifacts` domain
sub-servers. Like `sessions` (P2a), these domains instantiate **real ADK service objects**
and call them (all async), rather than authoring `agent.py` source.

## Lazy module loading

Both `google.adk.memory` and `google.adk.artifacts` use a lazy `__getattr__`: the concrete
service classes do **not** appear in `dir(module)` (which only shows
`BaseMemoryService` / `BaseArtifactService`), but they are importable by name. Confirmed:

```python
from google.adk.memory import (
    BaseMemoryService,
    InMemoryMemoryService,         # google.adk.memory.in_memory_memory_service
    VertexAiRagMemoryService,      # google.adk.memory.vertex_ai_rag_memory_service
    VertexAiMemoryBankService,     # google.adk.memory.vertex_ai_memory_bank_service
)
from google.adk.artifacts import (
    BaseArtifactService,
    InMemoryArtifactService,       # google.adk.artifacts.in_memory_artifact_service
    GcsArtifactService,            # google.adk.artifacts.gcs_artifact_service
)
```

## Memory service API is fully ASYNC

```text
async add_session_to_memory(self, session: Session) -> None          # positional `session`
async search_memory(self, *, app_name: str, user_id: str,            # keyword-only
                     query: str) -> SearchMemoryResponse
```

Both are coroutines (`inspect.iscoroutinefunction` → True). `add_session_to_memory` takes a
**positional** `Session` (the same `Session` object returned by the session service); it
returns `None`. `search_memory` is **keyword-only**.

### `SearchMemoryResponse` / `MemoryEntry` shape

- `SearchMemoryResponse` (pydantic) has a single field `memories: list[MemoryEntry]`.
- `MemoryEntry` (pydantic) fields: `content` (a `google.genai.types.Content`),
  `custom_metadata`, `id`, `author`, `timestamp` (an ISO-8601 string via
  `_utils.format_timestamp`).
- Serialize to a plain dict with `entry.model_dump(exclude_none=True)`. The `content` nests
  as `{"parts": [{"text": "..."}], "role": "user"}`.

### `InMemoryMemoryService` recall is KEYWORD-BASED (not semantic)

The docstring says "prototyping purpose only … uses keyword matching instead of semantic
search". Confirmed mechanism from source:

- `add_session_to_memory` stores only events whose `content.parts` is non-empty (keyed by
  `"{app_name}/{user_id}"` → `{session_id: [events]}`).
- `search_memory` lowercases and word-splits (`re.findall(r"[A-Za-z]+", text)`) both the
  query and each event's text, and includes an event if **any** query word appears in the
  event's words.

**Functional-test consequence:** the session must contain events that carry **text content**
(`Event(content=types.Content(parts=[types.Part.from_text(text=...)]))`); a bare
`state_delta` event is NOT indexed (no `content.parts`). A query word must literally appear
(case-insensitively) in some event's text. Verified: events "The capital of France is Paris"
/ "Paris is a beautiful city" → query `"Paris"` returns 2 memories; query `"zzzz"` returns 0.

## Artifact service API is fully ASYNC

```text
async save_artifact(self, *, app_name, user_id, filename,            # keyword-only
                    artifact: types.Part | dict, session_id=None,
                    custom_metadata=None) -> int            # returns the new version
async load_artifact(self, *, app_name, user_id, filename,
                    session_id=None, version=None) -> Optional[types.Part]
async list_artifact_keys(self, *, app_name, user_id, session_id=None) -> list[str]
async delete_artifact(self, *, app_name, user_id, filename, session_id=None) -> None
async list_versions(self, *, app_name, user_id, filename, session_id=None) -> list[int]
```

All keyword-only and async. Note `session_id` is **optional** on every method (a `user:`-
prefixed filename is user-scoped and shared across sessions — see below), but the `artifacts`
domain always passes it.

### Confirmed lifecycle (InMemoryArtifactService)

- `save_artifact` returns a **0-based version int**: first save → `0`, next → `1`, …
- `load_artifact()` (no `version`) returns the **latest**; `version=N` returns that version.
  Returns `None` for a missing/deleted artifact (tools treat `None` as a clean `err(...)`).
- `list_artifact_keys` → list of filenames; `list_versions` → e.g. `[0, 1]`.
- `delete_artifact` removes all versions; a subsequent `load` returns `None` and the key
  disappears from `list_artifact_keys`.
- `user:`-prefixed filenames (e.g. `user:profile.txt`) are accepted and round-trip fine;
  the prefix marks the artifact **user-scoped** (persisted across sessions for that user).

## Building a `types.Part` from text / bytes

```python
from google.genai import types
types.Part.from_text(text="hello")                       # keyword-only `text`
types.Part.from_bytes(data=b"...", mime_type="image/png")  # keyword-only `data`, `mime_type`
```

A loaded text `Part` exposes `.text` (and `.inline_data is None`). A loaded bytes `Part`
exposes `.inline_data.data` (raw `bytes`) and `.inline_data.mime_type` (and `.text is None`).
The `artifacts.load` tool inspects these: returns `text` when `.text` is set, otherwise
base64-encodes `.inline_data.data` and reports its `mime_type`.

## Which backends need which extras

| Backend kind                | Service                        | Extra needed | Failure mode without it |
|-----------------------------|--------------------------------|--------------|-------------------------|
| memory `in_memory`          | `InMemoryMemoryService`        | core (none)  | — |
| memory `vertex_rag`         | `VertexAiRagMemoryService`     | `gcp`        | `ImportError` at ctor: "'google-cloud-aiplatform' … install google-adk[gcp]" |
| memory `vertex_memory_bank` | `VertexAiMemoryBankService`    | `gcp`        | `ImportError` at ctor (same message) |
| artifacts `in_memory`       | `InMemoryArtifactService`      | core (none)  | — |
| artifacts `gcs`             | `GcsArtifactService`           | `gcp`        | `ModuleNotFoundError: No module named 'google.cloud'` at ctor |

`runtime.py` converts any `ImportError` (which `ModuleNotFoundError` subclasses) from these
constructors into a `ValueError` with an actionable message pointing at the `gcp` extra
(`uv add 'adk-toolkit-mcp[gcp]'`), mirroring the `db`-extra handling for `DatabaseSessionService`.

### Backend constructor signatures (for config wiring)

```text
VertexAiRagMemoryService(rag_corpus=None, similarity_top_k=None, vector_distance_threshold=10)
VertexAiMemoryBankService(project=None, location=None, agent_engine_id=None, *, express_mode_api_key=None)
GcsArtifactService(bucket_name: str, **kwargs)
```

`MemoryBackend` therefore carries `project` / `location` / `rag_corpus` / `agent_engine_id`
(only the relevant ones are used per kind); `ArtifactBackend` carries `bucket`.

## Runtime singleton requirement (same as sessions)

`InMemoryMemoryService` and `InMemoryArtifactService` hold all state in process memory, so two
tool calls sharing the same in-memory backend MUST receive the **same instance** or state is
lost. `runtime.py` caches each by a stable key (kind + project/location/rag_corpus/bucket),
exactly like `get_session_service`. `reset_service_cache()` clears all three caches (sessions,
memory, artifacts) for test isolation.
</content>
</invoke>
