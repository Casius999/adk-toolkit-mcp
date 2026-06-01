# 05 — Memory & artifacts (the `memory` and `artifacts` domains)

Two distinct runtime services. **Memory** = searchable long-term recall of past conversations.
**Artifacts** = versioned binary/text blobs (files) attached to a user/session. Both are real ADK
async services chosen by `runtime.json`. Maps to `memory_*` and `artifacts_*`.

## Memory vs Artifacts — when each

| | Memory | Artifacts |
|---|---|---|
| Stores | past **session content** (events with text) | named **`Part`** blobs (text or bytes), **versioned** |
| Retrieval | `search(query)` → matching memories | `load(filename, version?)` → the Part |
| Use for | "what did the user tell me before?", long-term recall across sessions | files the agent produces/consumes: images, reports, uploads, generated docs |
| In-memory recall | **keyword** match (not semantic) | exact filename + version |
| Backends | `in_memory`, `vertex_rag`, `vertex_memory_bank` | `in_memory`, `gcs` |

Rule of thumb: **remember a conversation → memory; store a file → artifacts.** They're independent;
configure each with its own `*_service_set`.

## Memory — `memory_*`

### `memory_service_set`
```
memory_service_set(path, app_name, kind, project=None, location=None,
                   rag_corpus=None, agent_engine_id=None)
```
- `kind` ∈ {`in_memory`, `vertex_rag`, `vertex_memory_bank`}. Shape-validate + persist (preserves
  session/artifacts backends).
- `vertex_rag` → needs `rag_corpus` (full RAG corpus name); extra **`gcp`**.
- `vertex_memory_bank` → needs `project` + `location` + `agent_engine_id`; extra **`gcp`**.

### `memory_add_session`
```
memory_add_session(path, app_name, user_id, session_id)   # async
```
Ingests an existing session into memory (`add_session_to_memory`). **Only events carrying text** are
indexed (a bare `state_delta` event is not recalled). Returns the session id + event count.

### `memory_search`
```
memory_search(path, app_name, user_id, query)   # async
```
Returns `{count, memories: [{author, timestamp, text, content}]}`.

> **⚠️ `InMemoryMemoryService` recall is KEYWORD-based, not semantic.** It word-splits the query and
> each indexed event's text and matches if **any** query word literally appears (case-insensitive).
> So "Paris" finds "The capital of France is Paris"; "zzzz" finds nothing. For semantic recall use a
> Vertex backend (`vertex_rag` / `vertex_memory_bank`). Don't expect embeddings from `in_memory`.

### Memory flow
1. `memory_service_set(path, app_name, kind="in_memory")`.
2. Have a session with **text** events (e.g. via `run_agent`, or `sessions_append_event(... text=...)`).
3. `memory_add_session(... user_id, session_id)` — ingest it.
4. `memory_search(... query="Paris")` — recall.

## Artifacts — `artifacts_*`

### `artifacts_service_set`
```
artifacts_service_set(path, app_name, kind, bucket=None)
```
- `kind` ∈ {`in_memory`, `gcs`}. `gcs` → needs `bucket`; extra **`gcp`**.

### `artifacts_save` (returns the new version int)
```
artifacts_save(path, app_name, user_id, session_id, filename,
               text=None, bytes_b64=None, mime_type="text/plain")   # async
```
Provide **exactly one** of `text` (→ `Part.from_text`) or `bytes_b64` (base64 → `Part.from_bytes` with
`mime_type`). Versions are **0-based**: first save → `0`, next → `1`, …

### Other artifact tools
| Tool | Key args | Notes |
|---|---|---|
| `artifacts_load` | `user_id, session_id, filename, version=None` | `version=None` → latest. Returns `{version, encoding, mime_type, text, bytes_b64}` (text Part → `text`; bytes Part → base64 `bytes_b64`). Missing → clean `err`. |
| `artifacts_list` | `user_id, session_id` | `{filenames: [...]}`. |
| `artifacts_delete` | `user_id, session_id, filename` | Removes all versions (idempotent). |
| `artifacts_versions` | `user_id, session_id, filename` | `{versions: [0, 1, ...]}`. |

### `user:` prefix = user-scoped artifacts

A `filename` prefixed **`user:`** (e.g. `user:profile.txt`) makes the artifact **user-scoped** —
shared across all of that user's sessions, not tied to one session. Unprefixed filenames are
session-scoped. (Mirrors the `user:` state prefix concept.)

### Artifact flow
1. `artifacts_service_set(path, app_name, kind="in_memory")`.
2. `artifacts_save(... filename="report.md", text="...")` → version 0.
3. `artifacts_save` again → version 1. `artifacts_versions` → `[0, 1]`.
4. `artifacts_load(... filename="report.md")` → latest; or `version=0` for the first.

## Extras & failure modes

- Vertex memory (`vertex_rag` / `vertex_memory_bank`) and GCS artifacts (`gcs`) need the **`gcp`** extra
  + Google Cloud credentials. The toolkit converts the missing-dependency `ImportError` (raised inside
  the service constructor) into an actionable `ValueError` → clean `err`: install `adk-toolkit-mcp[gcp]`.
- `in_memory` memory + `in_memory` artifacts are **core** (no extra) and the recommended dev default.
