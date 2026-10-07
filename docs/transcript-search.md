# Transcript search

Zeta keeps a disposable SQLite FTS5 index for each project at
`~/.zeta/projects/<project-id>/transcript-index.sqlite3`. Project memory remains
small and describes project state and decisions. The transcript index stores
sanitized evidence for other history.

The index contains completed parent-session turns and parent-visible child-agent
completion reports. A turn joins input, agent reply text, and bounded tool
activity digests. It does not copy reasoning, raw tool output, or child
transcripts. Oversized turns use deterministic 12 KiB chunks with one source
sequence range. Shared project-memory secret patterns redact indexed text.

Index writes run after durable transcript persistence and outside the event
loop. Each project database has a schema version, generation, and per-session
cursor. Rebuilds publish atomically. Appends replace one session projection in a
transaction, so retries after a crash do not duplicate units. Session deletion
or reassignment removes that session from its old project index.

Operator diagnostics are available from the CLI:

```sh
zeta project index <project> status
zeta project index <project> rebuild
zeta project index <project> search "query" --limit 10
```

Search first ranks documents that contain all query terms. If there are not
enough results, it adds matches for the rarest query terms. There is no
model-facing search tool and no automatic context injection in this release.
The index is derived data and can be deleted or rebuilt at any time.
