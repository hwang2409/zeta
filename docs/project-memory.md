# Project memory

Zeta stores project state and decisions as typed entries. New CLI-created,
explicitly initialized, and automatically discovered projects use entry memory
(format 2). Existing projects stay on the five-file format (format 1) until an
operator migrates each project. There is no global or implicit migration.

The Markdown files in `~/.zeta/projects/<id>/memory/` are generated, read-only
views. Direct edits are ignored and overwritten. The versioned store is the
source of truth. If its pointer is damaged, Zeta does not use a generated view
as recovery data. Mirror publication and repair are best effort and cannot fail
an authoritative update or context read.

Memory is loaded when a session starts or resumes. It does not refresh during a
running session. Agents can inspect memory, but only the background updater and
explicit user commands write entry memory. The model-facing `project_update`
tool remains available for format-1 projects. On a format-2 project it returns:
`this project uses entry memory; memory is maintained automatically`.

## Profiles

The default `zeta` profile contains `brief`, `decisions`, `state`, `backlog`,
and `changelog`. A messaging project instead uses `people`, `preferences`,
`routines`, `threads`, and `commitments`:

```sh
zeta project create ween --scope messaging --memory-profile messaging
zeta project init ~/src/ween --memory-profile messaging
zeta project memory ween schema set-profile messaging \
  --map-kind brief=people \
  --map-kind decisions=preferences \
  --map-kind state=threads \
  --map-kind backlog=commitments \
  --resolve-kind changelog
```

A profile is copied into the project state. It is not a live reference to a
global default. A profile change preserves entries in kinds with the same key.
It also preserves every explicit expiry. A new default expiry applies only to
later entries or later `seen_at` updates.

A removed populated kind requires one explicit action. Use `--map-kind OLD=NEW`
to retain its entries under a target kind, or `--resolve-kind KIND` to remove its
entries from the current view. Resolved entries remain in bounded version
history so undo can restore the complete schema transaction.

Messaging memory stores people, preferences, routines, threads, and
commitments. Messaging-agent behavior, such as quiet hours and follow-up
frequency, stays in the messaging agent and is not a memory rule.

## Automatic updates

Automatic reconciliation is enabled by default for sessions with a project.
One background worker reads durable transcript rows and applies validated entry
operations:

- before context eviction;
- after 50,000 estimated new transcript tokens;
- after 10 minutes without activity; or
- after a coalesced completed user turn.

Requests use the configured memory model. Zeta removes secrets and
agent-directed instructions before provider assembly. If active memory exceeds
the request budget, the request includes full text for the highest-priority,
most recently updated entries and an ID, kind, date, and short-text index for
every other active entry. Each automatic operation records its transcript
evidence and origin. Contradictions supersede older entries; completed work
resolves open entries; elapsed validity expires temporary entries. There is no
model-facing forget operation.

Configure `~/.zeta/settings.toml`:

```toml
[memory]
auto = true
model = "gpt-5.6-luna"
token_threshold = 50000
idle_minutes = 10
```

Use `--no-auto-memory` or `--auto-memory` for one session.

## Review and commands

Entry commands use entry IDs, not mirror filenames:

- `/memory log [kind|entry-id]` lists retained operation receipts.
- `/memory undo [entry-id|version-id]` restores a retained transaction.
- `/memory accept <entry-id>` accepts one active automatic entry.
- `zeta project memory <project> --json` prints structured entry state.
- `zeta project memory <project> --set <kind> <content>` replaces one kind with
  one accepted legacy-document entry.

Acceptance and manual mutation are user-only operations and require the normal
confirmation interface.

## Migrate, roll back, and finalize

Migration uses the deterministic migration engine. It does not call a model.
Each top-level list item and non-list paragraph becomes one typed entry; `##`
headings become stable section metadata. Oversized blocks split without
truncation. Entries keep their source order and the migration source digest and
version. A file's entries remain automatic only when that format-1 file was
automatic; other migrated entries are accepted. The exact five source files
remain protected for rollback. Activation first checks that updater, prompt
projection, commands/API, and synchronization all report format-2 support.

```sh
# Migrate one existing project. Its format-1 target remains protected.
zeta project memory <project-id-or-name> migrate

# Restore that protected format-1 pointer and its exact documents.
zeta project memory <project-id-or-name> rollback

# After verification, end special rollback protection.
zeta project memory <project-id-or-name> finalize
```

`migrate`, `rollback`, and `finalize` hold the project-registry lock. Migration
is idempotent for the source digest. Rollback remains available after normal
format-2 updates until `finalize`. Finalize is explicit and does not convert
entries back to files.

Zeta retains 128 recent versions. The pre-migration format-1 target is protected
from that bound only until finalize. After finalize, normal retention can remove
it. This retention is an undo facility, not secure erasure: session transcripts,
remote peers, filesystem snapshots, logs, and external backups can retain older
content beyond Zeta's memory-history window.

## Synchronization and safety

Format-2 synchronization transfers canonical entry state and schema. Mixed
format-1/format-2 peers fail closed for mutation; Zeta never flattens entries
back into five authoritative documents. Export and synchronization code must use
the logical registry interfaces, not copy private version-store paths.

The confirmation prompt prevents casual acceptance by an automatic model or a
non-interactive CLI caller. Shell access runs as the current user, so it is not
protection against a hostile process that edits the private store. Treat host
shell access and project files as trusted resources.
