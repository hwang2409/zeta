# Remote session transfer

Remote transfer is explicit and opt-in. Zeta does not run a server and does not
create credentials. It uses the SSH host configuration and keys that already
work with `ssh`.

## Configure a remote

Add an alias to the global `settings.toml`:

```toml
[remotes]
neenerair = "ssh://neenerair"
```

An absolute remote Zeta home can be part of the URL:

```toml
[remotes]
lab-test = "ssh://neenerair/tmp/zeta-test-home"
```

A command can also use an SSH host name directly. This is an explicit host,
not an automatically discovered destination. Use `--remote-home` when that
host must use a Zeta home other than `~/.zeta`.

## Sessions

```console
zeta session push neenerair [SESSION_ID]
zeta session pull neenerair SESSION_ID [--cwd PATH]
```

`push` uses the most recent session when the ID is omitted. The approval-gated
`session_push` agent tool uploads the current durable session, so a user can ask
an agent to save its own session to a configured host.

A session snapshot contains:

- the root transcript and session metadata;
- child-agent transcripts and lifecycle metadata;
- persisted background-task output and state, but not running processes;
- tool spill files, because they are referenced by transcript history;
- the linked project's `project.json`, five standard memory files, and memory
  history directories when present;
- `transfer.json`, which records the schema, session ID, last sequence, content
  digest, source cwd, mapped resume cwd, transfer time, and the Git origin URL,
  branch, and HEAD.

Spill files can contain sensitive material returned by tools. The remote
machine is therefore trusted with all session content. Zeta does not copy
`settings.toml`, provider credentials, OAuth tokens, or any path outside the
selected session and project directories. Credential-shaped files found in a
session are excluded unless they are spill history.

Zeta holds every session append lock while it copies a snapshot. A writer can
continue after the copy, and the local session stays usable. Publication uses a
private incoming directory and a rename, so an incomplete upload is never
published. A destination with a higher sequence or a divergent digest is not
replaced unless `--force` is explicit. A pull cannot replace a session while a
process has that session open, including with `--force`.

Transfer archives reject absolute paths, `..`, links, and special files. SSH
stdout streams to a private temporary file instead of process memory. Before
and during extraction, Zeta enforces configurable safety ceilings of 200,000
archive members and 1 GiB of total uncompressed content, and it checks free
disk space. These generous ceilings protect against malformed archives; they
are not limits on tool output. `SshTransport` callers can raise them for a
legitimate larger session.

## Working directory on another machine

The original absolute cwd often does not exist on the remote machine. During
publication or pull, Zeta maps it to:

```text
<remote ZETA_HOME>/remote-workspaces/<session-id>
```

You can select another location with `zeta session pull --cwd PATH`. Zeta
rewrites the root and child conversation headers, session metadata, and saved
shell cwd together, so resume does not fail because of a missing directory.
The command result and validated manifest contain the Git origin URL and a
re-clone notice. The transferred session's stored system context contains only
a generic harness-authored reminder to verify the mapped working directory;
untrusted manifest or Git values are never inserted into that prompt. Clone
the repository named by the command result into the mapped directory before
asking the resumed agent to work on the code.

A full resume on the destination also needs Zeta installed and the selected
provider logged in. Non-interactive SSH does not need `zeta` or `uv`; transfer
only needs `/usr/bin/python3` on the destination.

## Project memory

From a directory associated with a project:

```console
zeta project memory push neenerair
zeta project memory pull neenerair
```

Use `--project ID_OR_NAME` outside that directory. Memory sync is explicit.
Each standard file uses a saved per-remote digest as a compare-and-swap base.
When both copies changed, Zeta keeps the destination unchanged and writes the
incoming content beside it as `FILE.conflict-SOURCE-TIMESTAMP`. The shared
state records both observed digests and remains unresolved across retries; the
common baseline does not advance. Resolve it explicitly with one of:

```console
zeta project memory resolve neenerair --accept local
zeta project memory resolve neenerair --accept remote
```

The command reports conflicts and exits with status 1. It never silently
clobbers either version.

Memory sync uses the versioned project-memory store. It retains automatic
provenance, records an undoable remote-sync import, and never marks content as
accepted by the user.

For format-2 snapshots, the Zeta-installed side performs full semantic decoding
before initial publication. A push validates the exact bytes before upload; a
pull validates them locally before publication. The SSH destination needs only
`/usr/bin/python3`, so its shipped publisher applies the shared dependency-free
project schema plus structural and hash checks. Those hashes ensure that the
remote publishes the same bytes that the sender validated.
