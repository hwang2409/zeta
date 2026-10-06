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
replaced unless `--force` is explicit. Transfer archives reject absolute paths,
`..`, links, and special files.

## Working directory on another machine

The original absolute cwd often does not exist on the remote machine. During
publication or pull, Zeta maps it to:

```text
<remote ZETA_HOME>/remote-workspaces/<session-id>
```

You can select another location with `zeta session pull --cwd PATH`. Zeta
rewrites the root and child conversation headers, session metadata, and saved
shell cwd together, so resume does not fail because of a missing directory.
The command result and manifest contain the Git origin URL and a re-clone
notice. Zeta also adds this re-clone instruction to the transferred session's
stored system context, so the resumed agent sees the repository URL and mapped
cwd. Clone that repository into the mapped directory before asking the resumed
agent to work on the code.

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
incoming content beside it as `FILE.conflict-SOURCE-TIMESTAMP`. The command
reports conflicts and exits with status 1. It never silently clobbers either
version.

Automatic memory sync is intentionally not part of this feature. It can later
hook into the version/provenance events from the in-progress
`feat/memory-auto-reconcile` work without changing the explicit transfer
commands.
