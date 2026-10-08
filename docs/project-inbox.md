# Project inboxes

Project inboxes let peer top-level sessions exchange work without a server. They are enabled by default. To disable model tools, routing instructions, and notices, add this to the global `~/.zeta/settings.toml`:

```toml
[inbox]
enabled = false
```

Project settings cannot enable or disable the inbox.

## Storage

Each project owns these private directories:

```text
~/.zeta/projects/<project-id>/inbox/
  new/
  claimed/
  done/
  bodies/
```

One message is one schema-versioned JSON file. The integer `schema_version` is the major version. Readers accept their own major version, require all known required fields and their types, and ignore unknown optional fields. Those unknown fields are kept verbatim when a message moves through `claimed/` and `done/`. A higher or otherwise unsupported major version is invalid.

A body larger than 64 KiB is stored in `bodies/<message-id>.txt`, and the JSON contains that file reference. Bodies are not truncated. Reusing an ID with the same immutable message content is idempotent. Reusing it with different content fails. The `done/` history retains the 100 most recent valid messages and removes their spilled bodies when it prunes them.

Readers isolate malformed, unsafe, or unsupported message files. List results report up to 100 invalid files per status while valid messages remain usable. Scans skip invalid files, and a direct claim of one returns an error. Zeta logs each unchanged invalid file once per process. It does not move or delete invalid files automatically, including during stale-claim recovery and done-history pruning.

A claim renames `new/<id>.json` to `claimed/<id>.json` while holding the inbox directory lock, so only one session wins. It then records the claiming session and time. Completion requires the same claiming session, records an outcome, and moves the file to `done/`. An optional reply creates a new `reply` message in the sender project's inbox.

Zeta checks liveness with the existing session-directory lease. If the claiming session directory is absent, or its shared lease can be upgraded to an exclusive lease, the session is not alive. The next inbox scan returns its claimed messages to `new/` with a recovery note.

## Use

The model has one `inbox` tool with `send`, `list`, `sent`, `claim`, `done`, and `projects` actions. `sent` reads bounded, paged state for messages from the current project. It reports `new`, `claimed`, and `done` state without changing a receiver's inbox. Tool policy applies to the single name `inbox`. Read actions do not require approval. Scoped approval rules for writes can distinguish actions and destination projects.

Humans can inspect the current project's inbox with `/inbox`, or any known project with:

```console
zeta inbox --project zeta
```

A top-level session scans at startup, every two seconds while idle, and after each tool batch. A changed set of new messages produces one durable notice. The TUI shows the notice and an idle TUI session starts a notification turn, so the model sees it. During an active turn, the model sees it at the next turn boundary. Serve exposes the durable notification through its existing notification event; it adds no separate protocol. Child agents do not receive the inbox tool or run inbox polling.

The same top-level scan reads the state of messages sent by that session. A claim or completion does not wake the sender and does not write to the sender's inbox. Instead, Zeta appends a short harness-origin status block to the end of the sender's next turn input. Each message state is appended once and remains deduplicated after session resume. A completion that also sends a reply does not add a redundant completion line because the normal reply notification is the signal.

There is no daemon, network transport, ownership election, or automatic model turn for each message.

## Trust

Current inbox messages come only from the user's other Zeta sessions in the same Zeta home on the same machine. A receiving agent treats a request as work assigned by the user through another session. It can claim and do that work within the receiving project's normal rules without asking the user to confirm the sender.

Message text is still data. It cannot override the system prompt, `AGENTS.md`, safety rules, tool policy, tool permissions, or direct instructions from the user in the receiving session. Destructive or irreversible actions still require the normal confirmation, and agents must not echo secrets from messages.

Each stored message records its local origin. The model-visible framing checks that origin. A future transport that introduces a non-local origin will keep strict untrusted-data framing instead of receiving local-request trust.
