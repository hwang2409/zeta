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

One message is one schema-versioned JSON file. A body larger than 64 KiB is stored in `bodies/<message-id>.txt`, and the JSON contains that file reference. Bodies are not truncated. Reusing an ID with the same immutable message content is idempotent. Reusing it with different content fails. The `done/` history retains the 100 most recent messages and removes their spilled bodies when it prunes them.

A claim renames `new/<id>.json` to `claimed/<id>.json` while holding the inbox directory lock, so only one session wins. It then records the claiming session and time. Completion requires the same claiming session, records an outcome, and moves the file to `done/`. An optional reply creates a new `reply` message in the sender project's inbox.

Zeta checks liveness with the existing session-directory lease. If the claiming session directory is absent, or its shared lease can be upgraded to an exclusive lease, the session is not alive. The next inbox scan returns its claimed messages to `new/` with a recovery note.

## Use

The model has one `inbox` tool with `send`, `list`, `claim`, `done`, and `projects` actions. Tool policy applies to the single name `inbox`. Scoped approval rules receive subjects such as `send zeta`, so rules can distinguish actions and destination projects.

Humans can inspect the current project's inbox with `/inbox`, or any known project with:

```console
zeta inbox --project zeta
```

A top-level session scans at startup, every two seconds while idle, and after each tool batch. A changed set of new messages produces one durable notice. The TUI shows the notice and an idle TUI session starts a notification turn, so the model sees it. During an active turn, the model sees it at the next turn boundary. Serve exposes the durable notification through its existing notification event; it adds no separate protocol. Child agents do not receive the inbox tool or run inbox polling.

There is no daemon, network transport, ownership election, or automatic model turn for each message.
