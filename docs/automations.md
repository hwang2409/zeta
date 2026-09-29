# Automations

An automation is an ordinary zeta agent session started by an approved trigger.
Run the foreground daemon on the machine that should execute jobs:

```sh
zeta automation daemon
```

Use a process supervisor on a VPS to keep this command running. The daemon uses
`ZETA_HOME` (default `~/.zeta`), permits one daemon per home, and handles SIGINT
and SIGTERM. It executes one job at a time and checks schedules every 30 seconds.
A slow job can delay other jobs. Each attempt has a ten-minute timeout and each
agent phase has a 25-iteration limit. The webhook receiver defaults to the stable
loopback address `127.0.0.1:8765`; `--webhook-host` and `--webhook-port` override
it. A non-loopback bind is refused unless `--allow-non-loopback` is also present.
Public ingress/tunnelling is deliberately deferred to phase 2.

## Draft and approve

Ask the agent to draft an automation with its `automation` tool, or import a JSON
mapping with `zeta automation import jobs.json`. Both paths create inert drafts.
Example (substitute actual mounted tool names and an actual Slack recipient):

```json
{
  "morning-brief": {
    "prompt": "Read the selected services and summarize what needs my attention today.",
    "trigger": {"kind": "schedule", "cron": "0 8 * * 1-5", "timezone": "America/Toronto"},
    "servers": ["slack", "linear"],
    "allow": ["slack__slack_read_channel", "linear__list_issues"],
    "deliver": "slack:@austin",
    "provider": "claude",
    "model": "claude-sonnet-4-6",
    "cwd": "/srv/zeta-work"
  }
}
```

`provider`, `model`, and `cwd` can be omitted when drafting; the stored draft
resolves them from the originating session, home settings, and working directory.
Without configured provider defaults, CLI import uses Claude Sonnet 4.6. Review
these values for the daemon host before approving. `fake` is for injected test
backends, not a production automation provider.

```text
/automations
/automations morning-brief
/automations approve morning-brief
/automations approve morning-brief <displayed-token>
/automations disable morning-brief
/automations import .zeta/automations.json
```

Terminal equivalents are `zeta automation list`, `show <name>`, `approve <name>`,
`disable <name>`, and `import <file>`. Terminal approval shows the resolved job and
requires typing its review token; it cannot be piped a blanket yes. Approval
requires working selected services and an unambiguous Slack recipient. Explicit
Slack user/channel IDs are accepted; `@name` and `#channel` are resolved during
review. The resolved ID is pinned when approved.

Webhook triggers use one of these strict shapes (paths and URLs are never draft
fields):

```json
{"kind":"webhook", "verify":"github"}
```

```json
{"kind":"webhook", "verify":"hmac-sha256", "signature_header":"X-Service-Signature", "signature_prefix":"sha256=", "timestamp_header":"X-Service-Timestamp"}
```

The GitHub preset selects `X-Hub-Signature-256`, the `sha256=` prefix, and
`X-GitHub-Delivery`. A configured timestamp must be within five minutes and the
signed bytes are `<timestamp>.<body>`. Approval creates a separate random
32-byte secret and opaque route token for each job. They are harness-owned,
retained across reapproval, and stored only in the private automation store.
Human operators can use:

```text
zeta automation webhook url <job>
zeta automation webhook show-secret <job>
zeta automation webhook rotate-secret <job>
zeta automation webhook rotate-url <job>
```

The `/automations webhook ...` equivalents are also available. `url` prints
`http://127.0.0.1:8765/hooks/<opaque-token>` plus a phase-2 exposure note. Review
and model-tool output show trigger details but never credentials.

Only POST is accepted. The receiver looks up current approved/enabled state on
every request, reads no more than 1 MiB, verifies HMAC in constant time before
interpreting bytes, then applies a per-job rate limit. GitHub delivery IDs are
deduplicated permanently; otherwise a body hash is deduplicated within the
receiver window while all consumed records remain permanent. A valid duplicate
still receives 202 without creating a run. Accepted bytes, allowlisted headers,
and the accepted revision are fsync-durable in the same SQLite database as runs
before 202 is sent.

Every edit creates a new revision and suspends future execution until approved.
A running attempt retains its approved snapshot. Project JSON files are never
armed by discovery: explicitly import and approve a copy. Editing a source file
afterward does not change the approved copy. Imported entries cannot supply
approval state, cursors, or runtime state. Malformed entries appear as errors
without preventing valid entries from being scheduled.

## Permissions and services

Every model tool call is checked against the job's allow-list, including built-in
session tools and delegated calls. Unlisted tools deny; global approval settings
and `--yolo` do not broaden unattended permissions. Tool errors or denials fail
the run and prevent normal delivery. Tool names use zeta's `server__tool` spelling.
Use `/mcp` to inspect actual discovered tools; server tool catalogs can change.

A bare rule allows all arguments to that tool. An argument-scoped rule such as
`slack__slack_read_channel(C12345678)` applies only to a declared subject argument.
Built-ins already declare subjects. For MCP tools, declare trusted subject
mappings in home `mcp.json` (use names from the live tool schema). Project MCP
overlays cannot declare these mappings or redefine a trusted scope:


```json
{
  "servers": {
    "slack": {
      "transport": "streamable-http",
      "url": "https://mcp.slack.com/mcp",
      "auth": {
        "type": "oauth",
        "client_id": "${SLACK_CLIENT_ID}",
        "client_secret": "${SLACK_CLIENT_SECRET}",
        "callback_port": 3118,
        "scopes": ["chat:write", "search:read.users", "search:read.public", "channels:history"]
      },
      "approval_subjects": {"slack_read_channel": "channel_id"}
    },
    "linear": {
      "transport": "streamable-http",
      "url": "https://mcp.linear.app/mcp",
      "auth": {"type": "bearer", "token": "${LINEAR_API_KEY}"}
    }
  }
}
```

Adjust Slack scopes to the tools and conversations required. Register the exact
loopback redirect URI (`http://127.0.0.1:3118/callback`) with your own Slack app.
Complete `/mcp auth slack` interactively before running the daemon; on a VPS,
forward that loopback callback port over SSH and open the authorization URL on
your laptop. Fixed-client PKCE is supported, with an optional client secret;
omit `client_secret` if your app is configured for a public-client PKCE flow.
Do not use another harness's registered client ID.

The official Slack server uses registered-app OAuth, not a bot-token-only setup.
See [Slack's MCP documentation](https://docs.slack.dev/ai/slack-mcp-server/).
Linear supports API keys directly; see [Linear's MCP documentation](https://linear.app/docs/mcp).

Only home MCP configuration (or `ZETA_MCP_CONFIG`) is used. The daemon mounts the
job's selected servers and the Slack server selected by its delivery target.
Project MCP overlays, external Python tool modules, and lifecycle shell hooks
are not loaded. Secrets stay in the service configuration/token stores and never
become job fields. Protect the daemon home as you would the underlying credentials.

Agent provider authentication is unchanged: subscription OAuth is supported;
Claude API-key authentication requires both `ANTHROPIC_API_KEY` and
`ZETA_ALLOW_API_KEY=1`. It is the preferred unattended provider setup when available.

`deliver` grants the harness permission to send the final response to the pinned
recipient. It does not grant the model a messaging tool. Delivery uses one bounded
Slack message with the job name and session ID. Long responses are explicitly
truncated; the complete answer remains in the session.

## Time, polls, and recovery

Schedules use numeric five-field cron: wildcards, lists, ranges, and positive
steps. Day-of-month and day-of-week use POSIX matching. The default timezone is
`America/Toronto`. Nonexistent daylight-saving times skip; repeated local times
run once. Arming never backfills occurrences before approval. After downtime,
only the latest missed occurrence runs, and only if it is at most two hours old.

A poll trigger looks like:

```json
{"kind": "poll", "condition": "New urgent issues assigned to me", "interval_seconds": 300}
```

The first check runs after one interval. The check agent uses the same restricted
job permissions to evaluate activity in `(last_run, check_started_at]`. It must
return source event IDs, timestamps, and evidence in strict JSON. The harness
filters old/future events and deduplicates IDs. Source services must expose stable
IDs and timestamps; the relevance decision is still made by the agent.

Negative checks advance the consumed window. Positive checks consume IDs and
advance the window before executing the saved prompt in the same session.
Failed checks leave the window intact for the next scheduled check. A failed
execution does not replay consumed events. Downtime coalesces polls into one
current check rather than replaying every missed interval.

Runs use normal `sessions/<sid>/` transcripts. Inspect outcomes with
`/automations <name>` or `zeta automation show <name>`, then use the displayed
`zeta --resume <sid>` command. SQLite holds scheduling metadata under
`ZETA_HOME/automations/`; it is not a second transcript format.

Webhook deliveries use `pending → claimed(run_id) → done/interrupted`; claiming
and run creation are one SQLite transaction. Restart runs pending deliveries.
Once a run is claimed, interruption or worker failure is recorded and never
requeued. A delivery remains bound to the revision accepted at ingress; if that
revision is no longer the current approval before claim, it is recorded as
`skipped`/superseded rather than running old grants. Runs remain serialized per
job. Up to 64 KiB of decoded payload text and only allowlisted request headers
are appended to the saved prompt inside a conspicuous UNTRUSTED INPUT delimiter.
They cannot alter tools, prompt grants, recipient, provider/model, or cwd.

The loopback receiver is the extension point for phase-2 public ingress. The
per-job route and credential record is the extension point for phase-3 provider
self-registration; neither extension changes draft authority.

Interrupted executions and uncertain sends are never automatically replayed.
A network timeout can leave a Slack send uncertain even if Slack accepted it.
Inspect the session and destination before manually taking any further action.
Future scheduled occurrences continue normally.

## Verification status

Offline tests cover the complete draft/approve/fire/deliver/resume flow, strict
tool permissions, scheduler boundaries and recovery, plus fixed-client OAuth and
MCP mounting through simulated servers. Live service verification is separate:
the initial Linear discovery attempt returned `401 invalid_token`, and no Slack
configuration or designated test recipient was available. A successful live
mount and authorized test delivery are required before calling the deployment
verified.
