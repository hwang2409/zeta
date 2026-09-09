# zeta serve protocol

This document defines protocol version `1.1` (with `1.0` compatibility). It is the contract for native
clients. The transport is newline-delimited UTF-8 JSON. Each line is one
JSON-RPC 2.0 object. Frames are limited to 1 MiB.

The server accepts one client. A second connection receives a JSON-RPC error
with code `-32001`, then closes. The server uses a Unix socket by default:
`$ZETA_HOME/run/serve.sock`. `zeta serve --socket PATH` selects another Unix
socket. `zeta serve --port N` selects `127.0.0.1:N`.

## handshake

The first request must be `hello`. `protocol_version` is required and must be
`1.0` or `1.1`. A new client sends `protocol_version: "1.0"` plus
`client_version: "1.1"` so old servers accept the handshake. A new server returns
`1.1` for that request, or for an explicit `protocol_version: "1.1"`. A legacy
hello without the extra field receives `1.0` and only the legacy capabilities.
The GUI gates all extensions on the returned version; it never sends extension
requests to a 1.0 server. A mismatch returns `-32002` with `requested` and `supported` fields, then closes
the connection. Clients must not send other requests before `hello`.

Example request:

```json
{"jsonrpc":"2.0","id":1,"method":"hello","params":{"protocol_version":"1.0"}}
```

Example response:

```json
{"jsonrpc":"2.0","id":1,"result":{"protocol_version":"1.0","server":"zeta","capabilities":{"requests":["list_sessions","new_session","resume","send","steer","approve","deny","abort","status"],"notifications":["event"]}}}
```

## common JSON-RPC shapes

Every request has `jsonrpc` (`"2.0"`), `id` (a string or integer), `method`
(a non-empty string), and optional `params` (an object). Every response has
the request `id` and exactly one of `result` or `error`.
String request ids are limited to 128 UTF-8 bytes. This keeps error responses
within the frame limit.

Every notification has method `event`. Its `params.event` names the event and
its `params.session_id` identifies the active session when one exists.

The codec limits string request ids to 128 UTF-8 bytes and numeric request ids
to 128 decimal digits. An error uses the parsed id when the frame exposed a
safe id. It uses `null` when parsing did not expose an id or the id exceeded a
documented limit.

## requests

### `list_sessions`

Params: none. The result contains `sessions`, an array of session metadata.
Each metadata object has `version`, `session_id`, `created_at`, `updated_at`,
`provider`, `model`, `cwd`, `retained_tail`, `compaction_budget`,
`override_audit`, `system_prompt`, `context_files`, `vim_mode`, `budget_pinned`,
`plan_mode`, and `name`.

Server mode uses the effective launch provider after CLI and settings resolution.
Without either override, `zeta serve` uses fake mode.
Real-provider servers omit fake-provider sessions. A server whose effective launch
provider is `fake` lists only fake-provider sessions.

```json
{"jsonrpc":"2.0","id":2,"method":"list_sessions","params":{}}
```

```json
{"jsonrpc":"2.0","id":2,"result":{"sessions":[]}}
```

### `new_session`

Params: optional `provider` and `model`, both non-empty strings. The result has
`session`, containing the session metadata shape above. Settings provide
defaults when either field is absent.

```json
{"jsonrpc":"2.0","id":3,"method":"new_session","params":{"provider":"fake","model":"offline"}}
```

```json
{"jsonrpc":"2.0","id":3,"result":{"session":{"session_id":"abc123","provider":"fake","model":"offline"}}}
```

The example omits metadata fields for readability. A real response includes
all metadata fields.

### `resume`

Params: required `session_id`, a non-empty string. The result has `session`
with the full session metadata. The session must exist.

Real-provider servers reject fake sessions with RPC error `-32602`:
`session uses the offline test provider; open it with --provider fake`.
Fake-mode servers reject real sessions with the same code and a message
naming the required `--provider`. Both errors preserve the active session and
leave the rejected session's files untouched. Fake sessions still resume on
fake-mode servers; real sessions can resume across real providers.

```json
{"jsonrpc":"2.0","id":4,"method":"resume","params":{"session_id":"abc123"}}
```

```json
{"jsonrpc":"2.0","id":4,"result":{"session":{"session_id":"abc123","provider":"fake","model":"offline"}}}
```

### `send`

Params: required `text`, a non-empty string. The result acknowledges scheduling
with `accepted` (`true`) and `session_id`. Streaming starts as notifications.
Only one turn can run at a time.

```json
{"jsonrpc":"2.0","id":5,"method":"send","params":{"text":"hello"}}
```

```json
{"jsonrpc":"2.0","id":5,"result":{"accepted":true,"session_id":"abc123"}}
```

### `steer`

Params: required `text`, a non-empty string. The server queues this user
message at the next safe provider boundary. A turn must be running.

```json
{"jsonrpc":"2.0","id":6,"method":"steer","params":{"text":"also check the tests"}}
```

```json
{"jsonrpc":"2.0","id":6,"result":{"accepted":true}}
```

### `approve` and `deny`

Params: required `request_id`, a non-empty string matching an approval request.
The result contains `accepted`, `request_id`, and `decision` (`"approve"` or
`"deny"`). The decision wakes an active turn. For a resumed session, the
server executes the pending tool through the existing loop seam.

```json
{"jsonrpc":"2.0","id":7,"method":"approve","params":{"request_id":"tool-call-1"}}
```

```json
{"jsonrpc":"2.0","id":7,"result":{"accepted":true,"request_id":"tool-call-1","decision":"approve"}}
```

### `abort`

Params: none. The result contains `aborted` (`true` when a turn was canceled).
Abort cancels the active provider or tool task and persists the loop's partial
state.

```json
{"jsonrpc":"2.0","id":8,"method":"abort","params":{}}
```

```json
{"jsonrpc":"2.0","id":8,"result":{"aborted":true}}
```

### `status`

Params: none. The result contains `session` (full metadata or `null`), `state`
(`idle`, `running`, or `tool`), `pending_approvals`, `usage`, and
`compaction_markers`. Each pending approval has `request_id` and `tool_call`.

```json
{"jsonrpc":"2.0","id":9,"method":"status","params":{}}
```

```json
{"jsonrpc":"2.0","id":9,"result":{"session":null,"state":"idle","pending_approvals":[],"usage":{},"compaction_markers":0}}
```

## notifications

All examples below use the common event envelope:

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"turn_start","session_id":"abc123","data":{"turn":1}}}
```

`turn_start`, `turn_end`, `agent_start`, `agent_end`, `message_start`, and
`turn_aborted` carry `data` when the loop supplies it. `turn_end` data includes
`turn` and `tool_calls`.

`assistant_delta` carries `delta` (a bounded string) and `kind` (`assistant`,
`thinking`, or another content type). `assistant_message` carries `message`,
the committed message object with `role` and `content`.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"assistant_delta","session_id":"abc123","delta":"hello","kind":"assistant"}}
```

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"assistant_message","session_id":"abc123","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}}
```

`tool_start` and `tool_end` carry `tool_call` (`id`, `name`, `arguments`) and
`data`. `tool_end` also carries `tool_result` (`tool_call_id`, `content`,
`is_error`, and optional result fields). `tool_output` carries the same
`tool_call`, bounded `output`, and `data`.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"tool_start","session_id":"abc123","tool_call":{"id":"tool-call-1","name":"read","arguments":{"path":"README.md"}},"data":{}}}
```

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"tool_output","session_id":"abc123","tool_call":{"id":"tool-call-1","name":"read","arguments":{"path":"README.md"}},"output":"file contents","data":{}}}
```

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"tool_end","session_id":"abc123","tool_call":{"id":"tool-call-1","name":"read","arguments":{"path":"README.md"}},"tool_result":{"tool_call_id":"tool-call-1","content":"ok","is_error":false},"data":{}}}
```

`approval_request` carries `request_id` and `tool_call`. `approval_end` carries
the tool call and empty or loop-provided `data`.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"approval_request","session_id":"abc123","request_id":"tool-call-1","tool_call":{"id":"tool-call-1","name":"bash","arguments":{"command":"ls"}}}}
```

`usage` carries the provider usage object. `compaction_start` and
`compaction_end` carry loop `data`; the latter can include `token_count`.
`sub_agent_receipt` carries the existing agent notification `data` object.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"usage","session_id":"abc123","usage":{"input_tokens":10,"output_tokens":4}}}
```

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"compaction_end","session_id":"abc123","data":{"turn":2,"token_count":1200}}}
```

`error` is a loud structured event. It carries `error` with `code` and
`message`, plus a `data` object. The server emits it for provider and loop
failures, then returns to `idle`.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"error","session_id":"abc123","error":{"code":"backend_error","message":"provider failed"},"data":{}}}
```

## errors

Error responses use standard JSON-RPC codes where applicable:

| code | meaning |
| ---: | --- |
| `-32700` | invalid JSON or invalid UTF-8 frame |
| `-32600` | invalid JSON-RPC request or duplicate handshake |
| `-32601` | method is not supported |
| `-32602` | params have the wrong shape or value |
| `-32000` | unexpected server failure |
| `-32001` | another client owns the socket |
| `-32002` | handshake missing or protocol version mismatch |
| `-32003` | no active session |
| `-32004` | a turn is already running |
| `-32005` | no turn is running for steering |
| `-32006` | approval request is missing or already resolved |
| `-32007` | outbound frame exceeds the size limit |

Error objects have `code` and `message`, and may have a method-specific
`data` object. A malformed frame never terminates the server process. The
server returns its parse error and continues reading frames.

## normative wire schema

The following schema uses JSON Schema-like notation. `required` lists fields
that always exist. Other listed fields are optional.

```text
SessionMetadata = {
  required: {
    version: integer, session_id: string, created_at: string,
    updated_at: string, provider: string, model: string, cwd: string,
    retained_tail: integer, compaction_budget: integer,
    override_audit: array[object], system_prompt: string,
    context_files: array[string], vim_mode: boolean, budget_pinned: boolean,
    plan_mode: boolean, name: string
  }
}
ToolCall = { required: { id: string, name: string, arguments: object } }
ContentBlock = one of:
  { required: { type: "text", text: string }, optional: { path: string, size: integer } }
  { required: { type: "image", data: string, mimeType: string }, optional: { path: string, size: integer } }
  { required: { type: "thinking", text: string }, optional: { signature: string } }
  { required: { type: "redacted_thinking", data: string } }
  { required: { type: "tool_use", tool_call: ToolCall } }
Message = {
  required: { role: string, content: array[ContentBlock] },
  optional: { tool_result: ToolResult, metadata: object }
}
ToolResult = {
  required: { tool_call_id: string, content: string, is_error: boolean },
  optional: { is_canceled: boolean, content_blocks: array, structured_content: object }
}
Approval = { required: { request_id: string, tool_call: ToolCall } }
Usage = object with provider-defined JSON values
```

`created_at` and `updated_at` are ISO-8601 strings. IDs, names, paths, and
provider values are strings. `retained_tail` and `compaction_budget` are
positive integers. The server preserves usage keys and values.

Tool result `content_blocks` uses the MCP-compatible union. A `text` block
requires `text`, `truncated`, and `full_size`. An `image` block requires
`data` and `mimeType`. A `resource` block requires `resource` with `uri` and
exactly one of `text` or `blob`. Optional annotations contain `audience`,
`priority`, and `lastModified`. `structured_content` is a recursive JSON value.

The event envelope is always:

```text
Event = {
  required: { jsonrpc: "2.0", method: "event", params: { event: string } },
  optional params: { session_id: string, ...event_fields }
}
```

Event fields are:

| event | required fields | optional fields |
| --- | --- | --- |
| `turn_start`, `turn_end`, `agent_start`, `agent_end`, `message_start`, `turn_aborted`, `compaction_start`, `compaction_end` | `event` | `session_id`, `data: object` |
| `assistant_delta` | `event`, `delta: string`, `kind: string` | `session_id` |
| `assistant_message` | `event`, `message: Message` | `session_id` |
| `usage` | `event`, `usage: Usage` | `session_id` |
| `tool_start` | `event`, `tool_call: ToolCall`, `data: object` | `session_id` |
| `tool_output` | `event`, `tool_call: ToolCall`, `output: string`, `data: object` | `session_id` |
| `tool_end` | `event`, `tool_call: ToolCall`, `tool_result: ToolResult or null`, `data: object` | `session_id` |
| `approval_request` | `event`, `request_id: string`, `tool_call: ToolCall` | `session_id` |
| `approval_end` | `event`, `tool_call: ToolCall`, `data: object` | `session_id` |
| `sub_agent_receipt` | `event`, `data: object` | `session_id` |
| `retry` | `event`, `data: object` | `session_id` |
| `error` | `event`, `error: {code: string, message: string}`, `data: object` | `session_id` |

The `retry` data can contain `retry: integer`, `delay: number`, `text: string`,
and `is_stall: boolean`. Unknown provider keys remain allowed inside `data`.

## ordering and lifecycle rules

1. The server processes complete request lines in order. It parses inbound
   frames and serializes all responses and notifications through one bounded
   codec and one write lock.
2. The client sends `hello` first. Any rejected first request closes the client
   after its error response. A successful `hello` enables session operations.
3. `send` returns its acknowledgement before `turn_start`. Stream events keep
   loop order. `turn_end` follows that turn's tool events, then `agent_end`.
4. `approval_request` precedes `approval_end`. The tool does not execute until
   an allow decision exists. Delegated approval keys are opaque and map to the
   full `(child_instance_id, request_id)` key.
5. `tool_start`, `tool_output`, and `tool_end` identify one tool call.
   `status.state` is `tool` during tool execution or approval waits,
   `running` during model streaming, and `idle` after the active task ends.
6. `retry` can occur between provider stream attempts. Clients must not assume
   one provider attempt per turn.
7. The server awaits `drain()` after each frame. A slow client delays later
   notifications. A disconnected client drops writes and cancels its turn.
8. Disconnect marks unresolved approvals as `abort`. They are not pending after
   reconnect, and an approve request cannot run an aborted tool.
9. SIGINT, SIGTERM, and orderly close use one path. The server closes clients
   and turns, closes the listener, and removes the Unix socket.
10. Match responses by `id`. Continue processing notifications until the
    matching response arrives.
11. A session replacement closes the old loop before publishing the new
    session. Events from old background children keep the old session id.
    Background events do not change the foreground `status.state`.

## size and pagination rules

Every inbound and outbound frame is at most `MAX_FRAME_BYTES` bytes, including
the newline. An oversized inbound line gets a structured `-32600` error. The
server discards that line and keeps a handshaken connection usable. A valid
request can receive `-32007` when its result does not fit. That response keeps
the request id when it is within the request-id limits.

`list_sessions` returns the largest fitting prefix. A truncated result has
`truncated: true`, `sessions`, and zero-based `next_offset`. Version `1.0`
does not expose an offset request, so clients should treat this marker as a
safe display warning. Other oversized payloads become a bounded `-32007`
response or error event. A malformed frame returns a structured error. Once
the handshake succeeds, the server continues reading after malformed JSON,
wrong types, oversized string ids, and oversized numeric ids.

## conformance

The repository tests construct every request and event family. They check the
JSON-RPC envelope, required discriminators, size limit, ordering, pagination,
resume failure, second-client refusal, and disconnect cleanup. Changes to
event names, states, or field optionality must update this section and tests.


## Session extensions (1.1)

Every request below requires `session_id` equal to the active session ID.
Unknown or inactive sessions return `-32003`. A connection negotiated at 1.0
receives `-32601` for every extension. Existing notification shapes are unchanged;
there are no new event types. Mutation requests reject running turns, outstanding
tools, approvals, and background agents with `-32004`.

- `session_tree`: returns `branches`, each with `id` (head entry ID), `label`,
  `depth` (number of divergences), and `current`. The heads come from the core
  `list_branches` seam.
- `fork_message`: takes `message_id`, a user message entry on the active branch.
  Calls `append_message_fork` and returns the updated tree. Missing messages or
  non-user messages return `-32602`. The fork retains the selected user message.
- `switch_branch`: takes `head_id`, an existing leaf. Calls `switch_to_branch`
  and returns the tree. Selecting the current leaf is a no-op. Invalid heads
  return `-32602`. Fork and switch affect conversation state, not workspace files.
- `session_history`: takes optional nonnegative `offset` (default 0). Returns
  `messages` and `next_offset` (null at the end), eight messages per page. Each
  message has `id`, `role`, `content`, and optional `tool_result`. Text and tool
  output previews are bounded to 8,000 bytes. Image content becomes
  `{"type":"attachment","name":"shot.png","size":123}`; base64 stays off this
  response. Tool-use content retains the existing `tool_call` shape.
- `model_catalog`: returns `models`, sorted names from both built-in real provider
  catalogs, and `providers`, a model-name-to-provider map (`claude` or `codex`).
  Only a server whose effective launch provider is `fake` returns `faster` and
  `offline` instead, without a provider map; real models cannot enter that catalog.
- `session_settings`: returns `model` and `approval_mode`.
- `set_settings`: takes `model` from that catalog and `approval_mode` (`ask`,
  `allow`, or `deny`). Both persist atomically in session metadata and apply to
  future completions. Explicit approval tool rules retain precedence. Invalid
  choices return `-32602`. Cross-provider choices rebuild the completion and
  compaction backends while retaining the session, history, usage, and approval
  rules. Pinned budgets stay fixed; unpinned budgets track the target model.
  Running turns reject changes with `-32004`. Missing target-provider logins
  return `-32000` with a login message; failed swaps retain the previous settings.
  The response contains the applied settings.
- `send_images`: takes `text` (possibly empty) and `images`, a list of one to four
  objects with `name`, `mime_type`, and base64 `data`. Supported types are
  `image/png`, `image/jpeg`, `image/gif`, and `image/webp`. The combined decoded
  limit is 524,288 bytes. Names are plain filenames of at most 128 characters.
  Invalid base64, signatures, names, counts, types, or sizes return `-32602`.
  All images validate before any files are written. The server persists files at
  `sessions/<id>/attachments/<unique-id>/<name>` and builds normal `ImageContent`
  blocks with path and size for `AgentLoop.run_turn(user_message=...)`.
  The response matches `send`: `accepted` and `session_id`.

The generic 1 MiB frame bound applies to all requests and responses. Session
metadata adds nullable `approval_mode`; absent or null uses configured defaults.

### Session preview metadata

`list_sessions` includes an optional `first_message_preview` string on each
session. It contains a bounded, control-stripped, single-line preview of the
first user message. It does not change the persisted session name. Older clients
ignore it; newer clients fall back when an older server omits it. Provider-mode
filtering and response-size limits apply before the response is sent.
