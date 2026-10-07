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
The frontend client gates all extensions on the returned version; it never sends extension
requests to a 1.0 server. A mismatch returns `-32002` with `requested` and `supported` fields, then closes
the connection. Clients must not send other requests before `hello`.

A 1.1 client can also send `features`, an array of optional feature names
(see [optional features](#optional-features-11)). When the negotiated version
is `1.1` and the request had `features`, the result has
`capabilities.features`: the requested names that the server supports, in the
server's order. A client uses a feature only when this echo contains its name.
Old servers ignore `features` and do not return the key, so a client that
receives no echo uses no feature. A `features` value that is not an array of
strings returns `-32602` and closes the connection. A 1.0 negotiation ignores
`features`.

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

Params: none. With the `list_sessions_paging` feature, optional `offset` and
`limit` (see [size and pagination rules](#size-and-pagination-rules)). With the
`projects` feature, optional `project_id` filters the result to that exact
project and uses a read-only metadata scan. An unknown project returns `-32602`
with `data.code: "project_not_found"`. The result contains `sessions`, an array
of `SessionMetadata` objects, including `name`, `project_role`, and
`parent_session_id` on protocol 1.1 so clients can group orchestrators and
workers. The
[normative wire schema](#normative-wire-schema) lists every field. A 1.0
connection does not receive `name` in this response. `approval_mode` is
`"ask"`, `"allow"`, `"deny"`, or `null`: the effective session default the
server will apply on the next approval, and what the frontend client reads to
decide whether to paint the auto-approve indicator. It is `null` when the
session has never had a default set, and clients must fall back to their own
configured default in that case.

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

### `list_projects` (`projects` feature)

Params: optional `offset` (default `0`) and `limit` (default `100`, maximum
`1000`). The result is `{"projects":[ProjectSummary],"next_offset":N|null}`.
`ProjectSummary` has this exact shape:

```json
{"id":"p_0123456789abcdef0123456789abcdef","name":"zeta","scope":"git","roots":["/work/zeta"],"session_count":3,"last_activity":"2026-10-07T12:00:00.000000Z"}
```

`roots` is empty when no canonical integration root is registered. The result
is ordered like `zeta project list`. `next_offset` points to the next unread
project. A frame-size bound can end a page before `limit`; that result also has
`truncated: true`.

### `project_show` (`projects` feature)

Params: required non-empty `project_id`. The result has this shape:

```json
{"project":{"id":"p_0123456789abcdef0123456789abcdef","name":"zeta","scope":"git","roots":["/work/zeta"],"session_count":3,"last_activity":"2026-10-07T12:00:00.000000Z","created_at":"2026-10-01T12:00:00.000000Z","updated_at":"2026-10-07T12:00:00.000000Z"},"memory":{"version_id":"0123456789abcdef0123456789abcdef","digest":"<64 lowercase hex characters>","files":[{"name":"brief.md","content":"# Brief\n","automatic":false,"content_truncated":false}]}}
```

`memory.files` always contains `brief.md`, `state.md`, `backlog.md`,
`changelog.md`, and `decisions.md`, in that order. `content` comes from the
authoritative version store, never from the generated `memory/*.md` mirror.
`automatic` reports the current origin of each file. `version_id` is `null`
only for legacy memory that has no version pointer. If JSON escaping would make
the five bounded files exceed one frame, the server shortens the largest
contents and sets their `content_truncated` fields to `true` instead of failing.

### `project_memory_log` (`projects` feature)

List mode params: required `project_id`, optional `offset` (default `0`) and
`limit` (default `100`, maximum `1000`). The result is
`{"versions":[MemoryVersion],"next_offset":N|null}`. Records are oldest first,
as in `/memory log`. A frame-size bound can end a page before `limit` and adds
`truncated: true`. `MemoryVersion` has this shape:

```json
{"version_id":"0123456789abcdef0123456789abcdef","timestamp":"2026-10-07T12:00:00.000000Z","kind":"update","files_changed":["state.md"],"provenance":{"session_id":"abc123","seq_start":10,"seq_end":20,"model":"gpt-5.6-luna"}}
```

`kind` can include `update`, `import`, `accept`, or `undo`. The supported
provenance fields are `session_id`, `seq_start`, `seq_end`, `model`,
`accepted_by`, and remote-sync `source` and `peer`. Oversized provenance strings
are bounded and add `provenance_truncated: true`. Undo records also have
`target_version_id`.

Version mode params: required `project_id`, `version_id`, and `file`; `file`
must be one of the five memory filenames. `offset` and `limit` are not allowed.
The result has this shape:

```json
{"version":{"version_id":"0123456789abcdef0123456789abcdef","timestamp":"2026-10-07T12:00:00.000000Z","kind":"update","files_changed":["state.md"],"provenance":{},"file":"state.md","content":"current version content","content_truncated":false,"diff":"--- state.md@parent\n+++ state.md@0123456789abcdef0123456789abcdef\n","diff_truncated":false}}
```

`content` is that version's authoritative file content. `diff` is a unified
text diff against the version's recorded parent snapshot. The server limits
the UTF-8 diff to 64 KiB and sets `diff_truncated: true` instead of failing the
request. If JSON escaping still approaches the frame limit, it shortens the
diff and then the content and marks the applicable `*_truncated` field.

### `project_inbox` (`projects` feature)

Params: required `project_id`; optional `status`, one of `new` (default),
`claimed`, or `done`; and optional `offset` and `limit` with the same defaults
and bounds as `list_projects`. The result is
`{"status":"new","messages":[...],"untrusted":false,"next_offset":null}`.
Message objects are the same validated objects returned by inbox
`action: "list"`, including each message's `origin`. The page-level `untrusted`
value is `true` if any returned message has an origin other than `local`; it is
`false` for an empty page or a page of only local messages. If one stored
message would exceed a frame, the server shortens its largest text fields and
lists their names in `truncated_fields`. Frame-size pagination can also add
`truncated: true`. This request does not create inbox storage, recover or claim
messages, mark messages done, or change sessions.

Messages with a non-local origin are untrusted cross-project content. A client
must display them as data and must not treat their titles, bodies, outcomes, or
replies as trusted instructions.

All four project requests work before and after session attachment. Unknown
projects return `-32602` with structured data
`{"code":"project_not_found","project_id":"<requested id>"}`. Invalid params,
unknown versions return `-32602`. Unsafe stored data returns `-32000`. No project
request repairs or writes stored state.

### `new_session`

Params: optional `provider` and `model`, both non-empty strings. The result has
`session`, containing the session metadata shape above. Settings provide
defaults when either field is absent.

With the `session_cwd` feature, optional `cwd` selects the session working
directory. It must be an absolute path to an existing directory; the server
stores the resolved path. Other values return `-32602`. A `cwd` without the
negotiated feature also returns `-32602`, so a client never gets a session in
the wrong directory. Without `cwd`, the session uses the server launch
directory (`zeta serve --cwd`). A per-session `cwd` gets the same treatment as
`zeta serve --cwd`: its repository supplies the restriction-only project
settings layer, context files, skills, agents, and project association. No
other trust grant occurs. Project MCP servers still need `zeta mcp trust`.

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
with the full session metadata. The session must exist. Resuming the active
session keeps its runtime and background children alive; it does not rebuild
the runtime.

If the session does not exist, `resume` returns `-32602` with structured error
data `{"code":"session_not_found","session_id":"<requested id>"}`. Clients
must use `data.code`, not the human-readable message, for stale-session
recovery.

A resumed session runs in its stored `cwd`, not in the server launch
directory. If that directory no longer exists, `resume` returns `-32602` with
the message `session working directory no longer exists: <cwd>`, and the active
session does not change. The project settings layer for a resumed session
comes from the server launch directory, as for an explicit CLI `--resume`.

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
Only one turn can run at a time. With the `user_message_event` feature, a
`user_message` event with `mode: "send"` precedes the acknowledgement.

```json
{"jsonrpc":"2.0","id":5,"method":"send","params":{"text":"hello"}}
```

```json
{"jsonrpc":"2.0","id":5,"result":{"accepted":true,"session_id":"abc123"}}
```

### `steer`

Params: required `text`, a non-empty string. The server queues this user
message at the next safe provider boundary. A turn must be running. With the
`user_message_event` feature, a `user_message` event with `mode: "steer"`
precedes the acknowledgement.

```json
{"jsonrpc":"2.0","id":6,"method":"steer","params":{"text":"also check the tests"}}
```

```json
{"jsonrpc":"2.0","id":6,"result":{"accepted":true}}
```

### `approve` and `deny`

Params: required `request_id`, a non-empty string matching an approval request,
and optional `scope`. `scope` defaults to `"once"` (this request only).
`"always_tool"` on `approve` also adds the tool's name to the session's
`always_allow` list, so every later call to the same tool auto-approves for
the rest of the session. `"always_tool"` on `deny` is a `-32602`. `scope`
must be a string; arrays, objects, `null`, numbers, and booleans return
`-32602`. Legacy clients omitting `scope` see the same behavior as before.

Compatibility runs both ways. Old clients that never send `scope` reach a new
server as a one-time approval — the server treats a missing key as `"once"`.
A new client on an old server sends `scope: "always_tool"` and the server
ignores the extra key: the request approves once, the tool runs, and the
memory rule does not persist because the old server has no `always_allow`
list. The client sees the same accepted response shape either way. Clients
that need per-tool memory must degrade quietly on protocol `1.0` and re-ask.

Session-scoped memory is deliberate: per-command and per-directory scopes and
cross-session persistence are out of scope for this RPC. A restart discards
the extra rules.

The result contains `accepted`, `request_id`, and `decision` (`"approve"` or
`"deny"`). It also carries `scope` when the caller passed one other than
`"once"`. The decision wakes an active turn. For a resumed session, the
server executes the pending tool through the existing loop seam.

```json
{"jsonrpc":"2.0","id":7,"method":"approve","params":{"request_id":"tool-call-1"}}
```

```json
{"jsonrpc":"2.0","id":7,"result":{"accepted":true,"request_id":"tool-call-1","decision":"approve"}}
```

```json
{"jsonrpc":"2.0","id":8,"method":"approve","params":{"request_id":"tool-call-2","scope":"always_tool"}}
```

```json
{"jsonrpc":"2.0","id":8,"result":{"accepted":true,"request_id":"tool-call-2","decision":"approve","scope":"always_tool"}}
```

### `abort`

Params: none. The result contains `aborted` (`true` when a turn was canceled).
Abort cancels the active provider or tool task and persists the loop's partial
state. When `aborted` is `true`, the server emits one `turn_aborted` event
after the canceled task stops and before the `abort` response. This applies
while the model streams, while an approval waits, and while a tool runs. When
no turn runs, the result is `{"aborted": false}` and no event is emitted.

```json
{"jsonrpc":"2.0","id":8,"method":"abort","params":{}}
```

```json
{"jsonrpc":"2.0","id":8,"result":{"aborted":true}}
```

### `status`

Params: none. The result contains `session` (full metadata or `null`), `state`
(`idle`, `running`, or `tool`), `pending_approvals`, `usage`, and
`compaction_markers`. Each pending approval has `request_id`, `tool_call`, and `delegated`. Delegated approvals also have `agent_instance_id`, the child agent instance that owns the approval.

```json
{"jsonrpc":"2.0","id":9,"method":"status","params":{}}
```

```json
{"jsonrpc":"2.0","id":9,"result":{"session":null,"state":"idle","pending_approvals":[],"usage":{},"compaction_markers":0}}
```

### `ping`

Requires the `ping` feature; otherwise `-32601`. Params: none. The result is
`{"pong": true}`. The server answers `ping` with or without an active session
and while a turn runs, because turns run outside the request reader. A client
that gets no answer within its own timeout can treat the server as hung.

```json
{"jsonrpc":"2.0","id":10,"method":"ping","params":{}}
```

```json
{"jsonrpc":"2.0","id":10,"result":{"pong":true}}
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

`approval_request` carries `request_id` and `tool_call`. It also carries
`delegated: true` and `agent_instance_id` for delegated approvals. `approval_end`
carries the same `request_id` as its matching `approval_request`, the tool call,
and empty or loop-provided `data`. For delegated approvals, the request ID is an
opaque server-issued key, even when the parent and child use the same raw tool
call ID.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"approval_request","session_id":"abc123","request_id":"tool-call-1","delegated":false,"tool_call":{"id":"tool-call-1","name":"bash","arguments":{"command":"ls"}}}}
```

`usage` carries the provider usage object. `compaction_start` and
`compaction_end` carry loop `data`; the latter can include `token_count`.
`sub_agent_receipt` carries the existing agent notification `data` object.

### reconnecting sessions and background completions

A disconnected client does not stop the active served session or its
background children. When a client attaches with `hello`, `new_session`, or
`resume`, the server checks the attached session for pending durable
notifications. If notifications are pending and no turn is active, the server
synchronously reserves one parent notification turn in the `scheduled` state,
then streams it to that client. The optional wake delay occurs inside this
reservation. While the turn is `scheduled` or `running`, idle-only requests such
as `send` and `resume` return the existing `-32004` busy error; they cannot take
the reserved turn.

The wake claims one notification batch without consuming it. Receipt events
are streamed from that claim. The server acknowledges the full batch only
after the parent turn succeeds. Provider failure, cancellation, or client
disconnection releases the claim without acknowledgement, so the same batch
remains pending for the next attach. Repeated reconnects or resumes do not
schedule another parent turn for a batch that completed successfully. If a
turn is active, the server does not start a concurrent notification turn.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"usage","session_id":"abc123","usage":{"input_tokens":10,"output_tokens":4}}}
```

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"compaction_end","session_id":"abc123","data":{"turn":2,"token_count":1200}}}
```

`user_message` requires the `user_message_event` feature. The server emits it
when it accepts `send`, `steer`, or `send_images`, before the acknowledgement.
It carries `text` (the user text, bounded to 262,144 UTF-8 bytes), `mode`
(`"send"` or `"steer"`; `send_images` uses `"send"`), and `attachments`, an
array of `{name, mime_type, size}` objects (empty for text-only messages; no
image data). Observers and reconnecting clients use it to rebuild the
transcript.

```json
{"jsonrpc":"2.0","method":"event","params":{"event":"user_message","session_id":"abc123","text":"hello","mode":"send","attachments":[]}}
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
    compaction: string, compaction_pinned: boolean,
    override_audit: array[object], system_prompt: string,
    context_files: array[string], skill_catalog: array[object] or null,
    agent_catalog: array[object] or null, vim_mode: boolean,
    budget_pinned: boolean, plan_mode: boolean, name: string,
    approval_mode: string or null, project_id: string or null,
    project_role: string or null, parent_session_id: string or null,
    project_memory_offset: integer or null,
    project_memory_length: integer or null,
    project_memory_digest: string or null,
    tool_allow: array[string] or null, tool_deny: array[string]
  },
  optional: {
    tool_allow_layers: array[array[string]],
    first_message_preview: string
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
Approval = {
  required: { request_id: string, tool_call: ToolCall, delegated: boolean },
  optional: { agent_instance_id: string (delegated only) }
}
StatusPendingApproval = {
  required: { request_id: string, tool_call: ToolCall, delegated: boolean },
  optional: { agent_instance_id: string (delegated only) }
}
Attachment = { required: { name: string, mime_type: string, size: integer } }
Usage = object with provider-defined JSON values
```

`SessionMetadata` is the session serializer output; a test keeps this block
equal to it. `compaction` is the persisted compaction mode. `skill_catalog` and
`agent_catalog` are the session's snapshotted catalogs. `project_*` fields
describe the project association and the owned project-memory block in
`system_prompt`. `parent_session_id` names the parent of a child session.
`tool_allow` is `null` when no allowlist applies. `tool_allow_layers` appears
only when more than one allowlist layer applies. `first_message_preview`
appears only in `list_sessions`. A 1.0 `list_sessions` omits `name`.

`created_at` and `updated_at` are ISO-8601 strings. IDs, names, paths, and
provider values are strings. `retained_tail` and `compaction_budget` are
positive integers. The server preserves usage keys and values.

Tool result `content_blocks` uses the MCP-compatible union. A `text` block
requires `text`, `truncated`, and `full_size`. A truncated text block can also
include an absolute `spill_path` for the complete session-scoped output.
`full_size_chars` and `next_offset` describe paginated text. An `image` block
requires `data` and `mimeType`. A `resource` block requires `resource` with
`uri` and exactly one of `text` or `blob`. Optional annotations contain
`audience`, `priority`, and `lastModified`. `structured_content` is a recursive
JSON value.

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
| `assistant_reset` (feature `assistant_reset`) | `event` | `session_id`, `data: object` |
| `memory_updated` (feature `memory_updated`) | `event`, `session_id: string`, `message: string` | none |
| `usage` | `event`, `usage: Usage` | `session_id` |
| `tool_start` | `event`, `tool_call: ToolCall`, `data: object` | `session_id` |
| `tool_output` | `event`, `tool_call: ToolCall`, `output: string`, `data: object` | `session_id` |
| `tool_end` | `event`, `tool_call: ToolCall`, `tool_result: ToolResult or null`, `data: object` | `session_id` |
| `approval_request` | `event`, `request_id: string`, `tool_call: ToolCall`, `delegated: boolean` | `session_id`, `agent_instance_id: string` (delegated only) |
| `approval_end` | `event`, `request_id: string`, `tool_call: ToolCall`, `data: object` | `session_id` |
| `sub_agent_receipt` | `event`, `data: object` | `session_id` |
| `retry` | `event`, `data: object` | `session_id` |
| `error` | `event`, `error: {code: string, message: string}`, `data: object` | `session_id` |
| `user_message` (feature `user_message_event`) | `event`, `text: string`, `mode: string`, `attachments: array[Attachment]` | `session_id` |

The `retry` data can contain `retry: integer`, `delay: number`, `text: string`,
and `is_stall: boolean`. It is an informational notice that a retry is
scheduled. Unknown provider keys remain allowed inside `data`.

`assistant_reset` is emitted only when the client negotiated the
`assistant_reset` feature and a scheduled retry will actually start. It follows
every `assistant_delta` from the failed attempt and precedes every
`assistant_delta` from the next attempt. On receipt, clients must drop the
unfinished assistant message assembled from those earlier deltas. If the retry
wait is aborted, the server does not emit `assistant_reset`, so the visible
partial response remains consistent with the failed message in the store.

Negotiating `assistant_reset` enables retries after provider output has started.
A client that does not negotiate it keeps the behavior from before provider
stream retry: the server preserves the partial assistant output, reports the
provider error, and does not start another loop-level attempt. Transport retries
that happen before any provider event remain enabled because the client has no
attempt output to discard.

A server adds these approval identity fields unconditionally. They are part of
protocol 1.1's additive event and status shape; clients must ignore unknown
fields, as with the other 1.1 extensions. No feature negotiation is required.

## ordering and lifecycle rules

1. The server processes complete request lines in order. It parses inbound
   frames and serializes all responses and notifications through one bounded
   codec and one write lock.
2. The client sends `hello` first. Any rejected first request closes the client
   after its error response. A successful `hello` enables session operations.
3. `send` returns its acknowledgement before `turn_start`. Stream events keep
   loop order. `turn_end` follows that turn's tool events, then `agent_end`.
   `turn_end` ends one provider segment; it does not mean that the agent loop
   is idle. Clients must wait for `agent_end` before idle-only requests such as
   `new_session`; the server finalizes the turn before it publishes `agent_end`,
   while error and abort paths end with `error` or `turn_aborted` instead.
4. `approval_request` precedes `approval_end`. The server emits exactly one
   `approval_end` when the request stops being pending, including after a
   decision, child cancellation, turn abort, client close, or server shutdown.
   On turn abort, `approval_end` precedes `turn_aborted`. The tool does not
   execute until an allow decision exists. `approval_end.request_id` exactly
   matches the corresponding `approval_request.request_id` (and the `status`
   entry while it is pending). Delegated approval keys are opaque and map to
   the full `(child_instance_id, request_id)` key.
5. `tool_start`, `tool_output`, and `tool_end` identify one tool call.
   `status.state` is `tool` during tool execution or approval waits,
   `running` during model streaming, and `idle` after the active task ends.
6. With the `assistant_reset` feature, `retry` can occur between provider stream
   attempts. Clients must not assume one provider attempt per turn. When partial
   assistant output must be discarded, `assistant_reset` follows the `retry`
   notice after backoff and immediately precedes the next attempt's assistant
   events.
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
`truncated: true`, `sessions`, and zero-based `next_offset`. Without the
`list_sessions_paging` feature there is no offset request, so clients should
treat this marker as a safe display warning.

With the `list_sessions_paging` feature, `list_sessions` takes optional
`offset` (integer, at least 0, default 0) and `limit` (integer, at least 1,
default all remaining). The result always has `next_offset`: the zero-based
offset of the next page, or `null` when no sessions remain. `truncated: true`
appears only when the frame limit, not `limit`, cut the page. To page, send
`offset: next_offset` until `next_offset` is `null`. Other values return
`-32602`. `offset` or `limit` without the negotiated feature returns `-32602`. Other oversized payloads become a bounded `-32007`
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
A missing, empty, or non-string `session_id` returns `-32602`. Unknown or
inactive sessions return `-32003`. A connection negotiated at 1.0
receives `-32601` for every extension. Existing notification shapes are unchanged;
the only new event type is the opt-in `user_message` (see
[optional features](#optional-features-11)). Mutation requests reject running turns, outstanding
tools, approvals, and background agents with `-32004`.

Session fallback metadata is storage-only and never appears in wire responses.
For foreground provider failures, 1.1 error events use `model_access_error` when
HTTP 400/401/403/404 or an authentication/access code identifies a rejected
completion. They use `model_reverted` after restoring a pending model fallback.
The frontend client offers Open Settings for these two codes only. MCP setup and background
errors retain their original codes. Protocol 1.0 always retains the original
error code and message. Provider status and origin remain internal.

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
- `set_compaction`: takes `mode` (`evict` or `summary`) and switches the
  compaction mode of the active session. The mode persists in session metadata
  and applies from the next completion; durable compaction markers of either
  mode stay valid. `recall_history` is advertised only in `evict` mode and only
  when the session tool policy allows it. The response contains `compaction`
  (the applied mode), `previous`, and `recall_history` (whether the tool is
  advertised). Setting the current mode is a no-op. Unknown modes, and a switch
  to `summary` when the tool policy requires `recall_history`, return `-32602`.
  Busy sessions return `-32004`. The `/compaction` slash command over
  `slash_run` shows or switches the mode through the same path.
- `send_images`: takes `text` (possibly empty) and `images`, a list of one to four
  objects with `name`, `mime_type`, and base64 `data`. Supported types are
  `image/png`, `image/jpeg`, `image/gif`, and `image/webp`. The combined decoded
  limit is 524,288 bytes. Names are plain filenames of at most 128 characters.
  Invalid base64, signatures, names, counts, types, or sizes return `-32602`.
  All images validate before any files are written. The server persists files at
  `sessions/<id>/attachments/<unique-id>/<name>` and builds normal `ImageContent`
  blocks with path and size for `AgentLoop.run_turn(user_message=...)`.
  The response matches `send`: `accepted` and `session_id`.

The generic 1 MiB frame bound applies to all requests and responses.
`approval_mode` on `SessionMetadata` reflects the effective session default;
it is `null` when the session has never had a default set (a fresh session,
or one stored by a release that predates the field), and older 1.0 servers
omit the key entirely. Clients must fall back to their configured default in
both cases.

### Slash commands (ZETA-130)

Protocol 1.1 exposes the shared slash dispatcher without forking a second
implementation for the frontend client. Two RPCs cover the surface.

- `slash_list`: params `session_id`. Returns `commands` and `notices`.
  Each command entry has `name`, `description`, `kind` (`builtin`,
  `macro-prompt`, `macro-exec`, `skill`, or `mcp-prompt`), `source`
  (`builtin`, `home`, `project`, or `mcp:<server>`), `client_only`
  (`true` when the command needs a client-side surface — for example a
  picker or workspace mutation), and `unavailable` (a bounded reason
  string, or `null` when the command runs cleanly). The frontend client renders every
  entry so users see what is available, then routes runs by the flags.
- `slash_run`: params `session_id` and `text` (the raw composer value
  starting with `/`). Rejects the request when a turn is running with
  `-32004`. Returns a `kind` discriminator:
  - `output`: `text` is the composed notice to render as a quiet receipt.
  - `model_input`: `text` is the resolved prompt to send with `send`. The
    server does not enqueue it; the client sends normally so the composer
    stays authoritative.
  - `client_only`: the command exists but must run in the client (name is
    echoed back for the client's dispatch table).
  - `unknown`: no command matched the leading token.
  - `error`: the shared dispatcher reported a bounded failure `text`.

The scope floor served over `slash_run` is `/status`, `/compact`,
`/compaction`, `/model`, `/init`, `/help`, and user prompt macros (`.zeta/commands/*.md` with
`kind: prompt`) plus skills. `/model` splits by argument shape: argless
`/model` returns `client_only` so the frontend client can open Settings for the
picker surface, while `/model <name>` dispatches server-side through the
shared settings-apply path. Exec macros and every command in the
`client_only` set report themselves as client-only rather than
half-executing here.

Legacy clients receive `-32601` for both requests, matching every other
1.1 extension. A protocol-1.1 frontend client talking to a 1.0 server hides the menu
entirely: without `slash_list`, the composer keeps every `/`-prefixed
value in the composer and never posts it to the model as chat.

### Optional features (1.1)

The client requests these names in `hello.features`. The server enables only
the names it echoes in `capabilities.features`. A connection that did not
negotiate a feature sees the behavior from before the feature existed.

| feature | effect |
| --- | --- |
| `session_cwd` | `new_session` accepts `cwd` |
| `user_message_event` | `send`, `steer`, and `send_images` emit `user_message` |
| `list_sessions_paging` | `list_sessions` accepts `offset` and `limit` and always returns `next_offset` |
| `memory_updated` | automatic reconciliation emits `memory_updated` with a short `message` |
| `ping` | the `ping` request exists and appears in `capabilities.requests` |
| `assistant_reset` | enables post-stream provider retry; `assistant_reset` removes failed attempt output before replacement deltas |
| `projects` | adds `list_projects`, `project_show`, `project_memory_log`, and `project_inbox`; `list_sessions` accepts `project_id` |

Features keep the protocol version at `1.1`. A version bump would make a new
client that sends `client_version: "1.2"` negotiate `1.0` with a 1.1 server and
lose every 1.1 extension. Per-feature negotiation has no such failure mode.

### Session preview metadata

`list_sessions` includes an optional `first_message_preview` string on each
session. It contains a bounded, control-stripped, single-line preview of the
first user message. It does not change the persisted session name. Older clients
ignore it; newer clients fall back when an older server omits it. Provider-mode
filtering and response-size limits apply before the response is sent.
