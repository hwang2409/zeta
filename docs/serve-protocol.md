# zeta serve protocol

This document defines protocol version `1.0`. It is the contract for native
clients. The transport is newline-delimited UTF-8 JSON. Each line is one
JSON-RPC 2.0 object. Frames are limited to 1 MiB.

The server accepts one client. A second connection receives a JSON-RPC error
with code `-32001`, then closes. The server uses a Unix socket by default:
`$ZETA_HOME/run/serve.sock`. `zeta serve --socket PATH` selects another Unix
socket. `zeta serve --port N` selects `127.0.0.1:N`.

## handshake

The first request must be `hello`. `protocol_version` is required and must be
exactly `1.0`. The server returns its explicit version and capability lists.
A mismatch returns `-32002` with `requested` and `supported` fields, then closes
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

Every notification has method `event`. Its `params.event` names the event and
its `params.session_id` identifies the active session when one exists.

## requests

### `list_sessions`

Params: none. The result contains `sessions`, an array of session metadata.
Each metadata object has `version`, `session_id`, `created_at`, `updated_at`,
`provider`, `model`, `cwd`, `retained_tail`, `compaction_budget`,
`override_audit`, `system_prompt`, `context_files`, `vim_mode`, `budget_pinned`,
`plan_mode`, and `name`.

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
`message`, plus optional loop `data`. The server emits it for provider and
loop failures, then returns to `idle`.

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

Error objects have `code` and `message`, and may have a method-specific
`data` object. A malformed frame never terminates the server process. The
server returns its parse error and continues reading frames.
