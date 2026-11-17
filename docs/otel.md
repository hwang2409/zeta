# OpenTelemetry traces

Zeta can export traces of its own work (agents, turns, model calls, tools,
compaction, served requests, and MCP calls) to a local OpenTelemetry collector
over OTLP/HTTP. It is off by default. Traces contain metadata only: no prompt
or tool text is put in spans.

## enable it

1. Install the optional extra. It adds `opentelemetry-sdk` and
   `opentelemetry-exporter-otlp-proto-http`, both pinned to `1.45.2`. Zeta
   without the extra never imports OpenTelemetry.

   ```console
   uv tool install --editable '.[otel]'   # or, in a checkout: uv sync --extra otel
   ```

2. Set the endpoint in `~/.zeta/settings.toml` (or `$ZETA_HOME/settings.toml`):

   ```toml
   [otel]
   endpoint = "http://127.0.0.1:4318/v1/traces"
   ```

3. Restart zeta. Telemetry is read once at startup; the serve
   `set_settings` request does not change it.

Rules for `[otel]`:

- `endpoint` is the only key. Without it, telemetry is off: no OpenTelemetry
  import, thread, queue, or socket.
- It is global only. A project `.zeta/settings.toml` cannot set it.
- The URL must be `http` or `https` with a literal loopback IP host
  (`127.0.0.0/8` or `[::1]`), and no user, password, query, or fragment.
  `localhost` is refused. Any other value stops startup with
  `otel.endpoint must be an http(s) URL with a literal loopback host`.
- If the extra is not installed, startup stops with
  `OpenTelemetry support is not installed; install zeta[otel]`.
- Zeta sets the exporter endpoint, headers, timeout, resource, sampler, and
  limits itself. While `[otel]` is set, startup stops if any
  `OTEL_EXPORTER_OTLP*` or `OTEL_PYTHON_EXPORTER_OTLP*` variable is set, or if
  `OTEL_SDK_DISABLED=true`. Other `OTEL_*` variables (service name, resource
  attributes, sampler, limits, propagators) are ignored. Proxy variables are
  ignored and redirects are not followed, so spans go only to the configured
  address.

## what is exported

Spans carry operation names, fixed codes, safe identifiers, booleans, counts,
sizes, timings, and token usage. Every attribute passes one filter: a string that is
not a known code or a well-formed identifier is exported as `_invalid`.

Never exported: prompts, system instructions, reasoning, model output, tool
arguments and results, serve or MCP request bodies, headers, environment
values, account data, attachments, resource URIs, and exception messages.
Errors carry only the exception class name in `error.type`. Nothing is written
to disk.

The resource is `service.name=zeta` and `service.version=<installed version>`.
Attribute names follow the OpenTelemetry GenAI conventions (development
snapshot); zeta-specific attributes use the `zeta.` prefix.

Span tree for one prompt:

```text
invoke_agent <profile>                  INTERNAL  agent name, conversation id, model, depth, background
  zeta.turn                             INTERNAL  turn number, notification, tool count
    zeta.context.compact                INTERNAL  only when compaction runs: emergency, tokens before/after
      chat <summarizer model>           CLIENT    if a model summary runs
    chat <model>                        CLIENT    provider, model, stream, token usage, finish reason
                                                  one span per logical call; retries are gen_ai.client.retry events
    execute_tool <tool>                 INTERNAL  tool name/type, call id, argument/result byte counts
                                                  approvals are zeta.approval.requested/resolved events
      invoke_agent <child profile>      INTERNAL  child agents nest under the agent tool call
```

Parallel tool calls are siblings. A served request adds a `serve <method>`
span (`SERVER`) above the work it starts. Each prompt or served request starts
a new trace; sessions are joined by `gen_ai.conversation.id`.

## zeta serve

A client that sends `features: ["otel"]` in `hello` gets two additions (see
[serve-protocol.md](serve-protocol.md#trace-context-otel-feature)):

- Any later request may carry top-level `traceparent` and `tracestate` (W3C
  Trace Context). The request's `serve <method>` span, and the turn a `send`
  starts, become children of that context. A malformed value is ignored and
  recorded as a value-free `zeta.serve.invalid_propagation` event. Baggage is
  not supported. Responses and notifications carry no trace context.
- `session_settings` returns `telemetry`:
  `{"enabled": bool, "state": "off" | "healthy" | "degraded", "reason": null | "export_unavailable"}`.
  The endpoint is never returned.

Zeta's own serve clients send no trace context.

## MCP

For `tools/call`, `prompts/get`, `resources/list`, and `resources/read`, zeta
adds the caller's `traceparent` (and `tracestate`, if any) to the request's
`params._meta`, keeping other `_meta` keys. HTTP servers also get them as
request headers. A call the model makes adds MCP attributes (method, server,
tool name, request id, protocol version) to its `execute_tool` span; a prompt
or resource call with no tool span gets its own `CLIENT` span. Server startup,
refresh, and notifications carry no context.

## status

The TUI `/status` card has one line:

- `otel: off`
- `otel: healthy (127.0.0.1:4318)`
- `otel: degraded (export unavailable)`

Export runs on a background thread, batched every 5 seconds. When a batch
fails, the state becomes `degraded`; the next successful batch makes it
`healthy` again. A failed export never changes agent results, and exporter
errors are not printed. Spans that cannot be sent are dropped (queue limit
2048). On exit, zeta shuts the exporter down off the event loop; the SDK may
wait for its last export.

## local viewer (otel-gui)

This is one copy-paste proof journey. It downloads the exact-version, digest-pinned, MIT-licensed
`otel-gui` v3.1.0 macOS arm64 release into disposable space and runs it on
literal loopback only. It uses a disposable, credential-free `ZETA_HOME`, so it
does not overwrite real settings or mutate a globally installed zeta tool.

Set `OTEL_WORKTREE` to the absolute path of the checked-out revision before
starting. The revision check below is required: do not run this journey from
another checkout.

```console
(
  set -e
  OTEL_WORKTREE=/absolute/path/to/the/zeta-checkout
  OTEL_REVISION=663bc41baebf0aeb7e70a7605e75641240ee9abd
  ZETA_HOME="$(mktemp -d "${TMPDIR:-/tmp}/zeta-otel-home.XXXXXX")"
  VIEWER_HOME="$(mktemp -d "${TMPDIR:-/tmp}/otel-gui.XXXXXX")"
  VIEWER_URL=https://github.com/metafab/otel-gui/releases/download/v3.1.0/otel-gui-macos-arm64.tar.gz
  VIEWER_SHA256=2982fd3b3c44a459dd8a8ae973c7b6a835165ddaad4b500e414272122f3a281d

  cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ -n "${VIEWER_PID:-}" ]; then
      kill "$VIEWER_PID" 2>/dev/null || true
      wait "$VIEWER_PID" 2>/dev/null || true
    fi
    rm -rf "$ZETA_HOME" "$VIEWER_HOME"
    exit "$status"
  }
  trap cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  test "$(git -C "$OTEL_WORKTREE" rev-parse HEAD)" = "$OTEL_REVISION"
  cd "$OTEL_WORKTREE"
  uv run --extra otel python -c 'import zeta; print(zeta.__file__)'

  mkdir -p "$ZETA_HOME"
  cat > "$ZETA_HOME/settings.toml" <<'EOF'
[otel]
endpoint = "http://127.0.0.1:4319/v1/traces"
EOF
  curl -fsSL "$VIEWER_URL" -o "$VIEWER_HOME/otel-gui.tar.gz"
  printf '%s  %s\n' "$VIEWER_SHA256" "$VIEWER_HOME/otel-gui.tar.gz" | shasum -a 256 -c -
  tar -xzf "$VIEWER_HOME/otel-gui.tar.gz" -C "$VIEWER_HOME"
  VIEWER_BIN=$(find "$VIEWER_HOME" -type f -name otel-gui -perm -111 | head -1)
  HOST=127.0.0.1 PORT=4319 "$VIEWER_BIN" &
  VIEWER_PID=$!
  ZETA_HOME="$ZETA_HOME" uv run --extra otel zeta \
    --provider ollama --model qwen3:4b --no-session --yolo --max-turns 8 \
    -p 'you must use the agent tool now: ask exactly one child agent to run a short check. after the child returns, use the bash tool to print exactly otel-proof. do not answer until both tool calls complete. report only success.'
  open http://127.0.0.1:4319
)
```

The first `uv run` must print a `zeta` path under `OTEL_WORKTREE`, and the
revision check must pass. The local Ollama provider does not use account
credentials. In otel-gui, open the newest `zeta` trace. The tree should show
the parent `invoke_agent` span, its `execute_tool agent` span, the nested child
agent, and the child work. The viewer and OTLP endpoint above bind only to
`127.0.0.1`; do not replace it with a wildcard or public address. Zeta keeps
working after the viewer stops; `/status` shows `degraded` after the next
batch.

If you intentionally use a real home instead, set `ZETA_HOME="$HOME/.zeta"` and
edit only the existing `[otel]` setting without replacing the file. Restore the
setting and restart zeta processes after the proof. The disposable-home journey
above is the safe default.

## joining traces from another program (achilles)

A program that imports zeta and has its own span can pass that span to zeta
serve:

```python
from zeta.telemetry import Telemetry

telemetry = Telemetry.configure(settings)  # once per process; settings.otel_endpoint set
carrier = telemetry.current_trace_context()  # {"traceparent": ..., "tracestate": ...} or {}
request = {"jsonrpc": "2.0", "id": 7, "method": "send", "params": {"text": text}, **carrier}
```

`current_trace_context()` returns plain strings for the active OpenTelemetry
context, or `{}` when telemetry is off. Send it only after `hello` negotiated
`otel`. Code that receives a carrier uses `with telemetry.use_trace_context(carrier):`;
calls in the same task inherit context without it.

## not in this version

- Content capture. These GenAI attribute names are reserved for a later opt-in
  with redaction and are never emitted: `gen_ai.input.messages`,
  `gen_ai.output.messages`, `gen_ai.system_instructions`,
  `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`.
- Export to a non-loopback address. The loopback rule lives in one function,
  `zeta.telemetry.runtime.validate_loopback_endpoint`.
- Metrics and logs. Only traces are exported.
- The compaction span has no marker count.
