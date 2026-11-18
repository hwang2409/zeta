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

   A serve client with the `home_settings_v1` feature can turn the built-in
   endpoint on or off instead. That update changes only the endpoint line
   (under an existing `[otel]` header, else a top-level `otel.endpoint` line;
   it never adds or removes the header), so on then off restores the file's
   exact bytes. It publishes `settings.toml` with mode `0600` (an existing
   `0644` file becomes `0600` on its first write); a no-op update does not
   rewrite it.

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

## built-in receiver

Zeta has its own receiver, which stores traces for `zeta serve`'s
`telemetry_*` reads. To use it, set the endpoint to exactly:

```toml
[otel]
endpoint = "http://127.0.0.1:4323/v1/traces"
```

Every zeta process with this exact endpoint (the TUI, `zeta -p`, and
`zeta serve`) starts the receiver when none is running for its home. Any
other endpoint, even another loopback address or path, is an external
collector, and zeta never starts anything for it.

- The receiver runs as `zeta otel receiver --idle-exit` in its own process
  session. It outlives the zeta process that started it, because the home's
  other zeta processes export to it too. It exits by itself, and releases its
  lock, after 15 minutes with no OTLP ingest and no `/health` request. Every
  running zeta process with this endpoint checks `/health` every 10 seconds,
  so the receiver stays up while one runs, and exits about 15 minutes after
  the last one ends or turns telemetry off. Stop it sooner with `kill <pid>`;
  the next zeta process starts it again.
- When several processes start at the same time, the receiver's lock on
  `$ZETA_HOME/telemetry/receiver.lock` keeps one; the others exit. The lock is
  released when the process ends, also after a crash, so a new receiver can
  start.
- Each zeta process checks the receiver's `/health` every 10 seconds on a
  background thread. If a launch does not answer within 15 seconds, it tries
  again after 1, 2, 4, and 8 seconds. After five failed launches in a row it
  stops trying and shows `error`; run `zeta otel receiver` by hand to see why.
- If the `otel` extra is not installed, startup stops with
  `OpenTelemetry support is not installed; install zeta[otel]`.
- Each home has a random, non-secret id in `$ZETA_HOME/telemetry/home-id`,
  which `/health` also reports. With the built-in endpoint, spans carry it as
  the resource attribute `zeta.home.id`. The receiver stores only spans with
  its own home id; it drops other spans (from another home, or from an older
  zeta without the id), counts them in `/health` under
  `dropped.foreign_home`, and reports them as OTLP `rejected_spans`. The id is
  used only for this check and is not stored with the spans.
- If the receiver on port 4323 belongs to another zeta home, zeta starts
  nothing, sends no spans, and `/status` shows
  `error (port 4323 held by another zeta home)`. It checks again every 10
  seconds.
- Zeta sends spans only after a `/health` check confirmed this home's
  receiver. A flush before that (for example at shutdown) first makes one
  `/health` check with a 1 second limit; if it does not confirm this home, the
  spans are dropped, and `/status` shows the last export as an error.

`zeta otel receiver` still works by hand, for example with an external
endpoint or to see its errors. Started by hand without `--idle-exit`, it
never exits when idle; it runs until you stop it. It exits with `already running at
http://127.0.0.1:4323` when the home's receiver already runs.

## status

The TUI `/status` card has one line:

- `otel: off`
- `otel: healthy (127.0.0.1:4318)`
- `otel: degraded (export unavailable)`

With the built-in endpoint, a second line shows the receiver URL, its state
(`starting`, `ready`, `stopped`, or `error`), and this process's latest export:

- `otel receiver: http://127.0.0.1:4323/v1/traces ready, last export ok 3s ago`
- `otel receiver: http://127.0.0.1:4323/v1/traces starting, no export yet`

Serve `status` and `telemetry_status` show the same values (see
[serve-protocol.md](serve-protocol.md)). Reading them never waits on the
receiver.

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
  OWNED_ZETA_HOME="$(mktemp -d "${TMPDIR:-/tmp}/zeta-otel-home.XXXXXX")"
  OWNED_VIEWER_HOME="$(mktemp -d "${TMPDIR:-/tmp}/otel-gui.XXXXXX")"
  VIEWER_URL=https://github.com/metafab/otel-gui/releases/download/v3.1.0/otel-gui-macos-arm64.tar.gz
  VIEWER_SHA256=2982fd3b3c44a459dd8a8ae973c7b6a835165ddaad4b500e414272122f3a281d

  cleanup() {
    status=$?
    trap - EXIT INT TERM HUP
    if [ -n "${VIEWER_PID:-}" ]; then
      kill "$VIEWER_PID" 2>/dev/null || true
      wait "$VIEWER_PID" 2>/dev/null || true
    fi
    rm -rf "$OWNED_ZETA_HOME" "$OWNED_VIEWER_HOME"
    exit "$status"
  }
  trap cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  test "$(git -C "$OTEL_WORKTREE" rev-parse HEAD)" = "$OTEL_REVISION"
  cd "$OTEL_WORKTREE"
  uv run --extra otel python -c 'import zeta; print(zeta.__file__)'

  mkdir -p "$OWNED_ZETA_HOME"
  cat > "$OWNED_ZETA_HOME/settings.toml" <<'EOF'
[otel]
endpoint = "http://127.0.0.1:4319/v1/traces"
EOF
  curl -fsSL "$VIEWER_URL" -o "$OWNED_VIEWER_HOME/otel-gui.tar.gz"
  printf '%s  %s\n' "$VIEWER_SHA256" "$OWNED_VIEWER_HOME/otel-gui.tar.gz" | shasum -a 256 -c -
  tar -xzf "$OWNED_VIEWER_HOME/otel-gui.tar.gz" -C "$OWNED_VIEWER_HOME"
  VIEWER_BIN=$(find "$OWNED_VIEWER_HOME" -type f -name otel-gui -perm -111 | head -1)
  test -n "$VIEWER_BIN"
  HOST=127.0.0.1 PORT=4319 "$VIEWER_BIN" &
  VIEWER_PID=$!
  viewer_ready=false
  for _ in $(seq 1 30); do
    VIEWER_STATE=$(ps -p "$VIEWER_PID" -o state= 2>/dev/null || true)
    case "$VIEWER_STATE" in
      ''|Z*)
        wait "$VIEWER_PID"
        exit 1
        ;;
    esac
    if curl -fsS --max-time 1 http://127.0.0.1:4319/ >/dev/null; then
      viewer_ready=true
      break
    fi
    sleep 1
  done
  test "$viewer_ready" = true
  ZETA_HOME="$OWNED_ZETA_HOME" uv run --extra otel zeta \
    --provider ollama --model qwen3:4b --no-session --yolo --max-turns 8 \
    -p 'you must use the agent tool now: ask exactly one child agent to run a short check. after the child returns, use the bash tool to print exactly otel-proof. do not answer until both tool calls complete. report only success.'
  open http://127.0.0.1:4319
  printf 'inspect the newest zeta trace in otel-gui, then press enter to finish: '
  IFS= read -r _
)
```

The viewer readiness check waits up to about 60 seconds before failing. The
first `uv run` must print a `zeta` path under `OTEL_WORKTREE`, and the revision
check must pass. The local Ollama provider does not use account
credentials. In otel-gui, open the newest `zeta` trace. The tree should show
the parent `invoke_agent` span, its `execute_tool agent` span, the nested child
agent, and the child work. The viewer and OTLP endpoint above bind only to
`127.0.0.1`; do not replace it with a wildcard or public address. Zeta keeps
working after the viewer stops; `/status` shows `degraded` after the next
batch.

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
