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

## local viewer (Jaeger)

This is one copy-paste proof journey. It starts Jaeger, writes the exact global
settings, and runs a short prompt that uses both the `agent` child-agent tool
and the `bash` tool. Use a disposable `ZETA_HOME` if you do not want to change
your normal settings.

```console
docker run -d --rm --name zeta-jaeger \
  -p 127.0.0.1:4318:4318 \
  -p 127.0.0.1:16686:16686 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.23.0
mkdir -p "${ZETA_HOME:-$HOME/.zeta}"
cat > "${ZETA_HOME:-$HOME/.zeta}/settings.toml" <<'EOF'
[otel]
endpoint = "http://127.0.0.1:4318/v1/traces"
EOF
uv tool install --editable '.[otel]'
zeta --no-session --yolo --max-turns 4 -p 'use the agent tool to ask one child agent to run a short check, then use the bash tool to print "otel-proof"; report only success'
open http://127.0.0.1:16686
# when finished:
docker stop zeta-jaeger
```

In Jaeger, select service `zeta` and open the newest trace. The tree should
show the agent span, a nested child-agent span, and an `execute_tool` span for
the tool call. The JSON API in this Jaeger version is
`/api/v3/traces?query.service_name=zeta&...`. Zeta keeps working after
`docker stop zeta-jaeger`; `/status` shows `degraded` after the next batch.

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
