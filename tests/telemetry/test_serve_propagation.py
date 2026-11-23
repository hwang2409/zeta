"""W3C trace context on zeta serve requests, through the real codec and dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests import test_server_websocket as websocket
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer
from zeta.server.runtime import TEST_SCRIPTED_PROVIDER_ENV
from zeta.telemetry import Telemetry

pytest.importorskip("opentelemetry.sdk")

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
PARENT_ID = "b7ad6b7169203331"
TRACEPARENT = f"00-{TRACE_ID}-{PARENT_ID}-01"
PROMPT = "summarize the secret plan"
INVALID = "zeta.serve.invalid_propagation"


@pytest.fixture
def traced() -> Iterator[tuple[Telemetry, object]]:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    class DropExporter:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def export(self, spans: object) -> SpanExportResult:
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trace_exporter, "OTLPSpanExporter", DropExporter)
        telemetry = Telemetry.configure("http://127.0.0.1:4318/v1/traces")
    memory = InMemorySpanExporter()
    telemetry._implementation._provider.add_span_processor(SimpleSpanProcessor(memory))
    try:
        yield telemetry, memory
    finally:
        telemetry.shutdown()


class Wire:
    """One raw serve connection over the Unix socket or a WebSocket."""

    def __init__(self, server: ZetaServer, transport: str) -> None:
        self.server, self.transport = server, transport
        self.received: list[bytes] = []

    async def open(self) -> None:
        await await_or_fail(self.server.start(), what="serve start")
        if self.transport == "unix":
            self.reader, self.writer = await await_or_fail(
                asyncio.open_unix_connection(str(self.server.socket_path)), what="unix connect"
            )
            return
        assert self.server._websocket is not None
        self.reader, self.writer, status, _ = await websocket._connect(
            self.server, token=None, protocol_token=self.server._websocket.controller_token
        )
        assert status == 101

    async def _read(self) -> bytes:
        if self.transport == "unix":
            # test-contract: transport exact reply bytes are compared, and the websocket branch needs the raw frame decoder
            raw = await await_or_fail(self.reader.readline(), what="unix frame")
            assert raw
        else:
            opcode, raw = await await_or_fail(websocket._read_frame(self.reader), what="websocket frame")
            assert opcode == 1
        self.received.append(raw)
        return raw

    async def send(self, request_id: str, method: str, params: dict | None = None, **context: object) -> None:
        frame = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}, **context}
        payload = json.dumps(frame, separators=(",", ":")).encode()
        self.writer.write(payload + b"\n" if self.transport == "unix" else websocket._masked_frame(payload))
        await await_or_fail(self.writer.drain(), what="frame drain")

    async def rpc(self, request_id: str, method: str, params: dict | None = None, **context: object) -> bytes:
        """Send one request and return the exact bytes of its reply."""
        await self.send(request_id, method, params, **context)
        while True:
            raw = await self._read()
            if json.loads(raw).get("id") == request_id:
                return raw

    async def event(self, name: str) -> None:
        while json.loads(await self._read()).get("params", {}).get("event") != name:
            pass


def _server(tmp_path: Path, transport: str, telemetry: Telemetry, monkeypatch: pytest.MonkeyPatch) -> ZetaServer:
    monkeypatch.setenv(TEST_SCRIPTED_PROVIDER_ENV, "1")
    return ZetaServer(
        home=tmp_path, socket_path=short_socket_path(tmp_path), port=None,
        ws_port=0 if transport == "websocket" else None, provider="codex", telemetry=telemetry,
    )


def _serve_spans(memory: object, method: str) -> list:
    return [span for span in memory.get_finished_spans() if span.name == f"serve {method}"]


def _texts(spans: Sequence) -> list[str]:
    texts = []
    for span in spans:
        texts += [span.name, *(f"{key}={value}" for key, value in span.attributes.items())]
        texts += [event.name for event in span.events]
        texts += [f"{key}={value}" for event in span.events for key, value in event.attributes.items()]
    return texts


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_negotiated_context_parents_send_turn_and_read_methods(tmp_path, traced, monkeypatch, transport) -> None:
    from opentelemetry.trace import SpanKind

    telemetry, memory = traced
    server = _server(tmp_path, transport, telemetry, monkeypatch)
    wire = Wire(server, transport)
    await wire.open()
    try:
        context = {"traceparent": TRACEPARENT, "tracestate": "zeta=1"}
        hello = json.loads(await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["otel"]}, **context))
        assert "otel" in hello["result"]["capabilities"]["features"]
        created = json.loads(await wire.rpc("new", "new_session", {"provider": "codex"}))
        session_id = created["result"]["session"]["session_id"]
        await wire.rpc("send", "send", {"session_id": session_id, "text": PROMPT}, **context)
        await wire.event("turn_end")
        await wire.rpc("status", "status", {}, **context)
        await wire.rpc("settings", "session_settings", {"session_id": session_id}, **context)
    finally:
        await close_serve(server, wire.writer)

    (hello_span,) = _serve_spans(memory, "hello")
    (send_span,) = _serve_spans(memory, "send")
    (status_span,) = _serve_spans(memory, "status")
    (settings_span,) = _serve_spans(memory, "session_settings")
    (agent,) = [span for span in memory.get_finished_spans() if span.name.startswith("invoke_agent ")]
    # hello has no remote parent in v1
    assert hello_span.parent is None
    for span in (send_span, status_span, settings_span):
        assert span.kind is SpanKind.SERVER
        assert span.parent is not None, span.name
        assert (span.parent.trace_id, span.parent.span_id) == (int(TRACE_ID, 16), int(PARENT_ID, 16))
        assert span.parent.is_remote
        assert span.context.trace_state.get("zeta") == "1"
        assert not span.events
    assert dict(send_span.attributes) == {"rpc.system": "jsonrpc", "rpc.method": "send"}
    assert dict(status_span.attributes) == {"rpc.system": "jsonrpc", "rpc.method": "status"}
    # the turn task the send started inherits the request's span
    assert agent.parent is not None and agent.parent.span_id == send_span.context.span_id
    assert agent.context.trace_id == int(TRACE_ID, 16)
    # responses and notifications never echo the context
    assert not [raw for raw in wire.received if b"traceparent" in raw or TRACE_ID.encode() in raw or b"tracestate" in raw]
    assert not [text for text in _texts(memory.get_finished_spans()) if "secret" in text]


_MALFORMED = {
    "bad-traceparent": {"traceparent": "00-zz-secret-01"},
    "zero-trace-id": {"traceparent": f"00-{'0' * 32}-{PARENT_ID}-01"},
    "not-a-string": {"traceparent": 7},
    "null": {"traceparent": None},
    "bad-tracestate": {"traceparent": TRACEPARENT, "tracestate": "secret key=@@"},
    "tracestate-alone": {"tracestate": "zeta=1"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
@pytest.mark.parametrize("carrier", list(_MALFORMED.values()), ids=list(_MALFORMED))
async def test_malformed_context_is_ignored_with_one_event(tmp_path, traced, monkeypatch, caplog, transport, carrier) -> None:
    telemetry, memory = traced
    server = _server(tmp_path, transport, telemetry, monkeypatch)
    wire = Wire(server, transport)
    await wire.open()
    caplog.set_level(logging.DEBUG)
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["otel"]})
        plain = await wire.rpc("status", "status")
        malformed = await wire.rpc("status", "status", **carrier)
    finally:
        await close_serve(server, wire.writer)

    # the request is served exactly as without the context
    assert malformed == plain
    clean, ignored = _serve_spans(memory, "status")
    assert (clean.parent, clean.events) == (None, ())
    assert ignored.parent is None
    assert [(event.name, dict(event.attributes)) for event in ignored.events] == [(INVALID, {})]
    assert not [text for text in _texts([ignored]) if "secret" in text]
    # the SDK's tracestate parser logs the received member; that logger is owned and silent
    assert not [record for record in caplog.records if "secret" in record.getMessage()]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_context_is_ignored_without_negotiation(tmp_path, traced, monkeypatch, transport) -> None:
    telemetry, memory = traced
    server = _server(tmp_path, transport, telemetry, monkeypatch)
    wire = Wire(server, transport)
    await wire.open()
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["serve_observers"]})
        plain = await wire.rpc("status", "status")
        valid = await wire.rpc("status", "status", traceparent=TRACEPARENT, tracestate="zeta=1")
        malformed = await wire.rpc("status", "status", traceparent="00-zz-secret-01")
        unknown = json.loads(await wire.rpc("unknown", "secret_method", traceparent=TRACEPARENT))
    finally:
        await close_serve(server, wire.writer)

    assert valid == malformed == plain
    assert unknown["error"]["message"] == "method not found: secret_method"
    for span in _serve_spans(memory, "status"):
        assert (span.parent, span.events) == (None, ())
    (invalid_method,) = _serve_spans(memory, "_invalid")
    assert invalid_method.parent is None
    assert dict(invalid_method.attributes) == {"rpc.system": "jsonrpc", "rpc.method": "_invalid", "error.type": "ProtocolError", "zeta.error.code": "unknown"}
    assert not [text for text in _texts(memory.get_finished_spans()) if "secret" in text]


@pytest.mark.asyncio
async def test_disabled_telemetry_serves_context_fields_unchanged(tmp_path, monkeypatch) -> None:
    server = _server(tmp_path, "unix", Telemetry.disabled(), monkeypatch)
    wire = Wire(server, "unix")
    await wire.open()
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["otel"]})
        plain = await wire.rpc("status", "status")
        valid = await wire.rpc("status", "status", traceparent=TRACEPARENT)
        malformed = await wire.rpc("status", "status", traceparent=7)
    finally:
        await close_serve(server, wire.writer)
    assert valid == malformed == plain
