"""W3C trace context on zeta serve requests, through the real codec and dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests import test_server_websocket as websocket
from tests.support.credentials import named_account_credentials
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.sockets import short_socket_path
from zeta.accounts import AccountId, AccountStore
from zeta.oauth import OAuthTokens
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


@pytest.mark.asyncio
@pytest.mark.parametrize("negotiated", [False, True])
async def test_account_list_on_empty_serve(tmp_path, monkeypatch, negotiated) -> None:
    server = _server(tmp_path, "unix", Telemetry.disabled(), monkeypatch)
    wire = Wire(server, "unix")
    await wire.open()
    try:
        flags = ["account_list_idle_v1"] if negotiated else []
        hello = json.loads(await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": flags}))
        capabilities = hello["result"]["capabilities"]
        assert capabilities["requests"].count("account_list") == 1
        assert ("account_list_idle_v1" in capabilities["features"]) is negotiated
        reply = json.loads(await wire.rpc("list", "account_list"))
        if negotiated:
            result = reply["result"]
            assert result["data"] == {"accounts": []}
            assert len(result["revision"]) == 64
            assert int(result["revision"], 16) >= 0
            observer = Wire(server, "unix")
            observer.reader, observer.writer = await await_or_fail(
                asyncio.open_unix_connection(str(server.socket_path)), what="observer connect",
            )
            try:
                hello = json.loads(await observer.rpc("hello", "hello", {
                    "protocol_version": "1.2", "features": [*flags, "serve_observers"], "role": "observer",
                }))
                assert "account_list" not in hello["result"]["capabilities"]["requests"]
                refused = json.loads(await observer.rpc("list", "account_list", {"unexpected": True}))
                assert refused["error"] == {
                    "code": -32008,
                    "message": "method is unavailable to read-only observers",
                    "data": {"code": "observer_read_only", "reason": "observer_read_only", "method": "account_list"},
                }
            finally:
                observer.writer.close()
                await await_or_fail(observer.writer.wait_closed(), what="observer close")
        else:
            assert reply["error"] == {"code": -32602, "message": "session_id must be a nonempty string"}
            stale = json.loads(await wire.rpc("list", "account_list", {"session_id": "missing"}))
            assert stale["error"] == {"code": -32003, "message": "session is not active"}
    finally:
        await close_serve(server, wire.writer)


@pytest.mark.asyncio
async def test_account_list_isolates_corrupt_accounts_and_redacts_wire_logs_and_spans(
    tmp_path, traced, monkeypatch, caplog,
) -> None:
    telemetry, memory = traced
    server = _server(tmp_path, "unix", telemetry, monkeypatch)
    wire = Wire(server, "unix")
    token, refresh = "access-canary-never-export", "refresh-canary-never-export"
    path = str(tmp_path / "credential-path-canary")
    await wire.open()
    try:
        caplog.set_level(logging.DEBUG)
        tokens = OAuthTokens(token, refresh, 9_999_999_999, workspace=path)
        for name in ("good", "broken"):
            named_account_credentials(tmp_path, "codex", name).save(tokens)
        store = AccountStore(tmp_path)
        store.credential_path(AccountId("codex", "broken")).write_text(f"invalid {token} {path}")
        legacy = tmp_path / "anthropic-oauth.json"
        legacy.write_text(json.dumps(tokens.to_dict()))
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["account_list_idle_v1"]})
        first = await wire.rpc("list", "account_list")
        rows = {row["account"]: row for row in json.loads(first)["result"]["data"]["accounts"]}
        assert rows["codex:good"] == {
            "account": "codex:good", "available": True, "current": False, "default": True,
            "detail": None, "label": "good", "usage": {"error": None, "freshness": "usage unknown", "snapshot": None},
        }
        assert rows["codex:broken"]["available"] is False
        assert set(rows) == {"codex:good", "codex:broken"}
        assert json.loads(legacy.read_text()) == tokens.to_dict()

        invalid = await wire.rpc("invalid", "account_list", {"session_id": path, "token": token})
        assert json.loads(invalid)["error"] == {
            "code": -32602, "message": "invalid params", "data": {"code": "invalid_params"},
        }
        store.registry.write_text(f"invalid {token} {path}")
        unavailable = await wire.rpc("unavailable", "account_list")
        assert json.loads(unavailable)["error"] == {
            "code": -32005,
            "message": "stored state is not available to read",
            "data": {"code": "state_unavailable"},
        }
        spans = _serve_spans(memory, "account_list")
        assert len(spans) == 3
        output = "\n".join([*(reply.decode() for reply in (first, invalid, unavailable)), caplog.text, *_texts(spans)])
        for canary in (token, refresh, path, str(store.registry)):
            assert canary not in output
            assert hashlib.sha256(canary.encode()).hexdigest() not in output
    finally:
        await close_serve(server, wire.writer)


@pytest.mark.asyncio
async def test_account_list_serializes_with_registry_writer(tmp_path, monkeypatch) -> None:
    server = _server(tmp_path, "unix", Telemetry.disabled(), monkeypatch)
    wire = Wire(server, "unix")
    await wire.open()
    writing, reading, release = (threading.Event() for _ in range(3))
    store = AccountStore(tmp_path)
    write_registry = store._write_registry
    registry_guard = AccountStore._registry_guard

    def write(home_fd, value):
        writing.set()
        assert release.wait(10), "registry writer was not released"
        write_registry(home_fd, value)

    def guard(owner, home_fd):
        if owner is not store:
            reading.set()
        return registry_guard(owner, home_fd)

    writer = reader = None
    try:
        named_account_credentials(tmp_path, "codex", "work").save(OAuthTokens("access", "refresh", 9_999_999_999))
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["account_list_idle_v1"]})
        before = json.loads(await wire.rpc("before", "account_list"))["result"]
        # test-contract: seam hold the writer while it owns the registry lock
        monkeypatch.setattr(store, "_write_registry", write)
        # test-contract: seam signal when the reader reaches that lock without timing guesses
        monkeypatch.setattr(AccountStore, "_registry_guard", guard)
        writer = asyncio.create_task(asyncio.to_thread(store.rename, AccountId("codex", "work"), "renamed"))
        assert await asyncio.to_thread(writing.wait, 10), "registry writer never acquired its lock"
        reader = asyncio.create_task(wire.rpc("after", "account_list"))
        assert await asyncio.to_thread(reading.wait, 10), "account_list did not take the registry lock"
        release.set()
        await await_or_fail(writer, what="registry rename")
        after = json.loads(await await_or_fail(reader, what="locked account_list"))["result"]
        assert after["data"]["accounts"][0]["label"] == "renamed"
        assert before["revision"] != after["revision"]
        assert json.loads(await wire.rpc("repeat", "account_list"))["result"] == after
    finally:
        release.set()
        await close_serve(server, wire.writer)
        await await_or_fail(asyncio.gather(*(task for task in (writer, reader) if task), return_exceptions=True),
                            what="account catalog test cleanup")


@pytest.mark.asyncio
@pytest.mark.parametrize("controller_negotiated", [False, True])
async def test_auth_error_names_failed_account_for_negotiated_clients(
    tmp_path, traced, caplog, controller_negotiated,
):
    from tests.test_tui import renderable_plain
    from zeta.codex import CodexAuthError
    from zeta.providers.completion import owned_completion
    from zeta.tui.render import render_event

    telemetry, memory = traced
    token = "auth-error-token-canary"
    credentials = named_account_credentials(tmp_path, "codex", "work")
    await asyncio.to_thread(credentials.save, OAuthTokens(token, "refresh", 9_999_999_999))
    await asyncio.to_thread(credentials.store.rename, credentials.account, "night-shift")

    class RejectedBackend:
        token_store = credentials

        @owned_completion
        async def complete(self, messages, tool_schemas):
            if False:
                yield
            raise CodexAuthError(f"rejected {token}")

    server = ZetaServer(
        home=tmp_path, socket_path=short_socket_path(tmp_path), provider="codex",
        backend_factory=lambda _provider, model, _home: (RejectedBackend(), model), telemetry=telemetry,
    )
    rendered = []
    publish = server.control._event

    async def capture(event, **kwargs):
        if event.type.value == "error":
            rendered.append(renderable_plain(render_event(event)))
        await publish(event, **kwargs)

    server.control._event = capture
    controller = Wire(server, "unix")
    observers = []
    caplog.set_level(logging.DEBUG)
    await controller.open()
    try:
        await controller.rpc("hello", "hello", {
            "protocol_version": "1.2",
            "features": ["account_list_idle_v1"] if controller_negotiated else [],
        })
        session = json.loads(await controller.rpc("new", "new_session"))["result"]["session"]
        for features in (["account_list_idle_v1"], []):
            observer = Wire(server, "unix")
            observer.reader, observer.writer = await await_or_fail(
                asyncio.open_unix_connection(str(server.socket_path)), what="observer connect",
            )
            observers.append(observer)
            await observer.rpc("hello", "hello", {
                "protocol_version": "1.2", "role": "observer",
                "features": ["serve_observers", "session_replay_seq", *features],
            })
            joined = json.loads(await observer.rpc("join", "session_history_and_subscribe", {
                "session_id": session["session_id"],
            }))
            assert "replay" in joined["result"]
        refused = json.loads(await observers[0].rpc("login", "login_start", {"account": None}))
        assert refused["error"] == {
            "code": -32008,
            "message": "method is unavailable to read-only observers",
            "data": {"code": "observer_read_only", "reason": "observer_read_only", "method": "login_start"},
        }
        sent = json.loads(await controller.rpc("send", "send", {
            "session_id": session["session_id"], "text": "trigger auth error",
        }))
        assert "result" in sent, sent
        await observers[0].event("error")
        await observers[1].event("error")
        await controller.event("error")
        controller_error = json.loads(controller.received[-1])["params"]
        await controller.event("agent_end")
        error = json.loads(observers[0].received[-1])["params"]
        assert error["data"] == {
            "login_provider": "codex",
            "auth_account": {"account": "codex:work", "label": "night-shift", "provider": "codex"},
        }
        assert error["error"] == {"code": "auth_error", "message": "Account sign-in required."}
        legacy = json.loads(observers[1].received[-1])["params"]
        assert legacy["data"] == {"login_provider": "codex"}
        assert legacy["error"] == {"code": "auth_error", "message": f"rejected {token}"}
        expected = error if controller_negotiated else legacy
        assert controller_error["data"] == expected["data"]
        assert controller_error["error"] == expected["error"]
        assert "account: night-shift" in rendered[0]
        assert token not in rendered[0]
        assert token.encode() not in b"".join(observers[0].received)
        assert token not in caplog.text
    finally:
        for observer in observers:
            observer.writer.close()
            await await_or_fail(observer.writer.wait_closed(), what="observer close")
        await close_serve(server, controller.writer)
    assert not any(token in text for text in _texts(memory.get_finished_spans()))


@pytest.mark.asyncio
async def test_auth_error_survives_a_broken_account_label_lookup(tmp_path, traced, monkeypatch):
    from zeta.codex import CodexAuthError
    from zeta.providers.completion import owned_completion
    from zeta.storage_format import StorageFormatError

    telemetry, _memory = traced
    credentials = named_account_credentials(tmp_path, "codex", "work")
    await asyncio.to_thread(credentials.save, OAuthTokens("token", "refresh", 9_999_999_999))

    def broken(_self, _account):
        raise StorageFormatError("home format is newer than this zeta")

    monkeypatch.setattr(AccountStore, "public_identity", broken)

    class RejectedBackend:
        token_store = credentials

        @owned_completion
        async def complete(self, messages, tool_schemas):
            if False:
                yield
            raise CodexAuthError("rejected")

    server = ZetaServer(
        home=tmp_path, socket_path=short_socket_path(tmp_path), provider="codex",
        backend_factory=lambda _provider, model, _home: (RejectedBackend(), model), telemetry=telemetry,
    )
    controller = Wire(server, "unix")
    await controller.open()
    try:
        await controller.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["account_list_idle_v1"]})
        session = json.loads(await controller.rpc("new", "new_session"))["result"]["session"]
        await controller.rpc("send", "send", {"session_id": session["session_id"], "text": "trigger auth error"})
        await controller.event("error")
        error = json.loads(controller.received[-1])["params"]
        assert error["error"]["code"] == "auth_error", error
        assert error["data"] == {"login_provider": "codex"}
        await controller.event("agent_end")
    finally:
        await close_serve(server, controller.writer)
