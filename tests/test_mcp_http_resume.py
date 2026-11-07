from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import httpx
import pytest

from zeta.mcp.config import MCPServerConfig
from zeta.mcp.http import StreamableHTTPMCPClient

_REAL_ASYNC_HTTP = httpx.AsyncHTTPTransport.handle_async_request


class _ResumableNotificationHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    scenario = "replay"
    get_headers: ClassVar[list[str | None]] = []
    second_stream_open = threading.Event()
    close_second_stream = threading.Event()

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        request_id = request.get("id")
        if request_id is None:
            self._reply(b"", status=202)
            return
        if request["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            self._json(request_id, result, session_id="session-1")
            return
        self._json(
            request_id,
            {"tools": []},
            session_id="session-2" if type(self).scenario == "new-session" else None,
        )

    def do_GET(self) -> None:
        type(self).get_headers.append(self.headers.get("last-event-id"))
        get_number = len(type(self).get_headers)
        if type(self).scenario == "replay":
            bodies = {
                1: self._event("event-1", "notifications/first"),
                2: self._event("event-1", "notifications/first")
                + self._event("event-2", "notifications/second"),
            }
            self._stream_or_stop(bodies.get(get_number))
            return
        if type(self).scenario == "invalid-ids":
            bodies = {
                1: self._event("event-1", "notifications/first"),
                2: self._event("bad\x00id", "notifications/malformed")
                + self._event("x" * 9000, "notifications/oversized"),
                3: self._event("event-1", "notifications/first")
                + self._event("event-2", "notifications/second"),
            }
            self._stream_or_stop(bodies.get(get_number))
            return
        if get_number == 1:
            self._stream_or_stop(self._event("event-1", "notifications/first"))
            return
        if get_number == 2:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(self._event("event-1", "notifications/first"))
            self.wfile.flush()
            type(self).second_stream_open.set()
            type(self).close_second_stream.wait(timeout=2)
            return
        self._stream_or_stop(
            self._event("event-2", "notifications/second") if get_number == 3 else None
        )

    def _stream_or_stop(self, body: bytes | None) -> None:
        if body is None:
            self._reply(b"", status=405)
        else:
            self._reply(body, content_type="text/event-stream")

    @staticmethod
    def _event(event_id: str, method: str) -> bytes:
        payload = json.dumps(
            {"jsonrpc": "2.0", "method": method}, separators=(",", ":")
        )
        return f"id: {event_id}\ndata: {payload}\n\n".encode()

    def _json(
        self, request_id: int, result: object, *, session_id: str | None = None
    ) -> None:
        headers = {"mcp-session-id": session_id} if session_id is not None else None
        self._reply(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode(),
            headers=headers,
        )

    def _reply(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


@pytest.fixture
def resumable_notification_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _REAL_ASYNC_HTTP
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResumableNotificationHandler)
    _ResumableNotificationHandler.get_headers = []
    _ResumableNotificationHandler.second_stream_open = threading.Event()
    _ResumableNotificationHandler.close_second_stream = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        _ResumableNotificationHandler.close_second_stream.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def _notification_methods(url: str, expected: int) -> list[str]:
    methods: list[str] = []
    complete = asyncio.Event()
    client = StreamableHTTPMCPClient(
        MCPServerConfig("resumable", "streamable-http", url=url)
    )

    def receive(method: str) -> None:
        methods.append(method)
        if len(methods) == expected:
            complete.set()

    client.set_notification_sink(receive)
    try:
        await client.connect()
        await asyncio.wait_for(complete.wait(), 2)
        return methods
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_notification_resume_deduplicates_replayed_event(
    resumable_notification_server: str,
) -> None:
    _ResumableNotificationHandler.scenario = "replay"

    methods = await _notification_methods(resumable_notification_server, 2)

    assert methods == ["notifications/first", "notifications/second"]
    assert _ResumableNotificationHandler.get_headers[:2] == [None, "event-1"]


@pytest.mark.asyncio
async def test_http_notification_resume_ignores_invalid_event_ids(
    resumable_notification_server: str,
) -> None:
    _ResumableNotificationHandler.scenario = "invalid-ids"

    methods = await _notification_methods(resumable_notification_server, 4)

    assert methods == [
        "notifications/first",
        "notifications/malformed",
        "notifications/oversized",
        "notifications/second",
    ]
    assert _ResumableNotificationHandler.get_headers[:3] == [
        None,
        "event-1",
        "event-1",
    ]


@pytest.mark.asyncio
async def test_http_notification_resume_resets_for_new_session(
    resumable_notification_server: str,
) -> None:
    _ResumableNotificationHandler.scenario = "new-session"
    methods: list[str] = []
    second = asyncio.Event()
    client = StreamableHTTPMCPClient(
        MCPServerConfig(
            "resumable", "streamable-http", url=resumable_notification_server
        )
    )

    def receive(method: str) -> None:
        methods.append(method)
        if method == "notifications/second":
            second.set()

    client.set_notification_sink(receive)
    try:
        await client.connect()
        await asyncio.to_thread(_ResumableNotificationHandler.second_stream_open.wait)
        await client.list_tools()
        _ResumableNotificationHandler.close_second_stream.set()
        await asyncio.wait_for(second.wait(), 2)
    finally:
        await client.close()

    assert methods == ["notifications/first", "notifications/second"]
    assert _ResumableNotificationHandler.get_headers[:3] == [
        None,
        "event-1",
        None,
    ]
