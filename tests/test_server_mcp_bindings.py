"""real serve-protocol coverage for transient session MCP bindings."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.support.fake_backend import (
    FakeBackend,
    PerActorFakeBackend,
    ScriptedTurn,
)
from tests.support.server_client import event as _event
from tests.test_server import TIMEOUT, _connect, _request
from zeta.mcp import MCPServerConfig, StreamableHTTPMCPClient
from zeta.mcp.client import MCPHTTPError
from zeta.protocol.types import TextContent, ToolCall
from zeta.server import ZetaServer

SECRET = "page-secret-never-persisted"


class LocalMCP:
    def __init__(
        self, *, error_with_header: bool = False, reject_missing_auth: bool = False
    ) -> None:
        self.headers: list[str | None] = []
        self.methods: list[str] = []
        self.error_with_header = error_with_header
        self.reject_missing_auth = reject_missing_auth
        self.request_received = asyncio.Event()
        self.auth_received = asyncio.Event()
        self.notification_opened = asyncio.Event()
        self.notification_closed = asyncio.Event()
        self.send_tool_change = asyncio.Event()
        self.tool_change_sent = asyncio.Event()
        self.drop_notification = asyncio.Event()
        self.detached = False
        self.request_after_detach = asyncio.Event()
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            lines = raw.decode().split("\r\n")
            http_method = lines[0].split()[0]
            headers = dict(
                line.split(": ", 1) for line in lines[1:] if ": " in line
            )
            authorization = headers.get("authorization")
            self.headers.append(authorization)
            self.request_received.set()
            if self.detached:
                self.request_after_detach.set()
            if self.reject_missing_auth and authorization is None:
                payload = b"credentials rejected: missing bearer"
                writer.write(
                    b"HTTP/1.1 401 Unauthorized\r\ncontent-length: "
                    + str(len(payload)).encode()
                    + b"\r\n\r\n"
                    + payload
                )
                await writer.drain()
                return
            if authorization is not None:
                self.auth_received.set()
            if http_method == "GET":
                self.methods.append("GET")
                writer.write(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n"
                )
                await writer.drain()
                self.notification_opened.set()
                while True:
                    client_closed = asyncio.create_task(reader.read())
                    drop = asyncio.create_task(
                        asyncio.wait_for(self.drop_notification.wait(), 10)
                    )
                    send = asyncio.create_task(
                        asyncio.wait_for(self.send_tool_change.wait(), 10)
                    )
                    try:
                        done, _ = await asyncio.wait_for(
                            asyncio.wait(
                                (client_closed, drop, send),
                                return_when=asyncio.FIRST_COMPLETED,
                            ),
                            10,
                        )
                        if send in done:
                            self.send_tool_change.clear()
                            payload = json.dumps({
                                "jsonrpc": "2.0",
                                "method": "notifications/tools/list_changed",
                            }).encode()
                            writer.write(b"data: " + payload + b"\n\n")
                            await writer.drain()
                            self.tool_change_sent.set()
                            continue
                        self.notification_closed.set()
                        return
                    finally:
                        client_closed.cancel()
                        drop.cancel()
                        send.cancel()
                        await asyncio.wait_for(
                            asyncio.gather(
                                client_closed, drop, send, return_exceptions=True
                            ),
                            10,
                        )
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length)
            request = json.loads(body)
            method = request["method"]
            self.methods.append(method)
            if self.error_with_header:
                payload = (headers.get("authorization") or "missing").encode()
                writer.write(
                    b"HTTP/1.1 500 Internal Server Error\r\n"
                    b"content-type: text/plain\r\ncontent-length: "
                    + str(len(payload)).encode()
                    + b"\r\n\r\n"
                    + payload
                )
                await writer.drain()
                return
            if method == "notifications/initialized":
                writer.write(b"HTTP/1.1 202 Accepted\r\ncontent-length: 0\r\n\r\n")
                await writer.drain()
                return
            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
            elif method == "tools/list":
                result = {"tools": [{"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": "ok"}], "isError": False}
            else:
                result = {}
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
            payload = json.dumps(response).encode()
            session_header = (
                b"mcp-session-id: local-session\r\n" if method == "initialize" else b""
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                + session_header
                + b"content-length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def close(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


class LocalResponseStream(httpx.AsyncByteStream):
    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.reader = reader
        self.writer = writer

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while chunk := await self.reader.read(65536):
            yield chunk

    async def aclose(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()


class LocalTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        reader, writer = await asyncio.open_connection(request.url.host, request.url.port)
        body = request.content
        wire_headers = "".join(
            f"{name}: {value}\r\n" for name, value in request.headers.items()
        )
        writer.write(
            (
                f"{request.method} {request.url.raw_path.decode()} HTTP/1.1\r\n"
                f"{wire_headers}content-length: {len(body)}\r\n\r\n"
            ).encode()
            + body
        )
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        status = int(lines[0].split()[1])
        response_headers = {
            key.decode(): value.decode()
            for key, value in (
                line.split(b": ", 1) for line in lines[1:] if b": " in line
            )
        }
        if "content-length" in response_headers:
            payload = await reader.readexactly(int(response_headers["content-length"]))
            writer.close()
            await writer.wait_closed()
            return httpx.Response(
                status, headers=response_headers, content=payload, request=request
            )
        return httpx.Response(
            status,
            headers=response_headers,
            stream=LocalResponseStream(reader, writer),
            request=request,
        )


def allow_local_http(monkeypatch: pytest.MonkeyPatch) -> None:
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(
            *args,
            transport=LocalTransport(),
            **{key: value for key, value in kwargs.items() if key != "transport"},
        ),
    )


@pytest.mark.asyncio
async def test_session_mcp_binding_mounts_and_stays_out_of_wire_and_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcp = LocalMCP()
    port = await mcp.start()

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            mcp_reader, mcp_writer = await asyncio.open_connection(request.url.host, request.url.port)
            body = request.content
            wire_headers = "".join(f"{name}: {value}\r\n" for name, value in request.headers.items())
            mcp_writer.write((f"POST {request.url.raw_path.decode()} HTTP/1.1\r\n{wire_headers}content-length: {len(body)}\r\n" + "\r\n").encode() + body)
            await mcp_writer.drain()
            head = await mcp_reader.readuntil(b"\r\n\r\n")
            length = int(dict(line.split(b": ", 1) for line in head.split(b"\r\n")[1:] if b": " in line).get(b"content-length", b"0"))
            payload = await mcp_reader.readexactly(length)
            mcp_writer.close(); await mcp_writer.wait_closed()
            return httpx.Response(200, headers={"content-type": "application/json"}, content=payload, request=request)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: original_client(*args, transport=Transport(), **{k: v for k, v in kwargs.items() if k != "transport"}))
    try:
        (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {"transport": "streamable-http", "url": f"http://127.0.0.1:{port}"}}}))
        call = ToolCall("call-1", "page__echo", {})
        backend = FakeBackend([ScriptedTurn(tool_calls=[call]), ScriptedTurn(content=[TextContent("done")])])
        server = ZetaServer(home=tmp_path, port=0, provider="codex", backend_factory=lambda *_: (backend, "offline"))
        reader, writer = await _connect(server)
        frames: list[dict[str, Any]] = []
        try:
            frames += await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_mcp_bindings"]})
            session = (await _request(reader, writer, 2, "new_session", {"provider": "codex", "mcp_bindings": [{"server": "page", "headers": {"Authorization": f"Bearer {SECRET}"}}]}))[-1]["result"]["session"]
            frames += await _request(reader, writer, 3, "send", {"text": "use echo"})
            await asyncio.wait_for(mcp.auth_received.wait(), TIMEOUT)
            assert all(value == f"Bearer {SECRET}" for value in mcp.headers if value is not None)
            assert all(SECRET not in json.dumps(frame) for frame in frames)
            assert all(SECRET not in path.read_text(errors="ignore") for path in tmp_path.rglob("*") if path.is_file())
            assert session["session_id"]
        finally:
            writer.close(); await writer.wait_closed(); await server.close()
    finally:
        await mcp.close()


@pytest.mark.asyncio
async def test_session_mcp_bindings_requires_negotiation_and_global_server(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        error = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"}))[-1]
        assert "features" not in error["result"]["capabilities"]
        error = (await _request(reader, writer, 2, "new_session", {"mcp_bindings": [{"server": "missing", "headers": {}}]}))[-1]
        assert error["error"]["code"] == -32602
    finally:
        writer.close(); await writer.wait_closed(); await server.close()

@pytest.mark.asyncio
async def test_session_mcp_binding_rejects_project_definition(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".zeta").mkdir(parents=True)
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {"transport": "streamable-http", "url": "http://127.0.0.1:8"}}}))
    (project / ".zeta" / "mcp.json").write_text(json.dumps({"servers": {"page": {"transport": "streamable-http", "url": "http://127.0.0.1:9"}}}))
    server = ZetaServer(home=tmp_path, cwd=project, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_mcp_bindings"]})
        error = (await _request(reader, writer, 2, "new_session", {"mcp_bindings": [{"server": "page", "headers": {}}]}))[-1]
        assert error["error"]["code"] == -32602
        assert not list((tmp_path / "sessions").glob("*"))
    finally:
        writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_session_mcp_binding_rejects_control_header_without_leak(tmp_path: Path) -> None:
    secret = "nul-secret-should-not-appear"
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {"transport": "streamable-http", "url": "http://127.0.0.1:9"}}}))
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_mcp_bindings"]})
        error = (await _request(reader, writer, 2, "new_session", {"mcp_bindings": [{"server": "page", "headers": {"Authorization": "Bearer " + secret + chr(0)}}]}))[-1]
        assert error["error"]["code"] == -32602
        assert secret not in json.dumps(error)
    finally:
        writer.close(); await writer.wait_closed(); await server.close()


@pytest.mark.asyncio
async def test_session_mcp_binding_disconnect_requires_resume(tmp_path: Path) -> None:
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {"transport": "streamable-http", "url": "http://127.0.0.1:9"}}}))
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2", "features": ["session_mcp_bindings"]})
        session = (await _request(reader, writer, 2, "new_session", {"mcp_bindings": [{"server": "page", "headers": {"Authorization": "Bearer x"}}]}))[-1]["result"]["session"]
        writer.close(); await writer.wait_closed()
        async def wait_for_release() -> None:
            while server.control._controller_release_task is None:
                await asyncio.wait_for(asyncio.sleep(0), 10)
            await asyncio.wait_for(
                asyncio.shield(server.control._controller_release_task), 10
            )

        await asyncio.wait_for(wait_for_release(), 10)
        assert not server.runtime.loop._mcp_bindings
        reader2, writer2 = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            await _request(reader2, writer2, 3, "hello", {"protocol_version": "1.2", "features": ["session_mcp_bindings"]})
            blocked = (await _request(reader2, writer2, 4, "send", {"text": "needs tools"}))[-1]
            assert blocked["error"]["code"] == -32003
            resumed = (await _request(reader2, writer2, 5, "resume", {"session_id": session["session_id"], "mcp_bindings": [{"server": "page", "headers": {"Authorization": "Bearer x"}}]}))[-1]
            assert "result" in resumed
        finally:
            writer2.close(); await writer2.wait_closed()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_session_mcp_binding_reconnect_keeps_authenticated_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP(reject_missing_auth=True)
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    first_call = asyncio.Event()
    second_call = asyncio.Event()
    call_count = 0

    def record_call(messages: object, tools: object) -> bytes:
        nonlocal call_count
        call_count += 1
        (first_call if call_count == 1 else second_call).set()
        return json.dumps([]).encode()

    backend = FakeBackend([
        ScriptedTurn(content=[TextContent("first")]),
        ScriptedTurn(content=[TextContent("second")]),
    ], request_serializer=record_call)
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    binding = {"server": "page", "headers": {"Authorization": "Bearer reconnect"}}
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        session = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "mcp_bindings": [binding]
        }))[-1]["result"]["session"]
        first_sent = (await _request(reader, writer, 3, "send", {"text": "first"}))[-1]
        assert "result" in first_sent, first_sent
        await asyncio.wait_for(first_call.wait(), TIMEOUT)
        await asyncio.wait_for(mcp.auth_received.wait(), TIMEOUT)
        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 4, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        resumed = (await _request(reader, writer, 5, "resume", {
            "session_id": session["session_id"], "mcp_bindings": [binding]
        }))[-1]
        assert "result" in resumed
        sent = (await _request(reader, writer, 6, "send", {"text": "second"}))[-1]
        assert "result" in sent, sent
        await asyncio.wait_for(second_call.wait(), TIMEOUT)

        assert mcp.headers
        assert all(header == "Bearer reconnect" for header in mcp.headers)
        assert len(backend.calls) == 2, backend.calls
        assert "page__echo" in {tool["name"] for tool in backend.calls[1][1]}
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_session_mcp_binding_disconnect_closes_authenticated_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP(reject_missing_auth=True)
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    model_called = asyncio.Event()

    def record_call(messages: object, tools: object) -> bytes:
        model_called.set()
        return json.dumps([]).encode()

    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("mounted")])],
        request_serializer=record_call,
    )
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    binding = {"server": "page", "headers": {"Authorization": "Bearer old"}}
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "mcp_bindings": [binding]
        })
        await _request(reader, writer, 3, "send", {"text": "mount"})
        await asyncio.wait_for(model_called.wait(), TIMEOUT)
        loop = server.runtime.loop
        assert loop is not None
        await asyncio.wait_for(mcp.notification_opened.wait(), TIMEOUT)
        mcp.send_tool_change.set()
        await asyncio.wait_for(mcp.tool_change_sent.wait(), TIMEOUT)
        assert not mcp.notification_closed.is_set()

        mcp.detached = True
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        mount = loop._mcp_mount
        assert mount is not None
        assert mount.client_for("page") is None, "credential-bearing MCP client remained mounted"
        try:
            await asyncio.wait_for(mcp.notification_closed.wait(), 10)
            stream_closed = True
        except TimeoutError:
            stream_closed = False
        assert stream_closed, "credential-bearing notification stream remained open"

        request_count = len(mcp.headers)
        mcp.send_tool_change.set()
        mcp.drop_notification.set()
        try:
            await asyncio.wait_for(mcp.request_after_detach.wait(), 10)
        except TimeoutError:
            pass
        assert not mcp.request_after_detach.is_set()
        assert len(mcp.headers) == request_count
    finally:
        if not writer.is_closing():
            writer.close()
            await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_session_mcp_binding_reconnect_uses_only_new_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP(reject_missing_auth=True)
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    first_call = asyncio.Event()
    second_call = asyncio.Event()
    call_count = 0

    def record_call(messages: object, tools: object) -> bytes:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_call.set()
        elif call_count == 2:
            second_call.set()
        return json.dumps([]).encode()

    backend = FakeBackend([
        ScriptedTurn(content=[TextContent("first")]),
        ScriptedTurn(content=[TextContent("second")]),
    ], request_serializer=record_call)
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    old_binding = {"server": "page", "headers": {"Authorization": "Bearer old"}}
    new_binding = {"server": "page", "headers": {"Authorization": "Bearer new"}}
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        session = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "mcp_bindings": [old_binding]
        }))[-1]["result"]["session"]
        await _request(reader, writer, 3, "send", {"text": "first"})
        await asyncio.wait_for(first_call.wait(), TIMEOUT)
        await asyncio.wait_for(mcp.notification_opened.wait(), TIMEOUT)
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        await asyncio.wait_for(mcp.notification_closed.wait(), TIMEOUT)
        request_boundary = len(mcp.headers)

        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 4, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        resumed = (await _request(reader, writer, 5, "resume", {
            "session_id": session["session_id"], "mcp_bindings": [new_binding]
        }))[-1]
        assert "result" in resumed
        sent = (await _request(reader, writer, 6, "send", {"text": "second"}))[-1]
        assert "result" in sent, sent
        await asyncio.wait_for(second_call.wait(), TIMEOUT)

        assert mcp.headers[request_boundary:]
        assert all(header == "Bearer new" for header in mcp.headers[request_boundary:])
        assert len(backend.calls) == 2, backend.calls
        assert "page__echo" in {tool["name"] for tool in backend.calls[1][1]}
    finally:
        if not writer.is_closing():
            writer.close()
            await writer.wait_closed()
        await server.close()
        await mcp.close()


async def _wait_for_controller_release(server: ZetaServer) -> None:
    while server.control._controller_release_task is None:
        await asyncio.wait_for(asyncio.sleep(0), 10)
    await asyncio.wait_for(
        asyncio.shield(server.control._controller_release_task), 10
    )


@pytest.mark.asyncio
async def test_bound_mcp_tools_are_not_inherited_by_child_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    parent_call = ToolCall(
        "agent-1", "agent",
        {"prompt": "inspect child tools", "description": "child", "background": False},
    )
    backend = PerActorFakeBackend(
        [
            ScriptedTurn([TextContent("mounted")]),
            ScriptedTurn(tool_calls=[parent_call]),
            ScriptedTurn([TextContent("done")]),
        ],
        {"inspect child tools": [ScriptedTurn([TextContent("child done")])]},
        parent_prompt={"warm", "delegate"},
        gated_child_prompts=(),
        parent_waits_for_children=False,
    )
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        await _request(reader, writer, 2, "new_session", {
            "provider": "codex",
            "mcp_bindings": [{"server": "page", "headers": {"Authorization": "Bearer child-secret"}}],
        })
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await asyncio.wait_for(mcp.request_received.wait(), TIMEOUT)
        await asyncio.wait_for(server.runtime.loop._ensure_mcp_servers(), TIMEOUT)
        assert server.runtime.loop._mcp_mount.statuses["page"].state == "mounted", repr(
            server.runtime.loop._mcp_mount.statuses["page"]
        )
        assert server.control._turn_task is not None
        await asyncio.wait_for(asyncio.shield(server.control._turn_task), TIMEOUT)
        await _request(reader, writer, 4, "send", {"text": "delegate"})
        async def wait_for_child() -> None:
            async with asyncio.timeout(TIMEOUT):
                while not backend.children["inspect child tools"].calls:
                    await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_child(), TIMEOUT)
        child_tools = {
            tool["name"]
            for tool in backend.children["inspect child tools"].calls[0][1]
        }
        assert "page__echo" not in child_tools
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_binding_added_by_resume_is_hidden_from_existing_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    child_prompt = "use the existing page tool"
    parent_call = ToolCall(
        "agent-1", "agent",
        {"prompt": child_prompt, "description": "child", "background": True},
    )
    backend = PerActorFakeBackend(
        [
            ScriptedTurn([TextContent("mounted")]),
            ScriptedTurn(tool_calls=[parent_call]),
            ScriptedTurn([TextContent("delegated")]),
        ],
        {child_prompt: [
            ScriptedTurn(tool_calls=[ToolCall("page-1", "page__echo", {})]),
            ScriptedTurn([TextContent("child done")]),
        ]},
        parent_prompt={"warm", "delegate"},
        parent_waits_for_children=False,
    )
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        created = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex"
        }))[-1]["result"]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await asyncio.wait_for(mcp.request_received.wait(), TIMEOUT)
        await _event(reader, "turn_end")
        await _request(reader, writer, 4, "send", {"text": "delegate"})
        await asyncio.wait_for(backend.child_started[child_prompt].wait(), TIMEOUT)

        resumed = (await _request(reader, writer, 5, "resume", {
            "session_id": created["session"]["session_id"],
            "mcp_bindings": [{
                "server": "page",
                "headers": {"Authorization": "Bearer live-child-secret"},
            }],
        }))[-1]
        assert "result" in resumed
        backend.release_children.set()

        async def wait_for_child_retry() -> None:
            async with asyncio.timeout(TIMEOUT):
                while len(backend.children[child_prompt].calls) < 2:
                    await asyncio.sleep(0)

        child_retry = asyncio.create_task(wait_for_child_retry())
        leaked_approval = asyncio.create_task(_event(reader, "approval_request"))
        try:
            done, _ = await asyncio.wait(
                (child_retry, leaked_approval),
                timeout=TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            assert child_retry in done, "live child retained the newly bound MCP tool"
            await asyncio.wait_for(asyncio.shield(child_retry), TIMEOUT)
        finally:
            for task in (child_retry, leaked_approval):
                if not task.done():
                    task.cancel()
            await asyncio.wait_for(
                asyncio.gather(child_retry, leaked_approval, return_exceptions=True),
                TIMEOUT,
            )

        first_tools = {
            tool["name"] for tool in backend.children[child_prompt].calls[0][1]
        }
        later_tools = {
            tool["name"] for tool in backend.children[child_prompt].calls[1][1]
        }
        assert "page__echo" in first_tools
        assert "page__echo" not in later_tools
        assert "tools/call" not in mcp.methods
        assert all(
            header != "Bearer live-child-secret"
            for method, header in zip(mcp.methods, mcp.headers, strict=True)
            if method == "tools/call"
        )
    finally:
        backend.release_children.set()
        writer.close()
        await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_unbound_disconnect_preserves_ordinary_mcp_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    backend = FakeBackend([ScriptedTurn([TextContent("ready")])])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "mount"})
        await asyncio.wait_for(mcp.request_received.wait(), TIMEOUT)

        async def wait_for_mount() -> object:
            async with asyncio.timeout(TIMEOUT):
                while server.runtime.loop._mcp_mount is None:
                    await asyncio.sleep(0)
            return server.runtime.loop._mcp_mount

        mount = await asyncio.wait_for(wait_for_mount(), TIMEOUT)
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        assert server.runtime.loop._mcp_mount is mount
    finally:
        if not writer.is_closing():
            writer.close()
            await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_resume_without_binding_keeps_required_guard(tmp_path: Path) -> None:
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": "http://127.0.0.1:9"
    }}}))
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        session = (await _request(reader, writer, 2, "new_session", {
            "mcp_bindings": [{"server": "page", "headers": {"Authorization": "Bearer x"}}]
        }))[-1]["result"]["session"]
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        reader2, writer2 = await asyncio.open_connection("127.0.0.1", server.port)
        try:
            await _request(reader2, writer2, 3, "hello", {"protocol_version": "1.2"})
            resumed = (await _request(reader2, writer2, 4, "resume", {
                "session_id": session["session_id"]
            }))[-1]
            assert "result" in resumed
            blocked = (await _request(reader2, writer2, 5, "send", {"text": "blocked"}))[-1]
            assert blocked.get("error", {}).get("code") == -32003
        finally:
            writer2.close()
            await writer2.wait_closed()
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_bound_http_error_redacts_header_everywhere(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_local_http(monkeypatch)
    secret = "echoed-bound-secret"
    mcp = LocalMCP(error_with_header=True)
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    caplog.set_level(logging.DEBUG)
    frames: list[dict[str, Any]] = []
    try:
        frames += await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        frames += await _request(reader, writer, 2, "new_session", {
            "provider": "codex",
            "mcp_bindings": [{"server": "page", "headers": {"Authorization": secret}}],
        })
        frames += await _request(reader, writer, 3, "send", {"text": "mount"})
        await asyncio.wait_for(mcp.auth_received.wait(), TIMEOUT)

        async def wait_for_statuses() -> object:
            async with asyncio.timeout(TIMEOUT):
                while (
                    server.runtime.loop._mcp_mount is None
                    or server.runtime.loop._mcp_mount.statuses["page"].state == "connecting"
                ):
                    await asyncio.sleep(0)
            return server.runtime.loop._mcp_mount.statuses

        statuses = await asyncio.wait_for(wait_for_statuses(), TIMEOUT)
        exposed = json.dumps(frames) + caplog.text + repr(statuses)
        assert secret not in exposed
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
        await mcp.close()

    monkeypatch.undo()

    async def fail_with_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed with {secret}", request=request)

    config = MCPServerConfig(
        "page",
        "streamable-http",
        url="https://mcp.test",
        headers={"Authorization": secret},
    )
    client = StreamableHTTPMCPClient(
        config,
        client=httpx.AsyncClient(transport=httpx.MockTransport(fail_with_request)),
    )
    try:
        with pytest.raises(MCPHTTPError) as caught:
            await client.connect()
        assert secret not in repr(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_reconnect_rebinds_mcp_during_pending_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    mcp_config = {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }
    (tmp_path / "mcp.json").write_text(json.dumps({
        "servers": {"page": mcp_config, "other": mcp_config}
    }))
    backend = FakeBackend([
        ScriptedTurn([TextContent("mounted")]),
        ScriptedTurn(tool_calls=[ToolCall("page-call", "page__echo", {})]),
        ScriptedTurn([TextContent("done")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    features = ["approval_reconnect", "new_session_settings", "session_mcp_bindings"]
    old_binding = {"server": "page", "headers": {"Authorization": "Bearer old-secret"}}
    new_binding = {"server": "page", "headers": {"Authorization": "Bearer new-secret"}}
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        created = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
            "approval_mode": "ask", "mcp_bindings": [old_binding]
        }))[-1]["result"]
        session_id = created["session"]["session_id"]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "turn_end")
        await _request(reader, writer, 4, "send", {"text": "use page"})
        approval = await _event(reader, "approval_request")
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)

        reader2, writer2 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        try:
            hello = await _request(reader2, writer2, 5, "hello", {
                "protocol_version": "1.2",
                "features": features,
                "session_id": session_id,
                "mcp_bindings": [new_binding],
            })
            assert "error" not in hello[-1]
            allowed = await _request(reader2, writer2, 6, "approve", {
                "request_id": approval["request_id"]
            })
            assert "error" not in allowed[-1]
            async with asyncio.timeout(TIMEOUT):
                while mcp.methods.count("tools/call") != 1:
                    await asyncio.sleep(0)
            assert server.control._turn_task is not None
            await asyncio.wait_for(
                asyncio.shield(server.control._turn_task), TIMEOUT
            )
            await _event(reader2, "turn_end")
            assert mcp.methods.count("tools/call") == 1
            call_index = mcp.methods.index("tools/call")
            assert mcp.headers[call_index] == "Bearer new-secret"
            assert "Bearer old-secret" not in mcp.headers[call_index:]
        finally:
            writer2.close()
            await writer2.wait_closed()
    finally:
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_session_mcp_binding_rejects_invalid_and_reserved_header_names(
    tmp_path: Path,
) -> None:
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": "http://127.0.0.1:9"
    }}}))
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        names = (
            "bad:name", "Host", "Content-Length", "Transfer-Encoding", "Connection",
            "Upgrade", "TE", "Trailer", "Keep-Alive", "Proxy-Authorization",
            "Forwarded", "X-Forwarded-For", "X-Forwarded-Host",
            "X-Forwarded-Proto", "X-Real-IP", "Via",
        )
        errors = []
        for request_id, name in enumerate(names, 2):
            errors.append((await _request(reader, writer, request_id, "new_session", {
                "mcp_bindings": [{"server": "page", "headers": {name: "secret"}}]
            }))[-1])
        assert all(
            error.get("error", {}).get("code") == -32602 for error in errors
        )
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
