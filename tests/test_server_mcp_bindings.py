"""real serve-protocol coverage for transient session MCP bindings."""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import pytest_asyncio

from tests.support.fake_backend import (
    FakeBackend,
    PerActorFakeBackend,
    ScriptedTurn,
)
from tests.support.owned import await_or_fail, start_task_stack_watchdog
from tests.support.server_client import event as _event
from tests.test_server import TIMEOUT, _connect, _request
from zeta.mcp import MCPMount, MCPServerConfig, StreamableHTTPMCPClient
from zeta.mcp.call_deadline import MCPCallDeadline
from zeta.mcp.client import MCPHTTPError
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.loop.mcp_session import MCPSession
from zeta.server import ZetaServer
from zeta.server.protocol import EventDescriptor

SECRET = "page-secret-never-persisted"
_REAL_ASYNC_HTTP_REQUEST = httpx.AsyncHTTPTransport.handle_async_request


class LocalGETStream:
    def __init__(
        self,
        owner: LocalMCP,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.owner = owner
        self.reader = reader
        self.writer = writer

    async def run(self) -> None:
        response_headers = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
        if self.owner.chunked_notifications:
            response_headers += b"transfer-encoding: chunked\r\n"
        self.writer.write(response_headers + b"\r\n")
        await self.writer.drain()
        self.owner.notification_opened.set()
        while True:
            client_closed = asyncio.create_task(self.reader.read())
            drop = asyncio.create_task(
                asyncio.wait_for(self.owner.drop_notification.wait(), TIMEOUT)
            )
            send = asyncio.create_task(
                asyncio.wait_for(self.owner.send_tool_change.wait(), TIMEOUT)
            )
            try:
                done, _ = await asyncio.wait_for(
                    asyncio.wait(
                        (client_closed, drop, send),
                        return_when=asyncio.FIRST_COMPLETED,
                    ),
                    TIMEOUT,
                )
                if send in done:
                    self.owner.send_tool_change.clear()
                    payload = json.dumps({
                        "jsonrpc": "2.0",
                        "method": "notifications/tools/list_changed",
                    }).encode()
                    if self.owner.chunked_notifications:
                        event = b"data:" + payload + b"\r\n\r\n"
                        event = f"{len(event):x}\r\n".encode() + event + b"\r\n"
                    else:
                        event = b"data: " + payload + b"\n\n"
                    self.writer.write(event)
                    await self.writer.drain()
                    self.owner.tool_change_sent.set()
                    continue
                self.owner.notification_closed.set()
                return
            finally:
                client_closed.cancel()
                drop.cancel()
                send.cancel()
                await asyncio.wait_for(
                    asyncio.gather(
                        client_closed, drop, send, return_exceptions=True
                    ),
                    TIMEOUT,
                )


class LocalMCP:
    def __init__(
        self,
        *,
        error_with_header: bool = False,
        reject_missing_auth: bool = False,
        chunked_notifications: bool = False,
    ) -> None:
        self.headers: list[str | None] = []
        self.get_headers: dict[str, str] = {}
        self.chunked_notifications = chunked_notifications
        self.include_x = False
        self.tools_list_calls = 0
        self.tools_refetched = asyncio.Event()
        self.methods: list[str] = []
        self.error_with_header = error_with_header
        self.reject_missing_auth = reject_missing_auth
        self.request_received = asyncio.Event()
        self.auth_received = asyncio.Event()
        self.tool_called = asyncio.Event()
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
            headers = {
                name.lower(): value
                for name, value in (
                    line.split(": ", 1) for line in lines[1:] if ": " in line
                )
            }
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
                self.get_headers = headers
                self.methods.append("GET")
                await LocalGETStream(self, reader, writer).run()
                return
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
                result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {"listChanged": True}},
                }
            elif method == "tools/list":
                self.tools_list_calls += 1
                names = ["echo", "x"] if self.include_x else ["echo"]
                result = {
                    "tools": [
                        {
                            "name": name,
                            "description": name,
                            "inputSchema": {"type": "object"},
                        }
                        for name in names
                    ]
                }
                if self.include_x:
                    self.tools_refetched.set()
            elif method == "tools/call":
                self.tool_called.set()
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

    async def add_tool(self) -> None:
        await asyncio.wait_for(self.notification_opened.wait(), TIMEOUT)
        self.include_x = True
        self.send_tool_change.set()
        await asyncio.wait_for(self.tool_change_sent.wait(), TIMEOUT)

    async def close(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


class ListChangedNotificationStream(httpx.AsyncByteStream):
    def __init__(self, server: ListChangedMCP) -> None:
        self.server = server

    async def __aiter__(self):
        self.server.notification_open.set()
        await asyncio.wait_for(self.server.notification_ready.wait(), TIMEOUT)
        yield (
            b'event: message\ndata: {"jsonrpc":"2.0",'
            b'"method":"notifications/tools/list_changed"}\n\n'
        )
        yield b":" + b"x" * 65_536 + b"\n\n"
        await asyncio.wait_for(self.server.notification_release.wait(), TIMEOUT)

    async def aclose(self) -> None:
        self.server.notification_release.set()


class RefetchResponseStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes, server: ListChangedMCP) -> None:
        self.content = content
        self.server = server

    async def __aiter__(self):
        yield self.content
        self.server.tools_refetched.set()
        await asyncio.wait_for(self.server.refetch_response_release.wait(), TIMEOUT)


class ListChangedMCP(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.notification_open = asyncio.Event()
        self.notification_ready = asyncio.Event()
        self.notification_release = asyncio.Event()
        self.tools_refetched = asyncio.Event()
        self.refetch_response_release = asyncio.Event()
        self.tools_list_calls = 0
        self.include_x = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {SECRET}":
            return httpx.Response(401, request=request)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ListChangedNotificationStream(self),
                request=request,
            )
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202, request=request)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "list-changed", "version": "1"},
            }
            headers = {"mcp-session-id": "test-session"}
        elif method == "tools/list":
            self.tools_list_calls += 1
            tools = [
                {
                    "name": name,
                    "description": name,
                    "inputSchema": {"type": "object"},
                }
                for name in (["echo", "x"] if self.include_x else ["echo"])
            ]
            result = {"tools": tools}
            headers = {}
        else:
            result = {}
            headers = {}
        response = {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}
        if method == "tools/list" and self.include_x:
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=RefetchResponseStream(json.dumps(response).encode(), self),
                request=request,
            )
        return httpx.Response(200, headers=headers, json=response, request=request)

    async def add_tool(self) -> None:
        await asyncio.wait_for(self.notification_open.wait(), TIMEOUT)
        self.include_x = True
        self.notification_ready.set()

    async def aclose(self) -> None:
        self.notification_release.set()


class LocalResponseStream(httpx.AsyncByteStream):
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        close_barrier: tuple[asyncio.Event, asyncio.Event, asyncio.Event] | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.close_barrier = close_barrier

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while chunk := await self.reader.read(65536):
            yield chunk

    async def aclose(self) -> None:
        if self.close_barrier is not None:
            started, release, timed_out = self.close_barrier
            started.set()
            try:
                await asyncio.wait_for(release.wait(), 10)
            except asyncio.CancelledError:
                timed_out.set()
                raise
        self.writer.close()
        await self.writer.wait_closed()


class LocalTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        close_barriers: dict[int, tuple[asyncio.Event, asyncio.Event, asyncio.Event]] | None = None,
    ) -> None:
        self.close_barriers = close_barriers or {}

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
            stream=LocalResponseStream(
                reader, writer, self.close_barriers.get(request.url.port)
            ),
            request=request,
        )


def allow_local_http(
    monkeypatch: pytest.MonkeyPatch,
    close_barriers: dict[int, tuple[asyncio.Event, asyncio.Event, asyncio.Event]] | None = None,
) -> None:
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(
            *args,
            transport=LocalTransport(close_barriers),
            **{key: value for key, value in kwargs.items() if key != "transport"},
        ),
    )


def allow_streamable_local_http(
    monkeypatch: pytest.MonkeyPatch, transport: httpx.AsyncBaseTransport
) -> None:
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(
            *args,
            transport=transport,
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


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "config-only"])
@pytest.mark.asyncio
async def test_http_list_changed_refreshes_next_turn_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: bool,
) -> None:
    mcp = ListChangedMCP()
    allow_streamable_local_http(monkeypatch, mcp)
    original_ensure = MCPSession._ensure_mcp_servers
    original_wait = getattr(MCPMount, "wait_for_tool_refreshes", None)
    turn_started = asyncio.Event()
    turn_passed_refresh = asyncio.Event()
    refresh_wait_started = asyncio.Event()
    ensure_calls = 0

    async def observe_turn_start(session: MCPSession) -> None:
        nonlocal ensure_calls
        ensure_calls += 1
        this_call = ensure_calls
        if this_call == 2:
            turn_started.set()
        await original_ensure(session)
        if this_call == 2:
            turn_passed_refresh.set()

    async def observe_refresh_wait(mount: MCPMount) -> None:
        refresh_wait_started.set()
        assert original_wait is not None
        await original_wait(mount)

    # test-contract: seam observes turn and refresh ordering through the real server
    monkeypatch.setattr(MCPSession, "_ensure_mcp_servers", observe_turn_start)
    if original_wait is not None:
        monkeypatch.setattr(MCPMount, "wait_for_tool_refreshes", observe_refresh_wait)
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "page": {
                        "transport": "streamable-http",
                        "url": "https://mcp.test",
                        **(
                            {}
                            if bound
                            else {"headers": {"Authorization": f"Bearer {SECRET}"}}
                        ),
                    }
                }
            }
        )
    )
    backend = FakeBackend(
        [
            ScriptedTurn(content=[TextContent("first")]),
            ScriptedTurn(content=[TextContent("second")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        cwd=project,
        port=0,
        provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_mcp_bindings"],
            },
        )
        new_session: dict[str, Any] = {"provider": "codex"}
        if bound:
            new_session["mcp_bindings"] = [
                {
                    "server": "page",
                    "headers": {"Authorization": f"Bearer {SECRET}"},
                }
            ]
        await _request(reader, writer, 2, "new_session", new_session)
        await _request(reader, writer, 3, "send", {"text": "first"})
        await _event(reader, "agent_end")
        first_names = {schema["name"] for schema in backend.calls[0][1]}
        assert "page__echo" in first_names

        await mcp.add_tool()
        await asyncio.wait_for(mcp.tools_refetched.wait(), TIMEOUT)
        await _request(reader, writer, 4, "send", {"text": "second"})
        await asyncio.wait_for(turn_started.wait(), TIMEOUT)
        wait_started = asyncio.create_task(
            asyncio.wait_for(refresh_wait_started.wait(), TIMEOUT)
        )
        turn_passed = asyncio.create_task(
            asyncio.wait_for(turn_passed_refresh.wait(), TIMEOUT)
        )
        try:
            done, _ = await asyncio.wait_for(
                asyncio.wait(
                    {wait_started, turn_passed},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=TIMEOUT,
                ),
                TIMEOUT,
            )
            assert wait_started in done, "turn did not wait for the pending tool refresh"
        finally:
            for task in (wait_started, turn_passed):
                task.cancel()
            await asyncio.wait_for(
                asyncio.gather(wait_started, turn_passed, return_exceptions=True),
                TIMEOUT,
            )

        assert len(backend.calls) == 1
        mcp.refetch_response_release.set()
        await _event(reader, "agent_end")
        second_names = {schema["name"] for schema in backend.calls[1][1]}
        assert "page__x" in second_names
        assert mcp.tools_list_calls == 2
    finally:
        writer.close()
        await writer.wait_closed()
        mcp.notification_release.set()
        await server.close()
        await mcp.aclose()


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "config-only"])
@pytest.mark.asyncio
async def test_socket_http_list_changed_refreshes_next_turn_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: bool,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        _REAL_ASYNC_HTTP_REQUEST,
    )
    mcp = LocalMCP(chunked_notifications=True)
    port = await mcp.start()
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "page": {
                        "transport": "streamable-http",
                        "url": f"http://127.0.0.1:{port}",
                        **(
                            {}
                            if bound
                            else {"headers": {"Authorization": f"Bearer {SECRET}"}}
                        ),
                    }
                }
            }
        )
    )
    backend = FakeBackend(
        [
            ScriptedTurn(content=[TextContent("first")]),
            ScriptedTurn(content=[TextContent("second")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        cwd=project,
        port=0,
        provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_mcp_bindings"],
            },
        )
        new_session: dict[str, Any] = {"provider": "codex"}
        if bound:
            new_session["mcp_bindings"] = [
                {
                    "server": "page",
                    "headers": {"Authorization": f"Bearer {SECRET}"},
                }
            ]
        await _request(reader, writer, 2, "new_session", new_session)
        await _request(reader, writer, 3, "send", {"text": "first"})
        await _event(reader, "agent_end")
        first_names = {schema["name"] for schema in backend.calls[0][1]}
        assert "page__echo" in first_names
        assert mcp.get_headers["mcp-session-id"] == "local-session"
        assert mcp.get_headers["accept"] == "text/event-stream"
        assert mcp.get_headers["authorization"] == f"Bearer {SECRET}"

        await mcp.add_tool()
        try:
            await asyncio.wait_for(mcp.tools_refetched.wait(), 10)
        except TimeoutError:
            pass
        assert mcp.tools_list_calls == 2, "GET notification did not trigger tools/list"

        await _request(reader, writer, 4, "send", {"text": "second"})
        await _event(reader, "agent_end")
        second_names = {schema["name"] for schema in backend.calls[1][1]}
        assert "page__x" in second_names
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest_asyncio.fixture
async def approved_mcp_task_stack_watchdog() -> AsyncIterator[None]:
    watchdog = start_task_stack_watchdog(
        TIMEOUT, what="approved MCP socket test"
    )
    try:
        yield
    finally:
        watchdog.cancel()
        await asyncio.wait_for(
            asyncio.gather(watchdog, return_exceptions=True), 10
        )


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "config-only"])
@pytest.mark.parametrize("live_added", [True, False], ids=["live-added", "initial"])
@pytest.mark.asyncio
async def test_socket_http_approved_tool_runs_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: bool,
    live_added: bool,
    approved_mcp_task_stack_watchdog: None,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        _REAL_ASYNC_HTTP_REQUEST,
    )
    mcp = LocalMCP(chunked_notifications=True)
    mcp.include_x = not live_added
    port = await mcp.start()
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "mcp.json").write_text(
        json.dumps({
            "servers": {
                "page": {
                    "transport": "streamable-http",
                    "url": f"http://127.0.0.1:{port}",
                    **(
                        {}
                        if bound
                        else {"headers": {"Authorization": f"Bearer {SECRET}"}}
                    ),
                }
            }
        })
    )
    backend = FakeBackend([
        ScriptedTurn(content=[TextContent("first")]),
        ScriptedTurn(tool_calls=[ToolCall("x-call", "page__x", {})]),
        ScriptedTurn(content=[TextContent("done")]),
    ])
    server = ZetaServer(
        home=tmp_path,
        cwd=project,
        port=0,
        provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2",
            "features": [
                "approval_reconnect",
                "new_session_settings",
                "session_mcp_bindings",
            ],
        })
        new_session: dict[str, Any] = {
            "provider": "codex",
            "model": "gpt-5.6-sol",
            "approval_mode": "ask",
        }
        if bound:
            new_session["mcp_bindings"] = [{
                "server": "page",
                "headers": {"Authorization": f"Bearer {SECRET}"},
            }]
        created = await _request(reader, writer, 2, "new_session", new_session)
        assert "error" not in created[-1]
        sent = await _request(reader, writer, 3, "send", {"text": "first"})
        assert "error" not in sent[-1]
        await _event(reader, "agent_end")

        if live_added:
            await mcp.add_tool()
            await asyncio.wait_for(mcp.tools_refetched.wait(), TIMEOUT)

        await _request(reader, writer, 4, "send", {"text": "use x"})
        approval = await _event(reader, "approval_request")
        live_turn = server.control._turn_task
        assert live_turn is not None
        server.control._turn_task = None
        approved = await _request(reader, writer, 5, "approve", {
            "request_id": approval["request_id"],
        })
        assert "error" not in approved[-1]
        recovery_turn = server.control._turn_task
        assert recovery_turn is not None
        await await_or_fail(
            asyncio.gather(live_turn, recovery_turn),
            timeout=TIMEOUT,
            what="approved MCP live and recovery turns",
        )

        assert mcp.methods.count("tools/call") == 1
        assert sum(
            message.tool_result is not None
            for messages, _tools in backend.calls
            for message in messages
        ) == 1
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_bound_http_list_changed_disconnect_does_not_drop_refresh_coroutine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        _REAL_ASYNC_HTTP_REQUEST,
    )
    mcp = LocalMCP(chunked_notifications=True)
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http",
        "url": f"http://127.0.0.1:{port}",
    }}}))
    backend = FakeBackend([ScriptedTurn(content=[TextContent("mounted")])])
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        await _request(reader, writer, 2, "new_session", {
            "provider": "codex",
            "mcp_bindings": [{
                "server": "page",
                "headers": {"Authorization": f"Bearer {SECRET}"},
            }],
        })
        await _request(reader, writer, 3, "send", {"text": "mount"})
        await _event(reader, "agent_end")
        await mcp.add_tool()
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        gc.collect()
        assert not [
            warning
            for warning in recwarn
            if warning.category is RuntimeWarning
            and "was never awaited" in str(warning.message)
        ], "disconnect dropped an unawaited MCP tool refresh coroutine"
    finally:
        if not writer.is_closing():
            writer.close()
            await writer.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.parametrize("approved_tool", ["bash", "mcp"])
@pytest.mark.asyncio
async def test_socket_approved_tool_after_turn_end_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved_tool: str
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "bash-runs"
    backend = FakeBackend([
        ScriptedTurn(content=[TextContent("ready")]),
        ScriptedTurn(tool_calls=[ToolCall(
            "approval-call",
            "bash" if approved_tool == "bash" else "page__echo",
            (
                {"command": f"printf 'run\\n' >> {shlex.quote(str(marker))}"}
                if approved_tool == "bash"
                else {}
            ),
        )]),
        ScriptedTurn(content=[TextContent("done")]),
        ScriptedTurn(tool_calls=[ToolCall("mcp-call", "page__echo", {})]),
        ScriptedTurn(content=[TextContent("mcp done")]),
    ])
    server = ZetaServer(
        home=tmp_path,
        cwd=project,
        port=0,
        provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    first_agent_end = asyncio.Event()
    release_first_agent_end = asyncio.Event()
    notify = server.control._notify

    async def hold_first_agent_end(
        event: EventDescriptor, session_id: str | None = None, **fields: object
    ) -> bool:
        if event is EventDescriptor.AGENT_END and not first_agent_end.is_set():
            first_agent_end.set()
            await asyncio.wait_for(release_first_agent_end.wait(), TIMEOUT)
        return await notify(event, session_id, **fields)

    server.control._notify = hold_first_agent_end
    features = ["approval_reconnect", "new_session_settings", "session_mcp_bindings"]
    binding = {"server": "page", "headers": {"Authorization": "Bearer mixed"}}
    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        created = await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol", "approval_mode": "ask",
            "mcp_bindings": [binding],
        })
        assert "error" not in created[-1]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "turn_end")
        await asyncio.wait_for(first_agent_end.wait(), TIMEOUT)
        loop = server.runtime.loop
        assert loop is not None
        reservation_ready = asyncio.Event()
        recovery_waiting = asyncio.Event()
        recovery_released = asyncio.Event()
        live_claimed = asyncio.Event()
        live_claims: list[str] = []
        live_claim_owners: list[object] = []
        prepare_execution = loop.store.prepare_pending_approval_execution
        claim_execution = loop.store.claim_pending_approval_execution

        def prepare_and_release_live(*args: object, **kwargs: object) -> bool:
            prepared = prepare_execution(*args, **kwargs)
            reservation_ready.set()
            return prepared

        monkeypatch.setattr(
            loop.store,
            "prepare_pending_approval_execution",
            prepare_and_release_live,
        )

        def claim_and_release_first_turn(*args: object, **kwargs: object) -> str:
            claim = claim_execution(*args, **kwargs)
            if kwargs.get("prepared") is not True:
                live_claims.append(claim)
                # test-contract: the resident record exposes the claimed task identity.
                live_claim_owners.append(
                    loop.store._pending_approval_executions[args[0]].owner
                )
                asyncio.get_running_loop().call_soon(live_claimed.set)
            return claim

        monkeypatch.setattr(
            loop.store,
            "claim_pending_approval_execution",
            claim_and_release_first_turn,
        )
        approval_gate = loop.tool_registry._approval_gate
        approval_gate_run = type(approval_gate).run

        async def hold_live_after_approval(
            self: object, tool_call: ToolCall, *args: object, **kwargs: object
        ) -> object:
            result = await approval_gate_run(self, tool_call, *args, **kwargs)
            if self is approval_gate and tool_call.id == "approval-call":
                await asyncio.wait_for(reservation_ready.wait(), TIMEOUT)
                await asyncio.wait_for(recovery_waiting.wait(), TIMEOUT)
            return result

        monkeypatch.setattr(type(approval_gate), "run", hold_live_after_approval)
        await _request(reader, writer, 4, "send", {"text": "run bash"})
        approval = await _event(reader, "approval_request")
        assert not server.control.turn_busy()
        live_turn = server.control._turn_task
        assert live_turn is not None
        live_finished = asyncio.Event()
        live_turn.add_done_callback(lambda _task: live_finished.set())
        resume_pending_tool = loop.resume_pending_tool

        async def resume_after_live_turn(*args: object, **kwargs: object) -> object:
            recovery_waiting.set()
            await asyncio.wait_for(live_finished.wait(), TIMEOUT)
            await asyncio.wait_for(recovery_released.wait(), TIMEOUT)
            return await resume_pending_tool(*args, **kwargs)

        monkeypatch.setattr(loop, "resume_pending_tool", resume_after_live_turn)
        server.control._turn_task = None
        allowed = await _request(reader, writer, 5, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        recovery_turn = server.control._turn_task
        assert recovery_turn is not None
        await asyncio.wait_for(live_claimed.wait(), TIMEOUT)
        await asyncio.wait_for(live_finished.wait(), TIMEOUT)
        recovery_released.set()
        release_first_agent_end.set()
        await asyncio.wait_for(asyncio.gather(live_turn, recovery_turn), TIMEOUT)

        if approved_tool == "bash":
            runs = marker.read_text().splitlines() if marker.exists() else []
            assert runs == ["run"]
        else:
            assert mcp.methods.count("tools/call") == 1
        assert live_claims == ["claimed"]
        assert live_claim_owners == [live_turn]
        approval_results = [
            message.tool_result
            for message in backend.calls[-1][0]
            if message.tool_result is not None
            and message.tool_result.tool_call_id == "approval-call"
        ]
        assert len(approval_results) == 1
        assert not approval_results[0].is_error

        await _request(reader, writer, 6, "send", {"text": "run mcp"})
        mcp_approval = await _event(reader, "approval_request")
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        reader2, writer2 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        await _request(reader2, writer2, 7, "hello", {
            "protocol_version": "1.2", "features": features,
            "session_id": created[-1]["result"]["session"]["session_id"],
            "mcp_bindings": [binding],
        })
        allowed = await _request(reader2, writer2, 8, "approve", {
            "request_id": mcp_approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader2, "agent_end")

        runs = marker.read_text().splitlines() if marker.exists() else []
        assert runs == (["run"] if approved_tool == "bash" else [])
        assert mcp.methods.count("tools/call") == (1 if approved_tool == "bash" else 2)
        mcp_results = [
            message.tool_result
            for message in backend.calls[-1][0]
            if message.tool_result is not None
            and message.tool_result.tool_call_id == "mcp-call"
        ]
        assert len(mcp_results) == 1
        assert not mcp_results[0].is_error
    finally:
        release_first_agent_end.set()
        for connection in (writer2, writer):
            if connection is not None and not connection.is_closing():
                connection.close()
                await connection.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.parametrize("mcp_has_x", [False, True], ids=["host-prefix", "mcp-replaced"])
@pytest.mark.asyncio
async def test_reconnect_rebind_runs_only_the_pending_approved_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mcp_has_x: bool
) -> None:
    allow_local_http(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "host-runs"
    mcp = LocalMCP()
    mcp.include_x = mcp_has_x
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    backend = FakeBackend([
        ScriptedTurn(content=[TextContent("mounted")]),
        ScriptedTurn(tool_calls=[ToolCall("host-call", "page__x", {})]),
        ScriptedTurn(content=[TextContent("host done")]),
        ScriptedTurn(tool_calls=[ToolCall("page-call", "page__echo", {})]),
        ScriptedTurn(content=[TextContent("mcp done")]),
    ])
    server = ZetaServer(
        home=tmp_path,
        cwd=project,
        port=0,
        provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    features = ["approval_reconnect", "new_session_settings", "session_mcp_bindings"]
    binding = {"server": "page", "headers": {"Authorization": "Bearer secret"}}
    reader, writer = await _connect(server)
    writer2 = None
    writer3 = None
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        created = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex",
            "model": "gpt-5.6-sol",
            "approval_mode": "ask",
            "mcp_bindings": [binding],
        }))[-1]["result"]
        session_id = created["session"]["session_id"]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "agent_end")

        async def host_tool(arguments: dict[str, object], abort_signal: object) -> str:
            marker.write_text(marker.read_text() + "run\n" if marker.exists() else "run\n")
            return "ok"

        assert server.runtime.loop is not None
        server.runtime.loop.tool_registry.register("page__x", host_tool)
        await _request(reader, writer, 4, "send", {"text": "run host tool"})
        host_approval = await _event(reader, "approval_request")
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        reader2, writer2 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        await _request(reader2, writer2, 5, "hello", {
            "protocol_version": "1.2",
            "features": features,
            "session_id": session_id,
            "mcp_bindings": [binding],
        })
        allowed = await _request(reader2, writer2, 6, "approve", {
            "request_id": host_approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader2, "agent_end")
        assert marker.read_text().splitlines() == ["run"]
        assert mcp.methods.count("tools/call") == 0

        await _request(reader2, writer2, 7, "send", {"text": "use page"})
        mcp_approval = await _event(reader2, "approval_request")
        writer2.close()
        await writer2.wait_closed()
        await _wait_for_controller_release(server)
        reader3, writer3 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        await _request(reader3, writer3, 8, "hello", {
            "protocol_version": "1.2",
            "features": features,
            "session_id": session_id,
            "mcp_bindings": [binding],
        })
        allowed = await _request(reader3, writer3, 9, "approve", {
            "request_id": mcp_approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await _event(reader3, "agent_end")

        assert marker.read_text().splitlines() == ["run"]
        assert mcp.methods.count("tools/call") == 1
    finally:
        for connection in (writer3, writer2, writer):
            if connection is not None and not connection.is_closing():
                connection.close()
                await connection.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_mcp_refresh_expired_deadline_does_not_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PendingMount:
        tool_refresh_pending = True
        waited = False

        async def wait_for_tool_refreshes(self) -> None:
            self.waited = True

    mount = PendingMount()
    session = SimpleNamespace(_mcp_mount=mount)
    with caplog.at_level(logging.WARNING):
        await MCPSession._wait_for_pending_mcp_refreshes(
            session, deadline=MCPCallDeadline(1, lambda: 1)
        )

    assert not mount.waited
    assert "MCP tool refresh was late" in caplog.text


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
async def test_session_mcp_binding_disconnect_suspends_all_servers_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = LocalMCP(reject_missing_auth=True)
    second = LocalMCP(reject_missing_auth=True)
    first_port = await first.start()
    second_port = await second.start()
    first_close_started = asyncio.Event()
    release_first_close = asyncio.Event()
    first_close_timed_out = asyncio.Event()
    allow_local_http(
        monkeypatch,
        {first_port: (first_close_started, release_first_close, first_close_timed_out)},
    )
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {
        "first": {
            "transport": "streamable-http",
            "url": f"http://127.0.0.1:{first_port}",
        },
        "second": {
            "transport": "streamable-http",
            "url": f"http://127.0.0.1:{second_port}",
        },
    }}))
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
    bindings = [
        {"server": "first", "headers": {"Authorization": "Bearer old"}},
        {"server": "second", "headers": {"Authorization": "Bearer old"}},
    ]
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": ["session_mcp_bindings"]
        })
        await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "mcp_bindings": bindings
        })
        await _request(reader, writer, 3, "send", {"text": "mount"})
        await asyncio.wait_for(model_called.wait(), TIMEOUT)
        loop = server.runtime.loop
        assert loop is not None
        await asyncio.wait_for(first.notification_opened.wait(), TIMEOUT)
        await asyncio.wait_for(second.notification_opened.wait(), TIMEOUT)

        mount = loop._mcp_mount
        assert mount is not None
        first_suspended = asyncio.Event()
        second_suspended = asyncio.Event()
        suspend_server = mount.suspend_server

        async def observe_suspend(
            config: MCPServerConfig, *, source: Path
        ) -> None:
            await suspend_server(config, source=source)
            (first_suspended if config.name == "first" else second_suspended).set()

        monkeypatch.setattr(mount, "suspend_server", observe_suspend)
        first.detached = True
        second.detached = True
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(first_close_started.wait(), TIMEOUT)
        second.drop_notification.set()
        second_request = asyncio.create_task(
            asyncio.wait_for(second.request_after_detach.wait(), 10)
        )
        second_suspend = asyncio.create_task(
            asyncio.wait_for(second_suspended.wait(), 10)
        )
        try:
            done, _ = await asyncio.wait_for(
                asyncio.wait(
                    (second_request, second_suspend),
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                TIMEOUT,
            )
            assert second_suspend in done and second_request not in done, (
                "later MCP server used the detached controller's old header "
                "while the first server close was held"
            )
        finally:
            second_request.cancel()
            second_suspend.cancel()
            await asyncio.wait_for(
                asyncio.gather(
                    second_request, second_suspend, return_exceptions=True
                ),
                TIMEOUT,
            )
        assert second.notification_closed.is_set()
        release_task = server.control._controller_release_task
        assert release_task is not None and not release_task.done()
        release_first_close.set()
        await asyncio.wait_for(first_suspended.wait(), TIMEOUT)
        await asyncio.wait_for(release_task, TIMEOUT)
        assert not first.request_after_detach.is_set()
        assert not second.request_after_detach.is_set()
    finally:
        release_first_close.set()
        if not writer.is_closing():
            writer.close()
            await writer.wait_closed()
        await server.close()
        await first.close()
        await second.close()


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
        await _event(reader, "turn_end")
        await _event(reader, "agent_end")

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


@pytest.mark.parametrize(
    "approval_before_hello", [True, False], ids=["approval-before-hello", "approval-after-hello"]
)
@pytest.mark.asyncio
async def test_reconnect_rebinds_mcp_during_pending_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approval_before_hello: bool,
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
    writer2 = None
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
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "turn_end")
        approval = None
        if approval_before_hello:
            await _request(reader, writer, 4, "send", {"text": "use page"})
            approval = await _event(reader, "approval_request")
        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)

        reader2, writer2 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        hello = await _request(reader2, writer2, 5, "hello", {
            "protocol_version": "1.2", "features": features,
            "session_id": created["session"]["session_id"],
            "mcp_bindings": [new_binding],
        })
        assert "error" not in hello[-1]
        if approval is None:
            await _request(reader2, writer2, 6, "send", {"text": "use page"})
            approval = await _event(reader2, "approval_request")
        allowed = await _request(reader2, writer2, 7, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        await asyncio.wait_for(mcp.tool_called.wait(), TIMEOUT)

        assert mcp.methods.count("tools/call") == 1
        call_index = mcp.methods.index("tools/call")
        assert mcp.headers[call_index] == "Bearer new-secret"
        assert "Bearer old-secret" not in mcp.headers[call_index:]
    finally:
        for connection in (writer2, writer):
            if connection is not None and not connection.is_closing():
                connection.close()
                await connection.wait_closed()
        await server.close()
        await mcp.close()


@pytest.mark.asyncio
async def test_reconnect_rebind_removed_tool_returns_error_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allow_local_http(monkeypatch)
    mcp = LocalMCP()
    mcp.include_x = True
    port = await mcp.start()
    (tmp_path / "mcp.json").write_text(json.dumps({"servers": {"page": {
        "transport": "streamable-http", "url": f"http://127.0.0.1:{port}"
    }}}))
    backend = FakeBackend([
        ScriptedTurn([TextContent("mounted")]),
        ScriptedTurn(tool_calls=[ToolCall("page-call", "page__x", {})]),
        ScriptedTurn([TextContent("done")]),
    ])
    server = ZetaServer(
        home=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_: (backend, "offline"),
    )
    features = ["approval_reconnect", "new_session_settings", "session_mcp_bindings"]
    binding = {"server": "page", "headers": {"Authorization": "Bearer rebound"}}
    reader, writer = await _connect(server)
    writer2 = None
    try:
        await _request(reader, writer, 1, "hello", {
            "protocol_version": "1.2", "features": features
        })
        created = (await _request(reader, writer, 2, "new_session", {
            "provider": "codex", "model": "gpt-5.6-sol",
            "approval_mode": "ask", "mcp_bindings": [binding],
        }))[-1]["result"]
        await _request(reader, writer, 3, "send", {"text": "warm"})
        await _event(reader, "agent_end")
        await _request(reader, writer, 4, "send", {"text": "use x"})
        approval = await _event(reader, "approval_request")

        writer.close()
        await writer.wait_closed()
        await _wait_for_controller_release(server)
        mcp.include_x = False
        reader2, writer2 = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
        )
        await _request(reader2, writer2, 5, "hello", {
            "protocol_version": "1.2", "features": features,
            "session_id": created["session"]["session_id"],
            "mcp_bindings": [binding],
        })
        allowed = await _request(reader2, writer2, 6, "approve", {
            "request_id": approval["request_id"]
        })
        assert "error" not in allowed[-1]
        tool_end = await _event(reader2, "tool_end")
        await _event(reader2, "agent_end")

        assert mcp.methods.count("tools/call") == 0
        assert tool_end["tool_result"]["is_error"] is True
        assert tool_end["tool_result"]["content"] == "unknown tool: page__x"
    finally:
        for connection in (writer2, writer):
            if connection is not None and not connection.is_closing():
                connection.close()
                await connection.wait_closed()
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
