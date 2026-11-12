"""real serve-protocol coverage for transient session MCP bindings."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import TIMEOUT, _connect, _request
from zeta.protocol.types import TextContent, ToolCall
from zeta.server import ZetaServer

SECRET = "page-secret-never-persisted"


class LocalMCP:
    def __init__(self) -> None:
        self.headers: list[str | None] = []
        self.auth_received = asyncio.Event()
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            headers = dict(
                line.split(": ", 1)
                for line in raw.decode().split("\r\n")[1:]
                if ": " in line
            )
            length = int(headers.get("content-length", "0"))
            body = await reader.readexactly(length)
            request = json.loads(body)
            self.headers.append(headers.get("authorization"))
            if headers.get("authorization") is not None:
                self.auth_received.set()
            method = request["method"]
            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {}}
            elif method == "tools/list":
                result = {"tools": [{"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": "ok"}], "isError": False}
            else:
                result = {}
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
            payload = json.dumps(response).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def close(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()


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
