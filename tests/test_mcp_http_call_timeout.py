"""A slow streamable-HTTP tool call is bounded by the call deadline, not the
short HTTP client timeout. Times are scaled: a 1.0 s client cap stands in for
the real 10 s cap."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import httpx
import pytest

import zeta.mcp.http as mcp_http
from tests.support.owned import await_or_fail
from zeta.core.abort import AbortSignal
from zeta.mcp.client import make_error_result
from zeta.mcp.config import MCPConfig, MCPServerConfig
from zeta.mcp.envelope import MCPCallLimits
from zeta.mcp.mount import MCPMount, mount_mcp_servers
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry

_REAL_ASYNC_HTTP = httpx.AsyncHTTPTransport.handle_async_request
CLIENT_CAP = 1.0
SLOW_INSIDE_DEADLINE = 1.3


class _SlowToolHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    initializes = 0
    # seconds each tool sleeps before it answers; set per test.
    delays: ClassVar[dict[str, float]] = {}
    sse = False
    release = threading.Event()

    def do_GET(self) -> None:
        self._reply(b"", status=405)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        request_id = request.get("id")
        if request_id is None:
            self._reply(b"", status=202)
            return
        method = request["method"]
        if method == "initialize":
            type(self).initializes += 1
            self._result(
                request_id,
                {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}},
            )
            return
        if method == "tools/list":
            tools = [
                {"name": name, "description": "", "inputSchema": {"type": "object"}}
                for name in type(self).delays
            ]
            self._result(request_id, {"tools": tools})
            return
        name = request["params"]["name"]
        result = {"content": [{"type": "text", "text": f"{name} done"}]}
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})
        try:
            if type(self).sse:
                # headers first, then a silent wait: the read timeout covers the gap.
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                type(self).release.wait(type(self).delays[name])
                self.wfile.write(f"data: {body}\n\n".encode())
                return
            type(self).release.wait(type(self).delays[name])
            self._reply(body.encode())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _result(self, request_id: int, result: object) -> None:
        body = {"jsonrpc": "2.0", "id": request_id, "result": result}
        self._reply(json.dumps(body).encode())

    def _reply(
        self, body: bytes, *, status: int = 200, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


class _Server:
    def __init__(self) -> None:
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), _SlowToolHandler)
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.stopped = False

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        _SlowToolHandler.release.set()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=10)


@pytest.fixture
def slow_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _REAL_ASYNC_HTTP
    )
    monkeypatch.setattr(mcp_http, "HTTP_TIMEOUT_SECONDS", CLIENT_CAP)
    _SlowToolHandler.initializes = 0
    _SlowToolHandler.delays = {}
    _SlowToolHandler.sse = False
    _SlowToolHandler.release = threading.Event()
    server = _Server()
    try:
        yield server
    finally:
        server.stop()


async def _mount(
    tmp_path: Path, url: str, deadline: float
) -> tuple[MCPMount, ToolRegistry]:
    (tmp_path / "tools").mkdir()
    registry = ToolRegistry(
        tmp_path / "tools",
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        mcp_limits=MCPCallLimits(deadline_seconds=deadline),
    )
    config = MCPServerConfig("slow", "streamable-http", url=url)
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {"slow": config})
    )
    assert mount.statuses["slow"].state == "mounted"
    return mount, registry


async def _call(mount: MCPMount, tool: str):
    return await await_or_fail(
        mount.call_tool("slow", tool, {}, AbortSignal()), what=f"call {tool}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sse", [False, True], ids=["json", "sse"])
async def test_call_slower_than_client_cap_inside_deadline_succeeds(
    slow_server: _Server, tmp_path: Path, sse: bool
) -> None:
    _SlowToolHandler.delays = {"build": SLOW_INSIDE_DEADLINE}
    _SlowToolHandler.sse = sse
    mount, registry = await _mount(tmp_path, slow_server.url, deadline=5.0)
    try:
        result = await _call(mount, "build")

        assert result.get("isError") is not True, result
        assert result["content"][0]["text"] == "build done"
        assert mount.statuses["slow"].state == "mounted"
    finally:
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_call_past_deadline_times_out_without_degrading_server(
    slow_server: _Server, tmp_path: Path
) -> None:
    _SlowToolHandler.delays = {"hang": 30.0, "fast": 0.0}
    mount, registry = await _mount(tmp_path, slow_server.url, deadline=1.5)
    try:
        result = await _call(mount, "hang")

        assert result == make_error_result(
            "MCP tool slow__hang exceeded the 1.5s deadline"
        )
        assert mount.statuses["slow"].state == "mounted"
        follow_up = await _call(mount, "fast")
        assert follow_up["content"][0]["text"] == "fast done"
        # no reconnect happened between the two calls.
        assert _SlowToolHandler.initializes == 1
    finally:
        await mount.close()
        await registry.close()


@pytest.mark.asyncio
async def test_connect_failure_after_deadline_timeout_still_degrades(
    slow_server: _Server, tmp_path: Path
) -> None:
    _SlowToolHandler.delays = {"hang": 30.0, "fast": 0.0}
    mount, registry = await _mount(tmp_path, slow_server.url, deadline=1.5)
    try:
        await _call(mount, "hang")
        assert mount.statuses["slow"].state == "mounted"

        slow_server.stop()
        result = await _call(mount, "fast")

        assert result.get("isError") is True
        assert "MCP connection failed" in result["content"][0]["text"]
        assert mount.statuses["slow"].state == "degraded"
    finally:
        await mount.close()
        await registry.close()
