from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from rich.console import Console

from tests.support.fake_backend import FakeBackend
from zeta.config.settings import ResolvedConfig
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.mcp.commands import render_mcp_status
from zeta.mcp.config import MCPServerConfig
from zeta.mcp.http import StreamableHTTPMCPClient, _read_json_body
from zeta.mcp.mount import MCPMount
from zeta.runtime.loop import AgentLoop
from zeta.server.slash_commands import run_command
from zeta.skills import SkillCatalog
from zeta.tools._spill import SpillStore
from zeta.tools.registry import ToolRegistry
from zeta.tui.app import TUIApp


_REAL_ASYNC_HTTP = httpx.AsyncHTTPTransport.handle_async_request

LIMIT = 512
LIMIT_ERROR = f"MCP tool local__echo exceeded the {LIMIT} response limit"


def _limits(*, spill_after: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        deadline_seconds=300.0,
        max_wire_bytes=LIMIT,
        spill_after_bytes=spill_after,
    )


def _http_client(url: str, *, spill_store: SpillStore | None = None):
    config = MCPServerConfig("local", "streamable-http", url=url)
    kwargs: dict[str, object] = {"spill_store": spill_store}
    if "limits" in inspect.signature(StreamableHTTPMCPClient).parameters:
        kwargs["limits"] = _limits()
    return StreamableHTTPMCPClient(config, **kwargs)


class _MCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = 0
    mode = "json"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        request_id = request.get("id")
        if request_id is None:
            self._reply(b"", status=202)
            return
        method = request["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            self._json(request_id, result)
            return
        type(self).calls += 1
        if type(self).calls > 1:
            self._json(request_id, {"content": [{"type": "text", "text": "ok"}]})
            return
        if type(self).mode == "sse":
            notification = b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
            result = (
                b"data: "
                + json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"content": []}}).encode()
                + b"\n\n"
            )
            body = notification * ((LIMIT // len(notification)) + 2) + result
            self._reply(body, content_type="text/event-stream")
            return
        value = {"jsonrpc": "2.0", "id": request_id, "result": {"content": [], "padding": ""}}
        base = json.dumps(value, separators=(",", ":")).encode()
        value["result"]["padding"] = "x" * (LIMIT + 1 - len(base))
        body = json.dumps(value, separators=(",", ":")).encode()
        assert len(body) == LIMIT + 1
        if type(self).mode == "gzip":
            self._reply(gzip.compress(body), headers={"Content-Encoding": "gzip"})
        else:
            self._reply(body)

    def _json(self, request_id: int, result: object) -> None:
        self._reply(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode())

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
def mcp_http_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _REAL_ASYNC_HTTP)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPHandler)
    _MCPHandler.calls = 0
    _MCPHandler.mode = "json"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_http_json_call_rejected_at_limit_plus_one(mcp_http_server: str) -> None:
    client = _http_client(mcp_http_server)
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        assert result["isError"] is True
        assert result["content"][0]["text"] == LIMIT_ERROR
        follow_up = await client.call_tool("echo", {}, AbortSignal())
        assert follow_up["content"][0]["text"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_sse_call_counts_pre_result_notifications(mcp_http_server: str) -> None:
    _MCPHandler.mode = "sse"
    client = _http_client(mcp_http_server)
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        assert result["isError"] is True
        assert result["content"][0]["text"] == LIMIT_ERROR
        follow_up = await client.call_tool("echo", {}, AbortSignal())
        assert follow_up["content"][0]["text"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_gzip_counts_decoded_bytes(mcp_http_server: str) -> None:
    _MCPHandler.mode = "gzip"
    client = _http_client(mcp_http_server)
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        assert result["isError"] is True
        assert result["content"][0]["text"] == LIMIT_ERROR
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_notification_get_event_overflow_degrades_once() -> None:
    body = b"data: " + b" " * LIMIT + b"{}\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)

    client = StreamableHTTPMCPClient(
        MCPServerConfig("local", "streamable-http", url="https://mcp.test"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **({"limits": _limits()} if "limits" in inspect.signature(StreamableHTTPMCPClient).parameters else {}),
    )
    failures: list[str] = []
    notifications: list[str] = []
    client.set_failure_sink(failures.append)
    client.set_notification_sink(notifications.append)
    client._session_id = "session"
    listener = asyncio.create_task(client._listen_notifications())
    await asyncio.sleep(0.05)
    listener.cancel()
    await asyncio.gather(listener, return_exceptions=True)
    assert len(failures) == 1
    assert "response limit" in failures[0]
    assert notifications == []
    assert failures, "mount must receive one transport failure and degrade"
    await client.close()


@pytest.mark.asyncio
async def test_spill_reservation_rejects_before_cap_and_releases() -> None:
    spill = SpillStore()
    configure = getattr(spill, "configure_transient_limit", lambda _value: None)
    reserve = getattr(spill, "reserve_transient", lambda _value: None)
    release = getattr(spill, "release_transient", lambda _value: None)
    configure(10)
    reserve(6)
    with pytest.raises(RuntimeError, match="spill budget"):
        reserve(5)
    release(6)
    reserve(10)
    release(10)

    async def canceled() -> None:
        reserve(10)
        try:
            await asyncio.sleep(10)
        finally:
            release(10)

    task = asyncio.create_task(canceled())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    reserve(10)
    release(10)
    spill.close()


@pytest.mark.asyncio
async def test_large_http_parse_stays_off_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import zeta.mcp.http as http_module

    real_loads = json.loads

    def slow_loads(value: object, **kwargs: object):
        time.sleep(0.05)
        return real_loads(value, **kwargs)

    monkeypatch.setattr(http_module.json, "loads", slow_loads)
    response = httpx.Response(200, content=b'{"value":"' + b"x" * 200_000 + b'"}')
    spill = SpillStore()
    ticks = 0
    running = True

    async def heartbeat() -> None:
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0)

    beat = asyncio.create_task(heartbeat())
    try:
        parameter = (
            SimpleNamespace(deadline_seconds=300.0, max_wire_bytes=300_000, spill_after_bytes=64)
            if "limits" in inspect.signature(_read_json_body).parameters
            else 300_000
        )
        result = await _read_json_body(response, spill, parameter)
        assert result["value"].startswith("x")
    finally:
        running = False
        await beat
        spill.close()
    assert ticks > 10


def _status_mount(tmp_path) -> MCPMount:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        mcp_limits=_limits(),
    )
    return MCPMount(registry)


def test_mcp_status_shows_effective_limits_tui(tmp_path) -> None:
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skip_mcp_mount=True, skill_catalog=SkillCatalog.empty())
    app = TUIApp(loop, provider="codex", model="offline")
    rendered = []
    app._presenter.print_unit = rendered.append
    output = render_mcp_status(_status_mount(tmp_path / "tools"), home=None)
    app._handle_slash_output(output)
    assert "mcp limits: deadline 300s | wire 512 bytes | spill 64 bytes" in rendered[0].plain


@pytest.mark.asyncio
async def test_mcp_status_shows_effective_limits_serve(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    mount = _status_mount(tmp_path / "tools")
    output = render_mcp_status(mount, home=None)

    class Loop:
        async def refresh_skills(self) -> None:
            return None

        async def slash_mcp(self, args: str) -> str:
            assert args == "status"
            return output

    runtime = SimpleNamespace(loop=Loop())
    monkeypatch.setattr("zeta.server.slash_commands.build_registry", lambda _runtime: SimpleNamespace(notices=()))
    response = await run_command(runtime, "/mcp status")

    assert response == {"kind": "text", "text": output}
    assert "mcp limits: deadline 300s | wire 512 bytes | spill 64 bytes" in response["text"]
