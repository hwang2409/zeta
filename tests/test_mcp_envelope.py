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

from tests.support.fake_backend import FakeBackend
from zeta.config.settings import load_settings
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.mcp.commands import render_mcp_status
from zeta.mcp.config import MCPConfig, MCPServerConfig
from zeta.mcp.envelope import MCPCallEnvelope, MCPTransientQuotaError
from zeta.mcp.http import StreamableHTTPMCPClient, _iter_sse_events, _read_json_body
from zeta.mcp.mount import MCPMount, mount_mcp_servers
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


def _http_client(
    url: str, settings_home, *, spill_store: SpillStore | None = None
):
    settings_home.mkdir(parents=True, exist_ok=True)
    (settings_home / "settings.toml").write_text(
        "[mcp]\nmax_wire_bytes = 512\nspill_after_bytes = 64\n"
    )
    settings = load_settings(home=settings_home).settings
    config = MCPServerConfig("local", "streamable-http", url=url)
    kwargs: dict[str, object] = {"spill_store": spill_store}
    if "limits" in inspect.signature(StreamableHTTPMCPClient).parameters:
        kwargs["limits"] = SimpleNamespace(
            deadline_seconds=getattr(settings, "mcp_call_deadline_seconds", 300.0),
            max_wire_bytes=getattr(settings, "mcp_max_wire_bytes", 64 * 1024 * 1024),
            spill_after_bytes=getattr(settings, "mcp_spill_after_bytes", 256 * 1024),
        )
    return StreamableHTTPMCPClient(config, **kwargs)


class _MCPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    calls = 0
    mode = "json"

    def do_POST(self) -> None:
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
            remainder = LIMIT + 1 - len(notification) * 5
            final_value = {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"padding": ""},
            }
            final = b"data: " + json.dumps(final_value, separators=(",", ":")).encode() + b"\n\n"
            final_value["params"]["padding"] = "x" * (remainder - len(final))
            final = b"data: " + json.dumps(final_value, separators=(",", ":")).encode() + b"\n\n"
            body = notification * 5 + final
            assert len(body) == LIMIT + 1
            body += (
                b"data: "
                + json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {"content": []}}).encode()
                + b"\n\n"
            )
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
async def test_http_json_call_rejected_at_limit_plus_one(
    mcp_http_server: str, tmp_path
) -> None:
    client = _http_client(mcp_http_server, tmp_path / "home")
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
async def test_http_sse_call_counts_pre_result_notifications(
    mcp_http_server: str, tmp_path
) -> None:
    _MCPHandler.mode = "sse"
    client = _http_client(mcp_http_server, tmp_path / "home")
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
async def test_http_gzip_counts_decoded_bytes(
    mcp_http_server: str, tmp_path
) -> None:
    _MCPHandler.mode = "gzip"
    client = _http_client(mcp_http_server, tmp_path / "home")
    try:
        await client.connect()
        result = await client.call_tool("echo", {}, AbortSignal())
        assert result["isError"] is True
        assert result["content"][0]["text"] == LIMIT_ERROR
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_notification_get_event_overflow_degrades_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"data: " + b" " * LIMIT + b"{}\n\n"
    allow_get_response = asyncio.Event()
    get_requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_requests
        if request.method == "GET":
            get_requests += 1
            await allow_get_response.wait()
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
                request=request,
            )
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202, request=request)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                },
            },
            headers={"mcp-session-id": "session"},
            request=request,
        )

    client = StreamableHTTPMCPClient(
        MCPServerConfig("local", "streamable-http", url="https://mcp.test"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **({"limits": _limits()} if "limits" in inspect.signature(StreamableHTTPMCPClient).parameters else {}),
    )
    monkeypatch.setattr("zeta.mcp.mount._build_client", lambda _config: client)
    tools = tmp_path / "tools"
    tools.mkdir()
    registry_kwargs: dict[str, object] = {}
    if "mcp_limits" in inspect.signature(ToolRegistry).parameters:
        registry_kwargs["mcp_limits"] = _limits()
    registry = ToolRegistry(
        tools,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        **registry_kwargs,
    )
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"local": client.config}),
    )
    try:
        failures: list[str] = []
        actor_failure_sink = client._failure_sink
        assert actor_failure_sink is not None

        def record_failure(reason: str) -> None:
            failures.append(reason)
            actor_failure_sink(reason)

        client.set_failure_sink(record_failure)
        allow_get_response.set()
        for _ in range(100):
            if mount.statuses["local"].state == "degraded":
                break
            await asyncio.sleep(0.001)
        assert mount.statuses["local"].state == "degraded"
        assert failures == [
            f"MCP HTTP notification event exceeded the {LIMIT} response limit"
        ]
        assert get_requests == 1
        assert mount._actors["local"]._requests == {}
    finally:
        await mount.close()
        await registry.close()


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
    reserve(4)
    release(10)
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
    with pytest.raises(ValueError, match="exceeds its reservation"):
        release(1)
    spill.close()


@pytest.mark.asyncio
async def test_spill_reservation_does_not_wait_for_spill_write() -> None:
    spill = SpillStore()
    spill.configure_transient_limit(64)
    write_started = threading.Event()
    release_write = threading.Event()
    heartbeat_advanced = threading.Event()

    class BlockingPart:
        released_after_heartbeat = False

        def seek(self, _offset: int) -> None:
            return None

        def read(self, _size: int) -> bytes:
            write_started.set()
            release_write.wait(timeout=10)
            self.released_after_heartbeat = heartbeat_advanced.is_set()
            return b""

    part = BlockingPart()
    write = asyncio.create_task(spill.awrite_parts("tool", "call", 0, [part]))
    assert await asyncio.to_thread(write_started.wait, 10)

    envelope = MCPCallEnvelope(_limits(spill_after=1), spill)

    async def heartbeat() -> None:
        heartbeat_advanced.set()
        release_write.set()

    reserve = asyncio.create_task(envelope.append(b"{}"))
    beat = asyncio.create_task(heartbeat())
    guard = threading.Timer(10, release_write.set)
    guard.start()
    try:
        await reserve
        await beat
        await write
        assert part.released_after_heartbeat
    finally:
        guard.cancel()
        release_write.set()
        await envelope.close()
        await write
        spill.close()


@pytest.mark.asyncio
async def test_sse_cleanup_failure_still_releases_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spill = SpillStore()
    spill.configure_transient_limit(64)
    real_temporary_file = spill.temporary_file
    temporary_calls = 0

    class FailingClose:
        def __init__(self) -> None:
            self._context = real_temporary_file()
            self._closed = False

        def __enter__(self):
            return self._context.__enter__()

        def __exit__(self, exc_type, exc, traceback) -> None:
            if not self._closed:
                self._closed = True
                self._context.__exit__(exc_type, exc, traceback)
            raise OSError("injected close failure")

    def temporary_file():
        nonlocal temporary_calls
        temporary_calls += 1
        if temporary_calls == 1:
            return FailingClose()
        return real_temporary_file()

    monkeypatch.setattr(spill, "temporary_file", temporary_file)
    limits = _limits(spill_after=8)
    value = "x" * 40
    class BrokenResponse:
        async def aiter_bytes(self, *, chunk_size: int):
            assert chunk_size == 65_536
            yield b"data: " + json.dumps({"value": value}).encode()
            raise OSError("injected stream failure")

    response = BrokenResponse()

    with pytest.raises(OSError, match="injected close failure"):
        async for _event in _iter_sse_events(response, spill, limits):
            pass

    follow_up = httpx.Response(
        200,
        content=json.dumps({"value": value}, separators=(",", ":")).encode(),
    )
    try:
        result = await _read_json_body(follow_up, spill, limits)
    except MCPTransientQuotaError:
        result = None
    finally:
        spill.close()
    assert result == {"value": value}


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
    kwargs: dict[str, object] = {}
    if "mcp_limits" in inspect.signature(ToolRegistry).parameters:
        kwargs["mcp_limits"] = _limits()
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        **kwargs,
    )
    return MCPMount(registry)


@pytest.mark.asyncio
async def test_mcp_status_shows_effective_limits_tui(tmp_path) -> None:
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skip_mcp_mount=True, skill_catalog=SkillCatalog.empty())
    app = TUIApp(loop, provider="codex", model="offline")
    mount = _status_mount(tmp_path / "tools")
    rendered = []
    app._presenter.print_unit = rendered.append
    try:
        output = render_mcp_status(mount, home=None)
        app._handle_slash_output(output)
        assert "mcp limits: deadline 300s | wire 512 bytes | spill 64 bytes" in rendered[0].plain
    finally:
        await mount.close()
        await mount.registry.close()
        await loop.close()


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

    try:
        assert response == {"kind": "text", "text": output}
        assert "mcp limits: deadline 300s | wire 512 bytes | spill 64 bytes" in response["text"]
    finally:
        await mount.close()
        await mount.registry.close()
