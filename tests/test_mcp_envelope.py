from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import zeta.mcp.server_actor as server_actor_module
from tests.support.fake_backend import FakeBackend
from tests.support.serve_contract import ServeClient
from tests.support.tui_render import render_full_screen
from zeta.config.settings import load_settings
from zeta.core.abort import AbortSignal
from zeta.core.store import ConversationStore
from zeta.mcp.client import MCPTool, make_error_result
from zeta.mcp.config import MCPConfig, MCPServerConfig
from zeta.mcp.envelope import MCPCallEnvelope, MCPCallLimits, MCPTransientQuotaError
from zeta.mcp.http import StreamableHTTPMCPClient, _iter_sse_events, _read_json_body
from zeta.mcp.mount import MCPMount, mount_mcp_servers
from zeta.mcp.server_actor import MCPServerActor
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer
from zeta.skills import SkillCatalog
from zeta.tools._spill import SpillStore
from zeta.tools.registry import ToolRegistry
from zeta.tui.app import TUIApp

_REAL_ASYNC_HTTP = httpx.AsyncHTTPTransport.handle_async_request

LIMIT = 512
LIMIT_ERROR = f"MCP tool local__echo exceeded the {LIMIT} bytes response limit"


def _limits(*, spill_after: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        deadline_seconds=300.0,
        max_wire_bytes=LIMIT,
        spill_after_bytes=spill_after,
    )


def _http_client(url: str, settings_home, *, spill_store: SpillStore | None = None):
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
    release_deadline = threading.Event()
    deadline_streaming = threading.Event()

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        request_id = request.get("id")
        if request_id is None:
            self._reply(b"", status=202)
            return
        method = request["method"]
        if method == "initialize":
            capabilities = {"tools": {}} if type(self).mode == "deadline-sse" else {}
            result = {"protocolVersion": "2025-06-18", "capabilities": capabilities}
            self._json(request_id, result)
            return
        if method == "tools/list":
            self._json(
                request_id,
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            )
            return
        type(self).calls += 1
        if type(self).mode == "deadline-sse" and type(self).calls == 1:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            notification = b'data: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n\n'
            try:
                while not type(self).release_deadline.is_set():
                    self.wfile.write(notification)
                    self.wfile.flush()
                    type(self).deadline_streaming.set()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
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
            final = (
                b"data: "
                + json.dumps(final_value, separators=(",", ":")).encode()
                + b"\n\n"
            )
            final_value["params"]["padding"] = "x" * (remainder - len(final))
            final = (
                b"data: "
                + json.dumps(final_value, separators=(",", ":")).encode()
                + b"\n\n"
            )
            body = notification * 5 + final
            assert len(body) == LIMIT + 1
            body += (
                b"data: "
                + json.dumps(
                    {"jsonrpc": "2.0", "id": request_id, "result": {"content": []}}
                ).encode()
                + b"\n\n"
            )
            self._reply(body, content_type="text/event-stream")
            return
        value = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [], "padding": ""},
        }
        base = json.dumps(value, separators=(",", ":")).encode()
        value["result"]["padding"] = "x" * (LIMIT + 1 - len(base))
        body = json.dumps(value, separators=(",", ":")).encode()
        assert len(body) == LIMIT + 1
        if type(self).mode == "gzip":
            self._reply(gzip.compress(body), headers={"Content-Encoding": "gzip"})
        else:
            self._reply(body)

    def _json(self, request_id: int, result: object) -> None:
        self._reply(
            json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}).encode()
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
def mcp_http_server(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _REAL_ASYNC_HTTP
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPHandler)
    _MCPHandler.calls = 0
    _MCPHandler.mode = "json"
    _MCPHandler.release_deadline.clear()
    _MCPHandler.deadline_streaming.clear()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        _MCPHandler.release_deadline.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_http_deadline_retains_cancellation_resistant_call_until_shutdown(
    mcp_http_server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _MCPHandler.mode = "deadline-sse"
    signal = AbortSignal()
    abort_on_base: asyncio.Task[None] | None = None
    deadline_type = getattr(server_actor_module, "MCPCallDeadline", None)
    clock_offset = 0.0
    loop = asyncio.get_running_loop()

    def clock() -> float:
        return loop.time() + clock_offset

    if deadline_type is not None:

        async def expire_after_streaming(_deadline) -> None:
            nonlocal clock_offset
            await asyncio.to_thread(_MCPHandler.deadline_streaming.wait)
            clock_offset = 301.0

        monkeypatch.setattr(deadline_type, "wait", expire_after_streaming)
    else:

        async def abort_after_streaming() -> None:
            await asyncio.to_thread(_MCPHandler.deadline_streaming.wait)
            signal.abort()

        abort_on_base = asyncio.create_task(abort_after_streaming())

    release_call = asyncio.Event()
    call_resisted_cancellation = asyncio.Event()
    resistant_task: asyncio.Task[object] | None = None
    real_call_tool = StreamableHTTPMCPClient.call_tool

    async def cancellation_resistant_call(client, *args, **kwargs):
        nonlocal resistant_task
        try:
            return await real_call_tool(client, *args, **kwargs)
        except asyncio.CancelledError:
            resistant_task = asyncio.current_task()
            resistant_task.uncancel()
            call_resisted_cancellation.set()
            await release_call.wait()
            raise

    monkeypatch.setattr(
        StreamableHTTPMCPClient, "call_tool", cancellation_resistant_call
    )
    monkeypatch.setattr(server_actor_module, "CLIENT_CLOSE_TIMEOUT_SECONDS", 0)
    shutdown_joining = asyncio.Event()
    real_shutdown_children = MCPServerActor._shutdown_children

    async def observe_shutdown_children(actor) -> None:
        shutdown_joining.set()
        await real_shutdown_children(actor)

    monkeypatch.setattr(MCPServerActor, "_shutdown_children", observe_shutdown_children)
    limits = MCPCallLimits(
        deadline_seconds=300.0,
        max_wire_bytes=64 * 1024 * 1024,
        spill_after_bytes=64 * 1024,
    )
    (tmp_path / "tools").mkdir()
    registry = ToolRegistry(
        tmp_path / "tools",
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        tool_allow=("local__echo",),
        mcp_limits=limits,
    )
    config = MCPServerConfig("local", "streamable-http", url=mcp_http_server)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {"local": config}),
    )
    actor = mount._actors["local"]
    client = mount._clients["local"]
    if deadline_type is not None:
        actor._deadline_clock = clock
    close_task: asyncio.Task[None] | None = None
    try:
        result = await mount.call_tool("local", "echo", {}, signal)
        assert result == make_error_result(
            "MCP tool local__echo exceeded the 300s deadline"
        )
        await call_resisted_cancellation.wait()
        assert resistant_task is not None
        assert resistant_task in actor._children
        assert not resistant_task.done()

        close_task = asyncio.create_task(mount.close())
        await shutdown_joining.wait()
        assert not close_task.done()
        assert resistant_task in actor._children

        release_call.set()
        _MCPHandler.release_deadline.set()
        await close_task
        assert not actor._children
        assert not client._active_requests
        assert not any(not task.done() for task in client._deadline_cleanups)
    finally:
        release_call.set()
        _MCPHandler.release_deadline.set()
        if abort_on_base is not None:
            await abort_on_base
        if close_task is None:
            await mount.close()
        elif not close_task.done():
            await close_task
        await registry.close()


@pytest.mark.asyncio
async def test_deadline_covers_call_work_before_first_yield(tmp_path: Path) -> None:
    now = 0.0

    class Client:
        async def call_tool(self, _name, _arguments, _abort_signal):
            nonlocal now
            now = 301.0
            return {"content": [{"type": "text", "text": "late"}]}

        def deadline_exceeded(self, _owner) -> None:
            pass

        def set_failure_sink(self, _sink) -> None:
            pass

        async def close(self) -> None:
            pass

    async def connect_and_list(_client) -> list[MCPTool]:
        return [MCPTool("echo", "", {"type": "object"})]

    client = Client()
    actor = MCPServerActor(
        MCPServerConfig("local", "stdio", "unused"),
        source=tmp_path / "mcp.json",
        registry=None,
        publish=lambda *_args: None,
        build_client=lambda _config: client,
        connect_and_list=connect_and_list,
    )
    if hasattr(actor, "_deadline_clock"):
        actor._deadline_clock = lambda: now
        actor._call_deadline_seconds = 300.0
    actor.start()
    await actor.wait_started(None)
    try:
        result = await actor.call_tool(
            "echo", {}, AbortSignal(), generation=actor.generation
        )

        assert result == make_error_result(
            "MCP tool local__echo exceeded the 300s deadline"
        )
    finally:
        await actor.close()


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
async def test_http_gzip_counts_decoded_bytes(mcp_http_server: str, tmp_path) -> None:
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
        **(
            {"limits": _limits()}
            if "limits" in inspect.signature(StreamableHTTPMCPClient).parameters
            else {}
        ),
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
            f"MCP HTTP notification event exceeded the {LIMIT} bytes response limit"
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
            SimpleNamespace(
                deadline_seconds=300.0, max_wire_bytes=300_000, spill_after_bytes=64
            )
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
    )
    return MCPMount(registry)


@pytest.mark.asyncio
async def test_mcp_status_shows_effective_limits_in_real_tui_render(tmp_path) -> None:
    loop = AgentLoop(
        FakeBackend([]),
        ConversationStore(tmp_path / "sessions"),
        skip_mcp_mount=True,
        skill_catalog=SkillCatalog.empty(),
    )
    mount = _status_mount(tmp_path / "tools")
    loop.attach_mcp_mount(mount)
    app = TUIApp(loop, provider="codex", model="offline")
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    try:
        await app._handle_prompt_value("/mcp status")
        screen = render_full_screen(session, 120)
        visible = "\n".join(
            "".join(screen.data_buffer[row][column].char for column in range(120))
            for row in range(24)
        )
        assert (
            "mcp limits: deadline 300s | wire 67108864 bytes | spill 262144 bytes"
            in visible
        )
    finally:
        await loop.close()
        await mount.registry.close()


@pytest.mark.asyncio
async def test_mcp_status_shows_effective_limits_through_real_serve(tmp_path) -> None:
    server = ZetaServer(home=tmp_path / "home", cwd=tmp_path, port=0, provider="codex")
    await server.start()
    assert server.port is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    client = ServeClient(reader=reader, writer=writer)
    mount = _status_mount(tmp_path / "tools")
    try:
        hello = await client.request(
            "hello",
            "hello",
            {"protocol_version": "1.0", "client_version": "1.1"},
        )
        assert hello["result"]["protocol_version"] == "1.1"
        created = await client.request("new", "new_session", {"provider": "codex"})
        session_id = created["result"]["session"]["session_id"]
        loop = server.runtime.loop
        assert loop is not None
        assert loop._mcp_mount is not None
        await loop._mcp_mount.close()
        loop._mcp_mount = None
        loop.attach_mcp_mount(mount)
        response = await client.request(
            "status",
            "slash_run",
            {"session_id": session_id, "text": "/mcp status"},
        )
        assert response["result"]["kind"] == "text"
        assert (
            "mcp limits: deadline 300s | wire 67108864 bytes | spill 262144 bytes"
            in response["result"]["text"]
        )
    finally:
        await client.close()
        await server.close()
        await mount.registry.close()
