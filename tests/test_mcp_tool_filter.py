import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from zeta.automations.delivery import SlackDelivery
from zeta.core.abort import AbortSignal
from zeta.mcp import (
    MCPConfig,
    MCPServerConfig,
    MCPTool,
    StreamableHTTPMCPClient,
    load_mcp_config,
    mount_mcp_servers,
    server_to_json,
)
from zeta.mcp.http import NOTIFICATION_RECONNECT_INITIAL_SECONDS
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


class FilterClient:
    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.protocol_version = "2025-06-18"
        self.capabilities = {"tools": {"listChanged": True}}
        self.tools = [
            MCPTool("read_file", "", {"type": "object"}),
            MCPTool("write_file", "", {"type": "object"}),
            MCPTool("delete_file", "", {"type": "object"}),
        ]
        self.calls: list[str] = []
        self.list_calls = 0
        self.notification_sink: Callable[[str], None] | None = None

    def set_failure_sink(self, _sink) -> None:
        pass

    def set_notification_sink(self, sink: Callable[[str], None] | None) -> None:
        self.notification_sink = sink

    async def connect(self) -> None:
        pass

    async def list_tools(self) -> list[MCPTool]:
        self.list_calls += 1
        return list(self.tools)

    async def call_tool(self, name: str, arguments, abort_signal: AbortSignal):
        del arguments, abort_signal
        self.calls.append(name)
        return {"content": [], "isError": False, "structuredContent": None}

    async def close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("field", "value"),
    [("allowed_tools", "read_file"), ("disallowed_tools", ["", "write_file"])],
)
def test_mcp_tool_filter_config_rejects_invalid_lists(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "files": {
                        "transport": "stdio",
                        "command": "files",
                        field: value,
                    }
                }
            }
        )
    )

    config = load_mcp_config(path)

    assert "files" in config.malformed_servers
    assert field in (config.malformed_servers["files"].malformed_reason or "")


def test_mcp_tool_filter_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "files": {
                        "transport": "stdio",
                        "command": "files",
                        "allowed_tools": ["read_*", "search"],
                        "disallowed_tools": ["read_secret"],
                    }
                }
            }
        )
    )

    server = load_mcp_config(path).servers["files"]

    assert server.allowed_tools == ("read_*", "search")
    assert server.disallowed_tools == ("read_secret",)
    assert server_to_json(server)["allowed_tools"] == ["read_*", "search"]
    assert server_to_json(server)["disallowed_tools"] == ["read_secret"]


async def _mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: MCPServerConfig,
    *,
    tool_allow: tuple[str, ...] | None = None,
    tool_deny: tuple[str, ...] = (),
    notices: list[str] | None = None,
):
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
        tool_allow=tool_allow,
        tool_deny=tool_deny,
    )
    client = FilterClient(config)
    monkeypatch.setattr("zeta.mcp.mount._build_client", lambda _config: client)
    mount = await mount_mcp_servers(
        registry,
        MCPConfig(tmp_path / "mcp.json", {config.name: config}),
        notice_sink=None if notices is None else notices.append,
    )
    return registry, mount, client


@pytest.mark.asyncio
async def test_mcp_allowed_tools_filter_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused", allowed_tools=("read_*",))
    registry, mount, _client = await _mount(monkeypatch, tmp_path, config)
    try:
        assert registry.registered_names == {"files__read_file"}
        assert [tool.name for tool in mount._actors["files"]._tools] == ["read_file"]
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_disallowed_tools_filter_and_global_policy_combine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused", disallowed_tools=("write_*",))
    registry, mount, _client = await _mount(
        monkeypatch,
        tmp_path,
        config,
        tool_allow=("files__read_*", "files__write_*"),
        tool_deny=("files__delete_*",),
    )
    try:
        assert [tool.name for tool in mount._actors["files"]._tools] == [
            "read_file",
            "delete_file",
        ]
        assert registry.registered_names == {"files__read_file"}
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_unknown_filter_names_warn_and_zero_tools_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notices: list[str] = []
    config = MCPServerConfig(
        "files",
        "stdio",
        "unused",
        allowed_tools=("missing",),
        disallowed_tools=("also_missing",),
    )
    registry, mount, _client = await _mount(
        monkeypatch, tmp_path, config, notices=notices
    )
    try:
        assert registry.registered_names == frozenset()
        assert any(
            "unknown allowed_tools: missing; unknown disallowed_tools: also_missing"
            in notice
            for notice in notices
        )
        assert any("mounted with zero tools" in notice for notice in notices)
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_filtered_tool_call_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused", allowed_tools=("read_file",))
    registry, mount, client = await _mount(monkeypatch, tmp_path, config)
    try:
        actor = mount._actors["files"]
        result = await actor.call_tool(
            "write_file", {}, AbortSignal(), generation=actor.generation
        )
        assert result["isError"] is True
        assert "filtered" in result["content"][0]["text"]
        assert client.calls == []
        unknown = await registry.execute(ToolCall("old", "files__write_file", {}))
        assert unknown["isError"] is True
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_automation_resolve_fails_closed_for_filtered_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig(
        "slack", "stdio", "unused", allowed_tools=("slack_send_message",)
    )
    _registry, mount, client = await _mount(monkeypatch, tmp_path, config)
    try:
        with pytest.raises(ValueError, match="filtered"):
            await SlackDelivery(mount).resolve("slack:@austin")
        assert client.calls == []
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_tools_list_changed_reapplies_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused", allowed_tools=("read_*",))
    registry, mount, client = await _mount(monkeypatch, tmp_path, config)
    try:
        client.tools = [
            MCPTool("read_next", "", {"type": "object"}),
            MCPTool("write_next", "", {"type": "object"}),
        ]
        assert client.notification_sink is not None
        client.notification_sink("notifications/tools/list_changed")
        client.notification_sink("notifications/tools/list_changed")
        for _ in range(20):
            if (
                "files__read_next" in registry.registered_names
                and client.list_calls == 3
            ):
                break
            await asyncio.sleep(0)
        assert registry.registered_names == {"files__read_next"}
        assert client.list_calls == 3
    finally:
        await mount.close()


class SetupNotificationClient(FilterClient):
    async def connect(self) -> None:
        self.tools = [MCPTool("after_setup", "", {"type": "object"})]
        await super().connect()
        assert self.notification_sink is not None
        self.notification_sink("notifications/tools/list_changed")


class FailedSetupNotificationClient(FilterClient):
    async def connect(self) -> None:
        assert self.notification_sink is not None
        self.notification_sink("notifications/tools/list_changed")
        raise RuntimeError("setup failed")


class BlockingRefreshClient(FilterClient):
    def __init__(self, config: MCPServerConfig) -> None:
        super().__init__(config)
        self.refresh_started = asyncio.Event()
        self.release_refresh = asyncio.Event()

    async def list_tools(self) -> list[MCPTool]:
        self.list_calls += 1
        if self.list_calls > 1:
            self.refresh_started.set()
            await self.release_refresh.wait()
        return list(self.tools)


@pytest.mark.asyncio
async def test_mcp_setup_notification_is_refreshed_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused")
    client = SetupNotificationClient(config)
    monkeypatch.setattr("zeta.mcp.mount._build_client", lambda _config: client)
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {config.name: config})
    )
    try:
        assert registry.registered_names == {"files__after_setup"}
        assert client.list_calls == 2
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_failed_setup_notification_is_not_applied_to_next_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused")
    failed = FailedSetupNotificationClient(config)
    connected = FilterClient(config)
    clients = iter((failed, connected))
    monkeypatch.setattr("zeta.mcp.mount._build_client", lambda _config: next(clients))
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {config.name: config})
    )
    try:
        await mount.reconnect("files")
        assert connected.list_calls == 1
    finally:
        await mount.close()


@pytest.mark.asyncio
async def test_mcp_close_cancels_blocked_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MCPServerConfig("files", "stdio", "unused")
    client = BlockingRefreshClient(config)
    monkeypatch.setattr("zeta.mcp.mount._build_client", lambda _config: client)
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {config.name: config})
    )
    client.tools = [MCPTool("changed", "", {"type": "object"})]
    assert client.notification_sink is not None
    client.notification_sink("notifications/tools/list_changed")
    await asyncio.wait_for(client.refresh_started.wait(), 1)
    await asyncio.wait_for(mount.close(), 1)


def _http_rpc_response(
    request: httpx.Request, tools: set[str], requests: list[str]
) -> httpx.Response:
    payload = json.loads(request.content) if request.content else {}
    if "id" not in payload:
        return httpx.Response(202, request=request)
    requests.append(payload["method"])
    result = (
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": True}},
        }
        if payload["method"] == "initialize"
        else {
            "tools": [
                {"name": name, "description": "", "inputSchema": {"type": "object"}}
                for name in sorted(tools)
            ]
        }
    )
    headers = (
        {"mcp-session-id": "test-session"} if payload["method"] == "initialize" else {}
    )
    return httpx.Response(
        200,
        headers=headers,
        json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        request=request,
    )


@pytest.mark.asyncio
async def test_mcp_http_multiline_idle_notification_updates_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {"gone", "stay"}
    requests: list[str] = []
    notify = asyncio.Event()
    get_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            get_started.set()
            await notify.wait()
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"jsonrpc":"2.0",\n'
                    b'data: "method":"notifications/tools/list_changed"}\n\n'
                ),
                request=request,
            )
        return _http_rpc_response(request, tools, requests)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    monkeypatch.setattr(
        "zeta.mcp.mount._build_client",
        lambda _config: __import__(
            "zeta.mcp.http", fromlist=["StreamableHTTPMCPClient"]
        ).StreamableHTTPMCPClient(
            _config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {"http": config})
    )
    try:
        assert registry.registered_names == {"http__gone", "http__stay"}
        await asyncio.wait_for(get_started.wait(), 1)
        tools.remove("gone")
        notify.set()
        for _ in range(100):
            if registry.registered_names == {"http__stay"}:
                break
            await asyncio.sleep(0.01)
        assert registry.registered_names == {"http__stay"}
        assert "tools/list" in requests
    finally:
        await mount.close()


@pytest.mark.parametrize("first_stream", ["eof", "error"])
@pytest.mark.asyncio
async def test_mcp_http_notification_listener_reconnects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_stream: str
) -> None:
    tools = {"gone", "stay"}
    requests: list[str] = []
    get_count = 0
    first_get = asyncio.Event()
    allow_second = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_count
        if request.method != "GET":
            return _http_rpc_response(request, tools, requests)
        get_count += 1
        if get_count == 1:
            first_get.set()
            if first_stream == "error":
                raise httpx.ConnectError("stream failed", request=request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"",
                request=request,
            )
        if get_count == 2:
            await allow_second.wait()
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n',
                request=request,
            )
        return httpx.Response(405, request=request)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    monkeypatch.setattr(
        "zeta.mcp.mount._build_client",
        lambda _config: __import__(
            "zeta.mcp.http", fromlist=["StreamableHTTPMCPClient"]
        ).StreamableHTTPMCPClient(
            _config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {"http": config})
    )
    try:
        await asyncio.wait_for(first_get.wait(), 1)
        tools.remove("gone")
        allow_second.set()
        for _ in range(200):
            if registry.registered_names == {"http__stay"}:
                break
            await asyncio.sleep(0.01)
        assert registry.registered_names == {"http__stay"}
        assert get_count >= 2
    finally:
        await mount.close()


async def _run_notification_listener(
    config: MCPServerConfig,
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[StreamableHTTPMCPClient, list[str]]:
    failures: list[str] = []
    client = StreamableHTTPMCPClient(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client._session_id = "session-1"
    client.set_failure_sink(failures.append)
    await client._listen_notifications()
    await client.close()
    return client, failures


@pytest.mark.parametrize("status", [403, 404])
@pytest.mark.asyncio
async def test_mcp_http_notification_listener_reports_terminal_4xx(
    status: int,
) -> None:
    get_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_count
        get_count += 1
        return httpx.Response(
            status if get_count == 1 else 405,
            text="session rejected",
            request=request,
        )

    _, failures = await _run_notification_listener(
        MCPServerConfig("http", "streamable-http", url="https://mcp.test"), handler
    )

    assert get_count == 1
    assert failures == [f"MCP HTTP {status}: session rejected"]


@pytest.mark.asyncio
async def test_mcp_http_notification_listener_retries_503_with_fresh_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[str] = []
    delays: list[float] = []
    client: StreamableHTTPMCPClient

    def handler(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["mcp-session-id"])
        return httpx.Response(
            503 if len(sessions) <= 2 else 405, text="unavailable", request=request
        )

    async def sleep(delay: float) -> None:
        delays.append(delay)
        client._session_id = "session-2"

    monkeypatch.setattr("zeta.mcp.http.asyncio.sleep", sleep)
    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    client = StreamableHTTPMCPClient(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client._session_id = "session-1"
    await client._listen_notifications()
    await client.close()

    assert sessions == ["session-1", "session-2", "session-2"]
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_mcp_http_notification_listener_immediate_eof_backs_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[str] = []
    delays: list[float] = []
    client: StreamableHTTPMCPClient

    def handler(request: httpx.Request) -> httpx.Response:
        sessions.append(request.headers["mcp-session-id"])
        return httpx.Response(
            200 if len(sessions) <= 2 else 405,
            headers={"content-type": "text/event-stream"},
            content=b"",
            request=request,
        )

    async def sleep(delay: float) -> None:
        delays.append(delay)
        client._session_id = "session-2"

    monkeypatch.setattr("zeta.mcp.http.asyncio.sleep", sleep)
    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    client = StreamableHTTPMCPClient(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    client._session_id = "session-1"
    await client._listen_notifications()
    await client.close()

    assert sessions == ["session-1", "session-2", "session-2"]
    assert delays == [0.1, 0.2]


class _DelayedEOFStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.sleep(0.02)
        if False:
            yield b""


@pytest.mark.asyncio
async def test_mcp_http_notification_listener_healthy_idle_resets_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_at: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_at.append(time.monotonic())
        if len(requested_at) <= 2:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"",
                request=request,
            )
        if len(requested_at) == 3:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_DelayedEOFStream(),
                request=request,
            )
        return httpx.Response(405, request=request)

    monkeypatch.setattr("zeta.mcp.http.NOTIFICATION_STREAM_HEALTHY_SECONDS", 0.01)
    await _run_notification_listener(
        MCPServerConfig("http", "streamable-http", url="https://mcp.test"), handler
    )

    reconnect_delay = requested_at[3] - requested_at[2] - 0.02
    assert reconnect_delay == pytest.approx(
        NOTIFICATION_RECONNECT_INITIAL_SECONDS, abs=0.08
    )


@pytest.mark.asyncio
async def test_mcp_http_notification_listener_stops_retrying_on_405(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {"stay"}
    requests: list[str] = []
    get_count = 0
    get_finished = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_count
        if request.method == "GET":
            get_count += 1
            if get_count == 2:
                get_finished.set()
                return httpx.Response(405, request=request)
            return httpx.Response(500, request=request)
        return _http_rpc_response(request, tools, requests)

    config = MCPServerConfig("http", "streamable-http", url="https://mcp.test")
    monkeypatch.setattr(
        "zeta.mcp.mount._build_client",
        lambda _config: __import__(
            "zeta.mcp.http", fromlist=["StreamableHTTPMCPClient"]
        ).StreamableHTTPMCPClient(
            _config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ),
    )
    registry = ToolRegistry(
        tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
    )
    mount = await mount_mcp_servers(
        registry, MCPConfig(tmp_path / "mcp.json", {"http": config})
    )
    try:
        await asyncio.wait_for(get_finished.wait(), 1)
        await asyncio.sleep(0.2)
        assert get_count == 2
    finally:
        await mount.close()
