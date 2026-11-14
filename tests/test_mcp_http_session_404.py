from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tests.support.owned import await_or_fail
from zeta.core.abort import AbortSignal
from zeta.mcp.client import MCPHTTPError
from zeta.mcp.config import MCPServerConfig
from zeta.mcp.http import StreamableHTTPMCPClient


class _RestartingMCP:
    def __init__(self) -> None:
        self.session: str | None = None
        self.next_session = 0
        self.requests: list[tuple[str, str | None]] = []
        self.tool_calls = 0
        self.reject_retry = False
        self.block_initialize = False
        self.fail_initialize_once = False
        self.initialize_started = asyncio.Event()
        self.release_initialize = asyncio.Event()
        self.initialize_cancelled = asyncio.Event()

    def restart(self) -> None:
        self.session = None

    def block_next_initialize(self) -> None:
        self.block_initialize = True
        self.initialize_started.clear()
        self.release_initialize.clear()
        self.initialize_cancelled.clear()

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        session_id = request.headers.get("mcp-session-id")
        method = body["method"]
        self.requests.append((method, session_id))
        if method == "notifications/initialized":
            return httpx.Response(202, request=request)
        if method == "initialize":
            if self.block_initialize:
                self.block_initialize = False
                self.initialize_started.set()
                try:
                    await asyncio.wait_for(self.release_initialize.wait(), 10)
                except asyncio.CancelledError:
                    self.initialize_cancelled.set()
                    raise
            if self.fail_initialize_once:
                self.fail_initialize_once = False
                return httpx.Response(500, request=request, text="initialize failed")
            self.next_session += 1
            self.session = f"session-{self.next_session}"
            return httpx.Response(
                200,
                request=request,
                headers={"mcp-session-id": self.session},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                    },
                },
            )
        if session_id != self.session or self.reject_retry:
            return httpx.Response(404, request=request, text="unknown session")
        if method == "tools/call":
            self.tool_calls += 1
            result = {"content": [{"type": "text", "text": "ok"}]}
        else:
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        return httpx.Response(
            200,
            request=request,
            json={"jsonrpc": "2.0", "id": body["id"], "result": result},
        )


@pytest.fixture
def restarting_mcp() -> _RestartingMCP:
    return _RestartingMCP()


@pytest.fixture
async def client(restarting_mcp: _RestartingMCP):
    transport = httpx.MockTransport(restarting_mcp)
    http_client = httpx.AsyncClient(transport=transport)
    client = StreamableHTTPMCPClient(
        MCPServerConfig("restart", "streamable-http", url="http://mcp.test"),
        client=http_client,
    )
    await client.connect()
    try:
        yield client
    finally:
        await client.close()
        await http_client.aclose()


@pytest.mark.asyncio
async def test_http_404_reinitializes_and_retries_list(client, restarting_mcp):
    await client.list_tools()
    restarting_mcp.restart()

    tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["echo"]
    assert restarting_mcp.requests == [
        ("initialize", None),
        ("notifications/initialized", "session-1"),
        ("tools/list", "session-1"),
        ("tools/list", "session-1"),
        ("initialize", None),
        ("notifications/initialized", "session-2"),
        ("tools/list", "session-2"),
    ]


@pytest.mark.asyncio
async def test_http_404_retries_tool_call_once(client, restarting_mcp):
    await client.list_tools()
    restarting_mcp.restart()

    result = await client.call_tool("echo", {}, AbortSignal())

    assert result["isError"] is False
    assert restarting_mcp.tool_calls == 1


@pytest.mark.asyncio
async def test_http_404_concurrent_requests_share_reinitialize(client, restarting_mcp):
    await client.list_tools()
    restarting_mcp.restart()

    await await_or_fail(
        asyncio.gather(client.list_tools(), client.list_tools()),
        what="concurrent list requests",
    )

    assert sum(method == "initialize" for method, _session in restarting_mcp.requests) == 2


@pytest.mark.asyncio
async def test_http_404_on_retry_does_not_loop(client, restarting_mcp):
    await client.list_tools()
    restarting_mcp.restart()
    restarting_mcp.reject_retry = True

    with pytest.raises(MCPHTTPError, match="404"):
        await client.list_tools()

    assert sum(method == "initialize" for method, _session in restarting_mcp.requests) == 2


@pytest.mark.asyncio
async def test_http_404_without_session_is_plain_error(client, restarting_mcp):
    client._session_id = None

    with pytest.raises(MCPHTTPError, match="404"):
        await client.list_tools()

    assert sum(method == "initialize" for method, _session in restarting_mcp.requests) == 1


@pytest.mark.asyncio
async def test_cancelled_recovery_owner_does_not_cancel_recovery(
    client, restarting_mcp
):
    restarting_mcp.restart()
    restarting_mcp.block_next_initialize()
    first = asyncio.create_task(client.list_tools())
    await await_or_fail(
        restarting_mcp.initialize_started.wait(), what="recovery initialize start"
    )

    first.cancel()
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), 10)
    assert not restarting_mcp.initialize_cancelled.is_set()

    second = asyncio.create_task(client.list_tools())
    restarting_mcp.release_initialize.set()
    tools = await await_or_fail(second, what="list after cancelled recovery owner")

    assert [tool.name for tool in tools] == ["echo"]
    assert sum(method == "initialize" for method, _ in restarting_mcp.requests) == 2


@pytest.mark.asyncio
async def test_failed_recovery_is_retried_by_next_request(client, restarting_mcp):
    restarting_mcp.restart()
    restarting_mcp.fail_initialize_once = True

    with pytest.raises(MCPHTTPError, match="500"):
        await client.list_tools()

    tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["echo"]
    assert sum(method == "initialize" for method, _ in restarting_mcp.requests) == 3


@pytest.mark.asyncio
async def test_request_started_during_recovery_waits_for_shared_initialize(
    client, restarting_mcp, monkeypatch
):
    restarting_mcp.restart()
    restarting_mcp.block_next_initialize()
    first = asyncio.create_task(client.list_tools())
    await await_or_fail(
        restarting_mcp.initialize_started.wait(), what="recovery initialize start"
    )
    second_entered = asyncio.Event()
    original = client._send_with_auth

    async def observe_send(payload, request_id, state, **kwargs):
        if payload["method"] == "tools/list":
            second_entered.set()
        return await original(payload, request_id, state, **kwargs)

    # test-contract: seam observe entry before the transport waits on recovery
    monkeypatch.setattr(client, "_send_with_auth", observe_send)
    second = asyncio.create_task(client.list_tools())
    await await_or_fail(second_entered.wait(), what="second list request start")

    assert not second.done()
    assert restarting_mcp.requests[-1] == ("initialize", None)

    restarting_mcp.release_initialize.set()
    first_tools, second_tools = await await_or_fail(
        asyncio.gather(first, second), what="requests sharing recovery"
    )
    assert [tool.name for tool in first_tools] == ["echo"]
    assert [tool.name for tool in second_tools] == ["echo"]
    assert sum(method == "initialize" for method, _ in restarting_mcp.requests) == 2


@pytest.mark.asyncio
async def test_notification_404_recovers_and_replaces_listener(
    client, restarting_mcp, monkeypatch
):
    old_listener = client._notification_task
    assert old_listener is not None
    old_listener.cancel()
    await asyncio.wait_for(
        asyncio.gather(old_listener, return_exceptions=True), 10
    )
    restarting_mcp.restart()
    listener_calls: list[str | None] = []
    new_listener_started = asyncio.Event()
    release_listener = asyncio.Event()

    async def listen_once(_token):
        listener_calls.append(client._session_id)
        if client._session_id == "session-1":
            raise MCPHTTPError(404, "unknown session")
        new_listener_started.set()
        await asyncio.wait_for(release_listener.wait(), 10)
        return False

    # test-contract: seam force the GET stream's 404 and hold its replacement
    monkeypatch.setattr(client, "_listen_notification_stream", listen_once)
    replacement_owner = asyncio.create_task(client._listen_notifications())
    client._notification_task = replacement_owner
    try:
        await await_or_fail(new_listener_started.wait(), what="new session listener")
        assert listener_calls == ["session-1", "session-2"]
        assert client._notification_task is not replacement_owner
        assert sum(method == "initialize" for method, _ in restarting_mcp.requests) == 2
    finally:
        release_listener.set()


@pytest.mark.asyncio
async def test_close_cancels_recovery_after_caller_stops_waiting(
    client, restarting_mcp
):
    restarting_mcp.restart()
    restarting_mcp.block_next_initialize()
    request = asyncio.create_task(client.list_tools())
    await await_or_fail(
        restarting_mcp.initialize_started.wait(), what="recovery initialize start"
    )
    request.cancel()
    await asyncio.wait_for(asyncio.gather(request, return_exceptions=True), 10)

    assert not restarting_mcp.initialize_cancelled.is_set()

    await client.close()
    await await_or_fail(
        restarting_mcp.initialize_cancelled.wait(), what="recovery cancellation on close"
    )
    assert client._notification_task is None
    # test-contract: seam close must release both owned background task slots
    assert client._session_recovery_task is None
