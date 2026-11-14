from __future__ import annotations

import asyncio
import json

import httpx
import pytest

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

    def restart(self) -> None:
        self.session = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        session_id = request.headers.get("mcp-session-id")
        method = body["method"]
        self.requests.append((method, session_id))
        if method == "notifications/initialized":
            return httpx.Response(202, request=request)
        if method == "initialize":
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

    await asyncio.gather(client.list_tools(), client.list_tools())

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
