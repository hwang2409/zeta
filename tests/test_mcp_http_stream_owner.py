from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import pytest

from zeta.mcp.client import MCPHTTPError
from zeta.mcp.config import MCPServerConfig
from zeta.mcp.http import StreamableHTTPMCPClient, _ActiveRequest

Route = Callable[[StreamableHTTPMCPClient], Awaitable[object]]


def _client(handler: httpx.AsyncBaseTransport) -> StreamableHTTPMCPClient:
    return StreamableHTTPMCPClient(
        MCPServerConfig("test", "streamable-http", url="https://mcp.test/rpc"),
        client=httpx.AsyncClient(transport=handler),
    )


async def _send(client: StreamableHTTPMCPClient) -> object:
    return await client._send(
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        1,
        _ActiveRequest(),
        None,
    )


async def _listen(client: StreamableHTTPMCPClient) -> object:
    client._session_id = "session-1"
    return await client._listen_notification_stream(None)


async def _send_notification(client: StreamableHTTPMCPClient) -> object:
    return await client._send_notification_raw("notifications/test", {}, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [_send, _listen, _send_notification])
async def test_empty_401_has_the_same_auth_mapping(route: Route) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request)

    client = _client(httpx.MockTransport(handler))
    try:
        with pytest.raises(MCPHTTPError) as caught:
            await route(client)
        assert caught.value.status_code == 401
        assert "credentials rejected: unauthorized" in str(caught.value)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [_send, _listen, _send_notification])
async def test_network_errors_have_the_same_mapping(route: Route) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectError("offline")

    client = _client(httpx.MockTransport(handler))
    try:
        with pytest.raises(MCPHTTPError) as caught:
            await route(client)
        assert caught.value.status_code == 0
        assert "MCP connection failed: offline" in str(caught.value)
    finally:
        await client.close()
