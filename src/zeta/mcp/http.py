"""MCP streamable-http transport."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping

import httpx

from ..core.abort import AbortSignal
from .client import (
    MCPCanceled,
    MCPClient,
    MCPError,
    MCPHTTPError,
    MCPProtocolError,
    MCPTool,
    canceled_result,
    initialize_params,
    make_error_result,
    parse_rpc_response,
    tools_from_result,
    translate_call_result,
)
from .config import MCPServerConfig

logger = logging.getLogger(__name__)
HTTP_TIMEOUT_SECONDS = 10.0


class StreamableHTTPMCPClient(MCPClient):
    def __init__(self, config: MCPServerConfig, *, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        self._owns_client = client is None
        self._session_id: str | None = None
        self.protocol_version: str | None = None
        self._next_id = 0
        self._closed = False
        self._failure_sink: Callable[[str], None] | None = None

    def set_failure_sink(self, sink: Callable[[str], None] | None) -> None:
        self._failure_sink = sink

    async def connect(self) -> None:
        if self._closed:
            raise MCPHTTPError(0, "MCP HTTP client is closed")
        result = await self._request("initialize", initialize_params())
        protocol_version = result.get("protocolVersion")
        if type(protocol_version) is not str or not protocol_version:
            raise MCPProtocolError("MCP initialize response omitted protocolVersion")
        self.protocol_version = protocol_version
        await self._send_notification("notifications/initialized", {})

    async def list_tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        cursor: str | None = None
        while True:
            params: dict[str, object] = {}
            if cursor is not None:
                params["cursor"] = cursor
            result = await self._request("tools/list", params)
            tools.extend(tools_from_result(result))
            next_cursor = result.get("nextCursor")
            if type(next_cursor) is not str or not next_cursor:
                return tools
            if next_cursor == cursor:
                raise MCPProtocolError("MCP tools/list cursor did not advance")
            cursor = next_cursor

    async def call_tool(self, name: str, arguments: Mapping[str, object], abort_signal: AbortSignal):
        try:
            result = await self._request("tools/call", {"name": name, "arguments": dict(arguments)}, abort_signal)
        except MCPCanceled:
            return canceled_result()
        except MCPError as exc:
            if self._failure_sink is not None:
                self._failure_sink(str(exc))
            return make_error_result(str(exc))
        except Exception as exc:  # noqa: BLE001 - remote failures become tool results
            return make_error_result(str(exc))
        return translate_call_result(result)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def _request(self, method: str, params: Mapping[str, object], abort_signal: AbortSignal | None = None) -> dict[str, object]:
        if self._closed:
            raise MCPHTTPError(0, "MCP HTTP client is closed")
        self._next_id += 1
        request_id = self._next_id
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
        request_task = asyncio.create_task(self._send(payload, request_id))
        if abort_signal is None:
            return await request_task
        abort_task = asyncio.create_task(abort_signal.wait())
        try:
            done, _ = await asyncio.wait((request_task, abort_task), return_when=asyncio.FIRST_COMPLETED)
            if abort_task in done and request_task not in done:
                request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
                raise MCPCanceled()
            return await request_task
        finally:
            if not abort_task.done():
                abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)

    async def _send(self, payload: Mapping[str, object], request_id: int) -> dict[str, object]:
        headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        if self.config.auth_type == "bearer" and self.config.auth_token is not None:
            headers["authorization"] = f"Bearer {self.config.auth_token}"
        if self._session_id is not None:
            headers["mcp-session-id"] = self._session_id
        if self.protocol_version is not None:
            headers["mcp-protocol-version"] = self.protocol_version
        try:
            async with self._client.stream("POST", self.config.url, headers=headers, json=dict(payload)) as response:
                session_id = response.headers.get("mcp-session-id")
                if session_id is not None:
                    self._session_id = session_id
                if response.status_code == 401:
                    detail = await _response_detail(response)
                    raise MCPHTTPError(401, detail or "unauthorized")
                if response.status_code >= 400:
                    detail = await _response_detail(response)
                    raise MCPHTTPError(response.status_code, detail or "request failed")
                if "text/event-stream" in response.headers.get("content-type", ""):
                    return await _read_sse_response(response, request_id)
                try:
                    value = response.json()
                except ValueError as exc:
                    raise MCPProtocolError("MCP HTTP response was not JSON") from exc
                return parse_rpc_response(value, request_id)
        except httpx.HTTPError as exc:
            raise MCPHTTPError(0, str(exc)) from exc

    async def _send_notification(self, method: str, params: Mapping[str, object]) -> None:
        headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        if self.config.auth_type == "bearer" and self.config.auth_token is not None:
            headers["authorization"] = f"Bearer {self.config.auth_token}"
        if self._session_id is not None:
            headers["mcp-session-id"] = self._session_id
        if self.protocol_version is not None:
            headers["mcp-protocol-version"] = self.protocol_version
        payload = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        async with self._client.stream("POST", self.config.url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                detail = await _response_detail(response)
                raise MCPHTTPError(response.status_code, detail or "request failed")


async def _response_detail(response: httpx.Response) -> str:
    try:
        value = response.json()
    except ValueError:
        return (await response.aread()).decode(errors="replace")[:1024]
    if type(value) is dict:
        error = value.get("error")
        if type(error) is dict and type(error.get("message")) is str:
            return error["message"]
        message = value.get("message")
        if type(message) is str:
            return message
    return str(value)


async def _read_sse_response(response: httpx.Response, request_id: int) -> dict[str, object]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line or not data_lines:
            continue
        value = json.loads("\n".join(data_lines))
        data_lines.clear()
        if type(value) is dict and value.get("id") == request_id:
            return parse_rpc_response(value, request_id)
        if type(value) is dict and type(value.get("method")) is str:
            logger.debug("MCP stream notification: %s", value["method"])
    if data_lines:
        value = json.loads("\n".join(data_lines))
        return parse_rpc_response(value, request_id)
    raise MCPProtocolError("MCP SSE response ended before the result")


__all__ = ["StreamableHTTPMCPClient"]
