"""MCP streamable-http transport."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace

import httpx

from ..core.abort import AbortSignal
from .client import (
    MCPCanceled,
    MCPClient,
    MCPError,
    MCPHTTPError,
    MCPPrompt,
    MCPProtocolError,
    MCPResource,
    MCPTool,
    MCPTransportError,
    canceled_result,
    initialize_params,
    make_error_result,
    parse_rpc_response,
    prompt_text_from_result,
    prompts_from_result,
    resource_text_from_result,
    resources_from_result,
    tools_from_result,
    translate_call_result,
)
from .config import MCPServerConfig
from .oauth import (
    MCPOAuthError,
    refresh_access_token,
)
from .oauth_store import (
    MCPOAuthToken,
    load_token,
    save_token,
    token_is_expired,
)

logger = logging.getLogger(__name__)
HTTP_TIMEOUT_SECONDS = 10.0

OAUTH_HINT = "run /mcp auth {name} to reauthorize"


class StreamableHTTPMCPClient(MCPClient):
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        client: httpx.AsyncClient | None = None,
        home: str | None = None,
    ) -> None:
        self.config = config
        self._client = client or httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)
        self._owns_client = client is None
        self._session_id: str | None = None
        self.protocol_version: str | None = None
        self.capabilities: dict[str, object] = {}
        self._next_id = 0
        self._closed = False
        self._failure_sink: Callable[[str], None] | None = None
        self._home = home
        self._token_lock = asyncio.Lock()
        self._current_token: MCPOAuthToken | None = None
        if config.auth_type == "oauth":
            self._current_token = load_token(config.name, home=home)

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
        capabilities = result.get("capabilities", {})
        self.capabilities = dict(capabilities) if type(capabilities) is dict else {}
        await self._send_notification("notifications/initialized", {})

    async def list_tools(self) -> list[MCPTool]:
        return await _drain_pages(
            self._request, "tools/list", tools_from_result, "tools/list"
        )

    async def list_prompts(self) -> list[MCPPrompt]:
        return await _drain_pages(
            self._request, "prompts/list", prompts_from_result, "prompts/list"
        )

    async def list_resources(self) -> list[MCPResource]:
        return await _drain_pages(
            self._request,
            "resources/list",
            resources_from_result,
            "resources/list",
        )

    async def read_resource(self, uri: str) -> str:
        result = await self._request("resources/read", {"uri": uri})
        return resource_text_from_result(result)

    async def get_prompt(self, name: str, arguments: Mapping[str, str]) -> str:
        try:
            result = await self._request(
                "prompts/get", {"name": name, "arguments": dict(arguments)}
            )
            return prompt_text_from_result(result)
        except MCPTransportError as exc:
            if self._failure_sink is not None:
                self._failure_sink(str(exc))
            raise

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
            if self._failure_sink is not None:
                self._failure_sink(str(exc))
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
        request_task = asyncio.create_task(self._send_with_auth(payload, request_id))
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

    async def _send_with_auth(
        self, payload: Mapping[str, object], request_id: int
    ) -> dict[str, object]:
        if self.config.auth_type != "oauth":
            return await self._send(payload, request_id)
        try:
            return await self._send(payload, request_id)
        except MCPHTTPError as exc:
            if exc.status_code != 401:
                raise
            refreshed = await self._refresh_token_once(exc)
            if not refreshed:
                raise self._auth_error(exc)
            return await self._send(payload, request_id)

    async def _refresh_token_once(self, exc: MCPHTTPError) -> bool:
        async with self._token_lock:
            token = self._current_token
            if token is None or token.refresh_token is None:
                self._record_refresh_failure(token, str(exc))
                return False
            try:
                response = await refresh_access_token(
                    token_endpoint=token.token_endpoint,
                    refresh_token=token.refresh_token,
                    client_id=token.client_id,
                    client_secret=token.client_secret,
                    resource=token.resource,
                    http_client=self._client,
                )
            except MCPOAuthError as refresh_exc:
                self._record_refresh_failure(token, str(refresh_exc))
                return False
            access = response.get("access_token")
            if type(access) is not str or not access:
                self._record_refresh_failure(token, "refresh response missing access_token")
                return False
            import time as _time
            expires_at: float | None = None
            expires_in = response.get("expires_in")
            if type(expires_in) in {int, float}:
                expires_at = _time.time() + float(expires_in)  # type: ignore[arg-type]
            refresh_value = response.get("refresh_token")
            new_refresh = (
                refresh_value if type(refresh_value) is str and refresh_value else token.refresh_token
            )
            scope_value = response.get("scope")
            new_scope = scope_value if type(scope_value) is str else token.scope
            token_type_value = response.get("token_type") or token.token_type
            new_token = replace(
                token,
                access_token=access,
                refresh_token=new_refresh,
                expires_at=expires_at,
                scope=new_scope,
                token_type=token_type_value if type(token_type_value) is str else token.token_type,
                refresh_error=None,
            )
            save_token(self.config.name, new_token, home=self._home)
            self._current_token = new_token
            return True

    def _record_refresh_failure(
        self, token: MCPOAuthToken | None, reason: str
    ) -> None:
        if token is None:
            return
        marked = replace(token, refresh_error=reason)
        try:
            save_token(self.config.name, marked, home=self._home)
        except OSError:
            logger.warning("could not persist refresh failure for %s", self.config.name)
        self._current_token = marked

    def _auth_error(self, exc: MCPHTTPError) -> MCPHTTPError:
        hint = OAUTH_HINT.format(name=self.config.name)
        return MCPHTTPError(
            401,
            f"MCP OAuth token refresh failed ({exc}); {hint}",
        )

    async def _send(self, payload: Mapping[str, object], request_id: int) -> dict[str, object]:
        headers = self._auth_headers()
        headers.update(
            {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        )
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
        headers = self._auth_headers()
        headers.update(
            {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        )
        if self._session_id is not None:
            headers["mcp-session-id"] = self._session_id
        if self.protocol_version is not None:
            headers["mcp-protocol-version"] = self.protocol_version
        payload = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        async with self._client.stream("POST", self.config.url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                detail = await _response_detail(response)
                raise MCPHTTPError(response.status_code, detail or "request failed")

    def _auth_headers(self) -> dict[str, str]:
        if self.config.auth_type == "bearer" and self.config.auth_token is not None:
            return {"authorization": f"Bearer {self.config.auth_token}"}
        if self.config.auth_type == "oauth" and self._current_token is not None:
            token = self._current_token
            return {"authorization": f"{token.token_type} {token.access_token}"}
        return {}


async def _drain_pages(request, method: str, parser, label: str) -> list:
    """Follow the cursor chain on a `<thing>/list` MCP method until it ends."""

    items: list = []
    cursor: str | None = None
    while True:
        params: dict[str, object] = {}
        if cursor is not None:
            params["cursor"] = cursor
        result = await request(method, params)
        items.extend(parser(result))
        next_cursor = result.get("nextCursor")
        if type(next_cursor) is not str or not next_cursor:
            return items
        if next_cursor == cursor:
            raise MCPProtocolError(f"MCP {label} cursor did not advance")
        cursor = next_cursor


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
