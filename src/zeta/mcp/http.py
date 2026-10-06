"""MCP streamable-http transport."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import BinaryIO, TypeVar

import httpx

from ..core.abort import AbortSignal
from ..tools._spill import SpillStore
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
    resource_content_from_result,
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
)
from .pagination import drain_pages
from .resources import RESOURCE_MAX_BYTES

logger = logging.getLogger(__name__)
HTTP_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 2 * RESOURCE_MAX_BYTES
MAX_LIST_ITEMS = 10_000
MAX_LIST_PAGES = 1_000
MAX_ERROR_DETAIL_BYTES = 8192
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

OAUTH_HINT = "run /mcp auth {name} to reauthorize"

T = TypeVar("T")


class StreamableHTTPMCPClient(MCPClient):
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        client: httpx.AsyncClient | None = None,
        home: str | None = None,
        spill_store: SpillStore | None = None,
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
        self._spill_store = spill_store or SpillStore()
        self._owns_spill_store = spill_store is None
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

    async def list_resources(
        self, abort_signal: AbortSignal | None = None
    ) -> list[MCPResource]:
        return await _drain_pages(
            self._request,
            "resources/list",
            resources_from_result,
            "resources/list",
            abort_signal=abort_signal,
        )

    async def read_resource(self, uri: str, abort_signal: AbortSignal | None = None):
        result = await self._request("resources/read", {"uri": uri}, abort_signal)
        return resource_content_from_result(result)

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

    async def call_tool(
        self, name: str, arguments: Mapping[str, object], abort_signal: AbortSignal
    ):
        try:
            result = await self._request(
                "tools/call", {"name": name, "arguments": dict(arguments)}, abort_signal
            )
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
        if self._owns_spill_store:
            self._spill_store.close()

    async def _request(
        self,
        method: str,
        params: Mapping[str, object],
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, object]:
        if self._closed:
            raise MCPHTTPError(0, "MCP HTTP client is closed")
        self._next_id += 1
        request_id = self._next_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        request_task = asyncio.create_task(self._send_with_auth(payload, request_id))
        if abort_signal is None:
            return await request_task
        abort_task = asyncio.create_task(abort_signal.wait())
        try:
            done, _ = await asyncio.wait(
                (request_task, abort_task), return_when=asyncio.FIRST_COMPLETED
            )
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
        return await self._with_auth(lambda: self._send(payload, request_id))

    async def _with_auth(self, send: Callable[[], Awaitable[T]]) -> T:
        if self.config.auth_type != "oauth":
            return await send()
        token_at_send = self._current_token
        try:
            return await send()
        except MCPHTTPError as exc:
            if exc.status_code != 401:
                raise
            refreshed = await self._refresh_token_once(exc, token_at_send)
            if not refreshed:
                raise self._auth_error(exc)
            return await send()

    async def _refresh_token_once(
        self, exc: MCPHTTPError, token_at_send: MCPOAuthToken | None
    ) -> bool:
        async with self._token_lock:
            token = self._current_token
            if token is not token_at_send:
                return True
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
                self._record_refresh_failure(
                    token, "refresh response missing access_token"
                )
                return False
            import time as _time

            expires_at: float | None = None
            expires_in = response.get("expires_in")
            if type(expires_in) in {int, float}:
                expires_at = _time.time() + float(expires_in)  # type: ignore[arg-type]
            refresh_value = response.get("refresh_token")
            new_refresh = (
                refresh_value
                if type(refresh_value) is str and refresh_value
                else token.refresh_token
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
                token_type=token_type_value
                if type(token_type_value) is str
                else token.token_type,
                refresh_error=None,
            )
            save_token(self.config.name, new_token, home=self._home)
            self._current_token = new_token
            return True

    def _record_refresh_failure(self, token: MCPOAuthToken | None, reason: str) -> None:
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

    async def _send(
        self, payload: Mapping[str, object], request_id: int
    ) -> dict[str, object]:
        headers = self._auth_headers()
        headers.update(
            {
                "accept": "application/json, text/event-stream",
                "content-type": "application/json",
            }
        )
        if self._session_id is not None:
            headers["mcp-session-id"] = self._session_id
        if self.protocol_version is not None:
            headers["mcp-protocol-version"] = self.protocol_version
        try:
            async with self._client.stream(
                "POST", self.config.url, headers=headers, json=dict(payload)
            ) as response:
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
                    return await _read_sse_response(
                        response,
                        request_id,
                        self._spill_store,
                        MAX_RESPONSE_BYTES,
                    )
                try:
                    value = await _read_json_body(
                        response, self._spill_store, MAX_RESPONSE_BYTES
                    )
                except ValueError as exc:
                    raise MCPProtocolError("MCP HTTP response was not JSON") from exc
                return parse_rpc_response(value, request_id)
        except httpx.HTTPError as exc:
            raise MCPHTTPError(0, str(exc)) from exc

    async def _send_notification(
        self, method: str, params: Mapping[str, object]
    ) -> None:
        await self._with_auth(lambda: self._send_notification_raw(method, params))

    async def _send_notification_raw(
        self, method: str, params: Mapping[str, object]
    ) -> None:
        headers = self._auth_headers()
        headers.update(
            {
                "accept": "application/json, text/event-stream",
                "content-type": "application/json",
            }
        )
        if self._session_id is not None:
            headers["mcp-session-id"] = self._session_id
        if self.protocol_version is not None:
            headers["mcp-protocol-version"] = self.protocol_version
        payload = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        try:
            async with self._client.stream(
                "POST", self.config.url, headers=headers, json=payload
            ) as response:
                if response.status_code == 401:
                    detail = await _response_detail(response)
                    raise MCPHTTPError(401, detail or "unauthorized")
                if response.status_code >= 400:
                    detail = await _response_detail(response)
                    raise MCPHTTPError(response.status_code, detail or "request failed")
        except httpx.HTTPError as exc:
            raise MCPHTTPError(0, str(exc)) from exc

    def _auth_headers(self) -> dict[str, str]:
        def resolve(value: str) -> str:
            def replace(match: re.Match[str]) -> str:
                variable = match.group(1)
                resolved = os.environ.get(variable)
                if resolved is None:
                    raise MCPHTTPError(
                        0, f"missing environment variable for MCP header: {variable}"
                    )
                return resolved

            return _ENV_PATTERN.sub(replace, value)

        headers = {
            name: resolve(value) for name, value in self.config.headers.items()
        }
        if self.config.auth_type == "bearer" and self.config.auth_token is not None:
            headers["authorization"] = f"Bearer {self.config.auth_token}"
        elif self.config.auth_type == "oauth" and self._current_token is not None:
            token = self._current_token
            headers["authorization"] = f"{token.token_type} {token.access_token}"
        return headers


async def _drain_pages(
    request, method: str, parser, label: str,
    *, abort_signal: AbortSignal | None = None,
) -> list:
    if abort_signal is not None:
        request = partial(request, abort_signal=abort_signal)
    return await drain_pages(
        request, method, parser, label,
        max_pages=MAX_LIST_PAGES, max_items=MAX_LIST_ITEMS,
    )


async def _read_json_body(
    response: httpx.Response,
    spill_store: SpillStore,
    memory_bound: int,
) -> object:
    """Stream JSON, moving oversized wire bytes to a private spill file."""

    chunks: list[bytes] = []
    total = 0
    temporary = None
    handle: BinaryIO | None = None
    try:
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if handle is None and total <= memory_bound:
                chunks.append(chunk)
                continue
            if handle is None:
                temporary = spill_store.temporary_file()
                handle = await asyncio.to_thread(temporary.__enter__)
                buffered = chunks
                chunks = []
                await asyncio.to_thread(handle.writelines, buffered)
            await asyncio.to_thread(handle.write, chunk)
        if handle is None:
            return json.loads(b"".join(chunks))
        await asyncio.to_thread(handle.seek, 0)
        return await asyncio.to_thread(json.load, handle)
    finally:
        if temporary is not None:
            await asyncio.to_thread(temporary.__exit__, None, None, None)


async def _response_detail(response: httpx.Response) -> str:
    body = await _read_capped_bytes(response, MAX_ERROR_DETAIL_BYTES)
    try:
        value = json.loads(body)
    except ValueError:
        return body.decode(errors="replace")[:1024]
    if type(value) is dict:
        error = value.get("error")
        if type(error) is dict and type(error.get("message")) is str:
            return error["message"]
        message = value.get("message")
        if type(message) is str:
            return message
    return str(value)


async def _read_capped_bytes(response: httpx.Response, cap: int) -> bytes:
    """Read at most ``cap`` bytes from a response, discarding the rest."""

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        remaining = cap - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            total = cap
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


async def _read_sse_response(
    response: httpx.Response,
    request_id: int,
    spill_store: SpillStore,
    memory_bound: int,
) -> dict[str, object]:
    """Parse SSE incrementally without materializing complete data lines."""

    data_parts: list[bytes] = []
    total = 0
    has_data_line = False
    temporary = None
    handle: BinaryIO | None = None

    async def append_data(data: bytes, *, start_line: bool = False) -> None:
        nonlocal temporary, handle, total, data_parts, has_data_line
        separator = b"\n" if start_line and has_data_line else b""
        if start_line:
            has_data_line = True
        total += len(separator) + len(data)
        if handle is None and total <= memory_bound:
            if separator:
                data_parts.append(separator)
            if data:
                data_parts.append(data)
            return
        if handle is None:
            temporary = spill_store.temporary_file()
            handle = await asyncio.to_thread(temporary.__enter__)
            initial = b"".join(data_parts)
            data_parts = []
            if initial:
                await asyncio.to_thread(handle.write, initial)
        if separator:
            await asyncio.to_thread(handle.write, separator)
        if data:
            await asyncio.to_thread(handle.write, data)

    async def parse_event() -> object:
        nonlocal temporary, handle, total, data_parts, has_data_line
        if handle is None:
            value = json.loads(b"".join(data_parts))
        else:
            await asyncio.to_thread(handle.seek, 0)
            value = await asyncio.to_thread(json.load, handle)
            await asyncio.to_thread(temporary.__exit__, None, None, None)
        data_parts = []
        total = 0
        has_data_line = False
        temporary = None
        handle = None
        return value

    async def finish_event() -> dict[str, object] | None:
        if not has_data_line:
            return None
        value = await parse_event()
        if type(value) is dict and value.get("id") == request_id:
            return parse_rpc_response(value, request_id)
        if type(value) is dict and type(value.get("method")) is str:
            logger.debug("MCP stream notification: %s", value["method"])
        return None

    line_prefix = bytearray()
    line_is_data = False
    line_ignored = False
    stripping_whitespace = True

    async def feed_data(data: bytes) -> None:
        nonlocal stripping_whitespace
        if stripping_whitespace:
            offset = 0
            while offset < len(data) and chr(data[offset]).isspace():
                offset += 1
            data = data[offset:]
            if data:
                stripping_whitespace = False
        await append_data(data)

    async def feed_line(data: bytes, *, end_line: bool) -> bool:
        nonlocal line_is_data, line_ignored, stripping_whitespace
        if not line_is_data and not line_ignored:
            needed = 5 - len(line_prefix)
            line_prefix.extend(data[:needed])
            data = data[needed:]
            if len(line_prefix) == 5:
                if line_prefix == b"data:":
                    line_is_data = True
                    stripping_whitespace = True
                    await append_data(b"", start_line=True)
                else:
                    line_ignored = True
        if line_is_data:
            await feed_data(data)
        if not end_line:
            return False
        return not line_is_data and not line_ignored and not line_prefix

    def reset_line() -> None:
        nonlocal line_is_data, line_ignored, stripping_whitespace
        line_prefix.clear()
        line_is_data = False
        line_ignored = False
        stripping_whitespace = True

    pending_cr = False

    try:
        chunks = (
            response.aiter_bytes(chunk_size=65_536)
            if response.is_stream_consumed
            else response.aiter_raw()
        )
        async for raw_chunk in chunks:
            for offset in range(0, len(raw_chunk), 65_536):
                window = raw_chunk[offset : offset + 65_536]
                position = 0
                if pending_cr:
                    pending_cr = False
                    if window.startswith(b"\n"):
                        position = 1
                while position < len(window):
                    cr = window.find(b"\r", position)
                    lf = window.find(b"\n", position)
                    delimiters = [index for index in (cr, lf) if index >= 0]
                    if not delimiters:
                        await feed_line(window[position:], end_line=False)
                        break
                    delimiter = min(delimiters)
                    blank = await feed_line(window[position:delimiter], end_line=True)
                    reset_line()
                    position = delimiter + 1
                    if window[delimiter] == 13:
                        if position < len(window) and window[position] == 10:
                            position += 1
                        elif position == len(window):
                            pending_cr = True
                    if blank:
                        result = await finish_event()
                        if result is not None:
                            return result
        if line_prefix or line_is_data or line_ignored:
            await feed_line(b"", end_line=True)
        if has_data_line:
            result = await finish_event()
            if result is not None:
                return result
        raise MCPProtocolError("MCP SSE response ended before the result")
    finally:
        if temporary is not None:
            await asyncio.to_thread(temporary.__exit__, None, None, None)


__all__ = ["StreamableHTTPMCPClient"]
