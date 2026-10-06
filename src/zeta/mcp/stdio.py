"""Line-delimited JSON-RPC MCP transport."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from functools import partial
from typing import BinaryIO

from ..core.abort import AbortSignal
from ..core.process_env import subprocess_env
from ..tools._shared.process import _kill_and_reap
from ..tools._spill import SpillStore

MAX_LIST_ITEMS = 10_000
MAX_LIST_PAGES = 1_000

from .client import (
    MCPCanceled,
    MCPClient,
    MCPError,
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
from .config import MCPServerConfig, mcp_log_path
from .pagination import drain_pages

logger = logging.getLogger(__name__)

MCP_STDIO_LINE_LIMIT = 16 * 1024 * 1024


class StdioMCPClient(MCPClient):
    def __init__(
        self, config: MCPServerConfig, *, spill_store: SpillStore | None = None
    ) -> None:
        self.config = config
        self.protocol_version: str | None = None
        self.capabilities: dict[str, object] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._stderr: BinaryIO | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, object]]] = {}
        self._closed = False
        self._suppress_failure = False
        self._failure_sink: Callable[[str], None] | None = None
        self._spill_store = spill_store or SpillStore()
        self._owns_spill_store = spill_store is None

    def set_failure_sink(self, sink: Callable[[str], None] | None) -> None:
        """Set a callback for unexpected transport termination."""

        self._failure_sink = sink

    def _report_failure(self, error: BaseException) -> None:
        if self._failure_sink is not None and not self._suppress_failure:
            self._failure_sink(_error_text(error))

    async def connect(self) -> None:
        if self._process is not None:
            return
        log_path = mcp_log_path(self.config.name)
        log_root = log_path.parent
        log_root.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log_handle,
                env=subprocess_env(self.config.env),
                limit=MCP_STDIO_LINE_LIMIT,
                start_new_session=True,
            )
        except BaseException:
            log_handle.close()
            raise
        self._stderr = log_handle
        self._reader_task = asyncio.create_task(self._read_stdout())
        try:
            result = await self._request("initialize", initialize_params())
            protocol_version = result.get("protocolVersion")
            if type(protocol_version) is not str or not protocol_version:
                raise MCPProtocolError(
                    "MCP initialize response omitted protocolVersion"
                )
            self.protocol_version = protocol_version
            capabilities = result.get("capabilities", {})
            self.capabilities = dict(capabilities) if type(capabilities) is dict else {}
            await self._notify("notifications/initialized", {})
        except BaseException:
            await self.close()
            raise

    async def _list_pages(
        self, method: str, parser, label: str,
        *, abort_signal: AbortSignal | None = None,
    ) -> list:
        request = self._request
        if abort_signal is not None:
            request = partial(self._request, abort_signal=abort_signal)
        return await drain_pages(
            request, method, parser, label,
            max_pages=MAX_LIST_PAGES, max_items=MAX_LIST_ITEMS,
        )

    async def list_tools(self) -> list[MCPTool]:
        return await self._list_pages("tools/list", tools_from_result, "tools/list")

    async def list_prompts(self) -> list[MCPPrompt]:
        return await self._list_pages(
            "prompts/list", prompts_from_result, "prompts/list"
        )

    async def list_resources(
        self, abort_signal: AbortSignal | None = None
    ) -> list[MCPResource]:
        return await self._list_pages(
            "resources/list", resources_from_result, "resources/list",
            abort_signal=abort_signal,
        )

    async def read_resource(self, uri: str, abort_signal: AbortSignal | None = None):
        result = await self._request(
            "resources/read", {"uri": uri}, abort_signal
        )
        return resource_content_from_result(result)

    async def get_prompt(self, name: str, arguments: Mapping[str, str]) -> str:
        try:
            result = await self._request(
                "prompts/get", {"name": name, "arguments": dict(arguments)}
            )
            return prompt_text_from_result(result)
        except MCPTransportError as exc:
            self._report_failure(exc)
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
            self._report_failure(exc)
            return make_error_result(str(exc))
        except Exception as exc:  # noqa: BLE001 - remote failures become tool results
            self._report_failure(exc)
            return make_error_result(str(exc))
        return translate_call_result(result)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._suppress_failure = True
        process = self._process
        reader_task = self._reader_task
        self._process = None
        self._reader_task = None
        if process is not None:
            await _kill_and_reap(
                process, [reader_task] if reader_task is not None else []
            )
            await process.wait()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPTransportError("MCP stdio server closed"))
        self._pending.clear()
        stderr = self._stderr
        self._stderr = None
        if stderr is not None:
            stderr.close()
        if self._owns_spill_store:
            self._spill_store.close()

    async def _request(
        self,
        method: str,
        params: Mapping[str, object],
        abort_signal: AbortSignal | None = None,
    ) -> dict[str, object]:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPTransportError("MCP stdio server is not connected")
        if self._closed:
            raise MCPTransportError("MCP stdio server is closed")
        self._next_id += 1
        request_id = self._next_id
        response = asyncio.get_running_loop().create_future()
        self._pending[request_id] = response
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        try:
            async with self._write_lock:
                process.stdin.write(
                    (json.dumps(request, separators=(",", ":")) + "\n").encode()
                )
                await process.stdin.drain()
            if abort_signal is None:
                raw_response = await response
            else:
                abort_task = asyncio.create_task(abort_signal.wait())
                try:
                    done, _ = await asyncio.wait(
                        (response, abort_task), return_when=asyncio.FIRST_COMPLETED
                    )
                    if abort_task in done and response not in done:
                        self._pending.pop(request_id, None)
                        try:
                            await asyncio.wait_for(
                                self._notify(
                                    "notifications/cancelled",
                                    {
                                        "requestId": request_id,
                                        "reason": "client canceled",
                                    },
                                ),
                                timeout=0.05,
                            )
                        except Exception:  # noqa: BLE001 - cancellation must continue to cleanup
                            logger.debug(
                                "MCP %s did not accept cancellation", self.config.name
                            )
                        await self._terminate_process(report_failure=True)
                        raise MCPCanceled()
                    raw_response = await response
                finally:
                    if not abort_task.done():
                        abort_task.cancel()
                    await asyncio.gather(abort_task, return_exceptions=True)
            return parse_rpc_response(raw_response, request_id)
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            if abort_signal is not None and abort_signal.is_set():
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
                await self._terminate_process(report_failure=True)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: Mapping[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or self._closed:
            return
        message = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        async with self._write_lock:
            process.stdin.write(
                (json.dumps(message, separators=(",", ":")) + "\n").encode()
            )
            await process.stdin.drain()

    async def _terminate_process(self, *, report_failure: bool = False) -> None:
        process = self._process
        reader_task = self._reader_task
        if process is None:
            return
        self._suppress_failure = True
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=0.25)
        except TimeoutError:
            pass
        await _kill_and_reap(
            process,
            [reader_task] if reader_task is not None else [],
        )
        await process.wait()
        if reader_task is not None and not reader_task.done():
            await asyncio.gather(reader_task, return_exceptions=True)
        error = MCPTransportError("MCP stdio server terminated after cancellation")
        self._fail_pending(error)
        self._process = None
        self._reader_task = None
        stderr = self._stderr
        self._stderr = None
        if stderr is not None:
            stderr.close()
        if report_failure:
            self._suppress_failure = False
            self._report_failure(error)

    async def _read_messages(
        self, stream: asyncio.StreamReader
    ) -> AsyncIterator[object]:
        """Yield framed JSON values without retaining oversized wire lines."""

        buffered = bytearray()
        temporary = None
        handle: BinaryIO | None = None

        async def spill(data: bytes) -> None:
            nonlocal temporary, handle, buffered
            if handle is None:
                temporary = self._spill_store.temporary_file()
                handle = await asyncio.to_thread(temporary.__enter__)
                initial = bytes(buffered)
                buffered.clear()
                if initial:
                    await asyncio.to_thread(handle.write, initial)
            if data:
                await asyncio.to_thread(handle.write, data)

        async def parse_message() -> object:
            nonlocal temporary, handle
            if handle is None:
                value = json.loads(buffered)
                buffered.clear()
                return value
            await asyncio.to_thread(handle.seek, 0)
            try:
                return await asyncio.to_thread(json.load, handle)
            finally:
                await asyncio.to_thread(temporary.__exit__, None, None, None)
                temporary = None
                handle = None

        try:
            while chunk := await stream.read(65_536):
                remaining = chunk
                while True:
                    separator = remaining.find(b"\n")
                    part = remaining if separator < 0 else remaining[:separator]
                    if handle is not None or len(buffered) + len(part) > MCP_STDIO_LINE_LIMIT:
                        await spill(part)
                    else:
                        buffered.extend(part)
                    if separator < 0:
                        break
                    yield await parse_message()
                    remaining = remaining[separator + 1 :]
            if handle is not None or buffered:
                yield await parse_message()
        finally:
            if temporary is not None:
                await asyncio.to_thread(temporary.__exit__, None, None, None)

    async def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            async for value in self._read_messages(process.stdout):
                if type(value) is not dict:
                    self._fail_pending(MCPProtocolError("MCP message must be an object"))
                    continue
                response_id = value.get("id")
                if type(response_id) is int and response_id in self._pending:
                    future = self._pending[response_id]
                    if not future.done():
                        future.set_result(value)
                elif type(value.get("method")) is str:
                    logger.debug(
                        "MCP %s notification: %s", self.config.name, value["method"]
                    )
        except asyncio.CancelledError:
            raise
        except json.JSONDecodeError as exc:
            self._fail_pending(MCPProtocolError(f"invalid MCP JSON: {exc.msg}"))
        except Exception as exc:  # noqa: BLE001 - reader failure is transport failure
            self._fail_pending(MCPTransportError(f"MCP stdio reader failed: {exc}"))
        finally:
            wait_cancelled = False
            if process.returncode is None:
                try:
                    await process.wait()
                except asyncio.CancelledError:
                    wait_cancelled = True
            if not wait_cancelled and self._process is process:
                error = MCPTransportError(
                    f"MCP stdio server exited with code {process.returncode}"
                )
                self._fail_pending(error)
                if not self._suppress_failure and self._failure_sink is not None:
                    self._failure_sink(str(error))

    def _fail_pending(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)


__all__ = ["MCP_STDIO_LINE_LIMIT", "StdioMCPClient"]


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask the failure
        return type(error).__name__
