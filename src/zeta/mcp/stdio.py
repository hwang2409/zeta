"""Line-delimited JSON-RPC MCP transport."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable, Mapping
from typing import BinaryIO

from ..core.abort import AbortSignal
from ..tools._process import _kill_and_reap
from .client import (
    MCPCanceled,
    MCPClient,
    MCPError,
    MCPPrompt,
    MCPProtocolError,
    MCPTool,
    canceled_result,
    initialize_params,
    make_error_result,
    parse_rpc_response,
    prompt_text_from_result,
    prompts_from_result,
    tools_from_result,
    translate_call_result,
)
from .config import MCPServerConfig, mcp_log_path

logger = logging.getLogger(__name__)


class StdioMCPClient(MCPClient):
    def __init__(self, config: MCPServerConfig) -> None:
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
            child_env = os.environ.copy()
            child_env.update(self.config.env)
            self._process = await asyncio.create_subprocess_exec(
                self.config.command, *self.config.args,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log_handle, env=child_env, start_new_session=True,
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
                raise MCPProtocolError("MCP initialize response omitted protocolVersion")
            self.protocol_version = protocol_version
            capabilities = result.get("capabilities", {})
            self.capabilities = dict(capabilities) if type(capabilities) is dict else {}
            await self._notify("notifications/initialized", {})
        except BaseException:
            await self.close()
            raise

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

    async def list_prompts(self) -> list[MCPPrompt]:
        prompts: list[MCPPrompt] = []
        cursor: str | None = None
        while True:
            params: dict[str, object] = {}
            if cursor is not None:
                params["cursor"] = cursor
            result = await self._request("prompts/list", params)
            prompts.extend(prompts_from_result(result))
            next_cursor = result.get("nextCursor")
            if type(next_cursor) is not str or not next_cursor:
                return prompts
            if next_cursor == cursor:
                raise MCPProtocolError("MCP prompts/list cursor did not advance")
            cursor = next_cursor

    async def get_prompt(self, name: str, arguments: Mapping[str, str]) -> str:
        try:
            result = await self._request(
                "prompts/get", {"name": name, "arguments": dict(arguments)}
            )
            return prompt_text_from_result(result)
        except Exception as exc:
            self._report_failure(exc)
            raise

    async def call_tool(self, name: str, arguments: Mapping[str, object], abort_signal: AbortSignal):
        try:
            result = await self._request("tools/call", {"name": name, "arguments": dict(arguments)}, abort_signal)
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
            await _kill_and_reap(process, [reader_task] if reader_task is not None else [])
            await process.wait()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPError("MCP stdio server closed"))
        self._pending.clear()
        stderr = self._stderr
        self._stderr = None
        if stderr is not None:
            stderr.close()

    async def _request(self, method: str, params: Mapping[str, object], abort_signal: AbortSignal | None = None) -> dict[str, object]:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPError("MCP stdio server is not connected")
        if self._closed:
            raise MCPError("MCP stdio server is closed")
        self._next_id += 1
        request_id = self._next_id
        response = asyncio.get_running_loop().create_future()
        self._pending[request_id] = response
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)}
        try:
            async with self._write_lock:
                process.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
                await process.stdin.drain()
            if abort_signal is None:
                raw_response = await response
            else:
                abort_task = asyncio.create_task(abort_signal.wait())
                try:
                    done, _ = await asyncio.wait((response, abort_task), return_when=asyncio.FIRST_COMPLETED)
                    if abort_task in done and response not in done:
                        self._pending.pop(request_id, None)
                        try:
                            await asyncio.wait_for(
                                self._notify(
                                    "notifications/cancelled",
                                    {"requestId": request_id, "reason": "client canceled"},
                                ),
                                timeout=0.05,
                            )
                        except Exception:  # noqa: BLE001 - cancellation must continue to cleanup
                            logger.debug("MCP %s did not accept cancellation", self.config.name)
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
            process.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
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
        error = MCPError("MCP stdio server terminated after cancellation")
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

    async def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            async for line in process.stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._fail_pending(MCPProtocolError(f"invalid MCP JSON: {exc.msg}"))
                    continue
                if type(value) is not dict:
                    self._fail_pending(MCPProtocolError("MCP message must be an object"))
                    continue
                response_id = value.get("id")
                if type(response_id) is int and response_id in self._pending:
                    future = self._pending[response_id]
                    if not future.done():
                        future.set_result(value)
                elif type(value.get("method")) is str:
                    logger.debug("MCP %s notification: %s", self.config.name, value["method"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reader failure is transport failure
            self._fail_pending(MCPError(f"MCP stdio reader failed: {exc}"))
        finally:
            if process.returncode is None:
                try:
                    await process.wait()
                except asyncio.CancelledError:
                    return
            if self._process is process:
                error = MCPError(f"MCP stdio server exited with code {process.returncode}")
                self._fail_pending(error)
                if not self._suppress_failure and self._failure_sink is not None:
                    self._failure_sink(str(error))

    def _fail_pending(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)


__all__ = ["StdioMCPClient"]


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask the failure
        return type(error).__name__
