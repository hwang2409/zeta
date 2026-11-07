"""Line-delimited JSON-RPC MCP transport."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import BinaryIO, Literal
from weakref import WeakKeyDictionary

from ..core.abort import AbortSignal
from ..core.process_env import subprocess_env
from ..protocol.types import StructuredToolResult
from ..tools._shared.background_spawn import _kill_and_reap
from ..tools._spill import SpillStore
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
from .envelope import (
    MCPCallEnvelope,
    MCPCallLimits,
    MCPTransientQuotaError,
    MCPWireLimitError,
    _format_wire_limit,
)
from .pagination import drain_pages

logger = logging.getLogger(__name__)

MAX_LIST_ITEMS = 10_000
MAX_LIST_PAGES = 1_000
MCP_STDIO_LINE_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _MalformedMessage:
    error: MCPProtocolError


@dataclass(slots=True)
class _Frame:
    total_bytes: int = 0
    bytes_by_pending_id: dict[int, int] = field(default_factory=dict)
    value: object | None = None

    def record(self, size: int, pending_ids: Iterable[int]) -> None:
        self.total_bytes += size
        for request_id in pending_ids:
            self.bytes_by_pending_id[request_id] = (
                self.bytes_by_pending_id.get(request_id, 0) + size
            )


@dataclass(slots=True)
class _PendingCall:
    future: asyncio.Future[dict[str, object]]
    owner: asyncio.Task[object] | None
    tool_name: str | None
    wire_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _FrameInvalidation:
    limit_ids: set[int] | None = None
    quota_ids: set[int] | None = None


async def _append_frame_part(
    envelope: MCPCallEnvelope,
    frame: _Frame,
    part: bytes,
    pending_ids: Iterable[int],
) -> _Frame | _FrameInvalidation:
    frame.record(len(part), pending_ids)
    try:
        await envelope.append(part)
    except MCPWireLimitError:
        return _FrameInvalidation(limit_ids=set(frame.bytes_by_pending_id))
    except MCPTransientQuotaError:
        return _FrameInvalidation(quota_ids=set(frame.bytes_by_pending_id))
    return frame


_Outcome = Literal["limit", "quota", "restart", "canceled", "transport", "closed"]


@dataclass(slots=True)
class _Invalidation:
    generation: int
    process: asyncio.subprocess.Process | None
    reader_task: asyncio.Task[None] | None
    stderr: BinaryIO | None
    accumulator: MCPCallEnvelope | None
    pending: dict[int, _PendingCall]
    outcomes: dict[int, _Outcome]
    report_failure: bool
    reason: MCPTransportError
    initiator: asyncio.Task[object] | None
    task: asyncio.Task[None] | None = None


class _StdioCallLimit(MCPTransportError):
    pass


class _StdioSpillQuota(MCPTransportError):
    pass


class _StdioRestart(MCPTransportError):
    pass


class StdioMCPClient(MCPClient):
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        spill_store: SpillStore | None = None,
        limits: MCPCallLimits | None = None,
    ) -> None:
        self.config = config
        self.protocol_version: str | None = None
        self.capabilities: dict[str, object] = {}
        self._process: asyncio.subprocess.Process | None = None
        self._stderr: BinaryIO | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, _PendingCall] = {}
        self._reader_error: BaseException | None = None
        self._closed = False
        self._failure_sink: Callable[[str], None] | None = None
        self._notification_sink: Callable[[str], None] | None = None
        self._spill_store = spill_store or SpillStore()
        self._owns_spill_store = spill_store is None
        self._limits = limits or MCPCallLimits()
        if spill_store is None or limits is not None:
            self._spill_store.configure_transient_limit(self._limits.max_wire_bytes)
        self._generation = 0
        self._active_accumulator: MCPCallEnvelope | None = None
        self._invalidation: _Invalidation | None = None
        self._invalidation_results: WeakKeyDictionary[
            asyncio.Task[object], StructuredToolResult
        ] = WeakKeyDictionary()

    def set_failure_sink(self, sink: Callable[[str], None] | None) -> None:
        self._failure_sink = sink

    def set_notification_sink(self, sink: Callable[[str], None] | None) -> None:
        self._notification_sink = sink

    def _report_failure(self, error: BaseException) -> None:
        if self._failure_sink is not None:
            self._failure_sink(_error_text(error))

    def take_invalidation_result(
        self, task: asyncio.Task[object]
    ) -> StructuredToolResult | None:
        """Return the result fixed before shared transport cleanup began."""

        return self._invalidation_results.pop(task, None)

    async def connect(self) -> None:
        if self._process is not None:
            return
        if self._closed:
            raise MCPTransportError("MCP stdio server is closed")
        invalidation = self._invalidation
        if invalidation is not None and invalidation.task is not None:
            await asyncio.shield(invalidation.task)
            if self._invalidation is invalidation:
                self._invalidation = None
        self._reader_error = None
        log_path = mcp_log_path(self.config.name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        try:
            process = await asyncio.create_subprocess_exec(
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
        self._process = process
        self._stderr = log_handle
        generation = self._generation
        self._reader_task = asyncio.create_task(self._read_stdout(generation))
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
        self,
        method: str,
        parser,
        label: str,
        *,
        abort_signal: AbortSignal | None = None,
    ) -> list:
        request = self._request
        if abort_signal is not None:
            request = partial(self._request, abort_signal=abort_signal)
        return await drain_pages(
            request,
            method,
            parser,
            label,
            max_pages=MAX_LIST_PAGES,
            max_items=MAX_LIST_ITEMS,
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
            "resources/list",
            resources_from_result,
            "resources/list",
            abort_signal=abort_signal,
        )

    async def read_resource(self, uri: str, abort_signal: AbortSignal | None = None):
        result = await self._request("resources/read", {"uri": uri}, abort_signal)
        return resource_content_from_result(result)

    async def get_prompt(self, name: str, arguments: Mapping[str, str]) -> str:
        result = await self._request(
            "prompts/get", {"name": name, "arguments": dict(arguments)}
        )
        return prompt_text_from_result(result)

    async def call_tool(
        self, name: str, arguments: Mapping[str, object], abort_signal: AbortSignal
    ) -> StructuredToolResult:
        try:
            result = await self._request(
                "tools/call", {"name": name, "arguments": dict(arguments)}, abort_signal
            )
        except MCPCanceled:
            return canceled_result()
        except _StdioCallLimit:
            return self._limit_result(name)
        except _StdioSpillQuota:
            return make_error_result(
                f"MCP tool {self.config.name}__{name} could not buffer the response "
                "because the session spill budget is full"
            )
        except _StdioRestart:
            return self._restart_result()
        except MCPError as exc:
            self._report_failure(exc)
            return make_error_result(str(exc))
        except Exception as exc:  # noqa: BLE001 - remote failures become tool results
            self._report_failure(exc)
            return make_error_result(str(exc))
        return translate_call_result(result)

    def _limit_result(self, tool_name: str) -> StructuredToolResult:
        return make_error_result(
            f"MCP tool {self.config.name}__{tool_name} exceeded the "
            f"{_format_wire_limit(self._limits.max_wire_bytes)} response limit"
        )

    def _restart_result(self) -> StructuredToolResult:
        return make_error_result(
            f"MCP stdio server {self.config.name} restarted after a call limit; "
            "retry the tool"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        invalidation = self._start_invalidation(
            MCPTransportError("MCP stdio server closed"),
            report_failure=False,
            default_outcome="closed",
        )
        if invalidation.task is not None:
            await asyncio.shield(invalidation.task)
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
            raise self._reader_error or MCPTransportError(
                "MCP stdio server is not connected"
            )
        if self._closed:
            raise MCPTransportError("MCP stdio server is closed")
        self._next_id += 1
        request_id = self._next_id
        response = asyncio.get_running_loop().create_future()
        owner = asyncio.current_task()
        tool_name = params.get("name") if method == "tools/call" else None
        pending = _PendingCall(
            response,
            owner,
            tool_name if type(tool_name) is str else None,
        )
        self._pending[request_id] = pending
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        try:
            await self._write_request(process, request, response, request_id)
            if abort_signal is None:
                raw_response = await response
            else:
                raw_response = await self._wait_for_response(
                    response, abort_signal, request_id
                )
            return parse_rpc_response(raw_response, request_id)
        except asyncio.CancelledError:
            if abort_signal is not None and abort_signal.is_set():
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
                invalidation = self._start_invalidation(
                    MCPTransportError("MCP stdio server terminated after cancellation"),
                    cancel_ids={request_id},
                    default_outcome="restart",
                )
                if invalidation.task is not None:
                    await asyncio.shield(invalidation.task)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def _wait_for_response(
        self,
        response: asyncio.Future[dict[str, object]],
        abort_signal: AbortSignal,
        request_id: int,
    ) -> dict[str, object]:
        abort_task = asyncio.create_task(abort_signal.wait())
        try:
            done, _ = await asyncio.wait(
                (response, abort_task), return_when=asyncio.FIRST_COMPLETED
            )
            if abort_task in done and response not in done:
                invalidation = self._start_invalidation(
                    MCPTransportError("MCP stdio server terminated after cancellation"),
                    cancel_ids={request_id},
                    default_outcome="restart",
                )
                if invalidation.task is not None:
                    await asyncio.shield(invalidation.task)
            return await response
        finally:
            if not abort_task.done():
                abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)

    async def _write_request(
        self,
        process: asyncio.subprocess.Process,
        request: Mapping[str, object],
        response: asyncio.Future[dict[str, object]],
        request_id: int,
    ) -> None:
        try:
            async with self._write_lock:
                if process.stdin is None:
                    raise MCPTransportError("MCP stdio server closed its input")
                process.stdin.write(
                    (json.dumps(request, separators=(",", ":")) + "\n").encode()
                )
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            reader_error = self._reader_error
            invalidation = self._start_invalidation(
                MCPTransportError("MCP stdio server closed its input")
            )
            if invalidation.task is not None:
                await asyncio.shield(invalidation.task)
            if response.done():
                response.exception()
            raise reader_error or MCPTransportError(
                "MCP stdio server closed its input"
            ) from exc
        except asyncio.CancelledError:
            invalidation = self._start_invalidation(
                MCPTransportError("MCP stdio server terminated after cancellation"),
                cancel_ids={request_id},
                default_outcome="restart",
            )
            if invalidation.task is not None:
                await asyncio.shield(invalidation.task)
            raise

    async def _notify(self, method: str, params: Mapping[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or self._closed:
            return
        message = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        try:
            async with self._write_lock:
                process.stdin.write(
                    (json.dumps(message, separators=(",", ":")) + "\n").encode()
                )
                await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _start_invalidation(
        self,
        reason: MCPTransportError,
        *,
        limit_ids: set[int] | None = None,
        quota_ids: set[int] | None = None,
        cancel_ids: set[int] | None = None,
        report_failure: bool = True,
        default_outcome: _Outcome = "transport",
    ) -> _Invalidation:
        limit_ids = limit_ids or set()
        quota_ids = quota_ids or set()
        cancel_ids = cancel_ids or set()
        current = self._invalidation
        if current is not None:
            self._record_outcomes(current, limit_ids, quota_ids, cancel_ids)
            return current

        process = self._process
        reader_task = self._reader_task
        stderr = self._stderr
        accumulator = self._active_accumulator
        pending = self._pending
        self._process = None
        self._reader_task = None
        self._stderr = None
        self._active_accumulator = None
        self._pending = {}
        self._generation += 1
        outcomes = {request_id: default_outcome for request_id in pending}
        invalidation = _Invalidation(
            self._generation - 1,
            process,
            reader_task,
            stderr,
            accumulator,
            pending,
            outcomes,
            report_failure,
            reason,
            asyncio.current_task(),
        )
        self._invalidation = invalidation
        self._record_outcomes(invalidation, limit_ids, quota_ids, cancel_ids)
        invalidation.task = asyncio.create_task(self._finish_invalidation(invalidation))
        return invalidation

    def _outcome_values(
        self, outcome: _Outcome, tool_name: str | None
    ) -> tuple[StructuredToolResult | None, BaseException | None]:
        if outcome == "limit":
            result = self._limit_result(tool_name) if tool_name is not None else None
            return result, _StdioCallLimit()
        if outcome == "quota":
            result = (
                make_error_result(
                    f"MCP tool {self.config.name}__{tool_name} could not buffer "
                    "the response because the session spill budget is full"
                )
                if tool_name is not None
                else None
            )
            return result, _StdioSpillQuota()
        if outcome == "canceled":
            return canceled_result() if tool_name is not None else None, MCPCanceled()
        if outcome == "restart":
            result = self._restart_result() if tool_name is not None else None
            return result, _StdioRestart()
        return None, None

    def _record_outcomes(
        self,
        invalidation: _Invalidation,
        limit_ids: set[int],
        quota_ids: set[int],
        cancel_ids: set[int],
    ) -> None:
        for request_id in limit_ids:
            if request_id in invalidation.pending:
                invalidation.outcomes[request_id] = "limit"
        for request_id in quota_ids:
            if request_id in invalidation.pending:
                invalidation.outcomes[request_id] = "quota"
        for request_id in cancel_ids:
            if request_id in invalidation.pending:
                invalidation.outcomes[request_id] = "canceled"
        for request_id, pending in invalidation.pending.items():
            if pending.owner is None or pending.tool_name is None:
                continue
            result, _ = self._outcome_values(
                invalidation.outcomes[request_id], pending.tool_name
            )
            if result is not None:
                self._invalidation_results[pending.owner] = result

    async def _finish_invalidation(self, invalidation: _Invalidation) -> None:
        reader_tasks = (
            [invalidation.reader_task]
            if invalidation.reader_task is not None
            and invalidation.reader_task is not invalidation.initiator
            else []
        )
        try:
            if invalidation.process is not None:
                await _kill_and_reap(invalidation.process, reader_tasks)
            elif reader_tasks:
                for task in reader_tasks:
                    task.cancel()
                await asyncio.gather(*reader_tasks, return_exceptions=True)
        finally:
            if invalidation.accumulator is not None:
                await invalidation.accumulator.close()
            if invalidation.stderr is not None:
                invalidation.stderr.close()
            for request_id, pending in invalidation.pending.items():
                if pending.future.done():
                    continue
                _, error = self._outcome_values(
                    invalidation.outcomes[request_id], pending.tool_name
                )
                pending.future.set_exception(error or invalidation.reason)
            if self._reader_error is None:
                self._reader_error = invalidation.reason
            if invalidation.report_failure:
                self._report_failure(invalidation.reason)

    async def _read_messages(
        self, stream: asyncio.StreamReader, generation: int | None = None
    ) -> AsyncIterator[_Frame]:
        """Yield bounded frames with pending-at-arrival byte ownership."""

        generation = self._generation if generation is None else generation
        envelope = MCPCallEnvelope(self._limits, self._spill_store)
        frame = _Frame()
        self._active_accumulator = envelope
        try:
            while chunk := await stream.read(65_536):
                pending_at_arrival = tuple(self._pending)
                remaining = chunk
                while remaining:
                    separator = remaining.find(b"\n")
                    if separator < 0:
                        part = remaining
                        remaining = b""
                    else:
                        part = remaining[: separator + 1]
                        remaining = remaining[separator + 1 :]
                    append_result = await _append_frame_part(
                        envelope, frame, part, pending_at_arrival
                    )
                    if isinstance(append_result, _FrameInvalidation):
                        invalidation = self._start_invalidation(
                            MCPTransportError("MCP stdio response exceeded its bounds"),
                            limit_ids=append_result.limit_ids,
                            quota_ids=append_result.quota_ids,
                            default_outcome="restart",
                        )
                        if invalidation.task is not None:
                            await asyncio.shield(invalidation.task)
                        return
                    frame = append_result
                    if separator < 0:
                        break
                    frame.value = await self._finish_frame(envelope)
                    if not await self._finish_read_frame(frame):
                        return
                    yield frame
                    if generation != self._generation:
                        return
                    envelope = MCPCallEnvelope(self._limits, self._spill_store)
                    frame = _Frame()
                    self._active_accumulator = envelope
            if frame.total_bytes:
                frame.value = await self._finish_frame(envelope)
                if await self._finish_read_frame(frame):
                    yield frame
                envelope = MCPCallEnvelope(self._limits, self._spill_store)
        finally:
            if self._active_accumulator is envelope:
                self._active_accumulator = None
            await envelope.close()

    async def _finish_frame(self, envelope: MCPCallEnvelope) -> object:
        try:
            return await envelope.finish_json()
        except json.JSONDecodeError as exc:
            return _MalformedMessage(MCPProtocolError(f"invalid MCP JSON: {exc.msg}"))

    async def _finish_read_frame(self, frame: _Frame) -> bool:
        overflow = self._charge_frame(frame)
        if not overflow:
            return True
        invalidation = self._start_invalidation(
            MCPTransportError("MCP stdio call exceeded its response limit"),
            limit_ids=overflow,
            default_outcome="restart",
        )
        if invalidation.task is not None:
            await asyncio.shield(invalidation.task)
        return False

    async def _read_stdout(self, generation: int | None = None) -> None:
        generation = self._generation if generation is None else generation
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            async for frame in self._read_messages(process.stdout, generation):
                if generation != self._generation:
                    return
                value = frame.value
                if isinstance(value, _MalformedMessage):
                    self._fail_pending(value.error)
                    continue
                if type(value) is not dict:
                    self._fail_pending(
                        MCPProtocolError("MCP message must be an object")
                    )
                    continue
                response_id = value.get("id")
                if type(response_id) is int and response_id in self._pending:
                    future = self._pending[response_id].future
                    if not future.done():
                        future.set_result(value)
                elif type(value.get("method")) is str:
                    method = value["method"]
                    logger.debug("MCP %s notification: %s", self.config.name, method)
                    if self._notification_sink is not None:
                        self._notification_sink(method)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reader failure is transport failure
            if generation == self._generation:
                invalidation = self._start_invalidation(
                    MCPTransportError(f"MCP stdio reader failed: {exc}")
                )
                if invalidation.task is not None:
                    await asyncio.shield(invalidation.task)
        finally:
            if generation == self._generation and self._process is process:
                invalidation = self._start_invalidation(
                    MCPTransportError(
                        f"MCP stdio server exited with code {process.returncode}"
                    )
                )
                if invalidation.task is not None:
                    await asyncio.shield(invalidation.task)

    def _charge_frame(self, frame: _Frame) -> set[int]:
        value = frame.value
        owner = value.get("id") if type(value) is dict else None
        if type(owner) is int and owner in self._pending:
            pending = self._pending[owner]
            pending.wire_bytes += frame.bytes_by_pending_id.get(owner, 0)
            return (
                {owner} if pending.wire_bytes > self._limits.max_wire_bytes else set()
            )
        overflow: set[int] = set()
        for request_id, byte_count in frame.bytes_by_pending_id.items():
            pending = self._pending.get(request_id)
            if pending is None:
                continue
            pending.wire_bytes += byte_count
            if pending.wire_bytes > self._limits.max_wire_bytes:
                overflow.add(request_id)
        return overflow

    def _fail_pending(self, error: BaseException) -> None:
        if self._reader_error is None:
            self._reader_error = error
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(error)


__all__ = ["MCP_STDIO_LINE_LIMIT", "StdioMCPClient"]


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask the failure
        return type(error).__name__
