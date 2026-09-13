"""Single-owner lifecycle actors for configured MCP servers."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..core.abort import AbortSignal
from ..tools.registry import ToolRegistry
from ..types import StructuredToolResult
from .client import MCPClient, MCPPrompt, MCPTool, make_error_result
from .config import MCPServerConfig, mcp_log_path, tool_prefix
from .prompt_actor import (
    CallFinished as _CallFinished,
    CallRequest as _CallRequest,
    cancel_request as _cancel_request,
    cancel_prompt_request,
    CancelRequest as _CancelRequest,
    PromptFinished as _PromptFinished,
    PromptRequest as _PromptRequest,
    discover_prompts,
    get_prompt as _get_prompt,
    handle_prompt as _handle_prompt,
    handle_prompt_finished as _handle_prompt_finished,
    prompt_unavailable,
    resolve_prompt_message,
    set_result as _set_result,
)

logger = logging.getLogger(__name__)

SERVER_SETUP_TIMEOUT_SECONDS = 10.0
AUTO_RECONNECT_BASE_DELAY_SECONDS = 1.0
AUTO_RECONNECT_MAX_DELAY_SECONDS = 30.0

MCPServerState = Literal[
    "mounted",
    "degraded",
    "failed",
    "skipped-missing-env",
    "timed-out",
    "malformed",
]
NoticeSink = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class MCPServerStatus:
    """Current connection state for one configured MCP server."""

    name: str
    transport: str
    state: MCPServerState
    tool_count: int = 0
    reason: str | None = None
    stderr_log_path: str = ""
    next_retry_at: float | None = None


@dataclass(frozen=True, slots=True)
class _SetupOutcome:
    status: MCPServerStatus
    client: MCPClient | None = None
    tools: tuple[MCPTool, ...] = ()
    prompts: tuple[MCPPrompt, ...] = ()
    failure_reason: str | None = None


@dataclass(slots=True)
class _Operation:
    identifier: int
    kind: Literal["initial", "manual", "auto"]
    config: MCPServerConfig
    source: Path
    generation: int
    preserve_degraded: bool
    request: asyncio.Future[MCPServerStatus] | None = None
    waiter: _CallRequest | None = None
    task: asyncio.Task[_SetupOutcome] | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _StartOperation:
    identifier: int
    kind: Literal["initial", "manual", "auto"]
    config: MCPServerConfig
    source: Path
    notice_sink: NoticeSink | None
    request: asyncio.Future[MCPServerStatus] | None
    waiter: _CallRequest | None = None


@dataclass(frozen=True, slots=True)
class _SetupFinished:
    identifier: int
    task: asyncio.Task[_SetupOutcome]
    outcome: _SetupOutcome | None


@dataclass(frozen=True, slots=True)
class _TransportFailure:
    operation: int | None
    client: MCPClient
    reason: str


@dataclass(frozen=True, slots=True)
class _ChildFinished:
    task: asyncio.Task[object]


@dataclass(frozen=True, slots=True)
class _CancelOperation:
    identifier: int
    acknowledged: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _Remove:
    result: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _RemoveWhenIdle:
    result: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _Close:
    result: asyncio.Future[None]


PublishSnapshot = Callable[
    ["MCPServerActor", MCPServerStatus, MCPClient | None], None
]


class MCPServerActor:
    """Own one server lifecycle and serialize all lifecycle messages."""

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        source: Path,
        registry: ToolRegistry | None,
        publish: PublishSnapshot,
        build_client: Callable[[MCPServerConfig], MCPClient],
        connect_and_list: Callable[[MCPClient], Awaitable[list[MCPTool]]],
        setup_timeout: float = SERVER_SETUP_TIMEOUT_SECONDS,
        auto_base_delay: float = AUTO_RECONNECT_BASE_DELAY_SECONDS,
        auto_max_delay: float = AUTO_RECONNECT_MAX_DELAY_SECONDS,
    ) -> None:
        self.config = config
        self.source = source
        self._registry = registry
        self._publish_callback = publish
        self._build_client = build_client
        self._connect_and_list = connect_and_list
        self._setup_timeout = setup_timeout
        self._auto_base_delay = auto_base_delay
        self._auto_max_delay = auto_max_delay
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._children: set[asyncio.Task[object]] = set()
        self._operation: _Operation | None = None
        self._pending_starts: list[_StartOperation] = []
        self._pending_removes: list[asyncio.Future[None]] = []
        self._requests: dict[int, _CallRequest | _PromptRequest] = {}
        self._scheduled_closes: dict[
            int, tuple[MCPClient, asyncio.Task[object]]
        ] = {}
        self._next_identifier = 0
        self._client: MCPClient | None = None
        self._tools: tuple[MCPTool, ...] = ()
        self._prompts: tuple[MCPPrompt, ...] = ()
        self._generation = 0
        self._failure_count = 0
        self._status = MCPServerStatus(
            config.name,
            config.transport,
            "failed",
            stderr_log_path=str(mcp_log_path(config.name)),
        )
        self._closed = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def status(self) -> MCPServerStatus:
        return self._status
    @property
    def prompts(self) -> tuple[MCPPrompt, ...]:
        return self._prompts
    @property
    def generation(self) -> int:
        return self._generation
    @property
    def is_terminal(self) -> bool:
        return self._closed or (self._task is not None and self._task.done())

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return self._task

    async def wait_started(
        self,
        notice_sink: NoticeSink | None = None,
    ) -> MCPServerStatus:
        if self.is_terminal:
            return _unavailable_status(self.name, self.config)
        future: asyncio.Future[MCPServerStatus] = asyncio.get_running_loop().create_future()
        self._next_identifier += 1
        self._queue.put_nowait(
            _StartOperation(
                self._next_identifier,
                "initial",
                self.config,
                self.source,
                notice_sink,
                future,
            )
        )
        return await asyncio.shield(future)

    async def reconnect(
        self,
        *,
        notice_sink: NoticeSink | None = None,
    ) -> MCPServerStatus:
        return await self._request_operation("manual", notice_sink)

    async def replace(
        self,
        config: MCPServerConfig,
        source: Path,
        *,
        notice_sink: NoticeSink | None = None,
    ) -> MCPServerStatus:
        return await self._request_operation(
            "manual", notice_sink, config=config, source=source
        )

    async def remove(self) -> None:
        if self.is_terminal:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Remove(future))
        await asyncio.shield(future)

    async def remove_when_idle(self) -> None:
        if self.is_terminal:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_RemoveWhenIdle(future))
        await asyncio.shield(future)

    async def close(self) -> None:
        if self._task is None:
            return
        if self._task.done():
            await asyncio.gather(self._task, return_exceptions=True)
            return
        if self._closed:
            await asyncio.shield(self._task)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Close(future))
        await asyncio.shield(future)
        await asyncio.shield(self._task)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        abort_signal: AbortSignal,
        *,
        generation: int,
    ) -> StructuredToolResult:
        future: asyncio.Future[StructuredToolResult] = asyncio.get_running_loop().create_future()
        self._next_identifier += 1
        request = _CallRequest(
            self._next_identifier,
            tool_name,
            dict(arguments),
            abort_signal,
            generation,
            future,
        )
        if self.is_terminal:
            return _unavailable_result(self.name)
        self._queue.put_nowait(request)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await _cancel_request(self, request.identifier)
            raise

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str],
        *,
        generation: int,
    ) -> str:
        return await _get_prompt(self, prompt_name, arguments, generation=generation)
    async def _request_operation(
        self,
        kind: Literal["manual", "auto"],
        notice_sink: NoticeSink | None,
        *,
        config: MCPServerConfig | None = None,
        source: Path | None = None,
    ) -> MCPServerStatus:
        if self.is_terminal:
            return _unavailable_status(self.name, self.config)
        future: asyncio.Future[MCPServerStatus] = asyncio.get_running_loop().create_future()
        self._next_identifier += 1
        identifier = self._next_identifier
        self._queue.put_nowait(
            _StartOperation(
                identifier,
                kind,
                config or self.config,
                source or self.source,
                notice_sink,
                future,
            )
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if self._task is not None and not self._task.done():
                acknowledged: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._queue.put_nowait(_CancelOperation(identifier, acknowledged))
                await asyncio.shield(acknowledged)
            raise

    async def _run(self) -> None:
        message: object | None = None
        try:
            while True:
                message = await self._queue.get()
                if isinstance(message, _Close):
                    await self._handle_close(message.result)
                    return
                if isinstance(message, _StartOperation):
                    self._handle_start(message)
                elif isinstance(message, _SetupFinished):
                    self._handle_setup_finished(message)
                elif isinstance(message, _TransportFailure):
                    self._handle_transport_failure(message)
                elif isinstance(message, _CallRequest):
                    self._handle_call(message)
                elif isinstance(message, _CallFinished):
                    self._handle_call_finished(message)
                elif isinstance(message, _PromptRequest):
                    _handle_prompt(self, message)
                elif isinstance(message, _PromptFinished):
                    _handle_prompt_finished(self, message)
                elif isinstance(message, _CancelOperation):
                    self._handle_cancel_operation(message)
                elif isinstance(message, _CancelRequest):
                    self._handle_cancel_request(message)
                elif isinstance(message, _ChildFinished):
                    self._children.discard(message.task)
                elif isinstance(message, _Remove):
                    await self._handle_remove(message.result)
                    return
                elif isinstance(message, _RemoveWhenIdle):
                    if self._operation is None:
                        await self._handle_remove(message.result)
                        return
                    self._pending_removes.append(message.result)
                if self._operation is None and self._pending_starts and not self._closed:
                    self._handle_start(self._pending_starts.pop(0))
                if self._operation is None and self._pending_removes and not self._closed:
                    await self._handle_remove(self._pending_removes.pop(0))
                    return
        except asyncio.CancelledError:
            await asyncio.shield(self._finalize_terminal(message, None))
            raise
        except BaseException as exc:
            await self._finalize_terminal(message, exc)

    def _handle_start(self, message: _StartOperation) -> None:
        if self._closed:
            if message.request is not None:
                _set_result(message.request, _unavailable_status(self.name, self.config))
            if message.waiter is not None:
                _set_result(message.waiter.result, _unavailable_result(self.name))
            return
        if message.kind == "initial" and self._operation is not None:
            if message.request is not None:
                _set_result(message.request, self._status)
            return
        if self._operation is not None:
            if message.kind == "manual" and self._operation.kind == "auto":
                self._abandon_current_operation()
            else:
                self._pending_starts.append(message)
                return
        self.config = message.config
        self.source = message.source
        preserve_degraded = self._status.state == "degraded"
        if message.kind == "manual" and self._client is not None:
            old_client = self._client
            self._client = None
            self._tools = ()
            self._prompts = ()
            self._unregister_tools()
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "failed",
                    reason="reconnecting",
                    stderr_log_path=str(mcp_log_path(self.name)),
                )
            )
            self._schedule_close(old_client)
        setup_generation = self._generation + 1
        operation = _Operation(
            message.identifier,
            message.kind,
            message.config,
            message.source,
            setup_generation,
            preserve_degraded,
            message.request,
            message.waiter,
        )
        self._operation = operation
        task = asyncio.create_task(
            self._prepare(
                operation,
                message.notice_sink,
            )
        )
        operation.task = task
        self._children.add(task)
        task.add_done_callback(
            lambda done, identifier=operation.identifier: self._queue_setup_result(
                done, identifier
            )
        )

    def _queue_setup_result(
        self,
        task: asyncio.Task[_SetupOutcome],
        identifier: int,
    ) -> None:
        if task.cancelled():
            self._queue.put_nowait(_SetupFinished(identifier, task, None))
            return
        try:
            outcome = task.result()
        except BaseException:
            self._queue.put_nowait(_SetupFinished(identifier, task, None))
        else:
            self._queue.put_nowait(_SetupFinished(identifier, task, outcome))

    async def _prepare(
        self,
        operation: _Operation,
        notice_sink: NoticeSink | None,
    ) -> _SetupOutcome:
        config = operation.config
        if config.malformed_reason is not None:
            reason = config.malformed_reason
            status = MCPServerStatus(
                self.name,
                config.transport,
                "malformed",
                reason=reason,
                stderr_log_path=str(mcp_log_path(self.name)),
            )
            _notice(notice_sink, f"mcp · {self.name} malformed ({reason})")
            return _SetupOutcome(status)
        if config.missing_env:
            variables = ", ".join(config.missing_env)
            reason = f"missing environment variable(s): {variables}"
            status = MCPServerStatus(
                self.name,
                config.transport,
                "skipped-missing-env",
                reason=reason,
                stderr_log_path=str(mcp_log_path(self.name)),
            )
            _notice(
                notice_sink,
                f"mcp · {self.name} skipped-missing-env ({variables})",
            )
            return _SetupOutcome(status)

        client: MCPClient | None = None
        failure_future: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        def report_failure(reason: str) -> None:
            if client is None:
                return
            if not failure_future.done():
                failure_future.set_result(reason)
            self._queue.put_nowait(
                _TransportFailure(operation.identifier, client, reason)
            )

        try:
            client = self._build_client(config)
            set_failure_sink = getattr(client, "set_failure_sink", None)
            if set_failure_sink is not None:
                set_failure_sink(report_failure)
            tools = await asyncio.wait_for(
                self._connect_and_list(client), timeout=self._setup_timeout
            )
            prompts = await discover_prompts(
                client,
                self.name,
                self._setup_timeout,
                lambda message: _notice(notice_sink, message),
            )
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            if client is not None:
                await _safe_close(client)
            raise
        except TimeoutError:
            reason = f"after {self._setup_timeout:.1f}s"
            logger.warning(
                "timed out connecting to MCP server %s %s", self.name, reason
            )
            status = MCPServerStatus(
                self.name,
                config.transport,
                "timed-out",
                reason=reason,
                stderr_log_path=str(mcp_log_path(self.name)),
            )
            _notice(notice_sink, f"mcp · {self.name} timed-out ({reason})")
            return _SetupOutcome(status, client)
        except Exception as exc:  # noqa: BLE001 - isolate one server
            reason = _error_text(exc)
            logger.warning("failed to mount MCP server %s: %s", self.name, reason)
            status = MCPServerStatus(
                self.name,
                config.transport,
                "failed",
                reason=reason,
                stderr_log_path=str(mcp_log_path(self.name)),
            )
            _notice(notice_sink, f"mcp · {self.name} failed: {reason}")
            return _SetupOutcome(status, client)
        status = MCPServerStatus(
            self.name,
            config.transport,
            "mounted",
            tool_count=len(tools),
            stderr_log_path=str(mcp_log_path(self.name)),
        )
        _notice(notice_sink, f"mcp · {self.name} mounted ({len(tools)} tools)")
        failure_reason = failure_future.result() if failure_future.done() else None
        return _SetupOutcome(status, client, tuple(tools), prompts, failure_reason)

    def _handle_setup_finished(self, message: _SetupFinished) -> None:
        self._children.discard(message.task)
        operation = self._operation
        if operation is None or operation.identifier != message.identifier:
            if message.outcome is not None and message.outcome.client is not None:
                self._schedule_close(message.outcome.client)
            return
        if message.outcome is None:
            self._finish_cancelled(operation)
            return
        outcome = message.outcome
        reason = outcome.failure_reason or operation.failure_reason
        if reason is not None and outcome.status.state == "mounted":
            outcome = _SetupOutcome(
                MCPServerStatus(
                    self.name,
                    operation.config.transport,
                    "failed",
                    reason=reason,
                    stderr_log_path=str(mcp_log_path(self.name)),
                ),
                outcome.client,
            )
        if outcome.status.state == "mounted" and outcome.client is not None:
            self._unregister_tools()
            self.config = operation.config
            self.source = operation.source
            self._client = outcome.client
            self._tools = outcome.tools
            self._prompts = outcome.prompts
            self._generation = operation.generation
            self._failure_count = 0
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "mounted",
                    tool_count=len(self._tools),
                    stderr_log_path=str(mcp_log_path(self.name)),
                )
            )
            for tool in self._tools:
                self._register_tool(tool, self._generation)
            self._publish_callback(
                self,
                self._status,
                self._client,
            )
            self._complete_operation(operation, self._status)
            if operation.waiter is not None:
                operation.waiter.generation = self._generation
                self._dispatch_call(operation.waiter, self._client)
            return
        if outcome.client is not None:
            self._schedule_close(outcome.client)
        if reason is not None and operation.kind != "auto":
            self._client = None
            self._prompts = ()
            if not operation.preserve_degraded:
                self._tools = ()
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "degraded",
                    tool_count=len(self._tools),
                    reason=reason,
                    stderr_log_path=str(mcp_log_path(self.name)),
                    next_retry_at=time.monotonic(),
                )
            )
            self._complete_operation(operation, self._status)
            if operation.waiter is not None:
                _set_result(operation.waiter.result, _degraded_result(self.name, self._status))
            return
        if operation.kind == "auto":
            self._failure_count += 1
            exponent = min(max(self._failure_count - 1, 0), 30)
            delay = min(
                self._auto_max_delay,
                self._auto_base_delay * 2**exponent,
            )
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "degraded",
                    tool_count=len(self._tools),
                    reason=outcome.status.reason or "automatic remount failed",
                    stderr_log_path=str(mcp_log_path(self.name)),
                    next_retry_at=time.monotonic() + delay,
                )
            )
            self._complete_operation(operation, self._status)
            if operation.waiter is not None:
                _set_result(operation.waiter.result, _degraded_result(self.name, self._status))
            return
        if operation.preserve_degraded:
            previous = self._status
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "degraded",
                    tool_count=len(self._tools),
                    reason=previous.reason,
                    stderr_log_path=str(mcp_log_path(self.name)),
                    next_retry_at=previous.next_retry_at or time.monotonic(),
                )
            )
        elif outcome.status.state == "degraded":
            self._set_status(outcome.status)
        else:
            self._client = None
            self._tools = ()
            self._prompts = ()
            self._set_status(outcome.status)
        self._complete_operation(operation, self._status)
        if operation.waiter is not None:
            _set_result(operation.waiter.result, _degraded_result(self.name, self._status))

    def _finish_cancelled(self, operation: _Operation) -> None:
        self._close_completed_setup_client(operation)
        self._client = None
        if operation.preserve_degraded:
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "degraded",
                    tool_count=len(self._tools),
                    reason=self._status.reason,
                    stderr_log_path=str(mcp_log_path(self.name)),
                    next_retry_at=self._status.next_retry_at,
                )
            )
        else:
            self._unregister_tools()
            self._tools = ()
            self._prompts = ()
            self._set_status(
                MCPServerStatus(
                    self.name,
                    self.config.transport,
                    "failed",
                    stderr_log_path=str(mcp_log_path(self.name)),
                )
            )
        if operation.waiter is not None:
            _set_result(operation.waiter.result, _degraded_result(self.name, self._status))
        self._complete_operation(operation, self._status)

    def _handle_transport_failure(self, message: _TransportFailure) -> None:
        operation = self._operation
        if operation is not None and message.operation == operation.identifier:
            operation.failure_reason = message.reason
            return
        if self._client is not message.client or self._closed:
            return
        self._degrade_current(message.reason)

    def _degrade_current(self, reason: str) -> None:
        client = self._client
        if client is None:
            return
        self._client = None
        self._prompts = ()
        self._detach_failure_sink(client)
        self._schedule_close(client)
        self._set_status(
            MCPServerStatus(
                self.name,
                self.config.transport,
                "degraded",
                tool_count=len(self._tools),
                reason=reason,
                stderr_log_path=str(mcp_log_path(self.name)),
                next_retry_at=time.monotonic(),
            )
        )

    def _handle_call(self, request: _CallRequest) -> None:
        if self._closed:
            _set_result(request.result, _unavailable_result(self.name))
            return
        if request.generation != self._generation:
            _set_result(request.result, _unavailable_result(self.name))
            return
        if self._status.state == "mounted" and self._client is not None:
            self._dispatch_call(request, self._client)
            return
        if self._status.state != "degraded":
            _set_result(request.result, _unavailable_result(self.name))
            return
        retry_at = self._status.next_retry_at
        if self._operation is not None or (
            retry_at is not None and time.monotonic() < retry_at
        ):
            _set_result(request.result, _degraded_result(self.name, self._status))
            return
        self._next_identifier += 1
        self._handle_start(
            _StartOperation(
                self._next_identifier,
                "auto",
                self.config,
                self.source,
                None,
                None,
                waiter=request,
            )
        )

    def _dispatch_call(
        self,
        request: _CallRequest,
        client: MCPClient,
    ) -> None:
        request.client = client
        request.task = asyncio.create_task(
            self._invoke_call(request, client)
        )
        self._requests[request.identifier] = request
        self._children.add(request.task)
        request.task.add_done_callback(
            lambda done, call=request: self._queue_call_result(done, call)
        )

    async def _invoke_call(
        self,
        request: _CallRequest,
        client: MCPClient,
    ) -> StructuredToolResult:
        return await client.call_tool(
            request.tool_name,
            request.arguments,
            request.abort_signal,
        )

    def _queue_call_result(
        self,
        task: asyncio.Task[StructuredToolResult],
        request: _CallRequest,
    ) -> None:
        if task.cancelled():
            self._queue.put_nowait(
                _CallFinished(request, task, None, asyncio.CancelledError())
            )
            return
        try:
            result = task.result()
        except BaseException as exc:
            self._queue.put_nowait(_CallFinished(request, task, None, exc))
        else:
            self._queue.put_nowait(_CallFinished(request, task, result, None))

    def _handle_call_finished(self, message: _CallFinished) -> None:
        self._children.discard(message.task)
        request = self._requests.pop(message.request.identifier, message.request)
        if isinstance(message.error, asyncio.CancelledError):
            _set_result(request.result, _unavailable_result(self.name))
            return
        current = (
            not self._closed
            and self._client is request.client
            and request.generation == self._generation
            and self._status.state == "mounted"
        )
        if message.error is not None:
            if current:
                self._degrade_current(_error_text(message.error))
                _set_result(request.result, _degraded_result(self.name, self._status))
            else:
                _set_result(request.result, _unavailable_result(self.name))
            return
        if not current:
            if self._status.state == "degraded":
                result = _degraded_result(self.name, self._status)
            else:
                result = _unavailable_result(self.name)
            _set_result(request.result, result)
            return
        if message.result is not None:
            _set_result(request.result, message.result)

    def _handle_cancel_operation(self, message: _CancelOperation) -> None:
        operation = self._operation
        if operation is not None and operation.identifier == message.identifier:
            self._cancel_current_operation()
            self._finish_cancelled(operation)
        _set_result(message.acknowledged, None)

    def _handle_cancel_request(self, message: _CancelRequest) -> None:
        request = self._requests.get(message.identifier)
        if request is None:
            _set_result(message.acknowledged, None)
            return
        request.result.cancel()
        if request.task is not None:
            request.task.cancel()
        self._requests.pop(message.identifier, None)
        _set_result(message.acknowledged, None)

    async def _handle_remove(self, result: asyncio.Future[None]) -> None:
        await self._terminate(result)

    async def _handle_close(self, result: asyncio.Future[None]) -> None:
        await self._terminate(result, reason="MCP mount is closed")

    def _cancel_current_operation(self) -> None:
        operation = self._operation
        if operation is not None and operation.task is not None:
            operation.task.cancel()

    def _resolve_pending_removes(self) -> None:
        while self._pending_removes:
            _set_result(self._pending_removes.pop(0), None)

    def _abandon_current_operation(self) -> None:
        operation = self._operation
        if operation is None:
            return
        self._cancel_current_operation()
        self._finish_cancelled(operation)

    async def _terminate(
        self,
        result: asyncio.Future[None] | None,
        *,
        reason: str | None = None,
    ) -> None:
        if self._closed:
            await self._shutdown_children()
            if result is not None:
                _set_result(result, None)
            return
        self._closed = True
        operation = self._operation
        if operation is not None:
            self._close_completed_setup_client(operation)
        if operation is not None and operation.task is not None:
            operation.task.cancel()
        self._operation = None
        if operation is not None:
            if operation.request is not None:
                _set_result(
                    operation.request,
                    _unavailable_status(self.name, self.config),
                )
            if operation.waiter is not None:
                _set_result(operation.waiter.result, _unavailable_result(self.name))
        while self._pending_starts:
            self._resolve_message(self._pending_starts.pop(0))
        self._resolve_pending_removes()
        for request in self._requests.values():
            if request.task is not None:
                request.task.cancel()
            if isinstance(request, _CallRequest):
                _set_result(request.result, _unavailable_result(self.name))
            else:
                cancel_prompt_request(request, self.name)
        self._requests.clear()
        self._prompts = ()
        self._unregister_tools()
        if self._client is not None:
            self._schedule_close(self._client, detach=False)
            self._client = None
        if reason is not None:
            terminal_status = MCPServerStatus(
                self.name,
                self.config.transport,
                "failed",
                reason=reason,
                stderr_log_path=str(mcp_log_path(self.name)),
            )
            self._status = terminal_status
            try:
                self._publish_callback(self, terminal_status, None)
            except BaseException:  # noqa: BLE001 - finalization must complete
                logger.exception("failed to publish terminal MCP state")
        self._drain_queue()
        await self._shutdown_children()
        if result is not None:
            _set_result(result, None)

    async def _finalize_terminal(
        self,
        message: object | None,
        error: BaseException | None,
    ) -> None:
        if error is not None:
            logger.error(
                "MCP server actor %s crashed: %s",
                self.name,
                _error_text(error),
                exc_info=(type(error), error, error.__traceback__),
            )
        await self._terminate(
            None,
            reason=(
                "MCP server actor crashed"
                if error is not None
                else "MCP server actor cancelled"
            ),
        )
        self._resolve_message(message)

    def _resolve_message(self, message: object | None) -> None:
        if isinstance(message, (_Remove, _RemoveWhenIdle, _Close)):
            _set_result(message.result, None)
        elif isinstance(message, (_CancelOperation, _CancelRequest)):
            _set_result(message.acknowledged, None)
        elif isinstance(message, _CallRequest):
            _set_result(message.result, _unavailable_result(self.name))
        elif isinstance(message, _CallFinished):
            _set_result(message.request.result, _unavailable_result(self.name))
        elif resolve_prompt_message(message, self.name):
            return
        elif isinstance(message, _StartOperation):
            if message.request is not None:
                _set_result(
                    message.request,
                    _unavailable_status(self.name, self.config),
                )
            if message.waiter is not None:
                _set_result(message.waiter.result, _unavailable_result(self.name))

    def _schedule_close(
        self,
        client: MCPClient,
        *,
        detach: bool = True,
    ) -> asyncio.Task[object]:
        existing = self._scheduled_closes.get(id(client))
        if existing is not None:
            return existing[1]
        if detach:
            self._detach_failure_sink(client)
        task = asyncio.create_task(_safe_close(client))
        self._scheduled_closes[id(client)] = (client, task)
        self._children.add(task)
        task.add_done_callback(
            lambda done, client_id=id(client): self._scheduled_close_finished(
                client_id, done
            )
        )
        return task

    def _scheduled_close_finished(
        self,
        client_id: int,
        task: asyncio.Task[object],
    ) -> None:
        entry = self._scheduled_closes.get(client_id)
        if entry is not None and entry[1] is task:
            self._scheduled_closes.pop(client_id)
        self._queue_child_finished(task)

    def _close_completed_setup_client(self, operation: _Operation) -> None:
        task = operation.task
        if task is None or not task.done() or task.cancelled():
            return
        try:
            outcome = task.result()
        except BaseException:
            return
        if outcome.client is not None:
            self._schedule_close(outcome.client)

    def _queue_child_finished(self, task: asyncio.Task[object]) -> None:
        self._queue.put_nowait(_ChildFinished(task))

    def _detach_failure_sink(self, client: MCPClient) -> None:
        set_failure_sink = getattr(client, "set_failure_sink", None)
        if set_failure_sink is not None:
            set_failure_sink(None)

    async def _shutdown_children(self) -> None:
        tasks = tuple(task for task in self._children if task is not asyncio.current_task())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._children.clear()

    def _drain_queue(self) -> None:
        while True:
            try:
                message = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(message, _SetupFinished):
                self._children.discard(message.task)
                if message.outcome is not None and message.outcome.client is not None:
                    self._schedule_close(message.outcome.client)
            elif isinstance(message, (_CallFinished, _PromptFinished)):
                self._children.discard(message.task)
                self._resolve_message(message)
            elif isinstance(message, _ChildFinished):
                self._children.discard(message.task)
            else:
                self._resolve_message(message)

    def _set_status(self, status: MCPServerStatus) -> None:
        self._status = status
        self._publish_callback(
            self,
            status,
            self._client,
        )

    def _tool_prefix(self) -> str:
        """Namespace prefix for this server's tools."""

        return tool_prefix(self.name)

    def _register_tool(self, tool: MCPTool, generation: int) -> None:
        if self._registry is None or self._client is None:
            return
        name = f"{self._tool_prefix()}{tool.name}"

        async def handler(
            arguments: dict[str, object], abort_signal: AbortSignal
        ) -> StructuredToolResult:
            return await self.call_tool(
                tool.name,
                arguments,
                abort_signal,
                generation=generation,
            )

        try:
            self._registry.register(
                name,
                handler,
                description=tool.description,
                parameters=tool.input_schema,
                validate_arguments=False,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("skipping MCP tool %s: invalid input schema: %s", name, exc)

    def _unregister_tools(self) -> None:
        if self._registry is None:
            return
        prefix = self._tool_prefix()
        for name in tuple(self._registry.definitions_by_name):
            if name.startswith(prefix):
                self._registry.unregister(name)

    def _complete_operation(
        self,
        operation: _Operation,
        status: MCPServerStatus,
    ) -> None:
        if operation.request is not None:
            _set_result(operation.request, status)
        if self._operation is operation:
            self._operation = None

async def _safe_close(client: MCPClient) -> None:
    try:
        await client.close()
    except Exception:  # noqa: BLE001 - cleanup cannot mask lifecycle state
        logger.exception("failed to close MCP server %s", client.config.name)


def _notice(sink: NoticeSink | None, message: str) -> None:
    if sink is not None:
        sink(message)


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask the failure
        return type(error).__name__


def _retry_text(retry_at: float | None) -> str:
    if retry_at is None:
        return "when backoff expires"
    remaining = retry_at - time.monotonic()
    if remaining <= 0:
        return "now"
    return f"in {math.ceil(remaining * 10) / 10:.1f}s"


def _degraded_result(
    name: str,
    status: MCPServerStatus | None,
) -> StructuredToolResult:
    reason = (
        status.reason
        if status is not None and status.reason
        else "transport failure"
    )
    return make_error_result(
        f"MCP server '{name}' is degraded: {reason}. "
        f"Use /mcp reconnect {name}."
    )


def _unavailable_result(name: str) -> StructuredToolResult:
    return make_error_result(
        f"MCP server '{name}' is unavailable. Use /mcp reconnect {name}."
    )


def _unavailable_status(name: str, config: MCPServerConfig) -> MCPServerStatus:
    return MCPServerStatus(
        name,
        config.transport,
        "failed",
        reason="MCP mount is closed",
        stderr_log_path=str(mcp_log_path(name)),
    )


__all__ = [
    "AUTO_RECONNECT_BASE_DELAY_SECONDS",
    "AUTO_RECONNECT_MAX_DELAY_SECONDS",
    "SERVER_SETUP_TIMEOUT_SECONDS",
    "MCPServerActor",
    "MCPServerState",
    "MCPServerStatus",
    "_degraded_result",
    "_retry_text",
]
