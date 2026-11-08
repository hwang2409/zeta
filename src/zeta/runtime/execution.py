"""Execution helpers shared by the tool registry and handlers."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

from ..core.abort import AbortSignal
from ..core.approval import ApprovedExecution, ApprovedPathExecution, canceled_result
from ..protocol.types import (
    StreamEvent,
    StreamEventType,
    StructuredToolResult,
    ToolCall,
    ToolResult,
)

ToolStream = Literal["stdout", "stderr"]
ToolHandlerResult = str | StructuredToolResult | ToolResult
ToolStreamSink = Callable[[StreamEvent], None]
ToolLifecycleSink = Callable[..., None]
logger = logging.getLogger(__name__)


class ToolStreamPublisher(Protocol):
    """Publish advisory output for one tool call."""

    def publish(self, text: str, stream: ToolStream) -> None:
        """Publish one output chunk."""

    def set_metadata(self, metadata: Mapping[str, object]) -> None: ...


@dataclass(slots=True)
class ToolExecutionContext:
    """Per-call state passed to handlers that need execution ownership."""

    tool_call: ToolCall
    agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None
    lifecycle_sink: ToolLifecycleSink | None
    approved_execution: ApprovedExecution | None = None
    host_environment: Mapping[str, str] = field(default_factory=dict)
    path_binding_consumed: bool = False

    def consume_path_binding(self, raw_path: str) -> ApprovedPathExecution | None:
        """Consume this call's path binding exactly once through the shared opener."""

        binding = self.approved_execution
        if not isinstance(binding, ApprovedPathExecution):
            return None
        if self.path_binding_consumed:
            raise ValueError("approved path binding was already consumed")
        if self.tool_call.arguments.get("path") != raw_path:
            raise ValueError("approved path does not match the tool target")
        self.path_binding_consumed = True
        return binding


@dataclass(slots=True)
class _ToolCallStreamPublisher:
    """Adapt handler chunks into non-blocking events for the agent loop."""

    tool_call: ToolCall
    abort_signal: AbortSignal
    sink: ToolStreamSink
    closed: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        self.metadata.update(metadata)

    def publish(self, text: str, stream: ToolStream) -> None:
        if self.closed or self.abort_signal.is_set():
            return
        self.sink(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_UPDATE,
                tool_call=self.tool_call,
                delta=text,
                data={"stream": stream, **self.metadata},
            )
        )

    def close(self) -> None:
        self.closed = True


class _ToolCanceled(Exception):
    pass


def _signal_is_set(signal_state: AbortSignal) -> bool:
    return signal_state.is_set()


async def _yield_for_abort(
    abort_signal: AbortSignal,
) -> None:
    if _signal_is_set(abort_signal):
        raise _ToolCanceled()
    await asyncio.sleep(0)
    if _signal_is_set(abort_signal):
        raise _ToolCanceled()


ToolHandler = Callable[..., ToolHandlerResult | Awaitable[ToolHandlerResult]]


async def invoke_handler(
    handler: ToolHandler,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
) -> ToolHandlerResult:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        result = handler(arguments, abort_signal)
    else:
        if stream_publisher is not None:
            result = _invoke_with_stream(
                handler, signature, arguments, abort_signal, stream_publisher
            )
        else:
            result = _invoke_without_stream(handler, signature, arguments, abort_signal)
    if inspect.isawaitable(result):
        result = await result
    return result


def _invoke_with_stream(
    handler: ToolHandler,
    signature: inspect.Signature,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher,
) -> ToolHandlerResult | Awaitable[ToolHandlerResult]:
    try:
        signature.bind(arguments, abort_signal, stream_publisher)
    except TypeError:
        try:
            signature.bind(
                arguments,
                abort_signal=abort_signal,
                stream_publisher=stream_publisher,
            )
        except TypeError:
            try:
                signature.bind(
                    arguments, abort_signal, stream_publisher=stream_publisher
                )
            except TypeError:
                return _invoke_without_stream(
                    handler, signature, arguments, abort_signal
                )
            return handler(
                arguments,
                abort_signal,
                stream_publisher=stream_publisher,
            )
        return handler(
            arguments,
            abort_signal=abort_signal,
            stream_publisher=stream_publisher,
        )
    return handler(arguments, abort_signal, stream_publisher)


def _invoke_without_stream(
    handler: ToolHandler,
    signature: inspect.Signature,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> ToolHandlerResult | Awaitable[ToolHandlerResult]:
    try:
        signature.bind(arguments, abort_signal)
    except TypeError:
        try:
            signature.bind(arguments, abort_signal=abort_signal)
        except TypeError:
            signature.bind(arguments)
            return handler(arguments)
        return handler(arguments, abort_signal=abort_signal)
    return handler(arguments, abort_signal)


def bind_execution_context(
    handler: ToolHandler,
    context: ToolExecutionContext,
) -> ToolHandler:
    try:
        inspect.signature(handler).parameters["execution_context"]
    except (KeyError, TypeError, ValueError):
        return handler
    return partial(handler, execution_context=context)


# How long the owner waits for a canceled handler to finish its cleanup.
# Read at call time, so tests lower it through the same code path.
HANDLER_CANCEL_DRAIN_SECONDS = 30.0
_detached_handlers: set[asyncio.Task[ToolHandlerResult]] = set()


async def run_handler_with_abort(
    handler: ToolHandler,
    arguments: dict[str, Any],
    execution_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None,
    tool_call_id: str,
) -> ToolHandlerResult:
    """Run one handler in a task that only this function cancels, at most once.

    The abort signal and every cancel of the caller's task are requests. The
    handler task gets one cancel for all of them, then has
    HANDLER_CANCEL_DRAIN_SECONDS to finish before it is detached.
    """
    handler_task = asyncio.create_task(
        _invoke_then_close(handler, arguments, execution_signal, stream_publisher)
    )
    clock = asyncio.get_running_loop()
    deadline: float | None = None  # set when the one cancel is sent

    def request_cancel() -> None:
        nonlocal deadline
        if deadline is None:
            handler_task.cancel()
            deadline = clock.time() + HANDLER_CANCEL_DRAIN_SECONDS

    async def cancel_on_signal() -> None:
        await execution_signal.wait()
        # Let a cooperative handler (bash, agent) see the signal first.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        request_cancel()

    signal_watch = asyncio.create_task(cancel_on_signal())
    try:
        while not handler_task.done():
            if deadline is not None and clock.time() >= deadline:
                _detach(handler_task, tool_call_id)
                break
            try:
                await asyncio.wait(
                    {handler_task} if signal_watch.done() else {handler_task, signal_watch},
                    timeout=None if deadline is None else deadline - clock.time(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                request_cancel()
    finally:
        # Also closes after a detach; the handler task closes on its own end.
        if stream_publisher is not None:
            stream_publisher.close()
        signal_watch.cancel()
    return _handler_result(handler_task, execution_signal, tool_call_id)


async def _invoke_then_close(
    handler: ToolHandler,
    arguments: dict[str, Any],
    execution_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None,
) -> ToolHandlerResult:
    # Output stops when the handler returns, not one loop turn later in the owner.
    try:
        return await invoke_handler(
            handler, arguments, execution_signal, stream_publisher
        )
    finally:
        if stream_publisher is not None:
            stream_publisher.close()


def _handler_result(
    handler_task: asyncio.Task[ToolHandlerResult],
    execution_signal: AbortSignal,
    tool_call_id: str,
) -> ToolHandlerResult:
    if handler_task.done() and not handler_task.cancelled():
        error = handler_task.exception()
        if error is None:
            return handler_task.result()
        if isinstance(error, _ToolCanceled):
            return canceled_result(tool_call_id)
        if isinstance(error, Exception):
            return ToolResult(tool_call_id, str(error), True)
        raise error
    if execution_signal.is_set():
        return canceled_result(tool_call_id)
    raise asyncio.CancelledError


def _detach(task: asyncio.Task[ToolHandlerResult], tool_call_id: str) -> None:
    _detached_handlers.add(task)
    task.add_done_callback(_reap_detached)
    logger.warning(
        "tool %s: cleanup still running %.0f s after its cancel; detached",
        tool_call_id,
        HANDLER_CANCEL_DRAIN_SECONDS,
    )


def _reap_detached(task: asyncio.Task[ToolHandlerResult]) -> None:
    _detached_handlers.discard(task)
    if not task.cancelled():
        task.exception()


def build_execution_arguments(
    arguments: dict[str, Any],
    *,
    log_path: str | Path | None,
    background: bool,
    capture_output: bool,
) -> dict[str, Any]:
    """Add harness-only arguments before invoking a registered handler."""

    execution_arguments = dict(arguments)
    if log_path is not None:
        execution_arguments["_log_path"] = str(log_path)
    if background:
        execution_arguments["_background"] = True
    if capture_output:
        execution_arguments["_capture_output"] = True
    return execution_arguments
