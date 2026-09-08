"""Execution helpers shared by the tool registry and handlers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol

from .core.abort import AbortSignal
from .core.approval import canceled_result
from .types import (
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


class ToolStreamPublisher(Protocol):
    """Publish advisory output for one tool call."""

    def publish(self, text: str, stream: ToolStream) -> None:
        """Publish one output chunk."""

    def set_metadata(self, metadata: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Per-call state passed to handlers that need execution ownership."""

    tool_call: ToolCall
    agent_runner: Callable[..., Awaitable[ToolHandlerResult]] | None
    lifecycle_sink: ToolLifecycleSink | None


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


async def run_handler_with_abort(
    handler: ToolHandler,
    arguments: dict[str, Any],
    execution_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None,
    tool_call_id: str,
) -> ToolHandlerResult:
    current = asyncio.current_task()

    async def cancel_on_abort() -> None:
        await execution_signal.wait()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if current is not None and not current.done():
            current.cancel()

    abort_wait = asyncio.create_task(cancel_on_abort())
    result: ToolHandlerResult | None = None
    result_produced = False
    try:
        try:
            result = await invoke_handler(
                handler,
                arguments,
                execution_signal,
                stream_publisher,
            )
            result_produced = True
        except _ToolCanceled:
            result = canceled_result(tool_call_id)
            result_produced = True
        except asyncio.CancelledError:
            if execution_signal.is_set():
                result = canceled_result(tool_call_id)
                result_produced = True
            else:
                raise
        except Exception as exc:  # noqa: BLE001 - tool handlers fail closed
            result = ToolResult(tool_call_id, str(exc), True)
            result_produced = True
    finally:
        if stream_publisher is not None:
            stream_publisher.close()
        if not abort_wait.done():
            abort_wait.cancel()
            async def cleanup_abort_wait() -> None:
                await asyncio.gather(abort_wait, return_exceptions=True)

            cleanup = asyncio.create_task(cleanup_abort_wait())
            cleanup_canceled = False
            while True:
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cleanup_canceled = True
                    continue
                break
            if cleanup_canceled and not result_produced:
                raise asyncio.CancelledError
    if not result_produced:
        raise RuntimeError("tool handler did not produce a result")
    return result


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
