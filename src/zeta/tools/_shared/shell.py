"""Shared shell support for inline commands and custom shell macros."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from ...core.abort import AbortSignal
from ...protocol.types import (
    StreamEvent,
    StructuredToolResult,
    ToolCall,
    ToolResult,
    flatten_tool_content,
)
from ..registry import ToolRegistry, validate_tool_result

INLINE_SHELL_TIMEOUT = 10.0
INLINE_SHELL_OUTPUT_LIMIT = 8_192
INLINE_SHELL_MAX_SPANS = 32
INLINE_SHELL_TOTAL_OUTPUT_LIMIT = 64 * 1024
INLINE_SHELL_BATCH_TIMEOUT = 30.0
INLINE_SHELL_SPAN_LIMIT_MESSAGE = "[inline shell failed: span limit exceeded]"
INLINE_SHELL_OUTPUT_LIMIT_MESSAGE = (
    "[inline shell failed: aggregate output limit exceeded]"
)
INLINE_SHELL_BATCH_TIMEOUT_MESSAGE = "[inline shell failed: batch time limit exceeded]"


@dataclass(frozen=True, slots=True)
class MacroDisplay:
    """Trusted, harness-side display data for a macro's tool call."""

    command: str
    argv: tuple[str, ...] = ()


_trusted_macro_display: dict[str, MacroDisplay] = {}


def register_macro_display(call_id: str, *, command: str, argv: tuple[str, ...]) -> None:
    """Bind trusted display strings to a macro tool call the harness created."""

    _trusted_macro_display[call_id] = MacroDisplay(command, argv)


def trusted_macro_display(call_id: str) -> MacroDisplay | None:
    """Return trusted display data for a call created by the macro runner."""

    return _trusted_macro_display.get(call_id)


def forget_macro_display(call_id: str) -> None:
    """Drop trusted display data once its owning macro is done."""

    _trusted_macro_display.pop(call_id, None)


async def run_shell_macro(
    registry: ToolRegistry,
    call: ToolCall,
    log_path: str | Path,
    *,
    stream_sink: Callable[[StreamEvent], None],
    lifecycle_sink: Callable[[str], None],
    abort_signal: AbortSignal | None = None,
    background: bool = False,
) -> ToolResult:
    """Run an approved shell macro without persisting tool entries."""

    scope_signal = abort_signal or registry.abort_signal.registry.new_generation()
    try:
        raw_result = await registry.execute(
            call,
            abort_signal=scope_signal,
            _scope_signal=scope_signal,
            _stream_sink=stream_sink,
            _lifecycle_sink=lifecycle_sink,
            _persist_approval=False,
            _log_path=log_path,
            _background=background,
        )
        result = validate_tool_result(raw_result)
        structured = dict(result["structuredContent"] or {})
        structured["log_path"] = str(log_path)
        return ToolResult(
            call.id,
            flatten_tool_content(result["content"]),
            result["isError"],
            content_blocks=result["content"],
            structured_content=structured,
            is_canceled=result.get("isCanceled", False),
        )
    except asyncio.CancelledError:
        scope_signal.abort()
        return ToolResult(
            call.id,
            "tool execution canceled",
            True,
            is_canceled=True,
            structured_content={"log_path": str(log_path)},
        )


async def run_inline_shell_batch(
    registry: ToolRegistry,
    commands: tuple[str, ...],
    *,
    lifecycle_sink: Callable[[str, ToolCall], None],
    abort_signal: AbortSignal | None = None,
    timeout: float = INLINE_SHELL_TIMEOUT,
    output_limit: int = INLINE_SHELL_OUTPUT_LIMIT,
    max_spans: int = INLINE_SHELL_MAX_SPANS,
    total_output_limit: int = INLINE_SHELL_TOTAL_OUTPUT_LIMIT,
    batch_timeout: float = INLINE_SHELL_BATCH_TIMEOUT,
) -> tuple[str, ...]:
    """Run inline shell spans with one approval for the complete batch."""

    if not commands:
        return ()
    scope_signal = abort_signal or registry.abort_signal.registry.new_generation()
    first_call_id = f"inline-{uuid4().hex}"
    register_macro_display(first_call_id, command="\n".join(commands), argv=())
    outputs: list[str] = []
    started_at: float | None = None
    aggregate_output = 0

    def first_lifecycle(kind: str, call: ToolCall | None = None) -> None:
        nonlocal started_at
        if kind == "execution_start" and started_at is None:
            started_at = time.monotonic()
        if kind in {"approval_start", "approval_end"} and call is not None:
            lifecycle_sink(kind, call)

    try:
        for index, command in enumerate(commands):
            if index >= max_spans:
                outputs.extend(
                    INLINE_SHELL_SPAN_LIMIT_MESSAGE for _ in commands[index:]
                )
                break
            remaining_time = (
                timeout
                if started_at is None
                else batch_timeout - (time.monotonic() - started_at)
            )
            if remaining_time <= 0:
                outputs.extend(
                    INLINE_SHELL_BATCH_TIMEOUT_MESSAGE for _ in commands[index:]
                )
                break
            call_id = first_call_id if index == 0 else f"inline-{uuid4().hex}"
            call = ToolCall(
                call_id,
                "bash",
                {
                    "command": command,
                    "timeout": min(timeout, remaining_time),
                    "max_output": output_limit,
                },
            )
            result = await registry.execute(
                call,
                abort_signal=scope_signal,
                _scope_signal=scope_signal,
                _persist_approval=False,
                _capture_output=True,
                _skip_approval=index > 0,
                _lifecycle_sink=(
                    lambda kind, call=call: first_lifecycle(kind, call)
                )
                if index == 0
                else None,
            )
            if started_at is not None and time.monotonic() - started_at >= batch_timeout:
                outputs.append(INLINE_SHELL_BATCH_TIMEOUT_MESSAGE)
                outputs.extend(
                    INLINE_SHELL_BATCH_TIMEOUT_MESSAGE
                    for _ in commands[index + 1 :]
                )
                break
            if _inline_shell_was_denied(result):
                outputs.append(_inline_shell_failure(result))
                outputs.extend(
                    "[inline shell failed: denied]" for _ in commands[index + 1 :]
                )
                break
            structured = result.get("structuredContent")
            if result.get("isError") is not True and isinstance(structured, dict):
                stdout = structured.get("stdout")
                if isinstance(stdout, str):
                    output_size = len(stdout.encode("utf-8"))
                    if aggregate_output + output_size > total_output_limit:
                        outputs.append(INLINE_SHELL_OUTPUT_LIMIT_MESSAGE)
                        outputs.extend(
                            INLINE_SHELL_OUTPUT_LIMIT_MESSAGE
                            for _ in commands[index + 1 :]
                        )
                        break
                    outputs.append(stdout)
                    aggregate_output += output_size
                    continue
            outputs.append(_inline_shell_failure(result))
            if _inline_shell_was_canceled(result):
                outputs.extend(
                    "[inline shell failed: canceled]" for _ in commands[index + 1 :]
                )
                break
    finally:
        if registry.approval_policy is not None:
            registry.approval_policy.forget_ephemeral(first_call_id)
        forget_macro_display(first_call_id)
    return tuple(outputs)


def _inline_shell_was_canceled(result: StructuredToolResult) -> bool:
    return result.get("isCanceled") is True


def _inline_shell_was_denied(result: StructuredToolResult) -> bool:
    content = result.get("content")
    return result.get("isError") is True and isinstance(content, list) and any(
        isinstance(block, dict)
        and isinstance(block.get("text"), str)
        and block["text"].startswith("tool execution denied")
        for block in content
    )


def _inline_shell_failure(result: StructuredToolResult) -> str:
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured.get("timed_out") is True:
        reason = "timed out"
    elif isinstance(structured, dict) and isinstance(structured.get("exit_code"), int):
        reason = f"exit {structured['exit_code']}"
    else:
        content = result.get("content")
        content_text = (
            content[0].get("text")
            if isinstance(content, list)
            and content
            and isinstance(content[0], dict)
            else None
        )
        reason = (
            "denied"
            if isinstance(content_text, str)
            and content_text.startswith("tool execution denied")
            else "canceled"
        )
    return f"[inline shell failed: {reason}]"
