"""The built-in shell execution tool."""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.abort import AbortSignal
from ..types import (
    StreamEvent,
    StructuredToolResult,
    ToolCall,
    ToolResult,
    ToolTextBlock,
    flatten_tool_content,
)
from ._process import _kill_and_reap
from .registry import (
    ToolRegistry,
    ToolStream,
    ToolStreamPublisher,
    _BoundedText,
    _success_result,
    _ToolCanceled,
    validate_tool_result,
)

INLINE_SHELL_TIMEOUT = 10.0
INLINE_SHELL_OUTPUT_LIMIT = 8_192


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
    """Return the trusted display data for a call the macro runner registered."""

    return _trusted_macro_display.get(call_id)


def forget_macro_display(call_id: str) -> None:
    """Drop the trusted display entry once its owning macro is done."""

    _trusted_macro_display.pop(call_id, None)


class _BoundedOutput:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._data = bytearray()
        self._full_size = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    @property
    def data(self) -> bytes:
        return bytes(self._data)

    @property
    def retained_bytes(self) -> int:
        return len(self._data)

    @property
    def full_size(self) -> int:
        return self._full_size

    def append(self, chunk: bytes) -> None:
        self._full_size += len(chunk)
        self._append_text(self._decoder.decode(chunk, final=False))

    def finish(self) -> None:
        self._append_text(self._decoder.decode(b"", final=True))

    def _append_text(self, text: str) -> None:
        for character in text:
            encoded = character.encode("utf-8")
            remaining = self.limit - len(self._data)
            if remaining >= len(encoded):
                self._data.extend(encoded)


async def _drain_stream(
    stream: object,
    capture: _BoundedOutput,
    *,
    stream_name: ToolStream,
    stream_publisher: ToolStreamPublisher | None,
    log_handle: Any | None,
) -> None:
    if stream is None:
        return
    read = getattr(stream, "read")  # noqa: B009 - process pipe interface
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = await read(65_536)
        if not chunk:
            capture.finish()
            text = decoder.decode(b"", final=True)
            if stream_publisher is not None and text:
                stream_publisher.publish(text, stream_name)
            return
        capture.append(chunk)
        if log_handle is not None:
            log_handle.write(chunk)
            log_handle.flush()
        if stream_publisher is not None:
            text = decoder.decode(chunk, final=False)
            if text:
                stream_publisher.publish(text, stream_name)


def _format_exec_result(
    returncode: int | None,
    stdout: bytes,
    stderr: bytes,
    output_limit: int,
    *,
    suffix: str | None = None,
    stdout_full_size: int | None = None,
    stderr_full_size: int | None = None,
) -> ToolTextBlock:
    output = _BoundedText(output_limit)
    output.append(f"exit_code: {returncode}\n")
    if suffix:
        output.append(f"{suffix}\n")
    output.append("stdout:\n")
    output.append_captured(
        stdout.decode(errors="replace"),
        len(stdout) if stdout_full_size is None else stdout_full_size,
    )
    output.append("\nstderr:\n")
    output.append_captured(
        stderr.decode(errors="replace"),
        len(stderr) if stderr_full_size is None else stderr_full_size,
    )
    return output.render()


async def _exec(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
) -> StructuredToolResult:
    timeout = arguments.get("timeout", 30.0)
    output_limit = arguments.get("max_output", registry.max_output_chars)
    log_path = arguments.get("_log_path")
    if arguments.get("_background") is True:
        task_id, pid = await registry.background_tasks.start(
            arguments["command"],
            registry.cwd,
            log_path=log_path,
        )
        return _success_result(
            {
                "type": "text",
                "text": f"background task {task_id} started (pid {pid})",
                "truncated": False,
                "full_size": len(
                    f"background task {task_id} started (pid {pid})".encode()
                ),
            },
            structured_content={
                "status": "running",
                "task_id": task_id,
                "pid": pid,
                "cwd": str(registry.cwd),
                **(
                    {"log_path": str(log_path)}
                    if log_path is not None
                    else {}
                ),
            },
        )
    log_handle = (
        await asyncio.to_thread(Path(log_path).open, "wb")
        if log_path is not None
        else None
    )
    try:
        process = await asyncio.create_subprocess_shell(
            arguments["command"],
            cwd=registry.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        if log_handle is not None:
            log_handle.close()
        raise ValueError(f"could not execute command: {exc}") from exc

    stdout_capture = _BoundedOutput(output_limit)
    stderr_capture = _BoundedOutput(output_limit)
    process_wait = asyncio.create_task(process.wait())
    stdout_drain = asyncio.create_task(
        _drain_stream(
            process.stdout,
            stdout_capture,
            stream_name="stdout",
            stream_publisher=stream_publisher,
            log_handle=log_handle,
        )
    )
    stderr_drain = asyncio.create_task(
        _drain_stream(
            process.stderr,
            stderr_capture,
            stream_name="stderr",
            stream_publisher=stream_publisher,
            log_handle=log_handle,
        )
    )
    process_tasks = (process_wait, stdout_drain, stderr_drain)
    abort_wait = asyncio.create_task(abort_signal.wait())
    timeout_wait = asyncio.create_task(asyncio.sleep(timeout))
    try:
        pending: set[asyncio.Task[Any]] = {*process_tasks, abort_wait, timeout_wait}
        while True:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if abort_signal.is_set():
                await _kill_and_reap(process, process_tasks)
                raise _ToolCanceled()
            if all(task.done() for task in process_tasks):
                process_wait.result()
                result = _format_exec_result(
                    process.returncode,
                    stdout_capture.data,
                    stderr_capture.data,
                    output_limit,
                    stdout_full_size=stdout_capture.full_size,
                    stderr_full_size=stderr_capture.full_size,
                )
                structured_content = {
                    "exit_code": process.returncode,
                    "cwd": str(registry.cwd),
                    **(
                        {"log_path": str(log_path)}
                        if log_path is not None
                        else {}
                    ),
                }
                if arguments.get("_capture_output") is True:
                    structured_content.update(
                        {
                            "stdout": stdout_capture.data.decode(errors="replace"),
                            "stderr": stderr_capture.data.decode(errors="replace"),
                        }
                    )
                if process.returncode:
                    return {
                        "content": [result],
                        "isError": True,
                        "structuredContent": structured_content,
                    }
                return _success_result(
                    result,
                    structured_content=structured_content,
                )
            if timeout_wait in done:
                break

        await _kill_and_reap(process, process_tasks)
        structured_content = {
            "timed_out": True,
            "exit_code": process.returncode,
            "cwd": str(registry.cwd),
            **({"log_path": str(log_path)} if log_path is not None else {}),
        }
        if arguments.get("_capture_output") is True:
            structured_content.update(
                {
                    "stdout": stdout_capture.data.decode(errors="replace"),
                    "stderr": stderr_capture.data.decode(errors="replace"),
                }
            )
        return {
            "content": [
                _format_exec_result(
                    process.returncode,
                    stdout_capture.data,
                    stderr_capture.data,
                    output_limit,
                    suffix="command timed out",
                    stdout_full_size=stdout_capture.full_size,
                    stderr_full_size=stderr_capture.full_size,
                )
            ],
            "isError": True,
            "structuredContent": structured_content,
        }
    except asyncio.CancelledError:
        await _kill_and_reap(process, process_tasks)
        raise
    except BaseException:
        await _kill_and_reap(process, process_tasks)
        raise
    finally:
        for waiter in (abort_wait, timeout_wait):
            if not waiter.done():
                waiter.cancel()
        await asyncio.gather(abort_wait, timeout_wait, return_exceptions=True)
        if log_handle is not None:
            log_handle.close()


async def run_exec_macro(
    registry: ToolRegistry,
    call: ToolCall,
    log_path: str | Path,
    *,
    stream_sink: Callable[[StreamEvent], None],
    lifecycle_sink: Callable[[str], None],
    abort_signal: AbortSignal | None = None,
    background: bool = False,
) -> ToolResult:
    """Run an approved macro without persisting tool conversation entries."""

    Path(log_path).touch()
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
        )
    except asyncio.CancelledError:
        scope_signal.abort()
        return ToolResult(
            call.id,
            "tool execution canceled",
            True,
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
) -> tuple[str, ...]:
    """Run inline shell spans with one approval for the complete batch."""

    if not commands:
        return ()
    scope_signal = abort_signal or registry.abort_signal.registry.new_generation()
    first_call_id = f"inline-{uuid4().hex}"
    register_macro_display(first_call_id, command="\n".join(commands), argv=())
    outputs: list[str] = []
    try:
        for index, command in enumerate(commands):
            call_id = first_call_id if index == 0 else f"inline-{uuid4().hex}"
            call = ToolCall(
                call_id,
                "exec",
                {
                    "command": command,
                    "timeout": timeout,
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
                    lambda kind, call=call: lifecycle_sink(kind, call)
                    if kind in {"approval_start", "approval_end"}
                    else None
                )
                if index == 0
                else None,
            )
            structured = result.get("structuredContent")
            if result.get("isError") is not True and isinstance(structured, dict):
                stdout = structured.get("stdout")
                if isinstance(stdout, str):
                    outputs.append(stdout)
                    continue
            outputs.append(_inline_shell_failure(result))
            if _inline_shell_was_canceled(result):
                outputs.extend("[inline shell failed: canceled]" for _ in commands[index + 1 :])
                break
    finally:
        if registry.approval_policy is not None:
            registry.approval_policy.forget_ephemeral(first_call_id)
        forget_macro_display(first_call_id)
    return tuple(outputs)


def _inline_shell_was_canceled(result: StructuredToolResult) -> bool:
    content = result.get("content")
    return result.get("isError") is True and isinstance(content, list) and any(
        isinstance(block, dict) and block.get("text") == "tool execution canceled"
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
        reason = "denied" if content_text == "tool execution denied" else "canceled"
    return f"[inline shell failed: {reason}]"


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "exec",
        _exec,
        description=(
            "Run a shell command from the session cwd. "
            "This tool is not a sandbox."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "timeout": {"type": "number", "exclusiveMinimum": 0},
                "max_output": {"type": "integer", "minimum": 1},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )
