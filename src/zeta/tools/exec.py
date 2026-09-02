"""The built-in shell execution tool."""

from __future__ import annotations

import asyncio
import codecs
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    while True:
        chunk = await read(65_536)
        if not chunk:
            capture.finish()
            return
        capture.append(chunk)
        if log_handle is not None:
            log_handle.write(chunk)
            log_handle.flush()
        if stream_publisher is not None:
            text = chunk.decode(errors="replace")
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
                if process.returncode:
                    return {
                        "content": [result],
                        "isError": True,
                        "structuredContent": {
                            "exit_code": process.returncode,
                            "cwd": str(registry.cwd),
                            **(
                                {"log_path": str(log_path)}
                                if log_path is not None
                                else {}
                            ),
                        },
                    }
                return _success_result(
                    result,
                    structured_content={
                        "exit_code": process.returncode,
                        "cwd": str(registry.cwd),
                        **({"log_path": str(log_path)} if log_path is not None else {}),
                    },
                )
            if timeout_wait in done:
                break

        await _kill_and_reap(process, process_tasks)
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
            "structuredContent": {
                "exit_code": process.returncode,
                "cwd": str(registry.cwd),
                **({"log_path": str(log_path)} if log_path is not None else {}),
            },
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
) -> ToolResult:
    """Run an approved macro without persisting tool conversation entries."""

    Path(log_path).touch()
    registry.start_batch()
    try:
        raw_result = await registry.execute(
            call,
            _stream_sink=stream_sink,
            _lifecycle_sink=lifecycle_sink,
            _persist_approval=False,
            _log_path=log_path,
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
        registry.abort()
        return ToolResult(
            call.id,
            "tool execution canceled",
            True,
            structured_content={"log_path": str(log_path)},
        )


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
