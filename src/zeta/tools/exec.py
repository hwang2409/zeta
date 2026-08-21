"""The built-in shell execution tool."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Sequence
from typing import Any

from ..core.abort import AbortSignal
from ..types import StructuredToolResult, ToolTextBlock
from .registry import (
    _BoundedText,
    _success_result,
    ToolRegistry,
    _ToolCanceled,
)


class _BoundedOutput:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._data = bytearray()
        self._full_size = 0

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
        remaining = self.limit - len(self._data)
        if remaining > 0:
            self._data.extend(chunk[:remaining])


async def _drain_stream(stream: object, capture: _BoundedOutput) -> None:
    if stream is None:
        return
    read = getattr(stream, "read")
    while True:
        chunk = await read(65_536)
        if not chunk:
            return
        capture.append(chunk)


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    process_tasks: Sequence[asyncio.Task[Any]],
) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()
    for task in process_tasks:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        if not task.cancelled():
            task.exception()


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
) -> StructuredToolResult:
    timeout = arguments.get("timeout", 30.0)
    output_limit = arguments.get("max_output", registry.max_output_chars)
    try:
        process = await asyncio.create_subprocess_shell(
            arguments["command"],
            cwd=registry.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ValueError(f"could not execute command: {exc}") from exc

    stdout_capture = _BoundedOutput(output_limit)
    stderr_capture = _BoundedOutput(output_limit)
    process_wait = asyncio.create_task(process.wait())
    stdout_drain = asyncio.create_task(_drain_stream(process.stdout, stdout_capture))
    stderr_drain = asyncio.create_task(_drain_stream(process.stderr, stderr_capture))
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
                        },
                    }
                return _success_result(
                    result,
                    structured_content={
                        "exit_code": process.returncode,
                        "cwd": str(registry.cwd),
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


def register(registry: ToolRegistry) -> None:
    registry.register(
        "exec",
        lambda arguments, abort_signal: _exec(registry, arguments, abort_signal),
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
