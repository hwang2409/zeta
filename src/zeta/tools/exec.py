"""The built-in shell execution tool."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.abort import AbortSignal
from .registry import (
    ToolRegistry,
    _BoundedOutput,
    _ToolCanceled,
    _drain_stream,
    _format_exec_result,
    _kill_and_reap,
)


async def _exec(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
) -> str:
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
                )
                if process.returncode:
                    raise ValueError(result)
                return result
            if timeout_wait in done:
                break

        await _kill_and_reap(process, process_tasks)
        raise ValueError(
            _format_exec_result(
                process.returncode,
                stdout_capture.data,
                stderr_capture.data,
                output_limit,
                suffix="command timed out",
            )
        )
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
