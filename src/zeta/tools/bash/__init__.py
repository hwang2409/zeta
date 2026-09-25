"""The built-in general shell tool."""

from __future__ import annotations

import asyncio
import codecs
import os
import shlex
import uuid
from pathlib import Path
from typing import Any, TypedDict

from ...core.abort import AbortSignal
from ...protocol.types import StructuredToolResult
from .._shared.process import _kill_and_reap, tool_subprocess_env
from .._shared.sandbox import expand_user_path
from ..registry import (
    ToolRegistry,
    ToolStream,
    ToolStreamPublisher,
    _error_result,
    _success_result,
    text_block,
)


class BashArguments(TypedDict, total=False):
    command: str
    cmd: str
    cwd: str | None
    timeout: float
    max_output: int


class BashStructuredContent(TypedDict):
    stdout: str
    stderr: str
    exit_code: int | None
    cwd_after: str


class _OutputCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._data = bytearray()
        self.full_size = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    @property
    def data(self) -> bytes:
        return bytes(self._data)

    def append(self, chunk: bytes) -> None:
        self.full_size += len(chunk)
        self._append_text(self._decoder.decode(chunk, final=False))

    def finish(self) -> None:
        self._append_text(self._decoder.decode(b"", final=True))

    def _append_text(self, text: str) -> None:
        for character in text:
            encoded = character.encode("utf-8")
            if len(self._data) + len(encoded) > self.limit:
                break
            self._data.extend(encoded)


def _extract_command(arguments: BashArguments) -> str:
    command = arguments.get("command")
    if isinstance(command, str) and command:
        return command
    legacy = arguments.get("cmd")
    if isinstance(legacy, str) and legacy:
        return legacy
    raise ValueError("command is required (accepts legacy alias cmd)")


def _start_cwd(registry: ToolRegistry, arguments: BashArguments) -> str:
    cwd = arguments.get("cwd")
    if cwd is None:
        return registry.bash_cwd
    if type(cwd) is not str or not cwd:
        raise ValueError("cwd must be a nonempty string or null")
    candidate = Path(expand_user_path(cwd))
    if not candidate.is_absolute():
        candidate = registry.cwd / candidate
    return os.path.abspath(candidate)


async def _read_pipe(
    pipe: asyncio.StreamReader,
    capture: _OutputCapture,
    stream: ToolStream,
    stream_publisher: ToolStreamPublisher | None,
    log_handle: Any | None,
    abort_signal: AbortSignal,
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while chunk := await pipe.read(65_536):
        capture.append(chunk)
        if log_handle is not None:
            log_handle.write(chunk)
            log_handle.flush()
        if stream_publisher is not None and not abort_signal.is_set():
            text = decoder.decode(chunk, final=False)
            if text:
                stream_publisher.publish(text, stream)
    text = decoder.decode(b"", final=True)
    if stream_publisher is not None and not abort_signal.is_set() and text:
        stream_publisher.publish(text, stream)
    capture.finish()


def _read_cwd_channel(read_fd: int, nonce: str) -> str:
    os.set_blocking(read_fd, False)
    channel_data = bytearray()
    while True:
        try:
            chunk = os.read(read_fd, 65_536)
        except BlockingIOError:
            break
        if not chunk:
            break
        channel_data.extend(chunk)
    nonce_prefix = f"{nonce}\t".encode()
    reported_cwd = ""
    for line in channel_data.splitlines():
        if line.startswith(nonce_prefix):
            candidate = line[len(nonce_prefix) :].decode(errors="replace")
            if Path(candidate).is_absolute():
                reported_cwd = candidate
    return reported_cwd


async def _bash(
    registry: ToolRegistry,
    arguments: BashArguments,
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
) -> StructuredToolResult:
    start_cwd = _start_cwd(registry, arguments)
    timeout = arguments.get("timeout", 30.0)
    output_limit = arguments.get("max_output", registry.max_output_chars)
    log_path = arguments.get("_log_path")
    if arguments.get("_background") is True:
        task_id, pid = await registry.background_tasks.start(
            _extract_command(arguments), start_cwd, log_path=log_path
        )
        message = f"background task {task_id} started (pid {pid})"
        return _success_result(
            text_block(message),
            structured_content={
                "status": "running",
                "task_id": task_id,
                "pid": pid,
                "cwd": start_cwd,
                **({"log_path": str(log_path)} if log_path is not None else {}),
            },
        )

    read_fd, write_fd = os.pipe()
    nonce = uuid.uuid4().hex
    bind_fd = "" if write_fd == 3 else f"exec 3>&{write_fd}; exec {write_fd}>&-; "
    script = (
        f'{bind_fd}'
        'cd -- "$1" || exit $?; '
        f'trap \'printf "%s\\t%s\\n" "{nonce}" "$PWD" >&3\' EXIT; '
        'eval "$2"; status=$?; '
        f'printf "%s\\t%s\\n" "{nonce}" "$PWD" >&3; '
        'exit "$status"'
    )
    command = shlex.join(
        ["bash", "-c", script, "zeta-bash", start_cwd, _extract_command(arguments)]
    )
    process: asyncio.subprocess.Process | None = None
    log_handle = None
    stdout_capture = _OutputCapture(output_limit)
    stderr_capture = _OutputCapture(output_limit)
    stdout_reader: asyncio.Task[None] | None = None
    stderr_reader: asyncio.Task[None] | None = None
    process_wait: asyncio.Task[int] | None = None
    abort_wait: asyncio.Task[None] | None = None
    timeout_wait: asyncio.Task[None] | None = None
    timed_out = False
    try:
        if log_path is not None:
            log_handle = registry.background_tasks.open_log(log_path)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=registry.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=(write_fd,),
            env=tool_subprocess_env(),
        )
        os.close(write_fd)
        write_fd = -1
        if process.stdout is None or process.stderr is None:
            raise OSError("command output pipes were not created")
        stdout_reader = asyncio.create_task(
            _read_pipe(process.stdout, stdout_capture, "stdout", stream_publisher, log_handle, abort_signal)
        )
        stderr_reader = asyncio.create_task(
            _read_pipe(process.stderr, stderr_capture, "stderr", stream_publisher, log_handle, abort_signal)
        )
        process_wait = asyncio.create_task(process.wait())
        abort_wait = asyncio.create_task(abort_signal.wait())
        timeout_wait = asyncio.create_task(asyncio.sleep(timeout))
        process_tasks = (process_wait, stdout_reader, stderr_reader)
        pending: set[asyncio.Task[Any]] = set(process_tasks) | {abort_wait, timeout_wait}
        while True:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if abort_wait in done and abort_signal.is_set():
                await _kill_and_reap(process, process_tasks)
                return _error_result("tool execution canceled")
            if all(task.done() for task in process_tasks):
                break
            if timeout_wait in done:
                timed_out = True
                await _kill_and_reap(process, process_tasks)
                break

        reported_cwd = _read_cwd_channel(read_fd, nonce)
        stdout = stdout_capture.data.decode(errors="replace")
        stderr = stderr_capture.data.decode(errors="replace")
        cwd_after = reported_cwd if Path(reported_cwd).is_absolute() else start_cwd
        if cwd_after != start_cwd:
            registry.update_bash_cwd(cwd_after)
        exit_code = process.returncode if process.returncode is not None else 1
        marker = (
            f"[timed out after {timeout:g}s; process group killed]"
            if timed_out
            else None
        )
        combined = "stdout:\n" + stdout + "\nstderr:\n" + stderr
        if marker is not None:
            combined = marker + "\n" + combined
        combined_full_size = (
            len((marker + "\n").encode()) if marker is not None else 0
        )
        combined_full_size += len(b"stdout:\n")
        combined_full_size += stdout_capture.full_size
        combined_full_size += len(b"\nstderr:\n")
        combined_full_size += stderr_capture.full_size
        structured_content: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "cwd_after": cwd_after,
        }
        if timed_out:
            structured_content.update({"timed_out": True, "timeout_seconds": float(timeout)})
        return {
            "content": [
                text_block(combined, cap=output_limit, full_size=combined_full_size)
            ],
            "isError": timed_out or exit_code != 0,
            "structuredContent": structured_content,
        }
    except asyncio.CancelledError:
        if process is not None and stdout_reader is not None and stderr_reader is not None and process_wait is not None:
            await _kill_and_reap(process, (process_wait, stdout_reader, stderr_reader))
        raise
    except OSError as exc:
        raise ValueError(f"could not execute command: {exc}") from exc
    finally:
        for waiter in (abort_wait, timeout_wait):
            if waiter is not None and not waiter.done():
                waiter.cancel()
        await asyncio.gather(*(waiter for waiter in (abort_wait, timeout_wait) if waiter is not None), return_exceptions=True)
        if log_handle is not None:
            log_handle.close()
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "bash",
        _bash,
        approval_subject="command",
        description=(
            "Run a shell command. Session cwd persists after cd. "
            "Paths outside the session cwd are allowed. "
            "Timeouts are in seconds. For a long-running command, use run_background instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "cmd": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Deprecated alias for command.",
                },
                "cwd": {},
                "timeout": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Maximum runtime in seconds.",
                },
                "max_output": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    )
