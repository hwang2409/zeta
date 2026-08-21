"""The built-in general shell tool.

Sandboxing is truly best-effort and weaker than the file tools. An explicit
``cwd`` is checked lexically against the session root, but shell commands can
change directory to any path. The reported shell cwd is persisted as session
state for the next call.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import uuid
from pathlib import Path
from typing import TypedDict

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from ._process import _kill_and_reap
from .registry import (
    ToolRegistry,
    ToolStream,
    ToolStreamPublisher,
    _error_result,
    text_block,
)


class BashArguments(TypedDict, total=False):
    cmd: str
    cwd: str | None


class BashStructuredContent(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    cwd_after: str


def _start_cwd(registry: ToolRegistry, arguments: BashArguments) -> str:
    cwd = arguments.get("cwd")
    if cwd is None:
        return registry.bash_cwd
    if type(cwd) is not str or not cwd:
        raise ValueError("cwd must be a nonempty string or null")
    candidate = Path(cwd)
    normalized = Path(os.path.normpath(cwd))
    if candidate.is_absolute():
        inside = normalized == registry.cwd or registry.cwd in normalized.parents
    else:
        inside = not (normalized == Path("..") or Path("..") in normalized.parents)
    if not inside:
        raise ValueError("path escaped sandbox")
    return os.path.abspath(
        os.fspath(candidate if candidate.is_absolute() else registry.cwd / candidate)
    )


async def _bash(
    registry: ToolRegistry,
    arguments: BashArguments,
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
) -> StructuredToolResult:
    start_cwd = _start_cwd(registry, arguments)
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
        [
            "bash",
            "-c",
            script,
            "zeta-bash",
            start_cwd,
            arguments["cmd"],
        ]
    )
    process: asyncio.subprocess.Process | None = None
    stdout_reader: asyncio.Task[bytes] | None = None
    stderr_reader: asyncio.Task[bytes] | None = None
    process_wait: asyncio.Task[int] | None = None
    abort_wait: asyncio.Task[None] | None = None
    stdout_bytes = b""
    stderr_bytes = b""
    reported_cwd = ""

    async def read_pipe(
        pipe: asyncio.StreamReader,
        stream: ToolStream,
    ) -> bytes:
        chunks: list[bytes] = []
        while chunk := await pipe.read(65_536):
            chunks.append(chunk)
            if stream_publisher is not None and not abort_signal.is_set():
                stream_publisher.publish(chunk.decode(errors="replace"), stream)
        return b"".join(chunks)

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=registry.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        write_fd = -1
        if process.stdout is None or process.stderr is None:
            raise OSError("command output pipes were not created")
        stdout_reader = asyncio.create_task(read_pipe(process.stdout, "stdout"))
        stderr_reader = asyncio.create_task(read_pipe(process.stderr, "stderr"))
        process_wait = asyncio.create_task(process.wait())
        abort_wait = asyncio.create_task(abort_signal.wait())
        done, _ = await asyncio.wait(
            (process_wait, abort_wait),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_wait in done and process_wait not in done:
            await _kill_and_reap(
                process,
                (stdout_reader, stderr_reader, process_wait),
            )
            return _error_result("tool execution canceled")
        stdout_bytes, stderr_bytes = await asyncio.gather(
            stdout_reader,
            stderr_reader,
        )
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
        for line in channel_data.splitlines():
            if line.startswith(nonce_prefix):
                candidate = line[len(nonce_prefix) :].decode(errors="replace")
                if Path(candidate).is_absolute():
                    reported_cwd = candidate
    except asyncio.CancelledError:
        if (
            process is not None
            and stdout_reader is not None
            and stderr_reader is not None
            and process_wait is not None
        ):
            await _kill_and_reap(
                process,
                (stdout_reader, stderr_reader, process_wait),
            )
        raise
    except OSError as exc:
        raise ValueError(f"could not execute command: {exc}") from exc
    finally:
        if abort_wait is not None and not abort_wait.done():
            abort_wait.cancel()
            await asyncio.gather(abort_wait, return_exceptions=True)
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)

    stdout = stdout_bytes.decode(errors="replace")
    cwd_after = start_cwd
    if reported_cwd and Path(reported_cwd).is_absolute():
        cwd_after = reported_cwd
    stderr = stderr_bytes.decode(errors="replace")
    exit_code = process.returncode if process.returncode is not None else 1
    if cwd_after != start_cwd:
        registry.update_bash_cwd(cwd_after)
    combined = "stdout:\n" + stdout + "\nstderr:\n" + stderr
    structured_content: BashStructuredContent = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "cwd_after": cwd_after,
    }
    return {
        "content": [text_block(combined)],
        "isError": exit_code != 0,
        "structuredContent": structured_content,
    }


def register(registry: ToolRegistry) -> None:
    registry.register(
        "bash",
        lambda arguments, abort_signal, stream_publisher=None: _bash(
            registry,
            arguments,
            abort_signal,
            stream_publisher,
        ),
        description=(
            "Run a shell command. Session cwd persists after cd. "
            "Sandboxing is truly best-effort."
        ),
        parameters={
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "minLength": 1},
                "cwd": {},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    )
