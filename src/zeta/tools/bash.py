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
from .registry import ToolRegistry, _error_result, text_block


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
    communicate: asyncio.Task[tuple[bytes, bytes]] | None = None
    abort_wait: asyncio.Task[None] | None = None
    reported_cwd = ""
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
        communicate = asyncio.create_task(process.communicate())
        abort_wait = asyncio.create_task(abort_signal.wait())
        done, _ = await asyncio.wait(
            (communicate, abort_wait),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_wait in done and communicate not in done:
            await _kill_and_reap(process, (communicate,))
            return _error_result("tool execution canceled")
        stdout_bytes, stderr_bytes = await communicate
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
        if process is not None and communicate is not None:
            await _kill_and_reap(process, (communicate,))
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
        lambda arguments, abort_signal: _bash(registry, arguments, abort_signal),
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
