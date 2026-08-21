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
    marker = f"__ZETA_BASH_CWD_{uuid.uuid4().hex}__"
    script = (
        'cd -- "$1" || exit $?; '
        'eval "$2"; status=$?; '
        'printf "\\n%s%s\\n" "$3" "$PWD"; '
        'exit "$status"'
    )
    command = shlex.join(
        ["bash", "-c", script, "zeta-bash", start_cwd, arguments["cmd"], marker]
    )
    abort_wait: asyncio.Task[None] | None = None
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communicate = asyncio.create_task(process.communicate())
        abort_wait = asyncio.create_task(abort_signal.wait())
        done, _ = await asyncio.wait(
            (communicate, abort_wait),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_wait in done and communicate not in done:
            process.kill()
            await communicate
            return _error_result("tool execution canceled")
        stdout_bytes, stderr_bytes = await communicate
    except OSError as exc:
        raise ValueError(f"could not execute command: {exc}") from exc
    finally:
        if abort_wait is not None and not abort_wait.done():
            abort_wait.cancel()
            await asyncio.gather(abort_wait, return_exceptions=True)

    stdout = stdout_bytes.decode(errors="replace")
    marker_position = stdout.rfind(f"\n{marker}")
    if marker_position < 0:
        cwd_after = start_cwd
    else:
        cwd_after = stdout[marker_position + len(marker) + 1 :].rstrip("\n")
        stdout = stdout[:marker_position]
    stderr = stderr_bytes.decode(errors="replace")
    exit_code = process.returncode if process.returncode is not None else 1
    if exit_code == 0 and cwd_after != start_cwd:
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
            "Run a shell command. Session cwd persists after successful cd. "
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
