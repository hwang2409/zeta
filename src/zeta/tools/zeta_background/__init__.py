"""Background process management tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from ...types import StructuredToolResult
from ..registry import ToolRegistry, _success_result, text_block


class BackgroundArguments(TypedDict, total=False):
    command: str
    cwd: str | None


def _result(content: str, structured: dict[str, Any]) -> StructuredToolResult:
    return _success_result(text_block(content), structured_content=structured)


async def _run_background(
    registry: ToolRegistry,
    arguments: BackgroundArguments,
) -> StructuredToolResult:
    cwd = arguments.get("cwd")
    start_cwd = registry.cwd
    if cwd is not None:
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
        start_cwd = Path(
            os.path.abspath(
                os.fspath(candidate if candidate.is_absolute() else registry.cwd / candidate)
            )
        )
    task_id, pid = await registry.background_tasks.start(arguments["command"], start_cwd)
    return _result(
        f"started background task {task_id} (pid {pid})",
        {"task_id": task_id, "pid": pid, "running": True},
    )


async def _task_output(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> StructuredToolResult:
    result = await registry.background_tasks.output(
        arguments["task_id"], arguments.get("since")
    )
    output = result["output"]
    note = result.get("note")
    metadata = (
        f"task_id: {result['task_id']}\n"
        f"cursor: {result['cursor']}\n"
        f"running: {result['running']}\n"
        f"exit_code: {result['exit_code']}\n"
    )
    content = metadata + (output if output else (note or ""))
    return _result(content, result)


async def _task_kill(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> StructuredToolResult:
    result = await registry.background_tasks.kill(arguments["task_id"])
    status = "running" if result["running"] else "exited"
    return _result(
        f"background task {result['task_id']} {status}",
        result,
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        "run_background",
        lambda arguments, abort_signal: _run_background(registry, arguments),
        description=(
            "Start a shell command as a session-scoped background task. "
            "Use task_output to monitor it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "cwd": {},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )
    registry.register(
        "task_output",
        lambda arguments, abort_signal: _task_output(registry, arguments),
        description="Read incremental output and status from a background task.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "since": {"type": "integer", "minimum": 0},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
    registry.register(
        "task_kill",
        lambda arguments, abort_signal: _task_kill(registry, arguments),
        description="Terminate a background task process group.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
