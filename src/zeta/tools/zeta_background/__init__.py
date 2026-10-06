"""Background process management tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from ...core.approval import ApprovedCwdExecution
from ...protocol.types import StructuredToolResult
from .._shared.sandbox import expand_user_path
from ..registry import ToolExecutionContext, ToolRegistry, _success_result, text_block


class BackgroundArguments(TypedDict, total=False):
    command: str
    cwd: str | None


def _result(content: str, structured: dict[str, Any]) -> StructuredToolResult:
    return _success_result(text_block(content), structured_content=structured)


async def _run_background(
    registry: ToolRegistry,
    arguments: BackgroundArguments,
    *,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    cwd = arguments.get("cwd")
    start_cwd = registry.cwd
    if cwd is not None:
        if type(cwd) is not str or not cwd:
            raise ValueError("cwd must be a nonempty string or null")
        candidate = Path(expand_user_path(cwd))
        if not candidate.is_absolute():
            candidate = registry.cwd / candidate
        start_cwd = Path(os.path.abspath(candidate))
    approved_execution = (
        execution_context.approved_execution
        if execution_context is not None
        else None
    )
    registry.verify_cwd_identity()
    cwd_fd = None
    if isinstance(approved_execution, ApprovedCwdExecution):
        start_cwd = Path(approved_execution.cwd)
        cwd_fd = registry.open_verified_directory(
            start_cwd, approved_execution.identity
        )
    task_id, pid = await registry.background_tasks.start(
        arguments["command"], start_cwd, cwd_fd=cwd_fd
    )
    return _result(
        f"started background task {task_id} (pid {pid})",
        {"task_id": task_id, "pid": pid, "running": True},
    )


async def _task_output(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> StructuredToolResult:
    task_id = arguments["task_id"]
    since = arguments.get("since")
    wait_seconds = arguments.get("wait_seconds", 0)
    if wait_seconds:
        await registry.background_tasks.wait(task_id, timeout=wait_seconds)
    output_limit = registry.max_output_chars
    for _ in range(8):
        result = await registry.background_tasks.output(
            task_id,
            since,
            max_chars=output_limit,
        )
        output = result["output"]
        note = result.get("note")
        metadata = (
            f"task_id: {result['task_id']}\n"
            f"cursor: {result['cursor']}\n"
            f"running: {result['running']}\n"
            f"exit_code: {result['exit_code']}\n"
        )
        more = ""
        if result["has_more"]:
            more = (
                "\n"
                "has_more: true\n"
                f"next_cursor: {result['cursor']}\n"
                f"total_bytes: {result['total_bytes']}\n"
                f"remaining_bytes: {result['remaining_bytes']}\n"
                f"output_location: {result['output_location']}\n"
                f"retrieve with task_output (since={result['cursor']}); do not use read"
            )
        content = metadata + (output if output else (note or "")) + more
        overflow = len(content) - registry.max_output_chars
        if overflow <= 0:
            break
        next_limit = max(0, output_limit - overflow)
        if next_limit == output_limit:
            break
        output_limit = next_limit
    return _result(content, result)


async def _task_input(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> StructuredToolResult:
    result = await registry.background_tasks.input(
        arguments["task_id"],
        arguments.get("data", ""),
        eof=arguments.get("eof", False),
    )
    if result["status"] == "indeterminate":
        suffix = "; delivery indeterminate, do not retry"
    else:
        suffix = " and EOF closed" if result["eof"] else ""
    return _result(
        f"queued {result['bytes_written']} bytes to background task "
        f"{result['task_id']} stdin{suffix}",
        result,
    )


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
    registry.register_session_tool(
        "run_background",
        _run_background,
        approval_subject="command",
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
    registry.register_session_tool(
        "task_output",
        _task_output,
        description=(
            "Read incremental output and status from a background task. "
            "Set wait_seconds to wait up to 300 seconds for completion. Large output "
            "uses a task-output://<task-id> location; retrieve it only with repeated "
            "task_output calls using the returned since cursor, not with read."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "since": {"type": "integer", "minimum": 0},
                "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 300},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
    registry.register_session_tool(
        "task_input",
        _task_input,
        approval_subject="data",
        description=(
            "Write bounded UTF-8 text to a live background task's stdin. "
            "Each data chunk is independently approval-scoped; approval for the "
            "original command does not authorize later input. Writes are serialized "
            "and backpressure-aware; set eof to close stdin."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "data": {"type": "string"},
                "eof": {"type": "boolean"},
            },
            "required": ["task_id"],
            "additionalProperties": False,
        },
    )
    registry.register_session_tool(
        "task_kill",
        _task_kill,
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
