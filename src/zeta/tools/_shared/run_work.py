"""Shared background-task and todo projections for runtime views."""

from __future__ import annotations

import time
from collections.abc import Sequence

from ...core.todo import TodoItem
from .background_task_types import BackgroundTaskInfo
from .process import BackgroundTaskRegistry

MAX_WIRE_COMMAND_BYTES = 8_000


def task_status(task: BackgroundTaskInfo) -> str:
    if task.running:
        return "running"
    if task.terminal_phase in {"task_kill", "session_shutdown"}:
        return "killed"
    return "exited"


def task_elapsed_seconds(task: BackgroundTaskInfo, now: float) -> float:
    if task.started_at is None:
        return 0.0
    end = task.ended_at if task.ended_at is not None else now
    return max(0.0, end - task.started_at)


def _bounded_command(command: str) -> str:
    encoded = command.encode("utf-8")
    if len(encoded) <= MAX_WIRE_COMMAND_BYTES:
        return command
    return encoded[:MAX_WIRE_COMMAND_BYTES].decode("utf-8", errors="ignore")


def background_task_row(
    task: BackgroundTaskInfo, now: float | None = None
) -> dict[str, object]:
    current = time.monotonic() if now is None else now
    row: dict[str, object] = {
        "id": task.task_id,
        "status": task_status(task),
        "command": _bounded_command(task.command),
        "elapsed_seconds": task_elapsed_seconds(task, current),
        "cancelable": task.running,
    }
    if task.exit_code is not None:
        row["exit_code"] = task.exit_code
    return row


def run_work_snapshot(
    registry: BackgroundTaskRegistry,
    todo: Sequence[TodoItem],
    now: float | None = None,
) -> dict[str, object]:
    current = time.monotonic() if now is None else now
    return {
        "background_tasks": [
            background_task_row(task, current) for task in registry.snapshot()
        ],
        "todo": [dict(item) for item in todo],
    }
