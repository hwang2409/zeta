"""Lifecycle values shared by background-task owners and views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .process_output import command_headline

TUI_PRESENTED_TASK_KILL_CAUSES = frozenset(
    {
        "task_kill tool requested by the agent",
        "TUI tasks panel requested by the user",
        "background macro canceled by the user",
    }
)


@dataclass(frozen=True, slots=True)
class BackgroundTaskNotice:
    """Structured lifecycle notice emitted by a background task owner."""

    message: str
    task_id: str
    phase: Literal["started", "natural_exit", "task_kill", "session_shutdown"]
    owner: str


@dataclass(frozen=True, slots=True)
class BackgroundTaskShutdownNotice:
    """Structured notice emitted when the registry kills tasks on shutdown."""

    message: str
    phase: Literal["session_shutdown"] = "session_shutdown"
    tasks: tuple[tuple[str, str], ...] = ()


BackgroundTaskNoticeSinkValue = BackgroundTaskNotice | BackgroundTaskShutdownNotice


@dataclass(frozen=True, slots=True)
class TaskExitInput:
    """Detached task-exit data for the resident delivery flow."""

    task_id: str
    command: str
    exit_code: int | None
    output_tail: str = ""
    log_path: str | None = None
    note: str | None = None
    owner: str = "run_background"
    phase: str = "natural_exit"
    delivery_deadline: float | None = None


@dataclass(frozen=True, slots=True)
class BackgroundTaskInfo:
    """Immutable view of one background task for read-only inspection.

    The timestamps use ``time.monotonic`` so a consumer computes runtime
    against ``time.monotonic()`` without a wall-clock jump corrupting it.
    ``started_at`` is ``None`` for a task recovered from a previous session,
    whose runtime is not knowable.
    """

    task_id: str
    command: str
    pid: int
    owner: str
    running: bool
    exit_code: int | None
    note: str | None
    terminal_phase: str | None
    started_at: float | None
    ended_at: float | None
    output_bytes: int
    output_lines: int

    @property
    def headline(self) -> str:
        return command_headline(self.command)
