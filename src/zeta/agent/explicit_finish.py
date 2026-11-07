"""Shared rules for explicit completion of background children."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..tools._shared.process import BackgroundTaskInfo, BackgroundTaskRegistry
from .presets import AgentPreset

if TYPE_CHECKING:
    from .terminal_fence import TerminalTransitionFence


IDLE_CHILD_FD_FLOOR = 3
IDLE_CHILD_WARNING_THRESHOLD = 16

IdleAction = Literal["external_input", "nudge"]


async def _wait_for_idle_wake(
    wait: Callable[[], Awaitable[None]], timeout: float | None
) -> bool:
    """Return true for an external wake and false for the nudge deadline."""
    if timeout is None:
        await wait()
        return True
    try:
        await asyncio.wait_for(wait(), timeout=max(0.0, timeout))
    except TimeoutError:
        return False
    return True


class IdleNudgeLifecycle:
    """Own idle-period timing, lifetime limits, and atomic nudge admission."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        wait: Callable[
            [Callable[[], Awaitable[None]], float | None], Awaitable[bool]
        ]
        | None = None,
        nudge_seconds: float | None = None,
        nudge_limit: int = 3,
    ) -> None:
        self._clock = clock
        self._wait = wait or _wait_for_idle_wake
        self._nudge_seconds = 10 * 60 if nudge_seconds is None else nudge_seconds
        self._nudge_limit = nudge_limit
        self._idle_started: float | None = None
        self._nudge_count = 0
        self._immediate_reminded = False

    async def reserve_immediate_reminder(
        self, fence: TerminalTransitionFence
    ) -> bool:
        """Reserve one immediate reminder, sharing the nudge limit and fence."""
        if self._immediate_reminded or self._nudge_count >= self._nudge_limit:
            return False
        if not await fence.reserve_idle_nudge():
            return False
        self._nudge_count += 1
        self._immediate_reminded = True
        return True

    def reset_idle_period(self) -> None:
        self._immediate_reminded = False

    async def wait_for_idle_action(
        self,
        wake: Callable[[], Awaitable[None]],
        fence: TerminalTransitionFence,
    ) -> IdleAction:
        eligible = await fence.idle_nudge_eligible()
        if eligible and self._nudge_count < self._nudge_limit:
            now = self._clock()
            if self._idle_started is None:
                self._idle_started = now
            timeout = self._nudge_seconds - (now - self._idle_started)
        else:
            self._idle_started = None
            timeout = None

        if await self._wait(wake, timeout):
            self._idle_started = None
            return "external_input"
        if not await fence.reserve_idle_nudge():
            self._idle_started = None
            return "external_input"

        self._nudge_count += 1
        self._idle_started = None
        return "nudge"


def idle_resource_usage(states: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Return the known resource floor for one root agent tree."""

    idle_count = sum(
        1
        for state in states
        if isinstance(state, dict) and state.get("state") == "idle"
    )
    fd_floor = idle_count * IDLE_CHILD_FD_FLOOR
    return {
        "idle_child_count": idle_count,
        "idle_child_fd_floor": fd_floor,
        "idle_child_warning": idle_count > IDLE_CHILD_WARNING_THRESHOLD,
    }


def format_idle_resource_usage(usage: dict[str, object]) -> str:
    """Format one stable status line for tools and agent cards."""

    count = usage["idle_child_count"]
    floor = usage["idle_child_fd_floor"]
    warning = "; warning: more than 16 idle children" if usage["idle_child_warning"] else ""
    return f"idle children: {count}; known fd floor: {floor}{warning}"


def eligible(*, global_enabled: bool, preset: AgentPreset, background: bool) -> bool:
    """Return whether one spawned child must finish explicitly."""

    return (
        global_enabled
        and preset.explicit_finish
        and background
        and preset.accepts_follow_ups
    )


def owned_work_descriptions(child_loop: object) -> tuple[str, ...]:
    """Describe tasks and live children directly owned by one child."""

    children = child_loop._background_owner.owned_running(child_loop.agent_instance_id)
    tasks = tuple(
        (task.task_id, task.headline)
        for task in child_loop.tool_registry.background_tasks.snapshot()
        if task.running
    )
    return tuple(description for _, description in (*children, *tasks))


def register_handoff(child_loop: object) -> None:
    """Register the existing child-adoption operation for an explicit finisher."""

    from .adoption import adopt_agent_children
    from .finish_gate import _register_handoff

    async def handoff(arguments: dict[str, object]) -> str:
        owner = child_loop._background_owner
        child_id = child_loop.agent_instance_id
        parent_store, parent_id = owner.parent_edge(child_id, child_loop.store, None)
        adopt_agent_children(
            child_loop.store,
            parent_store,
            background_owner=owner,
            parent_instance_id=parent_id,
        )
        return (
            "background child agents handed off; "
            f"reason: {arguments['reason']}; outputs: {arguments['outputs']}"
        )

    _register_handoff(child_loop, handoff)


@dataclass(slots=True)
class _TaskSafetyWindow:
    task: BackgroundTaskInfo
    deadline: float
    warned: bool = False


class TaskSafetyLifecycle:
    """Wake a task owner before enforcing one per-task safety window."""

    def __init__(
        self,
        limit: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[
            [Callable[[], Awaitable[None]], float | None], Awaitable[bool]
        ]
        | None = None,
    ) -> None:
        self._limit = limit
        self._clock = clock
        self._wait = wait or _wait_for_idle_wake
        self._changed = asyncio.Event()
        self._windows: dict[str, _TaskSafetyWindow] = {}

    def owner_acted(self) -> None:
        """Start a fresh window for every task whose warning got an owner turn."""

        now = self._clock()
        for window in self._windows.values():
            if window.warned:
                window.deadline = now + self._limit
                window.warned = False
        self._changed.set()

    async def run(
        self,
        registry: BackgroundTaskRegistry,
        publish_notice: Callable[[str], Awaitable[None]],
    ) -> None:
        while True:
            running = {task.task_id: task for task in registry.snapshot() if task.running}
            self._windows = {
                task_id: window
                for task_id, window in self._windows.items()
                if task_id in running
            }
            now = self._clock()
            for task_id, task in running.items():
                if task_id not in self._windows:
                    started = task.started_at if task.started_at is not None else now
                    self._windows[task_id] = _TaskSafetyWindow(
                        task, started + self._limit
                    )

            deadline = min(
                (window.deadline for window in self._windows.values()),
                default=None,
            )
            self._changed.clear()
            timeout = None if deadline is None else max(0.0, deadline - now)
            if await self._wait(self._changed.wait, timeout):
                continue

            now = self._clock()
            for window in tuple(self._windows.values()):
                if window.deadline > now:
                    continue
                age = max(
                    0.0,
                    now
                    - (
                        window.task.started_at
                        if window.task.started_at is not None
                        else now - self._limit
                    ),
                )
                if window.warned:
                    await registry.kill(
                        window.task.task_id,
                        cause=(
                            "explicit-finish task safety limit expired at task age "
                            f"{age:g} seconds after no owner turn for "
                            f"{self._limit:g} seconds"
                        ),
                    )
                    continue
                window.warned = True
                window.deadline += self._limit
                await publish_notice(
                    f"task {window.task.task_id} ({window.task.headline}) has run "
                    f"{_format_duration(age)}; kill it with task_kill or leave it running"
                )


def _format_duration(seconds: float) -> str:
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds / 3600:g}h"
    return f"{seconds:g}s"
