"""Shared rules for explicit completion of background children."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from .presets import AgentPreset

if TYPE_CHECKING:
    from ..tools._shared.process import BackgroundTaskRegistry


IDLE_CHILD_FD_FLOOR = 3
IDLE_CHILD_WARNING_THRESHOLD = 16


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

    from .background import adopt_agent_children
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


async def kill_tasks_after_deadline(
    registry: BackgroundTaskRegistry, timeout: float
) -> tuple[str, ...]:
    """Kill tasks still running when the single idle safety deadline expires."""

    await asyncio.sleep(timeout)
    task_ids = tuple(task.task_id for task in registry.snapshot() if task.running)
    for task_id in task_ids:
        await registry.kill(task_id)
    return task_ids
