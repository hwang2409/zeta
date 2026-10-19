"""Keep child-owned background work alive until it is safe to finish."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..protocol.types import MessageOrigin, StreamEvent
from .background import finish_gate_message

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop

FinishGateOutcome = Literal[
    "completed", "handed_off", "model_bound_expired", "task_safety_expired"
]


@dataclass(slots=True)
class FinishGateControl:
    max_turns: int
    turns: int = 0
    handoff: tuple[str, str] | None = None

    def turn_ended(self) -> bool:
        self.turns += 1
        return self.handoff is not None or self.turns >= self.max_turns


@dataclass(frozen=True, slots=True)
class FinishGateResult:
    outcome: FinishGateOutcome
    reason: str | None = None
    handoff: tuple[str, str] | None = None


async def run_finish_gate(
    child_loop: AgentLoop,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
    consume_events: Callable[
        [AsyncIterator[StreamEvent], FinishGateControl | None], Awaitable[None]
    ],
    failed: Callable[[], bool],
) -> FinishGateResult:
    """Run the bounded finish gate while giving tasks one hard lifetime."""

    owner = child_loop._background_owner
    loop = asyncio.get_running_loop()
    gate_started = loop.time()
    child_deadline = gate_started + owner.finish_gate_timeout
    task_deadline = gate_started + owner.finish_gate_task_timeout
    control = FinishGateControl(owner.finish_gate_max_turns)
    task_outlived_child_bound = False

    def owned_children() -> tuple[tuple[str, str], ...]:
        instance_id = child_loop.agent_instance_id
        if instance_id is None:
            return ()
        return owner.owned_running(instance_id)

    def owned_tasks() -> tuple[tuple[str, str], ...]:
        return tuple(
            (task.task_id, task.headline)
            for task in child_loop.tool_registry.background_tasks.snapshot()
            if task.running
        )

    async def handoff(arguments: dict[str, Any]) -> str:
        control.handoff = (arguments["reason"], arguments["outputs"])
        return "background work handed off"

    async def run_model_turn(events: AsyncIterator[StreamEvent]) -> FinishGateResult | None:
        deadline = loop.time() + owner.finish_gate_timeout
        if owned_tasks():
            deadline = min(deadline, task_deadline)
        try:
            async with asyncio.timeout_at(deadline):
                await consume_events(events, control)
        except TimeoutError:
            if owned_tasks() and loop.time() >= task_deadline:
                return FinishGateResult(
                    "task_safety_expired",
                    "finish-gate task safety bound expired",
                )
            if not owned_tasks():
                return FinishGateResult(
                    "model_bound_expired", "finish-gate bound expired"
                )
            return None
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()
        if control.handoff is not None:
            return FinishGateResult("handed_off", handoff=control.handoff)
        if control.turns >= owner.finish_gate_max_turns:
            return FinishGateResult(
                "model_bound_expired", "finish-gate turn bound expired"
            )
        return None

    child_loop.tool_registry.register(
        "agent_handoff",
        handoff,
        description=(
            "Hand off running background work to the parent. Sub-agents "
            "continue and report to the parent. Background tasks cannot be "
            "adopted and are killed when this agent completes; provide where "
            "their output can be found."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "minLength": 1},
                "outputs": {"type": "string", "minLength": 1},
            },
            "required": ["reason", "outputs"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
    try:
        result = await run_model_turn(
            child_loop.run_turn(
                "",
                origin=MessageOrigin.HARNESS_NUDGE,
                user_message=finish_gate_message(children, tasks),
            )
        )
        if result is not None:
            return result

        while not failed():
            children = owned_children()
            tasks = owned_tasks()
            if tasks and loop.time() >= child_deadline:
                task_outlived_child_bound = True
            pending_notification = child_loop.notification_system_message() is not None
            if child_loop.store.pending_prompts():
                return FinishGateResult("completed")
            if not children and not tasks and not pending_notification:
                return FinishGateResult("completed")
            if (children or tasks) and not pending_notification:
                deadline = task_deadline if tasks else child_deadline
                if loop.time() >= deadline:
                    if tasks:
                        return FinishGateResult(
                            "task_safety_expired",
                            "finish-gate task safety bound expired",
                        )
                    if task_outlived_child_bound:
                        return FinishGateResult("completed")
                    return FinishGateResult(
                        "model_bound_expired", "finish-gate bound expired"
                    )
                try:
                    async with asyncio.timeout_at(deadline):
                        await owner.conversation_channel.wait(
                            child_loop.agent_instance_id or ""
                        )
                    if tasks and loop.time() >= child_deadline:
                        task_outlived_child_bound = True
                except TimeoutError:
                    if owned_tasks():
                        return FinishGateResult(
                            "task_safety_expired",
                            "finish-gate task safety bound expired",
                        )
                    return FinishGateResult(
                        "model_bound_expired", "finish-gate bound expired"
                    )
            if child_loop.notification_system_message() is None:
                continue
            result = await run_model_turn(child_loop.run_notification_turn())
            if result is not None:
                return result
        return FinishGateResult("completed")
    finally:
        child_loop.tool_registry.unregister("agent_handoff")
