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
ConsumeEvents = Callable[
    [AsyncIterator[StreamEvent], "FinishGateControl | None"], Awaitable[None]
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


@dataclass(slots=True)
class FinishGateState:
    """Own finish-gate deadlines and turn control for one complete child run."""

    child_deadline: float | None = None
    task_deadline: float | None = None
    control: FinishGateControl | None = None
    task_outlived_child_bound: bool = False
    turns_exhausted_with_task: bool = False

    def start(self, child_loop: AgentLoop) -> None:
        if self.control is not None:
            return
        owner = child_loop._background_owner
        started = asyncio.get_running_loop().time()
        self.child_deadline = started + owner.finish_gate_timeout
        self.task_deadline = started + owner.finish_gate_task_timeout
        self.control = FinishGateControl(owner.finish_gate_max_turns)

    def owned_children(self, child_loop: AgentLoop) -> tuple[tuple[str, str], ...]:
        instance_id = child_loop.agent_instance_id
        if instance_id is None:
            return ()
        return child_loop._background_owner.owned_running(instance_id)

    @staticmethod
    def owned_tasks(child_loop: AgentLoop) -> tuple[tuple[str, str], ...]:
        return tuple(
            (task.task_id, task.headline)
            for task in child_loop.tool_registry.background_tasks.snapshot()
            if task.running
        )

    async def run_model_turn(
        self,
        child_loop: AgentLoop,
        events: AsyncIterator[StreamEvent],
        consume_events: ConsumeEvents,
    ) -> FinishGateResult | None:
        owner = child_loop._background_owner
        loop = asyncio.get_running_loop()
        tasks_were_running = bool(self.owned_tasks(child_loop))
        deadline = loop.time() + owner.finish_gate_timeout
        if tasks_were_running:
            deadline = min(deadline, self._task_deadline())
        try:
            async with asyncio.timeout_at(deadline):
                await consume_events(events, self.control)
        except TimeoutError:
            return self._model_timeout_result(
                child_loop, tasks_were_running=tasks_were_running
            )
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()
        return self._model_turn_result(child_loop)

    def _model_timeout_result(
        self, child_loop: AgentLoop, *, tasks_were_running: bool
    ) -> FinishGateResult | None:
        tasks = self.owned_tasks(child_loop)
        if tasks and asyncio.get_running_loop().time() >= self._task_deadline():
            return _task_safety_expired()
        if tasks or tasks_were_running:
            return None
        return _model_bound_expired()

    def _model_turn_result(self, child_loop: AgentLoop) -> FinishGateResult | None:
        control = self._control()
        if control.handoff is not None:
            return FinishGateResult("handed_off", handoff=control.handoff)
        if control.turns < control.max_turns:
            return None
        if self.owned_tasks(child_loop):
            self.turns_exhausted_with_task = True
            return None
        return FinishGateResult(
            "model_bound_expired", "finish-gate turn bound expired"
        )

    async def wait_transition(
        self, child_loop: AgentLoop, *, tasks_running: bool
    ) -> FinishGateResult | None:
        loop = asyncio.get_running_loop()
        deadline = self._task_deadline() if tasks_running else self._child_deadline()
        if loop.time() >= deadline:
            return self._wait_timeout_result(child_loop, tasks_running=tasks_running)
        try:
            async with asyncio.timeout_at(deadline):
                await child_loop._background_owner.conversation_channel.wait(
                    child_loop.agent_instance_id or ""
                )
        except TimeoutError:
            return self._wait_timeout_result(child_loop, tasks_running=tasks_running)
        if tasks_running and loop.time() >= self._child_deadline():
            self.task_outlived_child_bound = True
        return None

    def _wait_timeout_result(
        self, child_loop: AgentLoop, *, tasks_running: bool
    ) -> FinishGateResult:
        if tasks_running and self.owned_tasks(child_loop):
            return _task_safety_expired()
        if self.task_outlived_child_bound:
            return FinishGateResult("completed")
        return _model_bound_expired()

    def should_wait_without_model(self, tasks_running: bool) -> bool:
        return tasks_running and self.turns_exhausted_with_task

    def begin_post_task_turns(self, tasks_running: bool) -> None:
        if tasks_running or not self.turns_exhausted_with_task:
            return
        self._control().turns = 0
        self.turns_exhausted_with_task = False

    def note_task_outlived_child_bound(self, tasks_running: bool) -> None:
        if tasks_running and asyncio.get_running_loop().time() >= self._child_deadline():
            self.task_outlived_child_bound = True

    def _control(self) -> FinishGateControl:
        assert self.control is not None
        return self.control

    def _child_deadline(self) -> float:
        assert self.child_deadline is not None
        return self.child_deadline

    def _task_deadline(self) -> float:
        assert self.task_deadline is not None
        return self.task_deadline


async def run_finish_gate(
    child_loop: AgentLoop,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
    consume_events: ConsumeEvents,
    failed: Callable[[], bool],
    state: FinishGateState | None = None,
    delivering_prompt_id: str | None = None,
) -> FinishGateResult:
    """Run one segment of the child run's persistent finish gate."""

    gate = state or FinishGateState()
    gate.start(child_loop)
    return await _run_finish_gate(
        gate,
        child_loop,
        children,
        tasks,
        consume_events,
        failed,
        delivering_prompt_id,
    )


async def _run_finish_gate(
    state: FinishGateState,
    child_loop: AgentLoop,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
    consume_events: ConsumeEvents,
    failed: Callable[[], bool],
    delivering_prompt_id: str | None,
) -> FinishGateResult:
    async def handoff(arguments: dict[str, Any]) -> str:
        state._control().handoff = (arguments["reason"], arguments["outputs"])
        return "background work handed off"

    _register_handoff(child_loop, handoff)
    try:
        result = await state.run_model_turn(
            child_loop,
            child_loop.run_turn(
                "",
                origin=MessageOrigin.HARNESS_NUDGE,
                user_message=finish_gate_message(children, tasks),
            ),
            consume_events,
        )
        if result is not None:
            return result
        return await _wait_for_finish(
            state,
            child_loop,
            consume_events,
            failed,
            delivering_prompt_id,
        )
    finally:
        child_loop.tool_registry.unregister("agent_handoff")


async def _wait_for_finish(
    state: FinishGateState,
    child_loop: AgentLoop,
    consume_events: ConsumeEvents,
    failed: Callable[[], bool],
    delivering_prompt_id: str | None,
) -> FinishGateResult:
    while not failed():
        children = state.owned_children(child_loop)
        tasks = state.owned_tasks(child_loop)
        state.note_task_outlived_child_bound(bool(tasks))
        state.begin_post_task_turns(bool(tasks))
        pending_notification = child_loop.notification_system_message() is not None
        terminal = _terminal_transition(
            child_loop,
            children,
            tasks,
            pending_notification,
            delivering_prompt_id,
        )
        if terminal is not None:
            return terminal
        if state.should_wait_without_model(bool(tasks)):
            result = await state.wait_transition(child_loop, tasks_running=True)
        elif pending_notification:
            result = await state.run_model_turn(
                child_loop, child_loop.run_notification_turn(), consume_events
            )
        else:
            result = await state.wait_transition(
                child_loop, tasks_running=bool(tasks)
            )
        if result is not None:
            return result
    return FinishGateResult("completed")


def _terminal_transition(
    child_loop: AgentLoop,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
    pending_notification: bool,
    delivering_prompt_id: str | None,
) -> FinishGateResult | None:
    pending_prompts = child_loop.store.pending_prompts()
    if any(prompt.id != delivering_prompt_id for prompt in pending_prompts):
        return FinishGateResult("completed")
    if not children and not tasks and not pending_notification:
        return FinishGateResult("completed")
    return None


def _register_handoff(
    child_loop: AgentLoop, handoff: Callable[[dict[str, Any]], Awaitable[str]]
) -> None:
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


def _task_safety_expired() -> FinishGateResult:
    return FinishGateResult(
        "task_safety_expired", "finish-gate task safety bound expired"
    )


def _model_bound_expired() -> FinishGateResult:
    return FinishGateResult("model_bound_expired", "finish-gate bound expired")
