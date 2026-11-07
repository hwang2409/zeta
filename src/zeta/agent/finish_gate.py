"""Keep child-owned background work alive until it is safe to finish."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..protocol.types import (
    Message,
    StreamEvent,
    harness_notice,
)

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop

FinishGateOutcome = Literal["completed", "handed_off", "task_safety_expired"]
POST_CAP_CAUSAL_TURNS = 3
ConsumeEvents = Callable[
    [AsyncIterator[StreamEvent], "FinishGateControl | None"], Awaitable[None]
]


def finish_gate_message(
    children: tuple[tuple[str, str], ...], tasks: tuple[tuple[str, str], ...]
) -> Message:
    running: list[str] = []
    if children:
        child_list = ", ".join(
            f"{handle}: {description}" for handle, description in children
        )
        running.append(
            f"{len(children)} background sub-agents still running: {child_list}"
        )
    if tasks:
        task_list = ", ".join(f"{task_id}: {headline}" for task_id, headline in tasks)
        running.append(f"{len(tasks)} background tasks still running: {task_list}")
    stop_choices = []
    if children:
        stop_choices.append("cancel sub-agents with agent_cancel")
    if tasks:
        stop_choices.append("stop tasks with task_kill")
    text = (
        f"You have {'; '.join(running)}. You cannot finish yet. Choose one: "
        "(a) wait: reply WAIT and end this turn; completions and task exits "
        "will wake you, with no polling; (b) "
        f"{' and '.join(stop_choices)}; "
        "(c) hand off with agent_handoff, providing the reason and where "
        "the outputs will appear. Sub-agents continue under the parent, but "
        "background tasks cannot be adopted and will be killed when this "
        "agent completes; task output remains in this child session's task archive. "
        "A finish-gate safety bound applies if a task never exits."
    )
    return harness_notice("finish gate", text, event="agent_finish_gate")


@dataclass(eq=False, slots=True)
class CausalEpisode:
    remaining: int = POST_CAP_CAUSAL_TURNS
    staged_ids: set[str] = field(default_factory=set)
    member_instance_ids: set[str] = field(default_factory=set)


_CURRENT_CAUSAL_EPISODE: ContextVar[CausalEpisode | None] = ContextVar(
    "zeta_finish_gate_causal_episode", default=None
)


class CausalEpisodeRegistry:
    """Bound provider work created by one post-cap durable delivery chain."""

    def __init__(self) -> None:
        self._by_staged_id: dict[str, CausalEpisode] = {}
        self._by_member: dict[str, CausalEpisode] = {}

    def associate(self, staged_ids: Iterable[str]) -> None:
        episode = _CURRENT_CAUSAL_EPISODE.get()
        if episode is None:
            return
        for staged_id in staged_ids:
            self._by_staged_id[staged_id] = episode
            episode.staged_ids.add(staged_id)

    def claim(
        self, instance_id: str, staged_ids: tuple[str, ...]
    ) -> tuple[CausalEpisode | None, tuple[str, ...]]:
        associated = {self._by_staged_id.get(item_id) for item_id in staged_ids}
        associated.discard(None)
        eligible = {episode for episode in associated if episode.remaining > 0}
        unassociated = tuple(
            item_id for item_id in staged_ids if item_id not in self._by_staged_id
        )
        for item_id in unassociated:
            fresh = CausalEpisode()
            eligible.add(fresh)
            self._by_staged_id[item_id] = fresh
            fresh.staged_ids.add(item_id)
        if not eligible:
            return None, staged_ids
        remaining = min(episode.remaining for episode in eligible) - 1
        episode = CausalEpisode(remaining=remaining)
        episode.member_instance_ids.add(instance_id)
        self._by_member[instance_id] = episode
        for source in eligible:
            source.remaining = remaining
        return episode, ()

    def enter(self, episode: CausalEpisode | None) -> Token[CausalEpisode | None]:
        return _CURRENT_CAUSAL_EPISODE.set(episode)

    @staticmethod
    def leave(token: Token[CausalEpisode | None]) -> None:
        _CURRENT_CAUSAL_EPISODE.reset(token)

    def associate_new_member(
        self, instance_id: str, parent_instance_id: str | None
    ) -> None:
        episode = _CURRENT_CAUSAL_EPISODE.get()
        if episode is None and parent_instance_id is not None:
            episode = self._by_member.get(parent_instance_id)
        if episode is None:
            return
        self._by_member[instance_id] = episode
        episode.member_instance_ids.add(instance_id)

    def associate_member(self, instance_id: str, staged_ids: Iterable[str]) -> None:
        episode = self._by_member.get(instance_id)
        if episode is None:
            return
        for staged_id in staged_ids:
            self._by_staged_id[staged_id] = episode
            episode.staged_ids.add(staged_id)

    def release_staged(self, staged_ids: Iterable[str]) -> None:
        for staged_id in staged_ids:
            episode = self._by_staged_id.pop(staged_id, None)
            if episode is not None:
                episode.staged_ids.discard(staged_id)

    def release_member(self, instance_id: str) -> None:
        episode = self._by_member.pop(instance_id, None)
        if episode is not None:
            episode.member_instance_ids.discard(instance_id)

    @property
    def association_count(self) -> int:
        return len(self._by_staged_id) + len(self._by_member)


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
    automatic_turns_exhausted: bool = False
    terminal_admitted: bool = False

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

    def can_wake_again(self, child_loop: AgentLoop) -> bool:
        """Return whether this idle child still owns a future wake source."""

        instance_id = child_loop.agent_instance_id
        flow = getattr(child_loop, "step_delivery", None)
        owned_children = bool(self.owned_children(child_loop))
        owned_tasks = bool(self.owned_tasks(child_loop))
        pending_question = instance_id is not None and (
            child_loop._background_owner.conversation_channel.has_pending_question(
                instance_id
            )
        )
        pending_delivery = flow is not None and flow.has_model_visible_delivery
        return any((owned_children, owned_tasks, pending_question, pending_delivery))

    async def run_model_turn(
        self,
        child_loop: AgentLoop,
        events: AsyncIterator[StreamEvent],
        consume_events: ConsumeEvents,
        episode: CausalEpisode | None = None,
    ) -> FinishGateResult | None:
        owner = child_loop._background_owner
        loop = asyncio.get_running_loop()
        task_registry = child_loop.tool_registry.background_tasks
        task_running_at_entry = bool(self.owned_tasks(child_loop))
        task_start_generation = task_registry.task_start_generation
        deadline = loop.time() + owner.finish_gate_timeout
        if self.task_outlived_child_bound:
            deadline = self._task_deadline()
        elif task_running_at_entry:
            deadline = min(deadline, self._task_deadline())
        registry = owner.causal_episodes
        token = registry.enter(episode)
        try:
            async with asyncio.timeout_at(deadline):
                try:
                    await consume_events(events, self.control)
                finally:
                    close = getattr(events, "aclose", None)
                    if close is not None:
                        await close()
        except TimeoutError:
            flow = getattr(child_loop, "step_delivery", None)
            if (
                flow is not None
                and (await flow.admit_terminal("finish_gate_timeout")).pending
            ):
                return None
            task_existed = (
                task_running_at_entry
                or task_registry.task_start_generation != task_start_generation
            )
            return self._model_timeout_result(
                task_existed=task_existed,
                task_running=task_registry.running_count > 0,
            )
        finally:
            registry.leave(token)
        task_existed = (
            task_running_at_entry
            or task_registry.task_start_generation != task_start_generation
        )
        return self._model_turn_result(task_existed=task_existed)

    def _model_timeout_result(
        self, *, task_existed: bool, task_running: bool
    ) -> FinishGateResult | None:
        if task_running and asyncio.get_running_loop().time() >= self._task_deadline():
            return _task_safety_expired()
        if task_existed:
            return None
        self.automatic_turns_exhausted = True
        return None

    def _model_turn_result(self, *, task_existed: bool) -> FinishGateResult | None:
        control = self._control()
        if control.handoff is not None:
            return FinishGateResult("handed_off", handoff=control.handoff)
        if control.turns < control.max_turns:
            return None
        if task_existed:
            self.turns_exhausted_with_task = True
        self.automatic_turns_exhausted = True
        return None

    async def wait_transition(
        self, child_loop: AgentLoop, *, tasks_running: bool
    ) -> FinishGateResult | None:
        loop = asyncio.get_running_loop()
        deadline = self._task_deadline() if tasks_running else self._child_deadline()
        if loop.time() >= deadline:
            flow = getattr(child_loop, "step_delivery", None)
            if (
                flow is not None
                and (await flow.admit_terminal("finish_gate_timeout")).pending
            ):
                return None
            return self._wait_timeout_result(child_loop, tasks_running=tasks_running)
        try:
            async with asyncio.timeout_at(deadline):
                await child_loop._background_owner.conversation_channel.wait(
                    child_loop.agent_instance_id or ""
                )
        except TimeoutError:
            flow = getattr(child_loop, "step_delivery", None)
            if (
                flow is not None
                and (await flow.admit_terminal("finish_gate_timeout")).pending
            ):
                return None
            return self._wait_timeout_result(child_loop, tasks_running=tasks_running)
        if tasks_running and loop.time() >= self._child_deadline():
            self.task_outlived_child_bound = True
        return None

    def _wait_timeout_result(
        self, child_loop: AgentLoop, *, tasks_running: bool
    ) -> FinishGateResult:
        if tasks_running and self.owned_tasks(child_loop):
            return _task_safety_expired()
        return FinishGateResult("completed")

    def should_wait_without_model(self, tasks_running: bool) -> bool:
        return tasks_running and self.turns_exhausted_with_task

    def begin_post_task_turns(self, tasks_running: bool) -> None:
        if tasks_running or not self.turns_exhausted_with_task:
            return
        self._control().turns = 0
        self.turns_exhausted_with_task = False

    def note_task_outlived_child_bound(self, tasks_running: bool) -> None:
        if (
            tasks_running
            and asyncio.get_running_loop().time() >= self._child_deadline()
        ):
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


async def await_owned_work(
    child_loop: AgentLoop, state: FinishGateState | None = None
) -> None:
    """Admit one already-decided terminal outcome after owned work drains."""

    gate = state or FinishGateState()
    gate.start(child_loop)
    while True:
        children = gate.owned_children(child_loop)
        tasks = gate.owned_tasks(child_loop)
        if not children and not tasks:
            return
        deadline = gate._task_deadline() if tasks else gate._child_deadline()
        if asyncio.get_running_loop().time() >= deadline:
            return
        try:
            async with asyncio.timeout_at(deadline):
                await child_loop._background_owner.conversation_channel.wait(
                    child_loop.agent_instance_id or ""
                )
        except TimeoutError:
            return


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
    )


async def _run_finish_gate(
    state: FinishGateState,
    child_loop: AgentLoop,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
    consume_events: ConsumeEvents,
    failed: Callable[[], bool],
) -> FinishGateResult:
    async def handoff(arguments: dict[str, Any]) -> str:
        state._control().handoff = (arguments["reason"], arguments["outputs"])
        return "background work handed off"

    _register_handoff(child_loop, handoff)
    try:
        result = await state.run_model_turn(
            child_loop,
            child_loop.run_harness_notice_turn(finish_gate_message(children, tasks)),
            consume_events,
        )
        if result is not None:
            return result
        return await _wait_for_finish(
            state,
            child_loop,
            consume_events,
            failed,
        )
    finally:
        child_loop.tool_registry.unregister("agent_handoff")


async def _wait_for_finish(
    state: FinishGateState,
    child_loop: AgentLoop,
    consume_events: ConsumeEvents,
    failed: Callable[[], bool],
) -> FinishGateResult:
    while not failed():
        children = state.owned_children(child_loop)
        tasks = state.owned_tasks(child_loop)
        state.note_task_outlived_child_bound(bool(tasks))
        state.begin_post_task_turns(bool(tasks))
        pending_notification = child_loop.notification_system_message() is not None
        flow = getattr(child_loop, "step_delivery", None)
        delivery_pending = bool(
            flow is not None and (await flow.admit_terminal("finish_gate_wait")).pending
        )
        terminal = await _terminal_transition(
            child_loop,
            children,
            tasks,
            pending_notification or delivery_pending,
        )
        if terminal is not None:
            return terminal
        if state.should_wait_without_model(bool(tasks)):
            result = await state.wait_transition(child_loop, tasks_running=True)
        elif pending_notification or delivery_pending:
            result = await _handle_pending_notification(
                state,
                child_loop,
                consume_events,
                flow,
                tasks_running=bool(tasks),
            )
        else:
            result = await state.wait_transition(child_loop, tasks_running=bool(tasks))
        if result is not None:
            return result
    return FinishGateResult("completed")


async def _handle_pending_notification(
    state: FinishGateState,
    child_loop: AgentLoop,
    consume_events: ConsumeEvents,
    flow: Any | None,
    *,
    tasks_running: bool,
) -> FinishGateResult | None:
    episode = None
    if state.automatic_turns_exhausted:
        if flow is None:
            return await state.wait_transition(child_loop, tasks_running=tasks_running)
        episode, exhausted_ids = child_loop._background_owner.causal_episodes.claim(
            child_loop.agent_instance_id or "", flow.pending_ids()
        )
        if episode is None:
            flow.suppress_pending(exhausted_ids)
            child_loop._background_owner.conversation_channel.clear_wake(
                child_loop.agent_instance_id or ""
            )
            return await state.wait_transition(child_loop, tasks_running=tasks_running)
    return await state.run_model_turn(
        child_loop,
        child_loop.run_notification_turn(),
        consume_events,
        episode,
    )


async def _terminal_transition(
    child_loop: AgentLoop,
    children: tuple[tuple[str, str], ...],
    tasks: tuple[tuple[str, str], ...],
    pending_notification: bool,
) -> FinishGateResult | None:
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
