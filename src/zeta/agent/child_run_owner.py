"""Own setup and terminal transitions for one child run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

from ..core.agent_state import LoadedAgentLifecycle, MissingAgentLifecycle
from ..core.child_runs import TERMINAL_AGENT_STATES, read_child_lifecycle
from ..core.store import ConversationStore
from ..core.store._async_writes import (
    _ChildQuestionsCommand,
    _FinishAgentLifecycleCommand,
    _ReserveQuestionSettlementCommand,
)
from ..core.store._child_runs import child_run_segments
from ..protocol.types import StreamEvent, StreamEventType
from ..providers.recovery_probe import recovery_account
from ..providers.recovery_schedule import RecoverySchedule
from ..runtime.offload import await_joined_offload

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop
    from .background_owner import BackgroundAgentOwner, CancellationRequest
    from .turn_limit import ChildTurnControl


# An action that writes state completes its phase change only after the write.
_PhaseEvent = Literal["send", "cancel", "shutdown", "probe_success", "probe_failure", "deadline", "finish"]
_PHASE_ACTIONS: dict[str, dict[_PhaseEvent, str]] = {
    "running": {
        "send": "enqueue", "cancel": "cancel", "shutdown": "preserve", "finish": "finish",
        "probe_success": "ignore", "probe_failure": "ignore", "deadline": "ignore",
    },
    "provider_suspending": {
        "send": "activate_after_write", "cancel": "cancel", "shutdown": "preserve", "finish": "finish",
        "probe_success": "ignore", "probe_failure": "ignore", "deadline": "ignore",
    },
    "provider_suspended": {
        "send": "activate", "cancel": "cancel", "shutdown": "preserve", "finish": "finish",
        "probe_success": "activate", "probe_failure": "reschedule", "deadline": "stop_retrying",
    },
    "activating": {
        "send": "enqueue", "cancel": "cancel", "shutdown": "preserve", "finish": "finish",
        "probe_success": "ignore", "probe_failure": "ignore", "deadline": "ignore",
    },
}


class ChildRunTransitionOwner:
    """Serialize live winners; restart repair enters with an exclusive subtree lease."""

    def __init__(
        self,
        store: ConversationStore,
        tool_call_id: str,
        *,
        explicit_finish: bool = False,
        background: bool = False,
        notification_pending: Callable[[], bool] = lambda: False,
        persisted_notification_ids: Callable[[], frozenset[str]] = frozenset,
        owned_work: Callable[[], tuple[str, ...]] = tuple,
        delivering_prompt_id: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._store = store
        self._tool_call_id = tool_call_id
        self._explicit_finish = explicit_finish
        self._background = background
        self._cancellation: CancellationRequest | None = None
        self._winner: str | None = None
        self.quiescing = False
        self._shutdown_suspended = False
        self._shutdown_settled = False
        self._provider_phase = "running"
        self._provider_activation_requested = False
        self._retry_window: dict[str, str] = {}
        self._retry_schedule: RecoverySchedule | None = None
        self._provider_ready: asyncio.Event | None = None
        self._close_admission: Callable[[], None] | None = None
        self._setup_task: asyncio.Task | None = None
        self._execution: asyncio.Task | None = None
        self._abort_runtime: Callable[[], None] | None = None
        self._cleanup_runtime: Callable[[], Awaitable[None]] | None = None
        self._prepare_cancel: Callable[[], Awaitable[None]] | None = None
        self._tree: BackgroundAgentOwner | None = None
        self._notification_pending = notification_pending
        self._persisted_notification_ids = persisted_notification_ids
        self._owned_work = owned_work
        self._delivering_prompt_id = delivering_prompt_id
        self._lock = asyncio.Lock()
        self.run_id = ""
        self.question_settlement = False
        self.finish_waits_for_answer = False

    def attach_setup(
        self,
        tree: BackgroundAgentOwner,
        task: asyncio.Task,
        prepare_cancel: Callable[[], Awaitable[None]],
    ) -> None:
        self._tree = tree
        self._setup_task = task
        self._prepare_cancel = prepare_cancel

    def attach_runtime(
        self,
        execution: asyncio.Task,
        abort: Callable[[], None],
        cleanup: Callable[[], Awaitable[None]],
        close_admission: Callable[[], None] | None = None,
    ) -> None:
        self._execution = execution
        self._close_admission = close_admission
        if self.quiescing and close_admission is not None:
            close_admission()
        self._abort_runtime = abort
        self._cleanup_runtime = cleanup
        self._setup_task = None
        self._prepare_cancel = None

    def bind_finish_checks(
        self,
        *,
        notification_pending: Callable[[], bool],
        persisted_notification_ids: Callable[[], frozenset[str]],
        owned_work: Callable[[], tuple[str, ...]],
        delivering_prompt_id: Callable[[], str | None],
    ) -> None:
        self._notification_pending = notification_pending
        self._persisted_notification_ids = persisted_notification_ids
        self._owned_work = owned_work
        self._delivering_prompt_id = delivering_prompt_id

    async def start(self, initialize: Callable[[], None]) -> None:
        async with self._lock:
            if self._winner is None:
                await self._store._to_thread_store(initialize)

    @property
    def history_store(self) -> ConversationStore:
        assert self._tree is not None
        return self._tree.notification_store

    @property
    def recoverable(self) -> bool:
        return self._background and self._store.child_recovery_enabled

    @property
    def durable_questions(self) -> bool:
        return self._background and self._store.child_question_hold_enabled

    def questions(self) -> tuple[dict, ...]:
        if not self.durable_questions:
            return ()
        return tuple(entry.data for entry in self._store._delivery_fold.child_questions.values())

    def pending_questions(self) -> tuple[dict, ...]:
        return tuple(q for q in self.questions() if q["state"] == "pending")

    async def write_questions(self, *rows: dict) -> None:
        if rows:
            await self._store._commit_storage_command(_ChildQuestionsCommand(rows=rows))

    async def reconcile_questions(self) -> None:
        if self.durable_questions:
            assert self._tree is not None
            await self._tree.conversation_channel.reconcile_questions(self.run_id, self.questions())

    async def repair_before_replay(self) -> None:
        if self.durable_questions:
            async with self._lock:
                await self.reconcile_questions()

    def _phase_action(self, event: _PhaseEvent) -> str:
        if self._winner is not None or self._shutdown_settled:
            return "ignore"
        if event not in {"cancel", "shutdown", "finish"} and self._provider_stopping():
            return "ignore"
        return _PHASE_ACTIONS[self._provider_phase][event]

    def _provider_stopping(self) -> bool:
        return (
            self.quiescing or self._cancellation is not None or self._winner is not None
            or self._execution is not None and self._execution.done()
        )

    async def send(self, enqueue: Callable[[], Awaitable[None]]) -> None:
        async with self._lock:
            if self._phase_action("send") == "ignore":
                raise LookupError(self.run_id)
            await enqueue()
            await self.reconcile_questions()
            action = self._phase_action("send")
            if action == "activate_after_write":
                self._provider_activation_requested = True
            elif action == "activate":
                await self._activate_provider(self._provider_episode)

    @property
    def provider_recovery_enabled(self) -> bool:
        return self._background and self._store.provider_recovery_enabled

    def bind_provider_recovery(self, parent: AgentLoop, backend, description: str) -> None:
        self._provider_backend = backend
        self._provider_account = recovery_account(backend)
        self._provider_parent = parent
        self._provider_description = description

    @property
    def provider_suspended(self) -> bool:
        return self._provider_phase in {"provider_suspending", "provider_suspended", "activating"}

    @property
    def _recovery(self):
        assert self._tree is not None
        return self._tree.provider_recovery

    def _provider_unsubscribe(self) -> None:
        if hasattr(self, "_provider_account"):
            self._recovery.unsubscribe(self._provider_account, self.run_id)

    async def publish_suspension_notice(self, parent: AgentLoop, description: str, episode: int, suspension=None) -> None:
        from ..core.store._async_writes import _AppendChildRecoveredCommand

        notice, created = await parent.store._commit_storage_command(_AppendChildRecoveredCommand(
            run_id=self.run_id, episode=episode, path=str(self._store.session_dir),
            description=description, suspension=suspension,
        ))
        if created:
            parent._publish_background_event(StreamEvent(
                StreamEventType.AGENT_NOTIFICATION, data={"notification_id": notice.id, **notice.data}, row_seq=notice.seq,
            ))

    def restore_provider_window(self, state: dict) -> None:
        self._retry_window = {key: state[key] for key in ("retry_started_at", "retry_deadline_at") if key in state}
        if "retry_probe_at" in state:
            self._retry_schedule = RecoverySchedule(state["retry_probe_at"], state["retry_step"])

    async def _persist_provider_wait(self, schedule: RecoverySchedule, *, stopped: bool) -> dict:
        from ..core.store._async_writes import _ChildSuspensionCommand

        self._retry_schedule = schedule
        next_at = min(datetime.fromisoformat(schedule.probe_at), datetime.fromisoformat(self._retry_window["retry_deadline_at"]))
        return await self._store._commit_storage_command(_ChildSuspensionCommand(provider_retry={
            **self._retry_window, "retry_state": "stopped" if stopped else "retrying",
            "retry_probe_at": schedule.probe_at, "retry_step": schedule.step,
            **({} if stopped else {"next_retry_at": next_at.isoformat()}),
        }))

    async def provider_suspend(self, code: str, elapsed: float) -> None:
        self._provider_phase = "provider_suspending"
        async with self._lock:
            if self._provider_stopping():
                raise asyncio.CancelledError
            if not self._retry_window:
                started = self._recovery.wall() - timedelta(seconds=elapsed)
                self._retry_window = {
                    "retry_started_at": started.isoformat(),
                    "retry_deadline_at": (started + timedelta(hours=1)).isoformat(),
                }
            deadline = self._retry_window["retry_deadline_at"]
            schedule = self._recovery.schedule(self._provider_account, self._retry_schedule)
            stopped = self._recovery.wall() >= datetime.fromisoformat(deadline)
            state = await self._persist_provider_wait(schedule, stopped=stopped)
            next_at = state.get("next_retry_at")
            self._provider_episode = episode = state["suspension_episode"]
            await self.publish_suspension_notice(self._provider_parent, self._provider_description, episode, {
                "text": state["current_step"], "code": code,
                **({"next_retry_at": next_at} if next_at is not None else {}),
            })
        # Let sends queued behind the durable write register intent before arming the timer.
        async with self._lock:
            if self._provider_stopping():
                raise asyncio.CancelledError
            self._provider_phase = "provider_suspended"
            ready = self._provider_ready = asyncio.Event()
            activate_now = self._provider_activation_requested
            self._provider_activation_requested = False
            if activate_now:
                await self._activate_provider(episode)
            elif not stopped:
                self._subscribe_provider(episode, deadline, schedule)
        await ready.wait()
        if self._provider_stopping():
            raise asyncio.CancelledError

    def _subscribe_provider(self, episode: int, deadline: str, schedule: RecoverySchedule) -> None:
        async def activate(next_schedule):
            await self.provider_probe_result(episode, "probe_success", next_schedule)

        async def update(next_schedule, stop):
            await self.provider_probe_result(episode, "deadline" if stop else "probe_failure", next_schedule)

        self._recovery.subscribe(
            self._provider_account, self.run_id, self._provider_backend, deadline=deadline,
            schedule=schedule, activate=activate, update=update,
        )

    async def provider_probe_result(
        self, episode: int, event: Literal["probe_success", "probe_failure", "deadline"], schedule: RecoverySchedule,
    ) -> None:
        if self._phase_action(event) == "ignore":
            return
        async with self._lock:
            if episode != self._provider_episode:
                return
            action = self._phase_action(event)
            if action == "activate":
                self._retry_schedule = schedule
                await self._activate_provider(episode)
            elif action in {"reschedule", "stop_retrying"}:
                await self._persist_provider_wait(schedule, stopped=action == "stop_retrying")

    async def _activate_provider(self, episode: int) -> None:
        self._provider_phase = "activating"
        try:
            schedule = self._retry_schedule
            assert schedule is not None
            committed = await self._commit_activation(self._provider_parent, self._provider_description, episode, schedule)
            if not committed or self._provider_stopping():
                return
            self._recovery.replaying(self._provider_account, self.run_id)
            self._provider_phase = "running"
            assert self._provider_ready is not None
            self._provider_ready.set()
        finally:
            if self._provider_phase == "activating":
                self._provider_phase = "provider_suspended"

    async def provider_replay_progress(self) -> None:
        from ..core.store._async_writes import _ChildSuspensionCommand

        if self._retry_window:
            await self._store._commit_storage_command(_ChildSuspensionCommand(retry_progress=True))
            self._retry_window = {}
            self._retry_schedule = None
            self._recovery.replay_progress(self._provider_account, self.run_id)

    async def hold_at_limit(self) -> bool:
        async with self._lock:
            pending = self.pending_questions()
            await self.write_questions(*(
                {**q, "held_at_turn_limit": True} for q in pending if not q["held_at_turn_limit"]
            ))
            return bool(pending)

    def settlement_due(self) -> bool:
        return any(q["held_at_turn_limit"] and "settlement_turn" not in q for q in self.questions() if q["state"] != "withdrawn")

    async def reserve_question_settlement(self, turn: int) -> bool:
        async with self._lock:
            if self.quiescing or self._winner is not None or self._cancellation is not None or self.pending_questions():
                return False
            ready = frozenset(
                q["question_id"] for q in self.questions()
                if q["held_at_turn_limit"] and q["state"] == "answered" and "settlement_turn" not in q
            )
            if not ready:
                return False
            await self.reconcile_questions()
            await self._store._commit_storage_command(_ReserveQuestionSettlementCommand(question_ids=ready, turn=turn))
            if self.quiescing or self._cancellation is not None:
                return False
            self.question_settlement = True
            return True

    async def withdraw_questions(self) -> None:
        await self.write_questions(*({**q, "state": "withdrawn"} for q in self.pending_questions()))
        await self.reconcile_questions()

    @property
    def suspending(self) -> bool:
        return self._shutdown_suspended and self._winner is None

    def quiesce(self) -> None:
        self.quiescing = True
        # Admission closes before the tree drains its coordinator. Claim preservation
        # now, before a blocked child can observe quiescing and settle its execution.
        if (
            self._phase_action("shutdown") == "preserve" and self.recoverable
            and (self._execution is None or not self._execution.done())
        ):
            self._shutdown_suspended = True
        if self._close_admission is not None:
            self._close_admission()

    async def suspend(self) -> None:
        async with self._lock:
            if self._phase_action("shutdown") != "preserve" or self._cancellation is not None:
                return
            if self._execution is not None and self._execution.done():
                return
            if self._explicit_finish and await self._store.explicit_finish_report() is not None:
                self._winner = "finish"
                return
            self._shutdown_suspended = True
            self._provider_unsubscribe()
            if self._abort_runtime is not None:
                self._abort_runtime()
            task = self._execution if self._execution is not None else self._setup_task
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()

    async def _commit_activation(
        self, parent: AgentLoop, description: str, episode: int, schedule: RecoverySchedule | None = None,
    ) -> bool:
        from ..core.store._async_writes import _ChildSuspensionCommand
        from .background_lifecycle import _complete_despite_cancellation

        async def commit_and_notify() -> bool:
            state = await self._store._commit_storage_command(_ChildSuspensionCommand(
                activate=episode, retry_schedule=(schedule.probe_at, schedule.step) if schedule is not None else None,
                activation_stopped=self._provider_stopping,
            ))
            if state is None or state["state"] not in {"running", "idle"}:
                return False
            await self.publish_suspension_notice(parent, description, episode)
            return True

        task = asyncio.create_task(commit_and_notify())
        if await _complete_despite_cancellation(task):
            raise asyncio.CancelledError
        return task.result()

    async def activate_recovered(self, parent: AgentLoop, description: str, episode: int) -> None:
        async with self._lock:
            if not await self._commit_activation(parent, description, episode) or self._provider_stopping():
                raise asyncio.CancelledError
            await self._store._to_thread_store(self._store.activate_runtime_lease)

    async def repair_activation_notice(self, parent: AgentLoop, description: str, state: dict) -> None:
        if (episode := state.get("last_activated_episode")) is not None:
            await self.publish_suspension_notice(parent, description, episode)

    async def block_recovery(self, reason: str) -> None:
        from ..core.store._async_writes import _ChildSuspensionCommand

        async with self._lock:
            await self._store._commit_storage_command(_ChildSuspensionCommand(blocked=reason))

    async def persist_suspension(self, reason: str) -> dict | None:
        from ..core.store._async_writes import _ChildSuspensionCommand

        async with self._lock:
            if self._prepare_cancel is not None:
                await self._prepare_cancel()
            return await self._store._commit_storage_command(_ChildSuspensionCommand(reason=reason))

    def request_cancel(self) -> CancellationRequest:
        from .background_owner import CancellationRequest

        assert self._tree is not None
        setup = self._setup_task
        if setup is not None and not setup.done() and not setup.cancelling():
            # Stop the setup before it can attach execution. Its cancellation
            # handler enters cancel(), including the lifecycle bootstrap.
            setup.cancel()

        duplicate = self._cancellation is not None and (
            not self._cancellation.confirmation.done()
            or self._winner == "cancel" and not self._cancellation.cleanup.done()
        )

        async def confirm() -> bool:
            if duplicate:
                return False
            if setup is not None:
                try:
                    await asyncio.shield(setup)
                except asyncio.CancelledError:
                    if not setup.cancelled():
                        raise
                return self._winner == "cancel"
            return await self.cancel()

        async def cleanup(confirmation: asyncio.Task[bool]) -> None:
            try:
                confirmed = await confirmation
            except Exception:  # noqa: BLE001 - requester reports write failures
                return
            if confirmed and self._cleanup_runtime is not None:
                await self._cleanup_runtime()

        confirmation = self._tree.create_cancellation_task(confirm())
        request = CancellationRequest(
            confirmation, self._tree.create_cancellation_task(cleanup(confirmation))
        )
        if not duplicate:
            self._cancellation = request
            self._provider_unsubscribe()
        return request

    async def cancel_and_join(self) -> None:
        request = self._cancellation
        if request is None or request.cleanup.done():
            request = self.request_cancel()
        await asyncio.shield(request.cleanup)
        # A foreground cancellation must not return a receipt on write failure.
        await asyncio.shield(request.confirmation)

    def owned_work_refusal(self) -> str | None:
        if not (work := self._owned_work()):
            return None
        shown = ", ".join(work[:16])
        if len(work) > 16:
            shown += f", … (+{len(work) - 16} more)"
        return "wait, cancel, or hand off owned work first: " + shown

    async def accept_finish(self, report: dict[str, object]) -> str | None:
        async with self._lock:
            if self._winner == "cancel":
                return "child is canceled"
            if self._winner is not None or self.quiescing:
                return "child is finishing"
            await self.reconcile_questions()
            if self.pending_questions():
                self.finish_waits_for_answer = True
                return "wait for your parent to answer the pending question before finish"
            if refusal := self.owned_work_refusal():
                return refusal
            if self._notification_pending():
                return "pending notification must be read before finish"
            reason = await self._store.accept_explicit_finish(
                report,
                self._persisted_notification_ids(),
                self._delivering_prompt_id(),
                question_settlement=self.question_settlement,
            )
            if reason is None:
                self._winner = "finish"
                await self._store._to_thread_store(
                    lambda: self._store.update_agent_lifecycle(
                        state="finishing", current_step="finishing"
                    )
                )
            return reason

    async def cancel(self) -> bool:
        """Confirm cancellation before interrupting runtime execution."""
        async with self._lock:
            if self._phase_action("cancel") != "cancel" or (
                self._background and self._execution is not None and self._execution.done() and not (self.provider_suspended or self.suspending)
            ):
                return False
            if (
                self._explicit_finish
                and self._prepare_cancel is None
                and not await self._store.cancel_explicit_child()
                and await self._store.terminal_transition() != "cancel"
            ):
                return False
            if self._prepare_cancel is not None:
                await self._prepare_cancel()
            await self._store.mark_agent_canceled_async(self._tool_call_id, current_step="canceled")
            self._winner = "cancel"
            if self._abort_runtime is not None:
                self._abort_runtime()
            return True

    async def complete_foreground(
        self, state: str, text: str, turns_used: int, *, canonical: bool = False,
    ) -> None:
        async with self._lock:
            if self._winner == "cancel" or self.suspending:
                raise asyncio.CancelledError
            if self._winner is None:
                await self._store._commit_storage_command(_FinishAgentLifecycleCommand(
                    state=state, final_result=text, turns_used=turns_used,
                    killed_task_ids=None, killed_task_count=None,
                    killed_task_ids_truncated=None, canonical_receipt=canonical, current_step=state,
                ))
                self._winner = "complete"

    def exhausted_turn_limit(self, control: ChildTurnControl | None) -> ChildTurnControl | None:
        """Return the exhausted budget only if no durable finish won."""
        if control is None or not control.exhausted or self.recovered_report() is not None:
            return None
        return control

    async def background_outcome(self) -> tuple[bool, dict[str, object] | None]:
        """Read the winning outcome after execution settles, before publication."""
        request = self._cancellation
        if (self.provider_suspended or self.suspending) and request is not None and not request.confirmation.done():
            await asyncio.gather(asyncio.shield(request.confirmation), return_exceptions=True)
        async with self._lock:
            report = None
            if self._explicit_finish:
                report = await self._store.explicit_finish_report()
                if (
                    report is None and self._winner is None
                    and await self._store.terminal_transition() == "cancel"
                ):
                    self._winner = "cancel"
            self._provider_unsubscribe()
            if self._phase_action("finish") == "finish":
                if report is not None:
                    self._winner = "finish"
                elif self.suspending:
                    self._shutdown_settled = True
                else:
                    self._winner = "complete"
            canceled = self._winner == "cancel"
            request = self._cancellation
        # Only the winning cancel request can have runtime cleanup to join.
        if canceled and request is not None:
            await asyncio.shield(request.cleanup)
        return canceled, report

    async def publish_background[Result](self, state: str, publish: Callable[[str], Awaitable[Result]]) -> Result:
        async with self._lock:
            await self.withdraw_questions()
            return await publish(state)

    def recovered_report(self) -> dict[str, object] | None:
        return self._store.pending_prompt_queue.explicit_finish_report_snapshot()

    def repair_terminal(self, status: str, text: str) -> None:
        """Settle a recovered run while the caller holds its subtree lease off-loop."""
        lifecycle = self._store.agent_lifecycle()
        running = lifecycle is not None and lifecycle.get("finished_at") is None
        if status == "canceled" and (lifecycle is None or running):
            self._store.mark_agent_canceled(self._tool_call_id, current_step="canceled")
            return
        if running:
            state = "failed" if status == "error" else "completed"
            self._store.finish_agent_lifecycle(
                state, final_result=text or "child completed", current_step=state,
            )
        self._store.finish_agent_parent()

    async def idle_nudge_eligible(self) -> bool:
        """Return whether the idle clock may run."""
        async with self._lock:
            return await self._idle_nudge_eligible()

    async def reserve_idle_nudge(self) -> bool:
        """Atomically let durable input win before reserving one nudge turn."""
        async with self._lock:
            return await self._idle_nudge_eligible()

    async def _idle_nudge_eligible(self) -> bool:
        if self.quiescing or self._winner is not None or self._notification_pending() or self._owned_work():
            return False
        return await self._store.idle_nudge_eligible(
            self._persisted_notification_ids()
        )


class ChildControlError(Exception):
    def __init__(self, code: str, *, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


async def cancel_child_run(loop: AgentLoop, run_id: str) -> str:
    store = loop.store
    try:
        segments = child_run_segments(store.session_id, run_id)
    except ValueError:
        raise ChildControlError("child_run_not_found") from None
    path = store.session_dir.absolute()
    for segment in segments:
        path = path / "agents" / str(segment)

    def state() -> str:
        try:
            lifecycle, _ = read_child_lifecycle(store, path)
        except FileNotFoundError:
            raise ChildControlError("child_run_not_found") from None
        if isinstance(lifecycle, MissingAgentLifecycle):
            raise ChildControlError("child_run_not_found")
        if not isinstance(lifecycle, LoadedAgentLifecycle):
            raise ChildControlError("state_unavailable")
        if lifecycle.value.get("handle") != run_id:
            raise ChildControlError("child_run_not_found")
        return lifecycle.value["state"]

    try:
        current = await await_joined_offload(state)
    except asyncio.CancelledError:
        # An interrupted reply must not withdraw cancellation of a live child.
        loop._background_owner.cancel(run_id)
        raise
    if current in TERMINAL_AGENT_STATES:
        return current
    outcome = await loop._background_owner.cancel_and_wait(run_id)
    if outcome.status == "confirmed":
        return "cancelling"
    current = await await_joined_offload(state)
    if current in TERMINAL_AGENT_STATES:
        return current
    if outcome.status == "failed" or not loop._background_owner.owns_running(run_id):
        raise ChildControlError("state_unavailable")
    raise ChildControlError("operation_busy")
