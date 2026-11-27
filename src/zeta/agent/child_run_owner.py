"""Own setup and terminal transitions for one child run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ..core.store import ConversationStore
from ..core.store._async_writes import _FinishAgentLifecycleCommand

if TYPE_CHECKING:
    from .background_owner import BackgroundAgentOwner, CancellationRequest


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
    def suspending(self) -> bool:
        return self._winner == "suspend"

    def quiesce(self) -> None:
        self.quiescing = True
        if self._close_admission is not None:
            self._close_admission()

    async def suspend(self) -> None:
        async with self._lock:
            if self._winner is not None or self._cancellation is not None:
                return
            if self._execution is not None and self._execution.done():
                return
            if self._explicit_finish and await self._store.explicit_finish_report() is not None:
                self._winner = "finish"
                return
            self._winner = "suspend"
            if self._abort_runtime is not None:
                self._abort_runtime()
            task = self._execution if self._execution is not None else self._setup_task
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()

    async def activate_recovered(self, episode: int) -> None:
        from ..core.store._async_writes import _ChildSuspensionCommand

        async with self._lock:
            if self.quiescing or self._winner is not None:
                raise asyncio.CancelledError
            await self._store._commit_storage_command(_ChildSuspensionCommand(activate=episode))
            await self._store._to_thread_store(self._store.activate_runtime_lease)

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
            if refusal := self.owned_work_refusal():
                return refusal
            if self._notification_pending():
                return "pending notification must be read before finish"
            reason = await self._store.accept_explicit_finish(
                report,
                self._persisted_notification_ids(),
                self._delivering_prompt_id(),
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
            if self._winner is not None or (
                self._background and self._execution is not None and self._execution.done()
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
            await self._store.mark_agent_canceled_async(self._tool_call_id)
            self._winner = "cancel"
            if self._abort_runtime is not None:
                self._abort_runtime()
            return True

    async def complete_foreground(
        self, state: str, text: str, turns_used: int, *, canonical: bool = False,
    ) -> None:
        async with self._lock:
            if self._winner in {"cancel", "suspend"}:
                raise asyncio.CancelledError
            if self._winner is None:
                await self._store._commit_storage_command(_FinishAgentLifecycleCommand(
                    state=state, final_result=text, turns_used=turns_used,
                    killed_task_ids=None, killed_task_count=None,
                    killed_task_ids_truncated=None, canonical_receipt=canonical,
                ))
                self._winner = "complete"

    async def background_outcome(self) -> tuple[bool, dict[str, object] | None]:
        """Read the winning outcome after execution settles, before publication."""
        async with self._lock:
            report = None
            if self._explicit_finish:
                report = await self._store.explicit_finish_report()
                if (
                    report is None and self._winner is None
                    and await self._store.terminal_transition() == "cancel"
                ):
                    self._winner = "cancel"
            if self._winner is None:
                self._winner = "finish" if report is not None else "complete"
            canceled = self._winner == "cancel"
            request = self._cancellation
        # Only the winning cancel request can have runtime cleanup to join.
        if canceled and request is not None:
            await asyncio.shield(request.cleanup)
        return canceled, report

    async def publish_background[Result](self, publish: Callable[[], Awaitable[Result]]) -> Result:
        async with self._lock:
            return await publish()

    def recovered_report(self) -> dict[str, object] | None:
        return self._store.pending_prompt_queue.explicit_finish_report_snapshot()

    def repair_terminal(self, status: str, text: str) -> None:
        """Settle a recovered run while the caller holds its subtree lease off-loop."""
        lifecycle = self._store.agent_lifecycle()
        running = lifecycle is not None and lifecycle.get("finished_at") is None
        if status == "canceled" and (lifecycle is None or running):
            self._store.mark_agent_canceled(self._tool_call_id)
            return
        if running:
            self._store.finish_agent_lifecycle(
                "failed" if status == "error" else "completed",
                final_result=text or "child completed",
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
