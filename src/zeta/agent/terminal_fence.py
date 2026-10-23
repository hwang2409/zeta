"""Serialized terminal transitions for one explicit-finish child."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..core.store import ConversationStore


class TerminalTransitionFence:
    """Make finish, cancellation, recovery, and nudge eligibility agree."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        notification_pending: Callable[[], bool] = lambda: False,
        persisted_notification_ids: Callable[[], frozenset[str]] = frozenset,
        owned_work: Callable[[], tuple[str, ...]] = tuple,
        delivering_prompt_id: Callable[[], str | None] = lambda: None,
    ) -> None:
        self._store = store
        self._notification_pending = notification_pending
        self._persisted_notification_ids = persisted_notification_ids
        self._owned_work = owned_work
        self._delivering_prompt_id = delivering_prompt_id
        self._lock = asyncio.Lock()

    async def accept_finish(self, report: dict[str, object]) -> str | None:
        async with self._lock:
            if work := self._owned_work():
                shown = ", ".join(work[:16])
                if len(work) > 16:
                    shown += f", … (+{len(work) - 16} more)"
                return "wait, cancel, or hand off owned work first: " + shown
            if self._notification_pending():
                return "pending notification must be read before finish"
            reason = await self._store._to_thread_durable(
                self._store._accept_explicit_finish,
                report,
                self._persisted_notification_ids(),
                self._delivering_prompt_id(),
            )
            if reason is None:
                self._store.update_agent_lifecycle(
                    state="finishing", current_step="finishing"
                )
            return reason

    async def cancel(self) -> bool:
        """Atomically let cancellation win only before finish acceptance."""
        async with self._lock:
            return await self._store._to_thread_durable(
                self._store._cancel_explicit_child
            )

    async def recover(self) -> dict[str, object] | None:
        """Read the durable winner used by asynchronous shutdown recovery."""
        async with self._lock:
            return await self._store._to_thread_durable(
                self._store._explicit_finish_report
            )

    def accepted_report_snapshot(self) -> dict[str, object] | None:
        """Read recovery state from a store that its owner just loaded."""
        return self._store.pending_prompt_queue.explicit_finish_report_snapshot()

    async def idle_nudge_eligible(self) -> bool:
        """Return whether the idle clock may run."""
        async with self._lock:
            return await self._idle_nudge_eligible()

    async def reserve_idle_nudge(self) -> bool:
        """Atomically let durable input win before reserving one nudge turn."""
        async with self._lock:
            return await self._idle_nudge_eligible()

    async def _idle_nudge_eligible(self) -> bool:
        if self._notification_pending() or self._owned_work():
            return False
        return await self._store._to_thread_durable(
            self._store._idle_nudge_eligible,
            self._persisted_notification_ids(),
        )
