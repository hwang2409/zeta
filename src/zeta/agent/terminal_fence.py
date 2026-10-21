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
    ) -> None:
        self._store = store
        self._notification_pending = notification_pending
        self._persisted_notification_ids = persisted_notification_ids
        self._owned_work = owned_work
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
            reason = self._store.pending_prompt_queue.accept_explicit_finish(
                report,
                persisted_notification_ids=self._persisted_notification_ids(),
            )
            if reason is None:
                self._store.update_agent_lifecycle(
                    state="finishing", current_step="finishing"
                )
            return reason

    def request_cancel(self) -> bool:
        """Atomically let cancellation win only before finish acceptance."""
        return self._store.pending_prompt_queue.cancel_explicit_child()

    def accepted_report(self) -> dict[str, object] | None:
        """Read the durable winner used by shutdown and restart recovery."""
        return self._store.pending_prompt_queue.explicit_finish_report()

    async def cancel(self) -> bool:
        async with self._lock:
            return self.request_cancel()

    async def recover(self) -> dict[str, object] | None:
        async with self._lock:
            return self.accepted_report()

    async def idle_nudge_eligible(self) -> bool:
        """Return phase-3 nudge eligibility through the shared fence."""
        async with self._lock:
            return (
                self._store.pending_prompt_queue.terminal_transition() is None
                and not self._store.pending_prompts()
                and not self._notification_pending()
                and not self._owned_work()
            )
