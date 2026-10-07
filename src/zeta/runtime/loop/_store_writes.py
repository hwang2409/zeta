"""Ordered store writes for asynchronous agent turns."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ...core.store import ConversationEntry
from ...core.store._approval_display import ApprovalAuditRequest
from ...protocol.types import Message

if TYPE_CHECKING:
    from .agent import AgentLoop


class StoreWriteMixin:
    """Offload root writes without reordering parallel child startup."""

    async def _append_turn_message(
        self: AgentLoop,
        message: Message,
        *,
        on_persisted: Callable[[], None] | None = None,
    ) -> ConversationEntry:
        if self.agent_depth > 0:
            # Parallel child loops are created in tool-call order. Their first
            # append must not introduce thread-pool completion order before
            # provider startup, which is observable by dispatch/result order.
            entry = self.store.append_message(message)
            if on_persisted is not None:
                on_persisted()
            return entry
        return await self.store.append_message_async(
            message, on_persisted=on_persisted
        )

    async def _append_turn_message_with_approvals(
        self: AgentLoop,
        message: Message,
        approval_requests: Sequence[ApprovalAuditRequest],
    ) -> ConversationEntry:
        # Provider completion and parallel tool dispatch observe this append as
        # one atomic transition; yielding to a worker here can reorder child
        # startup and queued TUI submissions around that boundary.
        return self.store.append_message_with_approval_requests(
            message, approval_requests
        )
