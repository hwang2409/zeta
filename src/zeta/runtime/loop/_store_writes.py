"""Ordered store writes for asynchronous agent turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ...core.store import ConversationEntry
from ...protocol.types import Message, ToolCall

if TYPE_CHECKING:
    from .agent import AgentLoop


class StoreWriteMixin:
    """Offload root writes without reordering parallel child startup."""

    async def _append_turn_message(
        self: AgentLoop, message: Message
    ) -> ConversationEntry:
        if self.agent_depth > 0:
            # Parallel child loops are created in tool-call order. Their first
            # append must not introduce thread-pool completion order before
            # provider startup, which is observable by dispatch/result order.
            return self.store.append_message(message)
        return await self.store.append_message_async(message)

    async def _append_turn_message_with_approvals(
        self: AgentLoop,
        message: Message,
        approval_requests: Sequence[
            tuple[str, ToolCall] | tuple[str, ToolCall, Mapping[str, object]]
        ],
    ) -> ConversationEntry:
        if self.agent_depth > 0:
            return self.store.append_message_with_approval_requests(
                message, approval_requests
            )
        return await self.store.append_message_with_approval_requests_async(
            message, approval_requests
        )
