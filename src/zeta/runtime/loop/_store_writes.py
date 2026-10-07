"""Ordered store writes for asynchronous agent turns."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ...core.store import ConversationEntry
from ...protocol.types import Message, ToolCall
from ...transcript_search.background import refresh_transcript_index

if TYPE_CHECKING:
    from .agent import AgentLoop


class StoreWriteMixin:
    """Offload root writes without reordering parallel child startup."""

    def _schedule_transcript_index(self: AgentLoop) -> None:
        """Refresh completed turns after their transcript rows are durable."""
        if self.agent_depth > 0 or self.root_project_id is None:
            return
        registry = self.project_registry
        if registry is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._create_task(
            refresh_transcript_index(
                registry.root,
                self.root_project_id,
                self.store.session_id,
                self.store.session_dir,
            )
        )

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
        # Provider completion and parallel tool dispatch observe this append as
        # one atomic transition; yielding to a worker here can reorder child
        # startup and queued TUI submissions around that boundary.
        return self.store.append_message_with_approval_requests(
            message, approval_requests
        )
