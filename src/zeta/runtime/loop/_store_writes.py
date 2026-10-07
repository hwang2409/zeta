"""Ordered store writes for asynchronous agent turns."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...core.store import ConversationEntry
from ...core.store._approval_display import ApprovalAuditRequest
from ...protocol.types import Message
from ...transcript_search.index import refresh_transcript_index

if TYPE_CHECKING:
    from ...core.session import SessionMetadata
    from .agent import AgentLoop


class StoreWriteMixin:
    """Offload root writes without reordering parallel child startup."""

    def _configure_transcript_index(self: AgentLoop) -> None:
        if (
            self.agent_depth == 0
            and self.root_project_id is not None
            and self.project_registry is not None
        ):
            self.store.enable_persisted_append_tracking()
        else:
            self.store.disable_persisted_append_tracking()

    def _set_runtime_project(self: AgentLoop, metadata: SessionMetadata) -> None:
        """Apply changed session-project metadata to the active runtime."""
        self.session_metadata = metadata
        self.root_project_id = metadata.project_id
        self._configure_transcript_index()
        tool_registry = getattr(self, "tool_registry", None)
        if tool_registry is not None:
            tool_registry.project_id = metadata.project_id
        if metadata.project_id is not None:
            self._schedule_transcript_index()

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
        receipts = self.store.take_persisted_appends()
        self._create_task(
            refresh_transcript_index(
                registry.root,
                self.root_project_id,
                self.store.session_id,
                self.store.session_dir,
                receipts or None,
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
        approval_requests: Sequence[ApprovalAuditRequest],
    ) -> ConversationEntry:
        # Provider completion and parallel tool dispatch observe this append as
        # one atomic transition; yielding to a worker here can reorder child
        # startup and queued TUI submissions around that boundary.
        return self.store.append_message_with_approval_requests(
            message, approval_requests
        )
