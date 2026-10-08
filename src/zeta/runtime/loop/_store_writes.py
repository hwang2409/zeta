"""Ordered store writes for asynchronous agent turns."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ...core.store import ConversationEntry
from ...core.store._approval_display import ApprovalAuditRequest
from ...protocol.types import Message
from ...transcript_search.index import (
    is_indexable_top_level_project_session,
    refresh_transcript_index,
)

if TYPE_CHECKING:
    from ...core.session import SessionMetadata
    from .agent import AgentLoop


class StoreWriteMixin:
    """Offload root writes without reordering parallel child startup."""

    def _indexable_project_session(self: AgentLoop) -> bool:
        return self.project_registry is not None and is_indexable_top_level_project_session(
            agent_depth=self.agent_depth,
            project_id=self.root_project_id,
            parent_session_id=self.parent_session_id,
            session_dir=self.store.session_dir,
        )

    def _configure_transcript_index(self: AgentLoop) -> None:
        if self._indexable_project_session():
            self.store.enable_persisted_append_tracking()
        else:
            self.store.disable_persisted_append_tracking()

    def _set_runtime_project(self: AgentLoop, metadata: SessionMetadata) -> None:
        """Apply changed session-project metadata to the active runtime."""
        self.session_metadata = metadata
        self.root_project_id = metadata.project_id
        self.parent_session_id = metadata.parent_session_id
        self._configure_transcript_index()
        tool_registry = getattr(self, "tool_registry", None)
        if tool_registry is not None:
            tool_registry.project_id = metadata.project_id
        if metadata.project_id is not None:
            self._schedule_transcript_index()

    def _schedule_transcript_index(self: AgentLoop) -> None:
        """Refresh completed turns after their transcript rows are durable."""
        if not self._indexable_project_session():
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
