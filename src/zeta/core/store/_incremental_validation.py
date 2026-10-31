"""Incremental integrity indexes for conversation log tails."""

from __future__ import annotations

import copy
import time

from ...protocol.types import Message, ToolResult, ToolUseContent
from ..checkpoints import ConversationEntry, ConversationIntegrityError
from ._validation import (
    AGENT_COMPLETION_NOTIFICATION_KIND,
    TASK_EXITED_NOTIFICATION_KIND,
)


class IncrementalValidationMixin:
    """Validate linear appended entries without rescanning the full log."""

    def _reset_incremental_validation_state(self) -> None:
        self._entry_ids: set[str] = set()
        self._active_approval_requests: dict[str, ConversationEntry] = {}
        self._active_approval_resolutions: dict[str, str] = {}
        self._active_tool_results: dict[str, ToolResult] = {}
        self._active_agent_call_ids: set[str] = set()
        self._active_agent_receipts: list[tuple[str, str]] = []
        self._notification_ids: set[str] = set()
        self._active_notifications: dict[str, ConversationEntry] = {}
        self._active_completion_notifications: dict[str, ConversationEntry] = {}
        self._active_notification_presentations: set[str] = set()
        self._pending_prompt_ids: set[str] = set()
        self._active_pending_prompt_entries: dict[str, ConversationEntry] = {}
        self._active_pending_prompt_messages: dict[str, Message] = {}
        self._active_notification_delivery_message: Message | None = None
        self._active_last_message: Message | None = None

    def _rebuild_incremental_validation_state(self) -> None:
        """Build O(1) integrity indexes after a complete validation pass."""
        self._reset_incremental_validation_state()
        by_id: dict[str, ConversationEntry] = {}
        for index, entry in enumerate(self._entries):
            if index and index % 64 == 0:
                time.sleep(0.0001)
            self._entry_ids.add(entry.id)
            by_id[entry.id] = entry
        branch: list[ConversationEntry] = []
        current = self._entries[-1] if self._entries else None
        while current is not None:
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        for index, entry in enumerate(reversed(branch)):
            if index and index % 64 == 0:
                time.sleep(0.0001)
            self._record_active_entry(entry)

    def _accept_incremental_entries(
        self, entries: list[ConversationEntry]
    ) -> bool:
        """Validate a tail against staged state, then publish it as one update."""
        staged = copy.copy(self)
        staged._entries = list(self._entries)
        staged._entry_ids = set(self._entry_ids)
        staged._active_approval_requests = dict(self._active_approval_requests)
        staged._active_approval_resolutions = dict(self._active_approval_resolutions)
        staged._active_tool_results = dict(self._active_tool_results)
        staged._active_agent_call_ids = set(self._active_agent_call_ids)
        staged._active_agent_receipts = list(self._active_agent_receipts)
        staged._notification_ids = set(self._notification_ids)
        staged._active_notifications = dict(self._active_notifications)
        staged._active_completion_notifications = dict(
            self._active_completion_notifications
        )
        staged._active_notification_presentations = set(
            self._active_notification_presentations
        )
        staged._pending_prompt_ids = set(self._pending_prompt_ids)
        staged._active_pending_prompt_entries = dict(
            self._active_pending_prompt_entries
        )
        staged._active_pending_prompt_messages = dict(
            self._active_pending_prompt_messages
        )
        staged._active_notification_delivery_message = (
            self._active_notification_delivery_message
        )
        staged._active_last_message = self._active_last_message
        staged._task_notification_ids = set(self._task_notification_ids)
        for index, entry in enumerate(entries):
            # Large off-thread refreshes must release the GIL often enough for
            # the owner event loop to remain responsive.
            if index and index % 64 == 0:
                time.sleep(0.001)
            if not staged._accept_incremental_entry(entry):
                return False

        self._before_incremental_state_install()
        self._entry_ids = staged._entry_ids
        self._active_approval_requests = staged._active_approval_requests
        self._active_approval_resolutions = staged._active_approval_resolutions
        self._active_tool_results = staged._active_tool_results
        self._active_agent_call_ids = staged._active_agent_call_ids
        self._active_agent_receipts = staged._active_agent_receipts
        self._notification_ids = staged._notification_ids
        self._active_notifications = staged._active_notifications
        self._active_completion_notifications = (
            staged._active_completion_notifications
        )
        self._active_notification_presentations = (
            staged._active_notification_presentations
        )
        self._pending_prompt_ids = staged._pending_prompt_ids
        self._active_pending_prompt_entries = staged._active_pending_prompt_entries
        self._active_pending_prompt_messages = staged._active_pending_prompt_messages
        self._active_notification_delivery_message = (
            staged._active_notification_delivery_message
        )
        self._active_last_message = staged._active_last_message
        self._task_notification_ids = staged._task_notification_ids
        # Publish entries last so readers never see rows without matching indexes.
        self._entries = staged._entries
        self._record_client_delivery_entries(entries)
        return True

    def _before_incremental_state_install(self) -> None:
        """Test hook immediately before a staged tail becomes visible."""

    def _accept_incremental_entry(self, entry: ConversationEntry) -> bool:
        """Validate and install one linear tail entry, or request a full reload."""
        expected_seq = self._entries[-1].seq + 1 if self._entries else 1
        if entry.seq != expected_seq:
            raise ConversationIntegrityError(
                f"non-monotonic conversation sequence at {entry.id}"
            )
        if entry.id in self._entry_ids:
            raise ConversationIntegrityError(f"duplicate conversation id: {entry.id}")
        if self._entries and entry.parent_id is None:
            raise ConversationIntegrityError(
                f"conversation entry {entry.id} is an orphaned root"
            )
        if entry.parent_id is not None and entry.parent_id not in self._entry_ids:
            raise ConversationIntegrityError(
                f"missing prior parent {entry.parent_id} for {entry.id}"
            )
        # A non-tip parent changes the active branch. It is uncommon and its
        # integrity state is deliberately rebuilt by the complete validator.
        if self._entries and entry.parent_id != self._entries[-1].id:
            return False
        self._validate_entry_payload(entry)
        if entry.type == "fork":
            self._validate_fork_entry(entry)
        self._record_active_entry(entry)
        self._entries.append(entry)
        self._entry_ids.add(entry.id)
        if entry.type == "notification" and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND:
            task_id = entry.data.get("task_id")
            if type(task_id) is str and task_id:
                self._task_notification_ids.add(task_id)
        return True

    def _record_agent_receipt(self, result: ToolResult) -> None:
        structured = result.structured_content
        if structured is None:
            return
        handle = structured.get("child_instance_id")
        child_path = structured.get("child_session_path")
        if type(handle) is str and handle and type(child_path) is str and child_path:
            self._active_agent_receipts.append((handle, child_path))

    def _record_message_entry(self, entry: ConversationEntry) -> None:
        message = Message.from_dict(entry.data["message"])
        self._active_last_message = message
        prompt_id = message.metadata.get("zeta.pending_prompt_id")
        if type(prompt_id) is str:
            self._active_pending_prompt_messages[prompt_id] = message
        identity = message.metadata.get("zeta.step_delivery")
        if isinstance(identity, dict) and identity.get("notification_ids"):
            self._active_notification_delivery_message = message
        tool_result = message.tool_result
        if tool_result is not None:
            self._active_tool_results[tool_result.tool_call_id] = tool_result
        for block in message.content:
            if (
                isinstance(block, ToolUseContent)
                and block.tool_call.name.casefold() == "agent"
            ):
                self._active_agent_call_ids.add(block.tool_call.id)
                prior_result = self._active_tool_results.get(block.tool_call.id)
                if prior_result is not None:
                    self._record_agent_receipt(prior_result)
        if (
            tool_result is not None
            and tool_result.tool_call_id in self._active_agent_call_ids
        ):
            self._record_agent_receipt(tool_result)
        for request in entry.data.get("approval_requests", []):
            request_id = request.get("request_id")
            if type(request_id) is str and request_id:
                if request_id in self._active_approval_requests:
                    raise ConversationIntegrityError(
                        f"duplicate approval request: {request_id}"
                    )
                self._active_approval_requests[request_id] = entry

    def _record_notification_entry(self, entry: ConversationEntry) -> None:
        handle = entry.data.get("child_instance_id")
        child_path = entry.data.get("child_session_path")
        if type(handle) is str and handle and type(child_path) is str and child_path:
            self._active_agent_receipts.append((handle, child_path))
        self._notification_ids.add(entry.id)
        self._active_notifications[entry.id] = entry
        if (
            entry.data.get("kind", AGENT_COMPLETION_NOTIFICATION_KIND)
            == AGENT_COMPLETION_NOTIFICATION_KIND
            and type(child_id := entry.data.get("child_instance_id")) is str
        ):
            self._active_completion_notifications.setdefault(child_id, entry)

    def _record_pending_prompt_entry(self, entry: ConversationEntry) -> None:
        self._pending_prompt_ids.add(entry.id)
        self._active_pending_prompt_entries[entry.id] = entry

    def _record_active_entry(self, entry: ConversationEntry) -> None:
        if entry.type == "message":
            self._record_message_entry(entry)
        elif entry.type == "approval_request":
            raise ConversationIntegrityError(
                "standalone approval requests are not supported"
            )
        elif entry.type == "approval_resolution":
            request_id = entry.data.get("request_id")
            if request_id not in self._active_approval_requests:
                raise ConversationIntegrityError(
                    f"approval resolution is not linked to request: {request_id}"
                )
            if request_id in self._active_approval_resolutions:
                raise ConversationIntegrityError(
                    f"duplicate approval resolution: {request_id}"
                )
            self._active_approval_resolutions[request_id] = entry.data["decision"]
        elif entry.type == "notification":
            self._record_notification_entry(entry)
        elif entry.type == "notification_ack":
            notification_id = entry.data.get("notification_id")
            if notification_id not in self._notification_ids:
                raise ConversationIntegrityError(
                    f"notification acknowledgement is not linked: {notification_id}"
                )
            if notification_id not in self._active_notifications:
                raise ConversationIntegrityError(
                    f"duplicate notification acknowledgement: {notification_id}"
                )
            del self._active_notifications[notification_id]
        elif entry.type == "notification_tui_presented":
            notification_id = entry.data.get("notification_id")
            if notification_id not in self._notification_ids:
                raise ConversationIntegrityError(
                    f"notification TUI presentation is not linked: {notification_id}"
                )
            if notification_id in self._active_notification_presentations:
                raise ConversationIntegrityError(
                    f"duplicate notification TUI presentation: {notification_id}"
                )
            self._active_notification_presentations.add(notification_id)
        elif entry.type == "pending_prompt":
            self._record_pending_prompt_entry(entry)
        elif entry.type == "pending_prompt_ack":
            prompt_id = entry.data.get("prompt_id")
            if prompt_id not in self._pending_prompt_ids:
                raise ConversationIntegrityError(
                    f"pending prompt acknowledgement is not linked: {prompt_id}"
                )
            if prompt_id not in self._active_pending_prompt_entries:
                raise ConversationIntegrityError(
                    f"duplicate pending prompt acknowledgement: {prompt_id}"
                )
            del self._active_pending_prompt_entries[prompt_id]
            self._active_pending_prompt_messages.pop(prompt_id, None)
