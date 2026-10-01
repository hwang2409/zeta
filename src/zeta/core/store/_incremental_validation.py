"""Incremental integrity indexes for conversation log tails."""

from __future__ import annotations

import copy

from ..checkpoints import ConversationEntry, ConversationIntegrityError
from ._validation import TASK_EXITED_NOTIFICATION_KIND


class IncrementalValidationMixin:
    """Validate linear appended entries without rescanning the full log."""

    def _reset_incremental_validation_state(self) -> None:
        self._entry_ids: set[str] = set()
        self._active_approval_requests: dict[str, ConversationEntry] = {}
        self._active_approval_resolutions: set[str] = set()
        self._active_notifications: set[str] = set()
        self._active_notification_acks: set[str] = set()
        self._active_notification_presentations: set[str] = set()
        self._active_pending_prompts: set[str] = set()
        self._active_pending_prompt_acks: set[str] = set()

    def _rebuild_incremental_validation_state(self) -> None:
        """Build O(1) integrity indexes after a complete validation pass."""
        self._reset_incremental_validation_state()
        self._entry_ids = {entry.id for entry in self._entries}
        by_id = {entry.id: entry for entry in self._entries}
        branch: list[ConversationEntry] = []
        current = self._entries[-1] if self._entries else None
        while current is not None:
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        for entry in reversed(branch):
            self._record_active_entry(entry)

    def _accept_incremental_entries(
        self, entries: list[ConversationEntry]
    ) -> bool:
        """Validate a tail against staged state, then publish it as one update."""
        staged = copy.copy(self)
        staged._entries = list(self._entries)
        staged._entry_ids = set(self._entry_ids)
        staged._active_approval_requests = dict(self._active_approval_requests)
        staged._active_approval_resolutions = set(self._active_approval_resolutions)
        staged._active_notifications = set(self._active_notifications)
        staged._active_notification_acks = set(self._active_notification_acks)
        staged._active_notification_presentations = set(
            self._active_notification_presentations
        )
        staged._active_pending_prompts = set(self._active_pending_prompts)
        staged._active_pending_prompt_acks = set(self._active_pending_prompt_acks)
        staged._task_notification_ids = set(self._task_notification_ids)
        for entry in entries:
            if not staged._accept_incremental_entry(entry):
                return False

        self._before_incremental_state_install()
        self._entry_ids = staged._entry_ids
        self._active_approval_requests = staged._active_approval_requests
        self._active_approval_resolutions = staged._active_approval_resolutions
        self._active_notifications = staged._active_notifications
        self._active_notification_acks = staged._active_notification_acks
        self._active_notification_presentations = (
            staged._active_notification_presentations
        )
        self._active_pending_prompts = staged._active_pending_prompts
        self._active_pending_prompt_acks = staged._active_pending_prompt_acks
        self._task_notification_ids = staged._task_notification_ids
        # Publish entries last so readers never see rows without matching indexes.
        self._entries = staged._entries
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

    def _record_active_entry(self, entry: ConversationEntry) -> None:
        if entry.type == "message":
            for request in entry.data.get("approval_requests", []):
                request_id = request.get("request_id")
                if type(request_id) is str and request_id:
                    if request_id in self._active_approval_requests:
                        raise ConversationIntegrityError(
                            f"duplicate approval request: {request_id}"
                        )
                    self._active_approval_requests[request_id] = entry
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
            self._active_approval_resolutions.add(request_id)
        elif entry.type == "notification":
            self._active_notifications.add(entry.id)
        elif entry.type == "notification_ack":
            notification_id = entry.data.get("notification_id")
            if notification_id not in self._active_notifications:
                raise ConversationIntegrityError(
                    f"notification acknowledgement is not linked: {notification_id}"
                )
            if notification_id in self._active_notification_acks:
                raise ConversationIntegrityError(
                    f"duplicate notification acknowledgement: {notification_id}"
                )
            self._active_notification_acks.add(notification_id)
        elif entry.type == "notification_tui_presented":
            notification_id = entry.data.get("notification_id")
            if notification_id not in self._active_notifications:
                raise ConversationIntegrityError(
                    f"notification TUI presentation is not linked: {notification_id}"
                )
            if notification_id in self._active_notification_presentations:
                raise ConversationIntegrityError(
                    f"duplicate notification TUI presentation: {notification_id}"
                )
            self._active_notification_presentations.add(notification_id)
        elif entry.type == "pending_prompt":
            self._active_pending_prompts.add(entry.id)
        elif entry.type == "pending_prompt_ack":
            prompt_id = entry.data.get("prompt_id")
            if prompt_id not in self._active_pending_prompts:
                raise ConversationIntegrityError(
                    f"pending prompt acknowledgement is not linked: {prompt_id}"
                )
            if prompt_id in self._active_pending_prompt_acks:
                raise ConversationIntegrityError(
                    f"duplicate pending prompt acknowledgement: {prompt_id}"
                )
            self._active_pending_prompt_acks.add(prompt_id)
