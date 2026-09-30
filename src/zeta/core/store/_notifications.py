"""Durable notification query, presentation, and consumption state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..checkpoints import ConversationEntry

if TYPE_CHECKING:
    from ._store import ConversationStore


class NotificationStateMixin:
    """Keep human presentation separate from parent-agent consumption."""

    def agent_notifications(
        self: ConversationStore, *, pending_only: bool = True
    ) -> list[ConversationEntry]:
        """Return durable background-child notifications on the active branch."""

        branch = self.replay()
        acknowledged = {
            entry.data["notification_id"]
            for entry in branch
            if entry.type == "notification_ack"
        }
        return [
            entry
            for entry in branch
            if entry.type == "notification"
            and (not pending_only or entry.id not in acknowledged)
        ]

    def is_agent_notification_presented_to_tui(
        self: ConversationStore, notification_id: str
    ) -> bool:
        """Return whether the TUI has already shown this notification."""

        return any(
            entry.type == "notification_tui_presented"
            and entry.data["notification_id"] == notification_id
            for entry in self.replay()
        )

    def mark_agent_notification_presented_to_tui(
        self: ConversationStore, notification_id: str
    ) -> None:
        """Durably record successful TUI output without consuming the notification."""

        notifications = self.agent_notifications(pending_only=False)
        if not any(entry.id == notification_id for entry in notifications):
            raise ValueError(f"unknown agent notification: {notification_id}")
        if self.is_agent_notification_presented_to_tui(notification_id):
            return
        self._append_row(
            "notification_tui_presented", {"notification_id": notification_id}
        )

    def acknowledge_agent_notification(
        self: ConversationStore, notification_id: str
    ) -> None:
        """Durably mark one notification as consumed by the parent agent."""

        notifications = self.agent_notifications(pending_only=False)
        if not any(entry.id == notification_id for entry in notifications):
            raise ValueError(f"unknown agent notification: {notification_id}")
        if any(
            entry.data["notification_id"] == notification_id
            for entry in self.replay()
            if entry.type == "notification_ack"
        ):
            return
        self._append_row("notification_ack", {"notification_id": notification_id})
