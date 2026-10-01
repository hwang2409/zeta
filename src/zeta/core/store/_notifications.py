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

    def record_agent_notification_delivery(
        self: ConversationStore,
        notification_id: str,
        *,
        presented: bool = False,
        acknowledged: bool = False,
    ) -> None:
        """Persist ordered presentation/ack markers in one durable append batch.

        When both flags are true the presentation marker precedes the
        acknowledgement and both rows are written under one flock and fsync.
        Existing markers are omitted, so retrying is idempotent.
        """
        if not presented and not acknowledged:
            return
        with self._append_lock():
            self._load()
            branch = self.replay()
            if not any(
                entry.type == "notification" and entry.id == notification_id
                for entry in branch
            ):
                raise ValueError(f"unknown agent notification: {notification_id}")
            rows: list[tuple[str, dict[str, str]]] = []
            if presented and not any(
                entry.type == "notification_tui_presented"
                and entry.data["notification_id"] == notification_id
                for entry in branch
            ):
                rows.append(
                    ("notification_tui_presented", {"notification_id": notification_id})
                )
            if acknowledged and not any(
                entry.type == "notification_ack"
                and entry.data["notification_id"] == notification_id
                for entry in branch
            ):
                rows.append(("notification_ack", {"notification_id": notification_id}))
            if rows:
                self._append_many_unlocked(rows)

    async def record_agent_notification_delivery_async(
        self: ConversationStore,
        notification_id: str,
        *,
        presented: bool = False,
        acknowledged: bool = False,
    ) -> None:
        """Persist delivery state off the event loop and in async call order."""
        await self._to_thread_durable(
            self.record_agent_notification_delivery,
            notification_id,
            presented=presented,
            acknowledged=acknowledged,
        )

    def mark_agent_notification_presented_to_tui(
        self: ConversationStore, notification_id: str
    ) -> None:
        """Durably record successful TUI output without consuming the notification."""

        self.record_agent_notification_delivery(notification_id, presented=True)

    def acknowledge_agent_notification(
        self: ConversationStore, notification_id: str
    ) -> None:
        """Durably mark one notification as consumed by the parent agent."""

        self.record_agent_notification_delivery(notification_id, acknowledged=True)
