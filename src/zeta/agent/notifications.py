"""Durable background-agent notification inputs and events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Collection, Iterator

from ..core.abort import AbortSignal as ToolAbortSignal
from ..core.store import ConversationStore
from ..protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)


def notification_events(
    store: ConversationStore,
    notification_ids: Collection[str] | None = None,
) -> Iterator[StreamEvent]:
    notifications = store.agent_notifications()
    if notification_ids is not None:
        notifications = [
            notification
            for notification in notifications
            if notification.id in notification_ids
        ]
    for notification in notifications:
        yield StreamEvent(
            StreamEventType.AGENT_NOTIFICATION,
            data={"notification_id": notification.id, **notification.data},
        )
        store.acknowledge_agent_notification(notification.id)


def build_notification_system_message(store: ConversationStore) -> Message | None:
    """Build the system input for one durable background completion wake."""

    notifications = store.agent_notifications()
    if not notifications:
        return None
    payload = [
        {"notification_id": entry.id, **entry.data}
        for entry in notifications
    ]
    return Message(
        MessageRole.SYSTEM,
        [
            TextContent(
                "background agent completion notifications:\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )
        ],
        metadata={"zeta_event": "agent_notifications", "notifications": payload},
    )


class AgentNotificationMixin:
    """Add durable notification wake inputs to an agent loop."""

    def set_background_wake_callback(
        self, callback: Callable[[], None] | None
    ) -> None:
        if callback is None:
            self._background_owner.set_wake_callback(None)
            return
        self._background_owner.set_wake_callback(
            lambda: callback() if not self._turn_active and not self._closed else None
        )

    def _background_notification_persisted(self) -> None:
        self._background_owner.notify_wake()

    def notification_system_message(self) -> Message | None:
        return build_notification_system_message(self.store)

    def drain_notification_batch(
        self,
        *,
        message_persisted: bool = False,
        message: Message | None = None,
    ) -> Iterator[StreamEvent]:
        if message is None:
            message = build_notification_system_message(self.store)
        if message is None:
            return
        notification_ids = tuple(
            entry["notification_id"] for entry in message.metadata["notifications"]
        )
        yield from notification_events(self.store, notification_ids)
        if not message_persisted:
            self.store.append_message(message)

    def run_notification_turn(
        self, *, abort_signal: ToolAbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        message = self.notification_system_message()
        if message is None:
            raise RuntimeError("no pending agent notifications")
        return self._run_turn(
            "",
            system_message=message,
            persist_user_message=False,
            abort_signal=abort_signal,
        )
