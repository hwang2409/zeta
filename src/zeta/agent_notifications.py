"""Durable background-agent notification inputs and events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable

from .core.abort import AbortSignal as ToolAbortSignal
from .core.store import ConversationStore
from .types import Message, MessageRole, StreamEvent, StreamEventType, TextContent


def notification_events(store: ConversationStore) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for notification in store.agent_notifications():
        events.append(
            StreamEvent(
                StreamEventType.AGENT_NOTIFICATION,
                data={"notification_id": notification.id, **notification.data},
            )
        )
        store.acknowledge_agent_notification(notification.id)
    return events


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
        self._background_wake_callback = callback

    def _background_notification_persisted(self) -> None:
        callback = self._background_wake_callback
        if callback is not None and not self._turn_active and not self._closed:
            callback()

    def notification_system_message(self) -> Message | None:
        return build_notification_system_message(self.store)

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
