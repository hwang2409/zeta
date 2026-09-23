"""Durable background-agent notification inputs and events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from .core.abort import AbortSignal as ToolAbortSignal
from .core.store import ConversationStore
from .types import Message, MessageRole, StreamEvent, StreamEventType, TextContent


def notification_events(store: ConversationStore) -> Iterator[StreamEvent]:
    for notification in store.agent_notifications():
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


def start_notification_wake(pipeline: Any, provider_done: Callable[..., Any]) -> None:
    """Start one notification turn through the submission owner."""

    if (
        pipeline._provider_entry is not None
        or pipeline._host.loop.notification_system_message() is None
    ):
        return
    submission = pipeline._new_submission("", 0, (), None, 1, steer=False)
    entry = pipeline._notification_entry(submission)
    pipeline._provider_entry = entry
    task = pipeline._provider_task = pipeline._host._start_turn(
        "",
        submission_id=submission.id,
        abort_signal=entry.signal,
        notification=True,
    )
    task.add_done_callback(
        lambda completed: pipeline._send(provider_done(submission, completed))
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
        self, *, message_persisted: bool = False
    ) -> Iterator[StreamEvent]:
        message = None if message_persisted else build_notification_system_message(self.store)
        if message is None and not message_persisted:
            return
        yield from notification_events(self.store)
        if message is not None:
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
