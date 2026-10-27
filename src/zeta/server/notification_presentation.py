"""Per-process exactly-once presentation for serve notifications."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from zeta.server import event_contract
from zeta.server.protocol import FrameWriteCancelled

NotificationSender = Callable[..., Awaitable[bool]]


class NotificationPresentation:
    """Publish each durable notification at most once per serve session."""

    def __init__(self) -> None:
        self._presented: set[tuple[str, str]] = set()

    async def publish(
        self,
        session_id: str | None,
        notification: Mapping[str, object],
        protocol_version: str,
        send: NotificationSender,
    ) -> None:
        notification_id = notification.get("notification_id")
        key = (
            (session_id, notification_id)
            if isinstance(session_id, str) and isinstance(notification_id, str)
            else None
        )
        if key is not None and key in self._presented:
            return
        if key is not None:
            self._presented.add(key)
        event_name = event_contract.notification_wire_event(
            notification.get("kind", "agent_completion"), protocol_version
        )
        # The claim survives only a frame that reached the client; a send
        # that wrote nothing releases it so a later publish can retry.
        try:
            written = await send(event_name, session_id, data=dict(notification))
        except FrameWriteCancelled as cancelled:
            if not cancelled.written and key is not None:
                self._presented.discard(key)
            raise
        except BaseException:
            if key is not None:
                self._presented.discard(key)
            raise
        if not written and key is not None:
            self._presented.discard(key)
