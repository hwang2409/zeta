"""Shared provider payload constants."""

from __future__ import annotations

from ..protocol.types import Message

HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER = (
    "[harness-injected system message; not written by the user]"
)


def is_harness_notification(message: Message) -> bool:
    """Return whether a system-role message is conversational harness data."""
    return message.metadata.get("zeta_event") == "agent_notifications"
