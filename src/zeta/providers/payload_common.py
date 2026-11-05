"""Shared provider payload constants."""

from __future__ import annotations

from collections.abc import Sequence

from ..protocol.types import ContentBlock, ImageContent, Message

HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER = (
    "[harness-injected system message; not written by the user]"
)


def is_harness_notification(message: Message) -> bool:
    """Return whether a system-role message is conversational harness data."""
    return message.metadata.get("zeta_event") == "agent_notifications"


def images_before_prompt(blocks: Sequence[ContentBlock]) -> list[ContentBlock]:
    """Place images before their prompt so vision models attend to them reliably."""
    return [
        *(block for block in blocks if isinstance(block, ImageContent)),
        *(block for block in blocks if not isinstance(block, ImageContent)),
    ]
