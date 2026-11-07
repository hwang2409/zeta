"""Shared provider payload constants."""

from __future__ import annotations

from ..protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    ContentBlock,
    ImageContent,
    Message,
    MessageOrigin,
    TextContent,
)

HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER = (
    "[harness-injected system message; not written by the user]"
)


def is_harness_notification(message: Message) -> bool:
    """Return whether a system-role message is conversational harness data."""
    return message.metadata.get("zeta_event") == "agent_notifications" or (
        message.role.value == "system"
        and message.metadata.get(MESSAGE_ORIGIN_METADATA)
        == MessageOrigin.HARNESS_NUDGE.value
    )


def images_before_prompt(message: Message) -> list[ContentBlock]:
    """Place canonical user attachments before their prompt."""
    blocks = message.content
    if (
        message.metadata.get(MESSAGE_ORIGIN_METADATA) == MessageOrigin.USER
        and blocks
        and isinstance(blocks[0], TextContent)
        and all(isinstance(block, ImageContent) for block in blocks[1:])
    ):
        return [*blocks[1:], blocks[0]]
    return list(blocks)
