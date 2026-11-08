"""Shared provider payload constants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..context_eviction.receipt_constructors import (
    RECEIPT_FIELDS_METADATA,
    RECEIPT_KIND_METADATA,
    REASONING_KIND,
    is_receipt_echo,
    reasoning_marker,
)
from ..protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    ContentBlock,
    ImageContent,
    Message,
    MessageOrigin,
    TextContent,
    ToolUseContent,
    harness_notice,
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


def provider_messages(messages: Sequence[Message]) -> list[Message]:
    """Keep eviction receipts out of assistant turns on the provider wire.

    A receipt sent as an assistant turn reads as something the model said, and
    models copy it as their next reply. Stored views stay unchanged; this only
    shapes the request, and returns unchanged messages as the same objects.
    """
    projected = (_provider_message(message) for message in messages)
    return [message for message in projected if message is not None]


def _provider_message(message: Message) -> Message | None:
    if is_receipt_echo(message):
        return None
    metadata = message.metadata
    if metadata.get("context_evicted") is not True or message.tool_result is not None:
        return message
    has_tool_use = any(isinstance(block, ToolUseContent) for block in message.content)
    kind = metadata.get(RECEIPT_KIND_METADATA)
    if kind in ("assistant", "range", "notification") and not has_tool_use:
        fields = metadata.get(RECEIPT_FIELDS_METADATA)
        if isinstance(fields, Mapping) and fields.get("kind") == REASONING_KIND:
            return None
        text = "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        )
        return harness_notice(
            "context eviction", text, event="eviction_receipt", visible=False
        )
    seq = metadata.get("source_seq")
    last = message.content[-1] if message.content else None
    if (
        set(metadata) == {"context_evicted", "source_seq"}
        and type(seq) is int
        and isinstance(last, TextContent)
        and last.text == reasoning_marker(seq)
    ):
        body = message.content[:-1]
        return Message(message.role, body, metadata=dict(metadata)) if body else None
    return message
