"""Small durable-message normalization shared by agent loops."""

from __future__ import annotations

from ..protocol.types import Message, ThinkingContent


def durable_message(message: Message) -> Message:
    content = [
        block
        for block in message.content
        if not isinstance(block, ThinkingContent) or not block.text or block.signature
    ]
    if len(content) == len(message.content):
        return message
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=dict(message.metadata),
    )
