"""Anthropic Messages request payload encoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..types import (
    SUPPORTED_IMAGE_MEDIA_TYPES,
    ContentBlock,
    ImageContent,
    Message,
    MessageRole,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
    ToolImageBlock,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    decoded_image_bytes,
    flatten_tool_content,
    image_description,
    image_dimensions,
    image_signature_matches,
)

ANTHROPIC_MAX_IMAGE_BYTES = 5 * 1024 * 1024
ANTHROPIC_MAX_IMAGE_DIMENSION = 8000


def _image_block_from_content(content: ImageContent) -> ToolImageBlock:
    block: ToolImageBlock = {
        "type": "image",
        "data": content.data,
        "mimeType": content.mime_type,
    }
    if content.path is not None:
        block["path"] = content.path
    if content.size is not None:
        block["size"] = content.size
    return block


def _image_wire_block(
    image: ToolImageBlock,
) -> tuple[dict[str, Any] | None, str | None]:
    data = decoded_image_bytes(image)
    dimensions = image_dimensions(image, data)
    if image["mimeType"] not in SUPPORTED_IMAGE_MEDIA_TYPES:
        return None, f"unsupported media type {image['mimeType']}"
    if data is None:
        return None, "invalid base64 payload"
    if len(data) > ANTHROPIC_MAX_IMAGE_BYTES:
        return None, (
            f"image is {len(data)} bytes; limit is "
            f"{ANTHROPIC_MAX_IMAGE_BYTES} bytes"
        )
    if not image_signature_matches(image["mimeType"], data):
        return None, "invalid image data"
    if dimensions is not None and any(
        dimension > ANTHROPIC_MAX_IMAGE_DIMENSION for dimension in dimensions
    ):
        return None, (
            f"image dimensions are {dimensions[0]}x{dimensions[1]}; "
            f"limit is {ANTHROPIC_MAX_IMAGE_DIMENSION}x"
            f"{ANTHROPIC_MAX_IMAGE_DIMENSION}"
        )
    return (
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image["mimeType"],
                "data": image["data"],
            },
        },
        None,
    )


def _wire_content(blocks: Sequence[ContentBlock]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            image = _image_block_from_content(block)
            wire, reason = _image_wire_block(image)
            if wire is not None:
                result.append(wire)
            else:
                result.append(
                    {
                        "type": "text",
                        "text": image_description(image, detailed=True, reason=reason),
                    }
                )
        elif isinstance(block, ThinkingContent):
            if not block.signature:
                continue
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.text,
                    "signature": block.signature,
                }
            )
        elif isinstance(block, RedactedThinkingContent):
            result.append({"type": "redacted_thinking", "data": block.data})
        elif isinstance(block, ToolUseContent):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.tool_call.id,
                    "name": block.tool_call.name,
                    "input": block.tool_call.arguments,
                }
            )
        else:
            raise ValueError("unsupported zeta content block")
    return result


def _wire_tool_result_content(result: ToolResult) -> str | list[dict[str, Any]]:
    """Encode tool images natively, with text fallback for API limits."""

    if result.content_blocks is None:
        return result.content
    has_native_image = False
    wire_blocks: list[dict[str, Any]] = []
    for block in result.content_blocks:
        if block["type"] == "text":
            wire_blocks.append({"type": "text", "text": flatten_tool_content([block])})
            continue
        if block["type"] != "image":
            wire_blocks.append(
                {"type": "text", "text": flatten_tool_content([block])}
            )
            continue
        image = block
        wire, reason = _image_wire_block(image)
        if wire is not None:
            caption = image.get("caption")
            if caption:
                wire_blocks.append({"type": "text", "text": f"caption: {caption}"})
            wire_blocks.append(wire)
            has_native_image = True
            continue
        wire_blocks.append(
            {
                "type": "text",
                "text": image_description(image, detailed=True, reason=reason),
            }
        )
    if has_native_image:
        return wire_blocks
    return "\n".join(block["text"] for block in wire_blocks)


def build_messages_payload(
    messages: Sequence[Message],
    tool_schemas: Sequence[ToolSchema],
    *,
    model: str,
    max_tokens: int,
    thinking_budget: int = 8_192,
) -> dict[str, Any]:
    _validate_thinking_parameters(max_tokens, thinking_budget)
    system: list[dict[str, Any]] = []
    wire_messages: list[dict[str, Any]] = []
    latest_user_wire_index: int | None = None
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            content = _wire_content(message.content)
            if content and any(
                block.get("type") != "text" or block.get("text", "").strip()
                for block in content
            ):
                system.extend(content)
            continue
        if message.role is MessageRole.TOOL_RESULT:
            if message.tool_result is None:
                raise ValueError("tool result message is missing its result")
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_result.tool_call_id,
                    "content": _wire_tool_result_content(message.tool_result),
                    "is_error": message.tool_result.is_error,
                }
            ]
            wire_messages.append({"role": "user", "content": content})
            continue
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
        content = _wire_content(message.content)
        if content or role != "assistant":
            if message.role is MessageRole.USER:
                latest_user_wire_index = len(wire_messages)
            wire_messages.append({"role": role, "content": content})

    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
    tools = [_wire_tool_schema(schema) for schema in tool_schemas]
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        "messages": wire_messages,
        "stream": True,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools
    # The active user turn can grow during tool calls, so cache only completed
    # conversation history before that turn.
    if latest_user_wire_index is not None and latest_user_wire_index > 0:
        prefix_message = wire_messages[latest_user_wire_index - 1]
        content = prefix_message["content"]
        if isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}
    return payload


def _validate_thinking_parameters(max_tokens: int, thinking_budget: int) -> None:
    if thinking_budget < 1024:
        raise ValueError("thinking_budget must be at least 1024 tokens")
    if thinking_budget >= max_tokens:
        raise ValueError("max_tokens must exceed thinking_budget")


def _wire_tool_schema(schema: ToolSchema) -> dict[str, Any]:
    name = schema.get("name")
    if type(name) is not str or not name:
        raise ValueError("tool schema name must be a nonempty string")
    input_schema = schema.get("input_schema", schema.get("parameters"))
    if not isinstance(input_schema, Mapping):
        input_schema = {
            key: value
            for key, value in schema.items()
            if key not in {"name", "description", "cache_control"}
        }
    result: dict[str, Any] = {"name": name, "input_schema": dict(input_schema)}
    description = schema.get("description")
    if type(description) is str:
        result["description"] = description
    return result


__all__ = [
    "ANTHROPIC_MAX_IMAGE_BYTES",
    "ANTHROPIC_MAX_IMAGE_DIMENSION",
    "build_messages_payload",
]
