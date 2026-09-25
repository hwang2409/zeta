"""Codex Responses request payload encoding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..media.images import (
    SUPPORTED_IMAGE_MEDIA_TYPES,
    decoded_image_bytes,
    image_description,
    image_signature_matches,
)
from ..protocol.types import (
    ContentBlock,
    ImageContent,
    Message,
    MessageRole,
    TextContent,
    ThinkingContent,
    ToolImageBlock,
    ToolSchema,
    ToolUseContent,
    flatten_tool_content,
)
from .codex_errors import CodexHTTPError
from .payload_common import HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER


def _image_input_block(image: ToolImageBlock) -> dict[str, Any] | None:
    data = decoded_image_bytes(image)
    if (
        data is None
        or image["mimeType"] not in SUPPORTED_IMAGE_MEDIA_TYPES
        or not image_signature_matches(image["mimeType"], data)
    ):
        return None
    return {
        "type": "input_image",
        "image_url": f"data:{image['mimeType']};base64,{image['data']}",
    }


def _wire_text(blocks: Sequence[ContentBlock], *, output: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "output_text" if output else "input_text", "text": block.text})
        elif isinstance(block, ImageContent):
            if output:
                raise CodexHTTPError("image content is not valid assistant output")
            image: ToolImageBlock = {
                "type": "image",
                "data": block.data,
                "mimeType": block.mime_type,
            }
            if block.path is not None:
                image["path"] = block.path
            if block.size is not None:
                image["size"] = block.size
            wire = _image_input_block(image)
            result.append(
                wire
                or {
                    "type": "input_text",
                    "text": image_description(image, detailed=True),
                }
            )
        elif isinstance(block, ThinkingContent):
            if output:
                reasoning: dict[str, Any] = {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": block.text}],
                }
                if block.signature:
                    reasoning["encrypted_content"] = block.signature
                result.append(reasoning)
        elif isinstance(block, ToolUseContent):
            if output:
                result.append(
                    {
                        "type": "function_call",
                        "call_id": block.tool_call.id,
                        "name": block.tool_call.name,
                        "arguments": json.dumps(
                            block.tool_call.arguments, separators=(",", ":")
                        ),
                    }
                )
            else:
                raise CodexHTTPError("tool calls are not valid user content")
        else:
            raise CodexHTTPError("unsupported zeta content block")
    return result


def _normalize_assistant_item(item: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
        return normalized
    content = []
    for part in item["content"]:
        if not isinstance(part, Mapping):
            content.append(part)
            continue
        normalized_part = dict(part)
        if normalized_part.get("type") == "input_text":
            normalized_part["type"] = "output_text"
        content.append(normalized_part)
    normalized["content"] = content
    return normalized


def build_responses_payload(
    messages: Sequence[Message],
    tool_schemas: Sequence[ToolSchema],
    *,
    model: str,
) -> dict[str, Any]:
    """Build a Responses request.

    Codex receives tool images as a following user image input. The function
    output remains a text receipt because it cannot carry image content here.
    """
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    system_at_head = True
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            if system_at_head:
                instructions.extend(
                    block.text
                    for block in message.content
                    if isinstance(block, TextContent)
                )
            else:
                wire_blocks = _wire_text(message.content, output=False)
                for block in wire_blocks:
                    if block.get("type") == "input_text":
                        block["text"] = f"{HARNESS_INJECTED_SYSTEM_MESSAGE_MARKER}\n{block['text']}"
                if wire_blocks:
                    input_items.append({"role": "user", "content": wire_blocks})
            continue
        system_at_head = False
        if message.role is MessageRole.TOOL_RESULT:
            if message.tool_result is None:
                raise CodexHTTPError("tool result message is missing its result")
            blocks = message.tool_result.content_blocks
            images: list[dict[str, Any]] = []
            if blocks is None:
                receipt = message.tool_result.content
            else:
                receipt_blocks = []
                has_text_block = any(block["type"] == "text" for block in blocks)
                for block in blocks:
                    if block["type"] == "image":
                        wire = _image_input_block(block)
                        if wire is not None:
                            images.append(wire)
                            if not has_text_block:
                                description = image_description(block, detailed=True)
                                receipt_blocks.append(
                                    {
                                        "type": "text",
                                        "text": description,
                                        "truncated": False,
                                        "full_size": len(description.encode()),
                                    }
                                )
                            continue
                    receipt_blocks.append(block)
                receipt = flatten_tool_content(receipt_blocks, detailed_images=True)
                if not receipt and images:
                    receipt = "image attached"
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_result.tool_call_id,
                    "output": receipt,
                }
            )
            if images:
                input_items.append({"role": "user", "content": images})
        else:
            output = message.role is MessageRole.ASSISTANT
            if output and "codex_output_items" in message.metadata:
                replayed = message.metadata["codex_output_items"]
                if type(replayed) is not list or any(
                    not isinstance(item, Mapping) for item in replayed
                ):
                    raise CodexHTTPError("Codex replay output items are invalid")
                input_items.extend(_normalize_assistant_item(item) for item in replayed)
                continue
            wire_blocks = _wire_text(message.content, output=output)
            if not output:
                input_items.append({"role": "user", "content": wire_blocks})
                continue
            text_blocks: list[dict[str, Any]] = []
            for block in wire_blocks:
                if block.get("type") == "output_text":
                    text_blocks.append(block)
                    continue
                if text_blocks:
                    input_items.append({"role": "assistant", "content": text_blocks})
                    text_blocks = []
                input_items.append(block)
            if text_blocks:
                input_items.append({"role": "assistant", "content": text_blocks})

    tools: list[dict[str, Any]] = []
    for schema in tool_schemas:
        name = schema.get("name")
        if type(name) is not str or not name:
            raise CodexHTTPError("tool schema name must be a nonempty string")
        if "parameters" in schema:
            parameters = schema["parameters"]
        elif "input_schema" in schema:
            parameters = schema["input_schema"]
        else:
            parameters = {"type": "object", "properties": {}}
        if not isinstance(parameters, Mapping):
            raise CodexHTTPError("tool schema parameters must be an object")
        tool: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": dict(parameters),
            "strict": False,
        }
        description = schema.get("description")
        if type(description) is str:
            tool["description"] = description
        tools.append(tool)

    payload: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": True,
        "instructions": "\n\n".join(instructions) or "You are a helpful assistant.",
        "input": input_items,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }
    if tools:
        payload["tools"] = tools
    return payload


__all__ = ["build_responses_payload"]
