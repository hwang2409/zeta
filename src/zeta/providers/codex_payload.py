"""Codex Responses request payload encoding."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..types import (
    ContentBlock,
    flatten_tool_content,
    Message,
    MessageRole,
    TextContent,
    ThinkingContent,
    ToolSchema,
    ToolUseContent,
)
from .codex_errors import CodexHTTPError


def _wire_text(blocks: Sequence[ContentBlock], *, output: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "output_text" if output else "input_text", "text": block.text})
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


def build_responses_payload(
    messages: Sequence[Message],
    tool_schemas: Sequence[ToolSchema],
    *,
    model: str,
) -> dict[str, Any]:
    """Build a Responses request.

    Codex has no native image tool-result block. Its one fallback is the
    provider-neutral text description emitted by ``flatten_tool_content``.
    """
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            instructions.extend(
                block.text for block in message.content if isinstance(block, TextContent)
            )
        elif message.role is MessageRole.TOOL_RESULT:
            if message.tool_result is None:
                raise CodexHTTPError("tool result message is missing its result")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_result.tool_call_id,
                    "output": (
                        flatten_tool_content(
                            message.tool_result.content_blocks,
                            detailed_images=True,
                        )
                        if message.tool_result.content_blocks is not None
                        else message.tool_result.content
                    ),
                }
            )
        else:
            output = message.role is MessageRole.ASSISTANT
            if output and "codex_output_items" in message.metadata:
                replayed = message.metadata["codex_output_items"]
                if type(replayed) is not list or any(
                    not isinstance(item, Mapping) for item in replayed
                ):
                    raise CodexHTTPError("Codex replay output items are invalid")
                input_items.extend(dict(item) for item in replayed)
                continue
            wire_blocks = _wire_text(message.content, output=output)
            if not output:
                input_items.append({"role": "user", "content": wire_blocks})
                continue
            text_blocks: list[dict[str, Any]] = []
            for block in wire_blocks:
                if block.get("type") == "output_text":
                    text_blocks.append({"type": "input_text", "text": block["text"]})
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
