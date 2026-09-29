"""Validate tool-call results at the loop boundary before persistence."""

from __future__ import annotations

from collections.abc import Mapping

from ...protocol.types import ToolResult, flatten_tool_content
from ...tools.registry import validate_tool_result


def _validated_tool_result(result: object, expected_id: str) -> ToolResult:
    if isinstance(result, ToolResult):
        if type(result.tool_call_id) is not str or not result.tool_call_id:
            return ToolResult(expected_id, "invalid tool result: call id", True)
        if type(result.content) is not str:
            return ToolResult(expected_id, "invalid tool result: content", True)
        if type(result.is_error) is not bool:
            return ToolResult(expected_id, "invalid tool result: is_error", True)
        if result.tool_call_id != expected_id:
            return ToolResult(
                expected_id,
                f"tool result id mismatch: expected {expected_id}, got {result.tool_call_id}",
                is_error=True,
            )
        return result
    if not isinstance(result, Mapping):
        return ToolResult(
            expected_id,
            "invalid tool result: expected structured result",
            True,
        )
    try:
        structured_result = validate_tool_result(result)
    except ValueError as exc:
        return ToolResult(expected_id, f"invalid tool result: {exc}", True)
    return ToolResult(
            expected_id,
            flatten_tool_content(structured_result["content"]),
            structured_result["isError"],
            content_blocks=structured_result["content"],
            structured_content=structured_result["structuredContent"],
            is_canceled=structured_result.get("isCanceled", False),
    )
