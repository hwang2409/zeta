"""Validation for immutable approval display facts in audit records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ...protocol.types import Message, ToolCall, ToolUseContent

_STRING_FIELDS = frozenset(
    {
        "project_id",
        "project_name",
        "filename",
        "preview",
        "effective_cwd",
        "resolved_path",
    }
)
_FIELDS = _STRING_FIELDS | {"utf8_bytes"}


def validated_approval_display(display: Mapping[str, object]) -> dict[str, object]:
    """Copy immutable display-only approval facts into the audit record."""

    if not isinstance(display, Mapping):
        raise TypeError("approval display must be an object")
    unknown = set(display) - _FIELDS
    if unknown:
        raise ValueError(f"unknown approval display fields: {sorted(unknown)}")
    copied: dict[str, object] = {}
    for name, value in display.items():
        if name in _STRING_FIELDS:
            if value is not None and type(value) is not str:
                raise ValueError(f"approval display {name} must be a string or null")
        elif value is not None and (type(value) is not int or value < 0):
            raise ValueError(
                "approval display utf8_bytes must be a nonnegative integer or null"
            )
        copied[name] = value
    return copied


def normalize_approval_requests(
    message: Message,
    approval_requests: Iterable[
        tuple[str, ToolCall] | tuple[str, ToolCall, Mapping[str, object]]
    ],
) -> list[dict[str, object]]:
    """Validate and serialize requests anchored in an assistant message."""

    request_data: list[dict[str, object]] = []
    request_ids: set[str] = set()
    anchored_calls = {
        block.tool_call.id: block.tool_call
        for block in message.content
        if isinstance(block, ToolUseContent)
    }
    for approval_request in approval_requests:
        if len(approval_request) == 2:
            request_id, tool_call = approval_request
            approval_display = None
        else:
            request_id, tool_call, raw_display = approval_request
            approval_display = validated_approval_display(raw_display)
        if type(request_id) is not str or not request_id:
            raise ValueError("approval request id must be a nonempty string")
        normalized_tool_call = ToolCall.from_dict(tool_call.to_dict())
        if request_id in request_ids:
            raise ValueError(f"duplicate approval request: {request_id}")
        if anchored_calls.get(request_id) != normalized_tool_call:
            raise ValueError("approval request must match an anchored tool call")
        request_ids.add(request_id)
        persisted_request: dict[str, object] = {
            "request_id": request_id,
            "tool_call": normalized_tool_call.to_dict(),
        }
        if approval_display is not None:
            persisted_request["approval_display"] = approval_display
        request_data.append(persisted_request)
    return request_data
