"""Exhaustive runtime-to-wire event ownership for zeta serve."""

from collections.abc import Mapping
from enum import Enum, auto

from ..core.checkpoints import ConversationEntry
from ..protocol.types import Message, StreamEvent, StreamEventType
from .protocol import TOOL_OUTPUT_MAX_BYTES, bounded


def delivery_status_row(
    entry: ConversationEntry, accepted_message: Message | None
) -> dict[str, object]:
    """Project one committed steer terminal transition."""
    data = entry.data
    row: dict[str, object] = {
        "id": entry.id,
        "role": "system",
        "event": "delivery_status",
        "delivery_id": data["delivery_id"],
        "method": "steer",
        "status": data["status"],
        "content": [],
    }
    if "reason" in data:
        row["reason"] = data["reason"]
    if accepted_message is not None:
        text = next(
            (
                block.text
                for block in accepted_message.content
                if hasattr(block, "text")
            ),
            "",
        )
        row["content"] = [{"type": "text", "text": bounded(text)}]
    return row


def image_stub(block: Mapping[str, object]) -> dict[str, object]:
    """Project stored image metadata without sending image bytes or paths."""
    stub = {
        key: block[key]
        for key in ("type", "mimeType", "caption", "width", "height", "size")
        if key in block
    }
    data = block.get("data")
    if "size" not in stub and isinstance(data, str):
        padding = len(data) - len(data.rstrip("="))
        stub["size"] = len(data) * 3 // 4 - padding
    stub["data_omitted"] = True
    return stub


def project_tool_value(value: object) -> tuple[object, bool]:
    """Bound string leaves in tool wire data."""
    if isinstance(value, str):
        projected = bounded(value, TOOL_OUTPUT_MAX_BYTES)
        return projected, projected != value
    if isinstance(value, dict):
        result: dict[object, object] = {}
        truncated = False
        for key, item in value.items():
            projected, item_truncated = project_tool_value(item)
            result[key] = projected
            truncated |= item_truncated
        return result, truncated
    if isinstance(value, (list, tuple)):
        result = []
        truncated = False
        for item in value:
            projected, item_truncated = project_tool_value(item)
            result.append(projected)
            truncated |= item_truncated
        return result, truncated
    return value, False

LEGACY_NOTIFICATION_WIRE_EVENT = "sub_agent_receipt"
NOTIFICATION_WIRE_EVENTS = {
    "agent_completion": LEGACY_NOTIFICATION_WIRE_EVENT,
    "child_idle": "child_idle",
    "task_exited": "task_exit_notification",
    "project_inbox": "project_inbox_notification",
    "child_question": "child_question",
    "child_question_withdrawn": "child_question_withdrawn",
}


def notification_wire_event(kind: object, protocol_version: str) -> str:
    """Preserve the legacy notification event until protocol 1.2."""

    if protocol_version != "1.2":
        return LEGACY_NOTIFICATION_WIRE_EVENT
    return NOTIFICATION_WIRE_EVENTS.get(kind, LEGACY_NOTIFICATION_WIRE_EVENT)


class AuxiliaryEvent(Enum):
    MEMORY_UPDATED = auto()
    TURN_ABORTED = auto()
    USAGE = auto()
    USER_MESSAGE = auto()


AUXILIARY_WIRE_EVENTS = {
    AuxiliaryEvent.MEMORY_UPDATED: "memory_updated",
    AuxiliaryEvent.TURN_ABORTED: "turn_aborted",
    AuxiliaryEvent.USAGE: "usage",
    AuxiliaryEvent.USER_MESSAGE: "user_message",
}

# Custom projections in server.py use this exhaustive contract.
def _tool_call(event: StreamEvent) -> dict[str, object] | None:
    return event.tool_call.to_dict() if event.tool_call is not None else None


def _data_text(data: Mapping[str, object]) -> str:
    value = data.get("output", data.get("text", ""))
    return value if isinstance(value, str) else ""


def assistant_message(event: StreamEvent) -> dict[str, object]:
    if event.message is None:
        return {}
    message = event.message.to_dict()
    message.pop("metadata", None)
    return {"message": message}


def _project_tool_call(tool_call: dict[str, object] | None) -> dict[str, object] | None:
    if tool_call is None:
        return None
    arguments, truncated = project_tool_value(tool_call.get("arguments", {}))
    return {**tool_call, "arguments": arguments, "arguments_truncated": truncated}


def structured_content_truncated(
    structured_content: object, *, wire_truncated: bool
) -> bool | None:
    if not isinstance(structured_content, dict):
        return True if wire_truncated else None
    stdout = structured_content.get("stdout_truncated")
    stderr = structured_content.get("stderr_truncated")
    if stdout is True or stderr is True or wire_truncated:
        return True
    if type(stdout) is bool and type(stderr) is bool:
        return False
    return None


def tool_start(event: StreamEvent, *, fidelity: bool = False) -> dict[str, object]:
    tool_call = _tool_call(event)
    if fidelity:
        tool_call = _project_tool_call(tool_call)
    return {"tool_call": tool_call, "data": dict(event.data)}


def tool_output(event: StreamEvent, *, fidelity: bool = False) -> dict[str, object]:
    data = dict(event.data)
    if fidelity:
        data, _ = project_tool_value(data)
    tool_call = _tool_call(event)
    if fidelity:
        tool_call = _project_tool_call(tool_call)
    return {
        "tool_call": tool_call,
        "output": bounded(event.delta or _data_text(event.data)),
        "data": data,
    }


def approval_request(payload: Mapping[str, object]) -> dict[str, object]:
    return dict(payload)


def approval_end(payload: Mapping[str, object]) -> dict[str, object]:
    return dict(payload)


def tool_end(event: StreamEvent, *, fidelity: bool = False) -> dict[str, object]:
    tool_call = _tool_call(event)
    result = event.tool_result.to_dict() if event.tool_result else None
    if fidelity:
        tool_call = _project_tool_call(tool_call)
        if result is not None:
            structured = result.get("structured_content")
            projected, _ = project_tool_value(result)
            if structured is not None:
                projected_structured, wire_truncated = project_tool_value(structured)
                projected["structured_content"] = projected_structured
                projected["structured_content_truncated"] = structured_content_truncated(
                    structured, wire_truncated=wire_truncated
                )
            result = projected
    return {
        "tool_call": tool_call,
        "tool_result": result,
        "data": dict(event.data),
    }


STREAM_EVENT_WIRE_EVENTS = {
    StreamEventType.AGENT_START: "agent_start",
    StreamEventType.TURN_START: "turn_start",
    StreamEventType.COMPACTION_START: "compaction_start",
    StreamEventType.COMPACTION_END: "compaction_end",
    StreamEventType.MESSAGE_START: "message_start",
    StreamEventType.MESSAGE_UPDATE: "assistant_delta",
    StreamEventType.MESSAGE_END: "assistant_message",
    StreamEventType.RETRY: "retry",
    StreamEventType.ASSISTANT_RESET: "assistant_reset",
    StreamEventType.TOOL_APPROVAL_START: "approval_request",
    StreamEventType.TOOL_APPROVAL_END: "approval_end",
    StreamEventType.TOOL_EXECUTION_START: "tool_start",
    StreamEventType.TOOL_EXECUTION_UPDATE: "tool_output",
    StreamEventType.TOOL_EXECUTION_END: "tool_end",
    StreamEventType.AGENT_NOTIFICATION: "notification_by_kind",
    StreamEventType.CHILD_MESSAGE_DELIVERED: "child_message_delivered",
    StreamEventType.ACCOUNT_LIMIT: "account_limit",
    StreamEventType.ACCOUNT_STATE_UNKNOWN: "account_state_unknown",
    StreamEventType.TURN_END: "turn_end",
    StreamEventType.AGENT_END: "agent_end",
    StreamEventType.ERROR: "error",
}
