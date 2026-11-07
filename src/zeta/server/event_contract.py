"""Exhaustive runtime-to-wire event ownership for zeta serve."""

from collections.abc import Mapping
from enum import Enum, auto

from ..core.checkpoints import ConversationEntry
from ..protocol.types import Message, StreamEvent, StreamEventType
from .protocol import TOOL_OUTPUT_MAX_BYTES, EventDescriptor, bounded


def delivery_status_row(
    entry: ConversationEntry, accepted_message: Message | None = None
) -> dict[str, object]:
    """Project one committed steer terminal transition."""
    data = entry.data
    status = data["status"]
    text = f"steer {status}"
    row: dict[str, object] = {
        "id": entry.id,
        "role": "system",
        "event": EventDescriptor.DELIVERY_STATUS,
        "delivery_id": data["delivery_id"],
        "method": "steer",
        "status": status,
        "content": [{"type": "text", "text": text}],
    }
    if "reason" in data:
        row["reason"] = data["reason"]
        row["content"] = [{"type": "text", "text": f"{text}: {data['reason']}"}]
    if accepted_message is not None:
        row["content"] = accepted_message.to_dict()["content"]
    return row


def served_turn_end_row(entry: ConversationEntry) -> dict[str, object]:
    """Project one durable served-turn completion fact."""
    return {
        "id": entry.id,
        "role": "system",
        "event": EventDescriptor.SERVED_TURN_END,
        "data": {
            "turn_id": entry.data["turn_id"],
            "reason": entry.data["reason"],
        },
    }


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


LEGACY_NOTIFICATION_WIRE_EVENT = EventDescriptor.SUB_AGENT_RECEIPT
NOTIFICATION_WIRE_EVENTS = {
    "agent_completion": LEGACY_NOTIFICATION_WIRE_EVENT,
    "child_idle": EventDescriptor.CHILD_IDLE,
    "task_exited": EventDescriptor.TASK_EXIT_NOTIFICATION,
    "project_inbox": EventDescriptor.PROJECT_INBOX_NOTIFICATION,
    "child_question": EventDescriptor.CHILD_QUESTION,
    "child_question_withdrawn": EventDescriptor.CHILD_QUESTION_WITHDRAWN,
}


def notification_wire_event(kind: object, protocol_version: str) -> EventDescriptor:
    """Preserve the legacy notification event until protocol 1.2."""

    if protocol_version != "1.2":
        return LEGACY_NOTIFICATION_WIRE_EVENT
    return NOTIFICATION_WIRE_EVENTS.get(kind, LEGACY_NOTIFICATION_WIRE_EVENT)


def notification_payload(notification: Mapping[str, object]) -> dict[str, object]:
    """Preserve additive durable notification fields on the serve wire."""

    return dict(notification)


class AuxiliaryEvent(Enum):
    DELIVERY_STATUS = auto()
    MEMORY_UPDATED = auto()
    TURN_ABORTED = auto()
    USAGE = auto()
    USER_MESSAGE = auto()


AUXILIARY_WIRE_EVENTS = {
    AuxiliaryEvent.DELIVERY_STATUS: EventDescriptor.DELIVERY_STATUS,
    AuxiliaryEvent.MEMORY_UPDATED: EventDescriptor.MEMORY_UPDATED,
    AuxiliaryEvent.TURN_ABORTED: EventDescriptor.TURN_ABORTED,
    AuxiliaryEvent.USAGE: EventDescriptor.USAGE,
    AuxiliaryEvent.USER_MESSAGE: EventDescriptor.USER_MESSAGE,
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
                projected["structured_content_truncated"] = (
                    structured_content_truncated(
                        structured, wire_truncated=wire_truncated
                    )
                )
            result = projected
    return {
        "tool_call": tool_call,
        "tool_result": result,
        "data": dict(event.data),
    }


STREAM_EVENT_WIRE_EVENTS = {
    StreamEventType.AGENT_START: EventDescriptor.AGENT_START,
    StreamEventType.TURN_START: EventDescriptor.TURN_START,
    StreamEventType.COMPACTION_START: EventDescriptor.COMPACTION_START,
    StreamEventType.COMPACTION_END: EventDescriptor.COMPACTION_END,
    StreamEventType.MESSAGE_START: EventDescriptor.MESSAGE_START,
    StreamEventType.MESSAGE_UPDATE: EventDescriptor.ASSISTANT_DELTA,
    StreamEventType.MESSAGE_END: EventDescriptor.ASSISTANT_MESSAGE,
    StreamEventType.RETRY: EventDescriptor.RETRY,
    StreamEventType.ASSISTANT_RESET: EventDescriptor.ASSISTANT_RESET,
    StreamEventType.TOOL_APPROVAL_START: EventDescriptor.APPROVAL_REQUEST,
    StreamEventType.TOOL_APPROVAL_END: EventDescriptor.APPROVAL_END,
    StreamEventType.TOOL_EXECUTION_START: EventDescriptor.TOOL_START,
    StreamEventType.TOOL_EXECUTION_UPDATE: EventDescriptor.TOOL_OUTPUT,
    StreamEventType.TOOL_EXECUTION_END: EventDescriptor.TOOL_END,
    StreamEventType.AGENT_NOTIFICATION: EventDescriptor.SUB_AGENT_RECEIPT,
    StreamEventType.CHILD_MESSAGE_DELIVERED: EventDescriptor.CHILD_MESSAGE_DELIVERED,
    StreamEventType.ACCOUNT_LIMIT: EventDescriptor.ACCOUNT_LIMIT,
    StreamEventType.ACCOUNT_STATE_UNKNOWN: EventDescriptor.ACCOUNT_STATE_UNKNOWN,
    StreamEventType.TURN_END: EventDescriptor.TURN_END,
    StreamEventType.AGENT_END: EventDescriptor.AGENT_END,
    StreamEventType.ERROR: EventDescriptor.ERROR,
}

WIRE_EVENT_NAMES = frozenset(event.value for event in EventDescriptor)
