"""Exhaustive runtime-to-wire event ownership for zeta serve."""

from collections.abc import Mapping
from enum import Enum, auto

from ..protocol.types import StreamEvent, StreamEventType
from .protocol import bounded

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


def tool_start(event: StreamEvent) -> dict[str, object]:
    return {"tool_call": _tool_call(event), "data": dict(event.data)}


def tool_output(event: StreamEvent) -> dict[str, object]:
    return {
        "tool_call": _tool_call(event),
        "output": bounded(event.delta or _data_text(event.data)),
        "data": dict(event.data),
    }


def tool_end(event: StreamEvent) -> dict[str, object]:
    return {
        "tool_call": _tool_call(event),
        "tool_result": event.tool_result.to_dict() if event.tool_result else None,
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
