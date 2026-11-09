"""The shared registry for conversation entry payload validation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ...protocol.types import (
    Message,
    ToolCall,
    ToolUseContent,
    is_passive_harness_message,
)
from ...storage_format import reject_unknown_keys


class EntryValidationContext(Protocol):
    session_id: str

    def _validate_checkpoint_or_fork_payload(self, entry: Any) -> None: ...


EntryValidator = Callable[[EntryValidationContext, Any], None]
_SENT_STATUS_EVENT = "project_inbox_sent_status"
_MESSAGE_ID = re.compile(r"[0-9a-f]{32}\Z")


def sent_message_receipts(entry: Any) -> tuple[tuple[str, str], ...]:
    """Return validated sent-status receipts from one physical entry."""

    if entry.type != "message":
        return ()
    raw = entry.data.get("message")
    if not isinstance(raw, dict) or not is_passive_harness_message(raw):
        return ()
    try:
        message = Message.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return ()
    if message.metadata.get("zeta_event") != _SENT_STATUS_EVENT:
        return ()
    raw_statuses = message.metadata.get("sent_statuses")
    if not isinstance(raw_statuses, list):
        return ()
    receipts: list[tuple[str, str]] = []
    for item in raw_statuses:
        if not isinstance(item, dict):
            continue
        message_id = item.get("message_id")
        status = item.get("status")
        if (
            isinstance(message_id, str)
            and _MESSAGE_ID.fullmatch(message_id)
            and status in {"claimed", "done"}
        ):
            receipts.append((message_id, status))
    return tuple(receipts)


@dataclass(frozen=True, slots=True)
class ConversationEntrySpec:
    validator: EntryValidator


def _message(_context: EntryValidationContext, entry: Any) -> None:
    message = entry.data.get("message")
    if type(message) is not dict:
        raise ValueError("message entry payload must contain an object")
    parsed_message = Message.from_dict(message)
    _message_approval_requests(parsed_message, entry.data.get("approval_requests", []))


def _validated_approval_requests(
    approval_requests: object,
) -> list[ToolCall]:
    from ..store._approval_display import validated_approval_display

    if type(approval_requests) is not list:
        raise ValueError("message approval_requests must be an array")
    request_ids: set[str] = set()
    tool_calls: list[ToolCall] = []
    for request in approval_requests:
        if type(request) is not dict:
            raise ValueError("message approval request must be an object")
        request_id = request.get("request_id")
        tool_call = request.get("tool_call")
        if type(request_id) is not str or not request_id:
            raise ValueError("approval request id must be a nonempty string")
        if request_id in request_ids:
            raise ValueError(f"duplicate approval request: {request_id}")
        request_ids.add(request_id)
        if type(tool_call) is not dict:
            raise ValueError("approval request tool_call must be an object")
        tool_calls.append(ToolCall.from_dict(tool_call))
        if "approval_display" in request:
            validated_approval_display(request["approval_display"])
    return tool_calls


def _message_approval_requests(
    parsed_message: Message, approval_requests: object
) -> None:
    anchored: dict[str, ToolCall] = {}
    for block in parsed_message.content:
        if isinstance(block, ToolUseContent):
            anchored.setdefault(block.tool_call.id, block.tool_call)
    for tool_call in _validated_approval_requests(approval_requests):
        if anchored.get(tool_call.id) != tool_call:
            raise ValueError("approval request must match an anchored tool call")


def _child_run_started(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._child_runs import validate_child_run_started_data

    validate_child_run_started_data(entry.data)


def _compaction(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._validation import validate_compaction_data

    validate_compaction_data(entry.data)


def _client_delivery(context: EntryValidationContext, entry: Any) -> None:
    from ..store._client_delivery import validate_client_delivery_data

    validate_client_delivery_data(entry.data, context.session_id)


def _warning(_context: EntryValidationContext, entry: Any) -> None:
    if type(entry.data.get("message")) is not str:
        raise ValueError("warning message must be a string")


def _checkpoint_or_fork(context: EntryValidationContext, entry: Any) -> None:
    context._validate_checkpoint_or_fork_payload(entry)


def _approval_resolution(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._incremental_validation import validate_approval_reason

    request_id = entry.data.get("request_id")
    decision = entry.data.get("decision")
    if type(request_id) is not str or not request_id:
        raise ValueError("approval resolution id must be a nonempty string")
    if decision not in {"allow", "deny", "abort"}:
        raise ValueError("invalid approval resolution")
    validate_approval_reason(decision, entry.data.get("reason"))


def _approval_history(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._approval_history import validate_approval_history_data

    validate_approval_history_data(entry.data)


def _turn_end(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._turn_history import validate_turn_end_data

    validate_turn_end_data(entry.data)


def _notification(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._validation import (
        AGENT_COMPLETION_NOTIFICATION_KIND,
        NOTIFICATION_KINDS,
        validate_notification_data,
    )

    kind = entry.data.get("kind", AGENT_COMPLETION_NOTIFICATION_KIND)
    if kind in NOTIFICATION_KINDS:
        validate_notification_data(entry.data)


def _notification_ack(_context: EntryValidationContext, entry: Any) -> None:
    notification_id = entry.data.get("notification_id")
    if type(notification_id) is not str or not notification_id:
        raise ValueError("notification id must be a nonempty string")


def _notification_tui_presented(_context: EntryValidationContext, entry: Any) -> None:
    _notification_ack(_context, entry)
    reject_unknown_keys(entry.data, "notification_marker")


def _pending_prompt(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._pending_prompts import MAX_PENDING_PROMPT_TEXT

    text = entry.data.get("text")
    if type(text) is not str or not text:
        raise ValueError("pending prompt text must be a nonempty string")
    if len(text) > MAX_PENDING_PROMPT_TEXT:
        raise ValueError("pending prompt text is too long")


def _pending_prompt_ack(_context: EntryValidationContext, entry: Any) -> None:
    prompt_id = entry.data.get("prompt_id")
    if type(prompt_id) is not str or not prompt_id:
        raise ValueError("pending prompt id must be a nonempty string")


def _pending_queue_closed(_context: EntryValidationContext, entry: Any) -> None:
    reject_unknown_keys(entry.data, "pending_queue_closed")


def _terminal_transition(_context: EntryValidationContext, entry: Any) -> None:
    from ..store._validation import validate_terminal_transition_data

    validate_terminal_transition_data(entry.data)


CONVERSATION_ENTRY_SPECS: dict[str, ConversationEntrySpec] = {
    "message": ConversationEntrySpec(_message),
    "child_run_started": ConversationEntrySpec(_child_run_started),
    "compaction": ConversationEntrySpec(_compaction),
    "client_delivery": ConversationEntrySpec(_client_delivery),
    "warning": ConversationEntrySpec(_warning),
    "checkpoint": ConversationEntrySpec(_checkpoint_or_fork),
    "fork": ConversationEntrySpec(_checkpoint_or_fork),
    "approval_resolution": ConversationEntrySpec(_approval_resolution),
    "approval_history": ConversationEntrySpec(_approval_history),
    "turn_end": ConversationEntrySpec(_turn_end),
    "notification": ConversationEntrySpec(_notification),
    "notification_ack": ConversationEntrySpec(_notification_ack),
    "notification_tui_presented": ConversationEntrySpec(_notification_tui_presented),
    "pending_prompt": ConversationEntrySpec(_pending_prompt),
    "pending_prompt_ack": ConversationEntrySpec(_pending_prompt_ack),
    "pending_queue_closed": ConversationEntrySpec(_pending_queue_closed),
    "terminal_transition": ConversationEntrySpec(_terminal_transition),
}


def validate_conversation_entry(context: EntryValidationContext, entry: Any) -> None:
    spec = CONVERSATION_ENTRY_SPECS.get(entry.type)
    if spec is None:
        raise ValueError(f"unsupported conversation entry type: {entry.type}")
    spec.validator(context, entry)
