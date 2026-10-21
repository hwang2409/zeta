"""Exhaustive runtime-to-wire event ownership for zeta serve."""

from ..protocol.types import StreamEventType

NOTIFICATION_WIRE_EVENTS = {
    "agent_completion": "sub_agent_receipt",
    "child_idle": "child_idle",
    "task_exited": "task_exit_notification",
    "project_inbox": "project_inbox_notification",
    "child_question": "child_question",
    "child_question_withdrawn": "child_question_withdrawn",
}

# Custom projections in server.py still use this exhaustive contract.
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
    StreamEventType.TURN_END: "turn_end",
    StreamEventType.AGENT_END: "agent_end",
    StreamEventType.ERROR: "error",
}
