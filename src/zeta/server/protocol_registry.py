"""Registered descriptors for every zeta serve error and event."""

from __future__ import annotations

from enum import Enum, StrEnum


class ErrorDescriptor(Enum):
    """One documented JSON-RPC error variant."""

    PARSE_ERROR = (-32700, None, "invalid JSON or invalid UTF-8 frame")
    INVALID_REQUEST = (-32600, None, "invalid JSON-RPC request or duplicate handshake")
    METHOD_NOT_FOUND = (-32601, None, "method or negotiated feature is not supported")
    INVALID_PARAMS = (
        -32602,
        None,
        "params have the wrong shape, value, or negotiated capability",
    )
    INTERNAL_ERROR = (-32000, None, "unexpected server or storage failure")
    CONNECTION_CAPACITY = (
        -32001,
        None,
        "requested controller or observer connection role is at capacity",
    )
    PROTOCOL_MISMATCH = (
        -32002,
        None,
        "handshake is missing or the protocol version does not match",
    )
    NO_ACTIVE_SESSION = (
        -32003,
        None,
        "no active session, or the active session requires a host binding",
    )
    OPERATION_BUSY = (
        -32004,
        None,
        "operation is blocked by active turn, tool, or background-agent work",
    )
    STATE_UNAVAILABLE = (-32005, None, "requested turn or account state is unavailable")
    APPROVAL_UNAVAILABLE = (
        -32006,
        None,
        "approval request is missing, resolved, or reserved",
    )
    FRAME_TOO_LARGE = (-32007, None, "outbound frame exceeds the size limit")
    REQUEST_FORBIDDEN = (
        -32008,
        None,
        "request is not allowed for this connection or account state",
    )
    HISTORY_SNAPSHOT_STALE = (
        -32009,
        "stale_history_snapshot",
        "session history snapshot is stale",
    )
    DELIVERY_ID_EVICTED = (-32010, None, "delivery ID details were evicted")
    OUTPUT_OVERFLOW = (-32011, None, "connection output queue overflowed")
    CONTENT_NOT_FOUND = (-32012, "content_not_found", "content handle was not found")
    ACTIVE_SESSION = (-32005, "active_session", "the active session cannot be deleted")
    ACCOUNT_LOCK_TIMEOUT = (
        -32008,
        "account_lock_timeout",
        "another process holds the selected account lock",
    )
    APPROVAL_PERSISTENCE_FAILED = (
        -32000,
        "approval_persistence_failed",
        "approval state could not be persisted",
    )
    HOST_BINDING_REQUIRED = (
        -32003,
        "host_binding_required",
        "the session requires a host binding",
    )
    INVALID_CONTENT_READ = (
        -32602,
        "invalid_content_read",
        "content_read parameters are invalid",
    )
    LOGIN_IN_PROGRESS = (-32005, "login_in_progress", "a login is already pending")
    OBSERVER_READ_ONLY = (
        -32008,
        "observer_read_only",
        "a read-only observer requested a write method",
    )
    PROJECT_NOT_FOUND = (
        -32602,
        "project_not_found",
        "the requested project does not exist",
    )
    PROJECT_REGISTRY_BUSY = (
        -32000,
        "project_registry_busy",
        "another process holds the project registry lock",
    )
    SESSION_NOT_FOUND = (
        -32602,
        "session_not_found",
        "the requested session does not exist",
    )

    def __init__(self, code: int, data_code: str | None, meaning: str) -> None:
        self.code = code
        self.data_code = data_code
        self.meaning = meaning

    def data(self, fields: dict[str, object] | None = None) -> dict[str, object] | None:
        if self.data_code is None:
            return fields
        return {**(fields or {}), "code": self.data_code}


class EventDescriptor(StrEnum):
    """One documented JSON-RPC event name."""

    AGENT_START = "agent_start"
    TURN_START = "turn_start"
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"
    MESSAGE_START = "message_start"
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_MESSAGE = "assistant_message"
    RETRY = "retry"
    ASSISTANT_RESET = "assistant_reset"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_END = "approval_end"
    TOOL_START = "tool_start"
    TOOL_OUTPUT = "tool_output"
    TOOL_END = "tool_end"
    SUB_AGENT_RECEIPT = "sub_agent_receipt"
    CHILD_IDLE = "child_idle"
    TASK_EXIT_NOTIFICATION = "task_exit_notification"
    PROJECT_INBOX_NOTIFICATION = "project_inbox_notification"
    CHILD_QUESTION = "child_question"
    CHILD_QUESTION_WITHDRAWN = "child_question_withdrawn"
    CHILD_MESSAGE_DELIVERED = "child_message_delivered"
    ACCOUNT_LIMIT = "account_limit"
    ACCOUNT_STATE_UNKNOWN = "account_state_unknown"
    TURN_END = "turn_end"
    AGENT_END = "agent_end"
    ERROR = "error"
    DELIVERY_STATUS = "delivery_status"
    MEMORY_UPDATED = "memory_updated"
    TURN_ABORTED = "turn_aborted"
    USAGE = "usage"
    USER_MESSAGE = "user_message"
