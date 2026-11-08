"""Key-shaped lookup of recent resident rows for serve replay.

The store owns these row shapes so readers never scan ``_entries`` themselves.
p08 must keep the same names behind a bounded resident key index; these are
synchronous owner-loop calls, so they may not query SQLite.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...protocol.types import (
    ASSISTANT_RESPONSE_ABORTED,
    ASSISTANT_RESPONSE_COMPLETED,
    ASSISTANT_RESPONSE_STATE,
    MESSAGE_ORIGIN_METADATA,
    MessageOrigin,
)
from ._client_delivery import CLIENT_DELIVERY_METADATA

if TYPE_CHECKING:
    from ._store import ConversationStore

# The newest rows a lookup can see; older rows are unknown, not absent.
RECENT_ROW_WINDOW = 1024


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The ``message`` row carrying ``tool_result.tool_call_id``."""

    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """The ``message`` row listing ``request_id`` in ``approval_requests``."""

    request_id: str


@dataclass(frozen=True, slots=True)
class ApprovalResolution:
    """The ``approval_resolution`` row for a store request id."""

    request_id: str


@dataclass(frozen=True, slots=True)
class ApprovalHistory:
    """The delegated ``approval_history`` row; request and end share the wire id."""

    event: Literal["approval_request", "approval_end"]
    wire_request_id: str


@dataclass(frozen=True, slots=True)
class DeliveredMessage:
    """The ``message`` row tagged with a client ``delivery_id``."""

    delivery_id: str


@dataclass(frozen=True, slots=True)
class DroppedSteer:
    """The ``client_delivery`` row that dropped a queued steer."""

    delivery_id: str


@dataclass(frozen=True, slots=True)
class AssistantResponseAfter:
    """The first completed or aborted assistant reply after ``seq``."""

    seq: int


@dataclass(frozen=True, slots=True)
class UserTurnMessageAfter:
    """The first user-authored message after ``seq``."""

    seq: int


type RecentRowKey = (
    ToolResult
    | ApprovalRequest
    | ApprovalResolution
    | ApprovalHistory
    | DeliveredMessage
    | DroppedSteer
    | AssistantResponseAfter
    | UserTurnMessageAfter
)


def _metadata(message: Mapping[str, Any]) -> Mapping[str, Any]:
    return message.get("metadata") or {}


def _predicate(key: RecentRowKey) -> tuple[str, Callable[[dict[str, Any]], bool]]:
    """Return the row type and a raw ``entry.data`` test; never decodes a Message."""
    match key:
        case ToolResult(tool_call_id):
            return "message", lambda data: (
                (data["message"].get("tool_result") or {}).get("tool_call_id")
                == tool_call_id
            )
        case ApprovalResolution(request_id):
            return "approval_resolution", lambda data: (
                data.get("request_id") == request_id
            )
        case ApprovalHistory(event, wire_request_id):
            return "approval_history", lambda data: (
                data.get("request_id") == wire_request_id and data.get("event") == event
            )
        case DeliveredMessage(delivery_id):
            return "message", lambda data: (
                _metadata(data["message"]).get(CLIENT_DELIVERY_METADATA) == delivery_id
            )
        case DroppedSteer(delivery_id):
            return "client_delivery", lambda data: (
                data.get("delivery_id") == delivery_id
                and data.get("status") == "dropped"
            )
        case AssistantResponseAfter():
            return "message", _is_assistant_response
        case UserTurnMessageAfter():
            return "message", lambda data: (
                data["message"].get("role") == "user"
                and _metadata(data["message"]).get(MESSAGE_ORIGIN_METADATA)
                == MessageOrigin.USER.value
            )
    raise TypeError(f"unknown recent row key: {key!r}")


def _is_assistant_response(data: dict[str, Any]) -> bool:
    message = data["message"]
    if message.get("role") != "assistant":
        return False
    metadata = _metadata(message)
    return metadata.get(ASSISTANT_RESPONSE_STATE) in {
        ASSISTANT_RESPONSE_COMPLETED,
        ASSISTANT_RESPONSE_ABORTED,
    } and not metadata.get("compaction_summary")


class RecentRowsMixin:
    """Owner-thread, I/O-free reads over the newest resident rows."""

    _reset_generation = 0

    def recent_row_floor(self: ConversationStore) -> int:
        """Return the lowest seq a lookup can see."""
        self._assert_owner_thread()
        return max(1, len(self._entries) - RECENT_ROW_WINDOW + 1)

    def recent_row_seq(self: ConversationStore, key: RecentRowKey) -> int | None:
        """Return the seq of the row ``key`` names, or None when not visible.

        Most keys name the newest match. ``*After`` keys name the first match
        after their seq, and return None when rows after it are below the floor.
        """
        floor = self.recent_row_floor()
        if type(key) is ApprovalRequest:
            entry = self._active_approval_requests.get(key.request_id)
            return None if entry is None else entry.seq
        entries = self._entries
        row_type, test = _predicate(key)
        if type(key) in (AssistantResponseAfter, UserTurnMessageAfter):
            if key.seq + 1 < floor:
                return None
            # seq == index + 1, so index key.seq is the first row after key.seq.
            for index in range(max(key.seq, 0), len(entries)):
                entry = entries[index]
                if entry.type == row_type and test(entry.data):
                    return entry.seq
            return None
        for index in range(len(entries) - 1, floor - 2, -1):
            entry = entries[index]
            if entry.type == row_type and test(entry.data):
                return entry.seq
        return None

    def row_id_at(self: ConversationStore, seq: int) -> str | None:
        """Return the stable entry id at ``seq`` while it is visible."""
        floor = self.recent_row_floor()
        if not floor <= seq <= len(self._entries):
            return None
        return self._entries[seq - 1].id

    def reset_generation(self: ConversationStore) -> int:
        """Count installs that replaced resident rows instead of extending them."""
        self._assert_owner_thread()
        return self._reset_generation
