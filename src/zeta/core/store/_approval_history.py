"""Durable parent-side history for delegated approval lifecycle events."""

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
    ToolCall,
)
from ..checkpoints import ConversationEntry
from ._approval_display import validated_approval_display
from ._async_writes import _AppendApprovalHistoryCommand
from ._client_delivery import CLIENT_DELIVERY_METADATA
from ._incremental_validation import APPROVAL_RESOLUTION_REASONS

if TYPE_CHECKING:
    from ._store import ConversationStore


def validate_approval_history_data(data: Mapping[str, Any]) -> None:
    event = data.get("event")
    if event not in {"approval_request", "approval_end"}:
        raise ValueError("invalid approval history event")
    request_id = data.get("request_id")
    if type(request_id) is not str or not request_id:
        raise ValueError("approval history id must be a nonempty string")
    tool_call = data.get("tool_call")
    if type(tool_call) is not dict:
        raise ValueError("approval history tool_call must be an object")
    ToolCall.from_dict(tool_call)
    if event == "approval_request":
        if data.get("delegated") is not True:
            raise ValueError("approval history request must be delegated")
        child_id = data.get("agent_instance_id")
        if type(child_id) is not str or not child_id:
            raise ValueError("approval history request must identify its child")
        if "approval_display" in data:
            validated_approval_display(data["approval_display"])
        return
    end_data = data.get("data")
    if type(end_data) is not dict:
        raise ValueError("approval history end data must be an object")
    decision = end_data.get("decision")
    if decision not in {"allow", "deny", "abort"}:
        raise ValueError("invalid approval history decision")
    reason = end_data.get("reason")
    if reason is not None and (
        type(reason) is not str
        or decision != "abort"
        or reason not in APPROVAL_RESOLUTION_REASONS
    ):
        raise ValueError("invalid approval history reason")
    if data.get("reason") != reason and ("reason" in data or reason is not None):
        raise ValueError("approval history reason must match end data")


class ApprovalHistoryStoreMixin:
    """Persist delegated wire identity and retrieve durable resolutions."""

    def approval_states_snapshot(self) -> dict[str, tuple[ToolCall, str | None]]:
        """Read approval state from the owner-installed indexes."""
        return self._approval_states_from_maps(
            self._active_approval_requests,
            self._active_approval_resolutions,
        )

    @staticmethod
    def _approval_states_from_maps(
        requests: Mapping[str, ConversationEntry],
        resolutions: Mapping[str, str],
    ) -> dict[str, tuple[ToolCall, str | None]]:
        states: dict[str, tuple[ToolCall, str | None]] = {}
        for request_id, entry in requests.items():
            request = next(
                request
                for request in entry.data.get("approval_requests", [])
                if request["request_id"] == request_id
            )
            states[request_id] = (
                ToolCall.from_dict(request["tool_call"]),
                resolutions.get(request_id),
            )
        return states

    def append_approval_history(
        self, payload: Mapping[str, Any]
    ) -> ConversationEntry:
        with self._append_lock():
            entry = self._append_row_unlocked("approval_history", dict(payload))
            return self._snapshot_entry(entry)

    async def append_approval_history_async(
        self, payload: Mapping[str, Any]
    ) -> ConversationEntry:
        """Append delegated history through the store's durable writer."""
        return await self._commit_storage_command(
            _AppendApprovalHistoryCommand(payload=payload)
        )

    def approval_resolution_data(self, request_id: str) -> dict[str, str] | None:
        """Read a published approval resolution without touching the log."""
        if request_id not in self._active_approval_resolutions:
            return None
        for entry in reversed(self._entries):
            if (
                entry.type == "approval_resolution"
                and entry.data.get("request_id") == request_id
            ):
                return dict(entry.data)
        return None


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
            indexes = range(max(key.seq, 0), len(entries))
        else:
            indexes = range(len(entries) - 1, floor - 2, -1)
        for index in indexes:
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
        """Count installs that replaced resident rows instead of extending them.

        The count can over-count: an async commit that installs after a pure
        external append also counts. Treat a change as "rows may be renumbered,
        rebuild", never as proof; use ``row_id_at`` for an exact check.
        """
        self._assert_owner_thread()
        return self._reset_generation
