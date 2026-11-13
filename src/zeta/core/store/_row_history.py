"""Recent-row history lookup keys and owner-thread accessors."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...protocol.types import (
    ASSISTANT_RESPONSE_ABORTED,
    ASSISTANT_RESPONSE_COMPLETED,
    ASSISTANT_RESPONSE_FAILED,
    ASSISTANT_RESPONSE_STATE,
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    assistant_text,
    is_passive_harness_message,
)
from ...storage_format import reject_unknown_keys
from ._async_writes import _AppendTurnEndCommand
from ._client_delivery import CLIENT_DELIVERY_METADATA

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
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
class AssistantResponseAfter:
    """The first completed, aborted, or failed assistant reply after ``seq``."""

    seq: int


@dataclass(frozen=True, slots=True)
class UserTurnMessageAfter:
    """The first user-role message with ``origin`` after ``seq``."""

    seq: int
    origin: MessageOrigin


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
        case UserTurnMessageAfter(_seq, origin):
            return "message", lambda data: (
                data["message"].get("role") == "user"
                and _metadata(data["message"]).get(MESSAGE_ORIGIN_METADATA)
                == origin.value
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
        ASSISTANT_RESPONSE_FAILED,
    } and not metadata.get("compaction_summary")


TurnEndReason = Literal["completed", "user_stop", "error", "disconnect"]
_TURN_END_REASONS = {"completed", "user_stop", "error", "disconnect"}
_TURN_ID = re.compile(r"[0-9a-f]{32}").fullmatch


def validate_turn_end_data(data: Mapping[str, object]) -> None:
    reject_unknown_keys(data, "turn_history")
    turn_id = data.get("turn_id")
    if (
        not isinstance(turn_id, str)
        or _TURN_ID(turn_id) is None
        or data.get("reason") not in _TURN_END_REASONS
    ):
        raise ValueError("invalid turn end")


class TurnHistoryMixin:
    def last_turn_end(self: ConversationStore) -> ConversationEntry | None:
        entry = self._delivery_fold.last_turn_end
        return self._snapshot_entry(entry) if entry is not None else None

    async def append_turn_end_async(
        self: ConversationStore, turn_id: str, reason: TurnEndReason
    ) -> None:
        data = {"turn_id": turn_id, "reason": reason}
        validate_turn_end_data(data)
        await self._commit_storage_command(_AppendTurnEndCommand(data=data))


class RecentRowsMixin:
    """Owner-thread, I/O-free reads over the newest resident rows."""

    _reset_generation = 0

    def recent_row_floor(self: ConversationStore) -> int:
        """Return the lowest seq a lookup can see."""
        self._assert_owner_thread()
        return max(1, len(self._resident_rows.entries) - RECENT_ROW_WINDOW + 1)

    def recent_row_seq(self: ConversationStore, key: RecentRowKey) -> int | None:
        """Return the seq of the row ``key`` names, or None when not visible.

        Most keys name the newest match. ``*After`` keys name the first match
        after their seq, and return None when rows after it are below the floor.
        """
        floor = self.recent_row_floor()
        if type(key) is ApprovalRequest:
            for entry in reversed(self._resident_rows.entries[floor - 1 :]):
                if entry.type != "message":
                    continue
                if any(
                    request.get("request_id") == key.request_id
                    for request in entry.data.get("approval_requests", ())
                ):
                    return entry.seq
            return None
        entries = self._resident_rows.entries
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
        if not floor <= seq <= len(self._resident_rows.entries):
            return None
        return self._resident_rows.entries[seq - 1].id

    def reset_generation(self: ConversationStore) -> int:
        """Count installs that replaced resident rows instead of extending them.

        The count can over-count: an async commit that installs after a pure
        external append also counts. Treat a change as "rows may be renumbered,
        rebuild", never as proof; use ``row_id_at`` for an exact check.
        """
        self._assert_owner_thread()
        return self._reset_generation


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS message_search ("
        "entry_seq INTEGER NOT NULL, entry_id TEXT NOT NULL, "
        "unit_ordinal INTEGER NOT NULL, text TEXT NOT NULL, "
        "PRIMARY KEY(entry_seq,unit_ordinal))"
    )


def searchable_units(
    entry_type: str, data: Mapping[str, object]
) -> tuple[str, ...]:
    """Return the visible search text for each unit painted by root replay."""

    if entry_type == "message":
        try:
            message = Message.from_dict(data["message"])
        except (KeyError, TypeError, ValueError):
            return ()
        if is_passive_harness_message(message):
            text = assistant_text(message)
            return (text,) if text else ()
        if message.role is MessageRole.USER:
            text = assistant_text(message)
            return (text,) if text else ()
        if message.role is MessageRole.ASSISTANT:
            units = [text] if (text := assistant_text(message)) else []
            if message.metadata.get(FAILED_TURN_MARKER):
                error = message.metadata.get(FAILED_TURN_ERROR)
                if isinstance(error, Mapping):
                    units.append(" ".join(str(value) for value in error.values()))
                else:
                    units.append("provider failure")
            return tuple(units)
        if message.role is MessageRole.TOOL_RESULT and message.tool_result is not None:
            return (str(message.tool_result.content),)
        return ()
    if entry_type == "checkpoint":
        return (f"checkpoint '{data.get('label', '')}' at seq",)
    if entry_type == "fork":
        label = data.get("label", data.get("name", ""))
        return (f"forked {label} seq {data.get('from_seq', '')}",)
    if entry_type == "compaction":
        return (
            (
                "compaction marker entries "
                f"{data.get('source_seq_start', '')} {data.get('source_seq_end', '')}"
            ),
        )
    if entry_type == "turn_end":
        reason = data.get("reason")
        text = {
            "user_stop": "turn stopped by the user",
            "error": "turn ended with an error",
            "disconnect": "turn ended when the controller disconnected",
        }.get(reason)
        return (text,) if text is not None else ()
    if entry_type == "warning" and data.get("kind") == "transcript_recovery":
        return (
            (
                "transcript recovered preserved corrupt bytes in quarantine "
                f"{data.get('quarantine_id', '')}"
            ),
        )
    if entry_type == "notification":
        text = "\n".join(
            value
            for key in ("title", "message", "body", "summary", "question")
            if isinstance((value := data.get(key)), str)
        )
        return (text,) if text else ()
    return ()


def write_text(
    connection: sqlite3.Connection,
    seq: int,
    entry_id: str,
    entry_type: str,
    data: Mapping[str, object],
) -> None:
    for unit_ordinal, text in enumerate(searchable_units(entry_type, data)):
        connection.execute(
            "INSERT INTO message_search(entry_seq,entry_id,unit_ordinal,text) "
            "VALUES(?,?,?,?)",
            (seq, entry_id, unit_ordinal, text.casefold()),
        )


def search(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
    cancel: threading.Event | None = None,
) -> tuple[tuple[str, int], ...]:
    needle = query.casefold()
    cursor = connection.execute(
        "SELECT s.entry_id,s.unit_ordinal,s.text FROM message_search s "
        "JOIN entries e ON e.seq=s.entry_seq "
        "WHERE e.tui_group IS NOT NULL "
        "ORDER BY s.entry_seq DESC,s.unit_ordinal DESC"
    )
    matches: list[tuple[str, int]] = []
    while rows := cursor.fetchmany(256):
        if cancel is not None and cancel.is_set():
            break
        for entry_id, unit_ordinal, text in rows:
            if needle in text:
                matches.append((entry_id, unit_ordinal))
                if len(matches) >= limit:
                    return tuple(matches)
    return tuple(matches)


def history_index_suffix_start(
    stored: Mapping[str, object],
    expected: Mapping[str, object],
    rows: Sequence[object],
    *,
    schema_version: int,
    row: Callable[[object], tuple[int, str, str | None, str, str, str]],
    digest_entry: Callable[[bytes, object], bytes],
) -> int | None:
    if stored.get("schema_version") != schema_version or any(
        stored.get(key) != expected.get(key)
        for key in ("st_dev", "st_ino", "repair_generation")
    ):
        return None
    stored_boundary = stored.get("byte_boundary")
    expected_boundary = expected.get("byte_boundary")
    if (
        not isinstance(stored_boundary, int)
        or not isinstance(expected_boundary, int)
        or stored_boundary > expected_boundary
    ):
        return None
    start = stored.get("last_seq")
    if not isinstance(start, int) or not 0 <= start <= len(rows):
        return None
    boundary = rows[start - 1] if start else None
    if stored.get("last_id") != (row(boundary)[1] if boundary else None):
        return None
    if stored.get("last_parent") != (row(boundary)[2] if boundary else None):
        return None
    digest_hex = stored.get("prefix_digest")
    if not isinstance(digest_hex, str):
        return None
    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError:
        return None
    suffix = rows[start:]
    if not suffix:
        return None
    parent_id = stored.get("last_id")
    for entry in suffix:
        projected = row(entry)
        if projected[2] != parent_id:
            return None
        parent_id = projected[1]
        digest = digest_entry(digest, entry)
    return start if digest.hex() == expected.get("prefix_digest") else None


def project_history_entries(
    entries: tuple[object, ...],
    total: int,
    tui_total: int,
    *,
    data: Callable[[object], Mapping[str, object]],
    belongs: Callable[[str, Mapping[str, object]], bool],
    starts: Callable[[str, Mapping[str, object]], bool],
) -> tuple[list[tuple[object, int | None, int]], int, int]:
    projected: list[tuple[object, int | None, int]] = []
    for entry in entries:
        entry_data = data(entry)
        group = total if belongs(entry.type, entry_data) else None
        projected.append((entry, group, tui_total))
        tui_total += 1
        if starts(entry.type, entry_data):
            total += 1
    return projected, total, tui_total
