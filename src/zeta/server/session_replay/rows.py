"""The replay route table: which history row each published serve event presents.

A replay frame's ``covers`` is the ``seq`` of the history row that represents the
same effect, or ``None`` when no row does. A stream drops a frame whose row is
already inside the history it joined with, so each effect reaches it once.

``ROUTES`` is the single source for ``covers``. Each published event kind has one
entry per variant; ``_VARIANTS`` picks the variant from the frame's fields and the
publisher's ``RowFacts``. A source is one of:

- a store key builder: the store finds the newest resident row for that key.
- ``CARRIED``: the publisher holds its committed row and passes ``RowFacts.seq``.
- ``LIVE``: the frame has no history row.
- ``PublishFirst(key)``: the row installs after publication (assistant_message,
  the user echo without a delivery id). The publisher passes ``RowFacts.seq`` =
  the newest resident seq at publication; replay copies wait until ``key`` finds
  the first matching row after it (see ``ReplayOutput.defer_replay``).

A kind or variant with no entry raises ``UnroutedEvent``; serve fails closed.

Scope rule: lookups use the scope's own store, and an event whose
``data.agent_instance_id`` differs from the scope's owner (``None`` for the top
session) is a child event with no row in that store. Child scopes reuse this
module with their store and owner.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...core.store import (
    ApprovalHistory,
    ApprovalRequest,
    ApprovalResolution,
    AssistantResponseAfter,
    DeliveredMessage,
    RecentRowKey,
    ToolResult,
    UserTurnMessageAfter,
)
from ...protocol.types import MessageOrigin
from ..protocol import EventDescriptor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RowFacts:
    """What a publisher knows about its frame's row; never sent on the wire."""

    variant: str | None = None
    seq: int | None = None
    delivery_id: str | None = None
    # the origin a send without delivery id passed to start_turn
    origin: MessageOrigin | None = None


# a publisher whose history write failed: the effect is live-only
WRITE_FAILED = RowFacts(variant="write_failed")
# a child-forwarded approval end; its row is the top store's approval_history
DELEGATED = RowFacts(variant="delegated")


@dataclass(frozen=True, slots=True)
class OldRow:
    """The row is older than the store's lookup window; every row <= bound is."""

    bound: int


Covers = int | OldRow | None


class _Source:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


CARRIED = _Source("CARRIED")
LIVE = _Source("LIVE")
KeyBuilder = Callable[[Mapping[str, Any], RowFacts], RecentRowKey]


class PublishFirst:
    """A route whose row installs after publication; ``key`` finds it later."""

    def __init__(self, key: Callable[[RowFacts], AssistantResponseAfter | UserTurnMessageAfter]) -> None:
        self.key = key

    def __repr__(self) -> str:
        return "PUBLISH_FIRST"


def _mark(facts: RowFacts) -> int:
    assert facts.seq is not None, "a publish-first publisher passes its mark"
    return facts.seq


def _user_turn(facts: RowFacts) -> UserTurnMessageAfter:
    assert facts.origin is not None, "a send without delivery id passes its origin"
    return UserTurnMessageAfter(_mark(facts), facts.origin)


Source = _Source | PublishFirst | KeyBuilder


def _data(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    data = fields.get("data")
    return data if isinstance(data, Mapping) else {}


def _child(fields: Mapping[str, Any], owner: str | None) -> bool:
    return _data(fields).get("agent_instance_id") != owner


def _tool_call_id(fields: Mapping[str, Any]) -> str:
    result = fields.get("tool_result")
    return str(result.get("tool_call_id")) if isinstance(result, Mapping) else ""


Classifier = Callable[[Mapping[str, Any], RowFacts, "str | None"], str]


def _user_message(fields: Mapping[str, Any], facts: RowFacts, owner: str | None) -> str:
    if fields.get("mode") == "steer":
        return "steer"
    return "delivered" if facts.delivery_id is not None else "undelivered"


def _tool_end(fields: Mapping[str, Any], facts: RowFacts, owner: str | None) -> str:
    if _child(fields, owner):
        return "child"
    return "background_terminal" if "notification_id" in _data(fields) else "foreground"


def _approval_request(fields: Mapping[str, Any], facts: RowFacts, owner: str | None) -> str:
    return "delegated" if fields.get("delegated") is True else "foreground"


def _approval_end(fields: Mapping[str, Any], facts: RowFacts, owner: str | None) -> str:
    return facts.variant or "foreground"


def _notification(fields: Mapping[str, Any], facts: RowFacts, owner: str | None) -> str:
    return "durable" if "notification_id" in _data(fields) else "display"


_NOTIFICATION_EVENTS = (
    EventDescriptor.SUB_AGENT_RECEIPT,
    EventDescriptor.CHILD_IDLE,
    EventDescriptor.TASK_EXIT_NOTIFICATION,
    EventDescriptor.PROJECT_INBOX_NOTIFICATION,
    EventDescriptor.CHILD_QUESTION,
    EventDescriptor.CHILD_QUESTION_WITHDRAWN,
)
_VARIANTS: dict[EventDescriptor, Classifier] = {
    EventDescriptor.USER_MESSAGE: _user_message,
    EventDescriptor.TOOL_END: _tool_end,
    EventDescriptor.APPROVAL_REQUEST: _approval_request,
    EventDescriptor.APPROVAL_END: _approval_end,
    **{event: _notification for event in _NOTIFICATION_EVENTS},
}

# events serve never publishes on a replay stream
NOT_PUBLISHED = frozenset(
    {
        EventDescriptor.SERVED_TURN_END,
        EventDescriptor.REPLAY_STREAM_ENDED,
        EventDescriptor.CHILD_HOLD,
    }
)
_LIVE_KINDS = (
    EventDescriptor.AGENT_START,
    EventDescriptor.TURN_START,
    EventDescriptor.COMPACTION_START,
    EventDescriptor.COMPACTION_END,
    EventDescriptor.MESSAGE_START,
    EventDescriptor.ASSISTANT_DELTA,
    EventDescriptor.RETRY,
    EventDescriptor.ASSISTANT_RESET,
    EventDescriptor.TOOL_START,
    EventDescriptor.TOOL_OUTPUT,
    EventDescriptor.CHILD_MESSAGE_DELIVERED,
    EventDescriptor.ACCOUNT_LIMIT,
    EventDescriptor.ACCOUNT_STATE_UNKNOWN,
    EventDescriptor.TURN_END,
    EventDescriptor.AGENT_END,
    EventDescriptor.ERROR,
    EventDescriptor.MEMORY_UPDATED,
    EventDescriptor.TURN_ABORTED,
    EventDescriptor.USAGE,
    EventDescriptor.SESSION_GRANTS_CHANGED,
    EventDescriptor.CONFIG_CHANGED,
    EventDescriptor.BACKGROUND_TASK_UPDATED,
    EventDescriptor.TODO_UPDATED,
)

ROUTES: dict[tuple[EventDescriptor, str], Source] = {
    **{(event, ""): LIVE for event in _LIVE_KINDS},
    (EventDescriptor.USER_MESSAGE, "delivered"): lambda fields, facts: DeliveredMessage(
        str(facts.delivery_id)
    ),
    # known gap G1, deferred to slice S3 (not settled): the queued steer row is not
    # in history, but its delivered/dropped client_delivery row is, with the steer
    # text (history_projection.py `_history_entry_rows`), and has no live event. A
    # join whose history read spans the delivery sees the steer twice.
    (EventDescriptor.USER_MESSAGE, "steer"): LIVE,
    (EventDescriptor.USER_MESSAGE, "undelivered"): PublishFirst(_user_turn),
    (EventDescriptor.ASSISTANT_MESSAGE, ""): PublishFirst(
        lambda facts: AssistantResponseAfter(_mark(facts))
    ),
    (EventDescriptor.TOOL_END, "foreground"): lambda fields, facts: ToolResult(
        _tool_call_id(fields)
    ),
    # a background child's terminal card presents its completion notification row
    (EventDescriptor.TOOL_END, "background_terminal"): CARRIED,
    (EventDescriptor.TOOL_END, "child"): LIVE,
    (EventDescriptor.APPROVAL_REQUEST, "foreground"): lambda fields, facts: ApprovalRequest(
        str(fields.get("request_id"))
    ),
    (EventDescriptor.APPROVAL_REQUEST, "delegated"): lambda fields, facts: ApprovalHistory(
        "approval_request", str(fields.get("request_id"))
    ),
    (EventDescriptor.APPROVAL_END, "foreground"): lambda fields, facts: ApprovalResolution(
        str(fields.get("request_id"))
    ),
    (EventDescriptor.APPROVAL_END, "delegated"): lambda fields, facts: ApprovalHistory(
        "approval_end", str(fields.get("request_id"))
    ),
    (EventDescriptor.APPROVAL_END, "write_failed"): LIVE,
    **{(event, "durable"): CARRIED for event in _NOTIFICATION_EVENTS},
    **{(event, "display"): LIVE for event in _NOTIFICATION_EVENTS},
    (EventDescriptor.DELIVERY_STATUS, ""): CARRIED,
}


# events with a publish-first variant: routed even before any stream exists
PUBLISH_FIRST_EVENTS = frozenset(
    event for (event, _variant), source in ROUTES.items() if isinstance(source, PublishFirst)
)


class UnroutedEvent(LookupError):
    """A published replay event kind or variant has no route table entry."""


def variant(
    event: EventDescriptor,
    fields: Mapping[str, Any],
    facts: RowFacts,
    owner: str | None = None,
) -> str:
    classify = _VARIANTS.get(event)
    return "" if classify is None else classify(fields, facts, owner)


def route(
    event: EventDescriptor,
    fields: Mapping[str, Any],
    facts: RowFacts,
    owner: str | None = None,
) -> Source:
    variant_name = variant(event, fields, facts, owner)
    try:
        return ROUTES[(event, variant_name)]
    except KeyError:
        raise UnroutedEvent(f"{event}/{variant_name or '-'}") from None


def covers(
    event: EventDescriptor,
    source: Source,
    store: Any,
    fields: Mapping[str, Any],
    facts: RowFacts,
) -> Covers:
    """Return the seq of the frame's history row; O(1) or a bounded store lookup."""
    if source is LIVE or isinstance(source, PublishFirst):
        return None
    if source is CARRIED:
        if facts.seq is None:
            logger.warning("replay row seq missing for %s; sent as live-only", event.value)
        return facts.seq
    assert callable(source)
    seq = store.recent_row_seq(source(fields, facts))
    if seq is not None:
        return seq
    floor = store.recent_row_floor()
    if floor <= 1:
        # the window spans the whole log, so the row is not there: a route bug
        logger.warning("replay row missing for %s; sent as live-only", event.value)
        return None
    return OldRow(floor - 1)
