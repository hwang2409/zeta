"""Render and select durable notifications for one admitted request."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

from ..core.checkpoints import ConversationEntry
from ..protocol.types import Message, MessageRole, TextContent

if TYPE_CHECKING:
    from ..core.store import ConversationStore

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)

def notification_batch_id(
    notifications: Sequence[Mapping[str, JSONValue]], notification_ids: Sequence[str]
) -> str:
    """Return the stable identity for one represented notification batch."""

    payload = {
        "version": 1,
        "notification_ids": list(notification_ids),
        "notifications": list(notifications),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

def _message_notification_ids(message: Message) -> frozenset[str]:
    identity = message.metadata.get("zeta.step_delivery")
    if not isinstance(identity, Mapping):
        return frozenset()
    ids = identity.get("notification_ids")
    if not isinstance(ids, list) or any(type(item) is not str for item in ids):
        return frozenset()
    return frozenset(ids)

def _bounded_text(value: object, limit: int) -> str:
    return str(value or "")[:limit]

def _notification_digest(entry: ConversationEntry) -> dict[str, JSONValue]:
    data = entry.data
    report = data.get("report")
    report_data = report if isinstance(report, Mapping) else {}
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    child_id = _bounded_text(data.get("child_instance_id"), 80)
    child_path = _bounded_text(data.get("child_session_path"), 200)
    result: dict[str, JSONValue] = {
        "notification_id": entry.id[:64],
        "seq": str(entry.seq)[:64],
        "kind": _bounded_text(data.get("kind", "agent_completion"), 32),
        "child_instance_id": child_id,
        "description": _bounded_text(data.get("description"), 120),
        "status": _bounded_text(data.get("status"), 32),
        "report_status": _bounded_text(report_data.get("status"), 32),
        "verdict": _bounded_text(report_data.get("verdict"), 32),
        "head_sha": _bounded_text(report_data.get("head_sha"), 40),
        "summary": _bounded_text(report_data.get("summary", data.get("text", "")), 256),
        "error": data.get("error") is True,
        "canceled": data.get("canceled") is True,
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    if child_id:
        result["full_output"] = f"agent action=output handle={child_id}"
    if child_path:
        result["child_transcript"] = child_path
    pr_number = report_data.get("pr_number", data.get("pr_number"))
    if pr_number is not None:
        result["pr_number"] = _bounded_text(pr_number, 10)
    return result

def _notification_message(
    entries: Sequence[ConversationEntry], *, digest: bool = False
) -> Message:
    payload = [
        _notification_digest(entry)
        if digest
        else {
            "notification_id": entry.id,
            "kind": entry.data.get("kind", "agent_completion"),
            **entry.data,
        }
        for entry in entries
    ]
    ids = [entry.id for entry in entries]
    identity = {
        "version": 1,
        "batch_id": notification_batch_id(payload, ids),
        "notification_ids": ids,
    }
    text = (
        "Background updates only. They do not replace the active user request. "
        "Continue and finish the active user request; use these updates only when "
        "relevant.\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    return Message(
        MessageRole.SYSTEM,
        [TextContent(text)],
        metadata={
            "zeta_event": "agent_notifications",
            "notifications": payload,
            "zeta.step_delivery": identity,
        },
    )

def _select_notification_delivery(
    store: ConversationStore,
    pending: Sequence[ConversationEntry],
    notification_budget: int | None,
    token_counter: Callable[[Sequence[Message]], int] | None,
) -> tuple[list[ConversationEntry], Message | None, bool]:
    persisted = _reuse_persisted_notification_delivery(
        store._active_notification_delivery_message,
        pending,
        notification_budget,
        token_counter,
    )
    if persisted is not None:
        return persisted
    selected, message = _largest_fitting_notification_delivery(
        pending, notification_budget, token_counter
    )
    return selected, message, False


def _reuse_persisted_notification_delivery(
    persisted: Message | None,
    pending: Sequence[ConversationEntry],
    notification_budget: int | None,
    token_counter: Callable[[Sequence[Message]], int] | None,
) -> tuple[list[ConversationEntry], Message, bool] | None:
    if persisted is None or notification_budget == 0:
        return None
    represented = _message_notification_ids(persisted)
    pending_ids = {entry.id for entry in pending}
    if not represented or not represented.issubset(pending_ids):
        return None
    if not _notification_fits(persisted, notification_budget, token_counter):
        return None
    return [entry for entry in pending if entry.id in represented], persisted, True


def _largest_fitting_notification_delivery(
    pending: Sequence[ConversationEntry],
    notification_budget: int | None,
    token_counter: Callable[[Sequence[Message]], int] | None,
) -> tuple[list[ConversationEntry], Message | None]:
    if not pending or notification_budget == 0:
        return [], None
    for selected, message in _ordered_notification_candidates(pending):
        if _notification_fits(message, notification_budget, token_counter):
            return selected, message
    return [], None


def _ordered_notification_candidates(
    pending: Sequence[ConversationEntry],
) -> Iterator[tuple[list[ConversationEntry], Message]]:
    yield list(pending), _notification_message(pending)
    for count in range(len(pending), 0, -1):
        selected = list(pending[:count])
        yield selected, _notification_message(selected, digest=True)


def _notification_fits(
    message: Message,
    notification_budget: int | None,
    token_counter: Callable[[Sequence[Message]], int] | None,
) -> bool:
    if len(json.dumps(message.to_dict()).encode()) > 64 * 1024:
        return False
    return (
        notification_budget is None
        or token_counter is None
        or token_counter((message,)) <= notification_budget
    )
