"""Durable step-delivery transactions owned by the conversation store."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...protocol.types import Message, MessageOrigin, MessageRole, TextContent
from ..checkpoints import ConversationEntry
from ._pending_prompts import PENDING_PROMPT_ID_METADATA
from ._validation import validate_notification_data

if TYPE_CHECKING:
    from ._store import ConversationStore

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)
type DeliveryKind = Literal[
    "task_exit",
    "inbox",
    "child_idle",
    "child_question",
    "child_question_close",
    "child_question_reroute",
    "question_answer_ack",
    "child_completion",
    "macro_completion",
]
type DeliverySource = Literal["prompt", "notification"]
type TerminalIntent = Literal[
    "segment_end",
    "finish_gate_wait",
    "finish_gate_timeout",
    "foreground_end",
    "idle_wake",
    "finish",
    "nudge",
]
type NotificationSelector = Callable[
    [
        "ConversationStore",
        Sequence[ConversationEntry],
        int | None,
        Callable[[Sequence[Message]], int] | None,
    ],
    tuple[list[ConversationEntry], Message | None, bool],
]

_LOGGER = logging.getLogger(__name__)
_MUTATION_ONLY_KINDS = {
    "child_question_close",
    "child_question_reroute",
    "question_answer_ack",
}


@dataclass(frozen=True, slots=True)
class PromptInput:
    text: str
    origin: MessageOrigin
    deadline: float | None


@dataclass(frozen=True, slots=True)
class NotificationInput:
    kind: DeliveryKind
    data: Mapping[str, JSONValue]
    dedupe_key: str | None = None
    after_commit: Callable[[], bool] | None = None
    deadline: float | None = None


type DeliveryInput = PromptInput | NotificationInput


@dataclass(frozen=True, slots=True)
class ProducerResult:
    entries: tuple[ConversationEntry, ...]
    changed: bool
    affected: int = 0


@dataclass(frozen=True, slots=True)
class DeliveryPublications:
    notifications: tuple[ConversationEntry, ...] = ()
    delivered_prompt: ConversationEntry | None = None


@dataclass(frozen=True, slots=True)
class StagedDelivery:
    notification_message: Message | None
    notification_ids: tuple[str, ...]
    prompt_message: Message | None
    prompt_id: str | None
    more_pending: bool


def _run_after_commit(item: NotificationInput, *, default: bool) -> bool:
    if item.after_commit is None:
        return default
    try:
        return item.after_commit() is True
    except Exception:
        _LOGGER.exception("step-delivery after-commit callback failed")
        return False


def _commit_delivery_input(
    store: ConversationStore, item: DeliveryInput
) -> ProducerResult:
    if isinstance(item, PromptInput):
        entry = store.pending_prompt_queue.append(
            item.text, origin=item.origin, deadline=item.deadline
        )
        return ProducerResult((entry,), True, 1)

    data = dict(item.data)
    kind = item.kind
    if kind in _MUTATION_ONLY_KINDS:
        return _commit_notification_mutation(store, kind, data)
    durable_kind = {
        "task_exit": "task_exited",
        "inbox": "project_inbox",
        "child_completion": "agent_completion",
        "macro_completion": "agent_completion",
    }.get(kind, kind)
    data["kind"] = durable_kind
    data.setdefault("origin", MessageOrigin.NOTIFICATION.value)
    validate_notification_data(data)
    with store._append_lock(deadline=item.deadline):
        store._load()
        store.pending_prompt_queue._check_deadline(item.deadline)
        if (
            kind == "child_idle"
            and data.get("child_instance_id") in store._active_completion_notifications
        ):
            return ProducerResult((), False)
        existing = _existing_notification(store, kind, data, item.dedupe_key)
        if existing is not None:
            changed = _run_after_commit(item, default=False)
            return ProducerResult((store._snapshot_entry(existing),), changed)
        if kind == "inbox":
            pending = next(
                (
                    entry
                    for entry in store._active_notifications.values()
                    if entry.data.get("kind") == "project_inbox"
                    and entry.data.get("project_id") == data.get("project_id")
                ),
                None,
            )
            if pending is not None:
                _ack, entry = store._append_many_unlocked(
                    [
                        ("notification_ack", {"notification_id": pending.id}),
                        ("notification", data),
                    ]
                )
                claimed = _run_after_commit(item, default=True)
                return ProducerResult((store._snapshot_entry(entry),), claimed, 1)
        entry = store._append_row_unlocked("notification", data)
        snapshot = store._snapshot_entry(entry)
        claimed = _run_after_commit(item, default=True)
        return ProducerResult((snapshot,), claimed, 1)


def _existing_notification(
    store: ConversationStore,
    kind: DeliveryKind,
    data: Mapping[str, Any],
    dedupe_key: str | None,
) -> ConversationEntry | None:
    if kind in {"child_completion", "macro_completion"}:
        child_id = data.get("child_instance_id")
        if type(child_id) is str:
            return store._active_completion_notifications.get(child_id)
    if dedupe_key is None:
        return None
    key_field = {
        "task_exit": "task_id",
        "inbox": "project_id",
        "child_question": "question_id",
    }.get(kind)
    if key_field is None:
        return None
    return next(
        (
            entry
            for entry in store._active_notifications.values()
            if entry.data.get(key_field) == data.get(key_field)
            and (
                kind != "inbox"
                or entry.data.get("message_ids") == data.get("message_ids")
            )
        ),
        None,
    )


def _commit_notification_mutation(
    store: ConversationStore, kind: DeliveryKind, data: Mapping[str, Any]
) -> ProducerResult:
    with store.pending_prompt_queue._lock():
        store._load()
        if kind == "question_answer_ack":
            entries, rows = _question_answer_ack_rows(store, data)
        else:
            entries, rows = _child_question_mutation_rows(store, kind, data)
        appended = store._append_many_unlocked(rows) if rows else []
        return ProducerResult(
            tuple(store._snapshot_entry(entry) for entry in appended),
            bool(rows),
            len(entries),
        )


def _question_answer_ack_rows(
    store: ConversationStore, data: Mapping[str, Any]
) -> tuple[list[ConversationEntry], list[tuple[str, dict[str, Any]]]]:
    entries = [
        entry
        for entry in store._active_notifications.values()
        if entry.data.get("kind") == "child_question"
        and entry.data.get("child_instance_id") == data.get("child_instance_id")
        and entry.data.get("question_id") == data.get("question_id")
    ]
    rows = [("notification_ack", {"notification_id": entry.id}) for entry in entries]
    return entries, rows


def _child_question_mutation_rows(
    store: ConversationStore, kind: DeliveryKind, data: Mapping[str, Any]
) -> tuple[list[ConversationEntry], list[tuple[str, dict[str, Any]]]]:
    child_id = data.get("child_instance_id")
    entries = [
        entry
        for entry in store._active_notifications.values()
        if entry.data.get("kind") == "child_question"
        and entry.data.get("child_instance_id") == child_id
    ]
    rows: list[tuple[str, dict[str, Any]]] = []
    for entry in entries:
        rows.append(("notification_ack", {"notification_id": entry.id}))
        if kind == "child_question_close":
            rows.append(
                (
                    "notification",
                    {
                        "origin": "notification",
                        "kind": "child_question_withdrawn",
                        "child_instance_id": child_id,
                        "question_id": entry.data["question_id"],
                        "description": "child question withdrawn",
                        "status": "withdrawn",
                        "text": (
                            "question withdrawn: child "
                            f"{data.get('reason', 'canceled')}"
                        ),
                    },
                )
            )
    return entries, rows


def _select_prompt_delivery(
    store: ConversationStore, pending_prompts: Sequence[ConversationEntry]
) -> tuple[ConversationEntry | None, Message | None, bool]:
    prompt_entry = pending_prompts[0] if pending_prompts else None
    if prompt_entry is None:
        return None, None, False
    prompt_message = store._active_pending_prompt_messages.get(prompt_entry.id)
    if prompt_message is not None:
        return prompt_entry, prompt_message, True
    origin = MessageOrigin(
        str(prompt_entry.data.get("origin", MessageOrigin.AGENT_SEND.value))
    )
    trailing = store._active_last_message
    if trailing is not None and _is_legacy_delivery_row(
        trailing, str(prompt_entry.data["text"]), origin
    ):
        return prompt_entry, trailing, True
    return (
        prompt_entry,
        Message(
            MessageRole.USER,
            [TextContent(str(prompt_entry.data["text"]))],
            metadata={
                "zeta.origin": origin.value,
                PENDING_PROMPT_ID_METADATA: prompt_entry.id,
            },
        ),
        False,
    )


def _persist_staged_messages(
    store: ConversationStore,
    selections: Sequence[tuple[Message | None, bool]],
) -> None:
    rows = [
        ("message", {"message": message.to_dict()})
        for message, persisted in selections
        if message is not None and not persisted
    ]
    if rows:
        store._append_many_unlocked(rows)


def _stage_delivery_request(
    store: ConversationStore,
    notification_budget: int | None,
    token_counter: Callable[[Sequence[Message]], int] | None,
    notification_selector: NotificationSelector,
) -> StagedDelivery:
    with store.pending_prompt_queue._lock():
        store._load()
        pending_prompts = store.pending_prompt_queue._pending_unlocked()
        pending_notifications = [
            entry
            for entry in store._active_notifications.values()
            if entry.data.get("display_only") is not True
        ]
        prompt_entry, prompt_message, prompt_persisted = _select_prompt_delivery(
            store, pending_prompts
        )
        notification_entries, notification_message, notification_persisted = (
            notification_selector(
                store, pending_notifications, notification_budget, token_counter
            )
        )
        _persist_staged_messages(
            store,
            (
                (notification_message, notification_persisted),
                (prompt_message, prompt_persisted),
            ),
        )
        represented_ids = tuple(entry.id for entry in notification_entries)
        return StagedDelivery(
            notification_message,
            represented_ids,
            prompt_message,
            prompt_entry.id if prompt_entry is not None else None,
            len(pending_prompts) > 1
            or len(pending_notifications) > len(notification_entries),
        )


def _acknowledge_staged_delivery(
    store: ConversationStore,
    prompt_id: str | None,
    notification_ids: tuple[str, ...],
) -> DeliveryPublications:
    with store.pending_prompt_queue._lock():
        store._load()
        rows: list[tuple[str, dict[str, JSONValue]]] = []
        prompt_entry = store._active_pending_prompt_entries.get(prompt_id or "")
        if prompt_entry is not None:
            rows.append(("pending_prompt_ack", {"prompt_id": prompt_id}))
        live_ids = {
            entry_id
            for entry_id in notification_ids
            if entry_id in store._active_notifications
        }
        notifications = tuple(
            store._snapshot_entry(store._active_notifications[entry_id])
            for entry_id in notification_ids
            if entry_id in live_ids
        )
        delivered_prompt = (
            store._snapshot_entry(prompt_entry) if prompt_entry is not None else None
        )
        rows.extend(
            ("notification_ack", {"notification_id": entry_id})
            for entry_id in notification_ids
            if entry_id in live_ids
        )
        if rows:
            store._append_many_unlocked(rows)
        return DeliveryPublications(notifications, delivered_prompt)


def _admit_terminal(
    store: ConversationStore, intent: TerminalIntent
) -> tuple[bool, bool]:
    with store.pending_prompt_queue._lock():
        store._load()
        prompts = store.pending_prompt_queue._pending_unlocked()
        notifications = [
            entry
            for entry in store._active_notifications.values()
            if entry.data.get("display_only") is not True
        ]
        pending = bool(prompts or notifications)
        closed = False
        if (
            not pending
            and intent in {"segment_end", "foreground_end", "finish"}
            and not store.pending_prompt_queue._queue_closed_unlocked()
        ):
            store._append_row_unlocked("pending_queue_closed", {})
            closed = True
        return pending, closed


def validate_delivery_input(item: DeliveryInput) -> DeliverySource:
    if isinstance(item, PromptInput):
        if not item.text.strip():
            raise ValueError("prompt text must be a nonempty string")
        if not isinstance(item.origin, MessageOrigin):
            raise TypeError("prompt origin must be a MessageOrigin")
        return "prompt"
    if not isinstance(item, NotificationInput):
        raise TypeError("delivery input must be PromptInput or NotificationInput")
    if item.kind not in {
        "task_exit",
        "inbox",
        "child_idle",
        "child_question",
        "child_question_close",
        "child_question_reroute",
        "question_answer_ack",
        "child_completion",
        "macro_completion",
    }:
        raise ValueError("unknown delivery kind")
    if not item.data:
        raise ValueError("notification data must not be empty")
    if (
        item.kind in {"task_exit", "inbox", "child_question", "child_completion"}
        and not item.dedupe_key
    ):
        raise ValueError(f"{item.kind} requires a dedupe key")
    if item.after_commit is not None and item.kind not in {
        "inbox",
        "child_completion",
        "macro_completion",
    }:
        raise ValueError("after_commit is not valid for this delivery kind")
    return "notification"


def _is_legacy_delivery_row(message: Message, text: str, origin: MessageOrigin) -> bool:
    return (
        message.role is MessageRole.USER
        and message.metadata.get("zeta.origin") == origin.value
        and message.content == [TextContent(text)]
    )


class StepDeliveryStoreMixin:
    """Own the durable transactions used by one step-delivery flow."""

    def _warm_step_delivery_writer(self) -> None:
        return None

    def _commit_delivery_input(self, item: DeliveryInput) -> ProducerResult:
        return _commit_delivery_input(self, item)

    def _stage_delivery_request(
        self,
        notification_budget: int | None,
        token_counter: Callable[[Sequence[Message]], int] | None,
        notification_selector: NotificationSelector,
    ) -> StagedDelivery:
        return _stage_delivery_request(
            self, notification_budget, token_counter, notification_selector
        )

    def _acknowledge_staged_delivery(
        self, prompt_id: str | None, notification_ids: tuple[str, ...]
    ) -> DeliveryPublications:
        return _acknowledge_staged_delivery(self, prompt_id, notification_ids)

    def _admit_step_delivery_terminal(
        self, intent: TerminalIntent
    ) -> tuple[bool, bool]:
        return _admit_terminal(self, intent)
