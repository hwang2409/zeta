"""Own pending-prompt sends, resume recovery, delivery, and acknowledgement."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..core.checkpoints import ConversationEntry, ConversationIntegrityError
from ..core.session_files import SessionNotFoundError
from ..core.store import (
    ConversationStore,
    PendingPromptCommitTimeoutError,
    PendingPromptsClosedError,
)
from ..core.store._pending_prompts import PENDING_PROMPT_ID_METADATA
from ..core.store._validation import validate_notification_data
from ..protocol.types import Message, MessageOrigin, MessageRole, TextContent
from .conversation_channel import ConversationChannel, has_follow_up_loop

if TYPE_CHECKING:
    from ..core.context import ContextAssembler
    from ..core.store._client_delivery import ClientSteeringBatch

AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0
QUESTION_ANSWER_PATTERN = re.compile(r"^Re ([^:\s]+):(?:\s|$)")
_PREVIEW_WHITESPACE = re.compile(r"\s+")

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
DeliveryKind = Literal[
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
DeliverySource = Literal["prompt", "notification"]


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


type DeliveryInput = PromptInput | NotificationInput


@dataclass(frozen=True, slots=True)
class ProducerResult:
    entries: tuple[ConversationEntry, ...]
    changed: bool
    affected: int = 0


def delivery_preview(text: str) -> str:
    """Return the shared bounded one-line delivery preview."""

    normalized = _PREVIEW_WHITESPACE.sub(" ", text).strip()
    if len(normalized) <= 120:
        return normalized
    return normalized[:119] + "…"


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


RequestCause = Literal["after_tools", "terminal_fallback", "idle_start", "retry"]
TerminalIntent = Literal[
    "segment_end",
    "finish_gate_wait",
    "finish_gate_timeout",
    "foreground_end",
    "idle_wake",
    "finish",
    "nudge",
]


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    messages: tuple[Message, ...]
    steering: ClientSteeringBatch | None
    notification_ids: tuple[str, ...] = ()
    prompt_id: str | None = None
    overflow_message: str | None = None


@dataclass(frozen=True, slots=True)
class DeliveryPublications:
    notifications: tuple[ConversationEntry, ...] = ()
    delivered_prompt: ConversationEntry | None = None


@dataclass(frozen=True, slots=True)
class TerminalResult:
    pending: bool
    closed_empty: bool


@dataclass(frozen=True, slots=True)
class _StagedDelivery:
    notification_message: Message | None
    notification_ids: tuple[str, ...]
    prompt_message: Message | None
    prompt_id: str | None
    more_pending: bool


_MUTATION_ONLY_KINDS = {
    "child_question_close",
    "child_question_reroute",
    "question_answer_ack",
}


class StepDeliveryFlow:
    """Own production, request staging, and acknowledgement for one store."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        wake_sink: Callable[[DeliverySource], None] | None = None,
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        self.store = store
        self._loop = asyncio.get_running_loop()
        self._wake_sink = wake_sink
        self._context_assembler = context_assembler
        self._edges: set[DeliverySource] = set()
        self._active: _StagedDelivery | None = None
        self._active_steering: ClientSteeringBatch | None = None
        self._prompt_priority: str | None = None

    async def produce(self, item: DeliveryInput, /) -> ProducerResult:
        source = _validate_delivery_input(item)
        self._require_owner_loop()
        result = await self.store._to_thread_durable(
            self.store._commit_delivery_input, item
        )
        if result.changed and not (
            isinstance(item, NotificationInput)
            and (
                item.kind in _MUTATION_ONLY_KINDS
                or item.data.get("display_only") is True
            )
        ):
            self._arm_edge(source)
        return result

    def wake(self, source: DeliverySource) -> None:
        if source not in {"prompt", "notification"}:
            raise ValueError("unknown delivery source")
        try:
            owner_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            owner_loop = False
        if owner_loop:
            self._arm_edge(source)
        else:
            self._loop.call_soon_threadsafe(self._arm_edge, source)

    async def prepare_request(
        self,
        cause: RequestCause,
        assembled: Sequence[Message],
        steering: ClientSteeringBatch | None,
    ) -> PreparedRequest:
        """Stage one fixed suffix and return it within the whole-request budget."""

        self._require_owner_loop()
        if cause not in {"after_tools", "terminal_fallback", "idle_start", "retry"}:
            raise ValueError("unknown request cause")
        assembler = self._context_assembler
        if cause == "retry" and self._active is not None:
            steering = self._active_steering
        else:
            # Clear before reload: a later producer re-arms the level edge.
            self._edges.clear()
            staged = await self.store._to_thread_durable(
                self.store._stage_delivery_request,
                0 if assembler is not None else None,
                None if assembler is None else assembler.estimate_messages,
            )
            if self._prompt_priority is not None:
                steering = None
            self._active = staged
            self._active_steering = steering
            if staged.more_pending:
                if staged.prompt_id is not None:
                    self._edges.add("prompt")
                self._edges.add("notification")
        staged = self._active
        assert staged is not None
        if assembler is None:
            return self._compose(staged, assembled, steering)

        prompt = staged.prompt_message
        prompt_tokens = assembler.estimate_messages(() if prompt is None else (prompt,))
        steering_messages = () if steering is None else steering.messages
        steering_tokens = assembler.estimate_messages(steering_messages)
        budget = assembler.estimator_token_budget
        context = self._without_staged(assembled, staged)

        request = self._compose(
            staged, context, steering, include_notification=False
        )
        if assembler.estimate_messages(request.messages) <= budget:
            return await self._add_notification(staged, context, steering, budget)

        reserved = prompt_tokens + steering_tokens
        try:
            context = await assembler.assemble(
                force=True,
                emergency=True,
                reserved_estimator_tokens=reserved,
            )
        except Exception as exc:
            from ..core.context import BudgetExceeded

            if not isinstance(exc, BudgetExceeded):
                raise
            context = await assembler.assemble(force=True, emergency=True)
        context = self._without_staged(context, staged)
        request = self._compose(
            staged, context, steering, include_notification=False
        )
        if assembler.estimate_messages(request.messages) <= budget:
            return await self._add_notification(staged, context, steering, budget)

        prompt_id = staged.prompt_id
        if prompt_id is not None:
            self._prompt_priority = prompt_id
        steering_only = self._compose(
            staged,
            context,
            steering,
            include_notification=False,
            include_prompt=False,
        )
        steering_only_tokens = assembler.estimate_messages(steering_only.messages)
        if steering is not None and steering_only_tokens <= budget:
            self._active = self._request_active(
                staged, notification=False, prompt=False
            )
            return steering_only

        self._active = self._request_active(
            staged, notification=False, prompt=False
        )
        self._active_steering = None
        overflow = (
            "your message was not sent: it is larger than this model's context "
            f"window ({steering_tokens} of {budget} tokens). shorten it and resend "
            "it, or switch to a larger model."
        )
        return PreparedRequest((), None, overflow_message=overflow)

    @staticmethod
    def _without_staged(
        assembled: Sequence[Message], staged: _StagedDelivery
    ) -> list[Message]:
        notification_ids = frozenset(staged.notification_ids)
        return [
            message
            for message in assembled
            if (
                staged.prompt_id is None
                or message.metadata.get(PENDING_PROMPT_ID_METADATA) != staged.prompt_id
            )
            and not notification_ids.intersection(_message_notification_ids(message))
        ]

    async def _add_notification(
        self,
        staged: _StagedDelivery,
        context: Sequence[Message],
        steering: ClientSteeringBatch | None,
        budget: int,
    ) -> PreparedRequest:
        assembler = self._context_assembler
        assert assembler is not None
        without_notification = self._compose(
            staged, context, steering, include_notification=False
        )
        capacity = min(
            assembler.delivery_notification_budget,
            budget - assembler.estimate_messages(without_notification.messages),
        )
        selected = staged
        if capacity > 0:
            selected = await self.store._to_thread_durable(
                self.store._stage_delivery_request,
                capacity,
                assembler.estimate_messages,
            )
        request = self._compose(
            selected,
            context,
            steering,
            include_notification=selected.notification_message is not None,
        )
        if assembler.estimate_messages(request.messages) > budget:
            selected = self._request_active(selected, notification=False)
            request = self._compose(
                selected, context, steering, include_notification=False
            )
        self._active = selected
        return request

    def _compose(
        self,
        staged: _StagedDelivery,
        assembled: Sequence[Message],
        steering: ClientSteeringBatch | None,
        *,
        include_notification: bool = True,
        include_prompt: bool = True,
    ) -> PreparedRequest:
        context = self._without_staged(assembled, staged)
        if include_notification and staged.notification_message is not None:
            context.append(staged.notification_message)
        if include_prompt and staged.prompt_message is not None:
            context.append(staged.prompt_message)
        if steering is not None:
            context.extend(steering.messages)
        return PreparedRequest(
            tuple(context),
            steering,
            staged.notification_ids if include_notification else (),
            staged.prompt_id if include_prompt else None,
        )

    @staticmethod
    def _request_active(
        staged: _StagedDelivery,
        *,
        notification: bool,
        prompt: bool = True,
    ) -> _StagedDelivery:
        return _StagedDelivery(
            staged.notification_message if notification else None,
            staged.notification_ids if notification else (),
            staged.prompt_message if prompt else None,
            staged.prompt_id if prompt else None,
            staged.more_pending or bool(staged.notification_ids and not notification),
        )

    async def acknowledge_active(self) -> DeliveryPublications:
        """Durably acknowledge the fixed suffix selected for provider evidence."""

        self._require_owner_loop()
        active = self._active
        if active is None:
            return DeliveryPublications()
        publications = await self.store._to_thread_durable(
            self.store._acknowledge_staged_delivery,
            active.prompt_id,
            active.notification_ids,
        )
        if active.prompt_id is not None and active.prompt_id == self._prompt_priority:
            self._prompt_priority = None
        self._active = None
        self._active_steering = None
        return publications

    def release_active(self) -> None:
        """Re-arm a staged request that ended before provider acknowledgement."""

        self._require_owner_loop()
        staged = self._active
        if staged is None:
            return
        if staged.prompt_id is not None:
            self._edges.add("prompt")
        if staged.notification_ids:
            self._edges.add("notification")
        if self._prompt_priority is not None:
            self._edges.add("prompt")
        self._active = None
        self._active_steering = None

    async def admit_terminal(self, intent: TerminalIntent) -> TerminalResult:
        """Atomically decide whether terminal processing has durable input."""

        self._require_owner_loop()
        if intent not in {
            "segment_end",
            "finish_gate_wait",
            "finish_gate_timeout",
            "foreground_end",
            "idle_wake",
            "finish",
            "nudge",
        }:
            raise ValueError("unknown terminal intent")
        pending, closed = await self.store._to_thread_durable(
            self.store._admit_step_delivery_terminal, intent
        )
        if pending:
            self._edges.update(("prompt", "notification"))
        return TerminalResult(pending, closed)

    def initialize_resident_edges(self) -> None:
        """Seed level state from the already-loaded resident store."""

        if self.store.pending_prompt_queue._pending_unlocked():
            self._edges.add("prompt")
        if any(
            entry_id not in self.store._active_notification_acks
            and entry.data.get("display_only") is not True
            for entry_id, entry in self.store._active_notifications.items()
        ):
            self._edges.add("notification")

    @property
    def has_prompt_edge(self) -> bool:
        return "prompt" in self._edges

    @property
    def has_pending_edge(self) -> bool:
        return bool(self._edges)

    @property
    def active_notification_ids(self) -> frozenset[str]:
        return frozenset(self._active.notification_ids) if self._active else frozenset()

    @property
    def active_prompt_id(self) -> str | None:
        return self._active.prompt_id if self._active else None

    @property
    def prompt_priority(self) -> str | None:
        return self._prompt_priority

    def _require_owner_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("step delivery flow used from a non-owning event loop")

    def _arm_edge(self, source: DeliverySource) -> None:
        self._edges.add(source)
        if self._wake_sink is not None:
            self._wake_sink(source)


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
        "summary": _bounded_text(
            report_data.get("summary", data.get("text", "")), 256
        ),
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
        "relevant.\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
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
    with store._append_lock():
        store._load()
        if (
            kind == "child_idle"
            and data.get("child_instance_id")
            in store._active_completion_notifications
        ):
            return ProducerResult((), False)
        existing = _existing_notification(store, kind, data, item.dedupe_key)
        if existing is not None:
            changed = item.after_commit() is True if item.after_commit else False
            return ProducerResult((store._snapshot_entry(existing),), changed)
        if kind == "inbox":
            pending = next(
                (
                    entry
                    for notification_id, entry in store._active_notifications.items()
                    if notification_id not in store._active_notification_acks
                    and entry.data.get("kind") == "project_inbox"
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
                claimed = (
                    item.after_commit() is True if item.after_commit else True
                )
                return ProducerResult((store._snapshot_entry(entry),), claimed, 1)
        entry = store._append_row_unlocked("notification", data)
        snapshot = store._snapshot_entry(entry)
        claimed = item.after_commit() is True if item.after_commit else True
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
                or entry.id not in store._active_notification_acks
                and entry.data.get("message_ids") == data.get("message_ids")
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
            entries = [
                entry
                for entry in store._active_notifications.values()
                if entry.id not in store._active_notification_acks
                and entry.data.get("kind") == "child_question"
                and entry.data.get("child_instance_id")
                == data.get("child_instance_id")
                and entry.data.get("question_id") == data.get("question_id")
            ]
            rows = [
                ("notification_ack", {"notification_id": entry.id})
                for entry in entries
            ]
        else:
            child_id = data.get("child_instance_id")
            reason = str(data.get("reason", "canceled"))
            entries = [
                entry
                for entry in store._active_notifications.values()
                if entry.id not in store._active_notification_acks
                and entry.data.get("kind") == "child_question"
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
                                "text": f"question withdrawn: child {reason}",
                            },
                        )
                    )
        appended = store._append_many_unlocked(rows) if rows else []
        return ProducerResult(
            tuple(store._snapshot_entry(entry) for entry in appended),
            bool(rows),
            len(entries),
        )


def _stage_delivery_request(
    store: ConversationStore,
    notification_budget: int | None = None,
    token_counter: Callable[[Sequence[Message]], int] | None = None,
) -> _StagedDelivery:
    with store.pending_prompt_queue._lock():
        store._load()
        pending_prompts = store.pending_prompt_queue._pending_unlocked()
        pending_notifications = [
            entry
            for entry_id, entry in store._active_notifications.items()
            if entry_id not in store._active_notification_acks
            and entry.data.get("display_only") is not True
        ]
        branch_messages = [
            Message.from_dict(entry.data["message"])
            for entry in store.replay()
            if entry.type == "message"
        ]
        prompt_entry = pending_prompts[0] if pending_prompts else None
        prompt_message = None
        if prompt_entry is not None:
            prompt_message = next(
                (
                    message
                    for message in reversed(branch_messages)
                    if message.metadata.get(PENDING_PROMPT_ID_METADATA)
                    == prompt_entry.id
                ),
                None,
            )
            if prompt_message is None and branch_messages:
                trailing = branch_messages[-1]
                origin = MessageOrigin(
                    str(prompt_entry.data.get("origin", MessageOrigin.AGENT_SEND.value))
                )
                if _is_legacy_delivery_row(
                    trailing, str(prompt_entry.data["text"]), origin
                ):
                    prompt_message = trailing
            if prompt_message is None:
                prompt_message = Message(
                    MessageRole.USER,
                    [TextContent(str(prompt_entry.data["text"]))],
                    metadata={
                        "zeta.origin": str(
                            prompt_entry.data.get(
                                "origin", MessageOrigin.AGENT_SEND.value
                            )
                        ),
                        PENDING_PROMPT_ID_METADATA: prompt_entry.id,
                    },
                )
        notification_entries: list[ConversationEntry] = []
        notification_message = None
        pending_notification_ids = {entry.id for entry in pending_notifications}
        if notification_budget != 0:
            for message in reversed(branch_messages):
                represented = _message_notification_ids(message)
                if represented and represented.issubset(pending_notification_ids):
                    if (
                        notification_budget is None
                        or token_counter is None
                        or token_counter((message,)) <= notification_budget
                    ):
                        notification_message = message
                        notification_entries = [
                            entry
                            for entry in pending_notifications
                            if entry.id in represented
                        ]
                    break
        if notification_message is None and pending_notifications and notification_budget != 0:
            # Full JSON yields to a digest, fewer FIFO members, then deferral.
            for digest in (False, True):
                notification_entries = []
                notification_message = None
                for entry in pending_notifications:
                    candidate = [*notification_entries, entry]
                    message = _notification_message(candidate, digest=digest)
                    if len(json.dumps(message.to_dict()).encode()) > 64 * 1024:
                        break
                    if (
                        notification_budget is not None
                        and token_counter is not None
                        and token_counter((message,)) > notification_budget
                    ):
                        break
                    notification_entries = candidate
                    notification_message = message
                if notification_message is not None:
                    break
        rows: list[tuple[str, dict[str, object]]] = []
        persisted = {message.to_dict().__repr__() for message in branch_messages}
        for message in (notification_message, prompt_message):
            if message is not None and message.to_dict().__repr__() not in persisted:
                rows.append(("message", {"message": message.to_dict()}))
        if rows:
            store._append_many_unlocked(rows)
        represented_ids = tuple(entry.id for entry in notification_entries)
        return _StagedDelivery(
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
        rows: list[tuple[str, dict[str, object]]] = []
        if prompt_id is not None and prompt_id in {
            entry.id for entry in store.pending_prompt_queue._pending_unlocked()
        }:
            rows.append(("pending_prompt_ack", {"prompt_id": prompt_id}))
        live_ids = {
            entry_id
            for entry_id in notification_ids
            if entry_id in store._active_notifications
            and entry_id not in store._active_notification_acks
        }
        rows.extend(
            ("notification_ack", {"notification_id": entry_id})
            for entry_id in notification_ids
            if entry_id in live_ids
        )
        if rows:
            store._append_many_unlocked(rows)
        notifications = tuple(
            store._snapshot_entry(store._active_notifications[entry_id])
            for entry_id in notification_ids
            if entry_id in store._active_notifications
        )
        delivered_prompt = next(
            (
                store._snapshot_entry(entry)
                for entry in store.replay()
                if entry.type == "pending_prompt" and entry.id == prompt_id
            ),
            None,
        )
        return DeliveryPublications(notifications, delivered_prompt)


def _admit_terminal(
    store: ConversationStore, intent: TerminalIntent
) -> tuple[bool, bool]:
    with store.pending_prompt_queue._lock():
        store._load()
        prompts = store.pending_prompt_queue._pending_unlocked()
        notifications = [
            entry
            for entry_id, entry in store._active_notifications.items()
            if entry_id not in store._active_notification_acks
            and entry.data.get("display_only") is not True
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


def _validate_delivery_input(item: DeliveryInput) -> DeliverySource:
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
    if item.kind in {"task_exit", "inbox", "child_question", "child_completion"} and not item.dedupe_key:
        raise ValueError(f"{item.kind} requires a dedupe key")
    if item.after_commit is not None and item.kind not in {
        "inbox",
        "child_completion",
        "macro_completion",
    }:
        raise ValueError("after_commit is not valid for this delivery kind")
    return "notification"


@dataclass(frozen=True)
class PendingDelivery:
    """One queued prompt and the persisted user row used for its next turn."""

    prompt_id: str
    message: Message
    persist_message: bool

    @classmethod
    def resume(
        cls,
        store: ConversationStore,
        prompt_id: str,
        text: str,
        origin: MessageOrigin,
    ) -> PendingDelivery:
        messages = store.messages()
        current = next(
            (
                message
                for message in reversed(messages)
                if message.metadata.get(PENDING_PROMPT_ID_METADATA) == prompt_id
            ),
            None,
        )
        if current is None and messages:
            trailing = messages[-1]
            if _is_legacy_delivery_row(trailing, text, origin):
                current = trailing
        if current is not None:
            return cls(prompt_id, current, False)
        return cls(
            prompt_id,
            Message(
                MessageRole.USER,
                [TextContent(text)],
                metadata={
                    "zeta.origin": origin.value,
                    PENDING_PROMPT_ID_METADATA: prompt_id,
                },
            ),
            True,
        )

    async def acknowledge(
        self, store: ConversationStore, context_messages: list[Message]
    ) -> None:
        await store.acknowledge_pending_prompt_messages(
            context_messages, prompt_ids=(self.prompt_id,)
        )


def pending_prompt_acknowledgement(
    store: ConversationStore,
    delivery: PendingDelivery | None,
    context_messages: list[Message],
) -> Callable[[], Awaitable[None]] | None:
    """Select the first-provider-event acknowledgement for this context."""

    if delivery is not None:
        return partial(delivery.acknowledge, store, context_messages)
    if any(
        PENDING_PROMPT_ID_METADATA in message.metadata for message in context_messages
    ):
        return partial(store.acknowledge_pending_prompt_messages, context_messages)
    return None


def _is_legacy_delivery_row(
    message: Message, text: str, origin: MessageOrigin
) -> bool:
    return (
        message.role is MessageRole.USER
        and message.metadata.get("zeta.origin") == origin.value
        and message.content == [TextContent(text)]
    )


def _answered_question_notification_id(
    store: ConversationStore, child_instance_id: str, message: str
) -> str | None:
    match = QUESTION_ANSWER_PATTERN.match(message)
    if match is None:
        return None
    question_id = match.group(1)
    return next(
        (
            entry.id
            for entry in store.agent_notifications()
            if entry.data.get("kind") == "child_question"
            and entry.data.get("child_instance_id") == child_instance_id
            and entry.data.get("question_id") == question_id
        ),
        None,
    )


async def pending_deliveries(
    store: ConversationStore,
) -> list[ConversationEntry]:
    """Read pending deliveries without file I/O on the event loop."""

    return await store._to_thread_durable(store.pending_prompts)


async def close_pending_delivery_queue_if_empty(
    store: ConversationStore,
) -> list[ConversationEntry]:
    """Atomically return pending deliveries or close empty admission off-loop."""

    return await store._to_thread_durable(store.close_pending_queue_if_empty)


async def close_pending_delivery_queue(store: ConversationStore) -> None:
    """Close pending-delivery admission off-loop."""

    await store._to_thread_durable(store.close_pending_queue)


def _commit_follow_up(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    *,
    channel: ConversationChannel,
) -> str | None:
    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    no_live_run = (
        f"no live run/child {child_instance_id!r}; it was canceled, finished, "
        "or never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    if not has_follow_up_loop(
        accepts_follow_ups=marker.get("accepts_follow_ups") is True,
        background=marker.get("background") is True,
    ):
        agent_type = marker.get("agent_type") or "general"
        reason = (
            "it is a foreground child"
            if marker.get("background") is not True
            else "its profile does not accept follow-ups"
        )
        return (
            f"agent_send rejected for {agent_type} child {child_instance_id!r}: "
            f"{reason}"
        )
    deadline = time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS
    try:
        child_path = Path(str(marker["child_session_path"]))
        with ConversationStore(
            child_path.parent,
            session_id=child_path.name,
            cwd=parent_store.cwd,
            _lock_deadline=deadline,
            _must_exist=True,
        ) as child_store:
            child_store.pending_prompt_queue.append(
                message,
                origin=MessageOrigin.AGENT_SEND,
                deadline=deadline,
            )
    except PendingPromptCommitTimeoutError:
        return "pending prompt commit timed out before the queue could be changed"
    except (PendingPromptsClosedError, SessionNotFoundError):
        return no_live_run
    except ConversationIntegrityError:
        if (child_path / "conversation.jsonl").exists():
            raise
        return no_live_run
    channel.wake(child_instance_id)
    return None


async def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    *,
    channel: ConversationChannel,
) -> str | None:
    """Queue one follow-up through the child's resident delivery flow."""

    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    no_live_run = (
        f"no live run/child {child_instance_id!r}; it was canceled, finished, "
        "or never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    if not has_follow_up_loop(
        accepts_follow_ups=marker.get("accepts_follow_ups") is True,
        background=marker.get("background") is True,
    ):
        agent_type = marker.get("agent_type") or "general"
        reason = (
            "it is a foreground child"
            if marker.get("background") is not True
            else "its profile does not accept follow-ups"
        )
        return (
            f"agent_send rejected for {agent_type} child {child_instance_id!r}: "
            f"{reason}"
        )
    try:
        await channel.produce_to(
            child_instance_id,
            PromptInput(
                message,
                MessageOrigin.AGENT_SEND,
                time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS,
            ),
        )
    except PendingPromptCommitTimeoutError:
        return "pending prompt commit timed out before the queue could be changed"
    except (LookupError, PendingPromptsClosedError):
        return no_live_run
    match = QUESTION_ANSWER_PATTERN.match(message)
    if match is not None:
        await channel.produce_to_parent_of(
            child_instance_id,
            NotificationInput(
                "question_answer_ack",
                {
                    "child_instance_id": child_instance_id,
                    "question_id": match.group(1),
                },
            ),
        )
    return None
