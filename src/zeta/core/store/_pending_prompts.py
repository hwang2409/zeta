"""Durable pending-prompt queue ownership."""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

from ...protocol.types import Message, MessageOrigin
from ..checkpoints import ConversationEntry

if TYPE_CHECKING:
    from ._store import ConversationStore

MAX_PENDING_PROMPT_TEXT = 16_000
PENDING_PROMPT_ID_METADATA = "zeta.pending_prompt_id"


class PendingPromptsClosedError(RuntimeError):
    """Raised when a run has already decided to finish and refuses new prompts."""


class PendingPromptCommitTimeoutError(TimeoutError):
    """Raised when a pending-prompt commit misses its pre-write deadline."""


class PendingPromptQueue:
    """Own every durable append, acknowledgement, close, and timeout decision."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = weakref.proxy(store)

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise PendingPromptCommitTimeoutError(
                "pending prompt commit deadline exceeded"
            )

    def append(
        self,
        text: str,
        *,
        origin: MessageOrigin = MessageOrigin.UNKNOWN,
        deadline: float | None = None,
    ) -> ConversationEntry:
        if type(text) is not str or not text.strip():
            raise ValueError("pending prompt text must be a nonempty string")
        if len(text) > MAX_PENDING_PROMPT_TEXT:
            raise ValueError("pending prompt text is too long")
        with self._store._append_lock(deadline=deadline):
            self._store._load()
            self._check_deadline(deadline)
            if self._queue_closed_unlocked():
                raise PendingPromptsClosedError("pending prompt queue is closed")
            entry = self._store._append_row_unlocked(
                "pending_prompt",
                {"text": text, "origin": origin.value},
                deadline=deadline,
            )
            return self._store._snapshot_entry(entry)

    def pending(self) -> list[ConversationEntry]:
        with self._store._append_lock():
            self._store._load()
            return self._pending_unlocked()

    def acknowledge(self, prompt_id: str) -> None:
        with self._store._append_lock():
            self._store._load()
            branch = self._store.replay()
            prompts = [entry for entry in branch if entry.type == "pending_prompt"]
            if not any(entry.id == prompt_id for entry in prompts):
                raise ValueError(f"unknown pending prompt: {prompt_id}")
            acknowledged = {
                entry.data["prompt_id"]
                for entry in branch
                if entry.type == "pending_prompt_ack"
            }
            if prompt_id not in acknowledged:
                self._store._append_row_unlocked(
                    "pending_prompt_ack", {"prompt_id": prompt_id}
                )

    def acknowledge_messages(self, messages: list[Message]) -> None:
        """Consume pending prompts included in one provider context."""
        prompt_ids = {
            prompt_id
            for message in messages
            if isinstance(
                prompt_id := message.metadata.get(PENDING_PROMPT_ID_METADATA), str
            )
        }
        if not prompt_ids:
            return
        with self._store._append_lock():
            self._store._load()
            pending_ids = {entry.id for entry in self._pending_unlocked()}
            rows = [
                ("pending_prompt_ack", {"prompt_id": prompt_id})
                for prompt_id in prompt_ids & pending_ids
            ]
            if rows:
                self._store._append_many_unlocked(rows)

    def close_if_empty(self) -> list[ConversationEntry]:
        with self._store._append_lock():
            self._store._load()
            pending = self._pending_unlocked()
            if not pending and not self._queue_closed_unlocked():
                self._store._append_row_unlocked("pending_queue_closed", {})
            return pending

    def close(self) -> None:
        with self._store._append_lock():
            self._store._load()
            if not self._queue_closed_unlocked():
                self._store._append_row_unlocked("pending_queue_closed", {})

    def accept_explicit_finish(
        self,
        report: dict[str, object],
        *,
        persisted_notification_ids: frozenset[str] = frozenset(),
        delivering_prompt_id: str | None = None,
    ) -> str | None:
        """Atomically close admission and persist one accepted finish report."""
        with self._store._append_lock():
            self._store._load()
            transition = self._terminal_transition_unlocked()
            if transition is not None:
                return (
                    "child is finishing"
                    if transition.data["action"] == "finish"
                    else "child is canceled"
                )
            if any(
                entry.id != delivering_prompt_id for entry in self._pending_unlocked()
            ):
                return "pending follow-up must be read before finish"
            pending_notifications = [
                entry
                for notification_id, entry in self._store._active_notifications.items()
                if notification_id not in self._store._active_notification_acks
                and notification_id not in persisted_notification_ids
                and entry.data.get("display_only") is not True
            ]
            if pending_notifications:
                return "pending notification must be read before finish"
            self._store._append_many_unlocked(
                [
                    ("pending_queue_closed", {}),
                    ("terminal_transition", {"action": "finish", "report": report}),
                ]
            )
            return None

    def cancel_explicit_child(self) -> bool:
        """Close admission unless an accepted finish already owns termination."""
        with self._store._append_lock():
            self._store._load()
            transition = self._terminal_transition_unlocked()
            if transition is not None:
                return False
            rows = [
                ("pending_prompt_ack", {"prompt_id": entry.id})
                for entry in self._pending_unlocked()
            ]
            rows.extend(
                [
                    ("pending_queue_closed", {}),
                    ("terminal_transition", {"action": "cancel"}),
                ]
            )
            self._store._append_many_unlocked(rows)
            return True

    def explicit_finish_report(self) -> dict[str, object] | None:
        with self._store._append_lock():
            self._store._load()
            transition = self._terminal_transition_unlocked()
            if transition is None or transition.data["action"] != "finish":
                return None
            return dict(transition.data["report"])

    def idle_nudge_eligible(
        self,
        *,
        admission_allowed: Callable[[], bool] = lambda: True,
        persisted_notification_ids: frozenset[str] = frozenset(),
    ) -> bool:
        """Atomically reject a nudge when input, work, or termination exists."""
        with self._store._append_lock():
            self._store._load()
            return admission_allowed() and self._idle_nudge_eligible_unlocked(
                persisted_notification_ids
            )

    def terminal_transition(self) -> str | None:
        with self._store._append_lock():
            self._store._load()
            transition = self._terminal_transition_unlocked()
            return str(transition.data["action"]) if transition is not None else None

    def _idle_nudge_eligible_unlocked(
        self, persisted_notification_ids: frozenset[str]
    ) -> bool:
        if self._terminal_transition_unlocked() is not None or self._pending_unlocked():
            return False
        return not any(
            notification_id not in self._store._active_notification_acks
            and notification_id not in persisted_notification_ids
            and entry.data.get("display_only") is not True
            for notification_id, entry in self._store._active_notifications.items()
        )

    def _terminal_transition_unlocked(self) -> ConversationEntry | None:
        return next(
            (
                entry
                for entry in reversed(self._store.replay())
                if entry.type == "terminal_transition"
            ),
            None,
        )

    def _queue_closed_unlocked(self) -> bool:
        return any(
            entry.type == "pending_queue_closed" for entry in self._store._entries
        )

    def _pending_unlocked(self) -> list[ConversationEntry]:
        branch = self._store.replay()
        acknowledged = {
            entry.data["prompt_id"]
            for entry in branch
            if entry.type == "pending_prompt_ack"
        }
        return [
            self._store._snapshot_entry(entry)
            for entry in branch
            if entry.type == "pending_prompt" and entry.id not in acknowledged
        ]
