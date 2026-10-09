"""Durable pending-prompt queue ownership."""

from __future__ import annotations

import time
import weakref
from typing import TYPE_CHECKING

from ...protocol.types import MessageOrigin
from ..checkpoints import ConversationEntry

if TYPE_CHECKING:
    from ._store import ConversationStore

MAX_PENDING_PROMPT_TEXT = 16_000


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
        question_id: str | None = None,
    ) -> ConversationEntry:
        if type(text) is not str or not text.strip():
            raise ValueError("pending prompt text must be a nonempty string")
        if len(text) > MAX_PENDING_PROMPT_TEXT:
            raise ValueError("pending prompt text is too long")
        if question_id is not None and not question_id:
            raise ValueError("question id must be nonempty")
        with self._store._append_lock(deadline=deadline):
            self._store._load()
            self._check_deadline(deadline)
            if self._queue_closed_unlocked():
                raise PendingPromptsClosedError("pending prompt queue is closed")
            data = {"text": text, "origin": origin.value}
            if question_id is not None:
                data["question_id"] = question_id
            entry = self._store._append_row_unlocked(
                "pending_prompt", data, deadline=deadline
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
