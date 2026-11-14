"""Durable pending-prompt queue ownership."""

from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ...protocol.types import MessageOrigin
from ..checkpoints import ConversationEntry

if TYPE_CHECKING:
    from ._store import ConversationStore

MAX_PENDING_PROMPT_TEXT = 16_000
PENDING_PROMPT_ID_METADATA = "zeta.pending_prompt_id"


class PendingPromptsClosedError(RuntimeError):
    """Raised when a run has already decided to finish and refuses new prompts."""


class PendingPromptCommitTimeoutError(TimeoutError):
    """Raised when a pending-prompt commit misses its pre-write deadline."""


class PendingPromptWriterReloadedError(PendingPromptCommitTimeoutError):
    """Raised after a writer reload consumes its timed commit window."""


class PendingPromptLoopThreadLockError(RuntimeError):
    """Raised before pending-prompt acknowledgement can block an event loop."""


class PendingPromptStoreMixin:
    """Expose pending-prompt transactions through one store interface."""

    pending_prompt_queue: PendingPromptQueue

    def append_pending_prompt(self, text: str) -> ConversationEntry:
        return self.pending_prompt_queue.append(text)

    def close_pending_queue_if_empty(self) -> list[ConversationEntry]:
        return self.pending_prompt_queue.close_if_empty()

    def close_pending_queue(self) -> None:
        self.pending_prompt_queue.close()

    def pending_prompts(self) -> list[ConversationEntry]:
        return self.pending_prompt_queue.pending()

    def acknowledge_pending_prompt(self, prompt_id: str) -> None:
        self.pending_prompt_queue.acknowledge(prompt_id)

    def _accept_explicit_finish(
        self,
        report: dict[str, object],
        persisted_notification_ids: frozenset[str],
        delivering_prompt_id: str | None,
    ) -> str | None:
        return self.pending_prompt_queue.accept_explicit_finish(
            report,
            persisted_notification_ids=persisted_notification_ids,
            delivering_prompt_id=delivering_prompt_id,
        )

    def _cancel_explicit_child(self) -> bool:
        return self.pending_prompt_queue.cancel_explicit_child()

    def _explicit_finish_report(self) -> dict[str, object] | None:
        return self.pending_prompt_queue.explicit_finish_report()

    def _terminal_transition(self) -> str | None:
        return self.pending_prompt_queue.terminal_transition()

    def _idle_nudge_eligible(
        self, persisted_notification_ids: frozenset[str]
    ) -> bool:
        return self.pending_prompt_queue.idle_nudge_eligible(
            persisted_notification_ids=persisted_notification_ids
        )


def _require_off_loop_lock() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise PendingPromptLoopThreadLockError(
        "pending-prompt store lock requested on an asyncio event-loop thread"
    )


class PendingPromptQueue:
    """Own every durable append, acknowledgement, close, and timeout decision."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = weakref.proxy(store)
        # Set by the terminal drain (off-loop, under the append lock) so a late
        # notification producer is refused and re-resolves to the live ancestor.
        self._drained = False
        self._run_ended = False

    @contextmanager
    def _lock(self, *, deadline: float | None = None) -> Iterator[None]:
        """Acquire the pending-store lock only from a blocking worker thread."""
        _require_off_loop_lock()
        with self._store._append_lock(deadline=deadline):
            yield

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
        starts_new_episode: bool = False,
    ) -> ConversationEntry:
        if type(text) is not str or not text.strip():
            raise ValueError("pending prompt text must be a nonempty string")
        if len(text) > MAX_PENDING_PROMPT_TEXT:
            raise ValueError("pending prompt text is too long")
        if type(starts_new_episode) is not bool:
            raise TypeError("starts_new_episode must be a bool")
        reload_started = time.monotonic()
        with self._lock(deadline=deadline):
            if (
                deadline is not None
                and reload_started < deadline <= time.monotonic()
            ):
                raise PendingPromptWriterReloadedError
            self._check_deadline(deadline)
            if self._run_ended:
                raise PendingPromptsClosedError("run ended")
            if self._queue_closed_unlocked() and (
                not starts_new_episode or self._terminal_transition_unlocked() is not None
            ):
                raise PendingPromptsClosedError("pending prompt queue is closed")
            entry = self._store._append_row_unlocked(
                "pending_prompt",
                {"text": text, "origin": origin.value},
                deadline=deadline,
            )
            return self._store._snapshot_entry(entry)

    def pending(self) -> list[ConversationEntry]:
        with self._lock():
            return self._pending_unlocked()

    def acknowledge(self, prompt_id: str) -> None:
        with self._lock():
            fold = self._store._delivery_fold
            if prompt_id not in fold.prompt_ids:
                raise ValueError(f"unknown pending prompt: {prompt_id}")
            if prompt_id in fold.active_prompts:
                self._store._append_row_unlocked(
                    "pending_prompt_ack", {"prompt_id": prompt_id}
                )

    def close_if_empty(self) -> list[ConversationEntry]:
        with self._lock():
            pending = self._pending_unlocked()
            if not pending and not self._queue_closed_unlocked():
                self._store._append_row_unlocked("pending_queue_closed", {})
            return pending

    def close(self) -> None:
        with self._lock():
            if not self._queue_closed_unlocked():
                self._store._append_row_unlocked("pending_queue_closed", {})

    def end_run(self) -> None:
        """Seal the queue so a prompt that meets an ended run is refused, not lost.

        The settle and segment_end lock scopes set this atomically for the common
        exits. The consumer's ``finally`` calls this once more, lock-free and
        best-effort, for the error and cancel exits.
        """
        self._run_ended = True

    def accept_explicit_finish(
        self,
        report: dict[str, object],
        *,
        persisted_notification_ids: frozenset[str] = frozenset(),
        delivering_prompt_id: str | None = None,
    ) -> str | None:
        """Atomically close admission and persist one accepted finish report."""
        with self._lock():
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
                for notification_id, entry in self._store._delivery_fold.active_notifications.items()
                if notification_id not in persisted_notification_ids
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
        with self._lock():
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
        with self._lock():
            return self.explicit_finish_report_snapshot()

    def explicit_finish_report_snapshot(self) -> dict[str, object] | None:
        """Read a report from state already loaded by this store's owner."""
        transition = self._terminal_transition_unlocked()
        if transition is None or transition.data["action"] != "finish":
            return None
        return dict(transition.data["report"])

    def idle_nudge_eligible(
        self,
        *,
        persisted_notification_ids: frozenset[str] = frozenset(),
    ) -> bool:
        """Atomically reject a nudge when durable input or termination exists."""
        with self._lock():
            return self._idle_nudge_eligible_unlocked(persisted_notification_ids)

    def terminal_transition(self) -> str | None:
        with self._lock():
            transition = self._terminal_transition_unlocked()
            return str(transition.data["action"]) if transition is not None else None

    def _idle_nudge_eligible_unlocked(
        self, persisted_notification_ids: frozenset[str]
    ) -> bool:
        if self._terminal_transition_unlocked() is not None or self._pending_unlocked():
            return False
        return not any(
            notification_id not in persisted_notification_ids
            and entry.data.get("display_only") is not True
            for notification_id, entry in self._store._delivery_fold.active_notifications.items()
        )

    def _terminal_transition_unlocked(self) -> ConversationEntry | None:
        # The fold holds the last terminal_transition on the branch; snapshot it
        # so a resident entry (and its report) never crosses to the event loop.
        transition = self._store._delivery_fold.last_terminal_transition
        return (
            self._store._snapshot_entry(transition)
            if transition is not None
            else None
        )

    def _queue_closed_unlocked(self) -> bool:
        return self._store._delivery_fold.queue_closed

    def _pending_unlocked(self) -> list[ConversationEntry]:
        return [
            self._store._snapshot_entry(entry)
            for entry in self._store._delivery_fold.active_prompts.values()
        ]
