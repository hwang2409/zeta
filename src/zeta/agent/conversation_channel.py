"""One durable conversation channel between background workers and parents."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.checkpoints import ConversationEntry
from ..core.store import (
    ConversationStore,
    PendingPromptCommitTimeoutError,
    PendingPromptsClosedError,
)
from ..protocol.types import MessageOrigin

if TYPE_CHECKING:
    from .background import BackgroundAgentOwner


def has_follow_up_loop(*, accepts_follow_ups: bool, background: bool) -> bool:
    """Return whether a child has a live turn-boundary follow-up loop."""

    return accepts_follow_ups and background


def recover_orphaned_questions(store: ConversationStore) -> int:
    """Withdraw persisted questions whose child did not survive restart."""

    child_ids = {
        child_id
        for entry in store.agent_notifications()
        if entry.data.get("kind") == "child_question"
        and type(child_id := entry.data.get("child_instance_id")) is str
    }
    return sum(
        store.close_child_questions(child_id, reason="canceled")
        for child_id in child_ids
    )


class ConversationChannel:
    """Own follow-up eligibility, question routing, and child-loop wakes."""

    def __init__(
        self,
        owner: BackgroundAgentOwner,
        root_store: ConversationStore,
    ) -> None:
        self._owner = owner
        self._root_store = root_store
        self._wakes: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}
        self._root_wake: Callable[[], None] | None = None

    def set_root_wake(self, callback: Callable[[], None] | None) -> None:
        self._root_wake = callback

    def register_loop(self, instance_id: str) -> None:
        self._wakes[instance_id] = (asyncio.get_running_loop(), asyncio.Event())

    def unregister_loop(self, instance_id: str) -> None:
        self._wakes.pop(instance_id, None)

    async def wait(self, instance_id: str) -> None:
        _loop, event = self._wakes[instance_id]
        await event.wait()
        event.clear()

    def wake(self, instance_id: str | None) -> None:
        if instance_id is None:
            if self._root_wake is not None:
                self._root_wake()
            return
        target = self._wakes.get(instance_id)
        if target is not None:
            loop, event = target
            loop.call_soon_threadsafe(event.set)

    def publish_follow_up(
        self,
        parent_store: ConversationStore,
        child_instance_id: object,
        message: object,
        *,
        commit_timeout: float,
    ) -> str | None:
        """Durably queue one eligible follow-up, then wake its child loop."""

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
            return (
                f"agent_send rejected for {agent_type} child {child_instance_id!r}: "
                "it does not accept follow-ups; reviewers are one-shot; start a "
                "fresh reviewer"
            )
        deadline = time.monotonic() + commit_timeout
        try:
            child_path = Path(str(marker["child_session_path"]))
            with ConversationStore(
                child_path.parent,
                session_id=child_path.name,
                cwd=parent_store.cwd,
                _lock_deadline=deadline,
            ) as child_store:
                child_store.pending_prompt_queue.append(
                    message,
                    origin=MessageOrigin.AGENT_SEND,
                    deadline=deadline,
                )
        except PendingPromptCommitTimeoutError:
            return "pending prompt commit timed out before the queue could be changed"
        except PendingPromptsClosedError:
            return no_live_run
        self.wake(child_instance_id)
        return None

    def publish_completion(
        self,
        *,
        child_instance_id: str,
        marker_key: str,
        child_session_path: str,
        description: str,
        status: str,
        text: str,
        stats: dict[str, Any] | None,
        killed_task_ids: list[str] | None,
        killed_task_count: int | None,
        killed_task_ids_truncated: bool,
    ) -> ConversationEntry:
        """Persist a child completion for its recipients, then wake its parent."""

        parent_store, parent_id = self._live_parent(child_instance_id)
        notification = self._root_store.append_agent_notification(
            child_instance_id,
            child_session_path=child_session_path,
            description=description,
            status=status,
            text=text,
            stats=stats,
            killed_task_ids=killed_task_ids,
            killed_task_count=killed_task_count,
            killed_task_ids_truncated=killed_task_ids_truncated,
        )
        if self._root_store is not parent_store:
            parent_store.append_agent_notification(
                child_instance_id,
                child_session_path=child_session_path,
                description=description,
                status=status,
                text=text,
                stats=stats,
                killed_task_ids=killed_task_ids,
                killed_task_count=killed_task_count,
                killed_task_ids_truncated=killed_task_ids_truncated,
            )
        parent_store.finish_agent_child(marker_key)
        self.wake(parent_id)
        return notification

    def publish_question(
        self,
        *,
        child_instance_id: str,
        question_id: str,
        question: str,
        options: list[str] | None,
    ) -> ConversationStore:
        parent_store, parent_id = self._live_parent(child_instance_id)
        routed = parent_id != self._owner.original_parent_instance_id(
            child_instance_id
        )
        parent_store.append_child_question(
            child_instance_id=child_instance_id,
            question_id=question_id,
            question=question,
            options=options,
            routed_from_parent=routed,
        )
        self.wake(parent_id)
        return parent_store

    def reroute_questions_from(self, dead_parent: ConversationStore) -> int:
        """Move unanswered descendant questions to their next live ancestor."""

        moved = 0
        for entry in dead_parent.agent_notifications():
            if entry.data.get("kind") != "child_question":
                continue
            child_id = entry.data.get("child_instance_id")
            if type(child_id) is not str or not self._owner.owns_running(child_id):
                continue
            target, parent_id = self._live_parent(child_id)
            if target is dead_parent:
                continue
            question_id = str(entry.data["question_id"])
            exists = any(
                item.data.get("kind") == "child_question"
                and item.data.get("question_id") == question_id
                for item in target.agent_notifications(pending_only=False)
            )
            if not exists:
                options = entry.data.get("options")
                target.append_child_question(
                    child_instance_id=child_id,
                    question_id=question_id,
                    question=str(entry.data["question"]),
                    options=options if type(options) is list else None,
                    routed_from_parent=True,
                )
            dead_parent.acknowledge_agent_notification(entry.id)
            self.wake(parent_id)
            moved += 1
        return moved

    def close_questions(self, child_instance_id: str, *, reason: str) -> int:
        closed = 0
        for store in self._owner.conversation_stores():
            closed += store.close_child_questions(child_instance_id, reason=reason)
        return closed

    def _live_parent(
        self, child_instance_id: str
    ) -> tuple[ConversationStore, str | None]:
        store, parent_id = self._owner.parent_edge(
            child_instance_id, self._root_store
        )
        visited = {child_instance_id}
        while parent_id is not None and parent_id not in self._wakes:
            if parent_id in visited:
                return self._root_store, None
            visited.add(parent_id)
            store, parent_id = self._owner.parent_edge(parent_id, self._root_store)
        return store, parent_id
