"""One durable conversation channel between background workers and parents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..core.store import ConversationStore

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
        self._wakes: dict[str, asyncio.Event] = {}
        self._root_wake: Callable[[], None] | None = None

    def set_root_wake(self, callback: Callable[[], None] | None) -> None:
        self._root_wake = callback

    def register_loop(self, instance_id: str) -> None:
        self._wakes[instance_id] = asyncio.Event()

    def unregister_loop(self, instance_id: str) -> None:
        self._wakes.pop(instance_id, None)

    async def wait(self, instance_id: str) -> None:
        event = self._wakes[instance_id]
        await event.wait()
        event.clear()

    def wake(self, instance_id: str | None) -> None:
        if instance_id is None:
            if self._root_wake is not None:
                self._root_wake()
            return
        event = self._wakes.get(instance_id)
        if event is not None:
            event.set()

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

    def notify_child_event(self, child_instance_id: str) -> None:
        _store, parent_id = self._live_parent(child_instance_id)
        self.wake(parent_id)

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
