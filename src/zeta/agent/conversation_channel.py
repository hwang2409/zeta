"""One durable conversation channel between background workers and parents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.checkpoints import ConversationEntry
from ..core.store import MAX_AGENT_NOTIFICATION_TEXT, ConversationStore

if TYPE_CHECKING:
    from .background import BackgroundAgentOwner
    from .pending_delivery import DeliveryInput, ProducerResult, StepDeliveryFlow


@dataclass(slots=True)
class _DeliveryTarget:
    loop: asyncio.AbstractEventLoop
    flow: StepDeliveryFlow
    accepted: set[asyncio.Future[ProducerResult]] = field(default_factory=set)


def has_follow_up_loop(*, accepts_follow_ups: bool, background: bool) -> bool:
    """Return whether a child has a live turn-boundary follow-up loop."""

    return accepts_follow_ups and background


def recover_orphaned_questions(
    store: ConversationStore, *, excluded_child_subtrees: set[str] | None = None
) -> int:
    """Withdraw persisted questions whose child did not survive restart."""

    excluded_child_subtrees = excluded_child_subtrees or set()
    child_ids = {
        child_id
        for entry in store.agent_notifications()
        if entry.data.get("kind") == "child_question"
        and type(child_id := entry.data.get("child_instance_id")) is str
        and not any(
            child_id == root or child_id.startswith(f"{root}:")
            for root in excluded_child_subtrees
        )
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
        self._delivery_targets: dict[str | None, _DeliveryTarget] = {}
        self._root_wake: Callable[[], None] | None = None

    def set_root_wake(self, callback: Callable[[], None] | None) -> None:
        self._root_wake = callback

    def register_loop(self, instance_id: str) -> None:
        self._wakes[instance_id] = (asyncio.get_running_loop(), asyncio.Event())

    def unregister_loop(self, instance_id: str) -> None:
        self._wakes.pop(instance_id, None)

    def register_delivery_flow(
        self,
        instance_id: str | None,
        flow: StepDeliveryFlow,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        if instance_id in self._delivery_targets:
            raise RuntimeError(f"delivery flow already registered: {instance_id!r}")
        self._delivery_targets[instance_id] = _DeliveryTarget(
            loop or asyncio.get_running_loop(), flow
        )

    async def unregister_delivery_flow(self, instance_id: str | None) -> None:
        target = self._delivery_targets.pop(instance_id, None)
        if target is not None and target.accepted:
            await asyncio.gather(*tuple(target.accepted), return_exceptions=True)

    async def produce_to(
        self, target_instance_id: str | None, item: DeliveryInput, /
    ) -> ProducerResult:
        target = self._delivery_targets.get(target_instance_id)
        if target is None:
            raise LookupError(f"no live delivery flow: {target_instance_id!r}")
        if target.loop is asyncio.get_running_loop():
            return await target.flow.produce(item)
        concurrent = asyncio.run_coroutine_threadsafe(
            target.flow.produce(item), target.loop
        )
        future = asyncio.wrap_future(concurrent)
        target.accepted.add(future)
        try:
            return await future
        finally:
            target.accepted.discard(future)

    async def produce_to_parent_of(
        self, child_instance_id: str, item: DeliveryInput, /
    ) -> ProducerResult:
        _store, parent_id = self._live_parent(child_instance_id)
        return await self.produce_to(parent_id, item)

    async def wait(self, instance_id: str) -> None:
        _loop, event = self._wakes[instance_id]
        await event.wait()
        event.clear()

    def clear_wake(self, instance_id: str) -> None:
        _loop, event = self._wakes[instance_id]
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

    async def publish_idle(
        self,
        *,
        child_instance_id: str,
        child_session_path: str,
        description: str,
        text: str,
    ) -> ConversationEntry | None:
        """Persist one nonterminal idle notice through recipient flows."""

        from .pending_delivery import NotificationInput

        parent_store, parent_id = self._live_parent(child_instance_id)
        question_id = next(
            (
                entry.data.get("question_id")
                for entry in reversed(parent_store.agent_notifications())
                if entry.data.get("kind") == "child_question"
                and entry.data.get("child_instance_id") == child_instance_id
            ),
            None,
        )

        def data_for(*, display_only: bool) -> dict[str, Any]:
            data: dict[str, Any] = {
                "origin": "notification",
                "kind": "child_idle",
                "child_instance_id": child_instance_id,
                "child_session_path": child_session_path,
                "description": description,
                "status": "child_idle",
                "text": text[-MAX_AGENT_NOTIFICATION_TEXT:],
            }
            if isinstance(question_id, str):
                data["ask_parent_question_id"] = question_id
            if display_only:
                data["display_only"] = True
            return data

        if self._root_store is parent_store:
            result = await self.produce_to(
                parent_id,
                NotificationInput("child_idle", data_for(display_only=False)),
            )
            return result.entries[0] if result.entries else None
        root_result = await self.produce_to(
            None,
            NotificationInput("child_idle", data_for(display_only=True)),
        )
        await self.produce_to(
            parent_id,
            NotificationInput("child_idle", data_for(display_only=False)),
        )
        return root_result.entries[0] if root_result.entries else None

    async def publish_completion(
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
        report: dict[str, object] | None = None,
    ) -> ConversationEntry:
        """Persist a child completion through each resident recipient flow."""

        from .pending_delivery import NotificationInput

        parent_store, parent_id = self._live_parent(child_instance_id)

        def data_for(store: ConversationStore, *, display_only: bool) -> dict[str, Any]:
            return store._agent_notification_data(
                child_instance_id,
                child_session_path=child_session_path,
                description=description,
                status=status,
                text=text,
                stats=stats,
                killed_task_ids=killed_task_ids,
                killed_task_count=killed_task_count,
                killed_task_ids_truncated=killed_task_ids_truncated,
                display_only=display_only,
                report=report,
            )

        def finish_parent_marker() -> bool:
            parent_store._finish_agent_child_unlocked(marker_key)
            return True

        if self._root_store is parent_store:
            result = await self.produce_to(
                parent_id,
                NotificationInput(
                    "child_completion",
                    data_for(parent_store, display_only=False),
                    dedupe_key=child_instance_id,
                    after_commit=finish_parent_marker,
                ),
            )
            notification = result.entries[0]
        else:
            root_result = await self.produce_to(
                None,
                NotificationInput(
                    "child_completion",
                    data_for(self._root_store, display_only=True),
                    dedupe_key=child_instance_id,
                ),
            )
            result = await self.produce_to(
                parent_id,
                NotificationInput(
                    "child_completion",
                    data_for(parent_store, display_only=False),
                    dedupe_key=child_instance_id,
                    after_commit=finish_parent_marker,
                ),
            )
            notification = root_result.entries[0]
        return notification

    async def publish_question(
        self,
        *,
        child_instance_id: str,
        question_id: str,
        question: str,
        options: list[str] | None,
    ) -> ConversationStore:
        from .pending_delivery import NotificationInput

        parent_store, parent_id = self._live_parent(child_instance_id)
        routed = parent_id != self._owner.original_parent_instance_id(
            child_instance_id
        )
        data: dict[str, Any] = {
            "origin": "notification",
            "kind": "child_question",
            "child_instance_id": child_instance_id,
            "question_id": question_id,
            "question": question,
            "description": "child question",
            "status": "pending",
            "text": (
                f"child {child_instance_id} asks: {question}\n"
                f"Answer with agent_send to {child_instance_id}; you can reference "
                f"question id {question_id} in the message."
            ),
        }
        if options is not None:
            data["options"] = list(options)
        if routed:
            data["routing_note"] = (
                "direct parent is no longer alive; routed to the next live ancestor"
            )
        await self.produce_to(
            parent_id,
            NotificationInput("child_question", data, dedupe_key=question_id),
        )
        return parent_store

    async def reroute_questions_from(self, dead_parent: ConversationStore) -> int:
        """Move unanswered descendant questions to their next live ancestor."""

        from .pending_delivery import NotificationInput

        moved = 0
        dead_parent_id = self._target_for_store(dead_parent)
        for entry in dead_parent.agent_notifications():
            if entry.data.get("kind") != "child_question":
                continue
            child_id = entry.data.get("child_instance_id")
            if type(child_id) is not str or not self._owner.owns_running(child_id):
                continue
            target, parent_id = self._live_parent(child_id)
            if target is dead_parent:
                continue
            data = dict(entry.data)
            data["routing_note"] = (
                "direct parent is no longer alive; routed to the next live ancestor"
            )
            await self.produce_to(
                parent_id,
                NotificationInput(
                    "child_question", data, dedupe_key=str(data["question_id"])
                ),
            )
            await self.produce_to(
                dead_parent_id,
                NotificationInput(
                    "question_answer_ack",
                    {
                        "child_instance_id": child_id,
                        "question_id": str(data["question_id"]),
                    },
                ),
            )
            moved += 1
        return moved

    async def close_questions(self, child_instance_id: str, *, reason: str) -> int:
        from .pending_delivery import NotificationInput

        closed = 0
        for store in self._owner.conversation_stores():
            target_id = self._target_for_store(store)
            result = await self.produce_to(
                target_id,
                NotificationInput(
                    "child_question_close",
                    {"child_instance_id": child_instance_id, "reason": reason},
                ),
            )
            closed += result.affected
        return closed

    def _target_for_store(self, store: ConversationStore) -> str | None:
        for target_id, target in self._delivery_targets.items():
            if target.flow.store is store:
                return target_id
        raise LookupError(f"no live delivery flow for store {store.session_id!r}")

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
