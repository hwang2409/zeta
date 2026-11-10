"""One durable conversation channel between background workers and parents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..accounts.session import account_provider
from ..core.agent_progress import publish_child_ended
from ..core.checkpoints import ConversationEntry
from ..core.store import MAX_AGENT_NOTIFICATION_TEXT, ConversationStore
from ..protocol.types import StreamEvent, StreamEventType
from ..providers.rate_limit_hold import LONG_HOLD_WAKE_SECONDS

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop
    from .background_owner import BackgroundAgentOwner
    from .pending_delivery import DeliveryInput, ProducerResult, StepDeliveryFlow


def _append_recovery_notification(
    store: ConversationStore,
    data: dict[str, object],
    *,
    display_only: bool,
) -> tuple[ConversationEntry, bool]:
    report = (
        {"status": data["report_status"]}
        | {
            key: data[key]
            for key in ("summary", "head_sha", "verdict")
            if key in data
        }
        if "report_status" in data
        else None
    )
    return store.append_agent_notification_if_absent(
        str(data["child_instance_id"]),
        child_session_path=str(data["child_session_path"]),
        description=str(data["description"]),
        status=str(data["status"]),
        text=str(data["text"]),
        stats=data.get("stats") if isinstance(data.get("stats"), dict) else None,
        killed_task_ids=(
            data.get("killed_task_ids")
            if isinstance(data.get("killed_task_ids"), list)
            else None
        ),
        killed_task_count=(
            data.get("killed_task_count")
            if isinstance(data.get("killed_task_count"), int)
            else None
        ),
        killed_task_ids_truncated=data.get("killed_task_ids_truncated") is True,
        display_only=display_only,
        report=report,
    )


def recover_completion_mirrors(
    root_store: ConversationStore,
    parent_store: ConversationStore,
    child_instance_id: str,
    fallback_data: dict[str, object] | None = None,
    child_lifecycle: object = None,
) -> ConversationEntry | None:
    """Durably restore each intended completion recipient before cleanup.

    A newly created immediate-parent receipt publishes ``child_ended`` once; an
    existing receipt or the root display mirror never does.
    """

    root_notification = root_store.sync_agent_completion_notification(
        child_instance_id
    )
    parent_notification = (
        root_notification
        if parent_store is root_store
        else parent_store.sync_agent_completion_notification(child_instance_id)
    )
    source = parent_notification or root_notification
    if source is None and fallback_data is None:
        return None
    data = dict(source.data) if source is not None else fallback_data
    assert data is not None
    if root_notification is None:
        root_notification, created = _append_recovery_notification(
            root_store,
            data,
            display_only=parent_store is not root_store,
        )
        if created and parent_store is root_store:
            publish_child_ended(child_instance_id, root_notification.seq, child_lifecycle)
    if parent_notification is None:
        parent_notification, created = _append_recovery_notification(
            parent_store,
            data,
            display_only=False,
        )
        if created:
            publish_child_ended(
                child_instance_id, parent_notification.seq, child_lifecycle
            )
    return root_notification


@dataclass(slots=True)
class _DeliveryTarget:
    loop: asyncio.AbstractEventLoop
    flow: StepDeliveryFlow
    accepted: set[asyncio.Future[Any]] = field(default_factory=set)


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


class HoldWake:
    """Wake the parent once per long provider hold of one background child.

    A long hold is a stall with a known end. Re-probes that re-enter the hold
    are the same hold; a response, turn end, or error ends it. A foreground
    parent is blocked on the agent call, so foreground children get the inert
    ``NO_HOLD_WAKE``, which only recognizes hold events.
    """

    def __init__(
        self,
        channel: ConversationChannel | None = None,
        child_instance_id: str = "",
        child_session_path: str = "",
        description: str = "",
        accounts: Mapping[str, object] | None = None,
    ) -> None:
        self._channel = channel
        self._child = (child_instance_id, child_session_path, description)
        self._accounts = accounts or {}
        self._woken = False

    async def observe(self, event: StreamEvent) -> bool:
        """Return whether ``event`` is a provider hold; wake on the first long one."""

        if event.type in _HOLD_ENDS:
            self._woken = False
        if (
            event.type is not StreamEventType.RETRY
            or event.data.get("kind") != "provider_rate_limit_hold"
        ):
            return False
        hold = event.data
        if (
            self._channel is None
            or self._woken
            or hold["retry_after"] <= LONG_HOLD_WAKE_SECONDS
        ):
            return True
        self._woken = True
        # a subscription hold names its account; a rate limit uses the child's pin
        accounts = (
            self._accounts.snapshot()
            if hasattr(self._accounts, "snapshot")
            else self._accounts
        )
        pinned = accounts.get(account_provider(str(hold["provider"])) or "")
        child_instance_id, child_session_path, description = self._child
        # no live parent flow (the tree is closing) must not end the held child
        with suppress(LookupError):
            await self._channel.publish_hold(
                child_instance_id=child_instance_id,
                child_session_path=child_session_path,
                description=description,
                hold=hold,
                account=hold.get("account") or (None if pinned is None else str(pinned)),
            )
        return True


_HOLD_ENDS = frozenset(
    {StreamEventType.MESSAGE_START, StreamEventType.TURN_END, StreamEventType.ERROR}
)
NO_HOLD_WAKE = HoldWake()


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
        self._pending_questions: dict[str, set[str]] = {}
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
        return await self._produce_to_target(target, item)

    async def _produce_to_target(
        self, target: _DeliveryTarget, item: DeliveryInput, /
    ) -> ProducerResult:
        if target.loop is asyncio.get_running_loop():
            result = await target.flow.produce(item)
        else:
            concurrent = asyncio.run_coroutine_threadsafe(
                target.flow.produce(item), target.loop
            )
            future = asyncio.wrap_future(concurrent)
            target.accepted.add(future)
            try:
                result = await future
            finally:
                target.accepted.discard(future)
        if result.changed:
            self._owner.causal_episodes.associate(entry.id for entry in result.entries)
        return result

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

    def has_pending_question(self, child_instance_id: str) -> bool:
        return bool(self._pending_questions.get(child_instance_id))

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

        _parent_store, parent_id = self._live_parent(child_instance_id)
        data: dict[str, Any] = {
            "origin": "notification",
            "kind": "child_idle",
            "child_instance_id": child_instance_id,
            "child_session_path": child_session_path,
            "description": description,
            "status": "child_idle",
            "text": text[-MAX_AGENT_NOTIFICATION_TEXT:],
        }
        result = await self.produce_to(
            parent_id,
            NotificationInput("child_idle", data),
        )
        return result.entries[0] if result.entries else None

    async def publish_hold(
        self,
        *,
        child_instance_id: str,
        child_session_path: str,
        description: str,
        hold: Mapping[str, Any],
        account: str | None,
    ) -> None:
        """Wake the parent once: a long provider hold is a stall with a known end."""

        from .pending_delivery import NotificationInput

        subscription = hold.get("limit") == "subscription"
        until = datetime.fromtimestamp(hold["until"], UTC).isoformat()
        cause = "subscription limit" if subscription else "provider rate limit"
        observed_at = datetime.now(UTC).isoformat()
        data: dict[str, Any] = {
            "origin": "notification",
            "kind": "child_hold",
            "child_instance_id": child_instance_id,
            "child_session_path": child_session_path,
            "description": description,
            "status": "held",
            "reason": "subscription_limit" if subscription else "rate_limit",
            "until": until,
            "provider": str(hold["provider"]),
            "account": account,
            "retry_after": float(hold["retry_after"]),
            "text": (
                f"child {child_instance_id} is held by a {cause} on "
                f"{account or hold['provider']} until {until} (as of {observed_at}); "
                "a running child resumes by itself; cancel and restart it on another model or "
                "account if you cannot wait."
            ),
        }
        _parent_store, parent_id = self._live_parent(child_instance_id)
        await self.produce_to(parent_id, NotificationInput("child_hold", data))

    async def publish_not_delivered(self, child_loop: AgentLoop, text: str) -> None:
        """Show input a stop held back on the root transcript; it wakes no agent."""

        from .pending_delivery import NotificationInput

        store = child_loop.store
        child_instance_id = child_loop.agent_instance_id or ""
        lifecycle = store.agent_lifecycle() or {}
        data: dict[str, Any] = {
            "origin": "notification",
            "kind": "child_idle",
            "child_instance_id": child_instance_id,
            "child_session_path": str(store.session_dir),
            "description": str(lifecycle.get("description") or child_instance_id),
            "status": "child_idle",
            "text": text[-MAX_AGENT_NOTIFICATION_TEXT:],
            # display-only: a waking notice would re-trigger the stop it reports.
            "display_only": True,
        }
        result = await self.produce_to(None, NotificationInput("child_idle", data))
        for entry in result.entries:
            child_loop._publish_background_event(
                StreamEvent(
                    StreamEventType.AGENT_NOTIFICATION,
                    data={"notification_id": entry.id, **entry.data},
                )
            )

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
        child_lifecycle: object = None,
    ) -> ConversationEntry:
        """Persist a child completion through each resident recipient flow.

        Only the immediate parent's newly installed receipt publishes ``child_ended``;
        the root display mirror never does.
        """

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
            if result.changed:
                publish_child_ended(child_instance_id, notification.seq, child_lifecycle)
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
            if result.changed:
                publish_child_ended(
                    child_instance_id, result.entries[0].seq, child_lifecycle
                )
            notification = root_result.entries[0]
        self._owner.causal_episodes.associate_member(
            child_instance_id, (entry.id for entry in result.entries)
        )
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
        routed = parent_id != self._owner.original_parent_instance_id(child_instance_id)
        data: dict[str, Any] = {
            "origin": "notification",
            "kind": "child_question",
            "child_instance_id": child_instance_id,
            "question_id": question_id,
            "question": question,
            "description": "child question",
            "agent_description": self._owner.description(child_instance_id),
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
        self._pending_questions.setdefault(child_instance_id, set()).add(question_id)
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

    def question_answered(self, child_instance_id: str, question_id: str) -> None:
        questions = self._pending_questions.get(child_instance_id)
        if questions is None:
            return
        questions.discard(question_id)
        if not questions:
            self._pending_questions.pop(child_instance_id, None)

    async def close_questions(self, child_instance_id: str, *, reason: str) -> int:
        from .pending_delivery import NotificationInput

        self._pending_questions.pop(child_instance_id, None)
        item = NotificationInput(
            "child_question_close",
            {"child_instance_id": child_instance_id, "reason": reason},
        )
        targets_by_store = {
            id(target.flow.store): target for target in self._delivery_targets.values()
        }
        targets = tuple(
            target
            for store in self._owner.conversation_stores()
            if (target := targets_by_store.get(id(store))) is not None
        )
        reservations = tuple(
            (target, asyncio.get_running_loop().create_future()) for target in targets
        )
        for target, reservation in reservations:
            target.accepted.add(reservation)
        try:
            results = [
                await self._produce_to_target(target, item) for target in targets
            ]
        finally:
            for target, reservation in reservations:
                target.accepted.discard(reservation)
                if not reservation.done():
                    reservation.set_result(None)
        return sum(result.affected for result in results)

    def _target_for_store(self, store: ConversationStore) -> str | None:
        for target_id, target in self._delivery_targets.items():
            if target.flow.store is store:
                return target_id
        raise LookupError(f"no live delivery flow for store {store.session_id!r}")

    def _live_parent(
        self, child_instance_id: str
    ) -> tuple[ConversationStore, str | None]:
        store, parent_id = self._owner.parent_edge(child_instance_id, self._root_store)
        visited = {child_instance_id}
        while parent_id is not None and parent_id not in self._wakes:
            if parent_id in visited:
                return self._root_store, None
            visited.add(parent_id)
            store, parent_id = self._owner.parent_edge(parent_id, self._root_store)
        return store, parent_id
