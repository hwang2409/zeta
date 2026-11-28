"""One durable conversation channel between background workers and parents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..accounts.session import SessionAccountContext, account_provider
from ..accounts.status import AccountStatus, account_status_text
from ..core.agent_progress import publish_child_ended
from ..core.checkpoints import ConversationEntry
from ..core.store import MAX_AGENT_NOTIFICATION_TEXT, ConversationStore
from ..core.store._child_runs import child_run_segments
from ..core.store._step_delivery import TerminalTargetSealedError
from ..protocol.types import StreamEvent, StreamEventType
from ..providers.rate_limit_hold import caller_outwaited
from ..runtime.offload import await_joined_offload

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
    are the same hold; a response, turn end, or error ends it. Foreground children
    record their status but do not wake their blocked parent.
    """

    def __init__(
        self,
        channel: ConversationChannel | None = None,
        child_instance_id: str = "",
        child_session_path: str = "",
        description: str = "",
        accounts: Mapping[str, object] | SessionAccountContext | None = None,
        child_store: ConversationStore | None = None,
    ) -> None:
        self._channel = channel
        self._child = (child_instance_id, child_session_path, description)
        self._accounts = accounts or {}
        self._woken = False
        self._store = child_store
        self._status: AccountStatus | None = None

    async def start(self) -> None:
        await self._update_status(held=False)

    async def _update_status(self, *, held: bool) -> None:
        if self._store is None or not isinstance(self._accounts, SessionAccountContext):
            return
        statuses = await asyncio.to_thread(self._accounts.status_snapshot, held=held)
        if statuses:
            self._status = statuses[0]
        elif self._status is not None:
            self._status = {
                **self._status, "held": held, "usable_alternatives": 0,
                "cached_used_percent": None,
            }
        if self._status is not None:
            self._store.update_agent_lifecycle(account_status=self._status)

    def held_account_text(self) -> str | None:
        if self._status is not None and self._status["held"]:
            return account_status_text(self._status)
        return None

    async def observe(self, event: StreamEvent) -> bool:
        """Return whether ``event`` is a provider hold; wake on the first long one."""

        if event.type in _HOLD_ENDS:
            self._woken = False
            if self._status is not None and self._status["held"]:
                self._status = {**self._status, "held": False}
                assert self._store is not None
                self._store.update_agent_lifecycle(
                    account_status=self._status, current_step="provider hold ended"
                )
        if (
            event.type is not StreamEventType.RETRY
            or event.data.get("kind") != "provider_rate_limit_hold"
        ):
            return False
        hold = event.data
        if self._status is None or not self._status["held"]:
            await self._update_status(held=True)
        if self._channel is None or self._woken or not caller_outwaited(hold):
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
                status=self._status,
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
        self._legacy_pending_questions: dict[str, set[str]] = {}
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

    async def produce_to_live_parent(
        self,
        child_instance_id: str,
        build: Callable[[ConversationStore], DeliveryInput],
    ) -> tuple[ProducerResult, ConversationStore]:
        """Single owner for every child->parent publication.

        Every producer that sends a notice up the tree (idle, hold, question,
        completion, and the question-answer ack) goes through here, so one place
        re-resolves past a sealed store. A notice for a child whose parent has
        already settled meets the drained parent store and is refused;
        ``_live_parent`` steps over drained stores, so each retry moves strictly
        up, and root is never drained, so the bounded loop ends there at the
        latest. ``build`` receives the store the notice will land in, so the
        payload can reflect the ancestor actually chosen.
        """

        for _ in range(len(self._delivery_targets) + 1):
            store, parent_id = self._live_parent(child_instance_id)
            try:
                return await self.produce_to(parent_id, build(store)), store
            except TerminalTargetSealedError:
                continue
        raise TerminalTargetSealedError(child_instance_id)

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
        owner = self._owner.run_transition(child_instance_id)
        if owner is not None and owner.durable_questions:
            return bool(owner.pending_questions())
        return bool(self._legacy_pending_questions.get(child_instance_id))

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

        data: dict[str, Any] = {
            "origin": "notification",
            "kind": "child_idle",
            "child_instance_id": child_instance_id,
            "child_session_path": child_session_path,
            "description": description,
            "status": "child_idle",
            "text": text[-MAX_AGENT_NOTIFICATION_TEXT:],
        }
        result, _store = await self.produce_to_live_parent(
            child_instance_id, lambda _s: NotificationInput("child_idle", data)
        )
        return result.entries[0] if result.entries else None

    async def publish_note(
        self, *, child_instance_id: str, child_session_path: str, note: str
    ) -> None:
        """Send a no-reply child note as an idle notice; it never opens a question."""

        await self.publish_idle(
            child_instance_id=child_instance_id,
            child_session_path=child_session_path,
            description=self._owner.description(child_instance_id),
            text=(
                f"child {child_instance_id} notes (no reply needed; it keeps "
                f"working): {note}"
            ),
        )

    async def publish_hold(
        self,
        *,
        child_instance_id: str,
        child_session_path: str,
        description: str,
        hold: Mapping[str, Any],
        account: str | None,
        status: AccountStatus | None = None,
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
                + (f"{account_status_text(status)}. " if status is not None else "")
                + (
                    "This child waits and retries on its pinned account. Switching the parent "
                    "account affects new children, not this hold. "
                    if account is not None else "This child waits and retries. "
                )
                + "A hold on one child does not mean the provider is exhausted."
            ),
        }
        await self.produce_to_live_parent(
            child_instance_id, lambda _s: NotificationInput("child_hold", data)
        )

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

        parent_store, _parent_id = self._live_parent(child_instance_id)

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

        def build(store: ConversationStore) -> NotificationInput:
            def finish_parent_marker() -> bool:
                store._finish_agent_child_unlocked(marker_key)
                return True

            return NotificationInput(
                "child_completion",
                data_for(store, display_only=False),
                dedupe_key=child_instance_id,
                after_commit=finish_parent_marker,
            )

        # Mirror a display-only copy onto the root transcript when the effective
        # parent is not root, so a non-root recipient shows one card and the user
        # still sees the completion. A sealed retry that lands at root upgrades
        # this mirror to the model-visible row.
        root_mirror: ConversationEntry | None = None
        if self._root_store is not parent_store:
            root_result = await self.produce_to(
                None,
                NotificationInput(
                    "child_completion",
                    data_for(self._root_store, display_only=True),
                    dedupe_key=child_instance_id,
                ),
            )
            root_mirror = root_result.entries[0]
        result, final_store = await self.produce_to_live_parent(
            child_instance_id, build
        )
        # child_ended fires once, for the receipt actually created in the store
        # that finally accepted the completion (the effective parent after any
        # re-resolution past a sealed store), never for a refused attempt and
        # never for the root display-only mirror.
        if result.changed:
            publish_child_ended(
                child_instance_id, result.entries[0].seq, child_lifecycle
            )
        notification = (
            result.entries[0] if self._root_store is final_store else root_mirror
        )
        assert notification is not None
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
        existing: dict | None = None,
    ) -> ConversationStore:
        from .pending_delivery import NotificationInput

        def build(store: ConversationStore) -> NotificationInput:
            parent_id = self._target_for_store(store)
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
            return NotificationInput("child_question", data, dedupe_key=question_id)

        owner = self._owner.run_transition(child_instance_id)
        if owner is not None and owner.durable_questions and existing is not None:
            _result, parent_store = await self.produce_to_live_parent(child_instance_id, build)
            await self._record_question_target(owner, existing, parent_store)
        elif owner is not None and owner.durable_questions:
            fact = {
                "question_id": question_id, "question": question, "options": options or [],
                "state": "pending", "held_at_turn_limit": False, "ancestor_target_ids": [],
            }
            async with owner._lock:
                await owner.write_questions(fact)
                _result, parent_store = await self.produce_to_live_parent(child_instance_id, build)
                await self._record_question_target(owner, fact, parent_store)
        else:
            _result, parent_store = await self.produce_to_live_parent(child_instance_id, build)
            self._legacy_pending_questions.setdefault(child_instance_id, set()).add(question_id)
        return parent_store

    async def drain_terminal(self, dead_parent: ConversationStore) -> int:
        """Seal a dead child's store and reroute its late descendant input.

        One store command seals the store and returns the pending notifications
        that must reroute, so nothing new can land in the dead store between the
        seal and the acks below. A still-running grandchild's question/idle/hold
        is re-published to the next live ancestor; a grandchild completion that
        landed after its parent settled is re-published to that ancestor
        (upgrading the root mirror) and acked. Each reroute moves to the next
        live ancestor; idle/hold and completions are acked in the dead store (a
        question is acked by its reroute).
        """

        from .pending_delivery import NotificationInput

        rerouted = await dead_parent.drain_step_delivery_terminal()
        # Resident flag for _live_parent; the writer flag set in the command is
        # the authority at commit time.
        dead_parent.pending_prompt_queue._drained = True
        moved = 0
        dead_parent_id = self._target_for_store(dead_parent)
        settled_ids: list[str] = []
        for entry in rerouted:
            kind = entry.data.get("kind")
            child_id = entry.data.get("child_instance_id")
            if type(child_id) is not str:
                continue
            if kind == "agent_completion":
                await self._reroute_completion(entry, child_id)
                settled_ids.append(entry.id)
                moved += 1
                continue
            if not self._owner.owns_running(child_id):
                continue
            if kind == "child_question":
                data = dict(entry.data)
                data["routing_note"] = (
                    "direct parent is no longer alive; routed to the next live ancestor"
                )
                def build(_store, data=data):
                    return NotificationInput("child_question", data, dedupe_key=str(data["question_id"]))

                owner = self._owner.run_transition(child_id)
                if owner is not None and owner.durable_questions:
                    async with owner._lock:
                        question = next((q for q in owner.questions() if q["question_id"] == data["question_id"]), None)
                        if question is not None and question["state"] == "pending":
                            _, target = await self.produce_to_live_parent(child_id, build)
                            await self._record_question_target(owner, question, target)
                        await owner.reconcile_questions()
                else:
                    await self.produce_to_live_parent(child_id, build)
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
            else:
                # Re-publish the idle/hold with its payload unchanged: both
                # validators are strict, so adding a routing note would be
                # rejected. Then ack it in the dead store.
                await self.produce_to_live_parent(
                    child_id,
                    lambda _store, k=kind, e=entry: NotificationInput(k, dict(e.data)),
                )
                settled_ids.append(entry.id)
            moved += 1
        if settled_ids:
            await dead_parent.acknowledge_staged_delivery(None, tuple(settled_ids))
        return moved

    async def _reroute_completion(
        self, entry: ConversationEntry, child_instance_id: str
    ) -> None:
        """Re-publish a grandchild completion to the dead parent's live ancestor.

        The completion landed in the dead store during window C. Re-publish it to
        the live ancestor and finish that ancestor's marker for the grandchild
        (its key is the grandchild's instance id in the transferred marker).
        """

        from .pending_delivery import NotificationInput

        data = dict(entry.data)

        def build(store: ConversationStore) -> NotificationInput:
            def finish_marker() -> bool:
                for key, marker in store.agent_children().items():
                    if marker.get("child_instance_id") == child_instance_id:
                        store._finish_agent_child_unlocked(key)
                        break
                return True

            return NotificationInput(
                "child_completion",
                data,
                dedupe_key=child_instance_id,
                after_commit=finish_marker,
            )

        await self.produce_to_live_parent(child_instance_id, build)

    async def _record_question_target(self, owner, question: dict, store: ConversationStore) -> None:
        target_id = self._target_for_store(store)
        targets = question["ancestor_target_ids"]
        if target_id not in targets:
            await owner.write_questions({**question, "ancestor_target_ids": [*targets, target_id]})

    async def reconcile_questions(self, child_id: str, questions: tuple[dict, ...]) -> None:
        from .pending_delivery import NotificationInput

        for q in questions:
            if q["state"] == "pending":
                if not q["ancestor_target_ids"]:
                    await self.publish_question(
                        child_instance_id=child_id, question_id=q["question_id"],
                        question=q["question"], options=q["options"] or None,
                        existing=q,
                    )
                continue
            item = NotificationInput(
                "question_answer_ack" if q["state"] == "answered" else "child_question_close",
                {"child_instance_id": child_id, "question_id": q["question_id"], **(
                    {"reason": "canceled"} if q["state"] == "withdrawn" else {}
                )},
            )
            _, live_store = await self.produce_to_live_parent(child_id, lambda _store, item=item: item)
            live_id = self._target_for_store(live_store)
            for target_id in q["ancestor_target_ids"]:
                if target_id == live_id:
                    continue
                if target_id in self._delivery_targets:
                    await self.produce_to(target_id, item)
                else:
                    await await_joined_offload(self._settle_inactive_question, child_id, target_id, item)

    def _settle_inactive_question(self, child_id: str, target_id: str, item: DeliveryInput) -> None:
        root = self._root_store
        segments = child_run_segments(root.session_id, target_id)
        child = child_run_segments(root.session_id, child_id)
        if len(segments) >= len(child) or child[:len(segments)] != segments:
            raise ValueError("question target is not an ancestor")
        path = root.session_dir
        for segment in segments:
            path = path / "agents" / str(segment)
        if not path.is_dir():
            return
        with ConversationStore(path.parent, session_id=path.name, _storage_lease=root._storage_lease) as store:
            store._commit_delivery_input(item)

    def question_answered(self, child_instance_id: str, question_id: str) -> None:
        questions = self._legacy_pending_questions.get(child_instance_id)
        if questions is None:
            return
        questions.discard(question_id)
        if not questions:
            self._legacy_pending_questions.pop(child_instance_id, None)

    async def close_questions(self, child_instance_id: str, *, reason: str) -> int:
        from .pending_delivery import NotificationInput

        self._legacy_pending_questions.pop(child_instance_id, None)
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
        while parent_id is not None and (
            parent_id not in self._wakes or store.pending_prompt_queue._drained
        ):
            if parent_id in visited:
                return self._root_store, None
            visited.add(parent_id)
            store, parent_id = self._owner.parent_edge(parent_id, self._root_store)
        return store, parent_id
