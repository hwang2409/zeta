"""Shared lifecycle helpers for background agent children."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

from ..core.store import ConversationStore
from .conversation_channel import ConversationChannel
from .finish_gate import CausalEpisodeRegistry

_AGENT_CANCEL_CONFIRM_TIMEOUT = 1.0


@dataclass(frozen=True)
class CancellationRequest:
    """Durable cancellation confirmation and its eventual cleanup."""

    confirmation: asyncio.Task[bool]
    cleanup: asyncio.Task[None]


@dataclass(frozen=True)
class CancellationOutcome:
    """Result of waiting briefly for durable cancellation ownership."""

    status: Literal["confirmed", "unconfirmed", "failed", "already_finished"]
    error: str | None = None


@dataclass(frozen=True)
class _CancellationConfirmation:
    instance_id: str
    task: asyncio.Task[bool] | asyncio.Task[None]

class BackgroundAgentOwner:
    """Own cancellation and watcher state for one complete agent tree."""

    def __init__(
        self,
        notification_store: ConversationStore,
        *,
        finish_gate_max_turns: int = 8,
        finish_gate_timeout: float = 600,
        finish_gate_task_timeout: float = 3600,
    ) -> None:
        if type(finish_gate_max_turns) is not int or finish_gate_max_turns < 1:
            raise ValueError("finish_gate_max_turns must be a positive integer")
        if (
            not isinstance(finish_gate_timeout, (int, float))
            or isinstance(finish_gate_timeout, bool)
            or not finish_gate_timeout > 0
        ):
            raise ValueError("finish_gate_timeout must be positive")
        if (
            not isinstance(finish_gate_task_timeout, (int, float))
            or isinstance(finish_gate_task_timeout, bool)
            or not finish_gate_task_timeout > 0
        ):
            raise ValueError("finish_gate_task_timeout must be positive")
        self.notification_store = notification_store
        self.finish_gate_max_turns = finish_gate_max_turns
        self.finish_gate_timeout = float(finish_gate_timeout)
        # This internal bound lets tasks outlive a turn but stops abandoned tasks
        # that never exit; it is not a user setting.
        self.finish_gate_task_timeout = float(finish_gate_task_timeout)
        # Receipts and adopted descendants can outlive their immediate loop.
        # The tree owns child leases until root shutdown joins all watchers.
        self.store_leases = ExitStack()
        self._stores: list[ConversationStore] = []
        self._pending_stores: set[ConversationStore] = set()
        self._store_close_tasks: set[asyncio.Task[Any]] = set()
        self._store_close_stores: dict[asyncio.Task[Any], ConversationStore] = {}
        self._observed_close_failures: set[asyncio.Task[Any]] = set()
        self._cancellation_tasks: set[asyncio.Task[Any]] = set()
        self._cancellers: dict[
            str, Callable[[], CancellationRequest | asyncio.Task[None] | None]
        ] = {}
        self._watchers: dict[str, asyncio.Task[Any]] = {}
        self._parent_stores: dict[str, ConversationStore] = {}
        self._active_stores: dict[str, tuple[ConversationStore, ...]] = {}
        self.project_context_loops: dict[str, Any] = {}
        self._descriptions: dict[str, str] = {}
        self._canceling = False
        self._cancel_requested: set[str] = set()
        self._cancel_confirmations: dict[
            str, asyncio.Task[bool] | asyncio.Task[None]
        ] = {}
        self._parent_ids: dict[str, str | None] = {}
        self._original_parent_ids: dict[str, str | None] = {}
        self._wake_callback: Callable[[], None] | None = None
        self.causal_episodes = CausalEpisodeRegistry()
        self.conversation_channel = ConversationChannel(self, notification_store)

    def _notify_frontend(self) -> None:
        if self._wake_callback is not None:
            self._wake_callback()

    def set_wake_callback(self, callback: Callable[[], None] | None) -> None:
        self._wake_callback = callback
        self.conversation_channel.set_root_wake(callback)

    def notify_wake(self) -> None:
        """Wake the frontend after a durable completion notification."""
        self._notify_frontend()

    def create_cancellation_task[TaskResult](
        self, awaitable: Awaitable[TaskResult]
    ) -> asyncio.Task[TaskResult]:
        """Keep cancellation work alive for the lifetime of the agent tree."""

        task = asyncio.create_task(awaitable)
        self._cancellation_tasks.add(task)
        task.add_done_callback(self._cancellation_tasks.discard)
        return task

    def cancel(self, instance_id: str) -> bool:
        """Request cancellation of one child and its owned descendants."""
        if instance_id not in self._cancellers:
            return False
        self.cancel_subtree(instance_id)
        return True

    async def cancel_and_wait(self, instance_id: str) -> CancellationOutcome:
        """Request cancellation and briefly await durable confirmation."""
        if instance_id not in self._cancellers:
            return CancellationOutcome("already_finished")
        confirmations = self.cancel_subtree(instance_id)
        if not confirmations:
            return CancellationOutcome("confirmed")
        done, pending = await asyncio.wait(
            [confirmation.task for confirmation in confirmations],
            timeout=_AGENT_CANCEL_CONFIRM_TIMEOUT,
        )
        by_task = {confirmation.task: confirmation for confirmation in confirmations}
        error: str | None = None
        unconfirmed = bool(pending)
        for confirmation in confirmations:
            if confirmation.task not in done:
                continue
            try:
                confirmed = confirmation.task.result()
            except asyncio.CancelledError:
                error = error or "cancellation confirmation was canceled"
                confirmed = False
            except Exception as exc:  # noqa: BLE001 - confirmation failures are data
                error = error or str(exc)
                confirmed = False
            if confirmed is False:
                self._cancel_requested.discard(confirmation.instance_id)
                if (
                    self._cancel_confirmations.get(confirmation.instance_id)
                    is confirmation.task
                ):
                    self._cancel_confirmations.pop(confirmation.instance_id, None)
                unconfirmed = True
        for task in pending:
            self._cancel_requested.discard(by_task[task].instance_id)
        if error is not None:
            return CancellationOutcome("failed", error)
        if unconfirmed:
            return CancellationOutcome("unconfirmed")
        return CancellationOutcome("confirmed")

    def _subtree_ids(self, instance_id: str) -> set[str]:
        selected = {instance_id}
        changed = True
        while changed:
            changed = False
            for child, parent in self._parent_ids.items():
                if child not in selected and parent in selected:
                    selected.add(child)
                    changed = True
        return selected

    def _request_cancellation(
        self, instance_id: str
    ) -> _CancellationConfirmation | None:
        confirmation = self._cancel_confirmations.get(instance_id)
        if instance_id not in self._cancel_requested:
            self._cancel_requested.add(instance_id)
            if confirmation is None:
                result = self._cancellers[instance_id]()
                if isinstance(result, CancellationRequest):
                    confirmation = result.confirmation
                elif isinstance(result, asyncio.Task):
                    confirmation = result
                if confirmation is not None:
                    self._cancel_confirmations[instance_id] = confirmation
        if confirmation is None:
            return None
        return _CancellationConfirmation(instance_id, confirmation)

    def cancel_subtree(
        self, instance_id: str
    ) -> tuple[_CancellationConfirmation, ...]:
        selected = self._subtree_ids(instance_id)
        return tuple(
            confirmation
            for child in tuple(self._cancellers)
            if child in selected
            for confirmation in (self._request_cancellation(child),)
            if confirmation is not None
        )

    async def wait_for_descendant_watchers(self, instance_id: str) -> None:
        """Keep the parent's routing alive through descendant publication."""

        descendants = self._subtree_ids(instance_id)
        watchers = tuple(
            watcher
            for child, watcher in self._watchers.items()
            if child != instance_id and child in descendants
        )
        await asyncio.gather(*watchers, return_exceptions=True)

    def owns_running(self, instance_id: str) -> bool:
        return instance_id in self._cancellers

    def original_parent_instance_id(self, instance_id: str) -> str | None:
        return self._original_parent_ids.get(instance_id)

    def description(self, instance_id: str) -> str:
        return self._descriptions.get(instance_id, "background task")

    def conversation_stores(self) -> tuple[ConversationStore, ...]:
        stores = [self.notification_store, *self._stores, *self._parent_stores.values()]
        return tuple(dict.fromkeys(stores))

    def track_store(self, store: ConversationStore) -> None:
        """Track a child store so completed trees can release its directory fd."""

        self._stores.append(store)

    def finish_store(self, instance_id: str, store: ConversationStore) -> None:
        """Schedule a finished child store while its watcher remains owned."""

        self._active_stores.pop(instance_id, None)
        self.mark_store_finished(store)
        self.release_unused_stores()

    def mark_store_finished(self, store: ConversationStore) -> None:
        self._pending_stores.add(store)

    def release_unused_stores(self) -> None:
        """Close stores no longer needed by an active adopted descendant.

        The owner ExitStack remains the final safety net for shutdown. Normally,
        however, completed children must release their descriptor immediately;
        a nested descendant keeps its ancestor store pinned until adoption or
        that descendant's own completion.
        """

        referenced = set(self._parent_stores.values())
        for stores in self._active_stores.values():
            referenced.update(stores)
        for store in self._stores:
            if store not in self._pending_stores or store in referenced:
                continue
            existing = next(
                (
                    task
                    for task, task_store in self._store_close_stores.items()
                    if task_store is store
                ),
                None,
            )
            if existing is not None:
                # Retry only a failure that wait_for_store_closes already raised;
                # an unobserved failure must still reach its waiter.
                if existing not in self._observed_close_failures:
                    continue
                self._observed_close_failures.discard(existing)
                self._store_close_tasks.discard(existing)
                self._store_close_stores.pop(existing, None)
            close_task = asyncio.create_task(
                store._to_thread_store(store.close),
                name=f"close child store {store.session_id}",
            )
            self._store_close_tasks.add(close_task)
            self._store_close_stores[close_task] = store

    def register(
        self,
        instance_id: str,
        cancel: Callable[[], CancellationRequest | asyncio.Task[None] | None],
        watcher: asyncio.Task[Any],
        parent_store: ConversationStore | None = None,
        description: str | None = None,
        active_store: ConversationStore | None = None,
        parent_instance_id: str | None = None,
    ) -> None:
        self._cancellers[instance_id] = cancel
        self._watchers[instance_id] = watcher
        if parent_store is not None:
            self._parent_stores[instance_id] = parent_store
        self._parent_ids[instance_id] = parent_instance_id
        self._original_parent_ids[instance_id] = parent_instance_id
        self._active_stores[instance_id] = tuple(
            store for store in (active_store, parent_store) if store is not None
        )
        if description is not None:
            self._descriptions[instance_id] = description

    def unregister(self, instance_id: str) -> None:
        self._cancellers.pop(instance_id, None)
        self._watchers.pop(instance_id, None)
        self._parent_stores.pop(instance_id, None)
        self._active_stores.pop(instance_id, None)
        self._descriptions.pop(instance_id, None)
        self._parent_ids.pop(instance_id, None)
        self._original_parent_ids.pop(instance_id, None)
        self._cancel_requested.discard(instance_id)
        self._cancel_confirmations.pop(instance_id, None)
        self.causal_episodes.release_member(instance_id)

    def adopt(
        self,
        instance_id: str,
        parent_store: ConversationStore,
        parent_instance_id: str | None = None,
    ) -> None:
        if instance_id in self._watchers:
            self._parent_stores[instance_id] = parent_store
            self._parent_ids[instance_id] = parent_instance_id
    def controls_child(self, parent: ConversationStore, child: str) -> bool:
        watcher = self._watchers.get(child)
        return bool(
            watcher and not watcher.done() and self._parent_stores.get(child) is parent
        )
    def parent_edge(
        self,
        instance_id: str,
        default_store: ConversationStore,
        default_instance_id: str | None = None,
    ) -> tuple[ConversationStore, str | None]:
        return (
            self._parent_stores.get(instance_id, default_store),
            self._parent_ids.get(instance_id, default_instance_id),
        )

    def parent_store(self, instance_id: str, default: ConversationStore) -> ConversationStore:
        return self.parent_edge(instance_id, default)[0]

    def cancel_all(self) -> None:
        if self._canceling:
            return
        self._canceling = True
        try:
            for instance_id in tuple(self._cancellers):
                self._request_cancellation(instance_id)
        finally:
            self._canceling = False

    @property
    def running(self) -> bool:
        return bool(self._cancellers)

    @property
    def active_descriptions(self) -> tuple[str, ...]:
        return tuple(self._descriptions.values())

    @property
    def active_instance_ids(self) -> frozenset[str]:
        return frozenset(self._cancellers)

    def owned_running(self, parent_instance_id: str) -> tuple[tuple[str, str], ...]:
        return tuple(
            (instance_id, self._descriptions.get(instance_id, "background task"))
            for instance_id in self._cancellers
            if self._parent_ids.get(instance_id) == parent_instance_id
        )

    async def wait_for_store_closes(self) -> None:
        """Wait until all scheduled child-store lease releases finish."""

        while self._store_close_tasks:
            close_tasks = tuple(self._store_close_tasks)
            try:
                # Shield: close tasks are shared, so one canceled waiter must not
                # cancel the release that other waiters are gathering.
                await asyncio.gather(*map(asyncio.shield, close_tasks))
            except Exception:
                self._observed_close_failures.update(
                    task
                    for task in close_tasks
                    if task.done() and not task.cancelled() and task.exception() is not None
                )
                raise
            for task in close_tasks:
                store = self._store_close_stores.pop(task, None)
                self._store_close_tasks.discard(task)
                if store is not None:
                    self._pending_stores.discard(store)
                    self._stores.remove(store)

    async def wait(self) -> None:
        current = asyncio.current_task()
        while True:
            for instance_id, watcher in tuple(self._watchers.items()):
                if watcher.done():
                    self.unregister(instance_id)
            watchers = tuple(
                watcher for watcher in self._watchers.values() if watcher is not current
            )
            if not watchers:
                await self.wait_for_store_closes()
                return
            await asyncio.gather(*watchers, return_exceptions=True)
