"""Runtime-lease ownership for child-tree recovery."""

from __future__ import annotations

import asyncio
import threading
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .pending_delivery import NotificationInput

from ..core.session_execution import SessionOwnedError
from ..core.store import ConversationStore
from ..protocol.types import ToolResult


class AgentLoopForRecovery(Protocol):
    store: ConversationStore
    _background_owner: Any
    _children_recovered: bool
    _pending_child_recovery: tuple[NotificationInput, ...]

    def _canceled_agent_result(
        self, tool_call_id: str, **kwargs: Any
    ) -> ToolResult: ...

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None: ...


class _RecoveryOwner:
    def __init__(self, notification_store: ConversationStore) -> None:
        self.notification_store = notification_store


class _DetachedRecoveryLoop:
    def __init__(
        self,
        source: AgentLoopForRecovery,
        store: ConversationStore,
        notification_store: ConversationStore,
    ) -> None:
        self.store = store
        self._background_owner = _RecoveryOwner(notification_store)
        self._source = source

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None:
        return self.store.tool_result(tool_call_id)

    def _canceled_agent_result(self, tool_call_id: str, **kwargs: Any) -> ToolResult:
        return self._source._canceled_agent_result(tool_call_id, **kwargs)


def open_child_store(store: ConversationStore, path: Path) -> ConversationStore | None:
    try:
        parts = path.relative_to(store.session_dir).parts
    except ValueError:
        return None
    if (
        len(parts) < 2
        or len(parts) % 2
        or any(parts[index] != "agents" for index in range(0, len(parts), 2))
        or any(not parts[index].isdigit() for index in range(1, len(parts), 2))
    ):
        return None
    return ConversationStore(
        path.parent,
        session_id=path.name,
        cwd=store.cwd,
        _storage_lease=store._storage_lease,
        _owner_surface="recovery",
        _recovery_owner=True,
    )


def acquire_recovery_subtree(
    parent_store: ConversationStore,
    child_path: Path,
    leases: ExitStack,
    recovery_stores: dict[Path, ConversationStore],
) -> bool:
    """Lease a complete child subtree before any recovery mutation."""

    try:
        child_store = open_child_store(parent_store, child_path)
    except SessionOwnedError:
        return False
    if child_store is None:
        return True
    leases.enter_context(child_store)
    if not leases.enter_context(child_store.orphan_recovery_lease()):
        return False
    recovery_stores[child_path] = child_store
    return all(
        acquire_recovery_subtree(
            child_store,
            Path(marker["child_session_path"]),
            leases,
            recovery_stores,
        )
        for marker in child_store.agent_children().values()
    )


def bootstrap_agent_children(loop: AgentLoopForRecovery) -> None:
    """Repair restart state before the opened store reaches the event loop."""

    from .background import recover_agent_children

    deliveries = recover_agent_children(loop)
    for item in deliveries:
        loop.store._commit_delivery_input(item)
    loop._pending_child_recovery = deliveries
    loop._children_recovered = True


async def publish_recovered_children(loop: AgentLoopForRecovery) -> None:
    async def publish() -> None:
        for item in loop._pending_child_recovery:
            await loop._background_owner.conversation_channel.produce_to(None, item)
        loop._pending_child_recovery = ()

    publication = asyncio.create_task(publish())
    try:
        await asyncio.shield(publication)
    except asyncio.CancelledError:
        await publication
        raise


def _has_recovery_work(loop: AgentLoopForRecovery) -> bool:
    return bool(loop.store.agent_children()) or any(
        entry.data.get("kind") == "child_question"
        for entry in loop.store.agent_notifications()
    )


async def activate_agent_children(loop: AgentLoopForRecovery) -> None:
    if not loop._children_recovered:
        if _has_recovery_work(loop):
            await recover_agent_children_off_loop(loop)
        loop._children_recovered = True
    if loop._pending_child_recovery:
        await publish_recovered_children(loop)


async def recover_agent_children_off_loop(loop: AgentLoopForRecovery) -> None:
    """Repair orphaned child trees without lease or close work on the loop."""

    root = loop.store
    notification = loop._background_owner.notification_store

    stop_recovery = threading.Event()

    def recover() -> tuple[NotificationInput, ...]:
        from .background import _recover_agent_children

        with ConversationStore(
            root.root_dir,
            session_id=root.session_id,
            cwd=root.cwd,
            _storage_lease=root._storage_lease,
            _execution_lease=root.execution_lease,
        ) as worker_root:
            if notification is root:
                return _recover_agent_children(
                    _DetachedRecoveryLoop(loop, worker_root, worker_root),
                    should_stop=stop_recovery.is_set,
                )
            with ConversationStore(
                notification.root_dir,
                session_id=notification.session_id,
                cwd=notification.cwd,
                _storage_lease=notification._storage_lease,
                _execution_lease=notification.execution_lease,
            ) as worker_notification:
                return _recover_agent_children(
                    _DetachedRecoveryLoop(loop, worker_root, worker_notification),
                    should_stop=stop_recovery.is_set,
                )

    async def complete_recovery() -> None:
        deliveries = await root._to_thread_store(recover)
        for item in deliveries:
            await loop._background_owner.conversation_channel.produce_to(None, item)
        await root._refresh_off_loop()
        if notification is not root:
            await notification._refresh_off_loop()

    recovery = asyncio.create_task(complete_recovery())
    try:
        await asyncio.shield(recovery)
    except asyncio.CancelledError:
        stop_recovery.set()
        await recovery
        raise
