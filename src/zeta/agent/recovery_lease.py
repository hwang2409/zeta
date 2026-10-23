"""Runtime-lease ownership for child-tree recovery."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .pending_delivery import NotificationInput

from ..core.store import ConversationStore
from ..protocol.types import ToolResult


class AgentLoopForRecovery(Protocol):
    store: ConversationStore
    _background_owner: Any

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


def open_child_store(
    store: ConversationStore, path: Path
) -> ConversationStore | None:
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
    return ConversationStore(path.parent, session_id=path.name, cwd=store.cwd)


def acquire_recovery_subtree(
    parent_store: ConversationStore,
    child_path: Path,
    leases: ExitStack,
    recovery_stores: dict[Path, ConversationStore],
) -> bool:
    """Lease a complete child subtree before any recovery mutation."""

    child_store = open_child_store(parent_store, child_path)
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


async def recover_agent_children_off_loop(loop: AgentLoopForRecovery) -> None:
    """Repair orphaned child trees without lease or close work on the loop."""

    root = loop.store
    notification = loop._background_owner.notification_store

    def recover() -> tuple[NotificationInput, ...]:
        from .background import recover_agent_children

        with ConversationStore(
            root.root_dir, session_id=root.session_id, cwd=root.cwd
        ) as worker_root:
            if notification is root:
                return recover_agent_children(
                    _DetachedRecoveryLoop(loop, worker_root, worker_root)
                )
            with ConversationStore(
                notification.root_dir,
                session_id=notification.session_id,
                cwd=notification.cwd,
            ) as worker_notification:
                return recover_agent_children(
                    _DetachedRecoveryLoop(loop, worker_root, worker_notification)
                )

    deliveries = await root._to_thread_store(recover)
    for item in deliveries:
        await loop._background_owner.conversation_channel.produce_to(None, item)
    root.refresh()
    if notification is not root:
        notification.refresh()
