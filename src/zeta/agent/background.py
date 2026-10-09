"""Shared lifecycle helpers for background agent children."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Protocol

from ..core.store import ConversationEntry, ConversationStore
from ..protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
)
from .conversation_channel import ConversationChannel, recover_orphaned_questions
from .receipt import (
    TerminalState,
    agent_stats,
    build_agent_receipt,
    receipt_tool_result,
)


class _AgentLoopForRecovery(Protocol):
    store: ConversationStore
    _background_owner: BackgroundAgentOwner

    def _canceled_agent_result(
        self,
        tool_call_id: str,
        *,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
    ) -> ToolResult: ...

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None: ...


class BackgroundAgentOwner:
    """Own cancellation and watcher state for one complete agent tree."""

    def __init__(
        self,
        notification_store: ConversationStore,
        *,
        finish_gate_max_turns: int = 8,
        finish_gate_timeout: float = 600,
    ) -> None:
        if type(finish_gate_max_turns) is not int or finish_gate_max_turns < 1:
            raise ValueError("finish_gate_max_turns must be a positive integer")
        if (
            not isinstance(finish_gate_timeout, (int, float))
            or isinstance(finish_gate_timeout, bool)
            or not finish_gate_timeout > 0
        ):
            raise ValueError("finish_gate_timeout must be positive")
        self.notification_store = notification_store
        self.finish_gate_max_turns = finish_gate_max_turns
        self.finish_gate_timeout = float(finish_gate_timeout)
        # Receipts and adopted descendants can outlive their immediate loop.
        # The tree owns child leases until root shutdown joins all watchers.
        self.store_leases = ExitStack()
        self._stores: list[ConversationStore] = []
        self._pending_stores: set[ConversationStore] = set()
        self._cancellers: dict[str, Callable[[], None]] = {}
        self._watchers: dict[str, asyncio.Task[Any]] = {}
        self._parent_stores: dict[str, ConversationStore] = {}
        self._active_stores: dict[str, tuple[ConversationStore, ...]] = {}
        self._descriptions: dict[str, str] = {}
        self._canceling = False
        self._cancel_requested: set[str] = set()
        self._parent_ids: dict[str, str | None] = {}
        self._original_parent_ids: dict[str, str | None] = {}
        self._wake_callback: Callable[[], None] | None = None
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

    def cancel(self, instance_id: str) -> bool:
        """Request cancellation of one child and its owned descendants."""
        if instance_id not in self._cancellers:
            return False
        self.cancel_subtree(instance_id)
        return True

    def cancel_subtree(self, instance_id: str) -> None:
        selected = {instance_id}
        changed = True
        while changed:
            changed = False
            for child, parent in self._parent_ids.items():
                if child not in selected and parent in selected:
                    selected.add(child)
                    changed = True
        for child in tuple(self._cancellers):
            if child in selected and child not in self._cancel_requested:
                self._cancel_requested.add(child)
                self._cancellers[child]()

    def owns_running(self, instance_id: str) -> bool:
        return instance_id in self._cancellers

    def original_parent_instance_id(self, instance_id: str) -> str | None:
        return self._original_parent_ids.get(instance_id)

    def conversation_stores(self) -> tuple[ConversationStore, ...]:
        stores = [self.notification_store, *self._stores, *self._parent_stores.values()]
        return tuple(dict.fromkeys(stores))

    def track_store(self, store: ConversationStore) -> None:
        """Track a child store so completed trees can release its directory fd."""

        self._stores.append(store)

    def mark_store_finished(self, store: ConversationStore) -> None:
        self._pending_stores.add(store)

    def release_unused_stores(self) -> None:
        """Close stores no longer needed by an active adopted descendant.

        The owner ExitStack remains the final safety net for shutdown. Normally,
        however, completed children must release their descriptor immediately;
        a nested descendant keeps its ancestor store pinned until adoption or
        that descendant's own completion.
        """

        retained: list[ConversationStore] = []
        referenced = set(self._parent_stores.values())
        for stores in self._active_stores.values():
            referenced.update(stores)
        for store in self._stores:
            if store not in self._pending_stores or store in referenced:
                retained.append(store)
            else:
                store.close()
                self._pending_stores.discard(store)
        self._stores = retained

    def register(
        self,
        instance_id: str,
        cancel: Callable[[], None],
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

    def adopt(
        self,
        instance_id: str,
        parent_store: ConversationStore,
        parent_instance_id: str | None = None,
    ) -> None:
        if instance_id in self._watchers:
            self._parent_stores[instance_id] = parent_store
            self._parent_ids[instance_id] = parent_instance_id

    def parent_edge(
        self,
        instance_id: str,
        default_store: ConversationStore,
        default_instance_id: str | None = None,
    ) -> tuple[ConversationStore, str | None]:
        """Read the child's current owner edge before it is unregistered."""
        return (
            self._parent_stores.get(instance_id, default_store),
            self._parent_ids.get(instance_id, default_instance_id),
        )

    def parent_store(
        self, instance_id: str, default: ConversationStore
    ) -> ConversationStore:
        return self.parent_edge(instance_id, default)[0]

    def cancel_all(self) -> None:
        if self._canceling:
            return
        self._canceling = True
        try:
            for instance_id in tuple(self._cancellers):
                self.cancel_subtree(instance_id)
        finally:
            self._canceling = False

    @property
    def running(self) -> bool:
        return bool(self._cancellers)

    @property
    def active_descriptions(self) -> tuple[str, ...]:
        return tuple(self._descriptions.values())

    def owned_running(self, parent_instance_id: str) -> tuple[tuple[str, str], ...]:
        """Return running children directly owned by one agent."""

        return tuple(
            (instance_id, self._descriptions.get(instance_id, "background task"))
            for instance_id in self._cancellers
            if self._parent_ids.get(instance_id) == parent_instance_id
        )

    async def wait_for_owned_completion(
        self, parent_instance_id: str, timeout: float
    ) -> bool:
        """Idle until one directly owned child finishes or the bound expires."""

        watchers = tuple(
            watcher
            for instance_id, watcher in self._watchers.items()
            if self._parent_ids.get(instance_id) == parent_instance_id
        )
        if not watchers:
            return True
        done, _ = await asyncio.wait(
            watchers, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        return bool(done)

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
                return
            await asyncio.gather(*watchers, return_exceptions=True)


def _open_child_store(store: ConversationStore, path: Path) -> ConversationStore | None:
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


def adopt_agent_children(
    child_store: ConversationStore,
    parent_store: ConversationStore,
    *,
    background_owner: BackgroundAgentOwner | None = None,
    parent_instance_id: str | None = None,
) -> None:
    """Move unfinished descendants into the surviving parent session."""

    for marker_key, marker in child_store.agent_children().items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        adopted_key = marker.get("child_instance_id", marker_key)
        if type(adopted_key) is not str:
            adopted_key = marker_key
        parent_store.register_agent_child(
            tool_call,
            child_session_path=marker["child_session_path"],
            description=marker["description"],
            agent_type=marker.get("agent_type"),
            background=marker.get("background", False),
            child_instance_id=marker.get("child_instance_id"),
            accepts_follow_ups=marker.get("accepts_follow_ups", False) is True,
        )
        turns_used = marker.get("turns_used", 0)
        if turns_used:
            parent_store.update_agent_child_turns(adopted_key, turns_used)
        child_instance_id = marker.get("child_instance_id")
        if background_owner is not None and type(child_instance_id) is str:
            background_owner.adopt(child_instance_id, parent_store, parent_instance_id)
        child_store.finish_agent_child(marker_key)


def _nested_canceled_result(marker: dict[str, object]) -> ToolResult:
    structured_content: dict[str, object] = {
        "turns_used": marker.get("turns_used", 0),
        "child_session_path": marker["child_session_path"],
    }
    if type(marker.get("agent_type")) is str:
        structured_content["agent_type"] = marker["agent_type"]
    child_instance_id = marker.get("child_instance_id")
    if type(child_instance_id) is str:
        structured_content["child_instance_id"] = child_instance_id
    result = build_agent_receipt(
        "canceled",
        "tool execution canceled",
        agent_stats(
            None,
            status="canceled",
            turns_used=marker.get("turns_used", 0)
            if type(marker.get("turns_used", 0)) is int
            else 0,
        ),
        structured_content=structured_content,
    )
    return receipt_tool_result(ToolCall.from_dict(marker["tool_call"]).id, result)


def _lifecycle_tool_result(
    tool_call: ToolCall, child_store: ConversationStore, child_path: Path
) -> ToolResult | None:
    lifecycle = child_store.agent_lifecycle()
    if lifecycle is None or lifecycle.get("finished_at") is None:
        return None
    state = lifecycle.get("state")
    final_result = lifecycle.get("final_result")
    if type(state) is not str or type(final_result) is not str or not final_result:
        return None
    structured: dict[str, object] = {
        "turns_used": lifecycle.get("turns_used", 0),
        "child_session_path": str(child_path),
        "status": state,
    }
    for key in ("agent_type", "handle", "depth"):
        if key in lifecycle:
            structured["child_instance_id" if key == "handle" else key] = lifecycle[key]
    state_value: TerminalState = (
        "completed"
        if state == "completed"
        else "canceled"
        if state == "canceled"
        else "failed"
    )
    if lifecycle.get("final_result_is_receipt") is True:
        return ToolResult(
            tool_call.id,
            final_result,
            is_error=state_value == "failed",
            structured_content=structured,
            is_canceled=state_value == "canceled",
        )
    result = build_agent_receipt(
        state_value,
        final_result,
        agent_stats(
            lifecycle,
            status=state_value,
            turns_used=structured["turns_used"]
            if type(structured["turns_used"]) is int
            else 0,
        ),
        structured_content=structured,
    )
    return receipt_tool_result(tool_call.id, result)


def _existing_tool_result(store: ConversationStore, tool_call_id: str) -> bool:
    return any(
        message.tool_result is not None
        and message.tool_result.tool_call_id == tool_call_id
        for message in store.messages()
    )


def _agent_notification_index(
    store: ConversationStore,
) -> dict[str, ConversationEntry]:
    """Index completion notifications with one active-branch scan."""

    return store.agent_completion_notifications_by_child()


def _sync_agent_notification(
    store: ConversationStore,
    notification_index: dict[str, ConversationEntry],
    child_instance_id: str,
) -> ConversationEntry | None:
    """Sync an appended tail and update one indexed active-branch result."""

    notification = store.sync_agent_completion_notification(child_instance_id)
    if notification is None:
        notification_index.pop(child_instance_id, None)
    else:
        notification_index[child_instance_id] = notification
    return notification


def _recover_nested_children(
    store: ConversationStore,
    notification_store: ConversationStore,
    notification_index: dict[str, ConversationEntry],
) -> None:
    """Cancel descendants left behind when an ancestor session exits."""

    child_markers = store.agent_children()
    if not child_markers:
        return
    store_notification_index = (
        notification_index
        if store is notification_store
        else _agent_notification_index(store)
    )
    for marker_key, marker in child_markers.items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        child_store = _open_child_store(store, child_path)
        with child_store if child_store is not None else nullcontext():
            if child_store is not None:
                _recover_nested_children(
                    child_store, notification_store, notification_index
                )
            existing_result = _existing_tool_result(store, tool_call.id)
            notification_status: str | None = None
            notification_text: str | None = None
            notification_stats: dict[str, object] | None = None
            if marker.get("background"):
                child_instance_id = marker.get(
                    "child_instance_id", f"{store.session_id}:{child_path.name}"
                )
                terminal = child_store.agent_lifecycle() if child_store else None
                reason = (
                    "finished"
                    if terminal and terminal.get("state") == "completed"
                    else "canceled"
                )
                store.close_child_questions(child_instance_id, reason=reason)
                if notification_store is not store:
                    notification_store.close_child_questions(
                        child_instance_id, reason=reason
                    )
                existing = _sync_agent_notification(
                    notification_store,
                    notification_index,
                    child_instance_id,
                )
                if notification_store is not store:
                    store_notification = _sync_agent_notification(
                        store,
                        store_notification_index,
                        child_instance_id,
                    )
                    if existing is None:
                        existing = store_notification
                if existing is None:
                    lifecycle = (
                        child_store.agent_lifecycle()
                        if child_store is not None
                        else None
                    )
                    lifecycle_state = lifecycle.get("state") if lifecycle else None
                    lifecycle_text = (
                        lifecycle.get("final_result") if lifecycle else None
                    )
                    notification_status = {
                        "completed": "completed",
                        "failed": "error",
                        "canceled": "canceled",
                    }.get(lifecycle_state, "canceled")
                    notification_text = (
                        lifecycle_text
                        if type(lifecycle_text) is str and lifecycle_text
                        else "background child canceled when the parent session exited"
                    )
                    notification_state: TerminalState = (
                        "completed"
                        if notification_status == "completed"
                        else "canceled"
                        if notification_status == "canceled"
                        else "failed"
                    )
                    notification_stats = agent_stats(
                        lifecycle,
                        status=notification_status,
                        turns_used=marker.get("turns_used", 0),
                    )
                    notification_result = build_agent_receipt(
                        notification_state,
                        notification_text,
                        notification_stats,
                    )
                    notification_text = notification_result["content"][0]["text"]
                    appended, created = (
                        notification_store.append_agent_notification_if_absent(
                            child_instance_id,
                            child_session_path=str(child_path),
                            description=marker["description"],
                            status=notification_status,
                            text=notification_text,
                            stats=notification_stats,
                        )
                    )
                    notification_index[child_instance_id] = appended
                    if not created:
                        notification_status = appended.data["status"]
                        notification_text = appended.data["text"]
                        existing_stats = appended.data.get("stats")
                        notification_stats = (
                            existing_stats if type(existing_stats) is dict else None
                        )
                else:
                    notification_status = existing.data["status"]
                    notification_text = existing.data["text"]
                    existing_stats = existing.data.get("stats")
                    if type(existing_stats) is dict:
                        notification_stats = existing_stats
                if (
                    notification_store is not store
                    and child_instance_id not in store_notification_index
                ):
                    appended, _ = store.append_agent_notification_if_absent(
                        child_instance_id,
                        child_session_path=str(child_path),
                        description=marker["description"],
                        status=notification_status,
                        text=notification_text,
                        stats=notification_stats,
                    )
                    store_notification_index[child_instance_id] = appended
                    notification_status = appended.data["status"]
                    notification_text = appended.data["text"]
                    appended_stats = appended.data.get("stats")
                    notification_stats = (
                        appended_stats if type(appended_stats) is dict else None
                    )
            elif not existing_result:
                lifecycle_result = (
                    _lifecycle_tool_result(tool_call, child_store, child_path)
                    if child_store is not None
                    else None
                )
                if lifecycle_result is not None:
                    store.append_message(
                        Message(
                            MessageRole.TOOL_RESULT,
                            [TextContent(lifecycle_result.content)],
                            tool_result=lifecycle_result,
                        )
                    )
                    child_store.finish_agent_parent()
                    store.finish_agent_child(marker_key)
                    continue
                store.append_message(
                    Message(
                        MessageRole.TOOL_RESULT,
                        [TextContent("tool execution canceled")],
                        tool_result=_nested_canceled_result(marker),
                    )
                )
            if child_store is not None:
                if marker.get("background") and notification_status == "canceled":
                    child_store.mark_agent_canceled(tool_call.id)
                elif existing_result or notification_status is not None:
                    if child_store.agent_lifecycle() is not None:
                        child_store.finish_agent_lifecycle(
                            "completed" if notification_status != "error" else "failed",
                            final_result=(
                                notification_text or "background child completed"
                            ),
                        )
                    child_store.finish_agent_parent()
                else:
                    child_store.mark_agent_canceled(tool_call.id)
            store.finish_agent_child(marker_key)


def recover_agent_children(loop: _AgentLoopForRecovery) -> None:
    """Resolve child markers left by a process exit before resuming."""

    child_markers = loop.store.agent_children()
    if not child_markers:
        recover_orphaned_questions(loop.store)
        return
    notification_store = loop._background_owner.notification_store
    notification_index = _agent_notification_index(notification_store)
    store_notification_index = (
        notification_index
        if notification_store is loop.store
        else _agent_notification_index(loop.store)
    )
    for marker_key, marker in child_markers.items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        agents_root = loop.store.session_dir / "agents"
        child_store = _open_child_store(loop.store, child_path)
        with child_store if child_store is not None else nullcontext():
            if child_store is not None:
                _recover_nested_children(
                    child_store,
                    notification_store,
                    notification_index,
                )
            if marker.get("background"):
                child_instance_id = marker.get(
                    "child_instance_id", f"{loop.store.session_id}:{child_path.name}"
                )
                terminal = child_store.agent_lifecycle() if child_store else None
                reason = (
                    "finished"
                    if terminal and terminal.get("state") == "completed"
                    else "canceled"
                )
                loop.store.close_child_questions(child_instance_id, reason=reason)
                if notification_store is not loop.store:
                    notification_store.close_child_questions(
                        child_instance_id, reason=reason
                    )
                notification = _sync_agent_notification(
                    notification_store,
                    notification_index,
                    child_instance_id,
                )
                if notification_store is not loop.store:
                    store_notification = _sync_agent_notification(
                        loop.store,
                        store_notification_index,
                        child_instance_id,
                    )
                    if notification is None:
                        notification = store_notification
                if notification is None:
                    lifecycle = (
                        child_store.agent_lifecycle()
                        if child_store is not None
                        else None
                    )
                    terminal_state = lifecycle.get("state") if lifecycle else None
                    terminal_text = lifecycle.get("final_result") if lifecycle else None
                    recovered_status = {
                        "completed": "completed",
                        "failed": "error",
                        "canceled": "canceled",
                    }.get(terminal_state, "canceled")
                    recovered_text = (
                        terminal_text
                        if type(terminal_text) is str and terminal_text
                        else "background child canceled when the parent session exited"
                    )
                    recovered_state: TerminalState = (
                        "completed"
                        if recovered_status == "completed"
                        else "canceled"
                        if recovered_status == "canceled"
                        else "failed"
                    )
                    recovered_stats = agent_stats(
                        lifecycle,
                        status=recovered_status,
                        turns_used=marker.get("turns_used", 0),
                    )
                    recovered_killed_ids = lifecycle.get("killed_task_ids") if lifecycle else None
                    recovered_killed_count = lifecycle.get("killed_task_count") if lifecycle else None
                    recovered_killed_truncated = lifecycle.get("killed_task_ids_truncated", False) if lifecycle else False
                    recovered_result = build_agent_receipt(
                        recovered_state,
                        recovered_text,
                        recovered_stats,
                    )
                    recovered_text = recovered_result["content"][0]["text"]
                    appended, created = (
                        notification_store.append_agent_notification_if_absent(
                            child_instance_id,
                            child_session_path=marker["child_session_path"],
                            description=marker["description"],
                            status=recovered_status,
                            text=recovered_text,
                            stats=recovered_stats,
                            killed_task_ids=recovered_killed_ids,
                            killed_task_count=recovered_killed_count,
                            killed_task_ids_truncated=recovered_killed_truncated,
                        )
                    )
                    notification_index[child_instance_id] = appended
                    if not created:
                        recovered_status = appended.data["status"]
                        recovered_text = appended.data["text"]
                        existing_stats = appended.data.get("stats")
                        recovered_stats = (
                            existing_stats if type(existing_stats) is dict else None
                        )
                    if notification_store is not loop.store:
                        appended, _ = (
                            loop.store.append_agent_notification_if_absent(
                                child_instance_id,
                                child_session_path=marker["child_session_path"],
                                description=marker["description"],
                                status=recovered_status,
                                text=recovered_text,
                                stats=recovered_stats,
                                killed_task_ids=recovered_killed_ids,
                                killed_task_count=recovered_killed_count,
                                killed_task_ids_truncated=recovered_killed_truncated,
                            )
                        )
                        store_notification_index[child_instance_id] = appended
                        recovered_status = appended.data["status"]
                        recovered_text = appended.data["text"]
                        appended_stats = appended.data.get("stats")
                        recovered_stats = (
                            appended_stats if type(appended_stats) is dict else None
                        )
                    if child_store is not None:
                        if (
                            recovered_status == "canceled"
                            and terminal_state != "canceled"
                        ):
                            child_store.mark_agent_canceled(tool_call.id)
                        else:
                            child_store.finish_agent_parent()
                elif child_store is not None:
                    if notification.data["status"] == "canceled":
                        child_store.mark_agent_canceled(tool_call.id)
                    else:
                        if child_store.agent_lifecycle() is not None:
                            child_store.finish_agent_lifecycle(
                                "completed"
                                if notification.data["status"] == "completed"
                                else "failed",
                                final_result=str(notification.data["text"]),
                            )
                        child_store.finish_agent_parent()
                loop.store.finish_agent_child(marker_key)
                continue
            existing_result = loop._existing_tool_result(tool_call.id)
            recovered_result = (
                _lifecycle_tool_result(tool_call, child_store, child_path)
                if child_store is not None
                else None
            )
            if existing_result is None:
                loop.store.append_message(
                    Message(
                        MessageRole.TOOL_RESULT,
                        [
                            TextContent(
                                recovered_result.content
                                if recovered_result
                                else "tool execution canceled"
                            )
                        ],
                        tool_result=recovered_result
                        or loop._canceled_agent_result(
                            tool_call.id,
                            child_session_path=marker["child_session_path"],
                            turns_used=marker.get("turns_used", 0),
                            agent_type=marker.get("agent_type"),
                            child_instance_id=marker.get(
                                "child_instance_id",
                                f"{loop.store.session_id}:{child_path.name}",
                            ),
                        ),
                    )
                )
            if child_path.parent == agents_root and child_path.name.isdigit():
                if existing_result is None and recovered_result is None:
                    child_store.mark_agent_canceled(tool_call.id)
                elif recovered_result is not None or existing_result is not None:
                    child_store.finish_agent_parent()
            loop.store.finish_agent_child(marker_key)
    recover_orphaned_questions(loop.store)


BuildResult = Callable[..., dict[str, object]]


async def finish_background_child(
    *,
    child_task: asyncio.Task[dict[str, object]],
    child_store: ConversationStore,
    parent_store: ConversationStore,
    notification_store: ConversationStore,
    tool_call: ToolCall,
    child_instance_id: str,
    child_path: str,
    description: str,
    child_turns: Callable[[], int],
    build_result: BuildResult,
    validate_result: Callable[[object, str], ToolResult],
    publish_event: Callable[[StreamEvent], None],
    cleanup: Callable[[], None],
    close_child: Callable[[], Awaitable[tuple[str, ...]]],
    error_message: Callable[[BaseException], str],
    marker_key: str | None = None,
    agent_instance_id: str | None = None,
    background_owner: BackgroundAgentOwner | None = None,
) -> None:
    """Persist a background child result and publish its terminal card event."""

    status = "completed"
    notification_text = ""
    try:
        try:
            result = await child_task
            content = result.get("content")
            if (
                isinstance(content, list)
                and content
                and isinstance(content[0], dict)
                and isinstance(content[0].get("text"), str)
            ):
                notification_text = content[0]["text"]
            else:
                notification_text = "background child returned no text"
            if result.get("isError") is True:
                status = "error"
        except asyncio.CancelledError:
            status = "canceled"
            notification_text = "background child canceled"
        except Exception as exc:  # noqa: BLE001 - child failures become receipts
            status = "error"
            notification_text = f"agent error: {error_message(exc)}"
        # Closing the child after its final turn terminates any task it still
        # owns. close() kills them silently and returns their ids so the parent
        # is never left guessing why child work disappeared.
        killed_tasks = list(await close_child() or ())
        # Completion metadata is persisted in more than one store. Keep the
        # legacy field bounded even when a child owned an unbounded task list.
        killed_task_metadata = [task_id[:64] for task_id in killed_tasks[:64]]
        killed_task_count = len(killed_tasks)
        killed_task_ids_truncated = len(killed_tasks) > len(killed_task_metadata) or any(
            len(task_id) > 64 for task_id in killed_tasks
        )
        killed_task_notice = (
            "\nbackground tasks killed on child completion: "
            + ", ".join(killed_tasks)
            if killed_tasks
            else None
        )
        lifecycle_state = {
            "completed": "completed",
            "canceled": "canceled",
            "error": "failed",
        }[status]
        if lifecycle_state == "canceled":
            child_store.mark_agent_canceled(tool_call.id)
        if background_owner is not None:
            effective_parent_store, effective_parent_id = background_owner.parent_edge(
                child_instance_id, parent_store, agent_instance_id
            )
        else:
            effective_parent_store, effective_parent_id = (
                parent_store,
                agent_instance_id,
            )
        if background_owner is not None:
            background_owner.conversation_channel.close_questions(
                child_instance_id,
                reason="canceled" if status == "canceled" else "finished",
            )
        if status != "canceled":
            adopt_agent_children(
                child_store,
                effective_parent_store,
                background_owner=background_owner,
                parent_instance_id=effective_parent_id,
            )
            if background_owner is not None:
                background_owner.conversation_channel.reroute_questions_from(
                    child_store
                )
            child_store.finish_agent_parent()
        terminal_stats = agent_stats(
            child_store.agent_lifecycle(),
            status=status,
            turns_used=child_turns(),
        )
        result_text = notification_text or "background child completed"
        build_parameters = inspect.signature(build_result).parameters
        build_kwargs: dict[str, object] = {}
        if "stats" in build_parameters:
            build_kwargs["stats"] = terminal_stats
        if "notice" in build_parameters:
            build_kwargs["notice"] = killed_task_notice
        if "notice_items" in build_parameters:
            build_kwargs["notice_items"] = killed_tasks or None
        if build_kwargs:
            terminal_payload = build_result(
                result_text,
                status != "completed",
                status,
                **build_kwargs,
            )
        elif "stats" in build_parameters:
            terminal_payload = build_result(
                result_text,
                status != "completed",
                status,
                terminal_stats,
            )
        else:
            terminal_payload = build_result(
                result_text,
                status != "completed",
                status,
            )
        payload_content = terminal_payload.get("content")
        if (
            isinstance(payload_content, list)
            and payload_content
            and isinstance(payload_content[0], dict)
            and isinstance(payload_content[0].get("text"), str)
        ):
            notification_text = payload_content[0]["text"]
        else:
            notification_text = result_text
        # Always replace the raw lifecycle result with the same bounded final
        # receipt delivered to notification/recovery consumers. The store
        # preserves the already-recorded terminal status and timestamp.
        child_store.finish_agent_lifecycle(
            lifecycle_state,
            final_result=notification_text,
            turns_used=child_turns(),
            killed_task_ids=killed_task_metadata or None,
            killed_task_count=killed_task_count or None,
            killed_task_ids_truncated=killed_task_ids_truncated,
        )
        child_store.update_agent_lifecycle_result(
            notification_text,
            turns_used=child_turns(),
            killed_task_ids=killed_task_metadata or None,
            killed_task_count=killed_task_count or None,
            killed_task_ids_truncated=killed_task_ids_truncated,
            canonical_receipt=True,
        )
        notification = notification_store.append_agent_notification(
            child_instance_id,
            child_session_path=child_path,
            description=description,
            status=status,
            text=notification_text,
            stats=terminal_stats,
            killed_task_ids=killed_task_metadata or None,
            killed_task_count=killed_task_count or None,
            killed_task_ids_truncated=killed_task_ids_truncated,
        )
        if notification_store is not effective_parent_store:
            effective_parent_store.append_agent_notification(
                child_instance_id,
                child_session_path=child_path,
                description=description,
                status=status,
                text=notification_text,
                stats=terminal_stats,
                killed_task_ids=killed_task_metadata or None,
                killed_task_count=killed_task_count or None,
                killed_task_ids_truncated=killed_task_ids_truncated,
            )
        effective_parent_store.finish_agent_child(marker_key or tool_call.id)
        if background_owner is not None:
            background_owner.conversation_channel.notify_child_event(child_instance_id)
        event_data: dict[str, object] = {"notification_id": notification.id}
        if agent_instance_id is not None:
            event_data["agent_instance_id"] = agent_instance_id
        publish_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=tool_call,
                tool_result=validate_result(terminal_payload, tool_call.id),
                data=event_data,
            )
        )
    finally:
        try:
            await close_child()
            # Let the parent finalize the running/terminal tool result before
            # releasing the child store descriptor.
            await asyncio.sleep(0)
        finally:
            cleanup()
