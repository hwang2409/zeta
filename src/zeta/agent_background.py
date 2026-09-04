"""Shared lifecycle helpers for background agent children."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from .core.store import ConversationEntry, ConversationStore
from .types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
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

    def __init__(self, notification_store: ConversationStore) -> None:
        self.notification_store = notification_store
        self._cancellers: dict[str, Callable[[], None]] = {}
        self._watchers: dict[str, asyncio.Task[Any]] = {}
        self._parent_stores: dict[str, ConversationStore] = {}
        self._canceling = False

    def register(
        self,
        instance_id: str,
        cancel: Callable[[], None],
        watcher: asyncio.Task[Any],
        parent_store: ConversationStore | None = None,
    ) -> None:
        self._cancellers[instance_id] = cancel
        self._watchers[instance_id] = watcher
        if parent_store is not None:
            self._parent_stores[instance_id] = parent_store

    def unregister(self, instance_id: str) -> None:
        self._cancellers.pop(instance_id, None)
        self._watchers.pop(instance_id, None)
        self._parent_stores.pop(instance_id, None)

    def adopt(self, instance_id: str, parent_store: ConversationStore) -> None:
        if instance_id in self._watchers:
            self._parent_stores[instance_id] = parent_store

    def parent_store(
        self, instance_id: str, default: ConversationStore
    ) -> ConversationStore:
        return self._parent_stores.get(instance_id, default)

    def cancel_all(self) -> None:
        if self._canceling:
            return
        self._canceling = True
        try:
            for cancel in tuple(self._cancellers.values()):
                cancel()
        finally:
            self._canceling = False

    @property
    def running(self) -> bool:
        return bool(self._cancellers)

    async def wait(self) -> None:
        current = asyncio.current_task()
        while True:
            for instance_id, watcher in tuple(self._watchers.items()):
                if watcher.done():
                    self.unregister(instance_id)
            watchers = tuple(
                watcher
                for watcher in self._watchers.values()
                if watcher is not current
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
        )
        turns_used = marker.get("turns_used", 0)
        if turns_used:
            parent_store.update_agent_child_turns(adopted_key, turns_used)
        child_instance_id = marker.get("child_instance_id")
        if background_owner is not None and type(child_instance_id) is str:
            background_owner.adopt(child_instance_id, parent_store)
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
    return ToolResult(
        ToolCall.from_dict(marker["tool_call"]).id,
        "tool execution canceled",
        is_error=True,
        structured_content=structured_content,
    )


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
    return ToolResult(
        tool_call.id,
        final_result,
        is_error=state != "completed",
        structured_content=structured,
    )


def _existing_tool_result(store: ConversationStore, tool_call_id: str) -> bool:
    return any(
        message.tool_result is not None
        and message.tool_result.tool_call_id == tool_call_id
        for message in store.messages()
    )


def _agent_notification(
    store: ConversationStore,
    notification_id: str,
) -> ConversationEntry | None:
    return next(
        (
            entry
            for entry in store.agent_notifications(pending_only=False)
            if entry.data["child_instance_id"] == notification_id
        ),
        None,
    )


def _recover_nested_children(
    store: ConversationStore,
    notification_store: ConversationStore,
) -> None:
    """Cancel descendants left behind when an ancestor session exits."""

    for marker_key, marker in store.agent_children().items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        child_store = _open_child_store(store, child_path)
        if child_store is not None:
            _recover_nested_children(child_store, notification_store)
        existing_result = _existing_tool_result(store, tool_call.id)
        notification_status: str | None = None
        notification_text: str | None = None
        if marker.get("background"):
            child_instance_id = marker.get(
                "child_instance_id", f"{store.session_id}:{child_path.name}"
            )
            existing = _agent_notification(notification_store, child_instance_id)
            if existing is None and notification_store is not store:
                existing = _agent_notification(store, child_instance_id)
            if existing is None:
                lifecycle = (
                    child_store.agent_lifecycle() if child_store is not None else None
                )
                lifecycle_state = lifecycle.get("state") if lifecycle else None
                lifecycle_text = lifecycle.get("final_result") if lifecycle else None
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
                notification_store.append_agent_notification(
                    child_instance_id,
                    child_session_path=str(child_path),
                    description=marker["description"],
                    status=notification_status,
                    text=notification_text,
                )
            else:
                notification_status = existing.data["status"]
                notification_text = existing.data["text"]
            if (
                notification_store is not store
                and _agent_notification(store, child_instance_id) is None
            ):
                store.append_agent_notification(
                    child_instance_id,
                    child_session_path=str(child_path),
                    description=marker["description"],
                    status=notification_status,
                    text=(
                        "background child canceled when the parent session exited"
                        if notification_status == "canceled"
                        else "background child completed"
                    ),
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

    for marker_key, marker in loop.store.agent_children().items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        agents_root = loop.store.session_dir / "agents"
        child_store = _open_child_store(loop.store, child_path)
        if child_store is not None:
            _recover_nested_children(
                child_store,
                loop._background_owner.notification_store,
            )
        if marker.get("background"):
            child_instance_id = marker.get(
                "child_instance_id", f"{loop.store.session_id}:{child_path.name}"
            )
            notification = _agent_notification(
                loop._background_owner.notification_store,
                child_instance_id,
            )
            if (
                notification is None
                and loop._background_owner.notification_store is not loop.store
            ):
                notification = _agent_notification(loop.store, child_instance_id)
            if notification is None:
                lifecycle = (
                    child_store.agent_lifecycle() if child_store is not None else None
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
                loop._background_owner.notification_store.append_agent_notification(
                    child_instance_id,
                    child_session_path=marker["child_session_path"],
                    description=marker["description"],
                    status=recovered_status,
                    text=recovered_text,
                )
                if loop._background_owner.notification_store is not loop.store:
                    loop.store.append_agent_notification(
                        child_instance_id,
                        child_session_path=marker["child_session_path"],
                        description=marker["description"],
                        status=recovered_status,
                        text=recovered_text,
                    )
                if child_store is not None:
                    if recovered_status == "canceled" and terminal_state != "canceled":
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


BuildResult = Callable[[str, bool, str], dict[str, object]]


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
    close_child: Callable[[], Awaitable[None]],
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
        except Exception as exc:
            status = "error"
            notification_text = f"agent error: {error_message(exc)}"
        lifecycle_state = {
            "completed": "completed",
            "canceled": "canceled",
            "error": "failed",
        }[status]
        lifecycle = child_store.agent_lifecycle()
        if lifecycle_state == "canceled":
            child_store.mark_agent_canceled(tool_call.id)
        elif lifecycle is None or lifecycle.get("finished_at") is None:
            child_store.finish_agent_lifecycle(
                lifecycle_state,
                final_result=notification_text or "background child completed",
                turns_used=child_turns(),
            )
        effective_parent_store = (
            background_owner.parent_store(child_instance_id, parent_store)
            if background_owner is not None
            else parent_store
        )
        if status != "canceled":
            adopt_agent_children(
                child_store,
                effective_parent_store,
                background_owner=background_owner,
            )
            child_store.finish_agent_parent()
        notification = notification_store.append_agent_notification(
            child_instance_id,
            child_session_path=child_path,
            description=description,
            status=status,
            text=notification_text or "background child completed",
        )
        if notification_store is not effective_parent_store:
            effective_parent_store.append_agent_notification(
                child_instance_id,
                child_session_path=child_path,
                description=description,
                status=status,
                text=notification_text or "background child completed",
            )
        effective_parent_store.finish_agent_child(marker_key or tool_call.id)
        terminal_payload = build_result(
            notification_text or "background child completed",
            status != "completed",
            status,
        )
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
        finally:
            cleanup()
