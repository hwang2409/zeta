"""Shared lifecycle helpers for background agent children."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from .core.store import ConversationStore
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

    def __init__(self) -> None:
        self._cancellers: dict[str, Callable[[], None]] = {}
        self._watchers: dict[str, asyncio.Task[Any]] = {}
        self._canceling = False

    def register(
        self,
        instance_id: str,
        cancel: Callable[[], None],
        watcher: asyncio.Task[Any],
    ) -> None:
        self._cancellers[instance_id] = cancel
        self._watchers[instance_id] = watcher

    def unregister(self, instance_id: str) -> None:
        self._cancellers.pop(instance_id, None)
        self._watchers.pop(instance_id, None)

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
            watchers = tuple(
                watcher
                for watcher in self._watchers.values()
                if watcher is not current
            )
            if not watchers:
                return
            await asyncio.gather(*watchers, return_exceptions=True)


def _open_child_store(store: ConversationStore, path: Path) -> ConversationStore | None:
    agents_root = store.session_dir / "agents"
    if path.parent != agents_root or not path.name.isdigit():
        return None
    return ConversationStore(agents_root, session_id=path.name, cwd=store.cwd)


def _nested_canceled_result(marker: dict[str, object]) -> ToolResult:
    structured_content: dict[str, object] = {
        "turns_used": marker.get("turns_used", 0),
        "child_session_path": marker["child_session_path"],
    }
    if type(marker.get("agent_type")) is str:
        structured_content["agent_type"] = marker["agent_type"]
    return ToolResult(
        ToolCall.from_dict(marker["tool_call"]).id,
        "tool execution canceled",
        is_error=True,
        structured_content=structured_content,
    )


def _existing_tool_result(store: ConversationStore, tool_call_id: str) -> bool:
    return any(
        message.tool_result is not None
        and message.tool_result.tool_call_id == tool_call_id
        for message in store.messages()
    )


def _recover_nested_children(store: ConversationStore) -> None:
    """Cancel descendants left behind when an ancestor session exits."""

    for tool_call_id, marker in store.agent_children().items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        child_store = _open_child_store(store, child_path)
        if child_store is not None:
            _recover_nested_children(child_store)
        existing_result = _existing_tool_result(store, tool_call_id)
        notification_status: str | None = None
        if marker.get("background"):
            child_instance_id = marker.get(
                "child_instance_id", f"{store.session_id}:{child_path.name}"
            )
            existing = next(
                (
                    entry
                    for entry in store.agent_notifications(pending_only=False)
                    if entry.data["child_instance_id"] == child_instance_id
                ),
                None,
            )
            if existing is None:
                store.append_agent_notification(
                    child_instance_id,
                    child_session_path=str(child_path),
                    description=marker["description"],
                    status="canceled",
                    text="background child canceled when the parent session exited",
                )
                notification_status = "canceled"
            else:
                notification_status = existing.data["status"]
        elif not existing_result:
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
                child_store.finish_agent_parent()
            else:
                child_store.mark_agent_canceled(tool_call.id)
        store.finish_agent_child(tool_call_id)


def recover_agent_children(loop: _AgentLoopForRecovery) -> None:
    """Resolve child markers left by a process exit before resuming."""

    for tool_call_id, marker in loop.store.agent_children().items():
        tool_call = ToolCall.from_dict(marker["tool_call"])
        child_path = Path(marker["child_session_path"])
        agents_root = loop.store.session_dir / "agents"
        child_store = _open_child_store(loop.store, child_path)
        if child_store is not None:
            _recover_nested_children(child_store)
        if marker.get("background"):
            child_instance_id = marker.get(
                "child_instance_id", f"{loop.store.session_id}:{child_path.name}"
            )
            notification = next(
                (
                    entry
                    for entry in loop.store.agent_notifications(pending_only=False)
                    if entry.data["child_instance_id"] == child_instance_id
                ),
                None,
            )
            if notification is None:
                loop.store.append_agent_notification(
                    child_instance_id,
                    child_session_path=marker["child_session_path"],
                    description=marker["description"],
                    status="canceled",
                    text="background child canceled when the parent session exited",
                )
                if child_store is not None:
                    child_store.mark_agent_canceled(tool_call.id)
            elif child_store is not None:
                if notification.data["status"] == "canceled":
                    child_store.mark_agent_canceled(tool_call.id)
                else:
                    child_store.finish_agent_parent()
            loop.store.finish_agent_child(tool_call_id)
            continue
        existing_result = loop._existing_tool_result(tool_call_id)
        if existing_result is None:
            loop.store.append_message(
                Message(
                    MessageRole.TOOL_RESULT,
                    [TextContent("tool execution canceled")],
                    tool_result=loop._canceled_agent_result(
                        tool_call_id,
                        child_session_path=marker["child_session_path"],
                        turns_used=marker.get("turns_used", 0),
                        agent_type=marker.get("agent_type"),
                    ),
                )
            )
        if child_path.parent == agents_root and child_path.name.isdigit():
            if existing_result is None:
                child_store.mark_agent_canceled(tool_call.id)
            else:
                child_store.finish_agent_parent()
        loop.store.finish_agent_child(tool_call.id)


BuildResult = Callable[[str, bool, str], dict[str, object]]


async def finish_background_child(
    *,
    child_task: asyncio.Task[dict[str, object]],
    child_store: ConversationStore,
    parent_store: ConversationStore,
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
) -> None:
    """Persist a background child result and publish its terminal card event."""

    status = "completed"
    notification_text = ""
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
    if status == "canceled":
        child_store.mark_agent_canceled(tool_call.id)
    else:
        child_store.finish_agent_parent()
    notification = parent_store.append_agent_notification(
        child_instance_id,
        child_session_path=child_path,
        description=description,
        status=status,
        text=notification_text or "background child completed",
    )
    parent_store.finish_agent_child(tool_call.id)
    terminal_payload = build_result(
        notification_text or "background child completed",
        status != "completed",
        status,
    )
    publish_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=tool_call,
            tool_result=validate_result(terminal_payload, tool_call.id),
            data={"notification_id": notification.id},
        )
    )
    cleanup()
    await close_child()
