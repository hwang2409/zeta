"""Durable notification query, presentation, and consumption state."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ...agent.receipt import valid_killed_task_fields
from ...protocol.types import Message, ToolUseContent
from ..checkpoints import ConversationEntry
from ._validation import (
    AGENT_COMPLETION_NOTIFICATION_KIND,
    MAX_AGENT_NOTIFICATION_TEXT,
    valid_agent_stats,
    validate_agent_notification_data,
)

if TYPE_CHECKING:
    from ._store import ConversationStore


@dataclass(frozen=True, slots=True)
class TUIReplayEntry:
    """Detached fields consumed while rebuilding the TUI transcript."""

    seq: int
    id: str
    type: str
    data: dict[str, Any]
    message: Message | None = None


class NotificationStateMixin:
    """Keep human presentation separate from parent-agent consumption."""

    @staticmethod
    def _agent_notification_data(
        child_instance_id: str,
        *,
        child_session_path: str,
        description: str,
        status: str,
        text: str,
        stats: dict[str, Any] | None = None,
        killed_task_ids: list[str] | None = None,
        killed_task_count: int | None = None,
        killed_task_ids_truncated: bool = False,
        background_metadata: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        if (
            not child_instance_id
            or not child_session_path
            or not description
            or status not in {"completed", "error", "canceled"}
            or not text
            or type(text) is not str
            or len(text) > MAX_AGENT_NOTIFICATION_TEXT
        ):
            raise ValueError("invalid agent notification")
        if stats is not None and not valid_agent_stats(stats):
            raise ValueError("invalid agent notification stats")
        fields = {
            "killed_task_ids": killed_task_ids,
            "killed_task_count": killed_task_count,
            "killed_task_ids_truncated": killed_task_ids_truncated,
        }
        if not valid_killed_task_fields(fields):
            raise ValueError("invalid killed task fields")
        data: dict[str, Any] = {
            "kind": AGENT_COMPLETION_NOTIFICATION_KIND,
            "child_instance_id": child_instance_id,
            "child_session_path": child_session_path,
            "description": description,
            "status": status,
            "text": text,
        }
        if stats is not None:
            data["stats"] = dict(stats)
        if killed_task_ids:
            data["killed_task_ids"] = list(killed_task_ids)
        if killed_task_count is not None:
            data["killed_task_count"] = killed_task_count
            data["killed_task_ids_truncated"] = killed_task_ids_truncated
        if background_metadata is not None:
            data["background_owner"], data["background_phase"] = background_metadata
        validate_agent_notification_data(data)
        return data

    def append_agent_notification(
        self: ConversationStore,
        child_instance_id: str,
        *,
        child_session_path: str,
        description: str,
        status: str,
        text: str,
        stats: dict[str, Any] | None = None,
        killed_task_ids: list[str] | None = None,
        killed_task_count: int | None = None,
        killed_task_ids_truncated: bool = False,
        background_metadata: tuple[str, str] | None = None,
    ) -> ConversationEntry:
        """Persist one agent-completion notification (legacy API)."""
        data = self._agent_notification_data(
            child_instance_id,
            child_session_path=child_session_path,
            description=description,
            status=status,
            text=text,
            stats=stats,
            killed_task_ids=killed_task_ids,
            killed_task_count=killed_task_count,
            killed_task_ids_truncated=killed_task_ids_truncated,
            background_metadata=background_metadata,
        )
        return self._append_row("notification", data)

    def append_agent_notification_if_absent(
        self: ConversationStore,
        child_instance_id: str,
        *,
        child_session_path: str,
        description: str,
        status: str,
        text: str,
        stats: dict[str, Any] | None = None,
        killed_task_ids: list[str] | None = None,
        killed_task_count: int | None = None,
        killed_task_ids_truncated: bool = False,
        background_metadata: tuple[str, str] | None = None,
    ) -> tuple[ConversationEntry, bool]:
        """Atomically return the first completion notification or append one."""
        data = self._agent_notification_data(
            child_instance_id,
            child_session_path=child_session_path,
            description=description,
            status=status,
            text=text,
            stats=stats,
            killed_task_ids=killed_task_ids,
            killed_task_count=killed_task_count,
            killed_task_ids_truncated=killed_task_ids_truncated,
            background_metadata=background_metadata,
        )
        with self._append_lock():
            self._load()
            existing = self._active_completion_notifications.get(child_instance_id)
            if existing is not None:
                return self._snapshot_entry(existing), False
            appended = self._append_row_unlocked("notification", data)
            return self._snapshot_entry(appended), True

    def agent_completion_notifications_by_child(
        self: ConversationStore,
    ) -> dict[str, ConversationEntry]:
        """Return detached first completion notifications keyed by child id."""
        return {
            child_id: self._snapshot_entry(entry)
            for child_id, entry in self._active_completion_notifications.items()
        }

    def sync_agent_completion_notification(
        self: ConversationStore, child_instance_id: str
    ) -> ConversationEntry | None:
        """Incrementally sync the log tail and return one active notification."""
        with self._append_lock():
            self._load()
            entry = self._active_completion_notifications.get(child_instance_id)
            return self._snapshot_entry(entry) if entry is not None else None

    def tui_replay_entries(self: ConversationStore) -> tuple[TUIReplayEntry, ...]:
        """Project the active branch into detached data needed by TUI replay."""
        projected: list[TUIReplayEntry] = []
        for entry in self._active_branch():
            message = None
            if entry.type == "message":
                parsed = Message.from_dict(entry.data["message"])
                content = [
                    replace(
                        block,
                        tool_call=replace(
                            block.tool_call,
                            arguments=copy.deepcopy(block.tool_call.arguments),
                        ),
                    )
                    if isinstance(block, ToolUseContent)
                    else block
                    for block in parsed.content
                ]
                tool_result = parsed.tool_result
                if tool_result is not None:
                    tool_result = replace(
                        tool_result,
                        structured_content=copy.deepcopy(
                            tool_result.structured_content
                        ),
                    )
                message = replace(
                    parsed,
                    content=content,
                    tool_result=tool_result,
                    metadata=copy.deepcopy(parsed.metadata),
                )
                data = {}
            elif entry.type in {
                "checkpoint",
                "fork",
                "compaction",
                "notification",
                "notification_ack",
                "notification_tui_presented",
            }:
                data = copy.deepcopy(entry.data)
            else:
                data = {}
            projected.append(
                TUIReplayEntry(
                    seq=entry.seq,
                    id=entry.id,
                    type=entry.type,
                    data=data,
                    message=message,
                )
            )
        return tuple(projected)

    def agent_notifications(
        self: ConversationStore, *, pending_only: bool = True
    ) -> list[ConversationEntry]:
        """Return durable background-child notifications on the active branch."""

        return [
            self._snapshot_entry(entry)
            for notification_id, entry in self._active_notifications.items()
            if not pending_only or notification_id not in self._active_notification_acks
        ]

    def is_agent_notification_presented_to_tui(
        self: ConversationStore, notification_id: str
    ) -> bool:
        """Return whether the TUI has already shown this notification."""

        return notification_id in self._active_notification_presentations

    def record_agent_notification_delivery(
        self: ConversationStore,
        notification_id: str,
        *,
        presented: bool = False,
        acknowledged: bool = False,
    ) -> None:
        """Persist ordered presentation/ack markers in one durable append batch.

        When both flags are true the presentation marker precedes the
        acknowledgement and both rows are written under one flock and fsync.
        Existing markers are omitted, so retrying is idempotent.
        """
        if not presented and not acknowledged:
            return
        with self._append_lock():
            self._load()
            branch = self.replay()
            if not any(
                entry.type == "notification" and entry.id == notification_id
                for entry in branch
            ):
                raise ValueError(f"unknown agent notification: {notification_id}")
            rows: list[tuple[str, dict[str, str]]] = []
            if presented and not any(
                entry.type == "notification_tui_presented"
                and entry.data["notification_id"] == notification_id
                for entry in branch
            ):
                rows.append(
                    ("notification_tui_presented", {"notification_id": notification_id})
                )
            if acknowledged and not any(
                entry.type == "notification_ack"
                and entry.data["notification_id"] == notification_id
                for entry in branch
            ):
                rows.append(("notification_ack", {"notification_id": notification_id}))
            if rows:
                self._append_many_unlocked(rows)

    async def record_agent_notification_delivery_async(
        self: ConversationStore,
        notification_id: str,
        *,
        presented: bool = False,
        acknowledged: bool = False,
    ) -> None:
        """Persist delivery state off the event loop and in async call order."""
        await self._to_thread_durable(
            self.record_agent_notification_delivery,
            notification_id,
            presented=presented,
            acknowledged=acknowledged,
        )

    def mark_agent_notifications_presented_to_tui(
        self: ConversationStore, notification_ids: list[str]
    ) -> None:
        """Persist presentation markers in one append batch without acknowledging."""

        ordered_ids = list(dict.fromkeys(notification_ids))
        if not ordered_ids:
            return
        with self._append_lock():
            self._load()
            branch = self._active_branch()
            notifications = {
                entry.id for entry in branch if entry.type == "notification"
            }
            unknown = next(
                (item for item in ordered_ids if item not in notifications), None
            )
            if unknown is not None:
                raise ValueError(f"unknown agent notification: {unknown}")
            presented = {
                entry.data["notification_id"]
                for entry in branch
                if entry.type == "notification_tui_presented"
            }
            rows = [
                ("notification_tui_presented", {"notification_id": notification_id})
                for notification_id in ordered_ids
                if notification_id not in presented
            ]
            if rows:
                self._append_many_unlocked(rows)

    async def mark_agent_notifications_presented_to_tui_async(
        self: ConversationStore, notification_ids: list[str]
    ) -> None:
        """Persist a presentation batch off the event loop."""

        await self._to_thread_durable(
            self.mark_agent_notifications_presented_to_tui, notification_ids
        )

    def mark_agent_notification_presented_to_tui(
        self: ConversationStore, notification_id: str
    ) -> None:
        """Durably record successful TUI output without consuming the notification."""

        self.mark_agent_notifications_presented_to_tui([notification_id])

    def acknowledge_agent_notifications(
        self: ConversationStore, notification_ids: list[str]
    ) -> None:
        """Durably consume one claimed notification batch atomically."""

        ordered_ids = list(dict.fromkeys(notification_ids))
        if not ordered_ids:
            return
        with self._append_lock():
            self._load()
            branch = self._active_branch()
            notifications = {
                entry.id for entry in branch if entry.type == "notification"
            }
            unknown = next(
                (item for item in ordered_ids if item not in notifications), None
            )
            if unknown is not None:
                raise ValueError(f"unknown agent notification: {unknown}")
            acknowledged = {
                entry.data["notification_id"]
                for entry in branch
                if entry.type == "notification_ack"
            }
            rows = [
                ("notification_ack", {"notification_id": notification_id})
                for notification_id in ordered_ids
                if notification_id not in acknowledged
            ]
            if rows:
                self._append_many_unlocked(rows)

    async def acknowledge_agent_notifications_async(
        self: ConversationStore, notification_ids: list[str]
    ) -> None:
        """Consume one claimed notification batch off the event loop."""

        await self._to_thread_durable(
            self.acknowledge_agent_notifications, notification_ids
        )

    def acknowledge_agent_notification(
        self: ConversationStore, notification_id: str
    ) -> None:
        """Durably mark one notification as consumed by the parent agent."""

        self.acknowledge_agent_notifications([notification_id])
