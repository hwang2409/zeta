"""Durable parent-side history for delegated approval lifecycle events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...protocol.types import ToolCall
from ..checkpoints import ConversationEntry
from ._approval_display import validated_approval_display
from ._incremental_validation import APPROVAL_RESOLUTION_REASONS


def validate_approval_history_data(data: Mapping[str, Any]) -> None:
    event = data.get("event")
    if event not in {"approval_request", "approval_end"}:
        raise ValueError("invalid approval history event")
    request_id = data.get("request_id")
    if type(request_id) is not str or not request_id:
        raise ValueError("approval history id must be a nonempty string")
    tool_call = data.get("tool_call")
    if type(tool_call) is not dict:
        raise ValueError("approval history tool_call must be an object")
    ToolCall.from_dict(tool_call)
    if event == "approval_request":
        if data.get("delegated") is not True:
            raise ValueError("approval history request must be delegated")
        child_id = data.get("agent_instance_id")
        if type(child_id) is not str or not child_id:
            raise ValueError("approval history request must identify its child")
        if "approval_display" in data:
            validated_approval_display(data["approval_display"])
        return
    end_data = data.get("data")
    if type(end_data) is not dict:
        raise ValueError("approval history end data must be an object")
    decision = end_data.get("decision")
    if decision not in {"allow", "deny", "abort"}:
        raise ValueError("invalid approval history decision")
    reason = end_data.get("reason")
    if reason is not None and (
        type(reason) is not str
        or decision != "abort"
        or reason not in APPROVAL_RESOLUTION_REASONS
    ):
        raise ValueError("invalid approval history reason")
    if data.get("reason") != reason and ("reason" in data or reason is not None):
        raise ValueError("approval history reason must match end data")


class ApprovalHistoryStoreMixin:
    """Persist delegated wire identity and retrieve durable resolutions."""

    def append_approval_history(
        self, payload: Mapping[str, Any]
    ) -> ConversationEntry:
        with self._append_lock():
            entry = self._append_row_unlocked("approval_history", dict(payload))
            return self._snapshot_entry(entry)

    async def append_approval_history_async(
        self, payload: Mapping[str, Any]
    ) -> ConversationEntry:
        """Append delegated history through the store's durable writer."""
        return await self._to_thread_durable(self.append_approval_history, payload)

    def approval_resolution_data(self, request_id: str) -> dict[str, str] | None:
        with self._append_lock():
            for entry in reversed(self.replay()):
                if (
                    entry.type == "approval_resolution"
                    and entry.data.get("request_id") == request_id
                ):
                    return dict(entry.data)
        return None
