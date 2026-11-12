"""Durable parent-side history for delegated approval lifecycle events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...protocol.types import ToolCall
from ..checkpoints import ConversationEntry


def validate_approval_history_data(data: Mapping[str, Any]) -> None:
    # imported lazily to avoid a cycle through the store facade.
    from ._approval_display import validated_approval_display
    from ._incremental_validation import APPROVAL_RESOLUTION_REASONS

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

    def approval_states_snapshot(self) -> dict[str, tuple[ToolCall, str | None]]:
        """Read approval state from the owner-installed indexes."""
        return self._approval_states_from_maps(
            self._active_approval_requests,
            self._active_approval_resolutions,
        )

    @staticmethod
    def _approval_states_from_maps(
        requests: Mapping[str, ConversationEntry],
        resolutions: Mapping[str, str],
    ) -> dict[str, tuple[ToolCall, str | None]]:
        states: dict[str, tuple[ToolCall, str | None]] = {}
        for request_id, entry in requests.items():
            request = next(
                request
                for request in entry.data.get("approval_requests", [])
                if request["request_id"] == request_id
            )
            states[request_id] = (
                ToolCall.from_dict(request["tool_call"]),
                resolutions.get(request_id),
            )
        return states

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
        from ._async_writes import _AppendApprovalHistoryCommand

        return await self._commit_storage_command(
            _AppendApprovalHistoryCommand(payload=payload)
        )

    def approval_resolution_data(self, request_id: str) -> dict[str, str] | None:
        """Read a published approval resolution without touching the log."""
        if request_id not in self._active_approval_resolutions:
            return None
        for entry in reversed(self._entries):
            if (
                entry.type == "approval_resolution"
                and entry.data.get("request_id") == request_id
            ):
                return dict(entry.data)
        return None

    def delegated_unmatched_requests(
        self, child_instance_id: str
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return (wire_request_id, tool_call) for a child's delegated requests
        that have no matching ``approval_end`` on the active branch.

        ``tool_call["id"]`` is the child store's request id, so a caller can
        reread the child's authoritative resolution and project a parent end.
        """
        requests: dict[str, dict[str, Any]] = {}
        ended: set[str] = set()
        for entry in self._active_branch():
            if entry.type != "approval_history":
                continue
            data = entry.data
            event = data.get("event")
            if event == "approval_request":
                if data.get("agent_instance_id") == child_instance_id:
                    requests[data["request_id"]] = data["tool_call"]
            elif event == "approval_end":
                request_id = data.get("request_id")
                if isinstance(request_id, str):
                    ended.add(request_id)
        return [
            (wire_id, tool_call)
            for wire_id, tool_call in requests.items()
            if wire_id not in ended
        ]

    def append_approval_history_end_if_absent(
        self, payload: Mapping[str, Any]
    ) -> bool:
        """Append one delegated ``approval_end`` only if the active branch has
        none for this wire id. The absence check and append share one lock, so
        a repeated restart never writes a duplicate end."""
        wire_id = payload["request_id"]
        with self._append_lock():
            for entry in self._active_branch():
                if (
                    entry.type == "approval_history"
                    and entry.data.get("event") == "approval_end"
                    and entry.data.get("request_id") == wire_id
                ):
                    return False
            self._append_row_unlocked(
                "approval_history", {"event": "approval_end", **payload}
            )
            return True
