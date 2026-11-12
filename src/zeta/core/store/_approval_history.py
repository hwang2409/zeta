"""Durable parent-side history for delegated approval lifecycle events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ...protocol.types import ToolCall
from ..checkpoints import ConversationEntry
from ._history_index import HistoryIndex, IndexedPendingApprovalPage


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

    @staticmethod
    def _request_entry(
        branch: list[ConversationEntry],
        request_id: str,
    ) -> ConversationEntry | None:
        for entry in branch:
            if entry.type == "message":
                for request in entry.data.get("approval_requests", []):
                    if request["request_id"] == request_id:
                        return entry
        return None

    @staticmethod
    def _approval_request_entries(
        entries: Iterable[ConversationEntry],
    ) -> dict[str, ConversationEntry]:
        return {
            request["request_id"]: entry
            for entry in entries
            if entry.type == "message"
            for request in entry.data.get("approval_requests", [])
        }

    @staticmethod
    def _approval_states_from_branch(
        branch: list[ConversationEntry],
    ) -> dict[str, tuple[ToolCall, str | None]]:
        states: dict[str, tuple[ToolCall, str | None]] = {}
        for entry in branch:
            if entry.type == "message":
                requests = entry.data.get("approval_requests", [])
                for request in requests:
                    request_id = request["request_id"]
                    states[request_id] = (
                        ToolCall.from_dict(request["tool_call"]),
                        None,
                    )
            elif entry.type == "approval_resolution":
                request_id = entry.data["request_id"]
                if request_id in states:
                    tool_call, _ = states[request_id]
                    states[request_id] = (tool_call, entry.data["decision"])
        return states

    # A read-only reader cannot read the shared sidecar for its own lagging
    # view: the sidecar is updated in place by the writer and may be mid-append.
    # It answers approval reads from its resident branch, like main's
    # latency-tolerant display path, without blocking on the writer's lock.
    def _resident_approval_state(
        self, request_id: str
    ) -> tuple[ToolCall, str | None] | None:
        return self._approval_states_from_branch(
            list(self._active_branch())
        ).get(request_id)

    # The owner's write path (ApprovalPolicy.authorize) needs the request it is
    # about to publish to be readable in the same task step, with no off-loop
    # yield before it. This is a resident, non-blocking scan (no sqlite on the
    # loop): it pulls any external append without waiting, then reads the
    # resident branch, matching main's synchronous approval_states() read.
    def approval_state_now(
        self, request_id: str
    ) -> tuple[ToolCall, str | None] | None:
        self._sync_log_without_waiting()
        return self._resident_approval_state(request_id)

    def _resident_approval_resolution(
        self, request_id: str
    ) -> Mapping[str, Any] | None:
        for entry in reversed(self._active_branch()):
            if (
                entry.type == "approval_resolution"
                and entry.data.get("request_id") == request_id
            ):
                return dict(entry.data)
        return None

    def _resident_pending_approval_page(
        self, after_seq: int, limit: int
    ) -> IndexedPendingApprovalPage:
        resolved: set[str] = set()
        pending: list[tuple[int, str, Mapping[str, object]]] = []
        for entry in self._active_branch():
            if entry.type == "message":
                for request in entry.data.get("approval_requests", []):
                    pending.append(
                        (entry.seq, request["request_id"], request["tool_call"])
                    )
            elif entry.type == "approval_resolution":
                resolved.add(entry.data["request_id"])
        rows = sorted(
            (
                (seq, request_id, tool_call)
                for seq, request_id, tool_call in pending
                if request_id not in resolved and seq > after_seq
            ),
            key=lambda row: row[0],
        )[:limit]
        return IndexedPendingApprovalPage(
            tuple((request_id, tool_call) for _seq, request_id, tool_call in rows),
            rows[-1][0] if len(rows) == limit else None,
        )

    # The display seed at resume needs the pending page in the same task step,
    # with no off-loop yield, matching approval_state_now. The async
    # pending_approval_page reconciles later off-loop.
    def pending_approval_page_now(
        self, after_seq: int = 0, limit: int = HistoryIndex.PAGE_LIMIT
    ) -> IndexedPendingApprovalPage:
        self._sync_log_without_waiting()
        return self._resident_pending_approval_page(after_seq, limit)
