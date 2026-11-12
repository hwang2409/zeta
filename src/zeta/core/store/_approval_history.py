"""Durable parent-side history for delegated approval lifecycle events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ...protocol.types import ToolCall
from ..checkpoints import ConversationEntry
from ._history_index import HistoryIndex, IndexedPendingApprovalPage


@dataclass(frozen=True, slots=True)
class _ApprovalNowSnapshot:
    """Resident active-branch approval projection, memoized by the active leaf."""

    states: dict[str, tuple[ToolCall, str | None]]
    resolutions: dict[str, Mapping[str, Any]]
    pending: tuple[tuple[int, str, Mapping[str, object]], ...]


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
    def _resident_pending_approval_page(
        self, after_seq: int, limit: int
    ) -> IndexedPendingApprovalPage:
        pending = [
            row for row in self._approval_now_snapshot().pending if row[0] > after_seq
        ]
        # Requests of one message share a seq; the cursor is that seq with a
        # strict ``>``. Ending a page inside a message would skip its remaining
        # requests forever, so a page stops only at a message boundary (and
        # always holds at least the first message, even past ``limit``). The
        # resident page counts rows while the SQL page (``pending_approval_page``)
        # counts message seqs; both end only at a message boundary, so callers
        # must loop until ``next_seq is None`` and must not assume one ``limit``
        # returns every pending request.
        page: list[tuple[int, str, Mapping[str, object]]] = []
        for row in pending:
            if len(page) >= limit and row[0] != page[-1][0]:
                break
            page.append(row)
        return IndexedPendingApprovalPage(
            tuple((request_id, tool_call) for _seq, request_id, tool_call in page),
            page[-1][0] if len(page) < len(pending) else None,
        )

    # One resident projection of the active branch, memoized by the active-leaf
    # (sequence, id). The key moves on every append (own writes included) and on
    # every branch switch or rewind, so a resolve made on the loop is visible to
    # the next `*_now` read at once, while the serve stale sweep that fires on
    # every published event rescans only when the leaf changes (O(1) otherwise).
    def _approval_now_snapshot(self) -> _ApprovalNowSnapshot:
        key = (self.active_branch_head_sequence(), self.active_branch_head_id())
        cached = self._approval_now_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        snapshot = self._build_approval_now_snapshot()
        self._approval_now_cache = (key, snapshot)
        return snapshot

    def _build_approval_now_snapshot(self) -> _ApprovalNowSnapshot:
        states: dict[str, tuple[ToolCall, str | None]] = {}
        resolutions: dict[str, Mapping[str, Any]] = {}
        resolved: set[str] = set()
        requested: list[tuple[int, str, Mapping[str, object]]] = []
        for entry in self._active_branch():
            if entry.type == "message":
                for request in entry.data.get("approval_requests", []):
                    request_id = request["request_id"]
                    states[request_id] = (
                        ToolCall.from_dict(request["tool_call"]),
                        None,
                    )
                    requested.append((entry.seq, request_id, request["tool_call"]))
            elif entry.type == "approval_resolution":
                request_id = entry.data["request_id"]
                resolved.add(request_id)
                resolutions[request_id] = dict(entry.data)
                if request_id in states:
                    states[request_id] = (states[request_id][0], entry.data["decision"])
        pending = tuple(
            sorted(
                (row for row in requested if row[1] not in resolved),
                key=lambda row: row[0],
            )
        )
        return _ApprovalNowSnapshot(states, resolutions, pending)

    # The owner loop reads the request it just published (ApprovalPolicy and the
    # serve admission) and the resolution it just wrote (the stale sweep) from
    # this resident projection, with no off-loop yield and no sqlite on the loop.
    def approval_state_now(
        self, request_id: str
    ) -> tuple[ToolCall, str | None] | None:
        return self._approval_now_snapshot().states.get(request_id)

    def approval_resolution_now(self, request_id: str) -> Mapping[str, Any] | None:
        return self._approval_now_snapshot().resolutions.get(request_id)

    def pending_approval_page_now(
        self, after_seq: int = 0, limit: int = HistoryIndex.PAGE_LIMIT
    ) -> IndexedPendingApprovalPage:
        return self._resident_pending_approval_page(after_seq, limit)

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
