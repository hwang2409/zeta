"""Server-owned wire identity and terminal state for approvals."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from ..core.approval import ApprovalRequest

ApprovalKey = str | tuple[str, str]


@dataclass(slots=True)
class _TrackedApproval:
    key: ApprovalKey
    request_id: str
    tool_call: dict[str, object]
    delegated: bool
    agent_instance_id: str | None
    ended: bool = False

    def pending_payload(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "tool_call": self.tool_call,
            "delegated": self.delegated,
            **(
                {"agent_instance_id": self.agent_instance_id}
                if self.agent_instance_id is not None
                else {}
            ),
        }

    def end_payload(self, data: dict[str, object]) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "tool_call": self.tool_call,
            "data": data,
        }


class ApprovalLifecycle:
    """Own stable approval IDs and make terminal events exactly once."""

    def __init__(self) -> None:
        self._by_key: dict[ApprovalKey, _TrackedApproval] = {}
        self._wires_by_key: dict[ApprovalKey, str] = {}
        self._keys_by_wire: dict[str, ApprovalKey] = {}

    def observe(self, request: ApprovalRequest) -> dict[str, object]:
        """Track a visible request and return its pending wire payload."""

        tracked = self._by_key.get(request.key)
        if tracked is None or tracked.ended:
            request_id = self.wire_id(request.key)
            tracked = _TrackedApproval(
                key=request.key,
                request_id=request_id,
                tool_call=request.tool_call.to_dict(),
                delegated=request.child_instance_id is not None,
                agent_instance_id=request.child_instance_id,
            )
            self._by_key[request.key] = tracked
        return tracked.pending_payload()

    def core_key(self, request_id: str) -> ApprovalKey:
        """Resolve an opaque wire ID, or preserve a foreground raw ID."""

        return self._keys_by_wire.get(request_id, request_id)

    def wire_id(self, key: ApprovalKey) -> str:
        """Return the current stable wire ID for a tracked core key."""

        wire = self._wires_by_key.get(key)
        if wire is None:
            wire = key if isinstance(key, str) else f"approval-{uuid4().hex}"
            self._wires_by_key[key] = wire
            self._keys_by_wire[wire] = key
        return wire

    def end(
        self, key: ApprovalKey, *, data: dict[str, object] | None = None
    ) -> dict[str, object] | None:
        """Return one terminal payload, or None after it was already emitted."""

        tracked = self._by_key.get(key)
        if tracked is None or tracked.ended:
            return None
        tracked.ended = True
        return tracked.end_payload(data or {})

    def active_keys(self) -> tuple[ApprovalKey, ...]:
        """Return visible approvals that do not have a terminal event."""

        return tuple(
            tracked.key for tracked in self._by_key.values() if not tracked.ended
        )

    def prune_ended(self) -> None:
        """Forget approvals after their turn has delivered all lifecycle events."""

        for key, tracked in tuple(self._by_key.items()):
            if not tracked.ended:
                continue
            del self._by_key[key]
            self._wires_by_key.pop(key, None)
            self._keys_by_wire.pop(tracked.request_id, None)

    def clear(self) -> None:
        """Forget all approval identities at a connection lifecycle transition."""

        self._by_key.clear()
        self._wires_by_key.clear()
        self._keys_by_wire.clear()
