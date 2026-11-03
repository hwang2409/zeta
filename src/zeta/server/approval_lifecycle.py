"""Server-owned wire identity and terminal state for approvals."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from uuid import uuid4

from ..core.approval import ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore

ApprovalKey = str | tuple[str, str]


def approval_display_fields(request: ApprovalRequest) -> dict[str, object]:
    """Return immutable trusted display facts for an approval request."""

    fields: dict[str, object] = {}
    if request.action is not None:
        fields["approval_action"] = request.action
    display = request.audit_display()
    if display:
        fields["approval_display"] = display
    return fields


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
            **({"reason": data["reason"]} if "reason" in data else {}),
        }


class ApprovalLifecycle:
    """Own stable approval IDs and make terminal events exactly once."""

    def __init__(self) -> None:
        self._by_key: dict[ApprovalKey, _TrackedApproval] = {}
        self._wires_by_key: dict[ApprovalKey, str] = {}
        self._keys_by_wire: dict[str, ApprovalKey] = {}

    def observe(
        self,
        request: ApprovalRequest,
        *,
        store: ConversationStore | None = None,
        fields: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Track a visible request and durably record delegated wire identity."""

        tracked = self._by_key.get(request.key)
        created = tracked is None or tracked.ended
        if created:
            request_id = self.wire_id(request.key)
            tracked = _TrackedApproval(
                key=request.key,
                request_id=request_id,
                tool_call=request.tool_call.to_dict(),
                delegated=request.child_instance_id is not None,
                agent_instance_id=request.child_instance_id,
            )
            self._by_key[request.key] = tracked
        payload = {**tracked.pending_payload(), **(fields or {})}
        if created and tracked.delegated and store is not None:
            store.append_approval_history({"event": "approval_request", **payload})
        return payload

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
        self,
        key: ApprovalKey,
        *,
        data: dict[str, object] | None = None,
        store: ConversationStore | None = None,
    ) -> dict[str, object] | None:
        """Return one terminal payload, or None after it was already emitted."""

        tracked = self._by_key.get(key)
        if tracked is None or tracked.ended:
            return None
        tracked.ended = True
        payload = tracked.end_payload(data or {})
        if tracked.delegated and store is not None:
            store.append_approval_history({"event": "approval_end", **payload})
        return payload

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


def durable_end_data(
    policy: ApprovalPolicy | None, key: object, data: dict[str, object] | None
) -> dict[str, object]:
    """Add the durable decision (and the abort reason) to an approval end."""
    data = dict(data or {})
    if policy is not None and isinstance(key, (str, tuple)):
        resolution = policy.durable_resolution(key)
        if resolution is not None:
            state = resolution["decision"]
            data.setdefault("decision", state)
            if state == "abort" and "reason" not in data:
                data["reason"] = resolution.get("reason", "turn_canceled")
    return data


def end_durably(
    lifecycle: ApprovalLifecycle,
    policy: ApprovalPolicy | None,
    key: ApprovalKey,
    store: ConversationStore | None,
    data: dict[str, object] | None,
) -> dict[str, object] | None:
    """Build one live end from durable state and record delegated identity."""

    return lifecycle.end(
        key, data=durable_end_data(policy, key, data), store=store
    )


def abort_pending(
    policy: ApprovalPolicy | None, foreground_only: bool, reason: str
) -> None:
    """Abort the policy's pending approvals with one durable reason."""
    if policy is None:
        return
    for request in policy.pending_requests():
        if foreground_only and request.child_instance_id is not None:
            continue
        with contextlib.suppress(ValueError, RuntimeError):
            policy.abort(request.key, reason=reason)
