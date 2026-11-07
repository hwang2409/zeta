"""Serve protocol adapter for active session approval grants."""

from __future__ import annotations

from typing import Any

from ..core.approval import ApprovalPolicy
from ..permissions import (
    PermissionSnapshot,
    SessionGrantChange,
    SessionGrantNotFound,
    permission_snapshot,
    revoke_session_grants,
)
from .protocol import ProtocolError

FEATURE = "session_grants"
READ = "session_grants"
REVOKE = "revoke_session_grants"
EVENT = "session_grants_changed"


def require_feature(features: frozenset[str]) -> None:
    if FEATURE not in features:
        raise ProtocolError(
            -32601, "session grants require the negotiated session_grants feature"
        )


def read(
    policy: ApprovalPolicy | None, session_id: str | None, params: dict[str, Any]
) -> dict[str, object]:
    if params:
        raise ProtocolError(
            -32602,
            "session_grants takes no params",
            {"code": "invalid_session_grants_params"},
        )
    if policy is None or session_id is None:
        raise ProtocolError(-32003, "no active session")
    return snapshot_result(permission_snapshot(policy), session_id)


def revoke(
    policy: ApprovalPolicy | None,
    session_id: str | None,
    params: dict[str, Any],
) -> tuple[dict[str, object], SessionGrantChange]:
    if policy is None or session_id is None:
        raise ProtocolError(-32003, "no active session")
    valid_grant = (
        set(params) == {"grant"}
        and isinstance(params.get("grant"), str)
        and bool(params["grant"])
    )
    valid_clear = set(params) == {"clear"} and params.get("clear") is True
    if not (valid_grant or valid_clear):
        raise ProtocolError(
            -32602,
            "invalid revoke_session_grants params",
            {"code": "invalid_session_grants_revoke"},
        )
    try:
        change = (
            revoke_session_grants(policy, exact=params["grant"])
            if valid_grant
            else revoke_session_grants(policy, clear=True)
        )
    except SessionGrantNotFound as exc:
        raise ProtocolError(
            -32602,
            "session grant not found",
            {"code": "session_grant_not_found", "grant": exc.grant},
        ) from exc
    return {
        "removed": list(change.changed),
        "grants": snapshot_result(change.snapshot, session_id),
    }, change


def event_data(change: SessionGrantChange, session_id: str) -> dict[str, object]:
    return {
        "change": change.change,
        "changed": list(change.changed),
        "grants": snapshot_result(change.snapshot, session_id),
    }


def snapshot_result(snapshot: PermissionSnapshot, session_id: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "default": snapshot.default.value,
        "precedence": list(snapshot.precedence),
        "rules": {
            "deny": list(snapshot.deny),
            "ask": list(snapshot.ask),
            "allow": list(snapshot.allow),
        },
        "session_grants": list(snapshot.session_grants),
    }
