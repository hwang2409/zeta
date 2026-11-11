from __future__ import annotations

from typing import Literal

from .approval import (
    PRECEDENCE,
    ApprovalPolicy,
    PermissionSnapshot,
    SessionGrantChange,
    SessionGrantNotFound,
)


def permission_snapshot(policy: ApprovalPolicy) -> PermissionSnapshot:
    """Return a canonical projection of the active approval policy."""

    return PermissionSnapshot(
        default=policy.default,
        precedence=PRECEDENCE,
        deny=tuple(sorted(map(str, policy.always_deny))),
        ask=tuple(sorted(map(str, policy.always_ask))),
        allow=tuple(sorted(map(str, policy.always_allow))),
        session_grants=tuple(sorted(map(str, policy.session_allow))),
    )


def revoke_session_grants(
    policy: ApprovalPolicy, *, exact: str | None = None, clear: bool = False
) -> SessionGrantChange:
    """Remove only existing session grants; never parse input into authority."""

    if (exact is None and not clear) or (exact is not None and clear):
        raise ValueError("choose exactly one session grant operation")
    if clear:
        removed = tuple(sorted(map(str, policy.clear_session_allow())))
        return SessionGrantChange("clear", removed, permission_snapshot(policy))
    rule = next((rule for rule in policy.session_allow if str(rule) == exact), None)
    if rule is None:
        assert exact is not None
        raise SessionGrantNotFound(exact)
    policy.revoke_session_allow(rule)
    return SessionGrantChange("revoke", (str(rule),), permission_snapshot(policy))


def session_grant_change(
    policy: ApprovalPolicy,
    change: Literal["grant", "reset"],
    changed: tuple[str, ...] = (),
) -> SessionGrantChange:
    return SessionGrantChange(change, changed, permission_snapshot(policy))
