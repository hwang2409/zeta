"""Inspect permissions and revoke grants for the active session."""

from __future__ import annotations

from pathlib import Path

from ...config.tool_policy import ToolPolicy
from ...core.approval import (
    ApprovalPolicy,
    PermissionSnapshot,
    SessionGrantChange,
    SessionGrantNotFound,
    permission_snapshot,
    revoke_session_grants,
)


def format_session_grant_change(change: SessionGrantChange) -> str:
    if change.change == "grant":
        return f"permissions · session grant added: {', '.join(change.changed)}"
    if change.change == "revoke":
        return f"permissions · session grant revoked: {', '.join(change.changed)}"
    if change.change == "clear":
        return f"permissions · session grants cleared: {', '.join(change.changed)}"
    return ""


def revoke_permissions(policy: ApprovalPolicy | None, args: str) -> str:
    """Handle explicit revocation commands; never edit settings or pending calls."""

    parts = args.strip().split(maxsplit=1)
    if not (parts == ["clear"] or len(parts) == 2 and parts[0] == "revoke"):
        return "usage: /permissions | /permissions revoke <exact grant> | /permissions clear"
    if policy is None:
        return "permissions · no approval policy is attached"
    if parts == ["clear"]:
        change = revoke_session_grants(policy, clear=True)
        if not change.changed:
            return "permissions · no session grants to clear"
    else:
        target = parts[1]
        try:
            change = revoke_session_grants(policy, exact=target)
        except SessionGrantNotFound:
            return (
                f"permissions · no session grant matches {target!r}; "
                "use /permissions to list exact grants"
            )
    snapshot = change.snapshot
    return "\n".join(
        [
            format_session_grant_change(change),
            "Remaining allow rules: " + (", ".join(snapshot.allow) or "none"),
            f"Default: {snapshot.default.value}; {' > '.join(snapshot.precedence)} still applies.",
            "Remaining allow rules or default allow can still permit calls; revocation adds no deny rule.",
            "Applies to future parent and delegated-child calls. Pending approvals and already authorized calls are unchanged.",
        ]
    )


def format_permissions(
    policy: ApprovalPolicy | None,
    tool_policy: ToolPolicy,
    *,
    settings_path: Path,
    plan_mode: bool,
) -> str:
    lines = ["Permissions · current session (read-only)"]
    if policy is None:
        lines.append("Approval policy: none attached")
    else:
        snapshot = permission_snapshot(policy)
        lines.extend(_format_policy(snapshot))
        if policy.notices:
            lines.append("Ignored approval rules:")
            lines.extend(f"  {notice}" for notice in policy.notices)
    lines.extend(
        [
            "",
            "/approve [key] allows one request; /deny [key] denies it.",
            "/always [key] grants the entire tool or resolved action, with any arguments.",
            "For example: bash permits any command; inbox(send) permits sending to any project.",
            "Session grants apply to future parent and delegated-child calls, still lose to deny/ask rules, and reset when the session is reopened.",
            "Delegated approvals cannot create session grants.",
            "Revoke one: /permissions revoke <exact grant> (for example, bash or inbox(send)).",
            "Revoke all session grants: /permissions clear. Configured rules remain unchanged.",
            f"Persistent approval rules: {settings_path} [approval] allow / deny / ask.",
            "Use scoped rules such as bash(git status*) to allow only matching commands.",
            "Project settings cannot set approval rules. /always does not edit settings.",
            "",
            "Tool availability (separate from approval):",
        ]
    )
    if tool_policy.allow_layers:
        lines.append("  Allow layers (a call must match every layer):")
        for layer in tool_policy.allow_layers:
            lines.append("    " + (", ".join(layer) or "none (blocks all tools)"))
    else:
        lines.append("  Allow: all registered tools")
    lines.append("  Deny: " + (", ".join(tool_policy.deny) or "none"))
    lines.extend(
        [
            f"  Plan mode: {'on (read-only tools)' if plan_mode else 'off'}",
            "Approval grants cannot enable unavailable tools. Use /tools to inspect tools.",
        ]
    )
    return "\n".join(lines)


def _format_policy(snapshot: PermissionSnapshot) -> list[str]:
    lines = [
        f"Default: {snapshot.default.value}",
        f"Rule precedence: {' > '.join(snapshot.precedence)}",
        "The default applies to calls requiring approval; some tools/actions are exempt.",
        "--yolo sets the default to allow; --no-yolo sets it to ask.",
        "Explicit rules still apply in either mode.",
    ]
    for label, rules in (
        ("Deny rules", snapshot.deny),
        ("Ask rules", snapshot.ask),
        ("Allow rules (including session grants)", snapshot.allow),
        ("Session grants (/always)", snapshot.session_grants),
    ):
        lines.append(f"{label}:")
        lines.extend(f"  {rule}" for rule in rules)
        if not rules:
            lines.append("  none")
    return lines
