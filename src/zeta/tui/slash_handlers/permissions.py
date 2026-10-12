"""Read-only presentation of the active session's permission policy."""

from __future__ import annotations

from pathlib import Path

from ...config.tool_policy import ToolPolicy
from ...core.approval import ApprovalPolicy


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
        lines.extend(
            [
                f"Default: {policy.default.value}",
                "Rule precedence: deny > ask > allow > default",
                "The default applies to calls requiring approval; some tools/actions are exempt.",
                "--yolo sets the default to allow; --no-yolo sets it to ask.",
                "Explicit rules still apply in either mode.",
            ]
        )
        for label, rules in (
            ("Deny rules", policy.always_deny),
            ("Ask rules", policy.always_ask),
            ("Allow rules (including session grants)", policy.always_allow),
            ("Session grants (/always)", policy.session_allow),
        ):
            lines.append(f"{label}:")
            lines.extend(f"  {rule}" for rule in sorted(map(str, rules)))
            if not rules:
                lines.append("  none")
        if policy.notices:
            lines.append("Ignored approval rules:")
            lines.extend(f"  {notice}" for notice in policy.notices)
    lines.extend(
        [
            "",
            "/approve [key] allows one request; /deny [key] denies it.",
            "/always [key] grants the entire tool or resolved action, with any arguments.",
            "For example: bash permits any command; inbox(send) permits sending to any project.",
            "Session grants still lose to deny/ask rules and reset when the session is reopened.",
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
