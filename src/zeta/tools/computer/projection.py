"""Bounded computer state shared by transcript and serve surfaces."""

from __future__ import annotations

from collections.abc import Mapping

CONTROL_STATES = frozenset({"active", "idle", "denied", "killed"})
KILL_SHORTCUT = "ctrl+opt+cmd+esc"
_MAX_ACTION = 64
_MAX_BUNDLE_ID = 255
_MAX_SHORTCUT = 64


def _optional_text(value: object, limit: int) -> str | None:
    return value if isinstance(value, str) and value and len(value) <= limit else None


def computer_state(
    tool_name: object,
    arguments: object,
    structured_content: object = None,
    *,
    running: bool,
) -> dict[str, str] | None:
    """Return the fixed public state for one computer call, or ``None``."""

    structured = structured_content if isinstance(structured_content, Mapping) else {}
    raw_state = structured.get("computer")
    state = raw_state if isinstance(raw_state, Mapping) else {}
    if not (
        isinstance(tool_name, str) and tool_name.strip().lower() == "computer"
    ) and not state:
        return None

    args = arguments if isinstance(arguments, Mapping) else {}
    action = _optional_text(state.get("action"), _MAX_ACTION)
    if action is None:
        action = _optional_text(args.get("action"), _MAX_ACTION) or "action"
    control = "active" if running else state.get("control", "idle")
    if control not in CONTROL_STATES:
        control = "idle"

    projected = {"action": action, "control": control}
    shortcut = (
        KILL_SHORTCUT
        if running
        else _optional_text(state.get("kill_switch"), _MAX_SHORTCUT)
    )
    bundle = _optional_text(state.get("frontmost_bundle_id"), _MAX_BUNDLE_ID)
    if shortcut is not None:
        projected["kill_switch"] = shortcut
    if bundle is not None:
        projected["frontmost_bundle_id"] = bundle
    return projected
