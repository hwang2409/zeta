"""provider-neutral native desktop computer tool."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from typing import Any

from ...media.image_normalization import normalize_image_bytes
from ...protocol.types import StructuredToolResult
from .._results import _error_result, text_block
from ..registry import ToolRegistry
from .backend import ComputerBackend, ComputerError, ScreenState
from .scaling import (
    OutOfBounds,
    ScreenGeometry,
    downscale_cap,
    to_backend_point,
    to_model_point,
)

_CLICK_ACTIONS = {"left_click": ("left", 1), "right_click": ("right", 1), "middle_click": ("middle", 1), "double_click": ("left", 2), "triple_click": ("left", 3)}
_SCROLL_UNIT = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_REFUSAL_REASON = {"locked": "the screen is locked", "secure_input": "a password field or secure input is active", "killed": "the kill switch is engaged; restart the helper to re-enable control"}


class _GuardError(ComputerError):
    def __init__(self, code: str, message: str, state: ScreenState) -> None:
        super().__init__(code, message)
        self.state = state


class _BadAction(ValueError):
    pass


def _point(geometry: ScreenGeometry | None, args: dict[str, Any], field: str = "coordinate") -> tuple[int, int]:
    if geometry is None:
        raise _BadAction("take a screenshot first")
    value = args.get(field)
    if not isinstance(value, list) or len(value) != 2 or not all(type(v) is int for v in value):
        raise _BadAction(f"{field} must be an [x, y] integer pair")
    try:
        return to_backend_point(geometry, value[0], value[1])
    except OutOfBounds as exc:
        raise _BadAction(str(exc)) from exc


def app_access(registry: ToolRegistry, bundle_id: str | None) -> bool:
    config = registry.computer_config
    return config.bundle_ids.get(bundle_id or "", config.default_app_access) == "allow"


def _state(action: str, state: ScreenState | None, control: str, screenshot: bool) -> dict[str, Any]:
    return {"computer": {"action": action, "control": control, "kill_switch": "ctrl+opt+cmd+esc", "frontmost_bundle_id": state.frontmost_bundle_id if state else None, "has_screenshot": screenshot}}


def computer_result(action: str, *, state: ScreenState | None, control: str, content: list[dict[str, Any]], error: bool, screenshot: bool = False) -> StructuredToolResult:
    return {"content": content, "isError": error, "structuredContent": _state(action, state, control, screenshot)}


async def _guard(registry: ToolRegistry, backend: ComputerBackend) -> ScreenState:
    state = await backend.state()
    if state.locked:
        raise _GuardError("locked", _REFUSAL_REASON["locked"], state)
    if state.secure_input:
        raise _GuardError("secure_input", _REFUSAL_REASON["secure_input"], state)
    if not app_access(registry, state.frontmost_bundle_id):
        bundle = state.frontmost_bundle_id or "<unknown>"
        raise _GuardError("not_allowed", f'frontmost app {bundle!r} is not allowed; add [computer.apps] "{bundle}" = "allow"', state)
    return state


async def _capture(registry: ToolRegistry, backend: ComputerBackend, display: int | None) -> tuple[dict[str, Any], ScreenState]:
    state = await _guard(registry, backend)
    frame = await backend.screenshot(display)
    cap = downscale_cap(frame.width, frame.height)
    existing = registry.image_policy.max_dimension
    policy = replace(registry.image_policy, max_dimension=cap if existing is None else min(cap, existing))
    prepared = await normalize_image_bytes(frame.image, policy)
    registry.computer_geometry = ScreenGeometry(frame.width, frame.height, prepared.sent_width, prepared.sent_height, frame.scale, frame.origin_x, frame.origin_y)
    image = {"type": "image", "data": base64.b64encode(prepared.data).decode("ascii"), "mimeType": prepared.media_type, "size": len(prepared.data)}
    if prepared.sent_width is not None:
        image["width"] = prepared.sent_width
    if prepared.sent_height is not None:
        image["height"] = prepared.sent_height
    receipt = f"screenshot {prepared.sent_width}x{prepared.sent_height} {prepared.media_type} scale={frame.scale} origin={frame.origin_x},{frame.origin_y}"
    return {"content": [text_block(receipt), image], "isError": False}, state


async def _input(registry: ToolRegistry, backend: ComputerBackend, action: str, args: dict[str, Any]) -> str | None:
    geometry = registry.computer_geometry
    if action in _CLICK_ACTIONS:
        x, y = _point(geometry, args); button, count = _CLICK_ACTIONS[action]; await backend.click(x, y, button, count)
    elif action == "mouse_move":
        await backend.move(*_point(geometry, args))
    elif action in {"left_mouse_down", "left_mouse_up"}:
        await getattr(backend, "mouse_down" if action.endswith("down") else "mouse_up")(*_point(geometry, args), "left")
    elif action == "left_click_drag":
        await backend.drag([_point(geometry, args, "start_coordinate"), _point(geometry, args)])
    elif action == "type":
        text = args.get("text")
        if type(text) is not str or not text: raise _BadAction("type requires nonempty text")
        await backend.type_text(text)
    elif action == "key":
        text = args.get("text")
        if type(text) is not str or not text: raise _BadAction("key requires a nonempty key combo in text")
        await backend.key(text)
    elif action == "scroll":
        x, y = _point(geometry, args); direction = args.get("scroll_direction"); amount = args.get("scroll_amount", 1)
        if direction not in _SCROLL_UNIT: raise _BadAction("scroll_direction must be up, down, left, or right")
        if type(amount) is not int or amount < 1: raise _BadAction("scroll_amount must be a positive integer")
        dx, dy = _SCROLL_UNIT[direction]; await backend.scroll(x, y, dx * amount, dy * amount)
    elif action == "cursor_position":
        x, y = to_model_point(geometry, *await backend.cursor_position()) if geometry else (_ for _ in ()).throw(_BadAction("take a screenshot before reading the cursor position"))
        return f"cursor_position x={x} y={y}"
    elif action == "wait":
        await asyncio.sleep(1)
    return None


_ACTIONS = ("screenshot", *(_CLICK_ACTIONS), "mouse_move", "left_click_drag", "left_mouse_down", "left_mouse_up", "type", "key", "scroll", "cursor_position", "wait")


async def run_action(registry: ToolRegistry, backend: ComputerBackend, action: str, args: dict[str, Any]) -> StructuredToolResult:
    state: ScreenState | None = None
    input_done = False
    capture_started = False
    try:
        state = await _guard(registry, backend)
        input_done = action not in {"screenshot", "wait", "cursor_position"}
        text = await _input(registry, backend, action, args)
        if action == "cursor_position":
            result = computer_result(action, state=state, control="idle", content=[text_block(text or "")], error=False)
            result["structuredContent"].update({"x": int((text or "").split("x=", 1)[1].split()[0]), "y": int((text or "").split("y=", 1)[1])})
            return result
        capture_started = True
        payload, capture_state = await _capture(registry, backend, args.get("display"))
        payload["structuredContent"] = _state(action, capture_state, "idle", True)
        return payload
    except _GuardError as exc:
        control = "denied" if exc.code == "not_allowed" else "killed" if exc.code == "killed" else "idle"
        message = f"screenshot failed: {exc.message}" if input_done else f"{action} failed: {exc.message}"
        return computer_result(action, state=exc.state, control=control, content=[text_block(message)], error=not input_done or not capture_started)
    except _BadAction as exc:
        return computer_result(action, state=state, control="idle", content=[text_block(str(exc))], error=True)
    except ComputerError as exc:
        control = "killed" if exc.code == "killed" else "idle"
        message = _REFUSAL_REASON.get(exc.code, exc.message)
        prefix = f"{action} done; screenshot failed" if input_done and capture_started else f"{action} failed"
        return computer_result(action, state=getattr(exc, "state", state), control=control, content=[text_block(f"{prefix}: {message}")], error=not input_done or not capture_started)
    except (ValueError, OSError) as exc:
        return computer_result(action, state=state, control="idle", content=[text_block(f"{action} failed: {exc}")], error=True)


async def _computer(registry: ToolRegistry, arguments: dict[str, Any]) -> StructuredToolResult:
    action = arguments.get("action", "action")
    backend = registry.computer_backend
    if backend is None:
        return _error_result("no computer backend is available on this machine", kind="error")
    if action not in _ACTIONS:
        return _error_result(f"unknown action {action!r}", kind="invalid_arguments")
    return await run_action(registry, backend, action, arguments)


def register(registry: ToolRegistry) -> None:
    if not registry.computer_enabled:
        return
    registry.register_session_tool("computer", _computer, binding="loop", approval_subject="action", description="Control the native desktop: capture a screenshot, move/click the mouse, type, press keys, drag, and scroll. Actions that change the screen return a fresh screenshot.", parameters={"type": "object", "properties": {"action": {"type": "string", "enum": list(_ACTIONS)}, "coordinate": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "start_coordinate": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "text": {"type": "string"}, "scroll_direction": {"type": "string", "enum": list(_SCROLL_UNIT)}, "scroll_amount": {"type": "integer", "minimum": 1}, "display": {"type": "integer", "minimum": 0}}, "required": ["action"], "additionalProperties": False})
