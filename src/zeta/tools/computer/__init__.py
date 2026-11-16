"""The provider-neutral ``computer`` tool: native desktop capture and input.

Opt-in: not registered unless ``[computer] enabled = true`` in settings. One
``ComputerBackend`` is the seam to the OS; tests inject a ``FakeBackend`` on
``registry.computer_backend``. There is no real backend yet (slice 2), so an
enabled tool with no backend returns a clear error. Actions that change the
screen return a fresh screenshot. Each screenshot is downscaled to a
model-friendly size and its geometry recorded, so model coordinates map back to
global display points through ``scaling``.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from ...media.image_normalization import normalize_image_bytes
from ...protocol.types import StructuredToolResult
from .._results import _error_result, text_block
from ..registry import ToolRegistry
from .backend import ComputerBackend, ComputerError
from .scaling import (
    OutOfBounds,
    ScreenGeometry,
    downscale_cap,
    to_backend_point,
    to_model_point,
)

# action -> (button, click count)
_CLICK_ACTIONS: dict[str, tuple[str, int]] = {
    "left_click": ("left", 1),
    "right_click": ("right", 1),
    "middle_click": ("middle", 1),
    "double_click": ("left", 2),
    "triple_click": ("left", 3),
}
# scroll direction -> unit (dx, dy); multiplied by scroll_amount
_SCROLL_UNIT: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
_ACTIONS = (
    "screenshot", *(_CLICK_ACTIONS), "mouse_move", "left_click_drag",
    "left_mouse_down", "left_mouse_up", "type", "key", "scroll",
    "cursor_position", "wait",
)
# backend error code -> human reason for the per-action failure message.
_REFUSAL_REASON: dict[str, str] = {
    "locked": "the screen is locked",
    "secure_input": "a password field or secure input is active",
    "killed": "the kill switch is engaged; restart the helper to re-enable control",
}


class _BadAction(ValueError):
    """Action-specific arguments were missing or malformed."""


def _point(
    geometry: ScreenGeometry | None,
    arguments: dict[str, Any],
    field: str = "coordinate",
) -> tuple[int, int]:
    if geometry is None:
        raise _BadAction("take a screenshot first")
    value = arguments.get(field)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(type(component) is int for component in value)
    ):
        raise _BadAction(f"{field} must be an [x, y] integer pair")
    try:
        return to_backend_point(geometry, value[0], value[1])
    except OutOfBounds as exc:
        raise _BadAction(str(exc)) from exc


def app_access(registry: ToolRegistry, bundle_id: str | None) -> bool:
    config = registry.computer_config
    return config.bundle_ids.get(bundle_id or "", config.default_app_access) == "allow"


def _state_content(action: str, state: Any, *, control: str, has_screenshot: bool) -> dict[str, Any]:
    return {"computer": {"action": action, "control": control, "kill_switch": "ctrl+opt+cmd+esc", "frontmost_bundle_id": state.frontmost_bundle_id if state else None, "has_screenshot": has_screenshot}}


async def _guard(registry: ToolRegistry, backend: ComputerBackend, action: str) -> Any:
    state = await backend.state()
    if state.locked:
        raise ComputerError("locked", "the screen is locked")
    if state.secure_input:
        raise ComputerError("secure_input", "a password field or secure input is active")
    if not app_access(registry, state.frontmost_bundle_id):
        bundle = state.frontmost_bundle_id or "<unknown>"
        raise ComputerError("not_allowed", f'frontmost app {bundle!r} is not allowed; add [computer.macos] bundle_ids = {{ "{bundle}" = "allow" }}')
    return state


async def _screenshot(
    registry: ToolRegistry, backend: ComputerBackend, display: int | None, action: str = "screenshot"
) -> StructuredToolResult:
    state = await _guard(registry, backend, action)
    frame = await backend.screenshot(display)
    # Downscale to the WXGA box so the model always sees one small coordinate
    # space; the provider's own cap still applies if it is smaller.
    cap = downscale_cap(frame.width, frame.height)
    existing = registry.image_policy.max_dimension
    policy = replace(
        registry.image_policy,
        max_dimension=cap if existing is None else min(cap, existing),
    )
    prepared = await normalize_image_bytes(frame.image, policy)
    registry.computer_geometry = ScreenGeometry(
        frame.width,
        frame.height,
        prepared.sent_width,
        prepared.sent_height,
        frame.scale,
        frame.origin_x,
        frame.origin_y,
    )
    image: dict[str, Any] = {
        "type": "image",
        "data": base64.b64encode(prepared.data).decode("ascii"),
        "mimeType": prepared.media_type,
        "size": len(prepared.data),
    }
    if prepared.sent_width is not None:
        image["width"] = prepared.sent_width
    if prepared.sent_height is not None:
        image["height"] = prepared.sent_height
    receipt = (
        f"screenshot {prepared.sent_width}x{prepared.sent_height} "
        f"{prepared.media_type} scale={frame.scale} "
        f"origin={frame.origin_x},{frame.origin_y}"
    )
    return {
        "content": [text_block(receipt), image],
        "isError": False,
        "structuredContent": {
            "width": prepared.sent_width,
            "height": prepared.sent_height,
            "scale": frame.scale,
            "origin_x": frame.origin_x,
            "origin_y": frame.origin_y,
            **_state_content(action, state, control="active", has_screenshot=True),
        },
    }


async def _run_input(registry: ToolRegistry, backend: ComputerBackend, action: str, arguments: dict[str, Any]) -> StructuredToolResult:
    await _guard(registry, backend, action)
    geometry = registry.computer_geometry
    if action in _CLICK_ACTIONS:
        x, y = _point(geometry, arguments); button, count = _CLICK_ACTIONS[action]
        await backend.click(x, y, button, count)
    elif action == "mouse_move":
        x, y = _point(geometry, arguments); await backend.move(x, y)
    elif action == "left_mouse_down":
        x, y = _point(geometry, arguments); await backend.mouse_down(x, y, "left")
    elif action == "left_mouse_up":
        x, y = _point(geometry, arguments); await backend.mouse_up(x, y, "left")
    elif action == "left_click_drag":
        await backend.drag([_point(geometry, arguments, "start_coordinate"), _point(geometry, arguments)])
    elif action == "type":
        text = arguments.get("text")
        if type(text) is not str or not text: raise _BadAction("type requires nonempty text")
        await backend.type_text(text)
    elif action == "key":
        combo = arguments.get("text")
        if type(combo) is not str or not combo: raise _BadAction("key requires a nonempty key combo in text")
        await backend.key(combo)
    elif action == "scroll":
        x, y = _point(geometry, arguments); direction = arguments.get("scroll_direction")
        if direction not in _SCROLL_UNIT: raise _BadAction("scroll_direction must be up, down, left, or right")
        amount = arguments.get("scroll_amount", 1)
        if type(amount) is not int or amount < 1: raise _BadAction("scroll_amount must be a positive integer")
        dx, dy = _SCROLL_UNIT[direction]; await backend.scroll(x, y, dx * amount, dy * amount)
    return await _screenshot(registry, backend, None, action)


async def _cursor(registry: ToolRegistry, backend: ComputerBackend, arguments: dict[str, Any]) -> StructuredToolResult:
    state = await _guard(registry, backend, "cursor_position")
    geometry = registry.computer_geometry
    if geometry is None: raise _BadAction("take a screenshot before reading the cursor position")
    x, y = to_model_point(geometry, *await backend.cursor_position())
    return {"content": [text_block(f"cursor_position x={x} y={y}")], "isError": False, "structuredContent": {"x": x, "y": y, **_state_content("cursor_position", state, control="active", has_screenshot=False)}}


async def _wait(registry: ToolRegistry, backend: ComputerBackend, arguments: dict[str, Any]) -> StructuredToolResult:
    await asyncio.sleep(1)
    return await _screenshot(registry, backend, None, "wait")


_ACTION_HANDLERS: dict[str, Callable[[ToolRegistry, ComputerBackend, dict[str, Any]], Awaitable[StructuredToolResult]]] = {
    "screenshot": lambda r, b, a: _screenshot(r, b, a.get("display")),
    **{name: (lambda r, b, a, name=name: _run_input(r, b, name, a)) for name in (*_CLICK_ACTIONS, "mouse_move", "left_click_drag", "left_mouse_down", "left_mouse_up", "type", "key", "scroll")},
    "cursor_position": _cursor,
    "wait": _wait,
}
_ACTIONS = tuple(_ACTION_HANDLERS)


async def _dispatch(registry: ToolRegistry, backend: ComputerBackend, action: str, arguments: dict[str, Any]) -> StructuredToolResult:
    handler = _ACTION_HANDLERS.get(action)
    if handler is None: raise _BadAction(f"unknown action {action!r}")
    return await handler(registry, backend, arguments)


async def _computer(registry: ToolRegistry, arguments: dict[str, Any]) -> StructuredToolResult:
    backend = registry.computer_backend
    action = arguments.get("action", "action")
    if backend is None: return _error_result("no computer backend is available on this machine", kind="error")
    try:
        return await _dispatch(registry, backend, arguments["action"], arguments)
    except _BadAction as exc:
        return _error_result(str(exc), kind="invalid_arguments")
    except ComputerError as exc:
        reason = _REFUSAL_REASON.get(exc.code, exc.message)
        if exc.code == "not_allowed": reason = exc.message
        result = _error_result(f"{action} failed: {reason}", kind="error")
        result["structuredContent"] = {"computer": {"action": action, "control": "denied", "kill_switch": "ctrl+opt+cmd+esc", "frontmost_bundle_id": None, "has_screenshot": False}}
        return result
    except ValueError as exc:
        return _error_result(f"{action} failed: {exc}", kind="error")

def register(registry: ToolRegistry) -> None:
    if not registry.computer_enabled:
        return
    registry.register_session_tool(
        "computer",
        _computer,
        binding="loop",
        approval_subject="action",
        description=(
            "Control the native desktop: capture a screenshot, move/click the "
            "mouse, type, press keys, drag, and scroll. Actions that change the "
            "screen return a fresh screenshot."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(_ACTIONS)},
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "start_coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "text": {"type": "string"},
                "scroll_direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                },
                "scroll_amount": {"type": "integer", "minimum": 1},
                "display": {"type": "integer", "minimum": 0},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    )
