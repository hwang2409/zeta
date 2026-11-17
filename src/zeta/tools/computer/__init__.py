"""provider-neutral native desktop computer tool."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...config.settings import computer_allow_guidance
from ...media.image_normalization import normalize_image_bytes
from ...protocol.types import StructuredToolResult
from .._results import text_block
from ..registry import ToolRegistry
from .backend import Button, ComputerBackend, ComputerError, Modifiers, ScreenState
from .scaling import (
    OutOfBounds,
    ScreenGeometry,
    downscale_cap,
    to_backend_point,
    to_model_point,
)

_SCROLL_UNIT = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_REFUSAL_REASON = {"locked": "the screen is locked", "secure_input": "a password field or secure input is active", "killed": "the kill switch is engaged; restart the helper to re-enable control"}


class _GuardError(ComputerError):
    def __init__(self, code: str, message: str, state: ScreenState) -> None:
        super().__init__(code, message)
        self.state = state


class _BadAction(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Handled:
    content: list[dict[str, Any]] | None = None
    structured: dict[str, Any] | None = None
    screenshot: bool = False


_ActionHandler = Callable[[ToolRegistry, ComputerBackend, dict[str, Any]], Awaitable[_Handled]]


@dataclass(frozen=True, slots=True)
class _ActionSpec:
    handler: _ActionHandler
    followup_capture: bool = False


def _point(geometry: ScreenGeometry | None, args: dict[str, Any], field: str = "coordinate") -> tuple[int, int] | None:
    value = args.get(field)
    if value is None:
        return None
    if geometry is None:
        raise _BadAction("take a screenshot first")
    if not isinstance(value, list) or len(value) != 2 or not all(type(v) is int for v in value):
        raise _BadAction(f"{field} must be an [x, y] integer pair")
    try:
        return to_backend_point(geometry, value[0], value[1])
    except OutOfBounds as exc:
        raise _BadAction(str(exc)) from exc


def _modifiers(args: dict[str, Any]) -> Modifiers:
    value = args.get("modifiers", ())
    if value == ():
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise _BadAction("modifiers must be a list of key names")
    return tuple(value)


def app_access(registry: ToolRegistry, bundle_id: str | None) -> bool:
    config = registry.computer_config
    return config.bundle_ids.get(bundle_id or "", config.default_app_access) == "allow"


async def _guard(registry: ToolRegistry, backend: ComputerBackend) -> ScreenState:
    state = await backend.state()
    if state.locked:
        raise _GuardError("locked", _REFUSAL_REASON["locked"], state)
    if state.secure_input:
        raise _GuardError("secure_input", _REFUSAL_REASON["secure_input"], state)
    if not app_access(registry, state.frontmost_bundle_id):
        bundle = state.frontmost_bundle_id or "<unknown>"
        reason = "the computer allow-list is empty, so every app is denied" if not registry.computer_config.bundle_ids else f"frontmost app {bundle!r} is not allowed"
        guidance = computer_allow_guidance(
            bundle,
            has_existing_table=registry.computer_config.bundle_ids_table_present,
        )
        raise _GuardError("not_allowed", f"{reason}. {guidance}", state)
    return state


async def _capture_frame(registry: ToolRegistry, backend: ComputerBackend, display: int | None) -> _Handled:
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
    return _Handled([text_block(receipt), image], screenshot=True)


async def _screenshot(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    return await _capture_frame(registry, backend, args.get("display"))


def _click(button: Button, count: int) -> _ActionHandler:
    async def handle(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
        point = _point(registry.computer_geometry, args)
        if point is None:
            point = await backend.cursor_position()
        modifiers = _modifiers(args)
        if modifiers:
            await backend.click(*(point or (None, None)), button, count, modifiers)
        else:
            await backend.click(*(point or (None, None)), button, count)
        return _Handled()

    return handle


async def _move(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    point = _point(registry.computer_geometry, args)
    if point is None:
        raise _BadAction("mouse_move requires coordinate")
    await backend.move(*point)
    return _Handled()


async def _drag(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    raw_path = args.get("path")
    if raw_path is not None:
        if not isinstance(raw_path, list) or not raw_path:
            raise _BadAction("path must be a nonempty list of coordinates")
        path = [_point(registry.computer_geometry, {"coordinate": value}) for value in raw_path]
    else:
        start = _point(registry.computer_geometry, args, "start_coordinate")
        end = _point(registry.computer_geometry, args)
        if start is None or end is None:
            raise _BadAction("drag requires start_coordinate and coordinate")
        path = [start, end]
    await backend.drag(path)
    return _Handled()


async def _mouse_down(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    point = _point(registry.computer_geometry, args)
    if point is None:
        raise _BadAction("mouse_down requires coordinate")
    await backend.mouse_down(*point, "left")
    return _Handled()


async def _mouse_up(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    point = _point(registry.computer_geometry, args)
    if point is None:
        raise _BadAction("mouse_up requires coordinate")
    await backend.mouse_up(*point, "left")
    return _Handled()


async def _type_text(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    text = args.get("text")
    if type(text) is not str or not text:
        raise _BadAction("type requires nonempty text")
    await backend.type_text(text)
    return _Handled()


async def _key(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    text = args.get("text")
    if type(text) is not str or not text:
        raise _BadAction("key requires a nonempty key combo in text")
    repeat = args.get("repeat", 1)
    if type(repeat) is not int or not 1 <= repeat <= 100:
        raise _BadAction("repeat must be an integer from 1 to 100")
    hold = args.get("hold", False)
    release = args.get("release", False)
    if type(hold) is not bool or type(release) is not bool or hold and release:
        raise _BadAction("hold and release must be exclusive boolean flags")
    await backend.key(text, repeat, hold, release)
    return _Handled()


async def _scroll(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    point = _point(registry.computer_geometry, args)
    if point is None:
        point = await backend.cursor_position()
    x, y = point
    direction = args.get("scroll_direction")
    amount = args.get("scroll_amount", 1)
    if direction not in _SCROLL_UNIT:
        raise _BadAction("scroll_direction must be up, down, left, or right")
    if type(amount) is not int or amount < 1:
        raise _BadAction("scroll_amount must be a positive integer")
    dx, dy = _SCROLL_UNIT[direction]
    modifiers = _modifiers(args)
    if modifiers:
        await backend.scroll(x, y, dx * amount, dy * amount, modifiers)
    else:
        await backend.scroll(x, y, dx * amount, dy * amount)
    return _Handled()


async def _cursor_position(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    geometry = registry.computer_geometry
    if geometry is None:
        raise _BadAction("take a screenshot before reading the cursor position")
    x, y = to_model_point(geometry, *await backend.cursor_position())
    return _Handled([text_block(f"cursor_position x={x} y={y}")], {"x": x, "y": y})


async def _wait(registry: ToolRegistry, backend: ComputerBackend, args: dict[str, Any]) -> _Handled:
    duration = args.get("duration", 1)
    if type(duration) not in (int, float) or not 0 <= duration <= 10:
        raise _BadAction("duration must be between 0 and 10 seconds")
    await asyncio.sleep(duration)
    return _Handled()


_ACTION_HANDLERS: dict[str, _ActionSpec] = {
    "screenshot": _ActionSpec(_screenshot),
    "left_click": _ActionSpec(_click("left", 1), followup_capture=True),
    "right_click": _ActionSpec(_click("right", 1), followup_capture=True),
    "middle_click": _ActionSpec(_click("middle", 1), followup_capture=True),
    "double_click": _ActionSpec(_click("left", 2), followup_capture=True),
    "triple_click": _ActionSpec(_click("left", 3), followup_capture=True),
    "mouse_move": _ActionSpec(_move, followup_capture=True),
    "left_click_drag": _ActionSpec(_drag, followup_capture=True),
    "left_mouse_down": _ActionSpec(_mouse_down, followup_capture=True),
    "left_mouse_up": _ActionSpec(_mouse_up, followup_capture=True),
    "type": _ActionSpec(_type_text, followup_capture=True),
    "key": _ActionSpec(_key, followup_capture=True),
    "scroll": _ActionSpec(_scroll, followup_capture=True),
    "cursor_position": _ActionSpec(_cursor_position),
    "wait": _ActionSpec(_wait, followup_capture=True),
}


def _result(
    action: str,
    *,
    state: ScreenState | None = None,
    control: str = "idle",
    content: list[dict[str, Any]] | None = None,
    structured: dict[str, Any] | None = None,
    screenshot: bool = False,
    error: bool = True,
) -> StructuredToolResult:
    structured_content = {"computer": {"action": action, "control": control, "kill_switch": "ctrl+opt+cmd+esc", "frontmost_bundle_id": state.frontmost_bundle_id if state else None, "has_screenshot": screenshot}}
    if structured:
        structured_content.update(structured)
    return {"content": content or [], "isError": error, "structuredContent": structured_content}


def _validation_error_result(
    arguments: Mapping[str, object], message: str
) -> StructuredToolResult:
    raw_action = arguments.get("action")
    action = raw_action if isinstance(raw_action, str) else "action"
    return _result(action, content=[text_block(message)])


async def run_action(registry: ToolRegistry, raw_action: object, args: dict[str, Any]) -> StructuredToolResult:
    action = raw_action if isinstance(raw_action, str) else "action"
    state: ScreenState | None = None
    control = "idle"
    content: list[dict[str, Any]] = []
    structured: dict[str, Any] | None = None
    screenshot = False
    error = True
    completed = False
    followup_capture = False
    backend = registry.computer_backend
    spec = _ACTION_HANDLERS.get(action)

    try:
        if not isinstance(raw_action, str):
            raise _BadAction("action must be a string")
        if backend is None:
            raise _BadAction("no computer backend is available on this machine")
        if spec is None:
            raise _BadAction(f"unknown action {action!r}")
        state = await _guard(registry, backend)
        handled = await spec.handler(registry, backend, args)
        completed = True
        if spec.followup_capture:
            followup_capture = True
            state = None
            state = await _guard(registry, backend)
            handled = await _capture_frame(registry, backend, args.get("display"))
        content = handled.content or []
        structured = handled.structured
        screenshot = handled.screenshot
        error = False
    except _GuardError as exc:
        state = exc.state
        control = "denied" if exc.code == "not_allowed" else "killed" if exc.code == "killed" else "idle"
        prefix = f"{action} done; screenshot failed" if completed and followup_capture else f"{action} failed"
        content = [text_block(f"{prefix}: {exc.message}")]
        error = not completed
    except _BadAction as exc:
        content = [text_block(str(exc))]
    except ComputerError as exc:
        control = "killed" if exc.code == "killed" else "idle"
        message = _REFUSAL_REASON.get(exc.code, exc.message)
        prefix = f"{action} done; screenshot failed" if completed and followup_capture else f"{action} failed"
        content = [text_block(f"{prefix}: {message}")]
        error = not completed
    except (ValueError, OSError) as exc:
        prefix = f"{action} done; screenshot failed" if completed and followup_capture else f"{action} failed"
        content = [text_block(f"{prefix}: {exc}")]
        error = not completed

    return _result(
        action,
        state=state,
        control=control,
        content=content,
        structured=structured,
        screenshot=screenshot,
        error=error,
    )


async def _computer(registry: ToolRegistry, arguments: dict[str, Any]) -> StructuredToolResult:
    return await run_action(registry, arguments.get("action"), arguments)


def register(registry: ToolRegistry) -> None:
    if not registry.computer_enabled:
        return
    registry.register_session_tool("computer", _computer, binding="loop", approval_subject="action", description="Control the native desktop: capture a screenshot, move/click the mouse, type, press keys, drag, and scroll. Actions that change the screen return a fresh screenshot.", parameters={"type": "object", "properties": {"action": {"type": "string", "enum": list(_ACTION_HANDLERS)}, "coordinate": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "start_coordinate": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "path": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "minItems": 1}, "text": {"type": "string"}, "modifiers": {"type": "array", "items": {"type": "string"}}, "repeat": {"type": "integer", "minimum": 1, "maximum": 100}, "hold": {"type": "boolean"}, "release": {"type": "boolean"}, "duration": {"type": "number", "minimum": 0, "maximum": 10}, "scroll_direction": {"type": "string", "enum": list(_SCROLL_UNIT)}, "scroll_amount": {"type": "integer", "minimum": 1}, "display": {"type": "integer", "minimum": 0}}, "required": ["action"], "additionalProperties": False}, validation_error_result=_validation_error_result)
