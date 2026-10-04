"""Pure feature logic for enhanced computer-use observations."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from typing import Any

from backend import MODEL_HEIGHT, MODEL_WIDTH, PHYSICAL_HEIGHT, PHYSICAL_WIDTH

FEATURES = frozenset({"batch", "zoom", "observe", "cursor", "settle"})
MAX_BATCH_ACTIONS = 10
ACTION_COORDINATES = {
    "click": (("x", "x"), ("y", "y")),
    "double_click": (("x", "x"), ("y", "y")),
    "drag": (("x1", "x"), ("y1", "y"), ("x2", "x"), ("y2", "y")),
    "scroll": (("x", "x"), ("y", "y")),
}
ACTION_FIELDS = {
    "click": frozenset({"type", "x", "y", "button"}),
    "double_click": frozenset({"type", "x", "y"}),
    "drag": frozenset({"type", "x1", "y1", "x2", "y2"}),
    "type": frozenset({"type", "text"}),
    "key": frozenset({"type", "keys"}),
    "scroll": frozenset({"type", "x", "y", "dx", "dy"}),
    "wait": frozenset({"type", "seconds"}),
}
ACTION_REQUIRED = {
    "click": frozenset({"type", "x", "y"}),
    "double_click": frozenset({"type", "x", "y"}),
    "drag": frozenset({"type", "x1", "y1", "x2", "y2"}),
    "type": frozenset({"type", "text"}),
    "key": frozenset({"type", "keys"}),
    "scroll": frozenset({"type", "x", "y", "dx", "dy"}),
    "wait": frozenset({"type", "seconds"}),
}


def parse_features(value: str | None) -> frozenset[str]:
    """Parse the opt-in comma list and reject misspelled feature names."""
    enabled = frozenset(
        item.strip() for item in (value or "").split(",") if item.strip()
    )
    unknown = enabled - FEATURES
    if unknown:
        raise ValueError(
            f"unknown ZETA_COMPUTER_FEATURES: {', '.join(sorted(unknown))}"
        )
    return enabled


def _number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a number")
    return float(value)


def model_crop(arguments: dict[str, object]) -> tuple[int, int, int, int]:
    """Validate a model-frame crop and return physical x, y, width, height."""
    x = _number(arguments.get("x"), "x")
    y = _number(arguments.get("y"), "y")
    width = _number(arguments.get("w"), "w")
    height = _number(arguments.get("h"), "h")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("zoom x/y must be nonnegative and w/h must be positive")
    if x + width > MODEL_WIDTH or y + height > MODEL_HEIGHT:
        raise ValueError(
            f"zoom crop must fit inside model frame {MODEL_WIDTH}x{MODEL_HEIGHT}"
        )
    left = round(x * PHYSICAL_WIDTH / MODEL_WIDTH)
    top = round(y * PHYSICAL_HEIGHT / MODEL_HEIGHT)
    right = round((x + width) * PHYSICAL_WIDTH / MODEL_WIDTH)
    bottom = round((y + height) * PHYSICAL_HEIGHT / MODEL_HEIGHT)
    return left, top, max(1, right - left), max(1, bottom - top)


def zoom_legend(arguments: dict[str, object]) -> str:
    """Explain how pixels in an enlarged crop map to global model coordinates."""
    x, y, width, height = (float(arguments[key]) for key in ("x", "y", "w", "h"))
    return (
        f"Zoom shows global model-frame region x={x:g}..{x + width:g}, "
        f"y={y:g}..{y + height:g}, enlarged to {MODEL_WIDTH}x{MODEL_HEIGHT}. "
        "All action tools still require GLOBAL model-frame coordinates, not enlarged-image "
        "pixel coordinates. Convert an enlarged pixel (u,v) with "
        f"global_x={x:g}+u*{width:g}/{MODEL_WIDTH}, "
        f"global_y={y:g}+v*{height:g}/{MODEL_HEIGHT}."
    )


def physical_bounds_to_model(bounds: dict[str, object]) -> dict[str, int]:
    """Convert a physical X11 rectangle to bounded model-frame coordinates."""
    x = round(int(bounds["x"]) * MODEL_WIDTH / PHYSICAL_WIDTH)
    y = round(int(bounds["y"]) * MODEL_HEIGHT / PHYSICAL_HEIGHT)
    width = round(int(bounds["width"]) * MODEL_WIDTH / PHYSICAL_WIDTH)
    height = round(int(bounds["height"]) * MODEL_HEIGHT / PHYSICAL_HEIGHT)
    model_x = min(MODEL_WIDTH - 1, max(0, x))
    model_y = min(MODEL_HEIGHT - 1, max(0, y))
    return {
        "x": model_x,
        "y": model_y,
        "w": min(MODEL_WIDTH - model_x, max(0, width)),
        "h": min(MODEL_HEIGHT - model_y, max(0, height)),
    }


def validate_action(
    action: object, coordinate: Callable[..., int]
) -> dict[str, object]:
    """Validate one complete batch action without executing it."""
    if type(action) is not dict or type(action.get("type")) is not str:
        raise ValueError("each batch action must be an object with a type")
    action_type = action["type"]
    if action_type not in ACTION_FIELDS:
        raise ValueError(f"unsupported batch action type: {action_type}")
    fields = set(action)
    missing = ACTION_REQUIRED[action_type] - fields
    extra = fields - ACTION_FIELDS[action_type]
    if missing:
        raise ValueError(f"{action_type} missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(
            f"{action_type} has unknown fields: {', '.join(sorted(extra))}"
        )
    for field, axis in ACTION_COORDINATES.get(action_type, ()):
        coordinate(action[field], axis=axis)
    normalized = dict(action)
    normalized.pop("type")
    return normalized


def validate_batch(
    arguments: dict[str, object], coordinate: Callable[..., int]
) -> list[tuple[str, dict[str, object]]]:
    """Validate every batch item before returning any executable action."""
    actions = arguments.get("actions")
    if type(actions) is not list or not 1 <= len(actions) <= MAX_BATCH_ACTIONS:
        raise ValueError(f"actions must contain 1..{MAX_BATCH_ACTIONS} items")
    if set(arguments) - {"actions", "screenshot"}:
        raise ValueError("computer_batch accepts only actions and screenshot")
    if "screenshot" in arguments and type(arguments["screenshot"]) is not bool:
        raise ValueError("screenshot must be a boolean")
    validated = []
    for index, action in enumerate(actions):
        try:
            assert isinstance(action, object)
            action_type = action.get("type") if type(action) is dict else "unknown"
            validated.append((str(action_type), validate_action(action, coordinate)))
        except ValueError as exc:
            raise ValueError(f"actions[{index}]: {exc}") from exc
    return validated


def frame_difference(first: bytes, second: bytes) -> float:
    """Return the fraction of grayscale pixel intensity that changed."""
    if len(first) != len(second) or not first:
        return 1.0
    difference = sum(
        abs(left - right) for left, right in zip(first, second, strict=True)
    )
    return difference / (255 * len(first))


def wait_for_stable(
    capture: Callable[[], bytes],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    interval: float = 0.1,
    timeout: float = 2.0,
    threshold: float = 0.002,
) -> float:
    """Wait for two consecutive near-identical sampled frames, with a hard cap."""
    started = monotonic()
    previous = capture()
    while monotonic() - started < timeout:
        sleep(interval)
        current = capture()
        if frame_difference(previous, current) <= threshold:
            return max(0.0, monotonic() - started)
        previous = current
    return max(0.0, monotonic() - started)


def format_observation(
    raw: dict[str, Any], screenshot: bytes, previous_hash: str | None
) -> tuple[str, str]:
    """Return stable structured observation JSON and the current frame hash."""
    current_hash = hashlib.sha256(screenshot).hexdigest()[:16]
    observation = {
        "active_window": raw.get("active_window"),
        "windows": raw.get("windows", []),
        "focused_widget": raw.get("focused_widget"),
        "mouse": raw.get("mouse"),
        "screen_change": {
            "hash": current_hash,
            "previous_hash": previous_hash,
            "changed": None if previous_hash is None else current_hash != previous_hash,
        },
    }
    return json.dumps(observation, separators=(",", ":"), sort_keys=True), current_hash
