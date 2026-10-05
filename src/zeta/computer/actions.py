"""Model-frame geometry and action validation for the computer tools.

The model always sees a fixed 1024x640 frame. The guest display is 1280x800.
Every coordinate the model sends is validated in the model frame and scaled to
the physical display here, so no backend repeats the rule.
"""

from __future__ import annotations

from collections.abc import Mapping

PHYSICAL_WIDTH = 1280
PHYSICAL_HEIGHT = 800
MODEL_WIDTH = 1024
MODEL_HEIGHT = 640
MAX_BATCH_ACTIONS = 10
MAX_KEYS_LENGTH = 100
MAX_WAIT_SECONDS = 5.0
MAX_SCROLL_DELTA = 10_000

ACTION_COORDINATES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "click": (("x", "x"), ("y", "y")),
    "double_click": (("x", "x"), ("y", "y")),
    "drag": (("x1", "x"), ("y1", "y"), ("x2", "x"), ("y2", "y")),
    "scroll": (("x", "x"), ("y", "y")),
}
ACTION_REQUIRED: Mapping[str, frozenset[str]] = {
    "click": frozenset({"x", "y"}),
    "double_click": frozenset({"x", "y"}),
    "drag": frozenset({"x1", "y1", "x2", "y2"}),
    "type": frozenset({"text"}),
    "key": frozenset({"keys"}),
    "scroll": frozenset({"x", "y", "dx", "dy"}),
    "wait": frozenset({"seconds"}),
}
ACTION_OPTIONAL: Mapping[str, frozenset[str]] = {"click": frozenset({"button"})}
BUTTONS = ("left", "middle", "right")


def _real(value: object) -> float | None:
    if type(value) is int or type(value) is float:
        return float(value)
    return None


def model_coordinate(value: object, *, axis: str) -> int:
    """Validate one model-frame coordinate and return its physical pixel."""

    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    limit = MODEL_WIDTH if axis == "x" else MODEL_HEIGHT
    physical = PHYSICAL_WIDTH if axis == "x" else PHYSICAL_HEIGHT
    number = _real(value)
    if number is None:
        raise ValueError(f"{axis} must be a number in model frame 0..{limit - 1}")
    if not 0 <= number < limit:
        raise ValueError(f"{axis}={value} is outside model frame 0..{limit - 1}")
    return min(physical - 1, max(0, round(number * physical / limit)))


def physical_bounds_to_model(bounds: Mapping[str, int]) -> dict[str, int]:
    """Convert a physical X11 rectangle to a clamped model-frame rectangle."""

    x = round(bounds["x"] * MODEL_WIDTH / PHYSICAL_WIDTH)
    y = round(bounds["y"] * MODEL_HEIGHT / PHYSICAL_HEIGHT)
    width = round(bounds["width"] * MODEL_WIDTH / PHYSICAL_WIDTH)
    height = round(bounds["height"] * MODEL_HEIGHT / PHYSICAL_HEIGHT)
    model_x = min(MODEL_WIDTH - 1, max(0, x))
    model_y = min(MODEL_HEIGHT - 1, max(0, y))
    return {
        "x": model_x,
        "y": model_y,
        "w": min(MODEL_WIDTH - model_x, max(0, width)),
        "h": min(MODEL_HEIGHT - model_y, max(0, height)),
    }


def _number(arguments: Mapping[str, object], name: str) -> float:
    number = _real(arguments.get(name))
    if number is None:
        raise ValueError(f"{name} must be a number")
    return number


def validate_action(action: str, arguments: Mapping[str, object]) -> dict[str, object]:
    """Validate one action completely, without side effects.

    Returns a copy of the arguments. Every field is checked here, so a batch
    rejects a bad item before the first item executes.
    """

    required = ACTION_REQUIRED.get(action)
    if required is None:
        raise ValueError(f"unsupported action: {action}")
    fields = set(arguments)
    missing = required - fields
    if missing:
        raise ValueError(f"{action} missing fields: {', '.join(sorted(missing))}")
    extra = fields - required - ACTION_OPTIONAL.get(action, frozenset())
    if extra:
        raise ValueError(f"{action} has unknown fields: {', '.join(sorted(extra))}")
    for field, axis in ACTION_COORDINATES.get(action, ()):
        model_coordinate(arguments[field], axis=axis)
    if "button" in arguments and arguments["button"] not in BUTTONS:
        raise ValueError("button must be left, middle, or right")
    if action == "type" and type(arguments["text"]) is not str:
        raise ValueError("text must be a string")
    if action == "key":
        keys = arguments["keys"]
        if type(keys) is not str or not keys or len(keys) > MAX_KEYS_LENGTH:
            raise ValueError(
                "keys must be a nonempty xdotool key string of at most "
                f"{MAX_KEYS_LENGTH} characters"
            )
    if action == "wait":
        seconds = _number(arguments, "seconds")
        if not 0 <= seconds <= MAX_WAIT_SECONDS:
            raise ValueError(f"seconds must be between 0 and {MAX_WAIT_SECONDS:g}")
    if action == "scroll":
        for name in ("dx", "dy"):
            if abs(_number(arguments, name)) > MAX_SCROLL_DELTA:
                raise ValueError(
                    f"dx and dy must be between -{MAX_SCROLL_DELTA} and {MAX_SCROLL_DELTA}"
                )
    return dict(arguments)


def validate_batch(arguments: Mapping[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Validate a whole batch before any item executes."""

    if set(arguments) - {"actions", "screenshot"}:
        raise ValueError("batch accepts only actions and screenshot")
    if "screenshot" in arguments and type(arguments["screenshot"]) is not bool:
        raise ValueError("screenshot must be a boolean")
    actions = arguments.get("actions")
    if type(actions) is not list or not 1 <= len(actions) <= MAX_BATCH_ACTIONS:
        raise ValueError(f"actions must contain 1..{MAX_BATCH_ACTIONS} items")
    validated: list[tuple[str, dict[str, object]]] = []
    for index, item in enumerate(actions):
        try:
            if type(item) is not dict or type(item.get("type")) is not str:
                raise ValueError("each batch action must be an object with a type")
            fields = dict(item)
            action = str(fields.pop("type"))
            validated.append((action, validate_action(action, fields)))
        except ValueError as exc:
            raise ValueError(f"actions[{index}]: {exc}") from exc
    return validated


__all__ = [
    "MAX_BATCH_ACTIONS",
    "MODEL_HEIGHT",
    "MODEL_WIDTH",
    "PHYSICAL_HEIGHT",
    "PHYSICAL_WIDTH",
    "model_coordinate",
    "physical_bounds_to_model",
    "validate_action",
    "validate_batch",
]
