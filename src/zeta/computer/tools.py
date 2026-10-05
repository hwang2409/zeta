"""Model-facing MCP tool definitions for the computer server.

The tool set is the benchmarked default ("BOS"): the eight single actions plus
``batch``, with a structured observation and a settle wait on every result.
"""

from __future__ import annotations

from .actions import (
    ACTION_REQUIRED,
    MAX_BATCH_ACTIONS,
    MAX_KEYS_LENGTH,
    MAX_WAIT_SECONDS,
    MODEL_HEIGHT,
    MODEL_WIDTH,
)

SERVER_NAME = "computer"
FRAME = f"model frame {MODEL_WIDTH}x{MODEL_HEIGHT}"
SANDBOX = "the isolated, networkless sandbox desktop (not the user's computer)"
BEHAVIOR = (
    " Each result waits up to 2 seconds for the screen to settle, then returns a fresh "
    f"{MODEL_WIDTH}x{MODEL_HEIGHT} screenshot and an observation JSON with the active "
    "window, window titles and bounds in model-frame coordinates, the focused widget, "
    "the pointer, and whether the screen changed."
)
X_COORD = {"type": "number", "minimum": 0, "exclusiveMaximum": MODEL_WIDTH}
Y_COORD = {"type": "number", "minimum": 0, "exclusiveMaximum": MODEL_HEIGHT}
BUTTON = {"type": "string", "enum": ["left", "middle", "right"]}
KEYS = {"type": "string", "minLength": 1, "maxLength": MAX_KEYS_LENGTH}
SECONDS = {"type": "number", "minimum": 0, "maximum": MAX_WAIT_SECONDS}


def _schema(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool(name: str, description: str, schema: dict[str, object]) -> dict[str, object]:
    return {"name": name, "description": description + BEHAVIOR, "inputSchema": schema}


TOOLS: tuple[dict[str, object], ...] = (
    _tool(
        "screenshot",
        f"Take a fresh screenshot of {SANDBOX}. Coordinates use the {FRAME}.",
        _schema({}, []),
    ),
    _tool(
        "click",
        f"Click in {SANDBOX}. Coordinates use the {FRAME}.",
        _schema({"x": X_COORD, "y": Y_COORD, "button": BUTTON}, ["x", "y"]),
    ),
    _tool(
        "double_click",
        f"Double-click in {SANDBOX}. Coordinates use the {FRAME}.",
        _schema({"x": X_COORD, "y": Y_COORD}, ["x", "y"]),
    ),
    _tool(
        "drag",
        f"Drag with the left button in {SANDBOX}. Coordinates use the {FRAME}.",
        _schema(
            {"x1": X_COORD, "y1": Y_COORD, "x2": X_COORD, "y2": Y_COORD},
            ["x1", "y1", "x2", "y2"],
        ),
    ),
    _tool(
        "type",
        f"Type text into the focused control in {SANDBOX}. Newlines press Return. "
        "No clipboard is used.",
        _schema({"text": {"type": "string"}}, ["text"]),
    ),
    _tool(
        "key",
        f"Send an xdotool key chord, such as ctrl+s or Return, to {SANDBOX}.",
        _schema({"keys": KEYS}, ["keys"]),
    ),
    _tool(
        "scroll",
        f"Scroll at a point in {SANDBOX}. Coordinates use the {FRAME}; "
        "positive dy scrolls down, about 100 per wheel step.",
        _schema(
            {"x": X_COORD, "y": Y_COORD, "dx": {"type": "number"}, "dy": {"type": "number"}},
            ["x", "y", "dx", "dy"],
        ),
    ),
    _tool(
        "wait",
        f"Wait up to {MAX_WAIT_SECONDS:g} seconds in {SANDBOX}.",
        _schema({"seconds": SECONDS}, ["seconds"]),
    ),
    _tool(
        "batch",
        f"Run 1 to {MAX_BATCH_ACTIONS} actions in order in {SANDBOX}, then return one "
        f"final screenshot. All coordinates are global {FRAME} coordinates. Every action "
        "is validated before the first one runs; a runtime error stops the batch and the "
        "result lists the status of each action. Use it for a known sequence, such as "
        "click a field, type, and press a key.",
        _schema(
            {
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_BATCH_ACTIONS,
                    "items": {
                        "type": "object",
                        "description": "One click, double_click, drag, type, key, "
                        "scroll, or wait action with that tool's fields.",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": sorted(ACTION_REQUIRED),
                            },
                            "x": X_COORD,
                            "y": Y_COORD,
                            "x1": X_COORD,
                            "y1": Y_COORD,
                            "x2": X_COORD,
                            "y2": Y_COORD,
                            "button": BUTTON,
                            "text": {"type": "string"},
                            "keys": KEYS,
                            "dx": {"type": "number"},
                            "dy": {"type": "number"},
                            "seconds": SECONDS,
                        },
                    },
                },
                "screenshot": {"type": "boolean", "default": True},
            },
            ["actions"],
        ),
    ),
)
TOOL_NAMES: tuple[str, ...] = tuple(str(tool["name"]) for tool in TOOLS)
QUALIFIED_TOOL_NAMES: tuple[str, ...] = tuple(f"{SERVER_NAME}__{name}" for name in TOOL_NAMES)
SINGLE_ACTIONS = frozenset(ACTION_REQUIRED)

__all__ = ["QUALIFIED_TOOL_NAMES", "SERVER_NAME", "SINGLE_ACTIONS", "TOOLS", "TOOL_NAMES"]
