#!/usr/bin/env python3
"""Minimal line-delimited JSON-RPC MCP server for a disposable desktop."""

from __future__ import annotations

import base64
import json
import os
import signal
import sys
from typing import TextIO

from backend import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    DesktopBackend,
    DockerDesktopBackend,
    model_coordinate,
)
from features import (
    format_observation,
    model_crop,
    parse_features,
    validate_batch,
    zoom_legend,
)

FRAME = f"model frame {MODEL_WIDTH}x{MODEL_HEIGHT}"
SANDBOX = "the isolated, networkless sandboxed VM desktop"


def _schema(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


X_COORD = {"type": "number", "minimum": 0, "exclusiveMaximum": MODEL_WIDTH}
Y_COORD = {"type": "number", "minimum": 0, "exclusiveMaximum": MODEL_HEIGHT}


def _base_tools() -> list[dict[str, object]]:
    return [
        {
            "name": "computer_screenshot",
            "description": f"Take a fresh screenshot of {SANDBOX}. Coordinates use the {FRAME}.",
            "inputSchema": _schema({}, []),
        },
        {
            "name": "computer_click",
            "description": f"Click in {SANDBOX}; returns a fresh screenshot. Coordinates use the {FRAME}.",
            "inputSchema": _schema(
                {
                    "x": X_COORD,
                    "y": Y_COORD,
                    "button": {"type": "string", "enum": ["left", "middle", "right"]},
                },
                ["x", "y"],
            ),
        },
        {
            "name": "computer_double_click",
            "description": f"Double-click in {SANDBOX}; returns a fresh screenshot. Coordinates use the {FRAME}.",
            "inputSchema": _schema({"x": X_COORD, "y": Y_COORD}, ["x", "y"]),
        },
        {
            "name": "computer_drag",
            "description": f"Drag in {SANDBOX}; returns a fresh screenshot. Coordinates use the {FRAME}.",
            "inputSchema": _schema(
                {"x1": X_COORD, "y1": Y_COORD, "x2": X_COORD, "y2": Y_COORD},
                ["x1", "y1", "x2", "y2"],
            ),
        },
        {
            "name": "computer_type",
            "description": (
                f"Type into the focused control in {SANDBOX}; returns a fresh "
                f"{MODEL_WIDTH}x{MODEL_HEIGHT} screenshot. No clipboard is used."
            ),
            "inputSchema": _schema({"text": {"type": "string"}}, ["text"]),
        },
        {
            "name": "computer_key",
            "description": (
                f"Send an xdotool key chord, such as ctrl+s, to {SANDBOX}; returns "
                f"a fresh {MODEL_WIDTH}x{MODEL_HEIGHT} screenshot."
            ),
            "inputSchema": _schema(
                {"keys": {"type": "string", "minLength": 1, "maxLength": 100}},
                ["keys"],
            ),
        },
        {
            "name": "computer_scroll",
            "description": (
                f"Scroll at a point in {SANDBOX}; returns a fresh screenshot. "
                f"Coordinates use the {FRAME}; positive dy scrolls down."
            ),
            "inputSchema": _schema(
                {
                    "x": X_COORD,
                    "y": Y_COORD,
                    "dx": {"type": "number"},
                    "dy": {"type": "number"},
                },
                ["x", "y", "dx", "dy"],
            ),
        },
        {
            "name": "computer_wait",
            "description": (
                f"Wait up to 5 seconds for {SANDBOX}, then return a fresh "
                f"{MODEL_WIDTH}x{MODEL_HEIGHT} screenshot."
            ),
            "inputSchema": _schema(
                {"seconds": {"type": "number", "minimum": 0, "maximum": 5}},
                ["seconds"],
            ),
        },
    ]


def tools_for(features: frozenset[str]) -> list[dict[str, object]]:
    """Return the legacy tools plus only explicitly enabled feature tools."""
    tools = _base_tools()
    if "batch" in features:
        action = {
            "type": "object",
            "description": "One click, double_click, drag, type, key, scroll, or wait action.",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "click",
                        "double_click",
                        "drag",
                        "type",
                        "key",
                        "scroll",
                        "wait",
                    ],
                },
                "x": X_COORD,
                "y": Y_COORD,
                "x1": X_COORD,
                "y1": Y_COORD,
                "x2": X_COORD,
                "y2": Y_COORD,
                "button": {"type": "string", "enum": ["left", "middle", "right"]},
                "text": {"type": "string"},
                "keys": {"type": "string"},
                "dx": {"type": "number"},
                "dy": {"type": "number"},
                "seconds": {"type": "number", "minimum": 0, "maximum": 5},
            },
        }
        tools.append(
            {
                "name": "computer_batch",
                "description": (
                    f"Execute 1 to 10 actions in order in {SANDBOX}, then return one final screenshot. "
                    f"All coordinates are global {FRAME} coordinates. All coordinates are validated before execution; runtime errors stop the batch."
                ),
                "inputSchema": _schema(
                    {
                        "actions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "items": action,
                        },
                        "screenshot": {"type": "boolean", "default": True},
                    },
                    ["actions"],
                ),
            }
        )
    if "zoom" in features:
        tools.append(
            {
                "name": "computer_zoom",
                "description": (
                    f"Crop a region given in global {FRAME} coordinates and enlarge it to the model frame. "
                    "The result includes conversion instructions; action coordinates remain GLOBAL, not crop pixels."
                ),
                "inputSchema": _schema(
                    {
                        "x": X_COORD,
                        "y": Y_COORD,
                        "w": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": MODEL_WIDTH,
                        },
                        "h": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": MODEL_HEIGHT,
                        },
                    },
                    ["x", "y", "w", "h"],
                ),
            }
        )
    notes = []
    if "observe" in features:
        notes.append(
            "screenshot results include structured window, mouse, and screen-change data"
        )
    if "cursor" in features:
        notes.append("screenshots visibly mark the current pointer")
    if "settle" in features:
        notes.append("actions wait up to 2 seconds for a stable screen")
    if notes:
        suffix = " Enabled behavior: " + "; ".join(notes) + "."
        for tool in tools:
            tool["description"] = str(tool["description"]) + suffix
    return tools


class ComputerServer:
    def __init__(
        self, backend: DesktopBackend, features: frozenset[str] | None = None
    ) -> None:
        self.backend = backend
        self.features = (
            parse_features(os.environ.get("ZETA_COMPUTER_FEATURES"))
            if features is None
            else features
        )
        self.tools = tools_for(self.features)
        self.previous_hash: str | None = None

    def call(self, name: str, arguments: object) -> dict[str, object]:
        if type(arguments) is not dict:
            return self.error("arguments must be an object")
        try:
            self.backend.start()
            if name == "computer_batch" and "batch" in self.features:
                return self._batch(arguments)
            if name == "computer_zoom" and "zoom" in self.features:
                crop = model_crop(arguments)
                return self._screenshot(crop=crop, messages=[zoom_legend(arguments)])
            coordinate_fields = {
                "computer_click": (("x", "x"), ("y", "y")),
                "computer_double_click": (("x", "x"), ("y", "y")),
                "computer_drag": (("x1", "x"), ("y1", "y"), ("x2", "x"), ("y2", "y")),
                "computer_scroll": (("x", "x"), ("y", "y")),
            }
            for field, axis in coordinate_fields.get(name, ()):
                model_coordinate(arguments[field], axis=axis)
            if name == "computer_screenshot":
                if arguments:
                    raise ValueError("computer_screenshot takes no arguments")
                return self._screenshot()
            actions = {
                "computer_click": "click",
                "computer_double_click": "double_click",
                "computer_drag": "drag",
                "computer_type": "type",
                "computer_key": "key",
                "computer_scroll": "scroll",
                "computer_wait": "wait",
            }
            action = actions.get(name)
            if action is None:
                raise ValueError(f"unknown or disabled tool: {name}")
            self.backend.input(action, arguments)
            messages = []
            if "settle" in self.features:
                messages.append(
                    f"Screen settled in {self.backend.settle():.3f} seconds."
                )
            return self._screenshot(messages=messages)
        except (KeyError, ValueError, RuntimeError) as exc:
            return self.error(str(exc))

    def _batch(self, arguments: dict[str, object]) -> dict[str, object]:
        actions = validate_batch(arguments, model_coordinate)
        results: list[dict[str, object]] = []
        failed = False
        for index, (action, action_arguments) in enumerate(actions):
            try:
                self.backend.input(action, action_arguments)
                results.append({"index": index, "type": action, "status": "ok"})
            except (KeyError, ValueError, RuntimeError) as exc:
                results.append(
                    {
                        "index": index,
                        "type": action,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                failed = True
                break
        messages = ["Batch results: " + json.dumps(results, separators=(",", ":"))]
        if "settle" in self.features:
            messages.append(f"Screen settled in {self.backend.settle():.3f} seconds.")
        if arguments.get("screenshot", True):
            return self._screenshot(messages=messages, is_error=failed)
        return {
            "content": [{"type": "text", "text": message} for message in messages],
            "isError": failed,
        }

    def _screenshot(
        self,
        *,
        crop: tuple[int, int, int, int] | None = None,
        messages: list[str] | None = None,
        is_error: bool = False,
    ) -> dict[str, object]:
        screenshot = self.backend.screenshot(
            crop=crop, cursor="cursor" in self.features
        )
        text = [
            f"Fresh {MODEL_WIDTH}x{MODEL_HEIGHT} screenshot ({len(screenshot.data)} bytes).",
            *(messages or []),
        ]
        if "observe" in self.features:
            observation, self.previous_hash = format_observation(
                self.backend.observe(), screenshot.data, self.previous_hash
            )
            text.append("Observation: " + observation)
        encoded = base64.b64encode(screenshot.data).decode("ascii")
        return {
            "content": [
                *({"type": "text", "text": message} for message in text),
                {"type": "image", "data": encoded, "mimeType": screenshot.media_type},
            ],
            "isError": is_error,
        }

    @staticmethod
    def error(message: str) -> dict[str, object]:
        return {"content": [{"type": "text", "text": message}], "isError": True}


def _write(sink: TextIO, payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":")), file=sink, flush=True)


def serve(
    backend: DesktopBackend,
    source: TextIO = sys.stdin,
    sink: TextIO = sys.stdout,
    *,
    destroy_on_exit: bool = True,
) -> None:
    server = ComputerServer(backend)
    try:
        for line in source:
            request_id: object = None
            try:
                request = json.loads(line)
                if type(request) is not dict:
                    continue
                request_id = request.get("id")
                method = request.get("method")
                if method == "notifications/initialized":
                    continue
                if method == "notifications/exit":
                    break
                if method == "initialize":
                    result: object = {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "zeta-computer-mcp",
                            "version": "0.1",
                        },
                    }
                elif method == "tools/list":
                    result = {"tools": server.tools}
                elif method == "tools/call":
                    params = request.get("params")
                    if type(params) is not dict or type(params.get("name")) is not str:
                        raise ValueError("tools/call requires a tool name")
                    result = server.call(params["name"], params.get("arguments", {}))
                elif method == "shutdown":
                    result = None
                else:
                    raise ValueError(f"unsupported method: {method}")
                if request_id is not None:
                    _write(
                        sink,
                        {"jsonrpc": "2.0", "id": request_id, "result": result},
                    )
            except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                if request_id is not None:
                    _write(
                        sink,
                        {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32602, "message": str(exc)},
                        },
                    )
    finally:
        if destroy_on_exit:
            backend.destroy()


if __name__ == "__main__":
    desktop = DockerDesktopBackend()

    def _terminate(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    serve(desktop)
