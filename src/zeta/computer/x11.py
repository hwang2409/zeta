"""Desktop actions for an X11 guest reached through a command transport.

``X11Desktop`` turns validated model-frame actions into xdotool, ImageMagick,
and AT-SPI commands that run inside the guest. A backend subclass supplies only
``_exec`` and the lifecycle; any guest image with the same tools reuses this
module unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from .actions import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    model_coordinate,
    physical_bounds_to_model,
)
from .backend import Screenshot
from .observe import wait_for_stable

DISPLAY = ":99"
SCREENSHOT_COMMAND = (
    "import",
    "-window",
    "root",
    "-resize",
    f"{MODEL_WIDTH}x{MODEL_HEIGHT}!",
    "-quality",
    "75",
    "jpeg:-",
)
SETTLE_SAMPLE_COMMAND = (
    "import",
    "-window",
    "root",
    "-resize",
    "64x40!",
    "-colorspace",
    "gray",
    "gray:-",
)
# Typing a newline with ``xdotool type`` is unreliable; send Return instead.
TYPE_SCRIPT = (
    "text=$1; while case $text in *$'\\n'*) true;; *) false;; esac; do "
    "head=${text%%$'\\n'*}; xdotool type --clearmodifiers --delay 1 -- "
    "\"$head\"; xdotool key Return; text=${text#*$'\\n'}; done; "
    'xdotool type --clearmodifiers --delay 1 -- "$text"'
)
MAX_FOCUSED_TEXT = 50_000
MAX_WINDOWS = 50

# Runs inside the guest. It reads window metadata and the focused widget's
# accessible text. It never reads the clipboard.
OBSERVE_SCRIPT = r"""import json, re, subprocess

def run(*args):
    value = subprocess.run(args, text=True, capture_output=True, check=False)
    return value.stdout.strip()

def geometry(window):
    fields = {}
    for line in run("xdotool", "getwindowgeometry", "--shell", window).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if value.lstrip("-").isdigit():
                fields[key] = int(value)
    return {"x": fields.get("X", 0), "y": fields.get("Y", 0),
            "width": fields.get("WIDTH", 0), "height": fields.get("HEIGHT", 0)}

def focused_widget():
    try:
        import pyatspi

        pending = [pyatspi.Registry.getDesktop(0)]
        visited = 0
        while pending and visited < 10000:
            node = pending.pop()
            visited += 1
            try:
                state = node.getState()
                if state.contains(pyatspi.STATE_FOCUSED):
                    multiline = None
                    if state.contains(pyatspi.STATE_MULTI_LINE):
                        multiline = True
                    elif state.contains(pyatspi.STATE_SINGLE_LINE):
                        multiline = False
                    value = None
                    try:
                        text = node.queryText()
                        value = text.getText(0, min(text.characterCount, LIMIT))
                    except Exception:
                        pass
                    return {"role": node.getRoleName(), "name": node.name or None,
                            "value": value, "multiline": multiline,
                            "truncated": value is not None and len(value) == LIMIT}
                pending.extend(reversed([node[i] for i in range(node.childCount)]))
            except Exception:
                continue
    except Exception:
        pass
    return None

active = run("xdotool", "getwindowfocus")
windows = []
stacking = run("xprop", "-root", "_NET_CLIENT_LIST_STACKING")
for hexadecimal in re.findall(r"0x[0-9a-fA-F]+", stacking)[:WINDOWS]:
    window = str(int(hexadecimal, 16))
    windows.append({"id": window, "title": run("xdotool", "getwindowname", window),
                    "bounds": geometry(window)})
mouse = {}
for line in run("xdotool", "getmouselocation", "--shell").splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        mouse[key] = value
print(json.dumps({"active_id": active,
                  "active_title": run("xdotool", "getwindowname", active),
                  "windows": windows,
                  "focused_widget": focused_widget(),
                  "mouse": {"x": int(mouse.get("X", 0)), "y": int(mouse.get("Y", 0))}}))
""".replace("LIMIT", str(MAX_FOCUSED_TEXT)).replace("WINDOWS", str(MAX_WINDOWS))


def input_command(action: str, arguments: Mapping[str, object]) -> tuple[str, ...]:
    """Return the guest command for one validated model-frame action."""

    if action in {"click", "double_click"}:
        x = model_coordinate(arguments["x"], axis="x")
        y = model_coordinate(arguments["y"], axis="y")
        button = {"left": "1", "middle": "2", "right": "3"}[str(arguments.get("button", "left"))]
        repeat = "2" if action == "double_click" else "1"
        return (
            "xdotool", "mousemove", str(x), str(y),
            "click", "--repeat", repeat, "--delay", "100", button,
        )
    if action == "drag":
        x1 = model_coordinate(arguments["x1"], axis="x")
        y1 = model_coordinate(arguments["y1"], axis="y")
        x2 = model_coordinate(arguments["x2"], axis="x")
        y2 = model_coordinate(arguments["y2"], axis="y")
        return (
            "xdotool", "mousemove", str(x1), str(y1), "mousedown", "1",
            "mousemove", "--sync", str(x2), str(y2), "mouseup", "1",
        )
    if action == "type":
        return ("bash", "-c", TYPE_SCRIPT, "zeta-type", str(arguments["text"]))
    if action == "key":
        return ("xdotool", "key", "--clearmodifiers", str(arguments["keys"]))
    if action == "wait":
        return ("sleep", str(float(arguments["seconds"])))
    if action == "scroll":
        x = model_coordinate(arguments["x"], axis="x")
        y = model_coordinate(arguments["y"], axis="y")
        command = ["xdotool", "mousemove", str(x), str(y)]
        for delta, negative, positive in (
            (float(arguments["dy"]), "4", "5"),
            (float(arguments["dx"]), "6", "7"),
        ):
            if delta:
                clicks = min(100, max(1, round(abs(delta) / 100)))
                command += ["click", "--repeat", str(clicks), positive if delta > 0 else negative]
        return tuple(command)
    raise ValueError(f"unsupported action: {action}")


def model_observation(raw: Mapping[str, object]) -> dict[str, object]:
    """Convert raw guest window metadata to model-frame geometry."""

    windows = raw.get("windows")
    windows = windows if isinstance(windows, list) else []
    active_bounds = next(
        (
            physical_bounds_to_model(item["bounds"])
            for item in windows
            if item.get("id") == raw.get("active_id")
        ),
        None,
    )
    mouse = raw.get("mouse")
    mouse = mouse if isinstance(mouse, dict) else {}
    pointer = physical_bounds_to_model(
        {"x": int(mouse.get("x", 0)), "y": int(mouse.get("y", 0)), "width": 0, "height": 0}
    )
    return {
        "active_window": {"title": raw.get("active_title"), "bounds": active_bounds},
        "windows": [
            {"title": item["title"], "bounds": physical_bounds_to_model(item["bounds"])}
            for item in windows
            if item.get("title")
        ],
        "focused_widget": raw.get("focused_widget"),
        "mouse": {"x": pointer["x"], "y": pointer["y"]},
    }


class X11Desktop:
    """Screenshot, input, observe, and settle over an abstract guest exec."""

    def _exec(self, command: Sequence[str], *, stdin: bytes | None = None) -> bytes:
        """Run one command in the guest with ``DISPLAY`` set; raise on failure."""

        raise NotImplementedError

    def screenshot(self) -> Screenshot:
        data = self._exec(SCREENSHOT_COMMAND)
        if not data.startswith(b"\xff\xd8"):
            raise RuntimeError("desktop returned an invalid JPEG screenshot")
        return Screenshot(data)

    def input(self, action: str, arguments: Mapping[str, object]) -> None:
        self._exec(input_command(action, arguments))

    def observe(self) -> dict[str, object]:
        raw = json.loads(self._exec(("python3", "-c", OBSERVE_SCRIPT)))
        if type(raw) is not dict:
            raise RuntimeError("desktop returned an invalid observation")
        return model_observation(raw)

    def settle(self) -> float:
        return wait_for_stable(lambda: self._exec(SETTLE_SAMPLE_COMMAND))


__all__ = ["DISPLAY", "X11Desktop", "input_command", "model_observation"]
