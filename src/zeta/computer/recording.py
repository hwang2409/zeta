"""Append-only recording of one computer session for replay and spectating.

Layout of a recording directory::

    metadata.json   version, active flag, start/end time, model frame size
    events.jsonl    one JSON object per tool call
    frames/         numbered JPEG screenshots referenced by events
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from .actions import MODEL_HEIGHT, MODEL_WIDTH, model_coordinate

MAX_SUMMARY_CHARS = 4000
RECORDING_VERSION = 1


def _coordinates(arguments: Mapping[str, object]) -> dict[str, object] | None:
    """Describe action points in both the model and the physical frame."""

    points: dict[str, object] = {}
    for label, x_key, y_key in (("point", "x", "y"), ("start", "x1", "y1"), ("end", "x2", "y2")):
        if x_key not in arguments or y_key not in arguments:
            continue
        try:
            physical = {
                "x": model_coordinate(arguments[x_key], axis="x"),
                "y": model_coordinate(arguments[y_key], axis="y"),
            }
        except ValueError:
            continue
        points[label] = {
            "model": {"x": arguments[x_key], "y": arguments[y_key]},
            "physical": physical,
        }
    return points or None


def event_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Copy tool arguments and add explicit coordinate conversions."""

    result = dict(arguments)
    coordinates = _coordinates(arguments)
    if coordinates:
        result["_coordinates"] = coordinates
    actions = arguments.get("actions")
    if type(actions) is list:
        result["actions"] = [
            event_arguments(action) if type(action) is dict else action for action in actions
        ]
    return result


def result_summary(result: Mapping[str, object]) -> str:
    """Return the bounded text content of a tool result, without images."""

    content = result.get("content")
    messages = [
        str(item.get("text", ""))
        for item in (content if isinstance(content, list) else [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    summary = " ".join(messages)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1] + "…"
    return summary


class SessionRecorder:
    """Write frames and tool events for one session."""

    def __init__(self, directory: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.directory = directory
        self.frames = directory / "frames"
        self.events = directory / "events.jsonl"
        self.metadata = directory / "metadata.json"
        self._clock = clock
        self._started = float(clock())
        self._frame_number = 0
        self._pending_frame: str | None = None
        self._closed = False
        self.frames.mkdir(parents=True, exist_ok=True)
        # Continue numbering when a resumed session reuses the directory.
        existing = sorted(self.frames.glob("*.jpg"))
        if existing:
            self._frame_number = int(existing[-1].stem)
        self._write_metadata(active=True)

    def _write_metadata(self, *, active: bool) -> None:
        payload = {
            "version": RECORDING_VERSION,
            "active": active,
            "started_at": datetime.fromtimestamp(self._started, UTC).isoformat(),
            "ended_at": None if active else datetime.now(UTC).isoformat(),
            "model_frame": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
        }
        temporary = self.metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.metadata)

    def record_frame(self, data: bytes) -> None:
        """Store a screenshot; the next ``record_tool`` event references it."""

        self._frame_number += 1
        relative = f"frames/{self._frame_number:06d}.jpg"
        (self.directory / relative).write_bytes(data)
        self._pending_frame = relative

    def record_tool(
        self, tool: str, arguments: Mapping[str, object], result: Mapping[str, object]
    ) -> None:
        now = float(self._clock())
        frame, self._pending_frame = self._pending_frame, None
        event = {
            "ts": datetime.fromtimestamp(now, UTC).isoformat(),
            "elapsed": max(0.0, now - self._started),
            "tool": tool,
            "args": event_arguments(arguments),
            "result": {"error": bool(result.get("isError")), "summary": result_summary(result)},
            "frame": frame,
        }
        with self.events.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._write_metadata(active=False)


def mark_finished(directory: Path) -> None:
    """Mark a recording inactive after its server exited without closing it."""

    path = directory / "metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if type(metadata) is not dict or not metadata.get("active"):
        return
    metadata["active"] = False
    metadata["ended_at"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["SessionRecorder", "event_arguments", "mark_finished", "result_summary"]
