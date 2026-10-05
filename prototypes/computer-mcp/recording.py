"""Host-side session recording for the computer-use prototype."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend import MODEL_HEIGHT, MODEL_WIDTH, model_coordinate


def default_recording_dir() -> Path | None:
    """Return the configured per-session directory, or None when disabled."""
    if os.environ.get("ZETA_COMPUTER_RECORDING", "1").lower() in {"0", "false", "no"}:
        return None
    configured = os.environ.get("ZETA_COMPUTER_RECORDING_DIR")
    if configured:
        return Path(configured)
    home = Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))
    run_id = os.environ.get("ZETA_COMPUTER_RUN_ID") or datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S-%fZ"
    )
    return home / "recordings" / run_id


def _coordinates(arguments: dict[str, object]) -> dict[str, object] | None:
    """Describe action points in both the model and physical display frames."""
    groups: list[tuple[str, tuple[str, str]]] = []
    if "x" in arguments and "y" in arguments:
        groups.append(("point", ("x", "y")))
    if all(key in arguments for key in ("x1", "y1")):
        groups.append(("start", ("x1", "y1")))
    if all(key in arguments for key in ("x2", "y2")):
        groups.append(("end", ("x2", "y2")))
    points: dict[str, object] = {}
    for label, (x_key, y_key) in groups:
        try:
            model_x = arguments[x_key]
            model_y = arguments[y_key]
            points[label] = {
                "model": {"x": model_x, "y": model_y},
                "physical": {
                    "x": model_coordinate(model_x, axis="x"),
                    "y": model_coordinate(model_y, axis="y"),
                },
            }
        except ValueError:
            continue
    return points or None


def event_arguments(arguments: dict[str, object]) -> dict[str, object]:
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


def result_summary(result: dict[str, object]) -> tuple[str, list[str]]:
    """Return bounded text and verification warnings without image payloads."""
    messages = []
    for item in result.get("content", []):
        if type(item) is dict and item.get("type") == "text":
            messages.append(str(item.get("text", "")))
    warnings = [line for message in messages for line in message.splitlines() if "WARNING:" in line]
    summary = " ".join(messages)
    if len(summary) > 4000:
        summary = summary[:3999] + "…"
    return summary, warnings


def append_usage_event(directory: Path, usage: dict[str, int], elapsed: float) -> None:
    """Append the final token usage of a finished run to its recording."""
    if not directory.is_dir():
        return
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "elapsed": max(0.0, elapsed),
        "tool": "session_usage",
        "args": {},
        "result": {"error": False, "summary": "Final model token usage."},
        "frame": None,
        "checklist": [],
        "verify_warnings": [],
        "token_usage": usage,
    }
    with (directory / "events.jsonl").open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


class SessionRecorder:
    """Write frames and append-only action events through one small interface."""

    def __init__(self, directory: Path, *, clock: Any = time.time) -> None:
        self.directory = directory
        self.frames = directory / "frames"
        self.events = directory / "events.jsonl"
        self.metadata = directory / "metadata.json"
        self.clock = clock
        self.started = float(clock())
        self.frame_number = 0
        self.last_frame: str | None = None
        self.closed = False
        self.frames.mkdir(parents=True, exist_ok=True)
        self._write_metadata(active=True)

    @classmethod
    def from_environment(cls) -> SessionRecorder | None:
        directory = default_recording_dir()
        return cls(directory) if directory is not None else None

    def _write_metadata(self, *, active: bool) -> None:
        payload = {
            "version": 1,
            "active": active,
            "started_at": datetime.fromtimestamp(self.started, UTC).isoformat(),
            "ended_at": None if active else datetime.now(UTC).isoformat(),
            "model_frame": {"width": MODEL_WIDTH, "height": MODEL_HEIGHT},
        }
        temporary = self.metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.metadata)

    def record_frame(self, data: bytes) -> str:
        self.frame_number += 1
        relative = f"frames/{self.frame_number:06d}.jpg"
        (self.directory / relative).write_bytes(data)
        self.last_frame = relative
        return relative

    def record_tool(
        self,
        tool: str,
        arguments: dict[str, object],
        result: dict[str, object],
        *,
        checklist: list[dict[str, object]],
        frame: str | None,
    ) -> None:
        summary, warnings = result_summary(result)
        now = float(self.clock())
        event = {
            "ts": datetime.fromtimestamp(now, UTC).isoformat(),
            "elapsed": max(0.0, now - self.started),
            "tool": tool,
            "args": event_arguments(arguments),
            "result": {"error": bool(result.get("isError")), "summary": summary},
            "frame": frame,
            "checklist": checklist,
            "verify_warnings": warnings,
        }
        with self.events.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self.closed:
            self._write_metadata(active=False)
            self.closed = True
