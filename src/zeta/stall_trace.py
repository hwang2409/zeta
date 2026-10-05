"""Opt-in event-loop stall tracing with content-free, bounded records."""

from __future__ import annotations

import asyncio
import gc
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .providers.stream_diagnostics import write_stream_diagnostic

DEFAULT_STALL_THRESHOLD_SECONDS = 0.1
DEFAULT_STALL_LOG_MAX_BYTES = 2 * 1024 * 1024
_MIN_GC_PAUSE_MS = 1.0


def _positive_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0 else default


class StallWatchdog:
    """Sample a running asyncio loop from a daemon thread and record stalls."""

    @classmethod
    def from_environment(
        cls,
        counts: Callable[[], Mapping[str, int]] | None = None,
    ) -> StallWatchdog | None:
        """Return a configured watchdog only when explicitly enabled."""

        if os.environ.get("ZETA_STALL_TRACE") != "1":
            return None
        home = Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))
        threshold = _positive_env(
            "ZETA_STALL_THRESHOLD_MS", DEFAULT_STALL_THRESHOLD_SECONDS * 1000
        ) / 1000
        max_bytes = int(
            _positive_env("ZETA_STALL_LOG_MAX_BYTES", DEFAULT_STALL_LOG_MAX_BYTES)
        )
        return cls(
            home / "logs" / "stalls.jsonl",
            threshold_seconds=threshold,
            max_bytes=max_bytes,
            counts=counts,
        )

    def __init__(
        self,
        path: Path,
        *,
        threshold_seconds: float,
        max_bytes: int,
        counts: Callable[[], Mapping[str, int]] | None = None,
    ) -> None:
        if threshold_seconds <= 0:
            raise ValueError("stall threshold must be positive")
        if max_bytes <= 0:
            raise ValueError("stall log size must be positive")
        self.path = path
        self.threshold_seconds = threshold_seconds
        self.max_bytes = max_bytes
        self._counts = counts or dict
        self._interval = min(0.025, max(0.005, threshold_seconds / 4))
        self._expected = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_stack: list[dict[str, Any]] | None = None
        self._gc_starts: dict[int, tuple[float, int]] = {}
        self._pending: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._gc_callback_ref = self._gc_callback

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start tracing ``loop``. Call this from the loop's thread."""

        if self._thread is not None:
            raise RuntimeError("stall watchdog is already started")
        self._loop = loop
        self._loop_thread_id = threading.get_ident()
        self._expected = time.monotonic() + self._interval
        gc.callbacks.append(self._gc_callback_ref)
        self._thread = threading.Thread(
            target=self._watch,
            name="zeta-stall-watchdog",
            daemon=True,
        )
        self._thread.start()
        loop.call_later(self._interval, self._heartbeat)

    def close(self) -> None:
        """Stop tracing and flush records already observed by the watcher."""

        try:
            gc.callbacks.remove(self._gc_callback_ref)
        except ValueError:
            pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self._interval * 4))
        self._drain_pending()

    def _heartbeat(self) -> None:
        if self._stop.is_set() or self._loop is None:
            return
        now = time.monotonic()
        if self._active_stack is not None:
            record: dict[str, Any] = {
                "kind": "loop_stall",
                "timestamp": time.time(),
                "duration_ms": round(max(0.0, now - self._expected) * 1000, 3),
                "threshold_ms": round(self.threshold_seconds * 1000, 3),
                "stack": self._active_stack,
            }
            try:
                record.update(
                    {
                        key: int(value)
                        for key, value in self._counts().items()
                        if type(key) is str and type(value) is int and value >= 0
                    }
                )
            except Exception:  # noqa: BLE001, S110 - diagnostics cannot affect TUI
                pass
            self._pending.put(record)
            self._active_stack = None
        self._expected = now + self._interval
        self._loop.call_later(self._interval, self._heartbeat)

    def _watch(self) -> None:
        while not self._stop.wait(self._interval):
            now = time.monotonic()
            if (
                self._active_stack is None
                and now - self._expected >= self.threshold_seconds
            ):
                frame = sys._current_frames().get(self._loop_thread_id)
                if frame is not None:
                    self._active_stack = [
                        {
                            "file": item.filename,
                            "line": item.lineno,
                            "function": item.name,
                        }
                        for item in traceback.extract_stack(frame)
                    ]
            self._drain_pending()
        self._drain_pending()

    def _gc_callback(self, phase: str, info: dict[str, Any]) -> None:
        thread_id = threading.get_ident()
        if phase == "start":
            self._gc_starts[thread_id] = (
                time.monotonic(),
                int(info.get("generation", -1)),
            )
            return
        started = self._gc_starts.pop(thread_id, None)
        if started is None:
            return
        duration_ms = (time.monotonic() - started[0]) * 1000
        if duration_ms < _MIN_GC_PAUSE_MS:
            return
        self._pending.put(
            {
                "kind": "gc_pause",
                "timestamp": time.time(),
                "duration_ms": round(duration_ms, 3),
                "generation": started[1],
                "collected": int(info.get("collected", 0)),
                "uncollectable": int(info.get("uncollectable", 0)),
            }
        )

    def _drain_pending(self) -> None:
        while True:
            try:
                record = self._pending.get_nowait()
            except queue.Empty:
                return
            self._write_record(record)

    def _write_record(self, record: Mapping[str, Any]) -> None:
        write_stream_diagnostic(self.path, record, max_bytes=self.max_bytes)


def _percentile(values: list[float], percentile: float) -> float | int:
    if not values:
        return 0
    value = sorted(values)[max(0, math.ceil(percentile * len(values)) - 1)]
    return int(value) if value.is_integer() else round(value, 3)


def _record_paths(path: Path) -> tuple[Path, ...]:
    rotated = path.with_name(f"{path.name}.1")
    return tuple(candidate for candidate in (rotated, path) if candidate.is_file())


def read_stall_summary(path: Path, *, top: int = 10) -> dict[str, Any]:
    """Aggregate bounded stall logs without failing on partial JSONL rows."""

    durations: list[float] = []
    gc_pauses = 0
    stacks: dict[str, list[float]] = defaultdict(list)
    for candidate in _record_paths(path):
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            if record.get("kind") == "gc_pause":
                gc_pauses += 1
                continue
            duration = record.get("duration_ms")
            stack = record.get("stack")
            if (
                record.get("kind") != "loop_stall"
                or not isinstance(duration, (int, float))
                or not isinstance(stack, list)
            ):
                continue
            durations.append(float(duration))
            frame = stack[-1] if stack else None
            if not isinstance(frame, dict):
                continue
            file = frame.get("file")
            line_number = frame.get("line")
            function = frame.get("function")
            if isinstance(file, str) and isinstance(line_number, int) and isinstance(function, str):
                stacks[f"{file}:{line_number} in {function}"].append(float(duration))
    ranked = sorted(
        (
            {
                "location": location,
                "count": len(values),
                "total_ms": round(sum(values), 3),
            }
            for location, values in stacks.items()
        ),
        key=lambda item: (-item["total_ms"], item["location"]),
    )[:top]
    return {
        "loop_stalls": len(durations),
        "duration_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": _percentile(durations, 1.0),
        },
        "gc_pauses": gc_pauses,
        "top_stacks": ranked,
    }


def render_stall_summary(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable stall report."""

    durations = report["duration_ms"]
    lines = [
        f"loop stalls: {report['loop_stalls']}",
        (
            f"duration: p50 {durations['p50']} ms · p95 {durations['p95']} ms "
            f"· max {durations['max']} ms"
        ),
        f"GC pauses: {report['gc_pauses']}",
        "top stacks by total blocked time:",
    ]
    stacks = report["top_stacks"]
    if not stacks:
        lines.append("  (none)")
    else:
        lines.extend(
            f"  {item['total_ms']} ms · {item['count']} stalls · {item['location']}"
            for item in stacks
        )
    return "\n".join(lines) + "\n"
