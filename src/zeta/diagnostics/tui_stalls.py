"""Bounded, always-on diagnostics for blocked TUI event loops."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import sys
import sysconfig
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

logger = logging.getLogger(__name__)

PROBE_INTERVAL_SECONDS = 0.05
STALL_THRESHOLD_SECONDS = 0.25
STALL_SAMPLE_INTERVAL_SECONDS = 0.1
STALL_LOG_MAX_BYTES = 5 * 1024 * 1024
_MAX_DISTINCT_STACKS = 8
_MAX_FRAMES = 20
_MAX_PATH_CHARS = 240
_MAX_FUNCTION_CHARS = 120
_IDLE_FUNCTIONS = frozenset({"get", "join", "poll", "select", "sleep", "wait"})
_PACKAGE_ROOT = Path(__file__).parents[1]
_STDLIB_ROOT = Path(sysconfig.get_paths()["stdlib"])


class TUIStallWatchdog:
    """Probe one asyncio loop from a daemon thread and log completed stalls."""

    def __init__(
        self,
        path: Path,
        *,
        session_id: str,
        probe_interval: float = PROBE_INTERVAL_SECONDS,
        stall_threshold: float = STALL_THRESHOLD_SECONDS,
        sample_interval: float = STALL_SAMPLE_INTERVAL_SECONDS,
        max_bytes: int = STALL_LOG_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if min(probe_interval, stall_threshold, sample_interval, max_bytes) <= 0:
            raise ValueError("watchdog timing and size limits must be positive")
        self.path = path
        self.session_id = session_id
        self.probe_interval = probe_interval
        self.stall_threshold = stall_threshold
        self.sample_interval = sample_interval
        self.max_bytes = max_bytes
        self._clock = clock
        self._utc_now = utc_now
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._logging_failure_reported = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start probing. Call this method from the loop thread."""

        if self._thread is not None:
            raise RuntimeError("TUI stall watchdog is already started")
        self._loop = loop
        self._loop_thread_id = threading.get_ident()
        self._thread = threading.Thread(
            target=self._watch_safely,
            name="zeta-tui-stall-watchdog",
            daemon=True,
        )
        self._thread.start()

    async def close(self) -> None:
        """Stop the daemon thread without blocking the event loop."""

        self._stop.set()
        thread = self._thread
        if thread is not None:
            try:
                await asyncio.to_thread(
                    thread.join, max(0.2, self.probe_interval * 4)
                )
            except Exception:  # noqa: BLE001, S110 - diagnostics cannot affect shutdown
                pass

    def _watch_safely(self) -> None:
        try:
            self._watch()
        except Exception:  # noqa: BLE001 - diagnostics must never affect the TUI
            self._stop.set()

    def _watch(self) -> None:
        loop = self._loop
        if loop is None:
            return
        while not self._stop.is_set():
            answered = threading.Event()
            started_monotonic = self._clock()
            started_utc = self._utc_now()
            loop.call_soon_threadsafe(answered.set)
            stacks: dict[tuple[tuple[str, int, str], ...], list[int]] = {}
            other_threads: dict[
                tuple[int, str],
                dict[tuple[tuple[str, int, str], ...], list[int]],
            ] = {}
            next_sample = started_monotonic + self.stall_threshold
            while not answered.wait(self.probe_interval):
                if self._stop.is_set():
                    return
                now = self._clock()
                if now < next_sample:
                    continue
                self._sample_stacks(stacks, other_threads)
                next_sample = now + self.sample_interval
            ended = self._clock()
            duration = ended - started_monotonic
            if duration >= self.stall_threshold:
                try:
                    self._append_record(
                        self._record(started_utc, duration, stacks, other_threads)
                    )
                except Exception:
                    if not self._logging_failure_reported:
                        logger.exception("failed to write TUI stall diagnostic")
                        self._logging_failure_reported = True
            if self._stop.wait(self.probe_interval):
                return

    def _sample_stacks(
        self,
        stacks: dict[tuple[tuple[str, int, str], ...], list[int]],
        other_threads: dict[
            tuple[int, str],
            dict[tuple[tuple[str, int, str], ...], list[int]],
        ],
    ) -> None:
        frames = sys._current_frames()
        loop_frame = frames.get(self._loop_thread_id)
        if loop_frame is not None:
            _count_stack(stacks, tuple(_trim_stack(loop_frame)))

        names = {
            thread.ident: thread.name
            for thread in threading.enumerate()
            if thread.ident is not None
        }
        watchdog_thread_id = threading.get_ident()
        for thread_id, frame in frames.items():
            if thread_id in {self._loop_thread_id, watchdog_thread_id}:
                continue
            stack = tuple(_trim_stack(frame))
            if not stack:
                continue
            samples = other_threads.setdefault(
                (thread_id, names.get(thread_id, "unknown")), {}
            )
            _count_stack(samples, stack)

    def _record(
        self,
        started: datetime,
        duration: float,
        stacks: dict[tuple[tuple[str, int, str], ...], list[int]],
        other_threads: dict[
            tuple[int, str],
            dict[tuple[tuple[str, int, str], ...], list[int]],
        ],
    ) -> dict[str, Any]:
        ordered = sorted(stacks.items(), key=lambda item: (-item[1][0], item[1][1]))
        threads = []
        for (thread_id, name), samples in other_threads.items():
            ordered_samples = sorted(
                samples.items(),
                key=lambda item: (_is_idle_stack(item[0]), -item[1][0], item[1][1]),
            )
            busy_samples = sum(
                count
                for stack, (count, _order) in ordered_samples
                if not _is_idle_stack(stack)
            )
            threads.append(
                {
                    "thread_id": thread_id,
                    "name": name,
                    "busy_samples": busy_samples,
                    "stacks": [
                        {
                            "count": count,
                            "idle": _is_idle_stack(stack),
                            "frames": _frame_records(stack),
                        }
                        for stack, (count, _order) in ordered_samples
                    ],
                }
            )
        threads.sort(
            key=lambda thread: (
                -thread["busy_samples"],
                -sum(stack["count"] for stack in thread["stacks"]),
                thread["name"],
                thread["thread_id"],
            )
        )
        return {
            "start_time": started.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "duration_ms": round(max(0.0, duration) * 1000, 3),
            "session_id": self.session_id,
            "stacks": [
                {
                    "count": count,
                    "frames": _frame_records(stack),
                }
                for stack, (count, _order) in ordered
            ],
            "other_threads": threads,
        }

    def _append_record(self, record: dict[str, Any]) -> None:
        encoded = (
            json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > self.max_bytes:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a+b") as lock:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if (
                    self.path.exists()
                    and self.path.stat().st_size + len(encoded) > self.max_bytes
                ):
                    rotated = self.path.with_name(f"{self.path.name}.1")
                    rotated.unlink(missing_ok=True)
                    self.path.replace(rotated)
                fd = os.open(
                    self.path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o600,
                )
                try:
                    if os.write(fd, encoded) != len(encoded):
                        raise OSError("incomplete TUI stall diagnostic write")
                finally:
                    os.close(fd)
                os.chmod(self.path, 0o600)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _count_stack(
    samples: dict[tuple[tuple[str, int, str], ...], list[int]],
    stack: tuple[tuple[str, int, str], ...],
) -> None:
    if not stack:
        return
    current = samples.get(stack)
    if current is not None:
        current[0] += 1
    elif len(samples) < _MAX_DISTINCT_STACKS:
        samples[stack] = [1, len(samples)]


def _frame_records(
    stack: tuple[tuple[str, int, str], ...],
) -> list[dict[str, str | int]]:
    return [
        {"file": file, "line": line, "function": function}
        for file, line, function in stack
    ]


def _is_idle_stack(stack: tuple[tuple[str, int, str], ...]) -> bool:
    return bool(stack and stack[0][2] in _IDLE_FUNCTIONS)


def _trim_stack(frame: FrameType) -> list[tuple[str, int, str]]:
    frames: list[tuple[str, int, str]] = []
    innermost = True
    while frame is not None and len(frames) < _MAX_FRAMES:
        path = Path(frame.f_code.co_filename)
        if innermost or _is_relevant_path(path):
            frames.append(
                (
                    str(path)[-_MAX_PATH_CHARS:],
                    frame.f_lineno,
                    frame.f_code.co_name[-_MAX_FUNCTION_CHARS:],
                )
            )
        innermost = False
        frame = frame.f_back
    return frames


def _is_relevant_path(path: Path) -> bool:
    try:
        path.relative_to(_PACKAGE_ROOT)
        return True
    except ValueError:
        pass
    try:
        path.relative_to(_STDLIB_ROOT)
    except ValueError:
        return False
    return "site-packages" not in path.parts


def start_tui_stall_watchdog(
    loop: asyncio.AbstractEventLoop,
    *,
    zeta_home: Path,
    session_id: str,
) -> TUIStallWatchdog | None:
    """Start the TUI watchdog unless the process explicitly opts out."""

    if os.environ.get("ZETA_STALL_WATCHDOG") == "0":
        return None
    try:
        watchdog = TUIStallWatchdog(
            zeta_home / "logs" / "tui-stalls.jsonl", session_id=session_id
        )
        watchdog.start(loop)
        return watchdog
    except Exception:  # noqa: BLE001 - diagnostics must never prevent TUI startup
        return None


__all__ = ["TUIStallWatchdog", "start_tui_stall_watchdog"]
