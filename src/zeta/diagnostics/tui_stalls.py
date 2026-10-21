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
_MAX_SAMPLED_THREADS = 32
_MAX_RETAINED_THREADS = 16
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
        frames: Callable[[], dict[int, FrameType]] = sys._current_frames,
        thread_names: Callable[[], dict[int, str]] | None = None,
        before_probe: Callable[[], None] | None = None,
        after_sample: Callable[[], None] | None = None,
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
        self._frames = frames
        self._thread_names = thread_names or _current_thread_names
        self._before_probe = before_probe
        self._after_sample = after_sample
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
            deadline = asyncio.get_running_loop().time() + 0.2
            try:
                while thread.is_alive() and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.01)
                thread.join(0)
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
            if self._before_probe is not None:
                self._before_probe()
            loop.call_soon_threadsafe(answered.set)
            stacks: dict[tuple[tuple[str, int, str], ...], list[int]] = {}
            other_threads: dict[
                tuple[int, str],
                dict[tuple[tuple[str, int, str], ...], list[int]],
            ] = {}
            observed_thread_count = 0
            next_sample = started_monotonic + self.stall_threshold
            while not answered.wait(self.probe_interval):
                if self._stop.is_set():
                    return
                now = self._clock()
                if now < next_sample:
                    continue
                observed_thread_count = max(
                    observed_thread_count,
                    self._sample_stacks(stacks, other_threads),
                )
                next_sample = now + self.sample_interval
            self._finish_probe(
                started_monotonic,
                started_utc,
                stacks,
                other_threads,
                observed_thread_count,
            )
            if self._stop.wait(self.probe_interval):
                return

    def _finish_probe(
        self,
        started_monotonic: float,
        started_utc: datetime,
        stacks: dict[tuple[tuple[str, int, str], ...], list[int]],
        other_threads: dict[
            tuple[int, str],
            dict[tuple[tuple[str, int, str], ...], list[int]],
        ],
        observed_thread_count: int,
    ) -> None:
        duration = self._clock() - started_monotonic
        if duration < self.stall_threshold:
            return
        try:
            self._append_record(
                self._record(
                    started_utc,
                    duration,
                    stacks,
                    other_threads,
                    observed_thread_count,
                )
            )
        except Exception:
            if not self._logging_failure_reported:
                logger.exception("failed to write TUI stall diagnostic")
                self._logging_failure_reported = True

    def _sample_stacks(
        self,
        stacks: dict[tuple[tuple[str, int, str], ...], list[int]],
        other_threads: dict[
            tuple[int, str],
            dict[tuple[tuple[str, int, str], ...], list[int]],
        ],
    ) -> int:
        frames = self._frames()
        loop_frame = frames.get(self._loop_thread_id)
        if loop_frame is not None:
            _count_stack(stacks, tuple(_trim_stack(loop_frame)))

        names = self._thread_names()
        watchdog_thread_id = threading.get_ident()
        candidates = [
            (thread_id, names.get(thread_id, "unknown"), frame)
            for thread_id, frame in frames.items()
            if thread_id not in {self._loop_thread_id, watchdog_thread_id}
        ]
        candidates.sort(
            key=lambda candidate: (
                _is_idle_frame(candidate[2]),
                candidate[1],
                candidate[0],
            )
        )
        for thread_id, name, frame in candidates[:_MAX_SAMPLED_THREADS]:
            stack = tuple(_trim_stack(frame))
            if not stack:
                continue
            samples = other_threads.setdefault((thread_id, name), {})
            _count_stack(samples, stack)
        if self._after_sample is not None:
            self._after_sample()
        return len(candidates)

    def _record(
        self,
        started: datetime,
        duration: float,
        stacks: dict[tuple[tuple[str, int, str], ...], list[int]],
        other_threads: dict[
            tuple[int, str],
            dict[tuple[tuple[str, int, str], ...], list[int]],
        ],
        observed_thread_count: int,
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
        retained_threads = threads[:_MAX_RETAINED_THREADS]
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
            "other_threads": retained_threads,
            "threads_truncated": max(
                0,
                observed_thread_count - len(retained_threads),
            ),
        }

    def _append_record(self, record: dict[str, Any]) -> None:
        _append_bounded_record(self.path, _encode_bounded_record(record, self.max_bytes), self.max_bytes)


def write_tui_diagnostic(
    path: Path, record: dict[str, Any], *, max_bytes: int = STALL_LOG_MAX_BYTES
) -> None:
    """Append one bounded TUI diagnostic without affecting normal operation."""

    try:
        _append_bounded_record(path, _encode_bounded_record(record, max_bytes), max_bytes)
    except Exception:  # noqa: BLE001 - diagnostics must never affect the TUI
        return


def _append_bounded_record(path: Path, encoded: bytes, max_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists() and path.stat().st_size + len(encoded) > max_bytes:
                rotated = path.with_name(f"{path.name}.1")
                rotated.unlink(missing_ok=True)
                path.replace(rotated)
            if path.exists():
                fd = os.open(path, os.O_APPEND | os.O_WRONLY, 0o600)
                try:
                    if os.write(fd, encoded) != len(encoded):
                        raise OSError("incomplete TUI diagnostic write")
                finally:
                    os.close(fd)
            else:
                temporary = path.with_name(
                    f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                try:
                    if os.write(fd, encoded) != len(encoded):
                        raise OSError("incomplete TUI diagnostic write")
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, path)
                finally:
                    os.close(fd)
                    temporary.unlink(missing_ok=True)
            os.chmod(path, 0o600)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _encode_bounded_record(record: dict[str, Any], max_bytes: int) -> bytes:
    limits = (
        (_MAX_DISTINCT_STACKS, _MAX_FRAMES, _MAX_RETAINED_THREADS),
        (4, 12, _MAX_RETAINED_THREADS),
        (2, 8, 12),
        (1, 4, 8),
        (1, 1, 4),
        (0, 0, 0),
    )
    for stack_limit, frame_limit, thread_limit in limits:
        reduced = _reduce_record(
            record,
            stack_limit=stack_limit,
            frame_limit=frame_limit,
            thread_limit=thread_limit,
        )
        encoded = (
            json.dumps(reduced, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        if len(encoded) <= max_bytes:
            return encoded
    raise ValueError("TUI stall diagnostic metadata exceeds the log size limit")


def _reduce_record(
    record: dict[str, Any],
    *,
    stack_limit: int,
    frame_limit: int,
    thread_limit: int,
) -> dict[str, Any]:
    reduced = dict(record)
    reduced["stacks"] = [
        {**stack, "frames": stack.get("frames", [])[:frame_limit]}
        for stack in record.get("stacks", [])[:stack_limit]
    ]
    threads = record.get("other_threads", [])
    retained_threads = []
    for thread in threads[:thread_limit]:
        retained_thread = dict(thread)
        retained_thread["stacks"] = [
            {**stack, "frames": stack.get("frames", [])[:frame_limit]}
            for stack in thread.get("stacks", [])[:stack_limit]
        ]
        retained_threads.append(retained_thread)
    reduced["other_threads"] = retained_threads
    reduced["threads_truncated"] = (
        record.get("threads_truncated", 0) + len(threads) - len(retained_threads)
    )
    return reduced


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


def _current_thread_names() -> dict[int, str]:
    return {
        thread.ident: thread.name
        for thread in threading.enumerate()
        if thread.ident is not None
    }


def _is_idle_stack(stack: tuple[tuple[str, int, str], ...]) -> bool:
    if not stack:
        return False
    file, _line, function = stack[0]
    return function in _IDLE_FUNCTIONS or (
        function == "_worker"
        and file.replace("\\", "/").endswith("/concurrent/futures/thread.py")
    )


def _is_idle_frame(frame: FrameType) -> bool:
    return _is_idle_stack(
        ((frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name),)
    )


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


__all__ = ["TUIStallWatchdog", "start_tui_stall_watchdog", "write_tui_diagnostic"]
