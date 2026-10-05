"""Pure observation and settle logic shared by every desktop backend."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping

SETTLE_INTERVAL_SECONDS = 0.1
SETTLE_TIMEOUT_SECONDS = 2.0
SETTLE_THRESHOLD = 0.002


def frame_difference(first: bytes, second: bytes) -> float:
    """Return the mean absolute difference of two grayscale samples, 0..1."""

    if len(first) != len(second) or not first:
        return 1.0
    difference = sum(abs(left - right) for left, right in zip(first, second, strict=True))
    return difference / (255 * len(first))


def wait_for_stable(
    capture: Callable[[], bytes],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    interval: float = SETTLE_INTERVAL_SECONDS,
    timeout: float = SETTLE_TIMEOUT_SECONDS,
    threshold: float = SETTLE_THRESHOLD,
) -> float:
    """Wait for two consecutive near-identical samples, with a hard time cap.

    Returns the elapsed seconds. The cap keeps animated screens (a blinking
    caret, a spinner) from stalling an action.
    """

    started = monotonic()
    previous = capture()
    while monotonic() - started < timeout:
        sleep(interval)
        current = capture()
        if frame_difference(previous, current) <= threshold:
            break
        previous = current
    return max(0.0, monotonic() - started)


def format_observation(
    raw: Mapping[str, object], screenshot: bytes, previous_hash: str | None
) -> tuple[str, str]:
    """Return stable observation JSON and the hash of the current frame."""

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
    text = json.dumps(observation, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return text, current_hash


__all__ = ["format_observation", "frame_difference", "wait_for_stable"]
