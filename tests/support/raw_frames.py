"""Assertions for raw JSON-RPC frame streams."""

from __future__ import annotations

import json
from collections import Counter


def assert_raw_frames_preserved(actual: bytes, expected: bytes) -> None:
    """Compare response bytes without asserting their undefined event interleaving."""
    actual_frames = actual.splitlines(keepends=True)
    expected_frames = expected.splitlines(keepends=True)
    actual_responses = Counter(
        (parsed["id"], frame)
        for frame in actual_frames
        if (parsed := json.loads(frame)).get("id") is not None
    )
    expected_responses = Counter(
        (parsed["id"], frame)
        for frame in expected_frames
        if (parsed := json.loads(frame)).get("id") is not None
    )
    actual_events = [
        frame for frame in actual_frames if json.loads(frame).get("method") == "event"
    ]
    expected_events = [
        frame for frame in expected_frames if json.loads(frame).get("method") == "event"
    ]
    assert actual_responses == expected_responses, (
        f"response frames differ:\nactual: {actual_responses}\nexpected: {expected_responses}"
    )
    assert actual_events == expected_events, (
        f"event frames differ:\nactual: {actual_events}\nexpected: {expected_events}"
    )
