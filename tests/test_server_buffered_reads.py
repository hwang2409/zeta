from __future__ import annotations

import ast
import asyncio
import json
from collections import deque
from pathlib import Path

import pytest

from tests.support.server_client import (
    assert_event_order,
    event_owner,
    frames_for_call,
    frames_until_event,
    read_until,
    request,
    send,
)
from tests.support.server_client import read as buffered_read

_READ_METHODS = {"readline", "readuntil"}


def _unbuffered_reader_reads(source: str, path: str) -> list[str]:
    tree = ast.parse(source, filename=path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        request_readers = {
            argument.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"_request", "_buffered_request"}
            for argument in call.args[:1]
            if isinstance(argument, ast.Name)
        }
        direct_readers = {
            expression.func.value.id
            for expression in ast.walk(node)
            if isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)
            and expression.func.attr in _READ_METHODS
            and isinstance(expression.func.value, ast.Name)
        }
        if request_readers & direct_readers:
            violations.append(f"{path}:{node.lineno}:{node.name}")
    return violations


def test_server_tests_do_not_mix_raw_reader_reads_with_buffered_requests() -> None:
    root = Path(__file__).parent
    violations = [
        violation
        for path in sorted(root.glob("test_server*.py"))
        for violation in _unbuffered_reader_reads(path.read_text(), str(path))
    ]
    assert not violations, (
        "use _read(reader); raw reader reads mixed with _request: "
        + ", ".join(violations)
    )


class _Reader:
    def __init__(self, *frames: dict[str, object]) -> None:
        self.lines = deque(json.dumps(frame).encode() + b"\n" for frame in frames)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)
        return self.lines.popleft() if self.lines else b""


def _event_frame(
    event: str,
    identifier: str | int,
    *,
    session_id: str = "parent",
) -> dict[str, object]:
    params: dict[str, object] = {"event": event, "session_id": session_id}
    if event.startswith("tool_"):
        params["tool_call"] = {"id": identifier}
    elif event.startswith("approval_"):
        params["request_id"] = identifier
    else:
        params["data"] = {"turn": identifier}
    return {"params": params}


class _Writer:
    def write(self, _: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass


@pytest.mark.parametrize(
    ("frames", "call_id", "expected_events"),
    [
        (
            [
                _event_frame("tool_start", "parent-call"),
                _event_frame("approval_request", "child-approval", session_id="child"),
                _event_frame("tool_end", "parent-call"),
                _event_frame("approval_end", "child-approval", session_id="child"),
                _event_frame("tool_start", "child-call", session_id="child"),
                _event_frame("tool_end", "child-call", session_id="child"),
            ],
            "child-call",
            ["tool_start", "tool_end"],
        ),
        (
            [
                _event_frame("tool_start", "duplicate", session_id="parent"),
                _event_frame("tool_start", "duplicate", session_id="child"),
            ],
            "duplicate",
            ["tool_start", "tool_start"],
        ),
    ],
)
def test_event_owner_associates_interleaved_and_duplicate_call_ids(
    frames: list[dict[str, object]],
    call_id: str,
    expected_events: list[str],
) -> None:
    selected = frames_for_call(frames, call_id)
    assert [frame["params"]["event"] for frame in selected] == expected_events
    assert len({event_owner(frame) for frame in frames}) == len(frames)


def test_event_order_uses_exact_owner_and_rejects_duplicates() -> None:
    approval_end = _event_frame("approval_end", "approval-1", session_id="child")
    child_tool_end = _event_frame("tool_end", "child-call", session_id="child")
    frames = [
        _event_frame("tool_end", "parent-call"),
        approval_end,
        child_tool_end,
    ]
    approval_owner = event_owner(approval_end)
    child_owner = event_owner(child_tool_end)
    assert approval_owner is not None
    assert child_owner is not None
    assert_event_order(frames, approval_owner, child_owner)

    with pytest.raises(AssertionError, match="events are out of order"):
        assert_event_order(frames, child_owner, approval_owner)
    with pytest.raises(AssertionError, match="found 2"):
        assert_event_order([approval_end, approval_end], approval_owner)


@pytest.mark.asyncio
async def test_read_until_timeout_lists_skipped_frames() -> None:
    skipped = _event_frame("tool_end", "parent-call")

    class TimeoutReader(_Reader):
        async def readline(self) -> bytes:
            if self.lines:
                return await super().readline()
            raise TimeoutError

    reader = TimeoutReader(skipped)
    with pytest.raises(
        AssertionError,
        match=r"timed out waiting for child tool_end; skipped frames: .*parent-call",
    ):
        await read_until(
            reader,
            lambda frame: event_owner(frame) == ("tool_end", "child", "child-call"),
            what="child tool_end",
        )


def _is_turn_end(frame: dict[str, object]) -> bool:
    return event_owner(frame) == ("turn_end", "parent", 1)


@pytest.mark.asyncio
async def test_read_until_keeps_reading_for_a_response_after_the_event() -> None:
    reader = _Reader(_event_frame("turn_end", 1), {"id": 4, "result": {}})
    await send(reader, _Writer(), 4, "deny")

    frames = await read_until(reader, _is_turn_end, what="turn_end")

    assert [frame.get("id") for frame in frames] == [None, 4]
    assert frames[1].raw == b'{"id": 4, "result": {}}\n'


@pytest.mark.parametrize("helper", ["read_until", "frames_until_event"])
@pytest.mark.asyncio
async def test_event_reads_fail_when_a_sent_request_gets_no_response(
    helper: str,
) -> None:
    reader = _Reader(_event_frame("turn_end", 1))
    await send(reader, _Writer(), 4, "deny")

    with pytest.raises(
        AssertionError, match=r"request id 4 has no response; frames read: .*turn_end"
    ):
        if helper == "read_until":
            await read_until(reader, _is_turn_end, what="turn_end")
        else:
            await frames_until_event(reader, "turn_end")


@pytest.mark.asyncio
async def test_buffered_read_regression_rejects_old_direct_read() -> None:
    reader = _Reader(
        {"params": {"event": "tool_end"}},
        {"params": {"event": "approval_end"}},
        {"id": 1, "result": {}},
    )
    await request(reader, _Writer(), 1, "deny")

    with pytest.raises(AssertionError, match="server closed"):
        while True:
            line = await reader.readline()
            assert line, "server closed the socket"
            if json.loads(line).get("params", {}).get("event") == "tool_end":
                break

    assert (await buffered_read(reader))["params"]["event"] == "tool_end"


# The shape of the pre-#647 mistake in test_server.py: a raw readline on the
# same reader that _request had already buffered frames from.
_PRE_647_SOURCE = """
async def test_parent_mode_resolves_live_delegated_approvals_independently(reader):
    await _request(reader, writer, 4, "status")
    frame = await asyncio.wait_for(reader.readline(), TIMEOUT)
"""


def test_buffered_read_guard_catches_pre_647_mistake() -> None:
    violations = _unbuffered_reader_reads(_PRE_647_SOURCE, "pre-647")
    assert violations == [
        "pre-647:2:test_parent_mode_resolves_live_delegated_approvals_independently"
    ]


def test_buffered_read_guard_allows_buffered_helpers() -> None:
    source = """
async def test_ok(reader):
    await _request(reader, writer, 1, "status")
    frames = await _frames_until_event(reader, "tool_end")
    frame = await _read(reader)
"""
    assert _unbuffered_reader_reads(source, "ok") == []
