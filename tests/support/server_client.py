"""Buffered JSON-RPC test client helpers."""

from __future__ import annotations

import asyncio
import json
import weakref
from collections import deque
from collections.abc import Callable
from typing import Any

TIMEOUT = 30
type EventOwner = tuple[str, str | None, str | int]
_TOOL_EVENTS = {"tool_start", "tool_output", "tool_end"}
_APPROVAL_EVENTS = {"approval_request", "approval_end"}
_TURN_EVENTS = {"turn_start", "turn_end"}
_buffers: weakref.WeakKeyDictionary[asyncio.StreamReader, deque[dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def _buffer(reader: asyncio.StreamReader) -> deque[dict[str, Any]]:
    return _buffers.setdefault(reader, deque())


async def read(reader: asyncio.StreamReader) -> dict[str, Any]:
    buffered = _buffer(reader)
    if buffered:
        return buffered.popleft()
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line, "server closed the socket"
    return json.loads(line)


async def request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: str | int,
    method: str,
    params: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    writer.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            + "\n"
        ).encode()
    )
    await asyncio.wait_for(writer.drain(), TIMEOUT)
    buffered = _buffer(reader)
    deferred: list[dict[str, Any]] = []
    try:
        while True:
            if buffered:
                frame = buffered.popleft()
            else:
                frame = await read(reader)
            if frame.get("id") == request_id:
                return [frame]
            deferred.append(frame)
    finally:
        buffered.extendleft(reversed(deferred))


async def approve_and_collect_grant(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: str | int,
    approval_request_id: str,
    expected_scope: str,
    expected_changed: list[str],
) -> dict[str, Any]:
    """Collect the approve response and its grant event as one transaction."""
    writer.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "approve",
                    "params": {
                        "request_id": approval_request_id,
                        "scope": expected_scope,
                    },
                }
            )
            + "\n"
        ).encode()
    )
    await asyncio.wait_for(writer.drain(), TIMEOUT)
    frames: list[dict[str, Any]] = []
    response: dict[str, Any] | None = None
    grant_events: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    try:
        try:
            async with asyncio.timeout(10):
                while response is None or not grant_events:
                    frame = await read(reader)
                    frames.append(frame)
                    is_response = frame.get("id") == request_id
                    is_grant_event = (
                        frame.get("params", {}).get("event") == "session_grants_changed"
                    )
                    if not is_response and not is_grant_event:
                        deferred.append(frame)
                        continue
                    if is_response:
                        response = frame
                        assert response.get("result", {}).get("accepted") is True, (
                            f"approve response was not accepted: response={response!r}; "
                            f"all observed frames={frames!r}"
                        )
                        assert (
                            response["result"].get("request_id") == approval_request_id
                        ), (
                            f"approve response has wrong request id: response={response!r}; "
                            f"all observed frames={frames!r}"
                        )
                        assert response["result"].get("scope") == expected_scope, (
                            f"approve response has wrong scope: response={response!r}; "
                            f"all observed frames={frames!r}"
                        )
                    if is_grant_event:
                        grant_events.append(frame["params"])
                        data = frame["params"].get("data")
                        assert isinstance(data, dict), (
                            f"grant event has no data: frame={frame!r}; "
                            f"all observed frames={frames!r}"
                        )
                        assert data.get("change") == "grant", (
                            f"grant event has wrong change: frame={frame!r}; "
                            f"all observed frames={frames!r}"
                        )
                        assert data.get("changed") == expected_changed, (
                            f"grant event has wrong data: frame={frame!r}; "
                            f"all observed frames={frames!r}"
                        )
        except (AssertionError, TimeoutError) as exc:
            if (
                response is not None
                and response.get("result", {}).get("accepted") is True
                and not grant_events
            ):
                raise AssertionError(
                    "approve response without session_grants_changed; "
                    f"response={response!r}; all observed frames={frames!r}"
                ) from None
            raise AssertionError(
                f"approve collection failed: {exc}; all observed frames={frames!r}"
            ) from None
    finally:
        _buffer(reader).extendleft(reversed(deferred))
    assert response is not None, (
        f"missing approve response; all observed frames={frames!r}"
    )
    assert len(grant_events) == 1, (
        f"expected exactly one grant event; all observed frames={frames!r}"
    )
    return grant_events[0]


async def frames_until_event(
    reader: asyncio.StreamReader, name: str
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        frame = await read(reader)
        frames.append(frame)
        if frame.get("params", {}).get("event") == name:
            return frames


async def event(reader: asyncio.StreamReader, name: str) -> dict[str, Any]:
    frames = await frames_until_event(reader, name)
    return frames[-1]["params"]


def event_owner(frame: dict[str, Any]) -> EventOwner | None:
    """Return the protocol identity that owns a tool, approval, or turn event."""
    params = frame.get("params", {})
    name = params.get("event")
    identifier: str | int | None = None
    if name in _TOOL_EVENTS:
        identifier = params.get("tool_call", {}).get("id")
    elif name in _APPROVAL_EVENTS:
        identifier = params.get("request_id")
    elif name in _TURN_EVENTS:
        data = params.get("data", {})
        identifier = data.get("turn_id", data.get("turn"))
    if not isinstance(name, str) or not isinstance(identifier, (str, int)):
        return None
    return name, params.get("session_id"), identifier


def frames_for_call(frames: list[dict[str, Any]], call_id: str) -> list[dict[str, Any]]:
    """Select tool events by tool-call id and approval events by request id."""
    return [
        frame
        for frame in frames
        if (owner := event_owner(frame)) is not None
        and owner[0] in _TOOL_EVENTS | _APPROVAL_EVENTS
        and owner[2] == call_id
    ]


async def read_until(
    reader: asyncio.StreamReader,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout: float = 10,
    what: str,
) -> list[dict[str, Any]]:
    """Read through a matching frame and return every skipped frame with it."""
    frames: list[dict[str, Any]] = []
    try:
        async with asyncio.timeout(timeout):
            while True:
                frame = await read(reader)
                frames.append(frame)
                if predicate(frame):
                    return frames
    except TimeoutError:
        raise AssertionError(
            f"timed out waiting for {what}; skipped frames: {frames!r}"
        ) from None


def assert_event_order(frames: list[dict[str, Any]], *owners: EventOwner) -> None:
    """Assert exact, unique protocol-owned events occur in the given order."""
    positions: list[int] = []
    for owner in owners:
        matches = [
            index for index, frame in enumerate(frames) if event_owner(frame) == owner
        ]
        assert len(matches) == 1, (
            f"expected exactly one event owned by {owner!r}, found {len(matches)}; "
            f"frames: {frames!r}"
        )
        positions.append(matches[0])
    assert positions == sorted(positions), (
        f"events are out of order: {list(zip(owners, positions, strict=True))!r}"
    )
