"""Tolerant decoding of observer serve frames into existing stream events.

Only size and frame-shape errors are fatal. Unknown event names, unknown keys,
and payloads this client cannot read are skipped, so a newer remote zeta never
breaks an older viewer.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache

from ..protocol.types import Message, StreamEvent, StreamEventType

MAX_FRAME_BYTES = 256 * 1024
MAX_HISTORY_ROWS = 4096
HISTORY_REQUEST_ID = 2
_EVENT_KEYS = ("message", "delta", "tool_call", "tool_result", "error", "data")


class RemoteWireError(ValueError):
    pass


class BoundedLineDecoder:
    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        lines: list[bytes] = []
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer[:] = rest
            if len(line) > MAX_FRAME_BYTES:
                raise RemoteWireError("remote frame is too large")
            if line:
                lines.append(bytes(line))
        if len(self._buffer) > MAX_FRAME_BYTES:
            raise RemoteWireError("remote frame is too large")
        return lines


@cache
def _renderable_events() -> dict[str, StreamEventType]:
    # lazy: importing zeta.server adds ~100 ms to every TUI start.
    from ..server.event_contract import STREAM_EVENT_WIRE_EVENTS

    return {wire.value: kind for kind, wire in STREAM_EVENT_WIRE_EVENTS.items()}


def decode_line(raw: bytes) -> tuple[StreamEvent, ...]:
    """Return the renderable events in one serve frame; raise only on a bad frame."""
    if len(raw) > MAX_FRAME_BYTES:
        raise RemoteWireError("remote frame is too large")
    try:
        frame = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RemoteWireError("invalid remote frame") from exc
    if not isinstance(frame, dict):
        raise RemoteWireError("invalid remote frame")
    if frame.get("id") == HISTORY_REQUEST_ID and isinstance(frame.get("result"), dict):
        return history_events(frame["result"])
    params = frame.get("params")
    if frame.get("method") != "event" or not isinstance(params, dict):
        return ()
    event = _event(params)
    return () if event is None else (event,)


def _event(params: Mapping[str, object]) -> StreamEvent | None:
    kind = _renderable_events().get(str(params.get("event")))
    if kind is None:
        return None
    if kind is StreamEventType.MESSAGE_UPDATE and params.get("kind", "assistant") != "assistant":
        return None
    value: dict[str, object] = {key: params[key] for key in _EVENT_KEYS if key in params}
    if kind is StreamEventType.TOOL_EXECUTION_UPDATE and "output" in params:
        value["delta"] = params["output"]
    if not isinstance(value.get("data", {}), dict):
        return None
    try:
        return StreamEvent.from_dict({**value, "type": kind.value})
    except (KeyError, TypeError, ValueError):
        return None


def history_events(result: Mapping[str, object]) -> tuple[StreamEvent, ...]:
    messages = result.get("messages")
    if not isinstance(messages, list) or len(messages) > MAX_HISTORY_ROWS:
        raise RemoteWireError("invalid remote history")
    events: list[StreamEvent] = []
    for row in messages:
        try:
            message = Message.from_dict(
                {key: row[key] for key in ("role", "content", "tool_result", "metadata") if key in row}
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        events.append(StreamEvent(StreamEventType.MESSAGE_END, message=message, data={"remote_history": True}))
    return tuple(events)
