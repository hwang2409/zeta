"""Strict decoding of observer serve frames into existing stream events."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

from ..protocol.types import StreamEvent, StreamEventType

MAX_FRAME_BYTES = 256 * 1024
MAX_LINE_BYTES = MAX_FRAME_BYTES + 1

class RemoteWireError(ValueError):
    pass


def decode_frame(raw: bytes | str) -> tuple[str, Mapping[str, object]]:
    data = raw.encode() if isinstance(raw, str) else raw
    if len(data) > MAX_FRAME_BYTES:
        raise RemoteWireError("remote frame is too large")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteWireError("invalid remote frame") from exc
    if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
        raise RemoteWireError("invalid remote frame")
    if value.get("method") != "event" or not isinstance(value.get("params"), dict):
        raise RemoteWireError("unsupported remote frame")
    params = value["params"]
    if not isinstance(params.get("event"), str):
        raise RemoteWireError("remote frame has no event")
    return params["event"], params


def decode_event(raw: bytes | str) -> StreamEvent:
    name, params = decode_frame(raw)
    try:
        kind = StreamEventType(name)
    except ValueError as exc:
        raise RemoteWireError("unknown remote event") from exc
    allowed = {"event", "session_id", "data", "delta", "kind", "message", "content", "tool_call", "tool_result", "error"}
    if set(params) - allowed:
        raise RemoteWireError("unknown remote event field")
    value = {key: item for key, item in params.items() if key != "session_id"}
    value["type"] = kind.value
    if "data" not in value:
        value["data"] = {}
    if not isinstance(value["data"], dict):
        raise RemoteWireError("invalid remote event data")
    try:
        return StreamEvent.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteWireError("invalid remote event payload") from exc


def decode_lines(chunks: Iterator[bytes]) -> Iterator[StreamEvent]:
    buffer = b""
    for chunk in chunks:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if len(line) > MAX_FRAME_BYTES:
                raise RemoteWireError("remote frame is too large")
            if line:
                yield decode_event(line)
        if len(buffer) > MAX_FRAME_BYTES:
            raise RemoteWireError("unterminated remote frame is too large")
    if buffer:
        raise RemoteWireError("unterminated remote frame")
