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
    if not isinstance(value, dict) or value.get("method") != "event":
        raise RemoteWireError("invalid remote frame")
    if "jsonrpc" in value and value.get("jsonrpc") != "2.0":
        raise RemoteWireError("invalid remote frame")
    if not isinstance(value.get("params"), dict):
        raise RemoteWireError("unsupported remote frame")
    params = value["params"]
    event = params.get("event", params.get("type"))
    if not isinstance(event, str):
        raise RemoteWireError("remote frame has no event")
    if "event" not in params:
        params = dict(params)
        params["event"] = event
    return event, params


def history_events(result: Mapping[str, object]) -> tuple[StreamEvent, ...]:
    messages = result.get("messages")
    if not isinstance(messages, list) or len(messages) > 4096:
        raise RemoteWireError("invalid remote history")
    events: list[StreamEvent] = []
    for row in messages:
        if not isinstance(row, dict) or set(row) - {"id", "role", "content", "tool_result", "metadata", "event", "data"}:
            raise RemoteWireError("invalid remote history row")
        if not isinstance(row.get("role"), str) or not isinstance(row.get("content"), list):
            raise RemoteWireError("invalid remote history row")
        try:
            from ..protocol.types import Message
            message = Message.from_dict({key: row[key] for key in ("role", "content", "tool_result", "metadata") if key in row})
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteWireError("invalid remote history row") from exc
        events.append(StreamEvent(StreamEventType.MESSAGE_END, message=message, data={"remote_history": True}))
    return tuple(events)


def decode_event(raw: bytes | str) -> StreamEvent:
    name, params = decode_frame(raw)
    aliases = {
        "assistant_delta": StreamEventType.MESSAGE_UPDATE,
        "assistant_message": StreamEventType.MESSAGE_END,
        "tool_start": StreamEventType.TOOL_EXECUTION_START,
        "tool_output": StreamEventType.TOOL_EXECUTION_UPDATE,
        "tool_end": StreamEventType.TOOL_EXECUTION_END,
        "approval_request": StreamEventType.TOOL_APPROVAL_START,
        "approval_end": StreamEventType.TOOL_APPROVAL_END,
        "sub_agent_receipt": StreamEventType.AGENT_NOTIFICATION,
        "usage": StreamEventType.AGENT_NOTIFICATION,
        "served_turn_end": StreamEventType.TURN_END,
    }
    try:
        kind = aliases[name] if name in aliases else StreamEventType(name)
    except ValueError as exc:
        raise RemoteWireError("unknown remote event") from exc
    allowed = {"event", "type", "session_id", "data", "delta", "kind", "message", "content", "tool_call", "tool_result", "error", "row_seq", "usage", "stream_id", "seq"}
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
