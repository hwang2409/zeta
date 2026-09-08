"""Wire-level helpers for the zeta frontend protocol."""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = "1.0"
MAX_FRAME_BYTES = 1_048_576
TOOL_OUTPUT_MAX_BYTES = 8_000


class ProtocolError(Exception):
    """A JSON-RPC error that can be returned to the client."""

    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def parse_request(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_FRAME_BYTES:
        raise ProtocolError(-32600, "request frame exceeds 1048576 bytes")
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(-32700, "invalid JSON frame") from exc
    if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
        raise ProtocolError(-32600, "request must be a JSON-RPC 2.0 object")
    method = value.get("method")
    request_id = value.get("id")
    if not isinstance(method, str) or not method:
        raise ProtocolError(-32600, "request method must be a nonempty string")
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise ProtocolError(-32600, "request id must be a string or integer")
    params = value.get("params", {})
    if not isinstance(params, dict):
        raise ProtocolError(-32602, "request params must be an object")
    return {"id": request_id, "method": method, "params": params}


def encode(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def response(request_id: str | int, result: object) -> bytes:
    return encode({"jsonrpc": "2.0", "id": request_id, "result": result})


def error_response(
    request_id: str | int | None, code: int, message: str, data: object | None = None
) -> bytes:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return encode({"jsonrpc": "2.0", "id": request_id, "error": error})


def notification(event: str, session_id: str | None, **fields: object) -> bytes:
    params: dict[str, object] = {"event": event}
    if session_id is not None:
        params["session_id"] = session_id
    params.update(fields)
    return encode({"jsonrpc": "2.0", "method": "event", "params": params})


def bounded(value: str, limit: int = TOOL_OUTPUT_MAX_BYTES) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    retained = raw[:limit].decode("utf-8", errors="ignore")
    return f"{retained}... [truncated: {len(raw) - limit} bytes]"


__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "TOOL_OUTPUT_MAX_BYTES",
    "ProtocolError",
    "bounded",
    "encode",
    "error_response",
    "notification",
    "parse_request",
    "response",
]
