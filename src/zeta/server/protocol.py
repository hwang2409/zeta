"""Bounded JSON-RPC framing for the zeta frontend protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = "1.0"
MAX_FRAME_BYTES = 1_048_576
MAX_REQUEST_ID_BYTES = 128
MAX_NUMERIC_ID_DIGITS = 128
TOOL_OUTPUT_MAX_BYTES = 8_000


class _OversizedInteger:
    def __init__(self, digits: int) -> None:
        self.digits = digits


class ProtocolError(Exception):
    """A JSON-RPC error that can be returned to the client."""

    def __init__(
        self,
        code: int,
        message: str,
        data: object | None = None,
        *,
        request_id: str | int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data
        self.request_id = _safe_request_id(request_id)


class FrameCodec:
    """Parse requests and serialize every bounded wire frame."""

    def parse_request(self, line: bytes) -> dict[str, Any]:
        request_id = _request_id_hint(line[:MAX_FRAME_BYTES])
        if len(line) > MAX_FRAME_BYTES:
            raise ProtocolError(
                -32600,
                f"request frame exceeds {MAX_FRAME_BYTES} bytes",
                request_id=request_id,
            )
        try:
            value = json.loads(
                line,
                parse_int=_parse_integer,
                parse_constant=_reject_constant,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as exc:
            raise ProtocolError(
                -32700,
                "invalid JSON frame",
                request_id=request_id,
            ) from exc
        if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
            raise ProtocolError(
                -32600,
                "request must be a JSON-RPC 2.0 object",
                request_id=request_id,
            )

        raw_request_id = value.get("id")
        request_id = _safe_request_id(raw_request_id)
        if isinstance(raw_request_id, _OversizedInteger):
            raise ProtocolError(
                -32600,
                f"numeric request id exceeds {MAX_NUMERIC_ID_DIGITS} digits",
                request_id=request_id,
            )
        if type(raw_request_id) is not str and type(raw_request_id) is not int:
            raise ProtocolError(
                -32600,
                "request id must be a string or integer",
                request_id=request_id,
            )
        if isinstance(raw_request_id, str) and request_id is None:
            raise ProtocolError(
                -32600,
                f"request id exceeds {MAX_REQUEST_ID_BYTES} UTF-8 bytes",
                request_id=request_id,
            )

        method = value.get("method")
        if not isinstance(method, str) or not method:
            raise ProtocolError(
                -32600,
                "request method must be a nonempty string",
                request_id=request_id,
            )
        params = value.get("params", {})
        if not isinstance(params, dict):
            raise ProtocolError(
                -32602,
                "request params must be an object",
                request_id=request_id,
            )
        return {"id": raw_request_id, "method": method, "params": params}

    def encode(self, value: Mapping[str, Any]) -> bytes:
        return self._bounded_json(value, request_id=None)

    def response(self, request_id: str | int | None, result: object) -> bytes:
        value = {"jsonrpc": "2.0", "id": request_id, "result": result}
        return self._bounded_json(value, request_id=request_id)

    def response_fits(self, request_id: str | int | None, result: object) -> bool:
        """Return whether a response can be sent without a size fallback."""

        try:
            return (
                len(
                    self._json_bytes(
                        {"jsonrpc": "2.0", "id": request_id, "result": result}
                    )
                )
                <= MAX_FRAME_BYTES
            )
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            return False

    def error_response(
        self,
        request_id: str | int | None,
        code: int,
        message: str,
        data: object | None = None,
    ) -> bytes:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        value = {"jsonrpc": "2.0", "id": request_id, "error": error}
        return self._bounded_json(value, request_id=request_id)

    def notification(
        self, event: str, session_id: str | None, **fields: object
    ) -> bytes:
        params: dict[str, object] = {"event": event}
        if session_id is not None:
            params["session_id"] = session_id
        params.update(fields)
        value = {"jsonrpc": "2.0", "method": "event", "params": params}
        try:
            payload = self._json_bytes(value)
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            return self._error_notification(
                session_id, event, "event could not be serialized"
            )
        if len(payload) <= MAX_FRAME_BYTES:
            return payload
        return self._error_notification(
            session_id, event, "event exceeds the size limit"
        )

    async def write(self, writer: Any, payload: bytes) -> None:
        """Write one codec-produced frame and absorb a disconnected peer."""

        if len(payload) > MAX_FRAME_BYTES:
            payload = self.error_response(
                None,
                -32007,
                f"outbound frame exceeds {MAX_FRAME_BYTES} bytes",
            )
        try:
            writer.write(payload)
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            pass

    def _bounded_json(
        self, value: Mapping[str, Any], *, request_id: str | int | None
    ) -> bytes:
        try:
            payload = self._json_bytes(value)
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            return self._minimal_error(
                request_id,
                -32000,
                "response could not be serialized",
            )
        if len(payload) <= MAX_FRAME_BYTES:
            return payload
        return self._minimal_error(
            request_id,
            -32007,
            f"outbound frame exceeds {MAX_FRAME_BYTES} bytes",
        )

    @staticmethod
    def _json_bytes(value: Mapping[str, Any]) -> bytes:
        return (
            json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def _minimal_error(
        self, request_id: str | int | None, code: int, message: str
    ) -> bytes:
        safe_id = _safe_request_id(request_id)
        value = {
            "jsonrpc": "2.0",
            "id": safe_id,
            "error": {"code": code, "message": message},
        }
        payload = self._json_bytes(value)
        if len(payload) <= MAX_FRAME_BYTES:
            return payload
        value["id"] = None
        return self._json_bytes(value)

    def _error_notification(
        self, session_id: str | None, event: str, message: str
    ) -> bytes:
        params: dict[str, object] = {
            "event": "error",
            "error": {"code": "frame_too_large", "message": message},
            "data": {"event": event},
        }
        if session_id is not None:
            params["session_id"] = session_id
        try:
            payload = self._json_bytes(
                {"jsonrpc": "2.0", "method": "event", "params": params}
            )
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            params.pop("session_id", None)
            payload = self._json_bytes(
                {"jsonrpc": "2.0", "method": "event", "params": params}
            )
        if len(payload) <= MAX_FRAME_BYTES:
            return payload
        params.pop("data", None)
        params.pop("session_id", None)
        return self._json_bytes({"jsonrpc": "2.0", "method": "event", "params": params})


def _parse_integer(value: str) -> int | _OversizedInteger:
    digits = value.lstrip("-")
    if len(digits) > MAX_NUMERIC_ID_DIGITS:
        return _OversizedInteger(len(digits))
    return int(value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _safe_request_id(value: object) -> str | int | None:
    try:
        if type(value) is str:
            return value if len(value.encode("utf-8")) <= MAX_REQUEST_ID_BYTES else None
        if type(value) is int:
            return value if len(str(abs(value))) <= MAX_NUMERIC_ID_DIGITS else None
    except (UnicodeEncodeError, ValueError):
        return None
    return None


def _request_id_hint(line: bytes) -> str | int | None:
    """Read a top-level id even when the rest of a frame is malformed."""

    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        text = line[: exc.start].decode("utf-8")
    decoder = json.JSONDecoder(
        parse_int=_parse_integer,
        parse_constant=_reject_constant,
    )
    position = 0
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        return None
    position += 1
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text) and text[position] == "}":
            return None
        try:
            key, position = decoder.raw_decode(text, position)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return None
        if not isinstance(key, str):
            return None
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != ":":
            return None
        position += 1
        while position < len(text) and text[position].isspace():
            position += 1
        try:
            value, position = decoder.raw_decode(text, position)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return None
        if key == "id":
            return _safe_request_id(value)
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != ",":
            return None
        position += 1
    return None


def bounded(value: str, limit: int = TOOL_OUTPUT_MAX_BYTES) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    retained = raw[:limit].decode("utf-8", errors="ignore")
    return f"{retained}... [truncated: {len(raw) - limit} bytes]"


__all__ = [
    "MAX_FRAME_BYTES",
    "MAX_NUMERIC_ID_DIGITS",
    "MAX_REQUEST_ID_BYTES",
    "PROTOCOL_VERSION",
    "TOOL_OUTPUT_MAX_BYTES",
    "FrameCodec",
    "ProtocolError",
    "bounded",
]
