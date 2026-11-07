"""Bounded JSON-RPC framing for the zeta frontend protocol."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from enum import Enum, StrEnum
from typing import Any


class ErrorDescriptor(Enum):
    """One documented JSON-RPC error variant."""

    PARSE_ERROR = (-32700, None, "invalid JSON or invalid UTF-8 frame")
    INVALID_REQUEST = (-32600, None, "invalid JSON-RPC request or duplicate handshake")
    METHOD_NOT_FOUND = (-32601, None, "method or negotiated feature is not supported")
    INVALID_PARAMS = (
        -32602,
        None,
        "params have the wrong shape, value, or negotiated capability",
    )
    INTERNAL_ERROR = (-32000, None, "unexpected server or storage failure")
    CONNECTION_CAPACITY = (
        -32001,
        None,
        "requested controller or observer connection role is at capacity",
    )
    PROTOCOL_MISMATCH = (
        -32002,
        None,
        "handshake is missing or the protocol version does not match",
    )
    NO_ACTIVE_SESSION = (
        -32003,
        None,
        "no active session, or the active session requires a host binding",
    )
    OPERATION_BUSY = (
        -32004,
        None,
        "operation is blocked by active turn, tool, or background-agent work",
    )
    STATE_UNAVAILABLE = (-32005, None, "requested turn or account state is unavailable")
    APPROVAL_UNAVAILABLE = (
        -32006,
        None,
        "approval request is missing, resolved, or reserved",
    )
    FRAME_TOO_LARGE = (-32007, None, "outbound frame exceeds the size limit")
    HISTORY_SNAPSHOT_STALE = (
        -32009,
        "stale_history_snapshot",
        "session history snapshot is stale",
    )
    DELIVERY_ID_EVICTED = (-32010, None, "delivery ID details were evicted")
    CONTROLLER_OUTPUT_OVERFLOW = (-32011, None, "controller output overflow")
    OBSERVER_OUTPUT_OVERFLOW = (-32011, None, "observer output overflow")
    CONTENT_NOT_FOUND = (-32012, "content_not_found", "content handle was not found")
    ACTIVE_SESSION = (-32005, "active_session", "the active session cannot be deleted")
    ACCOUNT_LOCK_TIMEOUT = (
        -32008,
        "account_lock_timeout",
        "another process holds the selected account lock",
    )
    APPROVAL_PERSISTENCE_FAILED = (
        -32000,
        "approval_persistence_failed",
        "approval state could not be persisted",
    )
    HOST_BINDING_REQUIRED = (
        -32003,
        "host_binding_required",
        "the session requires a host binding",
    )
    INVALID_CONTENT_READ = (
        -32602,
        "invalid_content_read",
        "content_read parameters are invalid",
    )
    LOGIN_IN_PROGRESS = (-32005, "login_in_progress", "a login is already pending")
    OBSERVER_READ_ONLY = (
        -32008,
        "observer_read_only",
        "a read-only observer requested a write method",
    )
    PROJECT_NOT_FOUND = (
        -32602,
        "project_not_found",
        "the requested project does not exist",
    )
    PROJECT_REGISTRY_BUSY = (
        -32000,
        "project_registry_busy",
        "another process holds the project registry lock",
    )
    SESSION_NOT_FOUND = (
        -32602,
        "session_not_found",
        "the requested session does not exist",
    )

    def __init__(self, code: int, data_code: str | None, meaning: str) -> None:
        self.code = code
        self.data_code = data_code
        self.meaning = meaning

    def data(self, fields: dict[str, object] | None = None) -> dict[str, object] | None:
        if self.data_code is None:
            return fields
        data = {"code": self.data_code}
        if fields is not None:
            data.update(fields)
            data["code"] = self.data_code
        return data


class EventDescriptor(StrEnum):
    """One documented JSON-RPC event name."""

    AGENT_START = "agent_start"
    TURN_START = "turn_start"
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"
    MESSAGE_START = "message_start"
    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_MESSAGE = "assistant_message"
    RETRY = "retry"
    ASSISTANT_RESET = "assistant_reset"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_END = "approval_end"
    TOOL_START = "tool_start"
    TOOL_OUTPUT = "tool_output"
    TOOL_END = "tool_end"
    SUB_AGENT_RECEIPT = "sub_agent_receipt"
    CHILD_IDLE = "child_idle"
    TASK_EXIT_NOTIFICATION = "task_exit_notification"
    PROJECT_INBOX_NOTIFICATION = "project_inbox_notification"
    CHILD_QUESTION = "child_question"
    CHILD_QUESTION_WITHDRAWN = "child_question_withdrawn"
    CHILD_MESSAGE_DELIVERED = "child_message_delivered"
    ACCOUNT_LIMIT = "account_limit"
    ACCOUNT_STATE_UNKNOWN = "account_state_unknown"
    TURN_END = "turn_end"
    AGENT_END = "agent_end"
    ERROR = "error"
    DELIVERY_STATUS = "delivery_status"
    MEMORY_UPDATED = "memory_updated"
    TURN_ABORTED = "turn_aborted"
    USAGE = "usage"
    USER_MESSAGE = "user_message"


PROTOCOL_VERSION = "1.2"
SUPPORTED_PROTOCOL_VERSIONS = ("1.0", "1.1", PROTOCOL_VERSION)
# Optional 1.1 features. A client opts in through ``hello.features``; the
# server echoes the supported subset in ``capabilities.features``.
FEATURES = (
    "session_cwd",
    "new_session_settings",
    "user_message_event",
    "user_message_delivery_id",
    "send_images_delivery_id",
    "list_sessions_paging",
    "memory_updated",
    "ping",
    "assistant_reset",
    "model_input_ids",
    "projects",
    "projects-memory-v2",
    "abort_scope",
    "turn_context",
    "delivery_id",
    "serve_observers",
    "approval_history",
    "approval_reconnect",
    "session_history_backward",
    "session_host_commands",
    "session_history_fidelity",
    "content_read",
)
MAX_FRAME_BYTES = 1_048_576
MAX_REQUEST_ID_BYTES = 128
MAX_NUMERIC_ID_DIGITS = 128
TOOL_OUTPUT_MAX_BYTES = 8_000


def hello_result(
    protocol_version: str, capabilities: dict[str, object], warning: str | None
) -> dict[str, object]:
    result: dict[str, object] = {
        "protocol_version": protocol_version,
        "server": "zeta",
        "capabilities": capabilities,
    }
    if warning is not None:
        result["notices"] = [warning]
    return result


class _OversizedInteger:
    def __init__(self, digits: int) -> None:
        self.digits = digits


class FrameWriteCancelled(asyncio.CancelledError):
    """Cancellation that arrived after a frame write started.

    ``written`` is false if the bounded write did not finish, including after
    transport abort when a cancellation-resistant drain had to be detached.
    """

    def __init__(self, written: bool) -> None:
        super().__init__()
        self.written = written


# Deadline for a canceled caller's started frame write before its transport is
# aborted. A stalled client must not hang abort or shutdown.
FRAME_CANCEL_GRACE = 1.0


async def finish_frame_write(codec: FrameCodec, writer: Any, payload: bytes) -> bool:
    """Write one frame; through cancellation, report whether it was written.

    Cancellation before this call writes nothing. Cancellation during the
    write waits at most ``FRAME_CANCEL_GRACE`` for it to finish, then aborts
    the transport; either way it raises ``FrameWriteCancelled``.
    """

    task = asyncio.ensure_future(codec.write(writer, payload))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        pass
    raise FrameWriteCancelled(await _settle_canceled_write(task, writer))


async def _settle_canceled_write(task: asyncio.Future[bool], writer: Any) -> bool:
    """Wait for the deadline, abort the transport, then finish cleanup."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + FRAME_CANCEL_GRACE
    aborted = False
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            if aborted:
                # Supported transports settle after abort. Keep observing an
                # abnormal transport without retaining it in process-global
                # state or coupling it to another connection.
                task.add_done_callback(_observe_write)
                return False
            aborted = True
            deadline = loop.time() + FRAME_CANCEL_GRACE
            transport = getattr(writer, "transport", None)
            if transport is not None:
                try:
                    transport.abort()
                except Exception:  # noqa: BLE001, S110 - abort must not block cleanup
                    pass
            task.cancel()
            continue
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            continue
    if task.cancelled():
        return False
    error = task.exception()
    if error is not None:
        raise error
    return not aborted and task.result()


def _observe_write(task: asyncio.Future[bool]) -> None:
    if not task.cancelled():
        task.exception()


class ProtocolError(Exception):
    """A JSON-RPC error that can be returned to the client."""

    def __init__(
        self,
        descriptor: ErrorDescriptor,
        message: str,
        data: dict[str, object] | None = None,
        *,
        request_id: str | int | None = None,
    ) -> None:
        super().__init__(message)
        self.descriptor = descriptor
        self.code = descriptor.code
        self.message = message
        self.data = descriptor.data(data)
        self.request_id = _safe_request_id(request_id)


def negotiate_protocol_version(requested: object, advertised: object) -> str:
    """Select one supported version from a hello request."""

    if requested not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ProtocolError(
            ErrorDescriptor.PROTOCOL_MISMATCH,
            "unsupported protocol version",
            {"requested": requested, "supported": list(SUPPORTED_PROTOCOL_VERSIONS)},
        )
    if requested == "1.0" and advertised in SUPPORTED_PROTOCOL_VERSIONS[1:]:
        return advertised
    return requested


class FrameCodec:
    """Parse requests and serialize every bounded wire frame."""

    def parse_request(self, line: bytes) -> dict[str, Any]:
        request_id = _request_id_hint(line[:MAX_FRAME_BYTES])
        if len(line) > MAX_FRAME_BYTES:
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST,
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
                ErrorDescriptor.PARSE_ERROR,
                "invalid JSON frame",
                request_id=request_id,
            ) from exc
        if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST,
                "request must be a JSON-RPC 2.0 object",
                request_id=request_id,
            )

        raw_request_id = value.get("id")
        request_id = _safe_request_id(raw_request_id)
        if isinstance(raw_request_id, _OversizedInteger):
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST,
                f"numeric request id exceeds {MAX_NUMERIC_ID_DIGITS} digits",
                request_id=request_id,
            )
        if type(raw_request_id) is not str and type(raw_request_id) is not int:
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST,
                "request id must be a string or integer",
                request_id=request_id,
            )
        if isinstance(raw_request_id, str) and request_id is None:
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST,
                f"request id exceeds {MAX_REQUEST_ID_BYTES} UTF-8 bytes",
                request_id=request_id,
            )

        method = value.get("method")
        if not isinstance(method, str) or not method:
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST,
                "request method must be a nonempty string",
                request_id=request_id,
            )
        params = value.get("params", {})
        if not isinstance(params, dict):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "request params must be an object",
                request_id=request_id,
            )
        return {"id": raw_request_id, "method": method, "params": params}

    def response(self, request_id: str | int | None, result: object) -> bytes:
        value = {"jsonrpc": "2.0", "id": request_id, "result": result}
        return self._bounded_json(value, request_id=request_id)

    def serialized_size(self, value: object) -> int:
        """Return the exact encoded size under this codec's JSON settings."""

        return len(self._json_bytes(value))

    def response_size(self, request_id: str | int | None, result: object) -> int:
        """Return the exact encoded response size without applying a fallback."""

        return self.serialized_size(
            {"jsonrpc": "2.0", "id": request_id, "result": result}
        )

    def response_fits(self, request_id: str | int | None, result: object) -> bool:
        """Return whether a response can be sent without a size fallback."""

        try:
            return self.response_size(request_id, result) <= MAX_FRAME_BYTES
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            return False

    def error_response(
        self,
        request_id: str | int | None,
        descriptor: ErrorDescriptor,
        message: str,
        data: dict[str, object] | None = None,
    ) -> bytes:
        error: dict[str, Any] = {"code": descriptor.code, "message": message}
        wire_data = descriptor.data(data)
        if wire_data is not None:
            error["data"] = wire_data
        value = {"jsonrpc": "2.0", "id": request_id, "error": error}
        return self._bounded_json(value, request_id=request_id)

    def notification(
        self, event: EventDescriptor, session_id: str | None, **fields: object
    ) -> bytes:
        params: dict[str, object] = {"event": event.value}
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

    async def write(self, writer: Any, payload: bytes) -> bool:
        """Write one codec-produced frame; report a disconnected peer."""

        if len(payload) > MAX_FRAME_BYTES:
            payload = self.error_response(
                None,
                ErrorDescriptor.FRAME_TOO_LARGE,
                f"outbound frame exceeds {MAX_FRAME_BYTES} bytes",
            )
        try:
            writer.write(payload)
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            return False
        return True

    def _bounded_json(
        self, value: Mapping[str, Any], *, request_id: str | int | None
    ) -> bytes:
        try:
            payload = self._json_bytes(value)
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            return self._minimal_error(
                request_id,
                ErrorDescriptor.INTERNAL_ERROR,
                "response could not be serialized",
            )
        if len(payload) <= MAX_FRAME_BYTES:
            return payload
        return self._minimal_error(
            request_id,
            ErrorDescriptor.FRAME_TOO_LARGE,
            f"outbound frame exceeds {MAX_FRAME_BYTES} bytes",
        )

    @staticmethod
    def _json_bytes(value: object) -> bytes:
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
        self, request_id: str | int | None, descriptor: ErrorDescriptor, message: str
    ) -> bytes:
        safe_id = _safe_request_id(request_id)
        value = {
            "jsonrpc": "2.0",
            "id": safe_id,
            "error": {"code": descriptor.code, "message": message},
        }
        payload = self._json_bytes(value)
        if len(payload) <= MAX_FRAME_BYTES:
            return payload
        value["id"] = None
        return self._json_bytes(value)

    def _error_notification(
        self, session_id: str | None, event: EventDescriptor, message: str
    ) -> bytes:
        params: dict[str, object] = {
            "event": EventDescriptor.ERROR.value,
            "error": {"code": "frame_too_large", "message": message},
            "data": {"event": event.value},
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
    "FEATURES",
    "MAX_FRAME_BYTES",
    "MAX_NUMERIC_ID_DIGITS",
    "MAX_REQUEST_ID_BYTES",
    "PROTOCOL_VERSION",
    "SUPPORTED_PROTOCOL_VERSIONS",
    "TOOL_OUTPUT_MAX_BYTES",
    "FrameCodec",
    "ProtocolError",
    "bounded",
    "negotiate_protocol_version",
]
