"""Bounded JSON-RPC framing for the zeta frontend protocol."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from enum import Enum, StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class _Frame(bytes):
    """A complete wire frame constructed by FrameCodec."""


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
    STALE_EVENT_SEQUENCE = (
        -32009,
        "stale_event_sequence",
        "replay continuity was invalidated",
    )
    DELIVERY_ID_EVICTED = (-32010, None, "delivery ID details were evicted")
    CONTROLLER_OUTPUT_OVERFLOW = (-32011, None, "controller output overflow")
    OBSERVER_OUTPUT_OVERFLOW = (-32011, None, "observer output overflow")
    CONTENT_NOT_FOUND = (-32012, "content_not_found", "content handle was not found")
    CHILD_RUN_NOT_FOUND = (-32012, "child_run_not_found", "child run was not found")
    TRANSCRIPT_REPAIR_REQUIRES_UPGRADE = (
        -32013,
        "transcript_repair_requires_upgrade",
        "transcript repair requires the stored-format upgrade",
    )
    CORRUPT_TRANSCRIPT = (
        -32000,
        "corrupt_transcript",
        "transcript is corrupt or uses an unsupported format",
    )
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
    INVALID_SESSION_GRANTS_PARAMS = (
        -32602,
        "invalid_session_grants_params",
        "session_grants parameters are invalid",
    )
    INVALID_SESSION_GRANTS_REVOKE = (
        -32602,
        "invalid_session_grants_revoke",
        "revoke_session_grants parameters are invalid",
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
    SESSION_GRANT_NOT_FOUND = (
        -32602,
        "session_grant_not_found",
        "the requested session grant does not exist",
    )
    SESSION_NOT_FOUND = (
        -32602,
        "session_not_found",
        "the requested session does not exist",
    )
    SESSION_ID_AMBIGUOUS = (
        -32602,
        "session_id_ambiguous",
        "the session id prefix matches more than one session",
    )
    SESSION_MISMATCH = (
        -32003,
        "session_mismatch",
        "the requested session is not the attached session",
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


class SlashResultKindDescriptor(StrEnum):
    """One documented slash_run result kind."""

    OUTPUT = "output"
    MODEL_INPUT = "model_input"
    CLIENT_ONLY = "client_only"
    UNKNOWN = "unknown"
    ERROR = "error"


class SlashErrorCodeDescriptor(StrEnum):
    """One documented structured slash_run error code."""

    MCP_CREDENTIAL_STORE = "mcp_credential_store"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


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
    CHILD_HOLD = "child_hold"
    TASK_EXIT_NOTIFICATION = "task_exit_notification"
    PROJECT_INBOX_NOTIFICATION = "project_inbox_notification"
    PROJECT_INBOX_HEALTH = "project_inbox_health"
    CHILD_QUESTION = "child_question"
    CHILD_QUESTION_WITHDRAWN = "child_question_withdrawn"
    CHILD_MESSAGE_DELIVERED = "child_message_delivered"
    ACCOUNT_LIMIT = "account_limit"
    ACCOUNT_STATE_UNKNOWN = "account_state_unknown"
    TURN_END = "turn_end"
    SERVED_TURN_END = "served_turn_end"
    AGENT_END = "agent_end"
    ERROR = "error"
    DELIVERY_STATUS = "delivery_status"
    MEMORY_UPDATED = "memory_updated"
    TURN_ABORTED = "turn_aborted"
    USAGE = "usage"
    USER_MESSAGE = "user_message"
    SESSION_GRANTS_CHANGED = "session_grants_changed"
    REPLAY_STREAM_ENDED = "replay_stream_ended"
    CONFIG_CHANGED = "config_changed"
    CHILD_STARTED = "child_started"
    CHILD_PROGRESS = "child_progress"
    CHILD_ENDED = "child_ended"
    BACKGROUND_TASK_UPDATED = "background_task_updated"
    TODO_UPDATED = "todo_updated"


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
    "live_config_changed",
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
    "session_replay_seq",
    "child_run_history",
    "child_run_events",
    "served_turn_end",
    "content_read",
    "session_grants",
    "path_completion",
    "child_runs",
    "delivery_session_guard",
    "run_work",
    "approval_restart_reason",
)


class Access(StrEnum):
    READ = "read"
    CONTROLLER_READ = "controller_read"
    SESSION_MUTATION = "session_mutation"
    DELIVERY = "delivery"
    CONTROLLER_WRITE = "controller_write"


class MethodDescriptor(Enum):
    """The single declaration of a served request's wire capability."""

    LIST_SESSIONS = ("list_sessions", None, None, Access.READ)
    NEW_SESSION = ("new_session", None, None, Access.CONTROLLER_WRITE)
    RESUME = ("resume", None, None, Access.CONTROLLER_WRITE)
    SEND = ("send", None, None, Access.DELIVERY)
    STEER = ("steer", None, None, Access.DELIVERY)
    APPROVE = ("approve", None, None, Access.SESSION_MUTATION)
    DENY = ("deny", None, None, Access.SESSION_MUTATION)
    ABORT = ("abort", None, None, Access.SESSION_MUTATION)
    STATUS = ("status", None, None, Access.READ)
    RENAME_SESSION = ("rename_session", "1.1", None, Access.SESSION_MUTATION)
    DELETE_SESSION = ("delete_session", "1.1", None, Access.CONTROLLER_WRITE)
    REPAIR_TRANSCRIPT = ("repair_transcript", "1.1", None, Access.SESSION_MUTATION)
    SESSION_TREE = ("session_tree", "1.1", None, Access.READ)
    SESSION_HISTORY = ("session_history", "1.1", None, Access.READ)
    SESSION_HISTORY_AND_SUBSCRIBE = (
        "session_history_and_subscribe",
        "1.1",
        "session_replay_seq",
        Access.READ,
    )
    CHILD_RUN_HISTORY = ("child_run_history", "1.1", "child_run_history", Access.READ)
    CHILD_RUN_EVENTS = ("child_run_events", "1.1", "child_run_events", Access.READ)
    CONTENT_READ = ("content_read", "1.1", "content_read", Access.READ)
    SWITCH_BRANCH = ("switch_branch", "1.1", None, Access.SESSION_MUTATION)
    FORK_MESSAGE = ("fork_message", "1.1", None, Access.SESSION_MUTATION)
    MODEL_CATALOG = ("model_catalog", "1.1", None, Access.READ)
    SESSION_SETTINGS = ("session_settings", "1.1", None, Access.READ)
    SET_SETTINGS = ("set_settings", "1.1", None, Access.SESSION_MUTATION)
    SET_COMPACTION = ("set_compaction", "1.1", None, Access.SESSION_MUTATION)
    SEND_IMAGES = ("send_images", "1.1", None, Access.DELIVERY)
    PATH_COMPLETE = (
        "path_complete",
        None,
        "path_completion",
        Access.CONTROLLER_READ,
    )
    SLASH_LIST = ("slash_list", "1.1", None, Access.CONTROLLER_READ)
    SLASH_RUN = ("slash_run", "1.1", None, Access.SESSION_MUTATION)
    ACCOUNT_LIST = ("account_list", "1.1", None, Access.CONTROLLER_READ)
    ACCOUNT_LOGOUT = ("account_logout", "1.1", None, Access.CONTROLLER_WRITE)
    ACCOUNT_REFRESH = ("account_refresh", "1.1", None, Access.CONTROLLER_WRITE)
    ACCOUNT_SWITCH = ("account_switch", "1.1", None, Access.CONTROLLER_WRITE)
    LOGIN_START = ("login_start", "1.1", None, Access.CONTROLLER_WRITE)
    LOGIN_STATUS = ("login_status", "1.1", None, Access.CONTROLLER_READ)
    LOGIN_CANCEL = ("login_cancel", "1.1", None, Access.CONTROLLER_WRITE)
    LOGIN_PROVIDERS = ("login_providers", "1.1", None, Access.CONTROLLER_READ)
    PING = ("ping", "1.1", "ping", Access.READ)
    SET_TURN_CONTEXT = (
        "set_turn_context",
        "1.1",
        "turn_context",
        Access.SESSION_MUTATION,
    )
    LIST_PROJECTS = ("list_projects", "1.1", "projects", Access.CONTROLLER_READ)
    PROJECT_SHOW = ("project_show", "1.1", "projects", Access.CONTROLLER_READ)
    PROJECT_MEMORY_LOG = (
        "project_memory_log",
        "1.1",
        "projects",
        Access.CONTROLLER_READ,
    )
    PROJECT_INBOX = ("project_inbox", "1.1", "projects", Access.CONTROLLER_WRITE)
    CLEAR_STEERING = ("clear_steering", None, "abort_scope", Access.SESSION_MUTATION)
    DELIVERY_STATUS = ("delivery_status", None, "delivery_id", Access.READ)
    SESSION_GRANTS = ("session_grants", "1.1", "session_grants", Access.READ)
    REVOKE_SESSION_GRANTS = (
        "revoke_session_grants",
        "1.1",
        "session_grants",
        Access.SESSION_MUTATION,
    )
    RUN_WORK_LIST = ("run_work_list", None, "run_work", Access.CONTROLLER_READ)
    BACKGROUND_TASK_CANCEL = (
        "background_task_cancel",
        None,
        "run_work",
        Access.SESSION_MUTATION,
    )
    CHILD_RUN_LIST = ("child_run_list", None, "child_runs", Access.READ)
    CHILD_RUN_SEND = ("child_run_send", None, "child_runs", Access.CONTROLLER_WRITE)

    def __init__(
        self, wire: str, since: str | None, feature: str | None, access: Access
    ) -> None:
        if feature is not None and feature not in FEATURES:
            raise ValueError(f"unknown method feature: {feature}")
        self.wire, self.since, self.feature, self.access = wire, since, feature, access

    @classmethod
    def for_wire(cls, wire: str) -> MethodDescriptor | None:
        return next((item for item in cls if item.wire == wire), None)

    @classmethod
    def advertised(cls, version: str, features: frozenset[str], role: str) -> list[str]:
        negotiated = tuple(int(part) for part in version.split("."))
        methods = [
            item.wire
            for item in cls
            if (
                item.since is None
                or negotiated >= tuple(int(part) for part in item.since.split("."))
            )
            and (item.feature is None or item.feature in features)
            and (role != "observer" or item.access is Access.READ)
        ]
        return sorted(methods) if role == "observer" else methods


EVENT_ACCESS = {
    EventDescriptor.BACKGROUND_TASK_UPDATED: Access.CONTROLLER_READ,
    EventDescriptor.TODO_UPDATED: Access.CONTROLLER_READ,
}


CHILD_RUN_EVENTS_PREREQUISITES = frozenset(
    descriptor.feature
    for descriptor in (
        MethodDescriptor.SESSION_HISTORY_AND_SUBSCRIBE,
        MethodDescriptor.CHILD_RUN_HISTORY,
        MethodDescriptor.CHILD_RUN_LIST,
    )
    if descriptor.feature is not None
)


_LOGIN_METHODS = frozenset(
    MethodDescriptor[name]
    for name in ("LOGIN_START", "LOGIN_STATUS", "LOGIN_CANCEL", "LOGIN_PROVIDERS")
)
_ERGONOMICS_METHODS = frozenset(
    MethodDescriptor[name]
    for name in (
        "RENAME_SESSION",
        "DELETE_SESSION",
        "REPAIR_TRANSCRIPT",
        "SESSION_TREE",
        "SESSION_HISTORY",
        "CONTENT_READ",
        "MODEL_CATALOG",
        "SESSION_SETTINGS",
        "SET_SETTINGS",
        "SET_COMPACTION",
        "SEND_IMAGES",
        "SWITCH_BRANCH",
        "FORK_MESSAGE",
        "SLASH_LIST",
        "SLASH_RUN",
        "ACCOUNT_LIST",
        "ACCOUNT_LOGOUT",
        "ACCOUNT_REFRESH",
        "ACCOUNT_SWITCH",
    )
)
_PROJECT_METHODS = frozenset(
    MethodDescriptor[name]
    for name in ("LIST_PROJECTS", "PROJECT_SHOW", "PROJECT_MEMORY_LOG", "PROJECT_INBOX")
)


MAX_FRAME_BYTES = 1_048_576
MAX_REQUEST_ID_BYTES = 128
MAX_NUMERIC_ID_DIGITS = 128
TOOL_OUTPUT_MAX_BYTES = 8_000
# A reconnecting client is given at most this many pending-approval rows.
RECONNECT_APPROVAL_LIMIT = 64


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


class FrameWriteOwner:
    def __init__(self) -> None:
        self._active: set[asyncio.Task[Any]] = set()
        self._orphaned: set[asyncio.Future[bool]] = set()

    async def write(
        self,
        codec: FrameCodec,
        writer: Any,
        payload: _Frame,
        lock: asyncio.Lock,
    ) -> bool:
        task = asyncio.current_task()
        assert task is not None
        self._active.add(task)
        try:
            async with lock:
                return await finish_frame_write(codec, writer, payload, owner=self)
        finally:
            self._active.discard(task)

    def register(self, task: asyncio.Future[bool]) -> None:
        self._orphaned.add(task)
        task.add_done_callback(self._forget)

    def _forget(self, task: asyncio.Future[bool]) -> None:
        self._orphaned.discard(task)
        _observe_write(task)

    async def aclose(self) -> None:
        deadline = asyncio.get_running_loop().time() + 2 * FRAME_CANCEL_GRACE
        while self._active or self._orphaned:
            for write in tuple(self._active):
                write.cancel()
            # Let canceled writers reach their handler and register the inner
            # write here, so this pass cancels it instead of the next one.
            await asyncio.sleep(0)
            pending = set(self._active) | self._orphaned
            if not pending:
                return
            for write in pending:
                write.cancel()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.wait(pending, timeout=min(remaining, FRAME_CANCEL_GRACE))


async def finish_frame_write(
    codec: FrameCodec,
    writer: Any,
    payload: _Frame,
    *,
    owner: FrameWriteOwner | None = None,
) -> bool:
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
    if owner is not None:
        owner.register(task)
    raise FrameWriteCancelled(await _settle_canceled_write(task, writer, owner=owner))


async def _settle_canceled_write(
    task: asyncio.Future[bool],
    writer: Any,
    *,
    owner: FrameWriteOwner | None = None,
) -> bool:
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
                if owner is None:
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


def protocol_error(descriptor: MethodDescriptor) -> ProtocolError:
    message = {
        **dict.fromkeys(_LOGIN_METHODS, "login is unavailable on this server"),
        **dict.fromkeys(_ERGONOMICS_METHODS, "request requires protocol 1.1"),
        MethodDescriptor.CONTENT_READ: "content_read requires the negotiated content_read feature",
        MethodDescriptor.PING: "ping requires the negotiated ping feature",
        MethodDescriptor.SET_TURN_CONTEXT: "set_turn_context requires negotiated turn_context feature",
        **dict.fromkeys(
            _PROJECT_METHODS,
            f"{descriptor.wire} requires the negotiated projects feature",
        ),
        MethodDescriptor.SESSION_GRANTS: "session grants require the negotiated session_grants feature",
        MethodDescriptor.REVOKE_SESSION_GRANTS: "session grants require the negotiated session_grants feature",
    }.get(descriptor, f"{descriptor.wire} requires protocol 1.1")
    return ProtocolError(ErrorDescriptor.METHOD_NOT_FOUND, message)


def feature_error(descriptor: MethodDescriptor) -> ProtocolError:
    message = {
        MethodDescriptor.CLEAR_STEERING: "clear_steering requires the negotiated abort_scope feature",
        MethodDescriptor.DELIVERY_STATUS: "delivery_status requires negotiated feature 'delivery_id'",
        MethodDescriptor.SET_TURN_CONTEXT: "set_turn_context requires negotiated turn_context feature",
        MethodDescriptor.SESSION_GRANTS: "session grants require the negotiated session_grants feature",
        MethodDescriptor.REVOKE_SESSION_GRANTS: "session grants require the negotiated session_grants feature",
        MethodDescriptor.PING: "ping requires the negotiated ping feature",
        MethodDescriptor.CONTENT_READ: "content_read requires the negotiated content_read feature",
        **dict.fromkeys(
            _PROJECT_METHODS,
            f"{descriptor.wire} requires the negotiated projects feature",
        ),
    }.get(
        descriptor,
        f"{descriptor.wire} requires the negotiated {descriptor.feature} feature",
    )
    return ProtocolError(ErrorDescriptor.METHOD_NOT_FOUND, message)


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

    def response(self, request_id: str | int | None, result: object) -> _Frame:
        value = {"jsonrpc": "2.0", "id": request_id, "result": result}
        return self._bounded_json(value, request_id=request_id)

    def _result_response(self, request_id: str | int | None, result: object) -> _Frame | None:
        """The result frame, or None when only an error frame could go out."""
        try:
            payload = self._json_bytes({"jsonrpc": "2.0", "id": request_id, "result": result})
        except (TypeError, UnicodeEncodeError, ValueError, RecursionError):
            return None
        return payload if len(payload) <= MAX_FRAME_BYTES else None

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
    ) -> _Frame:
        if not isinstance(descriptor, ErrorDescriptor):
            logger.error("refusing unregistered error descriptor: %r", descriptor)
            return self._minimal_error(
                request_id,
                ErrorDescriptor.INTERNAL_ERROR,
                ErrorDescriptor.INTERNAL_ERROR.meaning,
            )
        error: dict[str, Any] = {"code": descriptor.code, "message": message}
        wire_data = descriptor.data(data)
        if wire_data is not None:
            error["data"] = wire_data
        value = {"jsonrpc": "2.0", "id": request_id, "error": error}
        return self._bounded_json(value, request_id=request_id)

    def notification(
        self, event: EventDescriptor, session_id: str | None, **fields: object
    ) -> _Frame | None:
        if not isinstance(event, EventDescriptor):
            logger.error("refusing unregistered event descriptor: %r", event)
            return None
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

    async def write(self, writer: Any, payload: _Frame) -> bool:
        """Write one codec-produced frame; report a disconnected peer."""

        if not isinstance(payload, _Frame):
            logger.error("refusing frame not constructed by FrameCodec")
            return False
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
    ) -> _Frame:
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
    def _json_bytes(value: object) -> _Frame:
        return _Frame(
            (
                json.dumps(
                    value,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )

    def _minimal_error(
        self, request_id: str | int | None, descriptor: ErrorDescriptor, message: str
    ) -> _Frame:
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
    ) -> _Frame:
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


def bound_reconnect_approvals(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], bool]:
    """Cap a reconnecting client's pending-approval rows at RECONNECT_APPROVAL_LIMIT;
    the flag says the list was truncated. One owner for the bound so the live
    projection and the durable-store fallback cut off at the same place."""

    return rows[:RECONNECT_APPROVAL_LIMIT], len(rows) > RECONNECT_APPROVAL_LIMIT


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
    "bound_reconnect_approvals",
    "bounded",
    "negotiate_protocol_version",
]
