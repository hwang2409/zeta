from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.agent_progress import register_child_run_sink
from ..core.session_files import descend_directory
from ..core.store import ConversationStore, ObserverSnapshotChanged, TranscriptLoadError
from ..core.store._child_runs import (
    child_run_segments,
    parse_child_run_id,
    run_row_identity,
    verify_child_run_directory,
)
from ..core.thread_call import ThreadCallWorkers
from . import content, event_contract
from .diagnostics import log_controller_refusal
from .handlers import history, session, settings, validation
from .protocol import (
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    FrameWriteCancelled,
    ProtocolError,
    _Frame,
)
from .read_catalog import ReadCatalog
from .session_controllers import SessionController, SessionUse
from .session_replay import (
    DeferredFrame,
    ReplayScopeId,
    SessionReplay,
    history_features,
)
from .session_replay.children import ChildRunProjection
from .session_replay.output import ReplayOutput, _OutputFrame
from .session_replay.rows import RowFacts

logger = logging.getLogger(__name__)

CONNECTION_FLUSH_DEADLINE_SECONDS = 1.0

OBSERVER_REQUESTS = frozenset(
    {
        "list_sessions",
        "status",
        "session_tree",
        "session_history",
        "session_history_and_subscribe",
        "content_read",
        "model_catalog",
        "session_settings",
        "ping",
        "delivery_status",
        "session_grants",
    }
)
MAX_CONNECTIONS = 32
MAX_OBSERVERS = 8
OUTPUT_MAX_BYTES = 4 * 1024 * 1024
OUTPUT_MAX_FRAMES = 256
OUTPUT_OVERFLOW_CLOSE_DEADLINE = 1.0

# A contended-slot refusal logs once per holder, then at most once per this
# window per holder, with the suppressed count carried on the next log.
REFUSAL_LOG_INTERVAL_SECONDS = 30.0

# one owner for feature gates and optional wire fields on every output path
@dataclass(frozen=True, slots=True)
class _EventRoute:
    features: frozenset[str] = frozenset()
    legacy_observers: bool = False
    destination: Literal["session", "connection", "global"] = "session"


NO_FEATURE = _EventRoute()
EVENT_FEATURES = {
    EventDescriptor.AGENT_START: NO_FEATURE,
    EventDescriptor.TURN_START: NO_FEATURE,
    EventDescriptor.COMPACTION_START: NO_FEATURE,
    EventDescriptor.COMPACTION_END: NO_FEATURE,
    EventDescriptor.MESSAGE_START: NO_FEATURE,
    EventDescriptor.ASSISTANT_DELTA: NO_FEATURE,
    EventDescriptor.ASSISTANT_MESSAGE: NO_FEATURE,
    EventDescriptor.RETRY: NO_FEATURE,
    EventDescriptor.ASSISTANT_RESET: NO_FEATURE,
    EventDescriptor.APPROVAL_REQUEST: NO_FEATURE,
    EventDescriptor.APPROVAL_END: NO_FEATURE,
    EventDescriptor.TOOL_START: NO_FEATURE,
    EventDescriptor.TOOL_OUTPUT: NO_FEATURE,
    EventDescriptor.TOOL_END: NO_FEATURE,
    EventDescriptor.SUB_AGENT_RECEIPT: NO_FEATURE,
    EventDescriptor.CHILD_IDLE: NO_FEATURE,
    EventDescriptor.CHILD_RECOVERED: _EventRoute(frozenset({"child_recovery"})),
    EventDescriptor.CHILD_SUSPENDED: _EventRoute(frozenset({"child_recovery"})),
    EventDescriptor.CHILD_HOLD: NO_FEATURE,
    EventDescriptor.TASK_EXIT_NOTIFICATION: NO_FEATURE,
    EventDescriptor.PROJECT_INBOX_NOTIFICATION: NO_FEATURE,
    EventDescriptor.PROJECT_INBOX_HEALTH: NO_FEATURE,
    EventDescriptor.CHILD_QUESTION: NO_FEATURE,
    EventDescriptor.CHILD_STARTED: _EventRoute(frozenset({"child_run_events"})),
    EventDescriptor.CHILD_PROGRESS: _EventRoute(frozenset({"child_run_events"})),
    EventDescriptor.CHILD_ENDED: _EventRoute(frozenset({"child_run_events"})),
    EventDescriptor.CHILD_QUESTION_WITHDRAWN: NO_FEATURE,
    EventDescriptor.CHILD_MESSAGE_DELIVERED: NO_FEATURE,
    EventDescriptor.ACCOUNT_LIMIT: NO_FEATURE,
    EventDescriptor.ACCOUNT_STATE_UNKNOWN: NO_FEATURE,
    EventDescriptor.TURN_END: NO_FEATURE,
    EventDescriptor.SERVED_TURN_END: NO_FEATURE,
    EventDescriptor.AGENT_END: NO_FEATURE,
    EventDescriptor.ERROR: NO_FEATURE,
    EventDescriptor.DELIVERY_STATUS: _EventRoute(frozenset({"session_history_fidelity"})),
    EventDescriptor.MEMORY_UPDATED: _EventRoute(frozenset({"memory_updated"})),
    EventDescriptor.TURN_ABORTED: NO_FEATURE,
    EventDescriptor.USAGE: NO_FEATURE,
    EventDescriptor.USER_MESSAGE: _EventRoute(frozenset({"user_message_event"})),
    EventDescriptor.SESSION_GRANTS_CHANGED: _EventRoute(frozenset({"session_grants"}), legacy_observers=True),
    EventDescriptor.REPLAY_STREAM_ENDED: _EventRoute(destination="connection"),
    EventDescriptor.CONFIG_CHANGED: _EventRoute(frozenset({"live_config_changed"})),
    EventDescriptor.BACKGROUND_TASK_UPDATED: _EventRoute(frozenset({"run_work"})),
    EventDescriptor.SESSION_RETENTION: _EventRoute(frozenset({"session_retention"}), legacy_observers=True),
    EventDescriptor.TODO_UPDATED: _EventRoute(frozenset({"run_work"})),
    EventDescriptor.ZETA_UPDATED: _EventRoute(frozenset({"zeta_updated"}), destination="global"),
    EventDescriptor.PARKED_SESSION_CHANGED: _EventRoute(frozenset({"session_park"}), destination="connection"),
}
OPTIONAL_FIELD_FEATURES = {
    (EventDescriptor.USER_MESSAGE, "delivery_id"): frozenset(
        {"delivery_id", "user_message_delivery_id"}
    ),
}


async def _overflow_deadline_wait() -> None:
    await asyncio.sleep(OUTPUT_OVERFLOW_CLOSE_DEADLINE)


@dataclass(frozen=True, slots=True)
class Admission:
    kind: Literal[
        "admitted",
        "no_controller",
        "feature_rejected",
        "codec_rejected",
        "not_writable",
        "capacity_rejected",
    ]
    connection: Any | None = None

    write: Awaitable[bool] | None = None

    async def delivered(self) -> bool:
        return await self.write if self.write is not None else self.kind == "admitted"


@dataclass
class _OutputState:
    # closed: the writer died; nothing may enter the queue again
    phase: Literal["claimed_buffering", "flushing", "ready", "overflowing", "closed"] = (
        "claimed_buffering"
    )
    queue: deque[_OutputFrame] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    writer_task: asyncio.Task[None] | None = None
    current_frame: _OutputFrame | None = None
    current_write: asyncio.Task[tuple[bool, FrameWriteCancelled | None]] | None = None
    ready: asyncio.Future[bool] | None = None
    callers_settled: asyncio.Event = field(default_factory=asyncio.Event)
    pending_callers: int = 0
    retained_bytes: int = 0
    retained_frames: int = 0
    overflow_frame: _Frame | None = None
    overflow_deadline_task: asyncio.Task[None] | None = None
    overflowed: bool = False
    released: bool = False
    drained: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Connection:
    connection: Any
    role: str | None = None
    protocol_version: str = "1.0"
    features: frozenset[str] = frozenset()
    output: _OutputState | None = None
    paused: bool = False
    # set when the run loop sees the peer gone while a request still runs
    lost: bool = False
    # set when a controller hello starts waiting for this lost holder to release
    contended: asyncio.Event = field(default_factory=asyncio.Event)
    # set when the slot is released, so a waiting controller hello can proceed
    released: asyncio.Event = field(default_factory=asyncio.Event)


class ConnectionRegistry(ReplayOutput):
    """Owns connection claims, output, and bounded observer reads."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        publish_child_lifecycle: Callable[[EventDescriptor, ReplayScopeId, str, dict[str, object], int | None], None],
    ) -> None:
        self.loop = loop
        self._connections: dict[Any, _Connection] = {}
        # Writers of unregistered clients, held until they finish so shutdown can join them.
        self._retired_writers: set[asyncio.Task[None]] = set()
        # Rate-limit state for the contended-slot refusal log: the holder a
        # refusal was last logged against, that log's time, and how many later
        # refusals against the same holder were suppressed. The clock is an
        # attribute so a test can inject a deterministic one.
        self._refusal_clock: Callable[[], float] = time.monotonic
        self._refusal_logged_for: Any | None = None
        self._refusal_logged_at = 0.0
        self._refusal_suppressed = 0
        self._observers: list[_Connection] = []
        self._closed = False
        self.replay = SessionReplay()
        self.child_runs = ChildRunProjection(loop, publish_child_lifecycle)
        self._unregister_child_run_sink = register_child_run_sink(self.child_runs.sink)
        self._child_reader: tuple[tuple[str, str], ConversationStore] | None = None
        self._child_reader_lock = threading.Lock()
        self._reads = ThreadCallWorkers(loop, workers=2, name="zeta-read")
        self.state_reader = ReadCatalog(self.run_blocking)

    @property
    def frozen(self) -> bool:
        return self._closed

    @property
    def legacy_controller_holder(self) -> _Connection | None:
        return next((state for state in self._connections.values()
                     if state.role == "controller" and "multi_controller" not in state.features), None)

    def clients(self) -> tuple[Any, ...]:
        """The physical clients currently owned by this registry."""
        return tuple(state.connection for state in self._connections.values())

    def register(self, connection: Any) -> None:
        if self._closed:
            raise RuntimeError("connection registry is closed")
        if len(self._connections) >= MAX_CONNECTIONS:
            raise RuntimeError("connection limit reached")
        state = _Connection(connection)
        output = _OutputState()
        output.ready = self.loop.create_future()
        output.callers_settled.set()
        output.drained.set()
        state.output = output
        self._connections[connection] = state
        output.writer_task = asyncio.create_task(self._output_writer(state))

    def negotiate_role(
        self,
        connection: Any,
        requested_role: object,
        protocol_version: str,
        features: frozenset[str],
        *,
        max_role: str = "controller",
        multi_mode: bool = False,
    ) -> str:
        if requested_role not in {"controller", "observer"}:
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "role must be 'controller' or 'observer'",
            )
        role = str(requested_role)
        if max_role == "observer" and role == "controller":
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS, "credential permits observer role only"
            )
        if role == "observer" and (
            protocol_version == "1.0" or "serve_observers" not in features
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "observer requires protocol 1.1 and serve_observers",
            )
        try:
            if role == "observer" and not multi_mode and len(self._observers) >= MAX_OBSERVERS:
                raise RuntimeError("observer limit reached")
            self._claim(connection, role, protocol_version, features)
        except RuntimeError as exc:
            if role == "observer":
                raise ProtocolError(
                    ErrorDescriptor.CONNECTION_CAPACITY, "observer limit reached"
                ) from exc
            self._log_controller_refusal(connection)
            raise ProtocolError(
                ErrorDescriptor.CONNECTION_CAPACITY,
                "another client is already connected",
            ) from exc
        connection.role = role
        return role

    def require_observer_room(self, session_id: str) -> None:
        if sum(
            record.runtime.opened is not None and record.runtime.session_id == session_id
            for state in self._observers
            if (record := self.replay.controller_for(state.connection)) is not None
        ) >= MAX_OBSERVERS:
            raise ProtocolError(ErrorDescriptor.CONNECTION_CAPACITY, "observer limit reached")

    def log_controller_refusal(self, refused: Any) -> None:
        """Record the physical diagnostic for a semantic controller refusal."""
        self._log_controller_refusal(refused)

    def _log_controller_refusal(self, refused: Any) -> None:
        """Name the holder and the refused connection on a -32001 slot refusal.

        Diagnostic only. Logs the first refusal per holder, then at most one per
        ``REFUSAL_LOG_INTERVAL_SECONDS`` per holder, carrying the suppressed
        count. The holder's stuck request and await chain are added only when
        its connection is already closed while a request still runs.
        """
        holder = self.legacy_controller_holder
        if holder is None:
            return
        client = holder.connection
        now = self._refusal_clock()
        if (
            self._refusal_logged_for is client
            and now - self._refusal_logged_at < REFUSAL_LOG_INTERVAL_SECONDS
        ):
            self._refusal_suppressed += 1
            return
        suppressed = self._refusal_suppressed if self._refusal_logged_for is client else 0
        self._refusal_logged_for = client
        self._refusal_logged_at = now
        self._refusal_suppressed = 0
        log_controller_refusal(client, holder.output, refused, suppressed=suppressed)

    def _claim(
        self,
        connection: Any,
        role: str,
        protocol_version: str,
        features: frozenset[str],
    ) -> None:
        state = self._connections[connection]
        if state.role is not None:
            raise ValueError("connection already claimed")
        if role == "observer":
            self._observers.append(state)
        state.role = role
        # Legacy hello buffers its own pre-commit events. Multi claims activate independently.
        state.paused = role == "controller" and "multi_controller" in features
        assert state.output is not None
        state.output.overflow_frame = self._overflow_frame(role)
        state.protocol_version = protocol_version
        state.features = features

    def is_controller(self, connection: Any) -> bool:
        state = self._connections.get(connection)
        return state is not None and state.role == "controller"

    def output_state(self, connection: Any) -> _OutputState | None:
        state = self._connections.get(connection)
        return state.output if state is not None else None

    def mark_lost(self, connection: Any) -> asyncio.Event | None:
        """Mark a connection lost while its request still runs.

        Return the event the run loop waits on to learn a controller hello is
        now contending for the slot, or ``None`` if the connection is gone.
        """
        state = self._connections.get(connection)
        if state is None:
            return None
        state.lost = True
        return state.contended

    async def wait_for_lost_controller(self, deadline: float) -> None:
        """If the controller slot is held by a lost holder, signal contention
        and wait up to ``deadline`` seconds for the slot to be released."""
        state = self.legacy_controller_holder
        if state is None or not state.lost:
            return
        state.contended.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(state.released.wait(), deadline)

    def activate_controller(self, connection: Any) -> None:
        state = self._connections.get(connection)
        if state is not None and state.role == "controller":
            state.paused = False

    def _event_preflight(
        self,
        state: _Connection | None,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None,
        fields: dict[str, object],
        feature_fields: event_contract.FeatureFields | None = None,
    ) -> tuple[Admission, _Frame | None, _OutputState | None]:
        if state is None:
            return Admission("no_controller"), None, None
        if state.paused or state.role is None:
            return Admission("not_writable"), None, None
        if self._feature_rejected(state, event, required_feature, fields):
            return Admission("feature_rejected"), None, None
        fields = self._fields_for_connection(state, event, fields, feature_fields=feature_fields)
        frame = state.connection.codec.notification(event, session_id, **fields)
        if frame is None:
            return Admission("codec_rejected"), None, None
        output = state.output
        if output is None or output.phase not in {
            "claimed_buffering",
            "flushing",
            "ready",
        }:
            return Admission("not_writable"), None, None
        return Admission("admitted", state.connection), frame, output

    @staticmethod
    def _feature_rejected(
        state: _Connection,
        event: EventDescriptor,
        required_feature: str | None,
        fields: dict[str, object],
    ) -> bool:
        if event not in EVENT_FEATURES:
            # fail closed: an event the table does not name is never admitted.
            return True
        required = EVENT_FEATURES[event].features
        if required_feature is not None:
            required |= frozenset({required_feature})
        if not required <= history_features(state.features):
            return True
        return (
            event is EventDescriptor.APPROVAL_END
            and "approval_reconnect" not in state.features
            and isinstance(fields.get("data"), dict)
            and fields["data"].get("reason") == "grace_expired"
        )

    @staticmethod
    def _fields_for_connection(
        state: _Connection, event: EventDescriptor, fields: dict[str, object], *,
        feature_fields: event_contract.FeatureFields | None = None,
    ) -> dict[str, object]:
        shaped = dict(fields)
        for feature, overrides in (feature_fields or {}).items():
            if feature in state.features:
                shaped.update(overrides)
        for (field_event, name), required in OPTIONAL_FIELD_FEATURES.items():
            if field_event is event and name in shaped and not required <= history_features(
                state.features
            ):
                shaped.pop(name)
        return ConnectionRegistry._shape_event_fields(event, shaped, state.features)

    @staticmethod
    def _shape_event_fields(
        event: EventDescriptor, fields: dict[str, object], features: frozenset[str],
    ) -> dict[str, object]:
        from ..core.child_runs import child_run_for_features

        if event in {EventDescriptor.CHILD_STARTED, EventDescriptor.CHILD_PROGRESS, EventDescriptor.CHILD_ENDED}:
            child = fields.get("child")
            if isinstance(child, dict):
                row = child_run_for_features(child, child_recovery="child_recovery" in features)
                if row is not child:
                    fields = {**fields, "child": row}
        if event is EventDescriptor.TOOL_START and "tool_approval_origin" not in features:
            tool_call = fields.get("tool_call")
            if isinstance(tool_call, dict) and "approval_origin" in tool_call:
                fields["tool_call"] = {key: value for key, value in tool_call.items() if key != "approval_origin"}
        if event is EventDescriptor.APPROVAL_END:
            return event_contract.approval_end_for_features(
                fields, include_restart_reason="approval_restart_reason" in features,
            )
        return fields

    def publish_session(
        self,
        owner: Any | None,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None = None,
        replay_row: RowFacts | None = None,
        feature_fields: event_contract.FeatureFields | None = None,
        replay_scope: ReplayScopeId | None = None,
        covers: int | None = None,
        legacy_observers: bool = False,
        wait_for_write: bool = False,
        finalizer: bool = False,
        **fields: object,
    ) -> Admission:
        """Admit once per target. Only the owner may receive a plain frame."""
        if self._closed and not finalizer:
            return Admission("not_writable")
        route = EVENT_FEATURES.get(event)
        if route is None or route.destination != "session":
            return Admission("feature_rejected")
        coverage = (self._replay_covers(event, session_id, fields, replay_row, required_feature)
                    if replay_scope is None else covers)
        state = self._connections.get(owner)
        if not isinstance(coverage, DeferredFrame):
            self._publish_replay_subscribers(
                event, session_id, required_feature, fields, coverage, replay_row,
                exclude=owner, scope=replay_scope, feature_fields=feature_fields,
                legacy_observers=legacy_observers and route.legacy_observers,
            )
        outcome, frame, output = self._event_preflight(
            state, event, session_id, required_feature=required_feature,
            fields=fields, feature_fields=feature_fields,
        )
        if state is None or frame is None or output is None:
            return outcome
        if isinstance(coverage, DeferredFrame):
            if self.replay.scope_for(owner) == ReplayScopeId.top(coverage.session_id):
                return outcome
        elif self._publish_replay(
            state, event, session_id,
            self._fields_for_connection(state, event, fields, feature_fields=feature_fields),
            coverage, scope=replay_scope, replay_row=replay_row,
        ):
            return outcome
        return self._publish_local(state, output, frame, wait_for_write=wait_for_write)

    def _publish_local(
        self, state: _Connection, output: _OutputState, frame: _Frame, *, wait_for_write: bool
    ) -> Admission:
        if wait_for_write and output.phase == "ready":
            return Admission("admitted", state.connection, self._write_event(state, output, frame))
        if not self._admit(state, frame):
            return Admission("capacity_rejected")
        self._enqueue(output, _OutputFrame(frame), wake=self._phase_allows_writes(output))
        return Admission("admitted", state.connection)

    async def _write_event(self, state: _Connection, output: _OutputState, frame: _Frame) -> bool:
        if output.phase != "ready" or state.paused or not self._admit(state, frame):
            return False
        return await self._send_frame(output, frame)

    def publish_connection(
        self, connection: Any, event: EventDescriptor, session_id: str | None,
        *, wait_for_write: bool = False, **fields: object,
    ) -> Admission:
        if self._closed:
            return Admission("not_writable")
        state = self._connections.get(connection)
        outcome, frame, output = self._event_preflight(
            state, event, session_id, required_feature=None, fields=fields,
        )
        if state is None or frame is None or output is None:
            return outcome
        return self._publish_local(state, output, frame, wait_for_write=wait_for_write)

    async def publish_global(
        self, event: EventDescriptor, *, required_feature: str, data: dict[str, object]
    ) -> None:
        for state in tuple(self._connections.values()):
            if required_feature in state.features:
                await self.publish_connection(
                    state.connection, event, None, wait_for_write=state.role == "controller", data=data,
                ).delivered()

    def release(self, connection: Any) -> None:
        """Release physical role/output state, retaining the client until close."""
        self.invalidate_replay_connection(connection)
        state = self._connections.get(connection)
        if state is None or state.role is None:
            return
        state.released.set()
        state.paused = True
        if state in self._observers:
            self._observers.remove(state)
        output = state.output
        if output is not None:
            self._release_output(output, state.role)
        state.role = None

    def unregister(self, connection: Any) -> None:
        """Remove a physically closed client and settle its output writer."""
        state = self._connections.pop(connection, None)
        output = None if state is None else state.output
        if output is None:
            return
        self._settle_output(output)
        writer = output.writer_task
        if writer is not None and not writer.done():
            self._retired_writers.add(writer)
            writer.add_done_callback(self._retired_writers.discard)

    async def send(self, connection: Any, frame: _Frame) -> bool:
        state = self._connections.get(connection)
        if state is None or state.output is None:
            return False
        output = state.output
        if output.phase == "claimed_buffering":
            return await self._send_hello(state, frame)
        if output.phase != "ready" or not self._admit(state, frame):
            return False
        return await self._send_frame(output, frame)

    async def _send_hello(self, state: _Connection, frame: _Frame) -> bool:
        output = state.output
        assert output is not None
        if (
            output.overflow_frame is not None
            and not output.overflowed
            and not self._fits(output, len(frame))
        ):
            self._begin_overflow(state, enqueue_control=False)
        hello = _OutputFrame(frame)
        output.queue.appendleft(hello)
        self._charge(output, hello)
        if output.overflowed:
            output.phase = "overflowing"
            self._enqueue_overflow_control(output)
        else:
            output.phase = "flushing"
        output.wake.set()
        assert output.ready is not None
        self._begin_caller(output)
        try:
            return await asyncio.shield(output.ready)
        except asyncio.CancelledError:
            _, cancelled = await self._cancel_frame(output, hello)
            if cancelled is not None:
                raise cancelled
            if output.ready.done() and not output.ready.cancelled():
                return output.ready.result()
            raise
        finally:
            self._finish_caller(output)

    async def _send_frame(self, output: _OutputState, frame: _Frame) -> bool:
        future = self.loop.create_future()
        item = _OutputFrame(frame, future)
        self._enqueue(output, item)
        self._begin_caller(output)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            result, cancelled = await self._cancel_frame(output, item)
            if cancelled is not None:
                raise cancelled
            if result is not None:
                return result
            if future.done() and not future.cancelled():
                return future.result()
            raise
        finally:
            self._finish_caller(output)

    async def flush(
        self,
        connection: Any,
        timeout: float = CONNECTION_FLUSH_DEADLINE_SECONDS,
    ) -> None:
        state = self._connections.get(connection)
        if state is None or state.output is None:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(state.output.drained.wait(), timeout)

    async def wait_until_settled(self, connection: Any) -> None:
        state = self._connections.get(connection)
        if state is not None and state.output is not None:
            await state.output.callers_settled.wait()

    @staticmethod
    def _begin_caller(output: _OutputState) -> None:
        output.pending_callers += 1
        output.callers_settled.clear()

    def _finish_caller(self, output: _OutputState) -> None:
        self.loop.call_later(0, self._settle_caller, output)

    @staticmethod
    def _settle_caller(output: _OutputState) -> None:
        output.pending_callers -= 1
        if output.pending_callers == 0:
            output.callers_settled.set()

    @staticmethod
    def _phase_allows_writes(output: _OutputState) -> bool:
        return output.phase in {"flushing", "ready"}

    def _enqueue(
        self, output: _OutputState, frame: _OutputFrame, *, wake: bool = True
    ) -> None:
        output.drained.clear()
        output.queue.append(frame)
        self._charge(output, frame)
        if wake:
            output.wake.set()

    async def _output_writer(self, state: _Connection) -> None:
        output = state.output
        assert output is not None
        item: _OutputFrame | None = None
        try:
            while True:
                await output.wake.wait()
                while output.queue:
                    item = output.queue.popleft()
                    await self._write_output_frame(state, output, item)
                    item = None
                output.drained.set()
                output.wake.clear()
                if await self._complete_output_phase(state, output):
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            # a dead writer strands every caller and the controller slot: fail closed
            logger.exception(
                "serve output writer failed; closing the connection (role=%s phase=%s)",
                state.role,
                output.phase,
            )
            self._fail_dead_writer(state, output, item)

    def _fail_dead_writer(
        self, state: _Connection, output: _OutputState, item: _OutputFrame | None
    ) -> None:
        # the run loop then reads EOF and releases the connection and its role
        state.connection.writer.close()
        output.phase = "closed"
        # fail every waiter before any accounting step, which could raise again
        for frame in (item, *output.queue):
            if frame is not None:
                self._finish_future(frame.future, False)
        self._finish_future(output.ready, False)
        if output.current_frame is not None:
            # the writer failed before this frame left it: it is still charged
            self._end_frame_write(output, output.current_frame)
        output.current_write = None
        # queued frames still hold their replay budget; release it with them
        self._clear_output(output)

    async def _write_output_frame(
        self, state: _Connection, output: _OutputState, item: _OutputFrame
    ) -> None:
        output.current_frame = item
        try:
            output.current_write = asyncio.create_task(
                self._write_physical(state.connection, item.payload)
            )
            result, _ = await output.current_write
        except asyncio.CancelledError:
            self._end_frame_write(output, item)
            self._fail_replay_frame(state, item)
            if not output.released:
                self._finish_future(item.future, False)
            if asyncio.current_task().cancelling():
                self._finish_future(output.ready, False)
                raise
            if output.phase == "overflowing":
                return
            if output.phase == "flushing":
                self._reset_flush(output, False)
            return
        except Exception as exc:  # noqa: BLE001 - preserve write failure for its caller
            self._end_frame_write(output, item)
            self._fail_replay_frame(state, item)
            self._finish_future(item.future, error=exc)
            if output.phase == "flushing":
                self._reset_flush(output, error=exc)
            return
        self._end_frame_write(output, item)
        self._finish_future(item.future, result)
        if not result:
            self._fail_replay_frame(state, item)
        if not result and output.phase == "flushing":
            self._reset_flush(output, False)

    async def _complete_output_phase(
        self, state: _Connection, output: _OutputState
    ) -> bool:
        if output.released or output.phase == "closed":
            return True
        if output.phase == "flushing":
            output.phase = "ready"
            output.overflowed = False
            self._assert_retained_accounting(output)
            assert output.retained_bytes == 0
            assert output.retained_frames == 0
            self._finish_future(output.ready, True)
            return False
        if output.phase != "overflowing":
            return False
        self._cancel_overflow_deadline(output)
        await self._close_overflow_connection(state)
        self._finish_future(output.ready, False)
        return True

    def _reset_flush(
        self,
        output: _OutputState,
        result: bool | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self._clear_output(output)
        self._finish_future(output.ready, result, error=error)
        output.ready = self.loop.create_future()
        output.phase = "claimed_buffering"

    def _fits(self, output: _OutputState, size: int, frames: int = 1) -> bool:
        control = output.overflow_frame
        assert control is not None
        return (
            output.retained_bytes + size + len(control) <= OUTPUT_MAX_BYTES
            and output.retained_frames + frames + 1 <= OUTPUT_MAX_FRAMES
        )

    def _admit(self, state: _Connection, frame: _Frame) -> bool:
        output = state.output
        assert output is not None
        if output.overflowed or output.phase == "closed":
            return False
        if self._fits(output, len(frame)):
            return True
        self._begin_overflow(state, enqueue_control=output.phase != "claimed_buffering")
        return False

    def _begin_overflow(self, state: _Connection, *, enqueue_control: bool) -> None:
        output = state.output
        assert output is not None
        if output.overflowed:
            return
        output.overflowed = True
        self._clear_output(output, result=False)
        if enqueue_control:
            output.phase = "overflowing"
            self._enqueue_overflow_control(output)
        output.overflow_deadline_task = asyncio.create_task(
            self._close_after_overflow_deadline(state)
        )

    def _enqueue_overflow_control(self, output: _OutputState) -> None:
        control = output.overflow_frame
        assert control is not None
        self._enqueue(output, _OutputFrame(control))
        self._assert_retained_accounting(output)
        assert output.retained_bytes <= OUTPUT_MAX_BYTES
        assert output.retained_frames <= OUTPUT_MAX_FRAMES

    async def _close_after_overflow_deadline(self, state: _Connection) -> None:
        await _overflow_deadline_wait()
        output = state.output
        if output is None or output.phase != "overflowing":
            return
        await self._close_overflow(state)

    async def _close_overflow(self, state: _Connection) -> None:
        output = state.output
        assert output is not None
        self._clear_output(output)
        writer = output.writer_task
        if writer is not None:
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
        await self._close_overflow_connection(state)
        self._finish_future(output.ready, False)

    async def _close_overflow_connection(self, state: _Connection) -> None:
        if state.role == "controller" and "multi_controller" in state.features:
            # The connection runner settles its active request before releasing
            # a semantic slot, even when overflow originates in this writer.
            state.connection.stop_transport()
        else:
            await state.connection.close()

    @staticmethod
    def _overflow_frame(role: str) -> _Frame:
        descriptor = (
            ErrorDescriptor.CONTROLLER_OUTPUT_OVERFLOW
            if role == "controller"
            else ErrorDescriptor.OBSERVER_OUTPUT_OVERFLOW
        )
        return FrameCodec().error_response(None, descriptor, descriptor.meaning)

    @staticmethod
    async def _write_physical(
        connection: Any, payload: _Frame
    ) -> tuple[bool, FrameWriteCancelled | None]:
        try:
            return await connection._write_physical(payload), None
        except FrameWriteCancelled as cancelled:
            return cancelled.written, cancelled

    @staticmethod
    def _finish_future(
        future: asyncio.Future[bool] | None,
        result: bool | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        if future is None or future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            assert result is not None
            future.set_result(result)

    @staticmethod
    def _charge(output: _OutputState, item: _OutputFrame) -> None:
        output.retained_bytes += len(item.payload)
        output.retained_frames += 1

    def _end_frame_write(self, output: _OutputState, item: _OutputFrame) -> None:
        """The frame left the writer: it is no longer in flight, then uncharge it."""
        output.current_frame = None
        output.current_write = None
        self._finish_frame_accounting(output, item)

    def _finish_frame_accounting(
        self, output: _OutputState, item: _OutputFrame
    ) -> None:
        output.retained_bytes -= len(item.payload)
        output.retained_frames -= 1
        assert output.retained_bytes >= 0
        assert output.retained_frames >= 0
        if item.replay_scope is not None and item.replay_size:
            self.replay.release(item.replay_scope, item.replay_size)

    @staticmethod
    def _assert_retained_accounting(output: _OutputState) -> None:
        frames = list(output.queue)
        if output.current_frame is not None:
            frames.append(output.current_frame)
        assert output.retained_bytes == sum(len(item.payload) for item in frames)
        assert output.retained_frames == len(frames)

    async def _cancel_frame(
        self, output: _OutputState, item: _OutputFrame
    ) -> tuple[bool | None, FrameWriteCancelled | None]:
        if output.current_frame is item:
            if output.current_write is not None:
                output.current_write.cancel()
                settled = await asyncio.gather(
                    output.current_write, return_exceptions=True
                )
                if isinstance(settled[0], tuple):
                    return settled[0]
            return None, None
        try:
            output.queue.remove(item)
        except ValueError:
            return None, None
        self._finish_frame_accounting(output, item)
        if item.future is not None and not item.future.done():
            item.future.cancel()
        return None, None

    def _release_output(self, output: _OutputState, role: str | None) -> None:
        output.released = True
        self._cancel_overflow_deadline(output)
        if role == "observer":
            self._clear_output(output)
            if (
                output.writer_task is not None
                and output.writer_task is not asyncio.current_task()
            ):
                output.writer_task.cancel()
            return
        self._clear_output(output, result=False)
        writer_task = output.writer_task
        current_task = asyncio.current_task()
        if (
            writer_task is None
            or writer_task is current_task
            or output.current_write is current_task
        ):
            return
        if output.current_frame is not None:
            self._finish_future(output.current_frame.future, False)
        writer_task.cancel()

    def _settle_output(self, output: _OutputState) -> None:
        # A physical write can consume cancellation while reporting its outcome.
        # Close admission before joining the writer, and don't let it wait again.
        output.phase = "closed"
        self._clear_output(output, result=False)
        # an overflow close runs inside the writer; it returns on the closed phase
        if output.writer_task is not None and output.writer_task is not asyncio.current_task():
            output.writer_task.cancel()
        self._cancel_overflow_deadline(output)
        self._finish_future(output.ready, False)

    @staticmethod
    def _cancel_overflow_deadline(output: _OutputState) -> None:
        task = output.overflow_deadline_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        output.overflow_deadline_task = None

    def _clear_output(self, output: _OutputState, *, result: bool | None = None) -> None:
        while output.queue:
            item = output.queue.popleft()
            self._finish_frame_accounting(output, item)
            if item.future is None or item.future.done():
                continue
            if result is None:
                item.future.cancel()
            else:
                item.future.set_result(result)

    async def run_blocking(self, callable: Callable[[], Any]) -> Any:
        """Run one store or observer read away from the event loop."""
        if self._closed:
            raise RuntimeError("reads are closed")
        return await self._reads.run(callable)

    async def run_read(
        self,
        connection: Any,
        callable: Callable[[], Any],
        use: SessionUse | None = None,
        *,
        fence: bool = True,
    ) -> Any:
        """Run one read off-loop. ``use`` passes to the worker's completion, so
        its record cannot close under a read whose waiter was cancelled; with
        ``fence``, fail if that record no longer holds the session it held
        before the read. A replay view also has to remain selected."""
        if self._closed or connection not in self._connections:
            raise RuntimeError("connection is not connected")
        if use is None:
            return await self.run_blocking(callable)
        runtime = use.runtime
        session_id = None if runtime.opened is None else runtime.session_id
        handle = self._reads.submit(callable)
        use.transfer_to(handle.completed)
        result = await handle.result
        if fence or use.is_view:
            connection.server.controllers.require_current(connection, use, session_id)
        return result

    async def read_tree(
        self, connection: Any, params: dict[str, Any], controller: SessionUse | None = None
    ) -> object:
        """``controller`` is the request's owned use after a mutation; else select."""
        if controller is None:
            with connection.server.controllers.selected(
                connection, validation.required_string(params, "session_id")
            ) as selected:
                return await self.read_tree(connection, params, selected)
        store = validation.active(controller.runtime, params)
        return await self.run_read(
            connection, lambda: session.tree(_observer_runtime(store)), controller
        )

    async def read_history(self, connection: Any, params: dict[str, Any]) -> object:
        with connection.server.controllers.selected(
            connection, validation.required_string(params, "session_id")
        ) as selected:
            store = validation.active(selected.runtime, params)
            features = history_features(self._connections[connection].features)
            result = await self.run_read(
                connection,
                lambda: history.history(
                    _observer_runtime(store),
                    params,
                    include_delivery_id={"delivery_id", "user_message_event", "user_message_delivery_id"} <= features,
                    backward_enabled="session_history_backward" in features,
                    fidelity="session_history_fidelity" in features,
                    content_read="content_read" in features,
                    served_turn_end="served_turn_end" in features,
                    approval_origin="tool_approval_origin" in features,
                ),
                selected,
            )
        return self._shape_history_for_features(connection, result)

    async def read_content(
        self, connection: Any, params: dict[str, Any], controller: SessionUse
    ) -> object:
        if set(params) - {"handle", "offset", "limit"}:
            raise ProtocolError(ErrorDescriptor.INVALID_CONTENT_READ, "invalid content read")
        runtime = controller.runtime
        if runtime.opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        return await self.run_read(
            connection,
            lambda: content.read(
                runtime.opened.store,
                params.get("handle"),
                offset=params.get("offset", 0),
                limit=params.get("limit", content.MAX_CONTENT_READ),
            ),
            controller,
        )

    async def read_catalog(self, connection: Any, controller: SessionUse) -> object:
        runtime = controller.runtime
        return await self.run_read(
            connection, lambda: settings.catalog(runtime), controller, fence=False
        )

    async def read_settings(
        self, connection: Any, params: dict[str, Any], controller: SessionUse
    ) -> object:
        runtime = controller.runtime
        validation.active(runtime, params)
        telemetry = settings.telemetry(connection)
        return await self.run_read(
            connection, lambda: settings.settings(runtime, telemetry=telemetry), controller, fence=False
        )

    async def read_child_run_history(
        self, connection: Any, params: dict[str, Any]
    ) -> object:
        if (
            set(params) - {"session_id", "run_id", "before", "history_validator"}
            or type(params.get("session_id")) is not str
            or type(params.get("run_id")) is not str
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "session_id and run_id are required child history parameters",
            )
        with connection.server.controllers.selected(connection, params["session_id"]) as selected:
            runtime = selected.runtime
            if runtime.opened is None or params["session_id"] != runtime.session_id:
                raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
            features = self._connections[connection].features
            target = (runtime.session_id, runtime.opened.store)
            result = await self.run_read(
                connection,
                lambda: perform_child_history_read(
                    target,
                    params,
                    registry=self,
                    fidelity="session_history_fidelity" in features,
                    approval_origin="tool_approval_origin" in features,
                ),
                selected,
            )
        return self._shape_history_for_features(connection, result)

    async def read_child_run_events(
        self, connection: Any, params: dict[str, Any], controller: SessionUse
    ):
        # params are validated by the child_run_events handler before begin_replay_join
        runtime = controller.runtime
        if runtime.opened is None or params["session_id"] != runtime.session_id:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        features = self._connections[connection].features
        target = (runtime.session_id, runtime.opened.store)
        def read():
            result = perform_child_history_read(
                target,
                params,
                registry=self,
                fidelity="session_history_fidelity" in features,
                approval_origin="tool_approval_origin" in features,
            )
            from ..core.agent_state import LoadedAgentLifecycle
            from ..core.child_runs import project_child_run_row, read_child_lifecycle
            reader = self._child_reader_for(target[1], params["run_id"])
            lifecycle, _ = read_child_lifecycle(target[1], reader.session_dir.absolute())
            item = lifecycle.value if isinstance(lifecycle, LoadedAgentLifecycle) else None
            if item is not None:
                item = {"handle": item.get("handle"), "description": item.get("description", ""), "agent_type": item.get("agent_type", "general"), "state": item.get("state", "running"), "turns_used": item.get("turns_used", 0), "tool_calls": item.get("tool_calls", 0), "current_step": item.get("current_step", ""), "started_at": item.get("started_at"), "finished_at": item.get("finished_at"), "elapsed": item.get("elapsed", 0.0), "account_status": item.get("account_status")}
            row = None
            if item is not None:
                session_id, parent_run_id = run_row_identity(params["run_id"])
                row = project_child_run_row(item, session_id=session_id, parent_run_id=parent_run_id, accepts_follow_ups=False)
            if row is None:
                raise ProtocolError(ErrorDescriptor.CHILD_RUN_NOT_FOUND, "child run was not found")
            return result, row
        result = await self.run_read(connection, read, controller)
        return result[0], 0, result[1]

    def _child_reader_for(
        self, root_store: ConversationStore, run_id: str, *, allow_stale: bool = False
    ) -> ConversationStore:
        parse_child_run_id(root_store.session_id, run_id)
        segments = child_run_segments(root_store.session_id, run_id)
        key = (root_store.session_id, run_id)
        with self._child_reader_lock:
            if self._child_reader is not None and self._child_reader[0] == key:
                reader = self._child_reader[1]
                if not allow_stale:
                    reader.refresh()
                return reader
            if self._child_reader is not None:
                self._child_reader[1].close()
            with ExitStack() as cleanup:
                directory_fd = root_store.directory_fd
                components = tuple(
                    component
                    for segment in segments
                    for component in ("agents", str(segment))
                )
                directory_fd = cleanup.enter_context(
                    descend_directory(directory_fd, components)
                )
                try:
                    verify_child_run_directory(directory_fd, run_id)
                except (FileNotFoundError, OSError, ValueError) as exc:
                    raise ProtocolError(
                        ErrorDescriptor.CHILD_RUN_NOT_FOUND,
                        "child run was not found",
                    ) from exc
            parent = root_store.session_dir
            for segment in segments[:-1]:
                parent = parent / "agents" / str(segment)
            try:
                reader = ConversationStore(
                    parent / "agents",
                    session_id=str(segments[-1]),
                    _read_only=True,
                    _observer_snapshot=True,
                    _must_exist=True,
                    _storage_lease=root_store._storage_lease,
                )
            except TranscriptLoadError as exc:
                exc.session_id = root_store.session_id
                raise
            reader._child_history_reader = True
            self._child_reader = (key, reader)
            return reader

    def close_child_reader(self) -> None:
        with self._child_reader_lock:
            if self._child_reader is not None:
                self._child_reader[1].close()
                self._child_reader = None

    # merge-resolution: src/zeta/server/connections.py pr-lines ConnectionRegistry.read_status@9bb4d2bb8ef9 -> keep the shared status method after adding child history reads
    async def read_status(
        self, connection: Any, status: dict[str, Any], controller: SessionUse
    ) -> object:
        features = self._connections[connection].features
        return await self.run_read(
            connection,
            lambda: project_status(status, served_turn_end="served_turn_end" in features),
            controller,
        )

    def stop_admission(self) -> None:
        """Freeze new peers, requests and ordinary publications, not finalizers."""
        self._closed = True

    async def stop_producers(self) -> None:
        self._unregister_child_run_sink()

    async def stop_reads(self) -> None:
        """Abandon read results; transferred uses still track worker completion."""
        try:
            self.state_reader.close()
        finally:
            self._reads.close()

    async def drain_record_events(self, record: SessionController) -> None:
        """Settle replay output while its record is still readable."""
        async def drain(client: Any) -> None:
            try:
                await self.flush(client)
            finally:
                self.invalidate_replay_connection(client)

        clients = tuple(client for client in self.clients() if self.replay.controller_for(client) is record)
        results = await asyncio.gather(*(drain(client) for client in clients), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def stop_workers(self, clients: tuple[Any, ...]) -> None:
        """Drain the frozen peers without owning their physical lifetime."""
        first: BaseException | None = None

        async def attempt(operation: Awaitable[object]) -> None:
            nonlocal first
            try:
                await operation
            except BaseException as exc:  # noqa: BLE001
                if first is None:
                    first = exc

        async def settle_output(client: Any) -> None:
            if (state := self._connections.get(client)) is not None and state.output is not None:
                self._settle_output(state.output)

        await attempt(asyncio.to_thread(self.close_child_reader))
        writers = (*self._retired_writers, *(
            output.writer_task for client in clients
            if (output := self.output_state(client)) is not None and output.writer_task is not None
        ))
        for client in clients:
            await attempt(settle_output(client))
        await asyncio.gather(*writers, return_exceptions=True)
        for writer in writers:
            if not writer.cancelled():
                await attempt(writer)
        self.clear_replay_scopes()
        if first is not None:
            raise first

    async def aclose(self) -> None:
        self.stop_admission()
        try:
            await self.stop_producers()
        finally:
            try:
                await self.stop_reads()
            finally:
                await self.stop_workers(self.clients())


def perform_child_history_read(
    target: tuple[str | None, object],
    params: dict[str, Any],
    *,
    registry: ConnectionRegistry,
    fidelity: bool,
    approval_origin: bool,
) -> object:
    root_store = target[1]
    try:
        parse_child_run_id(root_store.session_id, params["run_id"])
    except ValueError as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "invalid child run id") from exc
    history_params = {key: value for key, value in params.items() if key != "run_id"}
    history_params.setdefault("before", None)

    def produce(final: bool) -> object:
        try:
            return registry._child_reader_for(
                root_store, params["run_id"], allow_stale=final
            )
        except (FileNotFoundError, OSError) as exc:
            raise ProtocolError(
                ErrorDescriptor.CHILD_RUN_NOT_FOUND, "child run was not found"
            ) from exc

    try:
        return history.child_history(
            produce,
            history_params,
            fidelity=fidelity,
            approval_origin=approval_origin,
        )
    except ObserverSnapshotChanged as exc:
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "child history source changed during read",
        ) from exc


def project_status(target: object, *, served_turn_end: bool = False) -> dict[str, Any]:
    status = dict(target)
    store = status.pop("store", None)
    status["compaction_markers"] = (
        store.compaction_marker_count() if store is not None else 0
    )
    if served_turn_end:
        status["last_turn_outcome"] = last_turn_outcome(store)
    return status


def last_turn_outcome(store: Any) -> dict[str, object] | None:
    """The last durable whole-turn outcome on the active branch, or None."""
    entry = store.last_turn_end() if store is not None else None
    if entry is None:
        return None
    return {"turn_id": entry.data["turn_id"], "reason": entry.data["reason"]}



def _observer_runtime(store: object) -> object:
    return type("ObserverRuntime", (), {"opened": type("Opened", (), {"store": store})()})()
