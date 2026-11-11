from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.agent_progress import register_child_run_sink
from ..core.session_files import descend_directory
from ..core.store import ConversationStore, ObserverSnapshotChanged, TranscriptLoadError
from ..core.store._child_runs import (
    child_run_segments,
    parse_child_run_id,
    verify_child_run_directory,
)
from ..core.thread_call import ThreadCallWorkers
from . import content
from .handlers import history, lifecycle, session, settings, validation
from .protocol import (
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    FrameWriteCancelled,
    ProtocolError,
    _Frame,
)
from .session_replay import (
    DeferredFrame,
    ReplayScopeId,
    SessionReplay,
    history_features,
)
from .session_replay.children import ChildRunProjection
from .session_replay.output import ReplayOutput, _OutputFrame
from .session_replay.rows import RowFacts

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
MAX_CONNECTIONS = 10  # controller, eight observers, and one handshake
OUTPUT_MAX_BYTES = 4 * 1024 * 1024
OUTPUT_MAX_FRAMES = 256
OUTPUT_OVERFLOW_CLOSE_DEADLINE = 1.0

# one owner for feature gates and optional wire fields on every output path
NO_FEATURE = frozenset()
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
    EventDescriptor.CHILD_HOLD: NO_FEATURE,
    EventDescriptor.TASK_EXIT_NOTIFICATION: NO_FEATURE,
    EventDescriptor.PROJECT_INBOX_NOTIFICATION: NO_FEATURE,
    EventDescriptor.CHILD_QUESTION: NO_FEATURE,
    EventDescriptor.CHILD_STARTED: frozenset({"child_run_events"}),
    EventDescriptor.CHILD_PROGRESS: frozenset({"child_run_events"}),
    EventDescriptor.CHILD_ENDED: frozenset({"child_run_events"}),
    EventDescriptor.CHILD_QUESTION_WITHDRAWN: NO_FEATURE,
    EventDescriptor.CHILD_MESSAGE_DELIVERED: NO_FEATURE,
    EventDescriptor.ACCOUNT_LIMIT: NO_FEATURE,
    EventDescriptor.ACCOUNT_STATE_UNKNOWN: NO_FEATURE,
    EventDescriptor.TURN_END: NO_FEATURE,
    EventDescriptor.SERVED_TURN_END: NO_FEATURE,
    EventDescriptor.AGENT_END: NO_FEATURE,
    EventDescriptor.ERROR: NO_FEATURE,
    EventDescriptor.DELIVERY_STATUS: frozenset({"session_history_fidelity"}),
    EventDescriptor.MEMORY_UPDATED: frozenset({"memory_updated"}),
    EventDescriptor.TURN_ABORTED: NO_FEATURE,
    EventDescriptor.USAGE: NO_FEATURE,
    EventDescriptor.USER_MESSAGE: frozenset({"user_message_event"}),
    EventDescriptor.SESSION_GRANTS_CHANGED: frozenset({"session_grants"}),
    EventDescriptor.REPLAY_STREAM_ENDED: NO_FEATURE,
    EventDescriptor.CONFIG_CHANGED: frozenset({"live_config_changed"}),
    EventDescriptor.BACKGROUND_TASK_UPDATED: frozenset({"run_work"}),
    EventDescriptor.TODO_UPDATED: frozenset({"run_work"}),
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


@dataclass(frozen=True)
class ControllerMetadata:
    connection: Any
    protocol_version: str
    features: frozenset[str]


@dataclass
class _OutputState:
    phase: Literal["claimed_buffering", "flushing", "ready", "overflowing"] = (
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


class ConnectionRegistry(ReplayOutput):
    """Owns connection claims, output, and bounded observer reads."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._connections: dict[Any, _Connection] = {}
        self._controller: _Connection | None = None
        self._controller_output_paused = False
        self._observers: list[_Connection] = []
        self._closed = False
        self.replay = SessionReplay()
        self.child_runs = ChildRunProjection(loop, self)
        self._unregister_child_run_sink = register_child_run_sink(self.child_runs.sink)
        self._child_reader: tuple[tuple[str, str], ConversationStore] | None = None
        self._child_reader_lock = threading.Lock()
        self._reads = ThreadCallWorkers(loop, workers=2, name="zeta-read")

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
            self._claim(connection, role, protocol_version, features)
        except RuntimeError as exc:
            if role == "observer":
                raise ProtocolError(
                    ErrorDescriptor.CONNECTION_CAPACITY, "observer limit reached"
                ) from exc
            raise ProtocolError(
                ErrorDescriptor.CONNECTION_CAPACITY,
                "another client is already connected",
            ) from exc
        connection.role = role
        return role

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
        if role == "controller":
            if self._controller is not None:
                raise RuntimeError("controller occupied")
            self._controller = state
        else:
            if len(self._observers) >= 8:
                raise RuntimeError("observer limit reached")
            self._observers.append(state)
        state.role = role
        assert state.output is not None
        state.output.overflow_frame = self._overflow_frame(role)
        state.protocol_version = protocol_version
        state.features = features

    def is_controller(self, connection: Any) -> bool:
        state = self._connections.get(connection)
        return state is not None and state is self._controller

    def activate_controller(self, connection: Any) -> None:
        if self.is_controller(connection):
            self._controller_output_paused = False

    def controller_metadata(self) -> ControllerMetadata | None:
        state = self._controller
        if state is None:
            return None
        return ControllerMetadata(
            state.connection, state.protocol_version, state.features
        )

    def _controller_preflight(
        self,
        state: _Connection | None,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None,
        fields: dict[str, object],
    ) -> tuple[Admission, _Frame | None, _OutputState | None]:
        if state is None:
            return Admission("no_controller"), None, None
        if state is self._controller and self._controller_output_paused:
            return Admission("not_writable"), None, None
        if self._feature_rejected(state, event, required_feature, fields):
            return Admission("feature_rejected"), None, None
        fields = self._fields_for_connection(state, event, fields)
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
        required = EVENT_FEATURES[event]
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
        state: _Connection, event: EventDescriptor, fields: dict[str, object]
    ) -> dict[str, object]:
        shaped = dict(fields)
        for (field_event, name), required in OPTIONAL_FIELD_FEATURES.items():
            if field_event is event and name in shaped and not required <= history_features(
                state.features
            ):
                shaped.pop(name)
        return shaped

    def try_admit_controller(
        self,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None = None,
        replay_row: RowFacts | None = None,
        **fields: object,
    ) -> Admission:
        """Admit one controller event without waiting for physical output."""
        covers = self._replay_covers(
            event, session_id, fields, replay_row, required_feature
        )
        if not isinstance(covers, DeferredFrame):
            self._publish_observer_replay(
                event, session_id, required_feature, fields, covers, replay_row
            )
        outcome, frame, output = self._controller_preflight(
            self._controller,
            event,
            session_id,
            required_feature=required_feature,
            fields=fields,
        )
        if outcome.kind != "admitted" or frame is None or output is None:
            return outcome
        assert self._controller is not None
        if self._replay_holds_copy(self._controller, covers):
            return outcome
        if not isinstance(covers, DeferredFrame) and self._publish_replay(
            self._controller,
            event,
            session_id,
            self._fields_for_connection(self._controller, event, fields),
            covers,
            replay_row=replay_row,
        ):
            return outcome
        if not self._admit(self._controller, frame):
            return Admission("capacity_rejected")
        self._enqueue(
            output,
            _OutputFrame(frame),
            wake=self._phase_allows_writes(output),
        )
        return outcome

    def _replay_holds_copy(self, state: _Connection, covers: object) -> bool:
        """True when a deferred record will send the controller's copy on its stream.

        A controller with no stream or join in the scope gets the frame plainly now.
        """
        return isinstance(covers, DeferredFrame) and self.replay.scope_for(
            state.connection
        ) == ReplayScopeId.top(covers.session_id)

    async def publish_controller(
        self,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None = None,
        replay_row: RowFacts | None = None,
        **fields: object,
    ) -> bool:
        covers = self._replay_covers(
            event, session_id, fields, replay_row, required_feature
        )
        if not isinstance(covers, DeferredFrame):
            self._publish_observer_replay(
                event, session_id, required_feature, fields, covers, replay_row
            )
        outcome, frame, output = self._controller_preflight(
            self._controller,
            event,
            session_id,
            required_feature=required_feature,
            fields=fields,
        )
        if outcome.kind != "admitted" or frame is None or output is None:
            return False
        state = self._controller
        assert state is not None
        if self._replay_holds_copy(state, covers):
            return True
        if not isinstance(covers, DeferredFrame) and self._publish_replay(
            state,
            event,
            session_id,
            self._fields_for_connection(state, event, fields),
            covers,
            replay_row=replay_row,
        ):
            return True
        if not self._admit(state, frame):
            return False
        if output.phase in {"claimed_buffering", "flushing"}:
            self._enqueue(
                output,
                _OutputFrame(frame),
                wake=self._phase_allows_writes(output),
            )
            return True
        return await self._send_frame(output, frame)

    async def publish_feature_event(
        self,
        feature: str,
        event: EventDescriptor,
        session_id: str,
        *,
        data: dict[str, object],
    ) -> None:
        """Queue one live event for every connection that negotiated a feature."""
        states = tuple(
            state
            for state in (self._controller, *self._observers)
            if state is not None
            and feature in state.features
            and (state is not self._controller or not self._controller_output_paused)
        )
        covers = self._replay_covers(event, session_id, {"data": data}, None)
        # feature events have no publish-first route
        assert not isinstance(covers, DeferredFrame)
        for state in states:
            frame = state.connection.codec.notification(event, session_id, data=data)
            if frame is None:
                continue
            if self._publish_replay(state, event, session_id, {"data": data}, covers):
                continue
            if state is self._controller:
                await self.send(state.connection, frame)
            else:
                self._publish_observer(state, frame)

    def release(self, connection: Any) -> None:
        self.invalidate_replay_connection(connection)
        state = self._connections.pop(connection, None)
        if state is None:
            return
        if state is self._controller:
            self._controller = None
            self._controller_output_paused = True
        if state in self._observers:
            self._observers.remove(state)
        output = state.output
        if output is not None:
            self._release_output(output, state.role)

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
        try:
            while True:
                await output.wake.wait()
                while output.queue:
                    await self._write_output_frame(
                        state, output, output.queue.popleft()
                    )
                output.drained.set()
                output.wake.clear()
                if await self._complete_output_phase(state, output):
                    return
        except asyncio.CancelledError:
            return

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
            self._finish_frame_accounting(output, item)
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
            self._finish_frame_accounting(output, item)
            self._fail_replay_frame(state, item)
            self._finish_future(item.future, error=exc)
            if output.phase == "flushing":
                self._reset_flush(output, error=exc)
            return
        finally:
            output.current_frame = None
            output.current_write = None
        self._finish_frame_accounting(output, item)
        self._finish_future(item.future, result)
        if not result:
            self._fail_replay_frame(state, item)
        if not result and output.phase == "flushing":
            self._reset_flush(output, False)

    async def _complete_output_phase(
        self, state: _Connection, output: _OutputState
    ) -> bool:
        if output.released:
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
        await state.connection.close()
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

    def _publish_observer(self, state: _Connection, frame: _Frame) -> None:
        output = state.output
        if (
            state.connection not in self._connections
            or output is None
            or output.phase != "ready"
            or not self._admit(state, frame)
        ):
            return
        self._enqueue(output, _OutputFrame(frame))

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
        if output.overflowed:
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
        await state.connection.close()
        self._finish_future(output.ready, False)

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
        while output.queue:
            item = output.queue.popleft()
            self._finish_frame_accounting(output, item)
            self._finish_future(item.future, False)
        if output.writer_task is not None:
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
        return await self._reads.run(callable)

    async def run_read(
        self,
        connection: Any,
        callable: Callable[[], Any],
        *,
        session_id: str | None = None,
    ) -> Any:
        if connection not in self._connections:
            raise RuntimeError("connection is not connected")
        result = await self.run_blocking(callable)
        if (
            session_id is not None
            and connection.server.runtime.session_id != session_id
        ):
            raise ProtocolError(
                ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active"
            )
        return result

    async def read_tree(self, connection: Any, params: dict[str, Any]) -> object:
        runtime = connection.server.runtime
        store = validation.active(runtime, params)
        return await self.run_read(
            connection,
            lambda: session.tree(_observer_runtime(store)),
            session_id=runtime.session_id,
        )

    async def read_history(self, connection: Any, params: dict[str, Any]) -> object:
        runtime = connection.server.runtime
        store = validation.active(runtime, params)
        features = history_features(self._connections[connection].features)
        return await self.run_read(
            connection,
            lambda: history.history(
                _observer_runtime(store),
                params,
                include_delivery_id={"delivery_id", "user_message_event", "user_message_delivery_id"} <= features,
                backward_enabled="session_history_backward" in features,
                fidelity="session_history_fidelity" in features,
                content_read="content_read" in features,
                served_turn_end="served_turn_end" in features,
            ),
            session_id=runtime.session_id,
        )

    async def read_content(self, connection: Any, params: dict[str, Any]) -> object:
        if set(params) - {"handle", "offset", "limit"}:
            raise ProtocolError(ErrorDescriptor.INVALID_CONTENT_READ, "invalid content read")
        runtime = connection.server.runtime
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
            session_id=runtime.session_id,
        )

    async def read_catalog(self, connection: Any) -> object:
        return await self.run_read(connection, lambda: settings.catalog(connection.server.runtime))

    async def read_settings(self, connection: Any, params: dict[str, Any]) -> object:
        runtime = connection.server.runtime
        if runtime.opened is None:
            raise ProtocolError(ErrorDescriptor.INTERNAL_ERROR, "server has no active session")
        return await self.run_read(connection, lambda: settings.settings(runtime))

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
        runtime = connection.server.runtime
        if runtime.opened is None or params["session_id"] != runtime.session_id:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        features = self._connections[connection].features
        target = (runtime.session_id, runtime.opened.store)
        return await self.run_read(
            connection,
            lambda: perform_child_history_read(
                target,
                params,
                registry=self,
                fidelity="session_history_fidelity" in features,
            ),
            session_id=runtime.session_id,
        )

    async def read_child_run_events(self, connection: Any, params: dict[str, Any]):
        if set(params) != {"session_id", "run_id"} or type(params.get("session_id")) is not str or type(params.get("run_id")) is not str:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "session_id and run_id are required child event parameters")
        runtime = connection.server.runtime
        if runtime.opened is None or params["session_id"] != runtime.session_id:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        features = self._connections[connection].features
        target = (runtime.session_id, runtime.opened.store)
        def read():
            result = perform_child_history_read(target, params, registry=self, fidelity="session_history_fidelity" in features)
            from ..core.agent_progress import read_agent_lifecycle
            from ..core.child_runs import project_child_run_row
            reader = self._child_reader_for(target[1], params["run_id"])
            lifecycle = read_agent_lifecycle(reader.session_dir / "agent_lifecycle.json")
            item = lifecycle if isinstance(lifecycle, dict) else None
            if item is not None:
                item = {"handle": item.get("handle"), "description": item.get("description", ""), "agent_type": item.get("agent_type", "general"), "state": item.get("state", "running"), "turns_used": item.get("turns_used", 0), "tool_calls": item.get("tool_calls", 0), "current_step": item.get("current_step", ""), "started_at": item.get("started_at"), "finished_at": item.get("finished_at"), "elapsed": item.get("elapsed", 0.0)}
            row = None
            if item is not None:
                _, parent_run_id = parse_child_run_id(target[0], params["run_id"])
                row = project_child_run_row(item, session_id=target[0], parent_run_id=parent_run_id, accepts_follow_ups=False)
            if row is None:
                raise ProtocolError(ErrorDescriptor.CHILD_RUN_NOT_FOUND, "child run was not found")
            return result, row
        result = await self.run_read(connection, read, session_id=runtime.session_id)
        return result[0], 0, result[1]

    def _child_reader_for(
        self, root_store: ConversationStore, run_id: str
    ) -> ConversationStore:
        parse_child_run_id(root_store.session_id, run_id)
        segments = child_run_segments(root_store.session_id, run_id)
        key = (root_store.session_id, run_id)
        with self._child_reader_lock:
            if self._child_reader is not None and self._child_reader[0] == key:
                reader = self._child_reader[1]
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
        self, connection: Any, status: dict[str, Any], session_id: str | None
    ) -> object:
        features = self._connections[connection].features
        return await self.run_read(
            connection,
            lambda: project_status(status, served_turn_end="served_turn_end" in features),
            session_id=session_id,
        )

    async def aclose(self) -> None:
        self._closed = True
        self._unregister_child_run_sink()
        states = tuple(self._connections.values())
        await asyncio.to_thread(self.close_child_reader)
        self._reads.close()
        for state in states:
            if state.output is not None:
                self._settle_output(state.output)
        await asyncio.gather(
            *(state.connection.close() for state in states), return_exceptions=True
        )
        self._connections.clear()
        self._observers.clear()
        self._controller = None


def _active_status_parts(
    client: Any, loop: Any, fallback_config: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = client.server.runtime
    transcript_index_failure = (
        loop.transcript_index_failure(runtime.session_id)
        if client.protocol_version != "1.0" and loop is not None
        else None
    )
    project_context_health = (
        (
            loop.project_context_health.summary
            if loop is not None and loop.project_context_health is not None
            else "healthy"
        )
        if client.protocol_version != "1.0"
        else None
    )
    status = {
        "session": lifecycle.session_snapshot(client),
        "state": (
            "idle"
            if client.server.control.unhealthy() or runtime.state is None
            else runtime.state.status
        ),
        "explicit_finish": (
            loop.explicit_finish
            if loop is not None
            else fallback_config.agents_explicit_finish
        ),
    }
    suffix = {
        "pending_approvals": [],
        "usage": dict(runtime.usage),
        **(
            {"transcript_index_failure": transcript_index_failure}
            if transcript_index_failure is not None
            else {}
        ),
        **(
            {"project_context_health": project_context_health}
            if project_context_health is not None
            else {}
        ),
        "store": runtime.opened.store,
    }
    slash_command = client.server.control.slash_lock.status()
    if slash_command is not None:
        suffix["slash_command"] = slash_command
    return status, suffix


def _capture_status(client: Any, fallback_config: Any) -> tuple[str | None, object]:
    runtime = client.server.runtime
    loop = runtime.loop
    if loop is None and fallback_config is None:
        raise RuntimeError("status fallback config was not resolved")
    if runtime.opened is None:
        status = {"session": None, "state": "idle", "explicit_finish": False}
        suffix = {
            "pending_approvals": [],
            "usage": {},
            "compaction_markers": 0,
        }
        session_id = None
    else:
        status, suffix = _active_status_parts(client, loop, fallback_config)
        session_id = runtime.session_id
    if client.protocol_version != "1.0":
        status["max_agent_depth"] = (
            loop.max_agent_depth
            if loop is not None
            else fallback_config.agents_max_depth
        )
    status.update(suffix)
    return session_id, status

def perform_child_history_read(
    target: tuple[str | None, object],
    params: dict[str, Any],
    *,
    registry: ConnectionRegistry,
    fidelity: bool,
) -> object:
    root_store = target[1]
    try:
        parse_child_run_id(root_store.session_id, params["run_id"])
    except ValueError as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "invalid child run id") from exc
    try:
        reader = registry._child_reader_for(root_store, params["run_id"])
    except ObserverSnapshotChanged as exc:
        raise ProtocolError(
            ErrorDescriptor.HISTORY_SNAPSHOT_STALE,
            "child history source changed during read",
        ) from exc
    except ProtocolError:
        raise
    except (FileNotFoundError, OSError) as exc:
        raise ProtocolError(
            ErrorDescriptor.CHILD_RUN_NOT_FOUND, "child run was not found"
        ) from exc
    runtime = type(
        "ObserverRuntime",
        (),
        {"session_id": target[0], "opened": type("Opened", (), {"store": reader})()},
    )()
    history_params = {key: value for key, value in params.items() if key != "run_id"}
    if "before" not in history_params:
        history_params["before"] = None
    return history.history(
        runtime,
        history_params,
        backward_enabled=True,
        fidelity=fidelity,
        child_runs=True,
    )


def project_status(target: object, *, served_turn_end: bool = False) -> dict[str, Any]:
    status = dict(target)
    store = status.pop("store", None)
    status["compaction_markers"] = (
        store.compaction_marker_count() if store is not None else 0
    )
    if served_turn_end:
        outcome = None
        entry = store.last_turn_end() if store is not None else None
        if entry is not None:
            outcome = {
                "turn_id": entry.data["turn_id"],
                "reason": entry.data["reason"],
            }
        status["last_turn_outcome"] = outcome
    return status



def _observer_runtime(store: object) -> object:
    return type("ObserverRuntime", (), {"opened": type("Opened", (), {"store": store})()})()
