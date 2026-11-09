from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.thread_call import ThreadCallWorkers
from . import content
from .handlers import history, session, settings, validation
from .protocol import (
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    FrameWriteCancelled,
    ProtocolError,
    _Frame,
)

CONNECTION_FLUSH_DEADLINE_SECONDS = 1.0

MAX_CONNECTIONS = 10  # controller, eight observers, and one handshake
OUTPUT_MAX_BYTES = 4 * 1024 * 1024
OUTPUT_MAX_FRAMES = 256
OUTPUT_OVERFLOW_CLOSE_DEADLINE = 1.0


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
class _OutputFrame:
    payload: _Frame
    future: asyncio.Future[bool] | None = None


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


class ConnectionRegistry:
    """Owns connection claims, output, and bounded observer reads."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._connections: dict[Any, _Connection] = {}
        self._controller: _Connection | None = None
        self._controller_output_paused = False
        self._observers: list[_Connection] = []
        self._closed = False
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
        if required_feature is not None and required_feature not in state.features:
            return Admission("feature_rejected"), None, None
        if (
            event is EventDescriptor.APPROVAL_END
            and "approval_reconnect" not in state.features
            and isinstance(fields.get("data"), dict)
            and fields["data"].get("reason") == "grace_expired"
        ):
            return Admission("feature_rejected"), None, None
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

    def try_admit_controller(
        self,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None = None,
        **fields: object,
    ) -> Admission:
        """Admit one controller event without waiting for physical output."""
        outcome, frame, output = self._controller_preflight(
            self._controller,
            event,
            session_id,
            required_feature=required_feature,
            fields=fields,
        )
        if outcome.kind != "admitted" or frame is None or output is None:
            return outcome
        if not self._admit(self._controller, frame):
            return Admission("capacity_rejected")
        self._enqueue(
            output,
            _OutputFrame(frame),
            wake=self._phase_allows_writes(output),
        )
        return outcome

    async def publish_controller(
        self,
        event: EventDescriptor,
        session_id: str | None,
        *,
        required_feature: str | None = None,
        **fields: object,
    ) -> bool:
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
        for state in states:
            frame = state.connection.codec.notification(event, session_id, data=data)
            if frame is None:
                continue
            if state is self._controller:
                await self.send(state.connection, frame)
            else:
                self._publish_observer(state, frame)

    def release(self, connection: Any) -> None:
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
            and not self._fits(output, frame)
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
            self._finish_future(item.future, error=exc)
            if output.phase == "flushing":
                self._reset_flush(output, error=exc)
            return
        finally:
            output.current_frame = None
            output.current_write = None
        self._finish_frame_accounting(output, item)
        self._finish_future(item.future, result)
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

    def _fits(self, output: _OutputState, frame: _Frame) -> bool:
        control = output.overflow_frame
        assert control is not None
        return (
            output.retained_bytes + len(frame) + len(control) <= OUTPUT_MAX_BYTES
            and output.retained_frames + 2 <= OUTPUT_MAX_FRAMES
        )

    def _admit(self, state: _Connection, frame: _Frame) -> bool:
        output = state.output
        assert output is not None
        if output.overflowed:
            return False
        if self._fits(output, frame):
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

    @staticmethod
    def _finish_frame_accounting(output: _OutputState, item: _OutputFrame) -> None:
        output.retained_bytes -= len(item.payload)
        output.retained_frames -= 1
        assert output.retained_bytes >= 0
        assert output.retained_frames >= 0

    @staticmethod
    def _assert_retained_accounting(output: _OutputState) -> None:
        frames = list(output.queue)
        if output.current_frame is not None:
            frames.append(output.current_frame)
        assert output.retained_bytes == sum(len(item.payload) for item in frames)
        assert output.retained_frames == len(frames)

    @staticmethod
    async def _cancel_frame(
        output: _OutputState, item: _OutputFrame
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
        ConnectionRegistry._finish_frame_accounting(output, item)
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

    @staticmethod
    def _clear_output(output: _OutputState, *, result: bool | None = None) -> None:
        while output.queue:
            item = output.queue.popleft()
            ConnectionRegistry._finish_frame_accounting(output, item)
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
        features = self._connections[connection].features
        return await self.run_read(
            connection,
            lambda: history.history(
                _observer_runtime(store),
                params,
                include_delivery_id={
                    "delivery_id", "user_message_event", "user_message_delivery_id"
                } <= features,
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
        store = runtime.opened.store
        return await self.run_read(
            connection,
            lambda: content.read(
                store,
                params.get("handle"),
                offset=params.get("offset", 0),
                limit=params.get("limit", content.MAX_CONTENT_READ),
            ),
            session_id=runtime.session_id,
        )

    async def read_catalog(self, connection: Any) -> object:
        runtime = connection.server.runtime
        return await self.run_read(connection, lambda: settings.catalog(runtime))

    async def read_settings(self, connection: Any, params: dict[str, Any]) -> object:
        runtime = connection.server.runtime
        return await self.run_read(
            connection,
            lambda: settings.settings(runtime),
            session_id=runtime.session_id,
        )

    async def read_status(
        self, connection: Any, status: dict[str, Any], session_id: str | None
    ) -> object:
        features = self._connections[connection].features
        return await self.run_read(
            connection,
            lambda: project_status(
                status, served_turn_end="served_turn_end" in features
            ),
            session_id=session_id,
        )

    async def aclose(self) -> None:
        self._closed = True
        states = tuple(self._connections.values())
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
            outcome = {"turn_id": entry.data["turn_id"], "reason": entry.data["reason"]}
        status["last_turn_outcome"] = outcome
    return status


def _observer_runtime(store: object) -> object:
    return type(
        "ObserverRuntime",
        (),
        {"opened": type("Opened", (), {"store": store})()},
    )()
