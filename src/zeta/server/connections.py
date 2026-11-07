from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from . import content, ergonomics, event_contract
from .protocol import FrameCodec, FrameWriteCancelled, ProtocolError

OBSERVER_REQUESTS = frozenset({
    "list_sessions", "status", "session_tree", "session_history", "content_read",
    "model_catalog", "session_settings", "ping", "delivery_status",
})
OBSERVER_READ_ONLY_CODE = "observer_read_only"
MAX_CONNECTIONS = 10  # controller, eight observers, and one handshake
_CONTROLLER_BUFFER_LIMIT = 4 * 1024 * 1024
_CONTROLLER_BUFFER_FRAMES = 256


@dataclass(frozen=True)
class ControllerMetadata:
    connection: Any
    protocol_version: str
    features: frozenset[str]


@dataclass
class _OutputFrame:
    payload: bytes
    future: asyncio.Future[bool] | None = None
    buffered_controller_event: bool = False


@dataclass
class _OutputState:
    phase: Literal[
        "claimed_buffering", "flushing", "ready", "overflowing"
    ] = "claimed_buffering"
    queue: deque[_OutputFrame] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    writer_task: asyncio.Task[None] | None = None
    current_frame: _OutputFrame | None = None
    current_write: asyncio.Task[tuple[bool, FrameWriteCancelled | None]] | None = None
    ready: asyncio.Future[bool] | None = None
    callers_settled: asyncio.Event = field(default_factory=asyncio.Event)
    pending_callers: int = 0
    overflow_done: asyncio.Future[bool] | None = None
    queued_bytes: int = 0
    pending_controller_bytes: int = 0
    pending_controller_frames: int = 0
    overflowed: bool = False
    overflowed_before_flush: bool = False
    released: bool = False


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
        self._observers: list[_Connection] = []
        self._closed = False
        self._read_condition = threading.Condition()
        self._read_jobs: deque[tuple[Callable[[], Any], asyncio.Future[Any]]] = deque()
        self._read_stopping = False
        self._read_threads = [threading.Thread(target=self._read_worker, daemon=True, name=f"zeta-read-{i}") for i in range(2)]
        for thread in self._read_threads:
            thread.start()

    def register(self, connection: Any) -> None:
        if self._closed:
            raise RuntimeError("connection registry is closed")
        if len(self._connections) >= MAX_CONNECTIONS:
            raise RuntimeError("connection limit reached")
        state = _Connection(connection)
        output = _OutputState()
        output.ready = self.loop.create_future()
        output.callers_settled.set()
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
            raise ProtocolError(-32602, "role must be 'controller' or 'observer'")
        role = str(requested_role)
        if max_role == "observer" and role == "controller":
            raise ProtocolError(-32602, "credential permits observer role only")
        if role == "observer" and (
            protocol_version == "1.0" or "serve_observers" not in features
        ):
            raise ProtocolError(
                -32602, "observer requires protocol 1.1 and serve_observers"
            )
        try:
            self._claim(connection, role, protocol_version, features)
        except RuntimeError as exc:
            if role == "observer":
                raise ProtocolError(-32001, "observer limit reached") from exc
            raise ProtocolError(-32001, "another client is already connected") from exc
        connection.role = role
        if role == "controller":
            connection.server.runtime.set_post_stream_provider_retry(
                "assistant_reset" in features
            )
        return role

    def _claim(
        self, connection: Any, role: str, protocol_version: str, features: frozenset[str]
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
        state.protocol_version = protocol_version
        state.features = features

    def is_controller(self, connection: Any) -> bool:
        state = self._connections.get(connection)
        return state is not None and state is self._controller

    def controller_metadata(self) -> ControllerMetadata | None:
        state = self._controller
        if state is None:
            return None
        return ControllerMetadata(state.connection, state.protocol_version, state.features)

    async def publish_controller(
        self,
        event: str,
        session_id: str | None,
        *,
        required_feature: str | None = None,
        **fields: object,
    ) -> bool:
        if event not in event_contract.WIRE_EVENT_NAMES:
            raise ValueError(
                f"undocumented serve event {event!r}; add it to event_contract "
                "and docs/serve-protocol.md"
            )
        state = self._controller
        if state is None or (
            required_feature is not None and required_feature not in state.features
        ):
            return False
        if (
            event == "approval_end"
            and "approval_reconnect" not in state.features
            and isinstance(fields.get("data"), dict)
            and fields["data"].get("reason") == "grace_expired"
        ):
            return False
        frame = state.connection.codec.notification(event, session_id, **fields)
        output = state.output
        assert output is not None
        if output.phase in {"claimed_buffering", "flushing"}:
            if output.phase == "flushing" and output.overflowed:
                # Honest rejection with -32011 belongs to phase B.
                return True
            if (
                output.pending_controller_bytes + len(frame) > _CONTROLLER_BUFFER_LIMIT
                or output.pending_controller_frames >= _CONTROLLER_BUFFER_FRAMES
            ):
                output.overflowed = True
                self._clear_output(output)
                return False
            output.pending_controller_bytes += len(frame)
            output.pending_controller_frames += 1
            self._enqueue(
                output,
                _OutputFrame(frame, buffered_controller_event=True),
                wake=output.phase == "flushing",
            )
            return True
        if output.phase != "ready":
            return False
        return await self._send_frame(output, frame)

    def release(self, connection: Any) -> None:
        state = self._connections.pop(connection, None)
        if state is None:
            return
        if state is self._controller:
            self._controller = None
        if state in self._observers:
            self._observers.remove(state)
        output = state.output
        if output is not None:
            self._release_output(output, state.role)

    async def send(self, connection: Any, frame: bytes) -> bool:
        state = self._connections.get(connection)
        if state is None or state.output is None:
            return False
        output = state.output
        if output.phase == "claimed_buffering":
            return await self._send_hello(output, frame)
        if output.phase != "ready":
            return False
        if state.role == "observer" and (
            output.queued_bytes + len(frame) > 4 * 1024 * 1024
            or len(output.queue) >= 256
        ):
            return await self._overflow(state)
        return await self._send_frame(output, frame)

    async def _send_hello(self, output: _OutputState, frame: bytes) -> bool:
        hello = _OutputFrame(frame)
        output.queue.appendleft(hello)
        output.queued_bytes += len(frame)
        output.overflowed_before_flush = output.overflowed
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

    async def _send_frame(self, output: _OutputState, frame: bytes) -> bool:
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

    def _enqueue(
        self, output: _OutputState, frame: _OutputFrame, *, wake: bool = True
    ) -> None:
        output.queue.append(frame)
        output.queued_bytes += len(frame.payload)
        if wake:
            output.wake.set()

    async def _output_writer(self, state: _Connection) -> None:
        output = state.output
        assert output is not None
        try:
            while True:
                await output.wake.wait()
                while output.queue:
                    await self._write_output_frame(state, output, output.queue.popleft())
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
            if output.phase == "overflowing":
                return
            if asyncio.current_task().cancelling():
                self._finish_future(output.ready, False)
                raise
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
            if output.overflowed_before_flush:
                output.phase = "overflowing"
                self._enqueue(
                    output, _OutputFrame(self._overflow_frame("controller"))
                )
                return False
            output.phase = "ready"
            self._finish_future(output.ready, True)
            return False
        if output.phase != "overflowing":
            return False
        await state.connection.close()
        self._finish_future(output.ready, False)
        self._finish_future(output.overflow_done, False)
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

    async def _overflow(self, state: _Connection) -> bool:
        output = state.output
        assert output is not None
        if state.connection not in self._connections:
            return False
        output.phase = "overflowing"
        self._clear_output(output)
        if output.current_write is not None:
            output.current_write.cancel()
        output.overflow_done = self.loop.create_future()
        self._enqueue(output, _OutputFrame(self._overflow_frame("observer")))
        return await output.overflow_done

    @staticmethod
    def _overflow_frame(role: str) -> bytes:
        return FrameCodec().error_response(
            None, -32011, f"{role} output overflow"
        )

    @staticmethod
    async def _write_physical(
        connection: Any, payload: bytes
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
    def _finish_frame_accounting(
        output: _OutputState, item: _OutputFrame
    ) -> None:
        output.queued_bytes -= len(item.payload)
        if item.buffered_controller_event:
            output.pending_controller_bytes -= len(item.payload)
            output.pending_controller_frames -= 1

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
        if role == "observer":
            self._clear_output(output)
            if output.writer_task is not None:
                output.writer_task.cancel()
            return
        if (
            output.current_frame is None
            and not output.queue
            and output.writer_task is not None
        ):
            output.writer_task.cancel()

    def _settle_output(self, output: _OutputState) -> None:
        while output.queue:
            item = output.queue.popleft()
            self._finish_frame_accounting(output, item)
            self._finish_future(item.future, False)
        if output.writer_task is not None:
            output.writer_task.cancel()
        self._finish_future(output.ready, False)
        self._finish_future(output.overflow_done, False)

    @staticmethod
    def _clear_output(output: _OutputState) -> None:
        while output.queue:
            item = output.queue.popleft()
            ConnectionRegistry._finish_frame_accounting(output, item)
            if item.future is not None and not item.future.done():
                item.future.cancel()

    async def run_blocking(self, callable: Callable[[], Any]) -> Any:
        """Run one store or observer read away from the event loop."""
        future = self.loop.create_future()
        with self._read_condition:
            if self._read_stopping:
                raise RuntimeError("reads are closed")
            self._read_jobs.append((callable, future))
            self._read_condition.notify()
        return await future

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
            raise ProtocolError(-32003, "session is not active")
        return result

    async def read_session(
        self, connection: Any, method: str, params: dict[str, Any]
    ) -> object:
        target = capture_read(connection, method, params)
        features = self._connections[connection].features
        include_delivery_id = {
            "delivery_id",
            "user_message_event",
            "user_message_delivery_id",
        } <= features
        return await self.run_read(
            connection,
            lambda: perform_read(
                method,
                target,
                params,
                include_delivery_id=include_delivery_id,
                backward_enabled="session_history_backward" in features, fidelity="session_history_fidelity" in features, content_read="content_read" in features,
            ),
            session_id=target[0],
        )

    async def read_status(
        self, connection: Any, status: dict[str, Any], session_id: str | None
    ) -> object:
        return await self.run_read(
            connection,
            lambda: project_status(status),
            session_id=session_id,
        )

    async def run_observer_read(self, connection: Any, callable: Callable[[], Any]) -> Any:
        state = self._connections.get(connection)
        if state is None or state.role != "observer":
            raise RuntimeError("observer is not connected")
        return await self.run_blocking(callable)

    def _read_worker(self) -> None:
        while True:
            with self._read_condition:
                while not self._read_jobs and not self._read_stopping:
                    self._read_condition.wait()
                if self._read_stopping and not self._read_jobs:
                    return
                job, future = self._read_jobs.popleft()
            if future.cancelled():
                continue
            try:
                result = job()
            except Exception as exc:  # noqa: BLE001 - worker must return every job failure
                self.loop.call_soon_threadsafe(self._read_done, future, None, exc)
            else:
                self.loop.call_soon_threadsafe(self._read_done, future, result, None)

    @staticmethod
    def _read_done(future: asyncio.Future[Any], result: Any, error: BaseException | None) -> None:
        if future.done():
            return
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)

    @staticmethod
    def requests_for_role(role: str, requests: list[str]) -> list[str]:
        if role == "observer":
            return sorted(OBSERVER_REQUESTS.intersection(requests))
        return requests

    async def route_request(
        self, connection: Any, request_id: str | int, method: str, params: dict[str, Any]
    ) -> tuple[bool, object]:
        state = self._connections.get(connection)
        if state is None or state.role != "observer":
            return False, None
        if method not in OBSERVER_REQUESTS or (
            method == "list_sessions" and "project_id" in params
        ):
            raise ProtocolError(
                -32008,
                "method is unavailable to read-only observers",
                {"code": OBSERVER_READ_ONLY_CODE, "method": method},
            )
        if method not in {
            "list_sessions",
            "status",
            "session_tree",
            "session_history",
            "model_catalog",
            "session_settings", "content_read",
        }:
            return False, None
        if method in {"session_tree", "session_history"}:
            return True, await self.read_session(connection, method, params)
        if method == "content_read":
            if "content_read" not in state.features:
                raise ProtocolError(-32601, "content_read requires the negotiated content_read feature")
            target = capture_read(connection, method, params)
            return True, await self.run_read(connection, lambda: perform_read(method, target, params), session_id=target[0])
        if method == "list_sessions":
            result, _ = await self.run_read(
                connection, lambda: connection._list_sessions(request_id, params)
            )
            return True, result
        target = capture_read(connection, method, params)
        result = await self.run_read(
            connection,
            lambda: perform_read(
                method, target, params,
                backward_enabled="session_history_backward" in state.features, fidelity="session_history_fidelity" in state.features, content_read="content_read" in state.features,
            ),
            session_id=target[0],
        )
        return True, result

    async def aclose(self) -> None:
        self._closed = True
        states = tuple(self._connections.values())
        with self._read_condition:
            self._read_stopping = True
            self._read_condition.notify_all()
        for state in states:
            if state.output is not None:
                self._settle_output(state.output)
        await asyncio.gather(
            *(state.connection.close() for state in states), return_exceptions=True
        )
        self._connections.clear()
        self._observers.clear()
        self._controller = None

def capture_read(client: Any, method: str, params: dict[str, Any]) -> tuple[str | None, object]:
    runtime = client.server.runtime
    if method == "content_read":
        if set(params) - {"handle", "offset", "limit"}:
            raise ProtocolError(
                -32602,
                "invalid content read",
                {"code": "invalid_content_read"},
            )
        if runtime.opened is None:
            raise ProtocolError(-32003, "session is not active")
        return runtime.session_id, runtime.opened.store
    if method in {"session_tree", "session_history"}:
        store = ergonomics.active(runtime, params)
        return runtime.session_id, store
    if method == "status":
        if runtime.opened is None:
            return None, {"session": None, "state": "idle", "explicit_finish": False, "pending_approvals": [], "usage": {}, "compaction_markers": 0}
        transcript_index_failure = (
            runtime.loop.transcript_index_failure(runtime.session_id)
            if client.protocol_version != "1.0" and runtime.loop is not None
            else None
        )
        return runtime.session_id, {
            "session": client._session_snapshot(),
            "state": (
                "idle"
                if client.server.control.unhealthy() or runtime.state is None
                else runtime.state.status
            ),
            "explicit_finish": runtime.loop.explicit_finish if runtime.loop is not None else runtime._config(None, None).agents_explicit_finish,
            "pending_approvals": [], "usage": dict(runtime.usage),
            **(
                {"transcript_index_failure": transcript_index_failure}
                if transcript_index_failure is not None
                else {}
            ),
            "store": runtime.opened.store,
        }
    if method == "model_catalog":
        return None, ergonomics.catalog(runtime)
    if method == "session_settings":
        return runtime.session_id, ergonomics.settings(runtime)
    raise ProtocolError(-32601, "method not found")


def project_status(target: object) -> dict[str, Any]:
    status = dict(target)
    store = status.pop("store", None)
    status["compaction_markers"] = (
        store.compaction_marker_count() if store is not None else 0
    )
    return status


def perform_read(
    method: str,
    target: tuple[str | None, object],
    params: dict[str, Any],
    *,
    include_delivery_id: bool = False,
    backward_enabled: bool = False,
    fidelity: bool = False,
    content_read: bool = False,
) -> object:
    if method == "status":
        return project_status(target[1])

    if method in {"model_catalog", "session_settings"}:
        return target[1]
    store = target[1]
    runtime = type("ObserverRuntime", (), {"opened": type("Opened", (), {"store": store})()})()
    if method == "session_tree":
        return ergonomics.tree(runtime)
    if method == "session_history":
        return ergonomics.history(
            runtime,
            params,
            include_delivery_id=include_delivery_id,
            backward_enabled=backward_enabled,
            fidelity=fidelity,
            content_read=content_read,
        )
    if method == "content_read":
        return content.read(store, params.get("handle"), offset=params.get("offset", 0), limit=params.get("limit", content.MAX_CONTENT_READ))
    raise ProtocolError(-32601, "method not found")
