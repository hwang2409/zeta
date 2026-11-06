from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import content, ergonomics
from .protocol import ProtocolError

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
class _Connection:
    connection: Any
    role: str | None = None
    protocol_version: str = "1.0"
    features: frozenset[str] = frozenset()
    queue: asyncio.Queue[tuple[bytes, asyncio.Future[bool]]] | None = None
    writer_task: asyncio.Task[None] | None = None
    queued_bytes: int = 0
    ready: bool = False
    flushing: bool = False
    buffered_frames: deque[bytes] = field(default_factory=deque)
    buffered_bytes: int = 0
    overflowed: bool = False


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
        self._connections[connection] = _Connection(connection)

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
                raise ProtocolError(-32004, "observer limit reached") from exc
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
            state.queue = asyncio.Queue(maxsize=256)
            state.writer_task = asyncio.create_task(self._observer_writer(state))
            self._observers.append(state)
        state.role = role
        state.protocol_version = protocol_version
        state.features = features

    def is_controller(self, connection: Any) -> bool:
        state = self._connections.get(connection)
        return state is not None and state is self._controller

    def controller_ready(self, connection: Any) -> None:
        if not self.is_controller(connection):
            raise RuntimeError("connection does not own the controller claim")
        state = self._connections[connection]
        state.ready = True
        state.flushing = bool(state.buffered_frames)

    async def flush_controller(self, connection: Any) -> bool:
        state = self._connections.get(connection)
        if state is None or state is not self._controller:
            return False
        if state.overflowed:
            error = json.dumps(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32011, "message": "controller output overflow"}},
                separators=(",", ":"),
            ).encode() + b"\n"
            with contextlib.suppress(ConnectionError, OSError, RuntimeError):
                await connection._write_physical(error)
            await connection.close()
            return False
        try:
            while state.buffered_frames:
                frame = state.buffered_frames[0]
                if not await connection._write_physical(frame):
                    return False
                state.buffered_frames.popleft()
                state.buffered_bytes -= len(frame)
            return True
        finally:
            state.flushing = False

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
        if not state.ready or state.flushing:
            if state.buffered_bytes + len(frame) > _CONTROLLER_BUFFER_LIMIT or len(state.buffered_frames) >= _CONTROLLER_BUFFER_FRAMES:
                state.overflowed = True
                state.buffered_frames.clear()
                state.buffered_bytes = 0
                return False
            state.buffered_frames.append(frame)
            state.buffered_bytes += len(frame)
            return True
        return await state.connection._write_physical(frame)

    def release(self, connection: Any) -> None:
        state = self._connections.pop(connection, None)
        if state is None:
            return
        if state is self._controller:
            self._controller = None
        state.buffered_frames.clear()
        state.buffered_bytes = 0
        if state in self._observers:
            self._observers.remove(state)
        if state.writer_task is not None:
            state.writer_task.cancel()
        if state.queue is not None:
            while not state.queue.empty():
                _, future = state.queue.get_nowait()
                future.cancel()

    async def send(self, connection: Any, frame: bytes) -> bool:
        state = self._connections.get(connection)
        if state is None:
            return False
        if state.role != "observer":
            return await connection._write_physical(frame)
        assert state.queue is not None
        if state.queued_bytes + len(frame) > 4 * 1024 * 1024 or state.queue.full():
            await self._overflow(state)
            return False
        future = self.loop.create_future()
        state.queue.put_nowait((frame, future))
        state.queued_bytes += len(frame)
        return await future

    async def _observer_writer(self, state: _Connection) -> None:
        assert state.queue is not None
        try:
            while True:
                frame, future = await state.queue.get()
                if future.cancelled():
                    state.queued_bytes -= len(frame)
                    continue
                try:
                    result = await state.connection._write_physical(frame)
                except (ConnectionError, OSError, RuntimeError) as exc:
                    if not future.done():
                        future.set_exception(exc)
                    return
                state.queued_bytes -= len(frame)
                if not future.done():
                    future.set_result(result)
        except asyncio.CancelledError:
            return

    async def _overflow(self, state: _Connection) -> None:
        if state.connection not in self._connections:
            return
        if state.queue is not None:
            while not state.queue.empty():
                frame, future = state.queue.get_nowait()
                state.queued_bytes -= len(frame)
                future.cancel()
        if state.writer_task is not None:
            state.writer_task.cancel()
            await asyncio.gather(state.writer_task, return_exceptions=True)
            state.writer_task = None
        error = json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32011, "message": "observer output overflow"}}, separators=(",", ":")).encode() + b"\n"
        with contextlib.suppress(ConnectionError, OSError, RuntimeError):
            await state.connection._write_physical(error)
        await state.connection.close()

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
            if state.writer_task is not None:
                state.writer_task.cancel()
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
        return runtime.session_id, {
            "session": client._session_snapshot(),
            "state": (
                "idle"
                if client.server.control.unhealthy() or runtime.state is None
                else runtime.state.status
            ),
            "explicit_finish": runtime.loop.explicit_finish if runtime.loop is not None else runtime._config(None, None).agents_explicit_finish,
            "pending_approvals": [], "usage": dict(runtime.usage), "store": runtime.opened.store,
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
