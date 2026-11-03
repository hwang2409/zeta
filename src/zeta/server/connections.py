from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

OBSERVER_REQUESTS = frozenset({
    "list_sessions", "status", "session_tree", "session_history",
    "model_catalog", "session_settings", "ping", "delivery_status",
})
OBSERVER_READ_ONLY_CODE = "observer_read_only"


@dataclass
class _Connection:
    connection: Any
    role: str | None = None
    features: frozenset[str] = frozenset()
    queue: asyncio.Queue[tuple[bytes, asyncio.Future[bool]]] | None = None
    writer_task: asyncio.Task[None] | None = None
    queued_bytes: int = 0


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
        self._read_threads = [threading.Thread(target=self._read_worker, daemon=True, name=f"zeta-observer-read-{i}") for i in range(2)]
        for thread in self._read_threads:
            thread.start()

    def register(self, connection: Any) -> None:
        if self._closed:
            raise RuntimeError("connection registry is closed")
        self._connections[connection] = _Connection(connection)

    def claim(self, connection: Any, role: str, features: frozenset[str]) -> None:
        state = self._connections[connection]
        if state.role is not None:
            raise ValueError("connection already claimed")
        if role == "controller":
            if self._controller is not None:
                raise RuntimeError("controller occupied")
            self._controller = state
        elif role == "observer":
            if len(self._observers) >= 8:
                raise RuntimeError("observer limit reached")
            state.queue = asyncio.Queue(maxsize=256)
            state.writer_task = asyncio.create_task(self._observer_writer(state))
            self._observers.append(state)
        else:
            raise ValueError("invalid role")
        state.role = role
        state.features = features

    def release(self, connection: Any) -> None:
        state = self._connections.pop(connection, None)
        if state is None:
            return
        if state is self._controller:
            self._controller = None
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

    async def run_observer_read(self, connection: Any, callable: Callable[[], Any]) -> Any:
        state = self._connections.get(connection)
        if state is None or state.role != "observer":
            raise RuntimeError("observer is not connected")
        future = self.loop.create_future()
        with self._read_condition:
            if self._read_stopping:
                raise RuntimeError("observer reads are closed")
            self._read_jobs.append((callable, future))
            self._read_condition.notify()
        return await future

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

    def allows(self, connection: Any, method: str, params: dict[str, Any]) -> bool:
        state = self._connections.get(connection)
        if state is None or state.role != "observer":
            return True
        return method in OBSERVER_REQUESTS and not (method == "list_sessions" and "project_id" in params)

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

    def close(self) -> None:
        self._closed = True
        with self._read_condition:
            self._read_stopping = True
            self._read_condition.notify_all()
        for state in tuple(self._connections.values()):
            if state.writer_task is not None:
                state.writer_task.cancel()
        self._connections.clear()
        self._observers.clear()
        self._controller = None
