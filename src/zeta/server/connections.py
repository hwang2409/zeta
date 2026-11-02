from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

OBSERVER_REQUESTS = frozenset({
    "list_sessions", "status", "session_tree", "session_history",
    "model_catalog", "session_settings", "ping", "delivery_status",
    "session_snapshot",
})
OBSERVER_READ_ONLY_CODE = "observer_read_only"

@dataclass
class _Connection:
    connection: Any
    role: str | None = None
    features: frozenset[str] = frozenset()
    queue: asyncio.Queue[tuple[bytes, asyncio.Future[bool]]] | None = None
    writer_task: asyncio.Task[None] | None = None

class ConnectionRegistry:
    """Owns serve connection claims and the single output fan-out seam."""
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._connections: dict[Any, _Connection] = {}
        self._controller: _Connection | None = None
        self._observers: list[_Connection] = []
        self._closed = False
        self._ingress_lock = threading.Lock()
        self._ingress: deque[tuple[str, Any]] = deque()
        self._ingress_rows = 0
        self._drain_scheduled = False
        self._ingress_overflow = False

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
                raise RuntimeError("another client is already connected")
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
        for item in list(state.queue._queue) if state.queue is not None else ():
            item[1].cancel()

    async def send(self, connection: Any, frame: bytes) -> bool:
        state = self._connections.get(connection)
        if state is None or state.role != "observer":
            return await connection._write_physical(frame)
        assert state.queue is not None
        if state.queue.full():
            await self._overflow(state)
            return False
        future: asyncio.Future[bool] = self.loop.create_future()
        state.queue.put_nowait((frame, future))
        return await future

    async def offer_live(self, frame: bytes, *, required_feature: str | None = None) -> bool:
        controller_result = True
        if self._controller is not None and (
            required_feature is None or required_feature in self._controller.features
        ):
            controller_result = await self.send(self._controller.connection, frame)
        for state in tuple(self._observers):
            if required_feature is not None and required_feature not in state.features:
                continue
            if state.queue is None or state.queue.full():
                await self._overflow(state)
                continue
            future: asyncio.Future[bool] = self.loop.create_future()
            state.queue.put_nowait((frame, future))
        return controller_result

    async def _observer_writer(self, state: _Connection) -> None:
        assert state.queue is not None
        try:
            while True:
                frame, future = await state.queue.get()
                if future.cancelled():
                    continue
                try:
                    result = await state.connection._write_physical(frame)
                except (ConnectionError, OSError, RuntimeError) as exc:
                    if not future.done():
                        future.set_exception(exc)
                    return
                if not future.done():
                    future.set_result(result)
        except asyncio.CancelledError:
            return

    async def _overflow(self, state: _Connection) -> None:
        if state.connection not in self._connections:
            return
        await state.connection.close()

    def offer_installed_rows(self, session_id: str, owned_delta: Any) -> None:
        try:
            count = len(owned_delta) if owned_delta != "reset" else 0
            with self._ingress_lock:
                if self._closed:
                    return
                if len(self._ingress) >= 256 or self._ingress_rows + count > 4096:
                    self._ingress_overflow = True
                    if not self._drain_scheduled:
                        self._drain_scheduled = True
                        self.loop.call_soon_threadsafe(self._drain_installed_ingress)
                    return
                self._ingress.append((session_id, owned_delta))
                self._ingress_rows += count
                if self._drain_scheduled:
                    return
                self._drain_scheduled = True
            self.loop.call_soon_threadsafe(self._drain_installed_ingress)
        except RuntimeError:
            return

    def _drain_installed_ingress(self) -> None:
        with self._ingress_lock:
            items = list(self._ingress)
            self._ingress.clear()
            self._ingress_rows = 0
            overflow = self._ingress_overflow
            self._ingress_overflow = False
            self._drain_scheduled = False
        if overflow:
            for state in tuple(self._observers):
                asyncio.create_task(self._overflow(state))
        # durable projection is attached by the session owner in a later slice;
        # consuming here keeps callback state loop-owned and bounded.
        del items

    def close(self) -> None:
        self._closed = True
        for state in tuple(self._connections.values()):
            if state.writer_task is not None:
                state.writer_task.cancel()
            asyncio.create_task(state.connection.close())
        self._connections.clear()
        self._observers.clear()
        self._controller = None

__all__ = ["OBSERVER_READ_ONLY_CODE", "OBSERVER_REQUESTS", "ConnectionRegistry"]
