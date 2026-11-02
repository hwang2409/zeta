from __future__ import annotations

import asyncio
import json
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
    snapshot_required: bool = True
    snapshot_session_id: str | None = None
    snapshot_seq: int = -1

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
        if role == "observer":
            state.snapshot_required = True

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
        # A full queue cannot safely accept another frame. Closing is the
        # fail-closed signal; the next connection must take a new snapshot.
        await state.connection.close()

    def begin_snapshot(self, observer: Any, session_id: str) -> None:
        state = self._connections.get(observer)
        if state is None or state.role != "observer":
            raise RuntimeError("observer is not connected")
        state.snapshot_required = True
        state.snapshot_session_id = session_id

    def activate_snapshot(
        self, observer: Any, session_id: str, page: dict[str, Any]
    ) -> bool:
        state = self._connections.get(observer)
        if state is None or state.role != "observer":
            return False
        if state.snapshot_session_id != session_id:
            return False
        anchor = page.get("anchor")
        state.snapshot_seq = anchor.get("seq", -1) if isinstance(anchor, dict) else -1
        state.snapshot_required = False
        return True

    @staticmethod
    def _row_projection(entry: Any) -> dict[str, Any]:
        data = getattr(entry, "data", {})
        display = None
        if getattr(entry, "type", None) == "message":
            message = data.get("message", {}) if isinstance(data, dict) else {}
            content = []
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    content.append({"type": "text", "text": str(block.get("text", ""))[:8000]})
                elif kind == "thinking":
                    content.append({"type": "thinking", "text": ""})
                elif kind == "image":
                    content.append({"type": "attachment", "name": str(block.get("path", "image")).split("/")[-1], "size": block.get("size", 0) or 0})
                elif kind == "tool_use":
                    call = block.get("tool_call", {})
                    content.append({"type": "tool_use", "tool_call": {"id": call.get("id", ""), "name": call.get("name", ""), "arguments": {}}})
            display = {"role": message.get("role"), "content": content, "tool_result": None}
        elif getattr(entry, "type", None) == "notification":
            display = {"role": "system", "content": [{"type": "text", "text": str(data.get("text", "background agent notification"))[:8000]}], "tool_result": None}
        return {"seq": entry.seq, "entry_id": entry.id, "parent_entry_id": entry.parent_id, "lane": entry.lane, "type": entry.type, "display": display}

    def _enqueue_projection(self, session_id: str, entry: Any) -> None:
        for state in tuple(self._observers):
            if state.snapshot_required or state.snapshot_session_id not in {None, session_id}:
                continue
            if entry.seq <= state.snapshot_seq:
                continue
            payload = json.dumps({"jsonrpc": "2.0", "method": "event", "params": {"event": "transcript_row", "session_id": session_id, "row": self._row_projection(entry)}}, separators=(",", ":")).encode()
            if state.queue is None or state.queue.full():
                asyncio.create_task(self._overflow(state))
                continue
            state.queue.put_nowait((payload, self.loop.create_future()))

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
            items.clear()
        for session_id, delta in items:
            if delta == "reset":
                for state in tuple(self._observers):
                    if state.snapshot_session_id == session_id:
                        state.snapshot_required = True
                continue
            for entry in delta:
                self._enqueue_projection(session_id, entry)

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
