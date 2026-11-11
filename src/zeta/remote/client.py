"""deep client for remote observer discovery and replay attachment."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

from ..protocol.types import StreamEvent
from .discovery import (
    RemoteSession,
    RemoteTarget,
    configured_hosts,
    dedupe_rows,
    split_target,
    validate_host,
)
from .transport import RemoteTransport, TransportItem, remote_bridge_command
from .wire import HISTORY_REQUEST_ID, BoundedLineDecoder, RemoteWireError, decode_line

COMPATIBILITY_MESSAGE = "remote zeta is missing or too old. update ~/me/fun/zeta on HOST, then run: cd ~/me/fun/zeta && uv tool install --editable ."
_OLD_ZETA_MARKERS = (b"command not found", b"invalid choice: '__remote_bridge'", b"unknown bridge_version")
MAX_DIAGNOSTIC_BYTES = 64 * 1024
MAX_LIST_BYTES = 1024 * 1024
MAX_ROW_BYTES = 8192
_ROW_FIELDS = ("session_id", "name", "state", "machine_id")
# the overflow item tells the consumer that _admit dropped the stream.
_OVERFLOW = TransportItem("overflow", b"")


def _check_compatible(status: bytes, diagnostics: bytes, display: str) -> None:
    if status == b"127" or any(marker in diagnostics for marker in _OLD_ZETA_MARKERS):
        raise RemoteWireError(COMPATIBILITY_MESSAGE.replace("HOST", display))


def _session_row(line: bytes, display: str) -> RemoteSession | None:
    if len(line) > MAX_ROW_BYTES:
        raise RemoteWireError("remote session row is too large")
    try:
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("kind") != "session":
            return None
        session_id, name, state, machine_id = (value.get(key) for key in _ROW_FIELDS)
        return RemoteSession(
            display, validate_host(str(session_id)), str(name or ""),
            state if isinstance(state, str) else None, str(machine_id or ""),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


class RemoteSessionClient:
    def __init__(self, home: Path, *, ssh: str = "ssh", transport_factory: Callable[..., RemoteTransport] = RemoteTransport) -> None:
        self.home, self.ssh, self._transport_factory = home, ssh, transport_factory
        self.transport: RemoteTransport | None = None
        self._queue: asyncio.Queue[TransportItem] | None = None
        self._events: asyncio.Queue[StreamEvent | str] | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._targets: dict[str, RemoteTarget] = {}

    async def discover(self, hosts: tuple[str | RemoteTarget, ...] | None = None) -> tuple[RemoteSession, ...]:
        targets = hosts or configured_hosts(self.home)
        results = await asyncio.gather(*(self._list(target) for target in targets), return_exceptions=True)
        rows = [row for result in results if not isinstance(result, BaseException) for row in result]
        failures = [result for result in results if isinstance(result, BaseException)]
        if not rows and failures:
            if isinstance(failures[0], RemoteWireError): raise failures[0]
            raise RemoteWireError("remote discovery failed") from failures[0]
        return dedupe_rows(rows)

    def _target(self, value: str | RemoteTarget) -> RemoteTarget:
        if isinstance(value, RemoteTarget):
            self._targets[value.name] = value
            return value
        return self._targets.get(value) or RemoteTarget(value, validate_host(value))

    def _transport(self, target: RemoteTarget, callback: Callable[[TransportItem], None]) -> RemoteTransport:
        return self._transport_factory(target.host, ssh=self.ssh, port=target.port, callback=callback)

    async def _list(self, value: str | RemoteTarget) -> tuple[RemoteSession, ...]:
        if isinstance(value, str) and split_target(value)[1] is not None:
            raise ValueError("discovery target cannot include a session prefix")
        target = self._target(value)
        loop = asyncio.get_running_loop()
        done: asyncio.Future[bytes] = loop.create_future()
        stdout, stderr = bytearray(), bytearray()
        def admit(item: TransportItem) -> None:
            if done.done(): return
            if item.stream == "exit": done.set_result(item.data)
            elif item.stream == "stderr": stderr.extend(item.data[:MAX_DIAGNOSTIC_BYTES - len(stderr)])
            elif len(stdout) + len(item.data) > MAX_LIST_BYTES: done.set_exception(RemoteWireError("remote discovery is too large"))
            else: stdout.extend(item.data)
        transport = self._transport(target, lambda item: loop.call_soon_threadsafe(admit, item))
        transport.start(remote_bridge_command("list", remote_home=target.remote_home))
        try: status = await done
        finally: await transport.close_async()
        _check_compatible(status, bytes(stderr), target.name)
        rows = (_session_row(line, target.name) for line in bytes(stdout).splitlines())
        return tuple(row for row in rows if row is not None)

    async def attach(self, host: str, session: RemoteSession | str) -> None:
        name, prefix = split_target(host)
        target = self._target(name)
        session_id = validate_host(session.session_id if isinstance(session, RemoteSession) else session)
        if prefix is not None and session_id != prefix: raise ValueError("attach requires a selected session")
        await self.detach()
        loop = asyncio.get_running_loop(); self._generation += 1; generation = self._generation
        self._queue = asyncio.Queue(maxsize=256); self._events = asyncio.Queue(maxsize=256)
        transport = self._transport(target, lambda item: loop.call_soon_threadsafe(self._admit, generation, item))
        transport.start(remote_bridge_command("attach", session_id, target.remote_home))
        for request in (
            {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol_version": "1.1", "role": "observer", "features": ["session_replay_seq", "serve_observers"]}},
            {"jsonrpc": "2.0", "id": HISTORY_REQUEST_ID, "method": "session_history_and_subscribe", "params": {"session_id": session_id}},
        ):
            transport.send((json.dumps(request, separators=(",", ":")) + "\n").encode())
        self.transport = transport
        self._event_task = asyncio.create_task(self._consume(self._queue, self._events, target.name))

    def _admit(self, generation: int, item: TransportItem) -> None:
        if generation != self._generation or self._queue is None: return
        try: self._queue.put_nowait(item)
        except asyncio.QueueFull:
            # the consumer is behind; drop the backlog and end the link instead of growing without bound.
            while not self._queue.empty(): self._queue.get_nowait()
            self._queue.put_nowait(_OVERFLOW)
            self._queue = None

    async def _consume(self, queue: asyncio.Queue[TransportItem], events: asyncio.Queue[StreamEvent | str], display: str) -> None:
        decoder = BoundedLineDecoder()
        diagnostics = bytearray()
        try:
            while (item := await queue.get()).stream != "exit":
                if item is _OVERFLOW: raise RemoteWireError("remote stream overflow")
                if item.stream == "stderr": diagnostics.extend(item.data[:MAX_DIAGNOSTIC_BYTES - len(diagnostics)])
                else:
                    for line in decoder.feed(item.data):
                        for event in decode_line(line): await events.put(event)
            _check_compatible(item.data, bytes(diagnostics), display)
            await events.put("disconnected")
        except RemoteWireError as exc:
            await events.put("fatal: " + str(exc))
            if self.transport is not None: await self.transport.close_async()

    async def next_event(self) -> StreamEvent | str:
        if self._events is None: raise RuntimeError("no remote attachment")
        return await self._events.get()

    async def detach(self) -> None:
        self._generation += 1
        task, transport = self._event_task, self.transport
        self._event_task = self._queue = self._events = self.transport = None
        if task is not None: task.cancel(); await asyncio.gather(task, return_exceptions=True)
        if transport is not None: await transport.close_async()
