"""deep client for remote observer discovery and replay attachment."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
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
from .wire import BoundedLineDecoder, RemoteWireError, decode_event, history_events

COMPATIBILITY_MESSAGE = "remote zeta is missing or too old. update ~/me/fun/zeta on HOST, then run: cd ~/me/fun/zeta && uv tool install --editable ."

@dataclass(frozen=True, slots=True)
class RemoteView:
    host: str
    session_id: str
    session_name: str
    replay_seq: int = 0

class RemoteSessionClient:
    def __init__(self, home: Path, *, ssh: str = "ssh", transport_factory: Callable[..., RemoteTransport] = RemoteTransport) -> None:
        self.home, self.ssh, self._transport_factory = home, ssh, transport_factory
        self.transport: RemoteTransport | None = None
        self.view: RemoteView | None = None
        self._queue: asyncio.Queue[TransportItem] | None = None
        self._events: asyncio.Queue[StreamEvent | str] | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._targets: dict[str, tuple[str, str]] = {}

    async def discover(self, hosts: tuple[str, ...] | None = None) -> tuple[RemoteSession, ...]:
        targets = hosts or configured_hosts(self.home)
        results = await asyncio.gather(*(self._list(target) for target in targets), return_exceptions=True)
        rows: list[RemoteSession] = []
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, BaseException): failures.append(result)
            else: rows.extend(result)
        if not rows and failures:
            first = failures[0]
            if isinstance(first, RemoteWireError): raise first
            raise RemoteWireError("remote discovery failed") from first
        return dedupe_rows(rows)

    async def _list(self, target: str | RemoteTarget) -> tuple[RemoteSession, ...]:
        if isinstance(target, RemoteTarget):
            host, remote_home, display = target.host, target.remote_home, target.name
            self._targets[display] = (host, remote_home)
        else:
            host, prefix = split_target(target)
            if prefix is not None: raise ValueError("discovery target cannot include a session prefix")
            remote_home, display = "~/.zeta", host
        loop = asyncio.get_running_loop()
        done: asyncio.Future[tuple[bytes, bytes, int]] = loop.create_future()
        stdout, stderr = bytearray(), bytearray()
        def callback(item: TransportItem) -> None:
            def admit() -> None:
                if done.done(): return
                if item.stream == "stdout":
                    stdout.extend(item.data)
                    if len(stdout) > 1024 * 1024: done.set_exception(RemoteWireError("remote discovery is too large"))
                elif item.stream == "stderr": stderr.extend(item.data)
                elif item.stream == "exit": done.set_result((bytes(stdout), bytes(stderr), int(item.data or b"2")))
            loop.call_soon_threadsafe(admit)
        transport = self._transport_factory(host, ssh=self.ssh, callback=callback)
        transport.start(remote_bridge_command("list", remote_home=remote_home))
        try: out, diagnostics, status = await done
        finally: await transport.close_async()
        if status == 127 or b"command not found" in diagnostics or b"invalid choice: '__remote_bridge'" in diagnostics or b"unknown bridge_version" in diagnostics:
            raise RemoteWireError(COMPATIBILITY_MESSAGE.replace("HOST", display))
        rows: list[RemoteSession] = []
        for line in out.splitlines():
            if len(line) > 8192: raise RemoteWireError("remote session row is too large")
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or value.get("kind") != "session": continue
                allowed = {"kind", "session_id", "name", "state", "machine_id", "socket", "pid", "cwd", "provider", "model"}
                if any(key not in allowed for key in value): raise RemoteWireError("invalid remote session row")
                session_id = validate_host(str(value["session_id"]))
                rows.append(RemoteSession(display, session_id, str(value.get("name", "")), value.get("state"), str(value.get("machine_id", ""))))
            except RemoteWireError: raise
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError): continue
        return tuple(rows)

    async def attach(self, host: str, session: RemoteSession | str, *, replay_seq: int = 0) -> RemoteView:
        host, prefix = split_target(host)
        remote_home = "~/.zeta"
        if host in self._targets:
            host, remote_home = self._targets[host]
        session_id = session.session_id if isinstance(session, RemoteSession) else session
        validate_host(session_id)
        if prefix is not None and session_id != prefix: raise ValueError("attach requires a selected session")
        await self.detach()
        self._loop = asyncio.get_running_loop(); self._generation += 1; generation = self._generation
        self._queue = asyncio.Queue(maxsize=256); self._events = asyncio.Queue(maxsize=256)
        def callback(item: TransportItem) -> None:
            if self._loop is not None: self._loop.call_soon_threadsafe(self._admit, generation, item)
        transport = self._transport_factory(host, ssh=self.ssh, callback=callback)
        transport.start(remote_bridge_command("attach", session_id, remote_home))
        transport.send((json.dumps({"jsonrpc":"2.0","id":1,"method":"hello","params":{"protocol_version":"1.1","role":"observer","features":["session_replay_seq","serve_observers"]}}, separators=(",", ":")) + "\n").encode())
        transport.send((json.dumps({"jsonrpc":"2.0","id":2,"method":"session_history_and_subscribe","params":{"session_id":session_id}}, separators=(",", ":")) + "\n").encode())
        self.transport = transport; self.view = RemoteView(host, session_id, session.name if isinstance(session, RemoteSession) else session_id, replay_seq)
        self._event_task = asyncio.create_task(self._consume(generation))
        return self.view

    def _admit(self, generation: int, item: TransportItem) -> None:
        if generation != self._generation or self._queue is None: return
        try: self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._queue = None
            if self._events is not None: self._events.put_nowait("fatal: remote stream overflow")
            if self.transport is not None: asyncio.create_task(self.transport.close_async())

    async def _consume(self, generation: int) -> None:
        queue = self._queue
        if queue is None: return
        decoder = BoundedLineDecoder()
        diagnostics = bytearray()
        while generation == self._generation:
            item = await queue.get()
            if item.stream == "exit":
                if self._events is not None:
                    status = int(item.data or b"2")
                    if status == 127 or b"command not found" in diagnostics or b"invalid choice: '__remote_bridge'" in diagnostics or b"unknown bridge_version" in diagnostics:
                        await self._events.put("fatal: " + COMPATIBILITY_MESSAGE.replace("HOST", self.view.host if self.view else "HOST"))
                    else:
                        await self._events.put("disconnected")
                return
            if item.stream == "stderr":
                diagnostics.extend(item.data)
                continue
            if item.stream != "stdout": continue
            try:
                lines = decoder.feed(item.data)
                for line in lines:
                    frame = json.loads(line)
                    if isinstance(frame, dict) and frame.get("id") == 2 and isinstance(frame.get("result"), dict):
                        events = history_events(frame["result"])
                    elif isinstance(frame, dict) and frame.get("method") == "event":
                        if isinstance(frame.get("params"), dict) and frame["params"].get("event") == "usage": continue
                        events = (decode_event(line),)
                    else: continue
                    if self._events is not None:
                        for event in events: await self._events.put(event)
            except (ValueError, TypeError, RemoteWireError, json.JSONDecodeError) as exc:
                if self._events is not None: await self._events.put("fatal: " + str(exc))
                if self.transport is not None: await self.transport.close_async()
                return

    async def next_event(self) -> StreamEvent | str:
        if self._events is None: raise RuntimeError("no remote attachment")
        return await self._events.get()

    async def detach(self) -> None:
        self._generation += 1
        task, transport = self._event_task, self.transport
        self._event_task = self._queue = None; self._events = None; self.transport = self.view = None
        if task is not None: task.cancel(); await asyncio.gather(task, return_exceptions=True)
        if transport is not None: await transport.close_async()
