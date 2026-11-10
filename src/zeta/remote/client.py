"""deep client for remote observer discovery and replay attachment."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..protocol.types import StreamEvent
from .discovery import RemoteSession, configured_hosts, dedupe_rows, split_target
from .transport import RemoteTransport, TransportItem, remote_bridge_command
from .wire import RemoteWireError, decode_event, history_events

COMPATIBILITY_MESSAGE = "remote zeta is missing or too old. update ~/me/fun/zeta on HOST, then run: cd ~/me/fun/zeta && uv tool install --editable ."


@dataclass(frozen=True, slots=True)
class RemoteView:
    host: str
    session_id: str
    session_name: str
    replay_seq: int = 0


class RemoteSessionClient:
    def __init__(self, home: Path, *, ssh: str = "ssh",
                 transport_factory: Callable[..., RemoteTransport] = RemoteTransport) -> None:
        self.home = home
        self.ssh = ssh
        self._transport_factory = transport_factory
        self.transport: RemoteTransport | None = None
        self.view: RemoteView | None = None
        self._queue: asyncio.Queue[TransportItem] | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events: asyncio.Queue[StreamEvent | str] = asyncio.Queue(maxsize=256)

    async def discover(self, hosts: tuple[str, ...] | None = None) -> tuple[RemoteSession, ...]:
        targets = hosts or configured_hosts(self.home)
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._list(host)) for host in targets]
        rows: list[RemoteSession] = []
        for task in tasks:
            rows.extend(task.result())
        return dedupe_rows(rows)

    async def _list(self, target: str) -> tuple[RemoteSession, ...]:
        host, prefix = split_target(target)
        if prefix is not None:
            raise ValueError("discovery target cannot include a session prefix")
        loop = asyncio.get_running_loop()
        done: asyncio.Future[tuple[bytes, bytes, int]] = loop.create_future()
        stdout = bytearray()
        stderr = bytearray()

        def callback(item: TransportItem) -> None:
            def admit() -> None:
                if done.done():
                    return
                if item.stream == "stdout":
                    stdout.extend(item.data)
                    if len(stdout) > 1024 * 1024:
                        done.set_exception(RemoteWireError("remote discovery is too large"))
                elif item.stream == "stderr":
                    stderr.extend(item.data)
                elif item.stream == "exit":
                    done.set_result((bytes(stdout), bytes(stderr), int(item.data or b"2")))
            loop.call_soon_threadsafe(admit)

        transport = self._transport_factory(host, ssh=self.ssh, callback=callback)
        transport.start(remote_bridge_command("list"))
        try:
            out, diagnostics, _ = await done
        finally:
            await transport.close_async()
        if any(token in diagnostics for token in (b"command not found", b"invalid choice")):
            raise RemoteWireError(COMPATIBILITY_MESSAGE.replace("HOST", host))
        rows: list[RemoteSession] = []
        for line in out.splitlines():
            try:
                value = json.loads(line)
                if value.get("kind") != "session":
                    continue
                rows.append(RemoteSession(host, str(value["session_id"]), str(value.get("name", "")), value.get("state"), str(value.get("machine_id", ""))))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                continue
        return tuple(rows)

    async def attach(self, host: str, session: RemoteSession | str, *, replay_seq: int = 0) -> RemoteView:
        host, prefix = split_target(host)
        session_id = session.session_id if isinstance(session, RemoteSession) else session
        if prefix is not None and session_id != prefix:
            raise ValueError("attach requires a selected session")
        await self.detach()
        self._loop = asyncio.get_running_loop()
        self._generation += 1
        generation = self._generation
        self._queue = asyncio.Queue(maxsize=256)

        def callback(item: TransportItem) -> None:
            loop = self._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._admit, generation, item)

        transport = self._transport_factory(host, ssh=self.ssh, callback=callback)
        transport.start(remote_bridge_command("attach", session_id))
        hello = {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol_version": "1.1", "role": "observer", "features": ["session_replay_seq", "serve_observers"]}}
        join = {"jsonrpc": "2.0", "id": 2, "method": "session_history_and_subscribe", "params": {"session_id": session_id}}
        transport.send((json.dumps(hello, separators=(",", ":")) + "\n").encode())
        transport.send((json.dumps(join, separators=(",", ":")) + "\n").encode())
        self.transport = transport
        self.view = RemoteView(host, session_id, session.name if isinstance(session, RemoteSession) else session_id, replay_seq)
        self._event_task = asyncio.create_task(self._consume(generation))
        return self.view

    def _admit(self, generation: int, item: TransportItem) -> None:
        if generation != self._generation or self._queue is None:
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._queue = None
            self._events.put_nowait("remote stream overflow")

    async def _consume(self, generation: int) -> None:
        queue = self._queue
        if queue is None:
            return
        decoder = bytearray()
        while generation == self._generation:
            item = await queue.get()
            if item.stream == "exit":
                await self._events.put("disconnected")
                return
            if item.stream != "stdout":
                continue
            decoder.extend(item.data)
            while b"\n" in decoder:
                line, _, rest = decoder.partition(b"\n")
                decoder[:] = rest
                if line:
                    try:
                        frame = json.loads(line)
                        if isinstance(frame, dict) and isinstance(frame.get("result"), dict):
                            for event in history_events(frame["result"]):
                                await self._events.put(event)
                        elif isinstance(frame, dict) and frame.get("method") == "event":
                            await self._events.put(decode_event(line))
                    except (ValueError, TypeError, RemoteWireError) as exc:
                        await self._events.put("fatal: " + str(exc))
                        return

    async def next_event(self) -> StreamEvent | str:
        return await self._events.get()

    async def detach(self) -> None:
        self._generation += 1
        task, transport = self._event_task, self.transport
        self._event_task = None
        self._queue = None
        self.transport = None
        self.view = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if transport is not None:
            await transport.close_async()
