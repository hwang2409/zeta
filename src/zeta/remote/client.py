"""Deep client for remote observer discovery and replay attachment."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..protocol.types import StreamEvent
from .discovery import RemoteSession, configured_hosts, dedupe_rows, split_target
from .transport import RemoteTransport, remote_bridge_command
from .wire import decode_event

COMPATIBILITY_MESSAGE = "remote zeta is missing or too old. update ~/me/fun/zeta on HOST, then run: cd ~/me/fun/zeta && uv tool install --editable ."

@dataclass(frozen=True, slots=True)
class RemoteView:
    host: str
    session_id: str
    session_name: str
    replay_seq: int = 0

class RemoteSessionClient:
    def __init__(self, home: Path, *, ssh: str = "ssh", transport_factory: Callable[..., RemoteTransport] = RemoteTransport) -> None:
        self.home = home
        self.ssh = ssh
        self._transport_factory = transport_factory
        self.transport: RemoteTransport | None = None
        self.view: RemoteView | None = None
        self._provisional = False

    async def discover(self, hosts: tuple[str, ...] | None = None) -> tuple[RemoteSession, ...]:
        targets = hosts or configured_hosts(self.home)
        rows: list[RemoteSession] = []
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._list(host)) for host in targets]
        for task in tasks:
            rows.extend(task.result())
        return dedupe_rows(rows)

    async def _list(self, host: str) -> tuple[RemoteSession, ...]:
        validate = split_target(host)[0]
        transport = self._transport_factory(validate, ssh=self.ssh)
        transport.start(remote_bridge_command("list"))
        chunks: list[bytes] = []
        while True:
            item = await asyncio.to_thread(transport.receive, 0.1)
            if item is None:
                continue
            if item.stream == "stderr":
                continue
            if item.stream == "exit":
                break
            chunks.append(item.data)
        transport.close()
        rows = []
        for line in b"".join(chunks).splitlines():
            try: value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError): continue
            if value.get("kind") != "session": continue
            rows.append(RemoteSession(validate, str(value["session_id"]), str(value.get("name", "")), value.get("state"), str(value.get("machine_id", ""))))
        return tuple(rows)

    async def attach(self, host: str, session: RemoteSession | str, *, replay_seq: int = 0) -> RemoteView:
        host, prefix = split_target(host)
        session_id = session.session_id if isinstance(session, RemoteSession) else session
        if prefix is not None and session_id != prefix:
            raise ValueError("attach requires a selected session")
        self.detach()
        transport = self._transport_factory(host, ssh=self.ssh)
        transport.start(remote_bridge_command("attach", session_id))
        self.transport = transport
        self.view = RemoteView(host, session_id, session.name if isinstance(session, RemoteSession) else session_id, replay_seq)
        return self.view

    def detach(self) -> None:
        if self.transport is not None:
            self.transport.close()
        self.transport = None
        self.view = None
        self._provisional = False

    def consume(self) -> tuple[StreamEvent, ...]:
        if self.transport is None:
            return ()
        events: list[StreamEvent] = []
        while True:
            item = self.transport.receive()
            if item is None: break
            if item.stream == "stdout":
                events.extend(decode_event(line) for line in item.data.splitlines() if line)
        return tuple(events)
