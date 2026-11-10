"""Observer-only unix endpoint owned by a foreground TUI."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import errno
import json
import os
import socket
import stat
from pathlib import Path

from ..runtime.foreground_events import ForegroundEventPublisher
from .descriptor import Descriptor, DescriptorError, publish, remove, socket_path


class ObserverHost:
    def __init__(self, *, home: Path, session_id: str, publisher: ForegroundEventPublisher, name: str = "", cwd: str = "", provider: str = "", model: str = "") -> None:
        self.home = home
        self.session_id = session_id
        self.publisher = publisher
        self.name, self.provider, self.model = name, provider, model
        self.cwd = "~" + cwd[len(str(home)):] if cwd == str(home) or cwd.startswith(str(home) + "/") else cwd
        self.cwd = self.cwd[:256]
        self.socket = socket_path(home, session_id)
        self._server: asyncio.AbstractServer | None = None
        self._subscription = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self._prepare_socket)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self.socket))
            os.chmod(self.socket, 0o600)
            self._server = await asyncio.start_unix_server(self._accept, sock=sock)
        except BaseException:
            sock.close()
            with contextlib.suppress(FileNotFoundError):
                self.socket.unlink()
            raise
        self._subscription = self.publisher.subscribe()
        self._task = asyncio.create_task(self._publish_events())
        publish(self.home, Descriptor(str(self.socket), self.session_id, os.getpid(), self.name, self.cwd, self.provider, self.model))

    async def _publish_events(self) -> None:
        assert self._subscription is not None
        async for event in self._subscription:
            payload = json.dumps({"method": "event", "params": dataclasses.asdict(event)}, separators=(",", ":"), default=str).encode() + b"\n"
            for writer in tuple(self._clients):
                try:
                    writer.write(payload)
                    await writer.drain()
                except (ConnectionError, OSError):
                    self._clients.discard(writer)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients.add(writer)
        try:
            while True:
                line = await reader.readline()
                if not line or len(line) > 65536:
                    return
                try:
                    request = json.loads(line)
                    method = request.get("method")
                    request_id = request.get("id")
                    if method == "hello":
                        response = {"id": request_id, "result": {"role": "observer", "session_id": self.session_id, "capabilities": {"features": ["serve_observers"]}}}
                    elif method == "status":
                        response = {"id": request_id, "result": {"session": {"status": "running" if self._task and not self._task.done() else "idle", "session_id": self.session_id}}}
                    else:
                        response = {"id": request_id, "error": {"code": -32601, "message": "observer method unavailable"}}
                    writer.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                    await writer.drain()
                except (ValueError, TypeError):
                    return
        finally:
            self._clients.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def close(self) -> None:
        if self._subscription is not None:
            self.publisher.unsubscribe(self._subscription)
            self._subscription = None
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        for writer in tuple(self._clients):
            writer.close()
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await asyncio.to_thread(self._unlink_socket)
        await asyncio.to_thread(remove, self.home, self.session_id)

    def _prepare_socket(self) -> None:
        self.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.socket.parent, 0o700)
        try:
            info = self.socket.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode):
            raise DescriptorError("refusing to replace non-socket observer path")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(self.socket))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise DescriptorError("observer socket is active") from exc
        else:
            raise DescriptorError("observer socket is active")
        finally:
            probe.close()
        self._unlink_socket(info)

    def _unlink_socket(self, identity=None) -> None:
        try:
            info = self.socket.lstat()
        except FileNotFoundError:
            return
        if identity is None or (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            self.socket.unlink(missing_ok=True)
