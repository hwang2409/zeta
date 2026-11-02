"""Authenticated WebSocket transport adapter for the serve protocol."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hmac
import json
import logging
import os
import secrets
import signal
import stat
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import MultipleValuesError
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from .protocol import MAX_FRAME_BYTES

if TYPE_CHECKING:
    from .server import ZetaServer

_UPGRADE_TIMEOUT = 5
_LOGGER = logging.getLogger("zeta.server.websocket")
_LOGGER.disabled = True


def validate_origins(origins: Sequence[str]) -> tuple[str, ...]:
    validated: list[str] = []
    for origin in origins:
        try:
            parsed = urlsplit(origin)
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid WebSocket origin: {origin!r}") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or origin != f"{parsed.scheme}://{parsed.netloc}"
        ):
            raise ValueError(
                f"WebSocket origin must be an exact scheme://host:port: {origin!r}"
            )
        if origin not in validated:
            validated.append(origin)
    return tuple(validated)


class WebSocketServer:
    """Own the secure loopback listener and its connection-discovery file."""

    def __init__(
        self,
        *,
        port: int,
        origins: Sequence[str],
        path: Path,
        accept: Callable[[object, object], Awaitable[None]],
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("ws_port must be between 0 and 65535")
        self.port = port
        self.origins = validate_origins(origins)
        self.path = path
        self.accept = accept
        self.token: str | None = None
        self.server: Server | None = None
        self._lock_fd: int | None = None

    async def start(self) -> None:
        self._lock_fd = await asyncio.to_thread(
            acquire_connection_lock, self.path.with_name("serve.ws.lock")
        )
        self.token = secrets.token_urlsafe(32)
        self.server = await serve(
            self._handle,
            "127.0.0.1",
            self.port,
            process_request=self._authorize,
            compression=None,
            open_timeout=_UPGRADE_TIMEOUT,
            close_timeout=1,
            max_size=MAX_FRAME_BYTES - 1,
            logger=_LOGGER,
        )
        if self.server.sockets:
            self.port = int(self.server.sockets[0].getsockname()[1])
        await asyncio.to_thread(
            write_connection_file,
            self.path,
            port=self.port,
            token=self.token,
            origins=self.origins,
        )
        await asyncio.to_thread(
            print,
            "zeta serve: websocket ready: "
            f"ws://127.0.0.1:{self.port}/?token={self.token}",
            file=sys.stderr,
        )

    def _authorize(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        supplied = parse_qs(urlsplit(request.path).query, keep_blank_values=True).get(
            "token", []
        )
        if (
            self.token is None
            or len(supplied) != 1
            or not hmac.compare_digest(supplied[0], self.token)
        ):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        try:
            origin = request.headers.get("Origin")
        except MultipleValuesError:
            return connection.respond(HTTPStatus.BAD_REQUEST, "Bad Request\n")
        if origin is not None and origin not in self.origins:
            return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
        return None

    async def _handle(self, connection: ServerConnection) -> None:
        adapter = WebSocketConnection(connection)
        await self.accept(adapter, adapter)

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if self._lock_fd is not None:
            await asyncio.to_thread(
                release_connection_lock,
                self.path,
                token=self.token,
                lock_fd=self._lock_fd,
            )
            self._lock_fd = None
            self.token = None


class WebSocketConnection:
    """Expose WebSocket messages as the stream interface used by ``_Client``."""

    def __init__(self, connection: ServerConnection) -> None:
        self._connection = connection
        self._pending: bytes | None = None
        self._close_task: asyncio.Task[None] | None = None
        self.transport = connection.transport

    async def read(self, _size: int) -> bytes:
        try:
            message = await self._connection.recv()
        except ConnectionClosed:
            return b""
        if isinstance(message, bytes):
            await self._connection.close(1003)
            return b""
        return message.encode("utf-8") + b"\n"

    def write(self, payload: bytes) -> None:
        if self._pending is not None:
            raise RuntimeError("WebSocket write already pending")
        self._pending = payload.removesuffix(b"\n")

    async def drain(self) -> None:
        if self._pending is None:
            return
        payload, self._pending = self._pending, None
        await self._connection.send(payload, text=True)

    def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._connection.close())

    async def wait_closed(self) -> None:
        if self._close_task is not None:
            with contextlib.suppress(ConnectionClosed):
                await self._close_task
        await self._connection.wait_closed()


async def run_server(server: ZetaServer) -> None:
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    await server.start()
    serving = asyncio.create_task(server.serve_forever())
    try:
        await stopped.wait()
    finally:
        await server.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        for signum in installed:
            loop.remove_signal_handler(signum)


def acquire_connection_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fd = -1
    try:
        os.fchmod(parent_fd, 0o700)
        fd = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=parent_fd,
        )
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"refusing unsafe WebSocket lock file: {path}")
        os.fchmod(fd, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            pid = ""
            for _ in range(10):
                pid = os.pread(fd, 64, 0).decode(errors="replace").strip()
                if pid:
                    break
                time.sleep(0.01)
            raise ValueError(
                "WebSocket server already running for this ZETA_HOME "
                f"(pid {pid or 'unknown'}); stop it before starting another"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        result, fd = fd, -1
        return result
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def release_connection_lock(path: Path, *, token: str | None, lock_fd: int) -> None:
    try:
        if token is not None:
            remove_connection_file(path, token=token)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def write_connection_file(
    path: Path, *, port: int, token: str, origins: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
    payload = json.dumps(
        {"port": port, "token": token, "pid": os.getpid(), "origins": list(origins)},
        separators=(",", ":"),
    ).encode()
    fd = -1
    try:
        os.fchmod(parent_fd, 0o700)
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=parent_fd)
        os.close(parent_fd)


def remove_connection_file(path: Path, *, token: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return
        with os.fdopen(fd) as handle:
            fd = -1
            value = json.load(handle)
        if value.get("token") == token:
            path.unlink()
    except (OSError, ValueError, AttributeError):
        return
    finally:
        if fd >= 0:
            os.close(fd)


__all__ = ["WebSocketServer", "run_server"]
