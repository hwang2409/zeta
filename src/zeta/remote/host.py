"""Observer-only serve endpoint owned by a foreground TUI."""
from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from pathlib import Path
from typing import Any

from ..runtime.foreground_events import ForegroundEventPublisher
from ..server.runtime import ServerRuntime
from ..server.server import ZetaServer
from .descriptor import (
    Descriptor,
    create_observer_socket,
    publish,
    remove,
    socket_path,
    unlink_observer_socket,
)


class ObserverHost:
    def __init__(
        self,
        *,
        home: Path,
        session_id: str,
        opened: Any | None = None,
        loop: Any | None = None,
        policy: Any | None = None,
        project_dir: Path | None = None,
        repo_root: Path | None = None,
        runtime: ServerRuntime | None = None,
        publisher: ForegroundEventPublisher,
        name: str = "",
        cwd: str = "",
        provider: str = "",
        model: str = "",
    ) -> None:
        self.home = home
        self.session_id = session_id
        self.runtime = runtime
        self._runtime_args = (opened, loop, policy, project_dir, repo_root)
        self.publisher = publisher
        self.name, self.provider, self.model = name, provider, model
        self.cwd = "~" + cwd[len(str(home)):] if cwd == str(home) or cwd.startswith(str(home) + "/") else cwd
        self.cwd = self.cwd[:256]
        self.socket = socket_path(home, session_id)
        self._server: ZetaServer | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._started = False

    def _prepare(self) -> tuple[socket.socket, tuple[int, int]]:
        # Off the TUI loop: the borrowed runtime opens a SessionManager and the socket setup probes.
        if self.runtime is None:
            opened, loop, policy, project_dir, repo_root = self._runtime_args
            if not all(value is not None for value in (opened, loop, policy, project_dir, repo_root)):
                raise RuntimeError("observer host requires a borrowed runtime")
            self.runtime = ServerRuntime.borrowed(
                home=self.home,
                opened=opened,
                loop=loop,
                policy=policy,
                project_dir=project_dir,
                repo_root=repo_root,
            )
        return create_observer_socket(self.socket)

    async def start(self) -> None:
        if self._started:
            return
        observer_socket, socket_identity = await asyncio.to_thread(self._prepare)
        self._socket_identity = socket_identity
        server: ZetaServer | None = None
        try:
            server = ZetaServer(
                home=self.home,
                socket_path=self.socket,
                runtime=self.runtime,
                foreground_event_publisher=self.publisher,
                max_role="observer",
                prebound_socket=observer_socket,
            )
            await server.start()
        except BaseException:
            if server is not None:
                await server.close()
            observer_socket.close()
            await asyncio.to_thread(unlink_observer_socket, self.socket, socket_identity)
            self._socket_identity = None
            raise
        self._server = server
        self._started = True
        await asyncio.to_thread(publish, self.home, Descriptor(str(self.socket), self.session_id, os.getpid(), self.name, self.cwd, self.provider, self.model))

    async def start_or_close(self) -> str | None:
        """Start, or close whatever started and return the reason; the endpoint is optional."""
        try:
            await self.start()
        except Exception as exc:  # noqa: BLE001 - any start failure becomes a TUI notice
            with contextlib.suppress(Exception):
                await self.close()
            return str(exc) or type(exc).__name__
        return None

    async def close(self) -> None:
        server, self._server = self._server, None
        self._started = False
        if server is not None:
            await server.close()
        if self._socket_identity is not None:
            await asyncio.to_thread(unlink_observer_socket, self.socket, self._socket_identity)
            self._socket_identity = None
        await asyncio.to_thread(remove, self.home, self.session_id)
