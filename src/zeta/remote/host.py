"""Observer-only serve endpoint owned by a foreground TUI."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ..runtime.foreground_events import ForegroundEventPublisher
from ..server.runtime import ServerRuntime
from ..server.server import ZetaServer
from .descriptor import Descriptor, publish, remove, socket_path


class ObserverHost:
    def __init__(
        self,
        *,
        home: Path,
        session_id: str,
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
        self.publisher = publisher
        self.name, self.provider, self.model = name, provider, model
        self.cwd = "~" + cwd[len(str(home)):] if cwd == str(home) or cwd.startswith(str(home) + "/") else cwd
        self.cwd = self.cwd[:256]
        self.socket = socket_path(home, session_id)
        self._server: ZetaServer | None = None

    async def start(self) -> None:
        server = ZetaServer(
            home=self.home,
            socket_path=self.socket,
            runtime=self.runtime,
            foreground_event_publisher=self.publisher,
            max_role="observer",
        )
        try:
            await server.start()
        except BaseException:
            await server.close()
            raise
        self._server = server
        publish(
            self.home,
            Descriptor(
                str(self.socket), self.session_id, os.getpid(), self.name,
                self.cwd, self.provider, self.model,
            ),
        )

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            await server.close()
        await asyncio.to_thread(remove, self.home, self.session_id)
