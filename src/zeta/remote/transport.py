"""off-loop ssh transport for remote observer streams."""
from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import shlex
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass

MAX_QUEUE = 256


def remote_bridge_command(mode: str, session_id: str | None = None, remote_home: str = "~/.zeta") -> str:
    if mode not in {"list", "attach"}:
        raise ValueError("invalid bridge mode")
    args = ["__remote_bridge", mode]
    if session_id is not None:
        if session_id != session_id.strip() or any(c in session_id for c in ":/;'\"\n\x00"):
            raise ValueError("invalid session id")
        args.append(session_id)
    bridge_args = " ".join(shlex.quote(arg) for arg in args)
    home = shlex.quote(remote_home)
    if remote_home == "~" or remote_home.startswith("~/"):
        # quoting "~" stops the remote shell from expanding it, so spell it as "$HOME".
        home = '"$HOME"' + shlex.quote(remote_home[1:])
    return (f'if command -v zeta >/dev/null 2>&1; then ZETA_HOME={home} exec zeta {bridge_args}; '
            f'elif [ -x "$HOME/.local/bin/zeta" ]; then ZETA_HOME={home} exec "$HOME/.local/bin/zeta" {bridge_args}; '
            "else exit 127; fi")


@dataclass(frozen=True, slots=True)
class TransportItem:
    stream: str
    data: bytes


class RemoteTransport:
    """run ssh and bridge io in a private thread and loop."""

    def __init__(self, host: str, *, ssh: str = "ssh", port: int | None = None, max_queue: int = MAX_QUEUE,
                 callback: Callable[[TransportItem], None] | None = None) -> None:
        from .discovery import validate_host
        validate_host(host)
        self.host, self.ssh, self.port = host, ssh, port
        self._input: queue.Queue[bytes | None] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._callback = callback

    def start(self, command: str) -> None:
        if self._thread is not None:
            raise RuntimeError("transport already started")
        self._thread = threading.Thread(target=self._run, args=(command,), daemon=True)
        self._thread.start()

    def send(self, data: bytes) -> None:
        self._input.put_nowait(data)

    async def close_async(self) -> None:
        self._callback = None
        self._stop.set()
        with contextlib.suppress(Exception):
            self._input.put_nowait(None)
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 5)

    def _publish(self, item: TransportItem) -> None:
        if self._callback is not None:
            self._callback(item)
        else:
            return

    def _run(self, command: str) -> None:
        asyncio.run(self._main(command))

    async def _main(self, command: str) -> None:
        try:
            # BatchMode: never prompt inside the TUI; ConnectTimeout: a sleeping host fails fast.
            process = await asyncio.create_subprocess_exec(
                self.ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                *(["-p", str(self.port)] if self.port else []), "--", self.host, command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            self._publish(TransportItem("stderr", str(exc).encode()))
            self._publish(TransportItem("exit", b"255"))
            return
        self._process = process

        async def read(name: str, stream: asyncio.StreamReader) -> None:
            while not self._stop.is_set():
                chunk = await stream.read(65536)
                if not chunk:
                    return
                self._publish(TransportItem(name, chunk))

        async def write() -> None:
            while not self._stop.is_set():
                data = await asyncio.to_thread(self._input.get)
                if data is None:
                    return
                if process.stdin is not None:
                    process.stdin.write(data)
                    await process.stdin.drain()

        readers = [asyncio.create_task(read("stdout", process.stdout)),
                   asyncio.create_task(read("stderr", process.stderr)),
                   asyncio.create_task(write())]
        waiter = asyncio.create_task(process.wait())
        stop_waiter = asyncio.create_task(asyncio.to_thread(self._stop.wait))
        done, _ = await asyncio.wait({waiter, *readers, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
        if process.returncode is None and (self._stop.is_set() or any(task in done for task in readers)):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        await process.wait()
        stop_waiter.cancel()
        # process.wait() can return before the pipes close; let the readers drain briefly.
        await asyncio.wait(readers[:2], timeout=1)
        for task in readers:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        self._publish(TransportItem("exit", str(process.returncode).encode()))
