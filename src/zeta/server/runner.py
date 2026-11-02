"""Signal-aware lifecycle for the local frontend server."""

from __future__ import annotations

import asyncio
import signal

from .server import ZetaServer


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


__all__ = ["run_server"]
