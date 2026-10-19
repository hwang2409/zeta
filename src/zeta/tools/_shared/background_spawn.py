"""Own a spawned process until the background registry can accept it."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any

_FD_SHELL_WRAPPER = (
    "import os,sys; os.fchdir(int(sys.argv[1])); "
    "os.execvpe('/bin/sh', ['sh','-c',sys.argv[2]], os.environ)"
)


async def create_subprocess_shell_in_fd(
    command: str, directory_fd: int, **kwargs: object
) -> asyncio.subprocess.Process:
    """Run a shell after fchdir'ing an inherited, identity-verified fd."""

    kwargs.pop("cwd", None)
    pass_fds = tuple(kwargs.pop("pass_fds", ()))
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _FD_SHELL_WRAPPER,
        str(directory_fd),
        command,
        pass_fds=(*pass_fds, directory_fd),
        **kwargs,
    )


def _group_exists(process_id: int) -> bool:
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _signal_group(process: asyncio.subprocess.Process, signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass
    except OSError:
        if process.returncode is None:
            process.kill()


class _DeferredCancellation:
    """Record caller cancellation without canceling work that still owns a child."""

    def __init__(self) -> None:
        self.cancelled = False

    async def finish(self, awaitable: Awaitable[Any]) -> Any:
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                self.cancelled = True
        return task.result()


class BackgroundSpawnCleanupError(RuntimeError):
    """An unregistered process could not be confirmed terminated and reaped."""


async def spawn_background(
    spawn: Awaitable[asyncio.subprocess.Process],
    *,
    require_open: Callable[[], None],
    term_grace: float,
    cleanup_timeout: float,
) -> asyncio.subprocess.Process:
    """Finish creation even when the caller cancels, then accept or reap it."""

    cancellation = _DeferredCancellation()
    process = await cancellation.finish(spawn)
    try:
        if cancellation.cancelled:
            raise asyncio.CancelledError
        require_open()
    except BaseException:
        # Keep ownership through repeated caller cancellation. Event-loop
        # teardown can cancel the cleanup task itself; retry its idempotent
        # cleanup rather than releasing the registry's pending-start token.
        cleanup = _DeferredCancellation()
        while True:
            try:
                await cleanup.finish(
                    _terminate_unregistered(process, term_grace, cleanup_timeout)
                )
                break
            except asyncio.CancelledError:
                cleanup.cancelled = True
            except Exception as exc:
                raise BackgroundSpawnCleanupError(
                    f"could not clean up background process group {process.pid}"
                ) from exc
        if cleanup.cancelled:
            raise asyncio.CancelledError
        raise
    return process


async def _terminate_unregistered(
    process: asyncio.subprocess.Process, term_grace: float, cleanup_timeout: float,
) -> None:
    if process.stdin is not None:
        process.stdin.close()

    async def drain(reader: asyncio.StreamReader | None) -> None:
        if reader is not None:
            while await reader.read(65_536):
                pass

    readers = [
        asyncio.create_task(drain(reader))
        for reader in (process.stdout, process.stderr)
        if reader is not None
    ]
    try:
        _signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=term_grace)
        except TimeoutError:
            pass
        if _group_exists(process.pid):
            _signal_group(process, signal.SIGKILL)
        await asyncio.wait_for(
            asyncio.gather(process.wait(), *readers), timeout=cleanup_timeout,
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + cleanup_timeout
        while _group_exists(process.pid):
            if loop.time() >= deadline:
                raise RuntimeError("background process group survived canceled start")
            await asyncio.sleep(0.01)
    finally:
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
