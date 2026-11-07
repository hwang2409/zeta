"""Own a spawned process until the background registry can accept it."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .tool_spawn import create_tool_subprocess_shell

logger = logging.getLogger(__name__)


async def create_subprocess_shell_in_fd(
    command: str, directory_fd: int, **kwargs: object
) -> asyncio.subprocess.Process:
    """Run a shell in an inherited, identity-verified directory."""

    kwargs.pop("cwd", None)
    return await create_tool_subprocess_shell(
        command, cwd_fd=directory_fd, **kwargs
    )


def _group_exists(process_id: int) -> bool:
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _parse_process_table(stdout: bytes) -> dict[int, list[tuple[int, str]]]:
    children: dict[int, list[tuple[int, str]]] = {}
    for line in stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            pid, ppid = int(fields[0]), int(fields[1])
            children.setdefault(ppid, []).append((pid, fields[2]))
    return children


def _descendant_pids_from_table(
    children: dict[int, list[tuple[int, str]]], root_pid: int
) -> dict[int, str]:
    descendants: dict[int, str] = {}
    pending = [root_pid]
    while pending:
        for pid, start in children.get(pending.pop(), ()):
            if pid not in descendants:
                descendants[pid] = start
                pending.append(pid)
    return descendants


def _snapshot_descendants_sync(
    root_pids: Sequence[int] | None,
) -> dict[int, dict[int, str]] | dict[int, str]:
    """Take one process-table snapshot for termination and identity checks."""
    if root_pids == () or root_pids == []:
        return {}
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "lstart="],
            timeout=2,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {} if root_pids is None else {pid: {} for pid in root_pids}
    if result.returncode != 0:
        return {} if root_pids is None else {pid: {} for pid in root_pids}
    children = _parse_process_table(result.stdout)
    if root_pids is None:
        return {
            pid: start
            for entries in children.values()
            for pid, start in entries
        }
    return {pid: _descendant_pids_from_table(children, pid) for pid in root_pids}


def _signal_pids(pids: dict[int, str], signal_number: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal_number)
        except ProcessLookupError:
            pass
        except OSError:
            logger.debug("could not signal background descendant %s", pid, exc_info=True)


async def _terminate_processes(
    processes: Sequence[asyncio.subprocess.Process],
    term_grace: float,
    *,
    finish: Callable[[Awaitable[Any]], Awaitable[Any]] | None = None,
) -> None:
    """Terminate processes and descendants without trusting reused pids."""
    if not processes:
        return
    snapshot = finish or (lambda awaitable: awaitable)
    descendants = await snapshot(
        asyncio.to_thread(
            _snapshot_descendants_sync, [process.pid for process in processes]
        )
    ) or {}
    for process in processes:
        _signal_group(process, signal.SIGTERM)
        _signal_pids(descendants.get(process.pid, {}), signal.SIGTERM)
    try:
        await snapshot(
            asyncio.wait_for(
                asyncio.gather(*(process.wait() for process in processes)),
                timeout=term_grace,
            )
        )
    except TimeoutError:
        pass
    current = await snapshot(asyncio.to_thread(_snapshot_descendants_sync, None))
    for process in processes:
        _signal_group(process, signal.SIGKILL)
        survivors = (
            descendants.get(process.pid, {})
            if current is None
            else {
                pid: start
                for pid, start in current.items()
                if descendants.get(process.pid, {}).get(pid) == start
            }
        )
        _signal_pids(survivors, signal.SIGKILL)


async def _terminate_process(
    process: asyncio.subprocess.Process,
    term_grace: float,
    close_stdin: Callable[[], Awaitable[object]] | None = None,
) -> None:
    cancellation = _DeferredCancellation()
    if close_stdin is not None:
        await cancellation.finish(close_stdin())
    await _terminate_processes(
        (process,), term_grace, finish=cancellation.finish
    )
    if process.returncode is None:
        await cancellation.finish(process.wait())
    if cancellation.cancelled:
        raise asyncio.CancelledError


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


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    process_tasks: Sequence[asyncio.Task[object]],
) -> None:
    # A descendant that calls setsid creates a new session and can escape this group.
    _signal_group(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=0.1)
    except TimeoutError:
        pass
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
    _signal_group(process, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()
    for task in process_tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*process_tasks, return_exceptions=True)
