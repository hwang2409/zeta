"""Starts in flight remain owned until registration or complete cleanup."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from zeta.tools._shared import background_spawn
from zeta.tools._shared import process as process_module
from zeta.tools._shared.process import BackgroundTaskRegistry, _group_exists


@pytest.mark.parametrize("interruption", [None, "spawn", "cleanup", "cleanup_failure"])
async def test_shutdown_joins_inflight_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str | None,
) -> None:
    registry = BackgroundTaskRegistry(term_grace=5 if interruption == "cleanup" else 0.03)
    spawn_ready = asyncio.Event()
    release_spawn = asyncio.Event()
    shutdown_started = asyncio.Event()
    term_sent = asyncio.Event()
    release_wait = asyncio.Event()
    spawned: list[asyncio.subprocess.Process] = []
    use_fd = interruption == "spawn"
    spawn = (
        process_module.create_subprocess_shell_in_fd
        if use_fd else asyncio.create_subprocess_shell
    )
    close = registry._shutdown

    async def paused_spawn(*args, **kwargs):
        process = await spawn(*args, **kwargs)
        spawned.append(process)
        assert await process.stdout.readline() == b"ready\n"
        spawn_ready.set()
        await release_spawn.wait()
        return process

    async def shutdown():
        shutdown_started.set()
        return await close()

    async def failed_cleanup(*args):
        raise OSError("injected terminate failure")

    if interruption == "cleanup_failure":
        monkeypatch.setattr(background_spawn, "_terminate_unregistered", failed_cleanup)
    if use_fd:
        monkeypatch.setattr(process_module, "create_subprocess_shell_in_fd", paused_spawn)
    else:
        monkeypatch.setattr(asyncio, "create_subprocess_shell", paused_spawn)
    monkeypatch.setattr(registry, "_shutdown", shutdown)
    command = "exec " + shlex.join((sys.executable, "-c", (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(30)"
    )))
    cwd_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY) if use_fd else None
    starting = asyncio.create_task(registry.start(command, tmp_path, cwd_fd=cwd_fd))
    closing = None
    try:
        async with asyncio.timeout(5):
            await spawn_ready.wait()
            process = spawned[0]
            original_wait = process.wait

            async def paused_wait():
                if interruption == "cleanup" and not term_sent.is_set():
                    term_sent.set()
                    await release_wait.wait()
                    # Trigger escalation without depending on a wall-clock race.
                    raise TimeoutError
                return await original_wait()

            monkeypatch.setattr(process, "wait", paused_wait)
            closing = asyncio.create_task(registry.close())
            await shutdown_started.wait()
            if interruption == "spawn":
                starting.cancel()
                await asyncio.sleep(0)
            release_spawn.set()
            if interruption == "cleanup":
                await term_sent.wait()
                starting.cancel()
                closing.cancel()
                await asyncio.sleep(0)
                starting.cancel()
                await asyncio.sleep(0)
                assert not starting.done()
                assert not closing.done()
                assert registry._directory_fd is not None
                assert _group_exists(process.pid)
                release_wait.set()
            result, close_result = await asyncio.gather(
                starting, closing, return_exceptions=True,
            )
            if interruption == "cleanup_failure":
                assert isinstance(result, background_spawn.BackgroundSpawnCleanupError)
                assert close_result is result
                assert not registry._closed
                return
            if interruption is None:
                assert isinstance(result, RuntimeError)
                assert "registry is closing" in str(result)
            else:
                assert isinstance(result, asyncio.CancelledError)
            if interruption == "cleanup":
                assert isinstance(close_result, asyncio.CancelledError)
            else:
                assert close_result == ()
            assert process.returncode == -signal.SIGKILL
            assert not _group_exists(process.pid)
            assert registry._closed
            assert registry.records == ()
            assert registry._directory_fd is None
            if cwd_fd is not None:
                with pytest.raises(OSError):
                    os.fstat(cwd_fd)
    finally:
        release_spawn.set()
        release_wait.set()
        for process in spawned:
            if _group_exists(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
        starting.cancel()
        await asyncio.gather(starting, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        monitors = [record.monitor for record in registry.records if record.monitor is not None]
        await asyncio.gather(*monitors, return_exceptions=True)
        for process in spawned:
            await process.communicate()
        await asyncio.gather(registry.close(), return_exceptions=True)
