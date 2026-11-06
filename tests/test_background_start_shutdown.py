"""Starts in flight remain owned until registration or complete cleanup."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
import threading
from pathlib import Path

import pytest

from zeta.tools._shared import background_spawn
from zeta.tools._shared import process as process_module
from zeta.tools._shared.process import (
    BackgroundTaskNotice,
    BackgroundTaskRegistry,
    _group_exists,
)


@pytest.mark.asyncio
async def test_start_notice_follows_off_loop_durable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[BackgroundTaskNotice] = []
    registry = BackgroundTaskRegistry(
        term_grace=0.03,
        notice_sink=lambda notice: notices.append(notice),  # type: ignore[arg-type]
    )
    entered = threading.Event()
    release = threading.Event()
    observations: list[bool] = []
    writer_threads: list[int] = []
    original_persist = registry._persist

    def gated_persist(rows=None) -> None:
        writer_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10)

    def observe_before_commit() -> None:
        assert entered.wait(10)
        observations.append(notices == [])
        release.set()

    monkeypatch.setattr(registry, "_persist", gated_persist)
    observer = threading.Thread(target=observe_before_commit)
    observer.start()
    task_id, _ = await registry.start("sleep 30", tmp_path)
    observer.join()

    assert observations == [True]
    assert writer_threads != [threading.get_ident()]
    assert [notice.phase for notice in notices] == ["started"]
    monkeypatch.setattr(registry, "_persist", original_persist)
    await registry.kill(task_id)
    await registry.close()


@pytest.mark.asyncio
async def test_start_persistence_failure_reaps_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[BackgroundTaskNotice] = []
    registry = BackgroundTaskRegistry(
        term_grace=0.03,
        notice_sink=lambda notice: notices.append(notice),  # type: ignore[arg-type]
    )
    spawned_pid: int | None = None
    original_spawn = process_module.spawn_background

    async def capture_spawn(*args, **kwargs):
        nonlocal spawned_pid
        process = await original_spawn(*args, **kwargs)
        spawned_pid = process.pid
        return process

    monkeypatch.setattr(process_module, "spawn_background", capture_spawn)
    original_persist = registry._persist
    monkeypatch.setattr(
        registry, "_persist", lambda rows: (_ for _ in ()).throw(OSError("disk full"))
    )

    with pytest.raises(OSError, match="disk full"):
        await registry.start("sleep 30", tmp_path)

    assert spawned_pid is not None
    assert not _group_exists(spawned_pid)
    assert registry.running_count == 0
    assert registry.records == ()
    assert notices == []
    monkeypatch.setattr(registry, "_persist", original_persist)
    await registry.close()


@pytest.mark.asyncio
async def test_terminal_notice_follows_durable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[BackgroundTaskNotice] = []
    registry = BackgroundTaskRegistry(
        term_grace=0.03,
        notice_sink=lambda notice: notices.append(notice),  # type: ignore[arg-type]
    )
    task_id, _ = await registry.start("sleep 30", tmp_path)
    notices.clear()
    original_persist = registry._persist
    monkeypatch.setattr(
        registry, "_persist", lambda rows: (_ for _ in ()).throw(OSError("disk full"))
    )

    with pytest.raises(OSError, match="disk full"):
        await registry.kill(task_id)

    assert not registry.snapshot()[0].running
    assert notices == []
    monkeypatch.setattr(registry, "_persist", original_persist)
    await registry.close()


@pytest.mark.asyncio
async def test_failed_registry_close_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = BackgroundTaskRegistry()
    directory_fd = registry._directory_fd
    assert directory_fd is not None
    original_persist = registry._persist
    attempts = 0

    def flaky_persist(rows) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("close failed")
        original_persist(rows)

    monkeypatch.setattr(registry, "_persist", flaky_persist)

    with pytest.raises(OSError, match="close failed"):
        await registry.close()
    assert registry._closing
    assert not registry._closed
    os.fstat(directory_fd)

    assert await registry.close() == ()
    assert attempts == 2
    with pytest.raises(OSError):
        os.fstat(directory_fd)


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
        if use_fd else process_module.create_tool_subprocess_shell
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
        monkeypatch.setattr(process_module, "create_tool_subprocess_shell", paused_spawn)
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
