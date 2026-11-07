from __future__ import annotations

import asyncio
import errno
import os
import threading
from contextlib import suppress
from pathlib import Path

import pytest

from zeta.core.thread_call import run_owned_thread_call


async def _next_loop_turn() -> None:
    resumed = asyncio.Event()
    asyncio.get_running_loop().call_soon(resumed.set)
    await resumed.wait()


@pytest.mark.asyncio
async def test_owned_thread_call_adopts_once_after_repeated_cancellation(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()
    descriptors: list[int] = []
    adopted: list[int] = []

    def acquire() -> int:
        descriptor = os.open(tmp_path / "owned", os.O_RDWR | os.O_CREAT, 0o600)
        descriptors.append(descriptor)
        started.set()
        assert release.wait(timeout=10)
        return descriptor

    def adopt(descriptor: int) -> None:
        adopted.append(descriptor)
        os.close(descriptor)

    task = asyncio.create_task(run_owned_thread_call(acquire, adopt))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await _next_loop_turn()
        task.cancel()
        await _next_loop_turn()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)

        assert adopted == descriptors
        with pytest.raises(OSError) as error:
            os.fstat(descriptors[0])
        assert error.value.errno == errno.EBADF
    finally:
        release.set()
        with suppress(BaseException):
            await asyncio.wait_for(task, timeout=10)
        for descriptor in descriptors:
            with suppress(OSError):
                os.close(descriptor)


@pytest.mark.asyncio
async def test_owned_thread_call_keeps_worker_failure_after_cancellation(
    tmp_path: Path,
) -> None:
    class WorkerFailure(Exception):
        pass

    started = threading.Event()
    release = threading.Event()
    descriptors: list[int] = []
    adopted: list[int] = []

    def fail() -> int:
        descriptor = os.open(tmp_path / "failed", os.O_RDWR | os.O_CREAT, 0o600)
        descriptors.append(descriptor)
        started.set()
        try:
            assert release.wait(timeout=10)
            raise WorkerFailure("worker failed")
        finally:
            os.close(descriptor)

    task = asyncio.create_task(run_owned_thread_call(fail, adopted.append))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await _next_loop_turn()
        release.set()
        with pytest.raises(WorkerFailure, match="worker failed"):
            await asyncio.wait_for(task, timeout=10)

        assert adopted == []
        with pytest.raises(OSError) as error:
            os.fstat(descriptors[0])
        assert error.value.errno == errno.EBADF
    finally:
        release.set()
        with suppress(BaseException):
            await asyncio.wait_for(task, timeout=10)
        for descriptor in descriptors:
            with suppress(OSError):
                os.close(descriptor)
