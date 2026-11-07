from __future__ import annotations

import asyncio
import errno
import os
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path

import pytest

from zeta.core.thread_call import run_owned_thread_call


async def _next_loop_turn() -> None:
    resumed = asyncio.Event()
    asyncio.get_running_loop().call_soon(resumed.set)
    await asyncio.wait_for(resumed.wait(), 10)


def _run_guarded_subprocess(source: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), environment.get("PYTHONPATH")))
    )
    try:
        return subprocess.run(
            [sys.executable, "-c", source],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("owned thread call did not complete within the hang guard", pytrace=False)


def test_owned_thread_call_propagates_worker_cancelled_error_once() -> None:
    completed = _run_guarded_subprocess(
        """
import asyncio

from zeta.core.thread_call import run_owned_thread_call

calls = 0


def cancel():
    global calls
    calls += 1
    raise asyncio.CancelledError


async def main():
    try:
        await run_owned_thread_call(cancel)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("worker cancellation was not propagated")
    assert calls == 1


asyncio.run(main())
"""
    )

    assert completed.returncode == 0, completed.stderr


def test_owned_thread_call_adopts_result_during_asyncio_run_shutdown(
    tmp_path: Path,
) -> None:
    owned_path = tmp_path / "shutdown-owned"
    completed = _run_guarded_subprocess(
        f"""
import asyncio
import errno
import os
import threading

from zeta.core.thread_call import run_owned_thread_call

started = threading.Event()
release = threading.Event()
descriptors = []
adopted = []


def acquire():
    descriptor = os.open({str(owned_path)!r}, os.O_RDWR | os.O_CREAT, 0o600)
    descriptors.append(descriptor)
    started.set()
    assert release.wait(timeout=10)
    return descriptor


def adopt(descriptor):
    adopted.append(descriptor)
    os.close(descriptor)


class ShutdownAwareTask(asyncio.Task):
    def cancel(self, *args, **kwargs):
        cancelled = super().cancel(*args, **kwargs)
        release.set()
        return cancelled


async def main():
    loop = asyncio.get_running_loop()
    loop.set_task_factory(
        lambda loop, coroutine, context=None: ShutdownAwareTask(
            coroutine, loop=loop, context=context
        )
    )
    asyncio.create_task(run_owned_thread_call(acquire, adopt))
    loop.set_task_factory(None)
    assert await asyncio.to_thread(started.wait, 10)


asyncio.run(main())
assert adopted == descriptors
try:
    os.fstat(descriptors[0])
except OSError as error:
    assert error.errno == errno.EBADF
else:
    raise AssertionError("adopted descriptor remained open")
"""
    )

    assert completed.returncode == 0, completed.stderr


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
