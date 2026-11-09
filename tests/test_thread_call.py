from __future__ import annotations

import asyncio
import errno
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import pytest

from zeta.core.thread_call import run_owned_thread_call


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
        pytest.fail(
            "owned thread call did not complete within the hang guard", pytrace=False
        )


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


def test_owned_thread_call_closes_result_during_asyncio_run_shutdown(
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


ROWS = (
    "success",
    "failure",
    "loop_affine",
    "cancel_before_start",
    "cancel_running",
    "cancel_running_failure",
    "duplicate_cancel",
    "cancel_result_ready",
    "loop_closed_running",
    "cancel_running_then_loop_closed",
)


@pytest.mark.parametrize("row", ROWS)
def test_owned_thread_call_closes_untaken_result_once(row: str, tmp_path: Path) -> None:
    """Each row drives a real loop and a one-thread executor it can join."""
    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    loop_thread = threading.get_ident()
    started = threading.Event()
    release = threading.Event()
    unblock = threading.Event()
    descriptors: list[int] = []
    closes: list[int] = []  # thread ident of each close

    def acquire() -> object:
        started.set()
        assert release.wait(10), "worker was not released"
        if row in {"failure", "cancel_running_failure"}:
            raise ValueError("acquire failed")
        if row == "loop_affine":
            return asyncio.Future(loop=loop)
        descriptor = os.open(tmp_path / "owned", os.O_RDWR | os.O_CREAT, 0o600)
        descriptors.append(descriptor)
        return descriptor

    def close(descriptor: object) -> None:
        closes.append(threading.get_ident())
        os.close(descriptor)  # type: ignore[arg-type]

    def step() -> None:
        loop.run_until_complete(asyncio.sleep(0))

    def assert_closed_once(*, on_loop: bool) -> None:
        assert len(closes) == 1, f"close ran {len(closes)} times"
        assert (closes[0] == loop_thread) is on_loop
        with pytest.raises(OSError) as error:
            os.fstat(descriptors[0])
        assert error.value.errno == errno.EBADF

    try:
        if row in {"success", "failure", "loop_affine"}:
            release.set()
            if row == "success":
                assert loop.run_until_complete(
                    run_owned_thread_call(acquire, close)
                ) == descriptors[0]
            else:
                expected = ValueError if row == "failure" else TypeError
                with pytest.raises(expected):
                    loop.run_until_complete(run_owned_thread_call(acquire, close))
            assert closes == []
        elif row == "cancel_before_start":
            executor.submit(unblock.wait, 10)  # occupy the only worker
            task = loop.create_task(run_owned_thread_call(acquire, close))
            step()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(task)
            unblock.set()
            executor.shutdown(wait=True)
            assert not started.is_set(), "a call cancelled before start ran"
            assert closes == []
        elif row == "cancel_result_ready":
            task = loop.create_task(run_owned_thread_call(acquire, close))
            step()
            assert started.wait(10)
            release.set()
            # The one worker runs this only after the call finished and handed back.
            executor.submit(lambda: None).result(10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                loop.run_until_complete(task)
            assert_closed_once(on_loop=True)
        else:
            task = loop.create_task(run_owned_thread_call(acquire, close))
            step()
            assert started.wait(10)
            if row != "loop_closed_running":
                task.cancel()
                if row == "duplicate_cancel":
                    task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    loop.run_until_complete(task)
                if row == "duplicate_cancel":
                    task.cancel()
            if row in {"loop_closed_running", "cancel_running_then_loop_closed"}:
                loop.close()
            release.set()
            executor.shutdown(wait=True)  # joins the worker: every close has run
            if row == "cancel_running_failure":
                assert closes == []
            else:
                assert_closed_once(on_loop=False)
    finally:
        release.set()
        unblock.set()
        executor.shutdown(wait=True)
        if not loop.is_closed():
            loop.close()
        for descriptor in descriptors:
            with suppress(OSError):
                os.close(descriptor)
