from __future__ import annotations

import ast
import asyncio
import errno
import gc
import logging
import os
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from zeta.core.thread_call import run_owned_thread_call
from zeta.storage_format import drain_releases


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
import gc
import logging
import os
import socket
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
    asyncio.create_task(
        run_owned_thread_call(acquire, adopt, close_is_thread_safe=True)
    )
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


class _Row:
    """A real loop and a one-thread executor that each row can join."""

    def __init__(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop.set_default_executor(self.executor)
        self.loop_thread = threading.get_ident()
        self.path = tmp_path / "owned"
        self.caplog = caplog
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = False
        self.descriptors: list[int] = []
        self.closes: list[int] = []  # thread ident of each close
        self.sockets: list[socket.socket] = []
        self.transport: asyncio.Transport | None = None

    def acquire(self) -> object:
        self.started.set()
        assert self.release.wait(10), "worker was not released"
        if self.fail:
            raise ValueError("acquire failed")
        if self.transport is not None:
            return self.transport
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self.descriptors.append(descriptor)
        return descriptor

    def close(self, result: object) -> None:
        self.closes.append(threading.get_ident())
        if isinstance(result, asyncio.Transport):
            result.close()
        else:
            os.close(result)  # type: ignore[arg-type]

    def use_transport(self) -> asyncio.Transport:
        """A real loop-affine closeable: closing it calls loop methods."""
        self.sockets.extend(socket.socketpair())
        transport, _ = self.loop.run_until_complete(
            self.loop.connect_accepted_socket(asyncio.Protocol, self.sockets[0])
        )
        self.transport = transport  # type: ignore[assignment]
        return transport  # type: ignore[return-value]

    def call(self) -> Any:
        return run_owned_thread_call(
            self.acquire, self.close, close_is_thread_safe=True
        )

    def start(self) -> asyncio.Task[Any]:
        task = self.loop.create_task(self.call())
        self.step()
        assert self.started.wait(10)
        return task

    def step(self) -> None:
        self.loop.run_until_complete(asyncio.sleep(0))

    def cancel(self, task: asyncio.Task[Any]) -> None:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            self.loop.run_until_complete(task)

    def finish_worker(self) -> None:
        """Release the worker and wait until it handed back; the loop stays idle."""
        self.release.set()
        # The one worker runs this only after the call finished and handed back.
        self.executor.submit(lambda: None).result(10)

    def assert_closed_once(self, *, on_loop: bool) -> None:
        assert len(self.closes) == 1, f"close ran {len(self.closes)} times"
        assert (self.closes[0] == self.loop_thread) is on_loop
        with pytest.raises(OSError) as error:
            os.fstat(self.descriptors[0])
        assert error.value.errno == errno.EBADF

    def cleanup(self) -> None:
        self.release.set()
        self.executor.shutdown(wait=True)
        if not self.loop.is_closed():
            if self.transport is not None:
                self.transport.close()
                self.step()
            self.loop.close()
        for descriptor in self.descriptors:
            with suppress(OSError):
                os.close(descriptor)
        for sock in self.sockets:
            sock.close()


def _success(row: _Row) -> None:
    row.release.set()
    assert row.loop.run_until_complete(row.call()) == row.descriptors[0]
    assert row.closes == []


def _failure(row: _Row) -> None:
    row.fail = True
    row.release.set()
    with pytest.raises(ValueError):
        row.loop.run_until_complete(row.call())
    assert row.closes == []


def _refused_without_thread_safe_flag(row: _Row) -> None:
    row.release.set()
    with pytest.raises(TypeError, match="close_is_thread_safe"):
        row.loop.run_until_complete(run_owned_thread_call(row.acquire, row.close))
    assert not row.started.is_set(), "a refused call ran its work"
    assert row.closes == []


def _cancel_before_start(row: _Row) -> None:
    unblock = threading.Event()
    row.executor.submit(unblock.wait, 10)  # occupy the only worker
    try:
        task = row.loop.create_task(row.call())
        row.step()
        row.cancel(task)
    finally:
        unblock.set()
    row.executor.shutdown(wait=True)
    assert not row.started.is_set(), "a call cancelled before start ran"
    assert row.closes == []


def _cancel_running(row: _Row) -> None:
    row.cancel(row.start())
    row.release.set()
    row.executor.shutdown(wait=True)  # joins the worker: every close has run
    row.assert_closed_once(on_loop=False)


def _cancel_running_failure(row: _Row) -> None:
    row.fail = True
    row.cancel(row.start())
    row.release.set()
    row.executor.shutdown(wait=True)
    assert row.closes == []


def _duplicate_cancel(row: _Row) -> None:
    task = row.start()
    task.cancel()
    row.cancel(task)
    task.cancel()
    row.release.set()
    row.executor.shutdown(wait=True)
    row.assert_closed_once(on_loop=False)


def _cancel_result_ready(row: _Row) -> None:
    task = row.start()
    row.finish_worker()
    row.cancel(task)
    row.assert_closed_once(on_loop=True)


def _loop_closed_running(row: _Row) -> None:
    row.start()
    row.loop.close()
    row.release.set()
    row.executor.shutdown(wait=True)
    row.assert_closed_once(on_loop=False)


def _cancel_running_then_loop_closed(row: _Row) -> None:
    row.cancel(row.start())
    row.loop.close()
    row.release.set()
    row.executor.shutdown(wait=True)
    row.assert_closed_once(on_loop=False)


def _loop_closed_after_handback(row: _Row) -> None:
    task = row.start()
    row.finish_worker()  # the hand-back callback is queued on the loop
    row.loop.close()  # discards the queued callback without running it
    del task
    gc.collect()
    assert row.closes == [], "a finalizer ran the close instead of queueing it"
    drain_releases()  # the next release point runs the queued close
    row.assert_closed_once(on_loop=True)
    assert "hand-back was dropped" in row.caplog.text


def _loop_closed_after_deliver(row: _Row) -> None:
    task = row.start()
    row.finish_worker()
    row.loop.call_soon(row.loop.stop)
    row.loop.run_forever()  # one turn: delivers, queues the task wake-up
    row.loop.close()  # discards the wake-up; the awaiter never takes it
    del task
    gc.collect()  # the dropped task's coroutine is closed and abandons the call
    assert row.closes == [], "a finalizer ran the close instead of queueing it"
    drain_releases()
    row.assert_closed_once(on_loop=True)
    assert "hand-back was dropped" not in row.caplog.text, "delivery did not run"


def _loop_affine_taken(row: _Row) -> None:
    transport = row.use_transport()
    row.release.set()
    assert row.loop.run_until_complete(row.call()) is transport
    assert row.closes == []
    assert not transport.is_closing()


def _loop_affine_cancel_running(row: _Row) -> None:
    transport = row.use_transport()
    row.cancel(row.start())
    row.release.set()
    row.executor.shutdown(wait=True)
    assert row.closes == [], "a loop-affine result was closed on the worker"
    row.step()  # the close was handed to the loop
    assert row.closes == [row.loop_thread], f"closes: {row.closes}"
    assert transport.is_closing()


def _loop_affine_loop_closed(row: _Row) -> None:
    row.use_transport()
    row.start()
    row.loop.close()
    row.release.set()
    row.executor.shutdown(wait=True)
    assert row.closes == [], "a loop-affine result was closed on the worker"
    assert "leaked loop-affine" in row.caplog.text, "the leak was dropped silently"


def _loop_affine_close_dropped(row: _Row) -> None:
    row.use_transport()
    row.cancel(row.start())
    row.release.set()
    row.executor.shutdown(wait=True)  # the worker queued the close on the loop
    row.loop.close()  # discards the queued close without running it
    gc.collect()
    drain_releases()
    assert row.closes == [], "a loop-affine result was closed on the worker"
    leaks = [r for r in row.caplog.records if "leaked loop-affine" in r.message]
    assert len(leaks) == 1, "the dropped loop-affine close was not logged once"
    assert "Transport" in leaks[0].message
    assert "_Row.acquire" in leaks[0].message


ROWS = {
    "success": _success,
    "failure": _failure,
    "refused_without_thread_safe_flag": _refused_without_thread_safe_flag,
    "cancel_before_start": _cancel_before_start,
    "cancel_running": _cancel_running,
    "cancel_running_failure": _cancel_running_failure,
    "duplicate_cancel": _duplicate_cancel,
    "cancel_result_ready": _cancel_result_ready,
    "loop_closed_running": _loop_closed_running,
    "cancel_running_then_loop_closed": _cancel_running_then_loop_closed,
    "loop_closed_after_handback": _loop_closed_after_handback,
    "loop_closed_after_deliver": _loop_closed_after_deliver,
    "loop_affine_taken": _loop_affine_taken,
    "loop_affine_cancel_running": _loop_affine_cancel_running,
    "loop_affine_loop_closed": _loop_affine_loop_closed,
    "loop_affine_close_dropped": _loop_affine_close_dropped,
}


@pytest.mark.parametrize("name", ROWS)
def test_owned_thread_call_closes_untaken_result_once(
    name: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="zeta.core.thread_call")
    row = _Row(tmp_path, caplog)
    try:
        ROWS[name](row)
    finally:
        row.cleanup()


def test_every_loop_callback_goes_through_the_owned_handoff() -> None:
    """A plain loop callback is dropped silently when the loop closes first."""
    import zeta.core.thread_call as module

    tree = ast.parse(Path(module.__file__).read_text())
    schedule = next(
        (
            fn for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef) and cls.name == "_Handoff"
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and fn.name == "schedule"
        ),
        None,
    )
    owned = {id(node) for node in ast.walk(schedule)} if schedule else set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "call_soon_threadsafe"
        ):
            assert id(node) in owned, (
                f"line {node.lineno} calls call_soon_threadsafe directly; use "
                "_Handoff.schedule(loop, on_loop, dropped) so a dropped "
                "callback runs its dropped action"
            )
