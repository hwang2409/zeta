"""one owned child process group; every blocking call runs on a worker thread."""

from __future__ import annotations

import asyncio
import functools
import os
import signal
import subprocess
import threading
from collections.abc import Callable, Sequence
from typing import Protocol

from ..core.thread_call import run_must_complete_thread_call

TERMINATE_SECONDS = 2.0
KILL_SECONDS = 2.0


class StdoutSink(Protocol):
    """Receives stdout chunks on the reader thread; ``close`` runs at EOF."""

    def write(self, chunk: bytes) -> None: ...
    def close(self) -> None: ...


class OwnedProcess:
    """A child in its own process group, reaped on exit.

    Popen, pipe reads, signals, and waits run on worker threads. The loop only
    receives bounded results.
    """

    def __init__(self, popen: subprocess.Popen[bytes], stdout_limit: int, stderr_limit: int) -> None:
        self._popen = popen
        self.stdout = bytearray()
        self.stderr = bytearray()
        self._readers: list[threading.Thread] = []
        self._stdout_limit = stdout_limit
        self._stderr_limit = stderr_limit

    @property
    def pid(self) -> int:
        return self._popen.pid

    @classmethod
    async def spawn(
        cls,
        argv: Sequence[str],
        *,
        stdout_sink: StdoutSink | None = None,
        stdout_limit: int = 64 * 1024,
        stderr_limit: int = 64 * 1024,
    ) -> OwnedProcess:
        spawn = functools.partial(cls._spawn_blocking, list(argv), stdout_sink, stdout_limit, stderr_limit)
        return await run_stoppable(lambda _stop: spawn(), cleanup=lambda process: process.kill_blocking())

    @classmethod
    def _spawn_blocking(
        cls, argv: list[str], sink: StdoutSink | None, stdout_limit: int, stderr_limit: int
    ) -> OwnedProcess:
        popen = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        process = cls(popen, stdout_limit, stderr_limit)
        assert popen.stdout is not None and popen.stderr is not None
        stdout = sink or _BoundedBuffer(process.stdout, stdout_limit)
        for stream, target in ((popen.stdout, stdout), (popen.stderr, _BoundedBuffer(process.stderr, stderr_limit))):
            reader = threading.Thread(target=_pump, args=(stream, target), name="zeta-voice-pipe", daemon=True)
            reader.start()
            process._readers.append(reader)
        return process

    async def wait(self) -> int:
        """Wait for exit and drained pipes; cancellation leaves the child to ``kill``."""
        return await asyncio.to_thread(self._wait_blocking)

    def _wait_blocking(self) -> int:
        code = self._popen.wait()
        for reader in self._readers:
            reader.join()
        return code

    async def request_stop(self, command: bytes, grace: float) -> int:
        """Ask the child to finish cleanly, then terminate and kill on deadlines."""
        return await run_must_complete_thread_call(functools.partial(self._stop_blocking, command, grace))

    def _stop_blocking(self, command: bytes, grace: float) -> int:
        stdin = self._popen.stdin
        if stdin is not None:
            try:
                stdin.write(command)
                stdin.close()
            except OSError:
                pass
        try:
            self._popen.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.kill_blocking()
        return self._wait_blocking()

    async def kill(self) -> None:
        await run_must_complete_thread_call(self.kill_blocking)

    def kill_blocking(self) -> None:
        """SIGTERM the group, escalate on timeout, and reap the exited child."""
        try:
            for sig, deadline in ((signal.SIGTERM, TERMINATE_SECONDS), (signal.SIGKILL, KILL_SECONDS)):
                if self._popen.poll() is not None:
                    break
                try:
                    os.killpg(self.pid, sig)
                except (ProcessLookupError, PermissionError):
                    # macOS can stop finding a signalable member before waitpid can reap it.
                    pass
                try:
                    self._popen.wait(timeout=deadline)
                except subprocess.TimeoutExpired:
                    continue
                break
            else:
                if self._popen.poll() is None:
                    raise RuntimeError(f"voice child pid {self.pid} did not exit by the SIGKILL deadline")
            self._wait_blocking()
        finally:
            if self._popen.stdin is not None:
                try:
                    self._popen.stdin.close()
                except OSError:
                    pass


async def run_stoppable[T](
    function: Callable[[threading.Event], T], *, cleanup: Callable[[T], None] | None = None
) -> T:
    """Run blocking work on a thread; on cancel, set its stop event and wait for it.

    Cancellation propagates only after the thread returns and ``cleanup`` (also
    off-loop) has released whatever it produced.
    """
    stop = threading.Event()
    worker = asyncio.get_running_loop().run_in_executor(None, function, stop)
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            stop.set()
    if not cancelled:
        return worker.result()
    if cleanup is not None and worker.exception() is None:
        await run_must_complete_thread_call(functools.partial(cleanup, worker.result()))
    raise asyncio.CancelledError


class _BoundedBuffer:
    def __init__(self, target: bytearray, limit: int) -> None:
        self._target = target
        self._limit = limit

    def write(self, chunk: bytes) -> None:
        self._target.extend(chunk[: max(0, self._limit - len(self._target))])

    def close(self) -> None:
        return None


def _pump(stream: object, sink: StdoutSink) -> None:
    read: Callable[[int], bytes] = stream.read1  # type: ignore[attr-defined]
    try:
        while chunk := read(64 * 1024):
            sink.write(chunk)
    finally:
        sink.close()
        stream.close()  # type: ignore[attr-defined]
