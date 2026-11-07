"""Cancellation-safe ordered writes for background task logs."""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from typing import IO

from ...core.thread_call import run_owned_thread_call


class _OrderedLogWriter:
    """Write and flush log chunks in order without blocking the event loop."""

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle
        self._queue: queue.SimpleQueue[
            tuple[bytes | None, concurrent.futures.Future[None]]
        ] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="zeta-background-log", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        failure: BaseException | None = None
        while True:
            data, completion = self._queue.get()
            try:
                if data is None:
                    self._handle.flush()
                elif failure is not None:
                    raise failure
                else:
                    view = memoryview(data)
                    while view:
                        written = self._handle.write(view)
                        if written is None:
                            written = len(view)
                        if written == 0:
                            raise OSError("zero-byte write to background task log")
                        view = view[written:]
                    self._handle.flush()
            except Exception as exc:  # noqa: BLE001 - report all I/O failures
                if data is not None:
                    failure = exc
                completion.set_exception(exc)
            else:
                completion.set_result(None)
            if data is None:
                return

    async def write(self, data: bytes) -> None:
        completion: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._queue.put((data, completion))
        await _await_worker(completion)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        completion: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._queue.put((None, completion))
        try:
            await _await_worker(completion)
        finally:
            await run_owned_thread_call(self._thread.join)


async def _await_worker(completion: concurrent.futures.Future[None]) -> None:
    await run_owned_thread_call(completion.result)
