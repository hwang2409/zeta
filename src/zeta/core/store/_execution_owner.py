"""Session-owner checks for append-lock acquisition."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

from ...locking import execution_lock, shape_lock
from ..session_files import open_session_file
from ._pending_prompts import PendingPromptCommitTimeoutError


class ExecutionOwnerStoreMixin:
    """Keep execution ownership and append locking in one store seam."""

    @contextmanager
    def _append_lock(
        self,
        *,
        deadline: float | None = None,
        wait: bool = True,
        observer: bool = False,
        nested_authority: AbstractContextManager[object] | None = None,
        after_shape: Callable[[], None] | None = None,
    ) -> Iterator[bool]:
        """Lock and synchronize the log before exposing resident state."""
        if self._closed:
            raise ValueError("session store is closed")
        if not observer:
            self.assert_execution_owner()
        authority = nullcontext() if observer else execution_lock()
        started = False
        try:
            with authority, nested_authority or nullcontext():
                with shape_lock(), os.fdopen(
                    open_session_file(
                        self.directory_fd, ".lock", os.O_RDWR | os.O_CREAT
                    ),
                    "r+",
                ) as handle:
                    if deadline is None:
                        try:
                            fcntl.flock(
                                handle.fileno(),
                                fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                        except BlockingIOError:
                            yield False
                            return
                    else:
                        while True:
                            if time.monotonic() >= deadline:
                                raise PendingPromptCommitTimeoutError(
                                    "pending prompt commit deadline exceeded"
                                )
                            try:
                                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                time.sleep(max(0, min(0.01, deadline - time.monotonic())))
                            else:
                                break
                    self._start_entries_installed()
                    started = True
                    try:
                        self._load()
                        yield True
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                if after_shape is not None:
                    after_shape()
        finally:
            # Callbacks may append again, so they run after every ranked lock is released.
            if started:
                self._finish_entries_installed()
