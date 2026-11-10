"""Single-owner lifecycle for one background process."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import replace
from enum import Enum, auto
from typing import IO, TYPE_CHECKING, Literal

from ...core.thread_call import run_must_complete_thread_call
from . import process_output
from .background_log_writer import _OrderedLogWriter
from .background_spawn import _group_exists, _signal_group
from .background_task_types import (
    BackgroundTaskKillOutcome,
    BackgroundTaskNotice,
    _BackgroundRecord,
)

if TYPE_CHECKING:
    from .process import BackgroundTaskRegistry

logger = logging.getLogger(__name__)

OUTPUT_CUTOFF_NOTE = (
    "background output was cut off because a process outside the task "
    "still held its output"
)
_NOTE_PRIORITY_ARCHIVE = 1
_NOTE_PRIORITY_CAUSE = 2


def _set_terminal_note(record: _BackgroundRecord, note: str, priority: int) -> None:
    if record.note is None:
        record.note = note
        record.note_priority = priority
    elif note not in record.note:
        existing_priority = record.note_priority or _NOTE_PRIORITY_ARCHIVE
        record.note = f"{note}; {record.note}" if priority > existing_priority else f"{record.note}; {note}"
        record.note_priority = max(existing_priority, priority)


class _OwnerState(Enum):
    RUNNING = auto()
    DRAINING = auto()
    ARCHIVING = auto()
    PERSISTING_TERMINAL = auto()
    DURABLE_TERMINAL = auto()
    TERMINAL = auto()


class _BackgroundTaskOwner:
    """Own every terminal transition for one background task."""

    def __init__(
        self,
        registry: BackgroundTaskRegistry,
        record: _BackgroundRecord,
        *,
        process: asyncio.subprocess.Process | None = None,
        log_handle: IO[bytes] | None = None,
    ) -> None:
        self.registry = registry
        self.record = record
        self.process = process
        self.log_handle = log_handle
        self.log_fd = log_handle.fileno() if log_handle is not None else None
        self.storage_lock = asyncio.Lock()
        self.state = _OwnerState.RUNNING if process is not None else _OwnerState.TERMINAL
        self.lifecycle: asyncio.Task[None] | None = None
        self.terminal_candidate: _BackgroundRecord | None = None
        self.kill_cause: str | None = record.note if record.note_priority == _NOTE_PRIORITY_CAUSE else None
        self.killed_by_claimant = False
        self._finish_output_drain = asyncio.Event()
        self._output_done = asyncio.Event()
        self.terminal_durable = asyncio.Event()
        self._close_termination: asyncio.Task[BackgroundTaskKillOutcome] | None = None
        if not record.running:
            self.terminal_durable.set()

    def start(self) -> None:
        if self.process is None or self.lifecycle is not None:
            return
        self.lifecycle = asyncio.create_task(self._run())

    async def wait(self, timeout: float | None = None) -> None:
        lifecycle = self.lifecycle
        if lifecycle is None:
            return
        if timeout is not None:
            await asyncio.wait((lifecycle,), timeout=timeout)
        if timeout is None or lifecycle.done():
            # test-contract: durability the owner must finish its terminal commit
            await asyncio.shield(lifecycle)

    async def _run(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        writer = _OrderedLogWriter(self.log_handle) if self.log_handle is not None else None
        cancelled = False

        async def read_output() -> None:
            try:
                await self.registry._read_output(self.record, process.stdout, writer)
            finally:
                self._output_done.set()

        reader = asyncio.create_task(read_output())
        archive_error: BaseException | None = None

        async def finish() -> None:
            nonlocal archive_error
            if not reader.done():
                reader.cancel()
            # test-contract: durability every exit joins the output reader and process
            await asyncio.gather(reader, return_exceptions=True)
            await process.wait()
            await self.registry._close_stdin(self.record)
            if writer is not None:
                await writer.close()
            if self.log_fd is not None:
                self.state = _OwnerState.ARCHIVING
                try:
                    await self._archive()
                except BaseException as exc:
                    archive_error = exc
                    _set_terminal_note(
                        self.record,
                        "background output archive failed; recovery will retry the retained log",
                        _NOTE_PRIORITY_ARCHIVE,
                    )
                    logger.exception(
                        "failed to archive background task output for %s",
                        self.record.task_id,
                    )
            await self._finish_terminal(process.returncode)

        try:
            await process.wait()
            self.state = _OwnerState.DRAINING
            # test-contract: durability output drain races the owned reader to completion
            drain_finished = asyncio.create_task(self._finish_output_drain.wait())
            try:
                await asyncio.wait(
                    (reader, drain_finished),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if drain_finished.done() and not reader.done():
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                else:
                    await reader
            finally:
                if not drain_finished.done():
                    drain_finished.cancel()
                await asyncio.gather(drain_finished, return_exceptions=True)
            while self.kill_cause is None and _group_exists(process.pid):
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled = True
            _signal_group(process, signal.SIGKILL)
        finally:
            cleanup = asyncio.create_task(finish())
            while True:
                try:
                    await asyncio.shield(cleanup)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    _signal_group(process, signal.SIGKILL)
                    if cleanup.cancelled():
                        cleanup = asyncio.create_task(finish())

        if archive_error is not None:
            raise archive_error
        if cancelled:
            raise asyncio.CancelledError

    async def _archive(self) -> None:
        async with self.storage_lock:
            if self.registry._archive is None or self.log_fd is None:
                raise RuntimeError("background output storage is unavailable")
            try:
                await run_must_complete_thread_call(
                    lambda: self.registry._archive.append(
                        self.record.task_id, self.log_fd, self.record.total_bytes
                    )
                )
                self.registry._archived_lengths[self.record.task_id] = self.record.total_bytes
            finally:
                if self.log_handle is not None:
                    self.log_handle.close()
                self.log_handle = None
                self.log_fd = None

    async def _finish_terminal(self, exit_code: int | None) -> None:
        async with self.registry._transition_lock:
            if not self.record.running:
                return
            self.state = _OwnerState.PERSISTING_TERMINAL
            candidate = replace(
                self.record,
                running=False,
                exit_code=exit_code,
                ended_at=time.monotonic(),
                process=None,
                receipt_pending=self.record.receipt_pending or self.record.notify_on_exit,
                terminal_phase=self.record.terminal_phase or "natural_exit",
            )
            self.terminal_candidate = candidate
            rows = self.registry._persistence_rows(
                tuple(
                    candidate
                    if item is self.record
                    else (
                        self.registry._owners[item.task_id].terminal_candidate
                        if (
                            self.registry._owners[item.task_id].terminal_durable.is_set()
                            and self.registry._owners[item.task_id].terminal_candidate is not None
                        )
                        else item
                    )
                    for item in self.registry._records.values()
                )
            )
            attempts = 0
            while not self.terminal_durable.is_set():
                try:
                    notice = BackgroundTaskNotice(
                        process_output.task_exit_headline(
                            candidate.task_id,
                            candidate.exit_code,
                            candidate.command,
                            candidate.note,
                        ),
                        candidate.task_id,
                        candidate.terminal_phase or "natural_exit",
                        candidate.owner,
                    )
                    await self.registry._commit_transition(
                        rows,
                        lambda: self._publish_terminal(candidate),
                        notice,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    attempts += 1
                    if attempts >= 3:
                        raise
                    # Keep the same terminal candidate. A failed write must not
                    # end the owner while the live record still says running.
                    await asyncio.sleep(0)
                    continue
                self.state = _OwnerState.DURABLE_TERMINAL
                self.terminal_durable.set()
        if candidate.notify_on_exit and not self.registry._closing:
            await self.registry._notify_terminal_exit(candidate)
        self.state = _OwnerState.TERMINAL

    def _publish_terminal(self, candidate: _BackgroundRecord) -> None:
        self.record.running = candidate.running
        self.record.exit_code = candidate.exit_code
        self.record.ended_at = candidate.ended_at
        self.record.process = candidate.process
        self.record.receipt_pending = candidate.receipt_pending
        self.record.terminal_phase = candidate.terminal_phase
        self.process = None

    async def _claim_termination(
        self,
        cause: str,
        phase: Literal["task_kill", "session_shutdown"],
        tui_presented: bool,
    ) -> tuple[asyncio.subprocess.Process | None, bool]:
        async with self.registry._transition_lock:
            process = self.process
            if (
                process is None
                or not self.record.running
                or self.kill_cause is not None
            ):
                return process, False
            candidate = replace(
                self.record,
                terminal_phase=phase,
                note=cause,
                tui_presented=tui_presented,
            )
            rows = self.registry._persistence_rows(
                tuple(
                    candidate if item is self.record else item
                    for item in self.registry._records.values()
                )
            )

            def publish() -> None:
                self.kill_cause = cause
                self.killed_by_claimant = True
                self.record.terminal_phase = phase
                _set_terminal_note(self.record, cause, _NOTE_PRIORITY_CAUSE)
                self.record.tui_presented = tui_presented

            await self.registry._commit_transition(rows, publish, None)
            return process, True

    async def terminate(
        self,
        cause: str,
        phase: Literal["task_kill", "session_shutdown"],
        tui_presented: bool,
    ) -> BackgroundTaskKillOutcome:
        process, claimant = await self._claim_termination(
            cause, phase, tui_presented
        )
        if claimant and process is not None:
            await self.registry._terminate_owned([process])
        # test-contract: durability kill returns the committed terminal outcome
        await self.wait()
        killed = (
            self.killed_by_claimant
            and self.record.terminal_phase in {"task_kill", "session_shutdown"}
            and self.record.exit_code in {-signal.SIGTERM, -signal.SIGKILL}
        )
        return BackgroundTaskKillOutcome(
            killed_by_this_call=claimant and killed,
            cause=self.record.note if killed else None,
            exit_code=self.record.exit_code,
            status="killed" if killed else "exited",
        )

    async def finish_for_close(
        self, cause: str
    ) -> asyncio.subprocess.Process | None:
        phase: Literal["task_kill", "session_shutdown"] = (
            "task_kill"
            if cause.startswith("finish-gate task safety bound expired")
            else "session_shutdown"
        )
        process, _ = await self._claim_termination(cause, phase, False)
        return process

    def finish_close_output_drain(self) -> None:
        self._finish_output_drain.set()
        if not self._output_done.is_set():
            _set_terminal_note(self.record, OUTPUT_CUTOFF_NOTE, _NOTE_PRIORITY_ARCHIVE)
        stream = self.process.stdout if self.process is not None else None
        transport = getattr(stream, "_transport", None)
        if transport is not None:
            transport.close()

    async def wait_until_durable(self) -> None:
        # test-contract: durability close waits for the terminal commit
        durable = asyncio.create_task(self.terminal_durable.wait())
        lifecycle = self.lifecycle
        if lifecycle is None:
            # test-contract: durability close waits for the terminal commit
            await durable
            return
        # test-contract: durability close observes either commit or owner failure
        done, pending = await asyncio.wait(
            (durable, lifecycle), return_when=asyncio.FIRST_COMPLETED
        )
        if durable in done:
            durable.result()
            return
        for task in pending:
            task.cancel()
        # test-contract: durability canceled close gates are fully drained
        await asyncio.gather(*pending, return_exceptions=True)
        if lifecycle in done:
            # Propagate the bounded terminal-write failure. Close must not
            # publish an aggregate success notice or release a receipt.
            lifecycle.result()
