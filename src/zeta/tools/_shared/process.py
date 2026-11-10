"""Process-group cleanup shared by shell tools."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.store import ConversationStore

from ...core.checkpoints import load_session_json
from ...core.process_env import subprocess_env as tool_subprocess_env
from ...core.session_files import (
    open_session_file,
    read_session_file,
    session_root,
    write_session_file,
)
from ...core.thread_call import run_owned_thread_call
from ...storage_format import WriterLease
from .background_log_writer import _OrderedLogWriter
from .background_output_archive import _BackgroundOutputArchive
from .background_spawn import (
    BackgroundSpawnCleanupError,
    _terminate_process,
    _terminate_processes,
    create_subprocess_shell_in_fd,
    spawn_background,
)
from .background_task_owner import _BackgroundTaskOwner
from .background_task_types import (
    BackgroundTaskInfo,
    BackgroundTaskNotice,
    BackgroundTaskNoticeSinkValue,
    BackgroundTaskShutdownNotice,
    TaskExitInput,
    _BackgroundRecord,
)
from .log_open import open_log_off_loop
from .process_output import command_headline, utf8_char_chunk, utf8_chunk
from .tool_spawn import create_tool_subprocess_shell

logger = logging.getLogger(__name__)


BACKGROUND_OUTPUT_LIMIT = 512 * 1024
BACKGROUND_OUTPUT_CALL_LIMIT = 32 * 1024
BACKGROUND_TERM_GRACE_SECONDS = 0.25
BACKGROUND_STDIN_LIMIT = 64 * 1024
BACKGROUND_STDIN_DRAIN_TIMEOUT = 1.0
BACKGROUND_OUTPUT_DRAIN_TIMEOUT = 0.25
class BackgroundTaskRegistry:
    """Own detached process groups and their bounded session output."""

    def __init__(
        self,
        *,
        session_dir: str | Path | None = None,
        directory_fd: int | None = None,
        output_limit: int = BACKGROUND_OUTPUT_LIMIT,
        call_limit: int = BACKGROUND_OUTPUT_CALL_LIMIT,
        term_grace: float = BACKGROUND_TERM_GRACE_SECONDS,
        stdin_drain_timeout: float = BACKGROUND_STDIN_DRAIN_TIMEOUT,
        notice_sink: Callable[[BackgroundTaskNoticeSinkValue], None] | None = None,
        notification_sink: Callable[[TaskExitInput], Awaitable[object]] | None = None,
        notification_exists: Callable[[str], bool] | None = None,
        command_niceness: int = 10,
        storage_lease: WriterLease | None = None,
    ) -> None:
        if type(output_limit) is not int or output_limit < 1:
            raise ValueError("output_limit must be a positive integer")
        if type(call_limit) is not int or call_limit < 1:
            raise ValueError("call_limit must be a positive integer")
        if term_grace <= 0:
            raise ValueError("term_grace must be positive")
        if stdin_drain_timeout <= 0:
            raise ValueError("stdin_drain_timeout must be positive")
        if type(command_niceness) is not int or not 0 <= command_niceness <= 19:
            raise ValueError("command_niceness must be an integer from 0 to 19")
        self.output_limit = output_limit
        self.call_limit = call_limit
        self.term_grace = term_grace
        self.stdin_drain_timeout = stdin_drain_timeout
        self.command_niceness = command_niceness
        self._notice_sink = notice_sink
        self._notification_sink = notification_sink
        self._notification_exists = notification_exists or (lambda _task_id: False)
        self._pending_recovery: dict[str, TaskExitInput] = {}
        self._records: dict[str, _BackgroundRecord] = {}
        self._owners: dict[str, _BackgroundTaskOwner] = {}
        self._session_dir: Path | None = None
        self._directory_fd: int | None = None
        self._storage_lease: WriterLease | None = None
        self._persistence_corrupt = False
        self._archive: _BackgroundOutputArchive | None = None
        self._archived_lengths: dict[str, int] = {}
        self._temporary_log_root: Path | None = None
        self._closing = False
        self._closed = False
        self._close_done: asyncio.Future[tuple[str, ...]] | None = None
        self._close_failure_generation = 0
        self._pending_starts: set[asyncio.Future[None]] = set()
        self._transition_lock = asyncio.Lock()
        self._task_start_generation = 0
        self._start_cleanup_error: BackgroundSpawnCleanupError | None = None
        self._shutdown_killed: tuple[str, ...] = ()
        self._shutdown_cause: str | None = None
        if session_dir is not None:
            if directory_fd is None:
                raise ValueError("session directory descriptor is required")
            if storage_lease is None:
                raise ValueError("stored-format writer lease is required")
            self.bind_session_dir(session_dir, directory_fd, storage_lease)
        else:
            self._temporary_log_root = Path(tempfile.mkdtemp(prefix="zeta-background-"))
            os.chmod(self._temporary_log_root, 0o700)
            temporary_fd = os.open(
                self._temporary_log_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            self._bind_storage(self._temporary_log_root, temporary_fd)
            os.close(temporary_fd)

    @property
    def running_count(self) -> int:
        return sum(record.running for record in self._records.values())

    def finish_gate_snapshot(self) -> tuple[tuple[str, str, object, int], ...]:
        """Project each owner once for the finish gate."""

        return tuple(
            (owner.record.task_id, owner.record.command, owner.state, owner.generation)
            for owner in self._owners.values()
            if owner.state.name != "TERMINAL"
        )

    async def wait_finish_gate_generation(self, generation: tuple[tuple[str, int], ...]) -> None:
        """Wake when an owner changes phase; no sink or registry lock is held."""

        current = self.observe_finish_gate()[1]
        if current != generation:
            return
        events = [
            owner.generation_event
            for owner in self._owners.values()
            if owner.state.name != "TERMINAL"
        ]
        if self.observe_finish_gate()[1] != generation:
            return
        waits = [asyncio.create_task(event.wait()) for event in events]
        if not waits:
            return
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                if not wait.done():
                    wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)

    def observe_finish_gate(
        self, generation: tuple[tuple[str, int], ...] | None = None
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, int], ...]]:
        """Return owned work and a stable owner-generation observation."""

        snapshot = self.finish_gate_snapshot()
        current = tuple(sorted((task_id, owner_generation) for task_id, _command, _state, owner_generation in snapshot))
        if generation is not None and generation != current:
            snapshot = self.finish_gate_snapshot()
            current = tuple(sorted((task_id, owner_generation) for task_id, _command, _state, owner_generation in snapshot))
        return tuple((task_id, command) for task_id, command, _state, _gen in snapshot), current

    @property
    def task_start_generation(self) -> int:
        """Return the monotonic count of tasks that reached running state."""

        return self._task_start_generation

    def close_failure_generation(self) -> int:
        return self._close_failure_generation

    @property
    def records(self) -> tuple[_BackgroundRecord, ...]:
        return tuple(self._records.values())

    def snapshot(self) -> tuple[BackgroundTaskInfo, ...]:
        """Return an immutable, read-only view of every task in this session."""

        return tuple(
            BackgroundTaskInfo(
                task_id=record.task_id,
                command=record.command,
                pid=record.pid,
                owner=record.owner,
                running=record.running,
                exit_code=record.exit_code,
                note=record.note,
                terminal_phase=record.terminal_phase,
                started_at=record.started_at,
                ended_at=record.ended_at,
                output_bytes=record.total_bytes,
                output_lines=record.total_lines,
            )
            for record in self._records.values()
        )

    def set_notice_sink(
        self, sink: Callable[[BackgroundTaskNoticeSinkValue], None] | None
    ) -> None:
        self._notice_sink = sink

    def set_notification_exists(self, lookup: Callable[[str], bool]) -> None:
        self._notification_exists = lookup

    def set_notification_sink(
        self,
        sink: Callable[[TaskExitInput], Awaitable[object]] | None,
    ) -> tuple[TaskExitInput, ...]:
        self._notification_sink = sink
        if sink is None:
            return ()
        return tuple(self._pending_recovery.values())

    async def deliver_recovered_exit(self, item: TaskExitInput) -> None:
        await self._owners[item.task_id].retry_evidence(item)

    def bind_session_dir(
        self,
        session_dir: str | Path,
        directory_fd: int,
        storage_lease: WriterLease | None = None,
    ) -> None:
        self._require_open()
        if storage_lease is None:
            raise ValueError("stored-format writer lease is required")
        if self._directory_fd is not None and self._temporary_log_root is None:
            if not os.path.samestat(
                os.fstat(self._directory_fd), os.fstat(directory_fd)
            ):
                raise ValueError("background task registry is already bound")
            return
        if self._records:
            raise ValueError(
                "cannot bind a registry after background tasks have started"
            )
        temporary_root = self._temporary_log_root
        self._close_storage()
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
            self._temporary_log_root = None
        retained_lease = storage_lease.retain()
        previous_lease = self._storage_lease
        self._storage_lease = retained_lease
        if previous_lease is not None:
            previous_lease.close()
        try:
            self._bind_storage(Path(session_dir), directory_fd)
            self._load_previous()
        except BaseException:
            self._close_storage()
            retained_lease.close()
            self._storage_lease = None
            raise

    def _bind_storage(self, directory: Path, directory_fd: int) -> None:
        self._session_dir = directory
        self._directory_fd = os.dup(directory_fd)
        self._release_directory = weakref.finalize(self, os.close, self._directory_fd)
        self._archive = _BackgroundOutputArchive(self._directory_fd)
        self._archived_lengths = self._archive.recover()

    def _open_task_log(self, task_id: str) -> IO[bytes]:
        if self._archive is None:
            raise RuntimeError("background output storage is unavailable")
        return os.fdopen(self._archive.open_task_log(task_id), "w+b", buffering=0)

    def open_log(self, path: str | Path) -> IO[bytes]:
        self._require_open()
        path = Path(path)
        directory = (
            nullcontext(self._directory_fd)
            if self._directory_fd is not None and path.parent == self._session_dir
            else session_root(path.parent, create=True)
        )
        with directory as directory_fd:
            fd = open_session_file(
                directory_fd,
                path.name,
                os.O_RDWR | os.O_CREAT | os.O_TRUNC,
            )
        return os.fdopen(fd, "w+b", buffering=0)

    async def start(
        self,
        command: str,
        cwd: str | Path,
        *,
        cwd_fd: int | None = None, log_path: str | Path | None = None,
        notify_on_exit: bool = True,
        owner: str = "run_background",
        environment: Mapping[str, str] | None = None,
    ) -> tuple[str, int]:
        """Start a task, taking ownership of supplied cwd and log handles."""
        pending = asyncio.get_running_loop().create_future()
        try:
            self._require_open()
            self._pending_starts.add(pending)
            return await self._start(
                command,
                cwd,
                cwd_fd=cwd_fd,
                log_path=log_path,
                notify_on_exit=notify_on_exit,
                owner=owner,
                environment=environment,
            )
        except BackgroundSpawnCleanupError as exc:
            self._start_cleanup_error = exc
            raise
        finally:
            self._pending_starts.discard(pending)
            pending.set_result(None)
            if cwd_fd is not None:
                os.close(cwd_fd)

    async def _start(
        self,
        command: str,
        cwd: str | Path,
        *,
        cwd_fd: int | None,
        log_path: str | Path | None,
        notify_on_exit: bool,
        owner: str,
        environment: Mapping[str, str] | None,
    ) -> tuple[str, int]:
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        effective_log_path = (
            Path(log_path) if log_path is not None else self._default_log_path(task_id)
        )
        with ExitStack() as cleanup:
            opener = (
                (lambda: self._open_task_log(task_id))
                if log_path is None
                else (lambda: self.open_log(effective_log_path))
            )
            log_handle = await open_log_off_loop(opener)
            log_handle = cleanup.enter_context(log_handle)
            self._require_open()
            try:
                spawn_kwargs = {
                    "stdin": asyncio.subprocess.PIPE,
                    "stdout": asyncio.subprocess.PIPE,
                    "stderr": asyncio.subprocess.STDOUT,
                    "env": tool_subprocess_env(environment),
                    "niceness": self.command_niceness,
                }
                if cwd_fd is None:
                    spawn = create_tool_subprocess_shell(
                        command, cwd=cwd, **spawn_kwargs
                    )
                else:
                    spawn = create_subprocess_shell_in_fd(
                        command, cwd_fd, **spawn_kwargs
                    )
                process = await spawn_background(
                    spawn,
                    require_open=self._require_open,
                    term_grace=self.term_grace,
                    cleanup_timeout=self.term_grace + self.stdin_drain_timeout,
                )
            except OSError as exc:
                # run_background's failed tool result is the canonical receipt;
                # persisting task_exited would render the same failure twice.
                # Macro-owned starts have no equivalent tool card, so retain
                # their durable notification.
                if (
                    notify_on_exit
                    and owner != "run_background"
                    and self._notification_sink is not None
                ):
                    await self._notification_sink(
                        TaskExitInput(
                            task_id=task_id,
                            command=command_headline(command),
                            exit_code=None,
                            output_tail=f"could not execute command: {exc}",
                            log_path=str(effective_log_path),
                            owner=owner,
                        )
                    )
                raise ValueError(f"could not execute command: {exc}") from exc
            # The monitor owns the log after the process starts.
            cleanup.pop_all()
        record = _BackgroundRecord(
            task_id=task_id,
            command=command,
            pid=process.pid,
            process=process,
            stdin=process.stdin,
            stdin_lock=asyncio.Lock(),
            output=bytearray(),
            started_at=time.monotonic(),
            log_path=str(effective_log_path),
            remove_log_on_archive=log_path is None,
            notify_on_exit=notify_on_exit,
            owner=owner,
        )
        try:
            await self._register_start(record, log_handle)
        except BaseException:
            await _terminate_process(
                process,
                self.term_grace,
                close_stdin=lambda: self._close_stdin(record),
            )
            owner_state = self._owners.get(record.task_id)
            if owner_state is None or owner_state.lifecycle is None:
                record.running = False
                record.process = None
            else:
                # test-contract: durability failed starts drain their owner lifecycle
                monitor_result = await asyncio.gather(
                    owner_state.lifecycle, return_exceptions=True
                )
                if isinstance(monitor_result[0], BaseException):
                    raise monitor_result[0]
            raise
        return task_id, process.pid

    async def _register_start(
        self, record: _BackgroundRecord, log_handle: IO[bytes]
    ) -> None:
        owner = _BackgroundTaskOwner(
            self, record, process=record.process, log_handle=log_handle
        )
        async with self._transition_lock:
            rows = self._persistence_rows(
                (*self._records.values(), record),
                include_notify_task_id=record.task_id,
            )

            def publish() -> None:
                self._records[record.task_id] = record
                self._owners[record.task_id] = owner
                owner.start()
                self._task_start_generation += 1

            await self._commit_transition(
                rows,
                publish,
                BackgroundTaskNotice(
                    f"background task {record.task_id} started: {command_headline(record.command)}",
                    record.task_id,
                    "started",
                    record.owner,
                ),
            )

    async def wait(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Wait for one background process and return its terminal status."""
        record = self._record(task_id)
        owner = self._owners[task_id]
        try:
            # test-contract: durability task wait observes the committed terminal row
            await owner.wait(timeout)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling() or record.running:
                raise
        return self._status(record)

    async def output(
        self,
        task_id: str,
        since: int | None = None,
        *,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        record = self._record(task_id)
        if since is not None and (type(since) is not int or since < 0):
            raise ValueError("since must be a nonnegative integer")
        if max_chars is not None and (type(max_chars) is not int or max_chars < 0):
            raise ValueError("max_chars must be a nonnegative integer")
        requested = 0 if since is None else min(since, record.total_bytes)
        start = requested
        if requested < record.base_cursor:
            available = await self._read_durable(record, start, self.call_limit + 1)
        else:
            output = bytes(record.output or b"")
            offset = start - record.base_cursor
            available = output[offset:]
        capped = len(available) > self.call_limit
        if capped:
            available = utf8_chunk(available, self.call_limit)
        if max_chars is not None:
            available = utf8_char_chunk(available, max_chars)
        next_cursor = start + len(available)
        remaining_bytes = max(0, record.total_bytes - next_cursor)
        result: dict[str, Any] = {
            "task_id": task_id,
            "output": available.decode(errors="replace"),
            "cursor": next_cursor,
            "running": record.running,
            "exit_code": record.exit_code,
            "has_more": remaining_bytes > 0,
            "total_bytes": record.total_bytes,
            "remaining_bytes": remaining_bytes,
            "output_location": f"task-output://{task_id}",
        }
        if record.note is not None:
            result["note"] = record.note
        return result

    async def input(
        self,
        task_id: str,
        data: str = "",
        *,
        eof: bool = False,
    ) -> dict[str, Any]:
        """Write one bounded UTF-8 chunk to a live task, optionally closing EOF."""

        record = self._record(task_id)
        if type(data) is not str:
            raise ValueError("stdin data must be a string")
        encoded = data.encode("utf-8")
        if len(encoded) > BACKGROUND_STDIN_LIMIT:
            raise ValueError(
                f"stdin data exceeds the {BACKGROUND_STDIN_LIMIT}-byte limit"
            )
        if type(eof) is not bool:
            raise ValueError("eof must be a boolean")
        lock = record.stdin_lock
        if lock is None:
            raise ValueError("background task stdin is closed")
        async with lock:
            if not record.running:
                raise ValueError("background task is not running")
            if record.stdin_closed or record.stdin is None:
                if eof and not data:
                    return {
                        "task_id": task_id,
                        "bytes_written": 0,
                        "eof": True,
                        "status": "committed",
                        "retry": False,
                    }
                raise ValueError("background task stdin is closed")
            writer = record.stdin
            if encoded:
                writer.write(encoded)
                committed = await self._drain_stdin(writer)
                if not committed:
                    record.stdin_closed = True
                    record.stdin = None
                    await self._abort_stdin(writer)
                    return {
                        "task_id": task_id,
                        "bytes_written": len(encoded),
                        "eof": eof,
                        "status": "indeterminate",
                        "retry": False,
                    }
            if eof:
                record.stdin_closed = True
                record.stdin = None
                writer.close()
            return {
                "task_id": task_id,
                "bytes_written": len(encoded),
                "eof": eof,
                "status": "committed",
                "retry": False,
            }

    def begin_shutdown(self, cause: str = "session shutdown") -> None:
        """Mark cancellations caused by session teardown as shutdown kills."""
        self._shutdown_cause = self._shutdown_cause or cause

    async def kill(
        self, task_id: str, *, cause: str, tui_presented: bool = False
    ) -> dict[str, Any]:
        self._record(task_id)
        outcome = await self._owners[task_id].terminate(
            self._shutdown_cause or cause,
            "task_kill",
            tui_presented,
        )
        result = self._status(self._record(task_id))
        result.update(
            killed_by_this_call=outcome.killed_by_this_call,
            cause=outcome.cause,
            status=outcome.status,
        )
        return result

    async def close(self) -> tuple[str, ...]:
        if self._closed:
            return ()
        done = self._close_done
        if done is None:
            # The first caller runs shutdown inline. A separately scheduled
            # task could be cancelled by event-loop teardown before its first
            # instruction; inline execution reaches the cancellation-safe
            # shutdown body without that gap. Concurrent callers share `done`.
            self._closing = True
            done = asyncio.get_running_loop().create_future()
            self._close_done = done
            try:
                result = await self._shutdown()
            except asyncio.CancelledError:
                if not done.done():
                    done.set_result(self._shutdown_killed)
                raise
            except BaseException as exc:
                if not done.done():
                    done.set_exception(exc)
                    done.exception()  # retrieved here; awaiting callers re-raise it
                self._close_done = None
                self._close_failure_generation += 1
                raise
            done.set_result(result)
            return result
        cancelled = False
        while not done.done():
            try:
                # test-contract: durability close ownership survives caller cancellation
                await asyncio.shield(done)
            except asyncio.CancelledError:
                cancelled = True
        result = done.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _finish_owned(self, awaitable: Any) -> Any:
        task = asyncio.ensure_future(awaitable)
        cancelled = False
        while not task.done():
            try:
                # test-contract: durability owned registry steps survive cancellation
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _terminate_owned(
        self,
        processes: Sequence[asyncio.subprocess.Process],
        *,
        after_signal: Callable[[], None] | None = None,
    ) -> None:
        cancelled = False

        async def finish_step(awaitable: Any) -> Any:
            nonlocal cancelled
            task = asyncio.ensure_future(awaitable)
            while not task.done():
                try:
                    # test-contract: durability termination finishes its claimed step
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            return task.result()

        await _terminate_processes(
            processes,
            self.term_grace,
            finish=finish_step,
            after_signal=after_signal,
        )
        if cancelled:
            raise asyncio.CancelledError

    async def _shutdown(self) -> tuple[str, ...]:
        cancelled = False

        async def finish(awaitable: Any) -> Any:
            nonlocal cancelled
            try:
                return await self._finish_owned(awaitable)
            except asyncio.CancelledError:
                cancelled = True
                return None

        if self._pending_starts:
            await finish(asyncio.gather(*self._pending_starts))
        owners = [
            owner
            for owner in self._owners.values()
            if owner.record.running and owner.process is not None
        ]
        killed = tuple(owner.record.task_id for owner in owners)
        cause = self._shutdown_cause or "background task registry closed"
        if owners:
            await finish(
                asyncio.gather(*(owner.finish_for_close(cause) for owner in owners))
            )
            processes = [owner.process for owner in owners if owner.process is not None]

            output_drain: asyncio.Task[None] | None = None

            async def finish_output_drain() -> None:
                await asyncio.sleep(BACKGROUND_OUTPUT_DRAIN_TIMEOUT)
                for owner in owners:
                    owner.finish_close_output_drain()

            def schedule_output_drain() -> None:
                nonlocal output_drain
                output_drain = asyncio.create_task(finish_output_drain())

            await finish(
                self._terminate_owned(processes, after_signal=schedule_output_drain)
            )
            if output_drain is not None:
                await finish(output_drain)
            await finish(
                asyncio.gather(*(owner.wait_until_durable() for owner in owners))
            )
            self._notice(
                BackgroundTaskShutdownNotice(
                    "background tasks killed on session exit: " + ", ".join(killed),
                    tasks=tuple(
                        (owner.record.task_id, owner.record.owner) for owner in owners
                    ),
                )
            )
            await finish(
                asyncio.gather(*(owner.release_close_receipt() for owner in owners))
            )
        if self._start_cleanup_error is not None:
            raise self._start_cleanup_error
        self._closing = False
        self._closed = True
        self._shutdown_killed = killed
        self._close_storage()
        if self._storage_lease is not None:
            self._storage_lease.close()
            self._storage_lease = None
        if self._temporary_log_root is not None:
            shutil.rmtree(self._temporary_log_root, ignore_errors=True)
            self._temporary_log_root = None
        if cancelled:
            raise asyncio.CancelledError
        return killed

    def _close_storage(self) -> None:
        self.release_directory()

    def _require_open(self) -> None:
        if self._closing:
            raise RuntimeError("background task registry is closing")
        if self._closed:
            raise RuntimeError("background task registry is closed")

    def release_directory(self) -> None:
        """Release storage after shutdown or before activation on setup failure."""
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        if self._directory_fd is not None:
            self._release_directory()
            self._directory_fd = None

    def _default_log_path(self, task_id: str) -> Path:
        root = self._session_dir or self._temporary_log_root
        if root is None:
            raise RuntimeError("background log storage is unavailable")
        return root / f"background-{task_id}.log"

    async def _read_durable(
        self, record: _BackgroundRecord, start: int, limit: int
    ) -> bytes:
        if limit <= 0:
            return b""
        owner = self._owners[record.task_id]
        async with owner.storage_lock:
            if owner.log_fd is not None:
                return await run_owned_thread_call(
                    lambda: os.pread(owner.log_fd, limit, start)
                )
            if self._archive is None:
                return b""
            return await run_owned_thread_call(
                lambda: self._archive.pread(record.task_id, start, limit)
            )

    async def _read_output(
        self,
        record: _BackgroundRecord,
        stream: asyncio.StreamReader,
        log_writer: _OrderedLogWriter | None = None,
    ) -> None:
        while chunk := await stream.read(65_536):
            if log_writer is not None:
                await log_writer.write(chunk)
            self._append(record, chunk)

    async def _drain_stdin(self, writer: asyncio.StreamWriter) -> bool:
        """Drain queued bytes without letting cancellation make them retryable."""

        drain = asyncio.create_task(writer.drain())
        committed = False
        try:
            try:
                await asyncio.wait_for(
                    asyncio.shield(drain), timeout=self.stdin_drain_timeout
                )
            except asyncio.CancelledError:
                # A caller cancellation cannot retract bytes already handed to
                # write(). Give the transport the same bounded chance to commit.
                await asyncio.wait_for(
                    asyncio.shield(drain), timeout=self.stdin_drain_timeout
                )
            committed = True
        except (
            asyncio.TimeoutError,
            BrokenPipeError,
            ConnectionError,
            OSError,
            asyncio.CancelledError,
        ):
            return False
        finally:
            if not committed and not drain.done():
                drain.cancel()
                # test-contract: durability canceled stdin drains are fully reaped
                await asyncio.gather(drain, return_exceptions=True)
        return True

    async def _abort_stdin(self, writer: asyncio.StreamWriter) -> None:
        transport = writer.transport
        transport.abort()
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), timeout=self.term_grace)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionError, OSError):
            pass

    async def _close_stdin(self, record: _BackgroundRecord) -> None:
        lock = record.stdin_lock
        if lock is None:
            return
        async with lock:
            if record.stdin_closed:
                return
            record.stdin_closed = True
            writer = record.stdin
            record.stdin = None
            if writer is not None:
                writer.close()

    def _append(self, record: _BackgroundRecord, chunk: bytes) -> None:
        if record.output is None:
            record.output = bytearray()
        record.output.extend(chunk)
        record.total_bytes += len(chunk)
        record.total_lines += chunk.count(b"\n")
        overflow = len(record.output) - self.output_limit
        if overflow > 0:
            trim = overflow
            while trim < len(record.output) and record.output[trim] & 0xC0 == 0x80:
                trim += 1
            del record.output[:trim]
            record.base_cursor += trim

    def _record(self, task_id: str) -> _BackgroundRecord:
        if type(task_id) is not str or not task_id:
            raise ValueError("task_id must be a nonempty string")
        try:
            return self._records[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown background task: {task_id}") from exc

    @staticmethod
    def _status(record: _BackgroundRecord) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": record.task_id,
            "running": record.running,
            "exit_code": record.exit_code,
        }
        if record.note is not None:
            result["note"] = record.note
        return result

    def terminal_metadata(self, task_id: str) -> tuple[str, str] | None:
        """Return internal owner/phase metadata without changing tool results."""

        record = self._record(task_id)
        if record.terminal_phase is None:
            return None
        return record.owner, record.terminal_phase

    def _notice(self, notice: BackgroundTaskNoticeSinkValue) -> None:
        if self._notice_sink is not None:
            self._notice_sink(notice)

    def _load_previous(self) -> None:
        if self._directory_fd is None:
            return
        try:
            rows = load_session_json(
                read_session_file(self._directory_fd, "background_tasks.json")
            )
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            self._persistence_corrupt = True
            return
        if type(rows) is not list:
            self._persistence_corrupt = True
            return
        for row in rows:
            if type(row) is not dict:
                continue
            task_id = row.get("task_id")
            command = row.get("command")
            pid = row.get("pid")
            if (
                type(task_id) is not str
                or type(command) is not str
                or type(pid) is not int
            ):
                continue
            was_running = row.get("running") is True
            archived_length = self._archived_lengths.get(task_id, 0)
            note = row.get("note")
            phase = row.get("terminal_phase")
            legacy_pending = row.get("receipt_pending") is True
            keyed = "notify_on_exit" in row
            notify_on_exit = row.get("notify_on_exit", True) is not False
            delivered = self._notification_exists(task_id)
            tui_presented = row.get("tui_presented") is True
            owner_name = row.get("owner")
            record = _BackgroundRecord(
                task_id=task_id,
                command=command,
                pid=pid,
                output=bytearray(),
                total_bytes=archived_length,
                base_cursor=archived_length,
                running=False,
                exit_code=(
                    row.get("exit_code")
                    if type(row.get("exit_code")) is int
                    else None
                ),
                note=(
                    note
                    if type(note) is str
                    else "task exited when the previous session ended"
                ),
                notify_on_exit=notify_on_exit,
                tui_presented=tui_presented,
                owner=owner_name if type(owner_name) is str else "run_background",
                terminal_phase=(
                    phase
                    if phase in {"natural_exit", "task_kill", "session_shutdown"}
                    else None
                ),
            )
            owner = _BackgroundTaskOwner(self, record)
            self._records[task_id] = record
            self._owners[task_id] = owner
            should_recover = notify_on_exit and not delivered and (
                was_running or legacy_pending or keyed
            )
            # Old terminal rows without the key were already treated as delivered.
            if should_recover:
                owner._set_state(owner.state.TERMINAL_DURABLE)
                self._pending_recovery[task_id] = TaskExitInput(
                    task_id=task_id,
                    command=command_headline(command),
                    exit_code=None if was_running else record.exit_code,
                    note=(
                        "exit not observed (zeta restarted)"
                        if was_running
                        else record.note
                    ),
                    owner=record.owner,
                    phase=record.terminal_phase or "natural_exit",
                    tui_presented=record.tui_presented,
                )

    async def _commit_transition(
        self,
        rows: tuple[dict[str, object], ...],
        publish: Callable[[], None],
        notice: BackgroundTaskNoticeSinkValue | None,
    ) -> None:
        worker = asyncio.create_task(asyncio.to_thread(self._persist, rows))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        worker.result()
        publish()
        if notice is not None:
            self._notice(notice)
        if cancelled:
            raise asyncio.CancelledError

    def _persistence_rows(
        self,
        records: Sequence[_BackgroundRecord] | None = None,
        *,
        include_notify_task_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for record in self._records.values() if records is None else records:
            row: dict[str, object] = {
                "task_id": record.task_id,
                "command": record.command,
                "pid": record.pid,
                "running": record.running,
                "exit_code": record.exit_code,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
                "note": record.note,
                "terminal_phase": record.terminal_phase,
                "tui_presented": record.tui_presented,
                "owner": record.owner,
            }
            if record.task_id == include_notify_task_id:
                row["notify_on_exit"] = record.notify_on_exit
            rows.append(row)
        return tuple(rows)

    def _persist(self, rows: Sequence[dict[str, object]] | None = None) -> None:
        if (
            self._directory_fd is None
            or self._temporary_log_root is not None
            or self._persistence_corrupt
            or self._storage_lease is None
        ):
            return
        self._storage_lease.publish_records(
            self._directory_fd,
            "background_tasks.json",
            self._persistence_rows() if rows is None else rows,
            identity="task_id",
            delete_ids=(),
            publish=write_session_file,
        )


def background_tasks_for_session(
    store: ConversationStore | None, *, command_niceness: int
) -> BackgroundTaskRegistry:
    if store is None:
        return BackgroundTaskRegistry(command_niceness=command_niceness)
    return BackgroundTaskRegistry(
        session_dir=store.session_dir,
        directory_fd=store.directory_fd,
        command_niceness=command_niceness,
        storage_lease=store._storage_lease,
        notification_exists=store.has_task_exit_notification,
    )


def bind_background_tasks_to_session(
    tasks: BackgroundTaskRegistry, store: ConversationStore
) -> None:
    tasks.set_notification_exists(store.has_task_exit_notification)
    tasks.bind_session_dir(store.session_dir, store.directory_fd, store._storage_lease)
