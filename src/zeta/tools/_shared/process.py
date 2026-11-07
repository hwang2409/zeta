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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Literal

from ...core.checkpoints import load_session_json
from ...core.process_env import subprocess_env as tool_subprocess_env
from ...core.session_files import (
    open_session_file,
    read_session_file,
    session_root,
    write_session_json,
)
from .background_log_writer import _OrderedLogWriter
from .background_output_archive import _BackgroundOutputArchive
from .background_spawn import (
    BackgroundSpawnCleanupError,
    _group_exists,
    _terminate_process,
    _terminate_processes,
    create_subprocess_shell_in_fd,
    spawn_background,
)
from .process_output import command_headline, output_tail, utf8_char_chunk, utf8_chunk
from .tool_spawn import create_tool_subprocess_shell

logger = logging.getLogger(__name__)


BACKGROUND_OUTPUT_LIMIT = 512 * 1024
BACKGROUND_OUTPUT_CALL_LIMIT = 32 * 1024
BACKGROUND_TERM_GRACE_SECONDS = 0.25
BACKGROUND_STDIN_LIMIT = 64 * 1024
BACKGROUND_STDIN_DRAIN_TIMEOUT = 1.0
BACKGROUND_OUTPUT_DRAIN_TIMEOUT = 0.25
OUTPUT_CUTOFF_NOTE = (
    "background output was cut off because a process outside the task still held its output"
)
@dataclass(frozen=True, slots=True)
class BackgroundTaskNotice:
    """Structured lifecycle notice emitted by a background task owner."""

    message: str
    task_id: str
    phase: Literal[
        "started", "natural_exit", "task_kill", "session_shutdown"
    ]
    owner: str


@dataclass(frozen=True, slots=True)
class BackgroundTaskShutdownNotice:
    """Structured notice emitted when the registry kills tasks on shutdown."""

    message: str
    phase: Literal["session_shutdown"] = "session_shutdown"
    tasks: tuple[tuple[str, str], ...] = ()


BackgroundTaskNoticeSinkValue = BackgroundTaskNotice | BackgroundTaskShutdownNotice


@dataclass(frozen=True, slots=True)
class TaskExitInput:
    """Detached task-exit data for the resident delivery flow."""

    task_id: str
    command: str
    exit_code: int | None
    output_tail: str = ""
    log_path: str | None = None
    note: str | None = None
    owner: str = "run_background"
    phase: str = "natural_exit"


@dataclass(frozen=True, slots=True)
class BackgroundTaskInfo:
    """Immutable view of one background task for read-only inspection.

    The timestamps use ``time.monotonic`` so a consumer computes runtime
    against ``time.monotonic()`` without a wall-clock jump corrupting it.
    ``started_at`` is ``None`` for a task recovered from a previous session,
    whose runtime is not knowable.
    """

    task_id: str
    command: str
    pid: int
    owner: str
    running: bool
    exit_code: int | None
    note: str | None
    terminal_phase: str | None
    started_at: float | None
    ended_at: float | None
    output_bytes: int
    output_lines: int

    @property
    def headline(self) -> str:
        return command_headline(self.command)


@dataclass(slots=True)
class _BackgroundRecord:
    task_id: str
    command: str
    pid: int
    process: asyncio.subprocess.Process | None = None
    stdin: asyncio.StreamWriter | None = None
    stdin_lock: asyncio.Lock | None = None
    stdin_closed: bool = False
    output: bytearray | None = None
    total_bytes: int = 0
    total_lines: int = 0
    base_cursor: int = 0
    started_at: float | None = None
    ended_at: float | None = None
    running: bool = True
    exit_code: int | None = None
    note: str | None = None
    log_path: str | None = None
    remove_log_on_archive: bool = False
    log_fd: int | None = None
    storage_lock: asyncio.Lock | None = None
    notify_on_exit: bool = True
    owner: str = "run_background"
    terminal_phase: Literal[
        "natural_exit", "task_kill", "session_shutdown"
    ] | None = None
    monitor: asyncio.Task[None] | None = None


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
        command_niceness: int = 10,
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
        self._pending_recovery: dict[str, str] = {}
        self._records: dict[str, _BackgroundRecord] = {}
        self._session_dir: Path | None = None
        self._directory_fd: int | None = None
        self._archive: _BackgroundOutputArchive | None = None
        self._archived_lengths: dict[str, int] = {}
        self._archive_workers: set[asyncio.Task[None]] = set()
        self._temporary_log_root: Path | None = None
        self._closing = False
        self._closed = False
        self._close_done: asyncio.Future[tuple[str, ...]] | None = None
        self._close_failure_generation = 0
        self._pending_starts: set[asyncio.Future[None]] = set()
        self._transition_lock = asyncio.Lock()
        self._pending_terminals: dict[
            str, tuple[_BackgroundRecord, BackgroundTaskNotice]
        ] = {}
        self._task_start_generation = 0
        self._start_cleanup_error: BackgroundSpawnCleanupError | None = None
        self._shutdown_killed: tuple[str, ...] = ()
        self._closing_for_shutdown = False
        if session_dir is not None:
            if directory_fd is None:
                raise ValueError("session directory descriptor is required")
            self.bind_session_dir(session_dir, directory_fd)
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

    def set_notice_sink(self, sink: Callable[[BackgroundTaskNoticeSinkValue], None] | None) -> None:
        self._notice_sink = sink

    def set_notification_sink(
        self,
        sink: Callable[[TaskExitInput], Awaitable[object]] | None,
    ) -> tuple[TaskExitInput, ...]:
        self._notification_sink = sink
        if sink is None:
            return ()
        recovered = tuple(
            TaskExitInput(
                task_id=task_id,
                command=command_headline(command),
                exit_code=None,
                note="exit not observed (zeta restarted)",
            )
            for task_id, command in self._pending_recovery.items()
        )
        self._pending_recovery.clear()
        return recovered

    def bind_session_dir(self, session_dir: str | Path, directory_fd: int) -> None:
        self._require_open()
        if self._directory_fd is not None and self._temporary_log_root is None:
            if not os.path.samestat(os.fstat(self._directory_fd), os.fstat(directory_fd)):
                raise ValueError("background task registry is already bound")
            return
        if self._records:
            raise ValueError("cannot bind a registry after background tasks have started")
        temporary_root = self._temporary_log_root
        self._close_storage()
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
            self._temporary_log_root = None
        self._bind_storage(Path(session_dir), directory_fd)
        try:
            self._load_previous()
        except BaseException:
            self._close_storage()
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
        return os.fdopen(
            self._archive.open_task_log(task_id), "w+b", buffering=0
        )

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
        cwd_fd: int | None = None,
        log_path: str | Path | None = None,
        notify_on_exit: bool = True,
        owner: str = "run_background",
        environment: Mapping[str, str] | None = None,
    ) -> tuple[str, int]:
        """Start a task, taking ownership of ``cwd_fd`` when one is supplied."""

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
        self._require_open()
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        effective_log_path = (
            Path(log_path) if log_path is not None else self._default_log_path(task_id)
        )
        with ExitStack() as cleanup:
            if log_path is None:
                if self._archive is None:
                    raise RuntimeError("background output storage is unavailable")
                log_handle = cleanup.enter_context(self._open_task_log(task_id))
            else:
                log_handle = cleanup.enter_context(self.open_log(effective_log_path))
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
                    spawn, require_open=self._require_open,
                    term_grace=self.term_grace,
                    cleanup_timeout=self.term_grace + self.stdin_drain_timeout,
                )
            except OSError as exc:
                # run_background's failed tool result is the canonical receipt;
                # persisting task_exited would render the same failure twice.
                # Macro-owned starts have no equivalent tool card, so retain
                # their durable notification.
                if notify_on_exit and owner != "run_background":
                    await self._notify_exit(
                        task_id,
                        command,
                        None,
                        f"could not execute command: {exc}",
                        effective_log_path,
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
            log_fd=log_handle.fileno(),
            storage_lock=asyncio.Lock(),
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
            if record.monitor is None:
                record.running = False
                record.process = None
            else:
                monitor_result = await asyncio.gather(
                    record.monitor, return_exceptions=True
                )
                if isinstance(monitor_result[0], BaseException):
                    raise monitor_result[0]
            raise
        return task_id, process.pid

    async def _register_start(self, record: _BackgroundRecord, log_handle: IO[bytes]) -> None:
        async with self._transition_lock:
            rows = self._persistence_rows((*self._records.values(), record))
            def publish() -> None:
                self._records[record.task_id] = record
                record.monitor = asyncio.create_task(self._monitor(record, log_handle))
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
        if record.monitor is not None:
            if timeout is None:
                await record.monitor
            else:
                await asyncio.wait((record.monitor,), timeout=timeout)
                if record.monitor.done():
                    await record.monitor
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
            available = await self._read_durable(
                record, start, self.call_limit + 1
            )
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

    def begin_shutdown(self) -> None:
        """Mark cancellations caused by session teardown as shutdown kills."""

        self._closing_for_shutdown = True

    async def kill(self, task_id: str) -> dict[str, Any]:
        record = self._record(task_id)
        if record.running and record.process is not None:
            phase: Literal["task_kill", "session_shutdown"] = (
                "session_shutdown" if self._closing_for_shutdown else "task_kill"
            )
            await self._terminate(
                record,
                reason="task killed on session exit" if phase == "session_shutdown" else "task killed",
                phase=phase,
            )
        return self._status(record)

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
                await asyncio.shield(done)
            except asyncio.CancelledError:
                cancelled = True
        result = done.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _shutdown(self) -> tuple[str, ...]:
        killed: list[str] = []
        shutdown_result: tuple[str, ...] = ()
        cancelled = False

        async def finish(awaitable: Any) -> Any:
            nonlocal cancelled
            task = asyncio.ensure_future(awaitable)
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            if task.cancelled():
                cancelled = True
                return None
            return task.result()

        try:
            # Only wait when a spawn is in flight: empty child registries must
            # not gain a scheduling boundary before completion is persisted.
            if self._pending_starts:
                await finish(asyncio.gather(*self._pending_starts))
            if self._pending_terminals:
                async with self._transition_lock:
                    await finish(self._retry_pending_terminals())
            entries = [
                (record, record.process)
                for record in self._records.values()
                if record.running and record.process is not None
            ]
            records = [record for record, _ in entries]
            killed = [record.task_id for record in records]
            # Whole-session and child-completion shutdown kills silently; callers
            # surface the ids (child receipt) instead. Prepare every record before
            # signaling so that monitors cannot emit notifications during shutdown.
            for record in records:
                record.notify_on_exit = False
                record.terminal_phase = "session_shutdown"
                record.note = "task killed on session exit"
            await finish(
                asyncio.gather(
                    *(self._close_stdin(record) for record in records)
                )
            )
            await finish(
                _terminate_processes(
                    [process for _, process in entries],
                    self.term_grace,
                    finish=finish,
                )
            )

            cleanup_timeout = self.term_grace + self.stdin_drain_timeout
            if entries:
                try:
                    await finish(
                        asyncio.wait_for(
                            asyncio.gather(
                                *(process.wait() for _, process in entries)
                            ),
                            timeout=cleanup_timeout,
                        )
                    )
                except asyncio.TimeoutError:
                    pass

                async def wait_for_groups_to_exit() -> bool:
                    loop = asyncio.get_running_loop()
                    deadline = loop.time() + cleanup_timeout
                    while any(
                        _group_exists(process.pid) for _, process in entries
                    ):
                        if loop.time() >= deadline:
                            return False
                        await asyncio.sleep(0.01)
                    return True

                if not await finish(wait_for_groups_to_exit()):
                    raise RuntimeError(
                        "background process groups survived registry shutdown"
                    )
            monitors = [
                record.monitor for record in records if record.monitor is not None
            ]
            if monitors:
                monitor_result = await finish(
                    asyncio.wait(
                        monitors,
                        timeout=self.term_grace + self.stdin_drain_timeout,
                    )
                )
                pending = set(monitors) if monitor_result is None else monitor_result[1]
                for record in records:
                    if record.monitor in pending:
                        record.note = f"{record.note}; {OUTPUT_CUTOFF_NOTE}"
                for monitor in pending:
                    monitor.cancel()
                if pending:
                    await finish(
                        asyncio.wait(
                            pending,
                            timeout=self.term_grace + self.stdin_drain_timeout,
                        )
                    )
            shutdown_notice = (
                BackgroundTaskShutdownNotice(
                    "background tasks killed on session exit: " + ", ".join(killed),
                    tasks=tuple(
                        (task_id, self._records[task_id].owner) for task_id in killed
                    ),
                )
                if killed
                else None
            )
            async with self._transition_lock:
                await self._commit_notice(shutdown_notice)
            if self._start_cleanup_error is not None:
                raise self._start_cleanup_error
            self._closing = False
            self._closed = True
            self._shutdown_killed = tuple(killed)
            shutdown_result = tuple(killed)
        finally:
            await finish(self._wait_for_archive_workers())
        self._close_storage()
        if self._temporary_log_root is not None:
            shutil.rmtree(self._temporary_log_root, ignore_errors=True)
            self._temporary_log_root = None
        if cancelled:
            raise asyncio.CancelledError
        return shutdown_result

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
        lock = record.storage_lock
        if lock is None:
            return b""
        async with lock:
            if record.log_fd is not None:
                return await asyncio.to_thread(os.pread, record.log_fd, limit, start)
            if self._archive is None:
                return b""
            return await asyncio.to_thread(
                self._archive.pread,
                record.task_id,
                start,
                limit,
            )

    async def _archive_record(
        self, record: _BackgroundRecord, log_handle: IO[bytes]
    ) -> None:
        lock = record.storage_lock
        if lock is None or record.log_fd is None:
            return
        async with lock:
            if self._archive is None or record.log_fd is None:
                raise RuntimeError("background output storage is unavailable")
            worker = asyncio.create_task(
                asyncio.to_thread(
                    self._archive.append,
                    record.task_id,
                    record.log_fd,
                    record.total_bytes,
                )
            )
            self._archive_workers.add(worker)
            worker.add_done_callback(self._archive_workers.discard)
            try:
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    await worker
                    self._archived_lengths[record.task_id] = record.total_bytes
                    raise
                self._archived_lengths[record.task_id] = record.total_bytes
            finally:
                log_handle.close()
                record.log_fd = None

    async def _wait_for_archive_workers(self) -> None:
        while self._archive_workers:
            await asyncio.gather(
                *tuple(self._archive_workers), return_exceptions=True
            )

    async def _monitor(self, record: _BackgroundRecord, log_handle: Any | None = None) -> None:
        process = record.process
        if process is None or process.stdout is None:
            return
        log_writer = _OrderedLogWriter(log_handle) if log_handle is not None else None
        reader = asyncio.create_task(
            self._read_output(record, process.stdout, log_writer)
        )
        try:
            await process.wait()
            await reader
            while _group_exists(process.pid):
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            raise
        finally:
            await self._close_stdin(record)
            if log_writer is not None:
                await log_writer.close()
            archive_error: BaseException | None = None
            if log_handle is not None:
                try:
                    await self._archive_record(record, log_handle)
                except BaseException as exc:
                    archive_error = exc
                    record.note = (
                        "background output archive failed; recovery will retry the retained log"
                    )
                    logger.exception(
                        "failed to archive background task output for %s", record.task_id
                    )
            await self._commit_terminal(record, process.returncode)
            if archive_error is not None:
                raise archive_error

    async def _commit_terminal(self, record: _BackgroundRecord, exit_code: int | None) -> bool:
        if not record.running:
            return False
        finalization = asyncio.create_task(self._finalize_terminal(record, exit_code))
        cancelled = False
        while not finalization.done():
            try:
                await asyncio.shield(finalization)
            except asyncio.CancelledError:
                cancelled = True
        committed = finalization.result()
        if cancelled:
            raise asyncio.CancelledError
        return committed

    async def _finalize_terminal(self, record: _BackgroundRecord, exit_code: int | None) -> bool:
        async with self._transition_lock:
            if not record.running:
                return False
            candidate = replace(
                record,
                running=False,
                exit_code=exit_code,
                ended_at=time.monotonic(),
                process=None,
                terminal_phase=record.terminal_phase or "natural_exit",
            )
            rows = self._persistence_rows(tuple(
                candidate if item is record else item for item in self._records.values()
            ))
            notice = BackgroundTaskNotice(
                f"background task {record.task_id} exited ({candidate.exit_code}): "
                f"{command_headline(record.command)}",
                record.task_id,
                candidate.terminal_phase,
                record.owner,
            )
            try:
                await self._commit_transition(
                    rows,
                    lambda: self._publish_terminal(record, candidate),
                    notice,
                )
            except Exception:
                self._pending_terminals[record.task_id] = candidate, notice
                raise
            self._pending_terminals.pop(record.task_id, None)
        await self._notify_terminal_exit(candidate)
        return True

    @staticmethod
    def _publish_terminal(record: _BackgroundRecord, candidate: _BackgroundRecord) -> None:
        record.running = candidate.running
        record.exit_code = candidate.exit_code
        record.ended_at = candidate.ended_at
        record.process = candidate.process
        record.terminal_phase = candidate.terminal_phase

    async def _retry_pending_terminals(self) -> None:
        pending = tuple(self._pending_terminals.items())
        candidates = {task_id: value[0] for task_id, value in pending}
        rows = self._persistence_rows(
            tuple(
                candidates.get(record.task_id, record)
                for record in self._records.values()
            )
        )

        def publish() -> None:
            for task_id, (candidate, notice) in pending:
                record = self._records[task_id]
                self._publish_terminal(record, candidate)
                self._notice(notice)
                self._pending_terminals.pop(task_id, None)

        await self._commit_transition(rows, publish, None)
        for _, (candidate, _) in pending:
            await self._notify_terminal_exit(candidate)

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

    async def _terminate(
        self,
        record: _BackgroundRecord,
        *,
        reason: str,
        phase: Literal["task_kill", "session_shutdown"],
    ) -> None:
        process = record.process
        if process is None:
            return
        # Set the phase before signaling: the monitor commonly wins the race to
        # finalize the record and must preserve the initiating lifecycle event.
        record.terminal_phase = phase
        record.note = reason
        await _terminate_process(
            process, self.term_grace, close_stdin=lambda: self._close_stdin(record)
        )
        monitor_cutoff = False
        if record.monitor is not None and record.monitor is not asyncio.current_task():
            done, _ = await asyncio.wait(
                (record.monitor,), timeout=BACKGROUND_OUTPUT_DRAIN_TIMEOUT
            )
            if not done:
                monitor_cutoff = True
                record.monitor.cancel()
            (monitor_result,) = await asyncio.gather(
                record.monitor, return_exceptions=True
            )
            if isinstance(monitor_result, BaseException) and not isinstance(
                monitor_result, asyncio.CancelledError
            ):
                raise monitor_result
        if monitor_cutoff:
            record.note = f"{record.note}; {OUTPUT_CUTOFF_NOTE}"
        await self._commit_terminal(record, process.returncode)

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

    async def _notify_terminal_exit(self, record: _BackgroundRecord) -> None:
        if not record.notify_on_exit:
            return
        await self._notify_exit(
            record.task_id,
            record.command,
            record.exit_code,
            output_tail(record.output),
            None if record.remove_log_on_archive else record.log_path,
            record.note,
            owner=record.owner,
            phase=record.terminal_phase or "natural_exit",
        )

    async def _notify_exit(
        self,
        task_id: str,
        command: str,
        exit_code: int | None,
        output_tail: str,
        log_path: str | Path | None,
        note: str | None = None,
        *,
        owner: str = "run_background",
        phase: str = "natural_exit",
    ) -> None:
        if self._notification_sink is None:
            return
        await self._notification_sink(
            TaskExitInput(
                task_id=task_id,
                command=command_headline(command),
                exit_code=exit_code,
                output_tail=output_tail,
                log_path=str(log_path) if log_path is not None else None,
                note=note,
                owner=owner,
                phase=phase,
            )
        )

    def _load_previous(self) -> None:
        if self._directory_fd is None:
            return
        try:
            rows = load_session_json(read_session_file(self._directory_fd, "background_tasks.json"))
        except (OSError, ValueError):
            return
        if type(rows) is not list:
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
            self._records[task_id] = _BackgroundRecord(
                task_id=task_id,
                command=command,
                pid=pid,
                output=bytearray(),
                total_bytes=archived_length,
                base_cursor=archived_length,
                running=False,
                exit_code=row.get("exit_code") if type(row.get("exit_code")) is int else None,
                note="task exited when the previous session ended",
                storage_lock=asyncio.Lock(),
            )
            if was_running:
                self._pending_recovery[task_id] = command

    async def _commit_notice(
        self, notice: BackgroundTaskNoticeSinkValue | None
    ) -> None:
        await self._commit_transition(self._persistence_rows(), lambda: None, notice)

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
        self, records: Sequence[_BackgroundRecord] | None = None
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "task_id": record.task_id,
                "command": record.command,
                "pid": record.pid,
                "running": record.running,
                "exit_code": record.exit_code,
                "started_at": record.started_at,
                "ended_at": record.ended_at,
            }
            for record in (
                self._records.values() if records is None else records
            )
        )

    def _persist(self, rows: Sequence[dict[str, object]] | None = None) -> None:
        if self._directory_fd is None or self._temporary_log_root is not None:
            return
        write_session_json(
            self._directory_fd,
            "background_tasks.json",
            self._persistence_rows() if rows is None else rows,
        )
