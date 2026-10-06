"""Process-group cleanup shared by shell tools."""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
import sys
import time
import uuid
import weakref
from collections.abc import Callable, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from ...core.checkpoints import load_session_json
from ...core.process_env import subprocess_env
from ...core.session_files import (
    open_session_file,
    read_session_file,
    session_root,
    write_session_json,
)
from ...core.store import ConversationStore


def tool_subprocess_env() -> dict[str, str]:
    """Return the scrubbed parent environment for a tool child."""

    return subprocess_env()


_FD_SHELL_WRAPPER = (
    "import os,sys; os.fchdir(int(sys.argv[1])); "
    "os.execvpe('/bin/sh', ['sh','-c',sys.argv[2]], os.environ)"
)


async def create_subprocess_shell_in_fd(
    command: str, directory_fd: int, **kwargs: object
) -> asyncio.subprocess.Process:
    """Run a shell after fchdir'ing an inherited, identity-verified fd."""

    kwargs.pop("cwd", None)
    pass_fds = tuple(kwargs.pop("pass_fds", ()))
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _FD_SHELL_WRAPPER,
        str(directory_fd),
        command,
        pass_fds=(*pass_fds, directory_fd),
        **kwargs,
    )


BACKGROUND_OUTPUT_LIMIT = 512 * 1024
BACKGROUND_OUTPUT_CALL_LIMIT = 32 * 1024
BACKGROUND_TERM_GRACE_SECONDS = 0.25
BACKGROUND_STDIN_LIMIT = 64 * 1024
BACKGROUND_STDIN_DRAIN_TIMEOUT = 1.0


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
        notification_store: ConversationStore | None = None,
        notification_callback: Callable[[], None] | None = None,
    ) -> None:
        if type(output_limit) is not int or output_limit < 1:
            raise ValueError("output_limit must be a positive integer")
        if type(call_limit) is not int or call_limit < 1:
            raise ValueError("call_limit must be a positive integer")
        if term_grace <= 0:
            raise ValueError("term_grace must be positive")
        if stdin_drain_timeout <= 0:
            raise ValueError("stdin_drain_timeout must be positive")
        self.output_limit = output_limit
        self.call_limit = call_limit
        self.term_grace = term_grace
        self.stdin_drain_timeout = stdin_drain_timeout
        self._notice_sink = notice_sink
        self._notification_store = notification_store
        self._notification_callback = notification_callback
        self._pending_recovery: dict[str, str] = {}
        self._records: dict[str, _BackgroundRecord] = {}
        self._session_dir: Path | None = None
        self._directory_fd: int | None = None
        self._closed = False
        self._closing_for_shutdown = False
        if session_dir is not None:
            if directory_fd is None:
                raise ValueError("session directory descriptor is required")
            self.bind_session_dir(session_dir, directory_fd)

    @property
    def running_count(self) -> int:
        return sum(record.running for record in self._records.values())

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

    def set_notification_sink(self, store: ConversationStore | None, callback: Callable[[], None] | None) -> None:
        self._notification_store = store
        self._notification_callback = callback
        # Recovery notifications are deferred until a store exists because the
        # registry is built before the notification sink is installed.
        self._flush_recovery()

    def bind_session_dir(self, session_dir: str | Path, directory_fd: int) -> None:
        if self._closed:
            raise RuntimeError("background task registry is closed")
        if self._directory_fd is not None:
            if not os.path.samestat(os.fstat(self._directory_fd), os.fstat(directory_fd)):
                raise ValueError("background task registry is already bound")
            return
        self._session_dir = Path(session_dir)
        self._directory_fd = os.dup(directory_fd)
        self._release_directory = weakref.finalize(self, os.close, self._directory_fd)
        try:
            self._load_previous()
        except BaseException:
            self.release_directory()
            raise

    def open_log(self, path: str | Path) -> IO[bytes]:
        if self._closed:
            raise RuntimeError("background task registry is closed")
        path = Path(path)
        if self._directory_fd is not None and path.parent != self._session_dir:
            raise ValueError("log must belong to the bound session")
        directory = (
            nullcontext(self._directory_fd)
            if self._directory_fd is not None
            else session_root(path.parent, create=True)
        )
        with directory as directory_fd:
            fd = open_session_file(
                directory_fd, path.name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            )
        return os.fdopen(fd, "wb")

    async def start(
        self,
        command: str,
        cwd: str | Path,
        *,
        cwd_fd: int | None = None,
        log_path: str | Path | None = None,
        notify_on_exit: bool = True,
        owner: str = "run_background",
    ) -> tuple[str, int]:
        """Start a task, taking ownership of ``cwd_fd`` when one is supplied."""

        try:
            return await self._start(
                command,
                cwd,
                cwd_fd=cwd_fd,
                log_path=log_path,
                notify_on_exit=notify_on_exit,
                owner=owner,
            )
        finally:
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
    ) -> tuple[str, int]:
        if self._closed:
            raise RuntimeError("background task registry is closed")
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        with ExitStack() as cleanup:
            log_handle = (
                cleanup.enter_context(self.open_log(log_path))
                if log_path is not None else None
            )
            try:
                spawn_kwargs = {
                    "stdin": asyncio.subprocess.PIPE,
                    "stdout": asyncio.subprocess.PIPE,
                    "stderr": asyncio.subprocess.STDOUT,
                    "start_new_session": True,
                    "env": tool_subprocess_env(),
                }
                if cwd_fd is None:
                    process = await asyncio.create_subprocess_shell(
                        command, cwd=cwd, **spawn_kwargs
                    )
                else:
                    process = await create_subprocess_shell_in_fd(
                        command, cwd_fd, **spawn_kwargs
                    )
            except OSError as exc:
                # run_background's failed tool result is the canonical receipt;
                # persisting task_exited would render the same failure twice.
                # Macro-owned starts have no equivalent tool card, so retain
                # their durable notification.
                if notify_on_exit and owner != "run_background":
                    self._notify_exit(task_id, command, None, f"could not execute command: {exc}", log_path)
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
            log_path=str(log_path) if log_path is not None else None,
            notify_on_exit=notify_on_exit,
            owner=owner,
        )
        self._records[task_id] = record
        record.monitor = asyncio.create_task(self._monitor(record, log_handle))
        self._notice(
            BackgroundTaskNotice(
                f"background task {task_id} started: {_command_headline(command)}",
                task_id,
                "started",
                owner,
            )
        )
        self._persist()
        return task_id, process.pid

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
        start = max(requested, record.base_cursor)
        marker = requested < record.base_cursor
        output = bytes(record.output or b"")
        offset = start - record.base_cursor
        available = output[offset:]
        capped = len(available) > self.call_limit
        if capped:
            available = _utf8_chunk(available, self.call_limit)
        if max_chars is not None:
            available = _utf8_char_chunk(available, max_chars)
        next_cursor = start + len(available)
        text = available.decode(errors="replace")
        if marker:
            dropped = record.base_cursor - requested
            text = f"[output truncated; dropped {dropped} bytes]\n" + text
        if capped:
            text += "\n[output capped; call task_output again]\n"
        result: dict[str, Any] = {
            "task_id": task_id,
            "output": text,
            "cursor": next_cursor,
            "running": record.running,
            "exit_code": record.exit_code,
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
        self._closed = True
        killed: list[str] = []
        try:
            for record in tuple(self._records.values()):
                if record.running and record.process is not None:
                    killed.append(record.task_id)
                    # Whole-session and child-completion shutdown kill silently;
                    # callers surface the ids (child receipt) instead.
                    record.notify_on_exit = False
                    await self._terminate(
                        record,
                        reason="task killed on session exit",
                        phase="session_shutdown",
                    )
            self._persist()
            if killed:
                self._notice(
                    BackgroundTaskShutdownNotice(
                        "background tasks killed on session exit: " + ", ".join(killed),
                        tasks=tuple(
                            (task_id, self._records[task_id].owner)
                            for task_id in killed
                        ),
                    )
                )
            return tuple(killed)
        finally:
            self.release_directory()

    def release_directory(self) -> None:
        """Release storage after shutdown or before activation on setup failure."""
        if self._directory_fd is not None:
            self._release_directory()
            self._directory_fd = None

    async def _monitor(self, record: _BackgroundRecord, log_handle: Any | None = None) -> None:
        process = record.process
        if process is None or process.stdout is None:
            return
        reader = asyncio.create_task(
            self._read_output(record, process.stdout, log_handle)
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
            if log_handle is not None:
                log_handle.close()
            if record.running:
                record.running = False
                record.exit_code = process.returncode
                record.ended_at = time.monotonic()
                record.process = None
                record.terminal_phase = record.terminal_phase or "natural_exit"
                self._notice(
                    BackgroundTaskNotice(
                        f"background task {record.task_id} exited ({record.exit_code}): "
                        f"{_command_headline(record.command)}",
                        record.task_id,
                        record.terminal_phase,
                        record.owner,
                    )
                )
                self._persist()
                if record.notify_on_exit:
                    self._notify_exit(
                        record.task_id,
                        record.command,
                        record.exit_code,
                        _output_tail(record.output),
                        record.log_path,
                        record.note,
                        owner=record.owner,
                        phase=record.terminal_phase,
                    )

    async def _read_output(
        self,
        record: _BackgroundRecord,
        stream: asyncio.StreamReader,
        log_handle: Any | None = None,
    ) -> None:
        while chunk := await stream.read(65_536):
            self._append(record, chunk)
            if log_handle is not None:
                log_handle.write(chunk)
                log_handle.flush()

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
        # Serialize shutdown with writes, then close stdin before signaling the group.
        await self._close_stdin(record)
        record.note = reason
        _signal_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=self.term_grace)
        except asyncio.TimeoutError:
            pass
        if _group_exists(process.pid):
            _signal_group(process, signal.SIGKILL)
        if process.returncode is None:
            await process.wait()
        if record.monitor is not None and record.monitor is not asyncio.current_task():
            await asyncio.gather(record.monitor, return_exceptions=True)
        if not record.running:
            return
        record.running = False
        record.exit_code = process.returncode
        record.ended_at = time.monotonic()
        record.process = None
        self._notice(
            BackgroundTaskNotice(
                f"background task {record.task_id} exited ({record.exit_code}): "
                f"{_command_headline(record.command)}",
                record.task_id,
                record.terminal_phase,
                record.owner,
            )
        )
        self._persist()
        if record.notify_on_exit:
            self._notify_exit(
                record.task_id,
                record.command,
                record.exit_code,
                _output_tail(record.output),
                record.log_path,
                record.note,
                owner=record.owner,
                phase=record.terminal_phase,
            )

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

    def _notify_exit(
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
        if self._notification_store is None:
            return
        self._notification_store.append_task_notification(
            task_id=task_id,
            command=_command_headline(command),
            exit_code=exit_code,
            output_tail=output_tail,
            log_path=str(log_path) if log_path is not None else None,
            note=note,
            background_metadata=(owner, phase),
        )
        if self._notification_callback is not None:
            self._notification_callback()

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
            self._records[task_id] = _BackgroundRecord(
                task_id=task_id,
                command=command,
                pid=pid,
                running=False,
                exit_code=row.get("exit_code") if type(row.get("exit_code")) is int else None,
                note="task exited when the previous session ended",
            )
            if was_running:
                self._pending_recovery[task_id] = command
        self._flush_recovery()

    def _flush_recovery(self) -> None:
        """Emit one recovery notification per unobserved previously-running task."""
        if self._notification_store is None or not self._pending_recovery:
            return
        observed = {
            entry.data.get("task_id")
            for entry in self._notification_store.agent_notifications(pending_only=False)
            if entry.data.get("kind", "agent_completion") == "task_exited"
        }
        for task_id, command in tuple(self._pending_recovery.items()):
            if task_id not in observed:
                self._notify_exit(
                    task_id, command, None, "", None, "exit not observed (zeta restarted)"
                )
            self._pending_recovery.pop(task_id, None)

    def _persist(self) -> None:
        if self._directory_fd is None:
            return
        rows = [
            {
                "task_id": record.task_id,
                "command": record.command,
                "pid": record.pid,
                "running": record.running,
                "exit_code": record.exit_code,
            }
            for record in self._records.values()
        ]
        write_session_json(self._directory_fd, "background_tasks.json", rows)


def _command_headline(command: str, limit: int = 80) -> str:
    headline = " ".join(command.split())
    if len(headline) <= limit:
        return headline
    return headline[: max(0, limit - 3)] + "..."


def _output_tail(output: bytearray | None) -> str:
    if not output:
        return ""
    return "\n".join(bytes(output)[-2_048:].decode(errors="replace").splitlines()[-20:])


def _utf8_chunk(data: bytes, limit: int) -> bytes:
    """Cap UTF-8 output without ending inside a character."""
    candidate = data[:limit]
    try:
        candidate.decode()
    except UnicodeDecodeError as exc:
        if exc.reason == "unexpected end of data" and exc.end == len(candidate):
            candidate = candidate[: exc.start]
    if candidate:
        return candidate
    for end in range(1, len(data) + 1):
        try:
            data[:end].decode()
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data" or exc.end != end:
                return data[:end]
        else:
            return data[:end]
    return data


def _utf8_char_chunk(data: bytes, limit: int) -> bytes:
    """Cap decoded UTF-8 characters without advancing past retained bytes."""
    if limit == 0:
        return b""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    character_start = 0
    characters = 0
    for index, value in enumerate(data):
        emitted = decoder.decode(bytes((value,)), final=False)
        if not emitted:
            continue
        if characters + len(emitted) > limit:
            return data[:character_start]
        characters += len(emitted)
        character_start = index + 1
    emitted = decoder.decode(b"", final=True)
    if characters + len(emitted) > limit:
        return data[:character_start]
    return data


def _group_exists(process_id: int) -> bool:
    try:
        os.killpg(process_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _signal_group(process: asyncio.subprocess.Process, signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass
    except OSError:
        if process.returncode is None:
            process.kill()


async def _kill_and_reap(
    process: asyncio.subprocess.Process,
    process_tasks: Sequence[asyncio.Task[object]],
) -> None:
    # A descendant that calls setsid creates a new session and can escape this group.
    _signal_group(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=0.1)
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
    _signal_group(process, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()
    for task in process_tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*process_tasks, return_exceptions=True)
