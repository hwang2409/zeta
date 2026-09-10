"""Process-group cleanup shared by shell tools."""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
import uuid
import weakref
from collections.abc import Callable, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ..core.checkpoints import load_session_json
from ..core.process_env import subprocess_env
from ..core.session_files import (
    open_session_file,
    read_session_file,
    session_root,
    write_session_json,
)


def tool_subprocess_env() -> dict[str, str]:
    """Return the scrubbed parent environment for a tool child."""

    return subprocess_env()


BACKGROUND_TASK_LIMIT = 8
BACKGROUND_OUTPUT_LIMIT = 512 * 1024
BACKGROUND_OUTPUT_CALL_LIMIT = 32 * 1024
BACKGROUND_TERM_GRACE_SECONDS = 0.25


@dataclass(slots=True)
class _BackgroundRecord:
    task_id: str
    command: str
    pid: int
    process: asyncio.subprocess.Process | None = None
    output: bytearray | None = None
    total_bytes: int = 0
    base_cursor: int = 0
    running: bool = True
    exit_code: int | None = None
    note: str | None = None
    monitor: asyncio.Task[None] | None = None


class BackgroundTaskRegistry:
    """Own detached process groups and their bounded session output."""

    def __init__(
        self,
        *,
        session_dir: str | Path | None = None,
        directory_fd: int | None = None,
        max_tasks: int = BACKGROUND_TASK_LIMIT,
        output_limit: int = BACKGROUND_OUTPUT_LIMIT,
        call_limit: int = BACKGROUND_OUTPUT_CALL_LIMIT,
        term_grace: float = BACKGROUND_TERM_GRACE_SECONDS,
        notice_sink: Callable[[str], None] | None = None,
    ) -> None:
        if type(max_tasks) is not int or max_tasks < 1:
            raise ValueError("max_tasks must be a positive integer")
        if type(output_limit) is not int or output_limit < 1:
            raise ValueError("output_limit must be a positive integer")
        if type(call_limit) is not int or call_limit < 1:
            raise ValueError("call_limit must be a positive integer")
        if term_grace <= 0:
            raise ValueError("term_grace must be positive")
        self.max_tasks = max_tasks
        self.output_limit = output_limit
        self.call_limit = call_limit
        self.term_grace = term_grace
        self._notice_sink = notice_sink
        self._records: dict[str, _BackgroundRecord] = {}
        self._session_dir: Path | None = None
        self._directory_fd: int | None = None
        self._closed = False
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

    def set_notice_sink(self, sink: Callable[[str], None] | None) -> None:
        self._notice_sink = sink

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
        log_path: str | Path | None = None,
    ) -> tuple[str, int]:
        if self._closed:
            raise RuntimeError("background task registry is closed")
        if self.running_count >= self.max_tasks:
            raise ValueError(f"background task limit reached ({self.max_tasks})")
        with ExitStack() as cleanup:
            log_handle = (
                cleanup.enter_context(self.open_log(log_path))
                if log_path is not None else None
            )
            try:
                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=tool_subprocess_env(),
                )
            except OSError as exc:
                raise ValueError(f"could not execute command: {exc}") from exc
            # The monitor owns the log after the process starts.
            cleanup.pop_all()
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        record = _BackgroundRecord(
            task_id=task_id,
            command=command,
            pid=process.pid,
            process=process,
            output=bytearray(),
        )
        self._records[task_id] = record
        record.monitor = asyncio.create_task(self._monitor(record, log_handle))
        self._notice(f"background task {task_id} started: {_command_headline(command)}")
        self._persist()
        return task_id, process.pid

    async def wait(self, task_id: str) -> dict[str, Any]:
        """Wait for one background process and return its terminal status."""

        record = self._record(task_id)
        if record.monitor is not None:
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

    async def kill(self, task_id: str) -> dict[str, Any]:
        record = self._record(task_id)
        if record.running and record.process is not None:
            await self._terminate(record, reason="task killed")
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
                    await self._terminate(record, reason="task killed on session exit")
            self._persist()
            if killed:
                self._notice("background tasks killed on session exit: " + ", ".join(killed))
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
            if log_handle is not None:
                log_handle.close()
            if not record.running:
                return
            record.running = False
            record.exit_code = process.returncode
            record.process = None
            self._notice(
                f"background task {record.task_id} exited ({record.exit_code}): "
                f"{_command_headline(record.command)}"
            )
            self._persist()

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

    async def _terminate(self, record: _BackgroundRecord, *, reason: str) -> None:
        process = record.process
        if process is None:
            return
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
        record.process = None
        self._notice(
            f"background task {record.task_id} exited ({record.exit_code}): "
            f"{_command_headline(record.command)}"
        )
        self._persist()

    def _append(self, record: _BackgroundRecord, chunk: bytes) -> None:
        if record.output is None:
            record.output = bytearray()
        record.output.extend(chunk)
        record.total_bytes += len(chunk)
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

    def _notice(self, message: str) -> None:
        if self._notice_sink is not None:
            self._notice_sink(message)

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
            self._records[task_id] = _BackgroundRecord(
                task_id=task_id,
                command=command,
                pid=pid,
                running=False,
                exit_code=row.get("exit_code") if type(row.get("exit_code")) is int else None,
                note="task exited when the previous session ended",
            )

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
    _signal_group(process, signal.SIGTERM)
    try:
        await asyncio.shield(asyncio.sleep(0.1))
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
    _signal_group(process, signal.SIGKILL)
    for task in process_tasks:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        if not task.cancelled():
            task.exception()
