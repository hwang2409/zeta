"""Keep unexpected asyncio errors visible inside the terminal UI."""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession

from ...core.session import env_home
from ...diagnostics.tui_stalls import write_tui_diagnostic
from ...protocol.types import ErrorInfo, StreamEvent, StreamEventType
from ..composer import FullScreenPromptSession
from ..render import render_event

_TUI_ERROR_LOG = "tui-errors.jsonl"
_DIAGNOSTIC_QUEUE: queue.Queue[tuple[Path, dict[str, Any]]] = queue.Queue(maxsize=32)
_STDERR_QUEUE: queue.Queue[tuple[Path, bytes] | None] = queue.Queue(maxsize=32)
_STDERR_DROPPED = 0
_STDERR_DROPPED_LOCK = threading.Lock()
_DIAGNOSTIC_WORKER_LOCK = threading.Lock()
_DIAGNOSTIC_WORKER_STARTED = False


def _write_diagnostics() -> None:
    while True:
        path, record = _DIAGNOSTIC_QUEUE.get()
        try:
            write_tui_diagnostic(path, record)
        except Exception:  # noqa: BLE001, S110 - diagnostics must never affect the TUI
            pass


def _schedule_diagnostic(path: Path, record: dict[str, Any]) -> None:
    global _DIAGNOSTIC_WORKER_STARTED
    try:
        with _DIAGNOSTIC_WORKER_LOCK:
            if not _DIAGNOSTIC_WORKER_STARTED:
                threading.Thread(
                    target=_write_diagnostics,
                    name="zeta-tui-diagnostics",
                    daemon=True,
                ).start()
                _DIAGNOSTIC_WORKER_STARTED = True
        _DIAGNOSTIC_QUEUE.put_nowait((path, record))
    except Exception:  # noqa: BLE001 - diagnostics must never affect the TUI
        return


def tui_error_log() -> Path:
    return env_home() / "logs" / _TUI_ERROR_LOG


def log_tui_crash(error: BaseException) -> Path:
    """Log a fatal TUI error and return the log path for one short exit line."""

    path = tui_error_log()
    stack = "".join(traceback.format_exception(error))
    write_tui_diagnostic(path, {"type": "crash", "traceback": stack})
    return path


def _write_stderr_queue() -> None:
    global _STDERR_DROPPED
    while item := _STDERR_QUEUE.get():
        path, chunk = item
        with _STDERR_DROPPED_LOCK:
            dropped = _STDERR_DROPPED
            _STDERR_DROPPED = 0
        record = {"type": "stderr", "text": chunk.decode("utf-8", "replace")}
        if dropped:
            record["dropped_bytes"] = dropped
        try:
            write_tui_diagnostic(path, record)
        except Exception:  # noqa: BLE001, S110 - diagnostics must never affect the TUI
            pass
        finally:
            _STDERR_QUEUE.task_done()


def _copy_stderr_to_log(read_fd: int, path: Path) -> None:
    # stderr draining never waits for disk I/O; a full queue drops and counts bytes.
    global _STDERR_DROPPED
    with os.fdopen(read_fd, "rb", buffering=0) as pipe:
        while chunk := pipe.read(65536):
            try:
                _STDERR_QUEUE.put_nowait((path, chunk))
            except queue.Full:
                with _STDERR_DROPPED_LOCK:
                    _STDERR_DROPPED += len(chunk)


@contextmanager
def stderr_to_log() -> Iterator[None]:
    """Send everything written to stderr to the TUI error log while the TUI runs.

    One owner for fd 2: Python writes (warnings, logging's last-resort handler,
    unraisable and thread exceptions, asyncio shutdown errors) and child
    processes that inherit fd 2 all reach the log instead of the terminal.
    """

    path = tui_error_log()
    sys.stderr.flush()
    read_fd, write_fd = os.pipe()
    writer = threading.Thread(
        target=_write_stderr_queue, name="zeta-tui-stderr-writer", daemon=True
    )
    writer.start()
    reader = threading.Thread(
        target=_copy_stderr_to_log, args=(read_fd, path), name="zeta-tui-stderr", daemon=True
    )
    reader.start()
    terminal_fd = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    previous = sys.stderr
    sys.stderr = open(  # noqa: SIM115 - lives until the terminal is restored below
        2, "w", buffering=1, encoding="utf-8", errors="backslashreplace", closefd=False
    )
    try:
        yield
    finally:
        sys.stderr.flush()
        sys.stderr = previous
        os.dup2(terminal_fd, 2)
        os.close(terminal_fd)
        # Children that still hold the pipe keep logging; do not wait for them.
        reader.join(timeout=1)
        _STDERR_QUEUE.join()


async def start_prompt_during_activation(
    app: Any,
    session: PromptSession[Any],
    prompt_runner: Any,
) -> tuple[asyncio.Task[None], asyncio.Task[object] | None, bool]:
    """Start TUI input first and report whether it exited during activation."""

    prompt_task: asyncio.Task[object] | None = None
    activation_task: asyncio.Task[None] | None = None
    try:
        if isinstance(session, FullScreenPromptSession):
            app._install_full_screen_layout(session)
            prompt_task = asyncio.create_task(prompt_runner(session, app))
            app._input_loop_active = True
            await asyncio.sleep(0)
        activation_task = asyncio.create_task(app.loop.activate())
        if prompt_task is None:
            return activation_task, None, False
        done, _ = await asyncio.wait(
            (prompt_task, activation_task), return_when=asyncio.FIRST_COMPLETED
        )
        if prompt_task not in done:
            return activation_task, prompt_task, False
        await app._run_full_screen(session, prompt_task)
        return activation_task, prompt_task, True
    except BaseException:
        # The caller never received these handles, so this helper must stop them.
        app._input_loop_active = False
        started = [task for task in (prompt_task, activation_task) if task is not None]
        for task in started:
            task.cancel()
        await asyncio.gather(*started, return_exceptions=True)
        raise


async def cancel_activation(task: asyncio.Task[None]) -> None:
    if not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def run_prompt_toolkit(
    session: PromptSession[Any],
    app: Any,
) -> object:
    """Run prompt-toolkit while routing background errors to the TUI."""
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()

    def handle_exception(
        _loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        error = context.get("exception")
        log_path = tui_error_log()
        if isinstance(error, BaseException):
            message = f"unexpected ui error: {type(error).__name__}"
            stack = "".join(traceback.format_exception(error))
        else:
            message = "unexpected ui error"
            stack = message
        message = f"{message} (details in {log_path})"
        app._print_unit(
            render_event(
                StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("ui_error", message),
                )
            )
        )
        app._invalidate_prompt()
        _schedule_diagnostic(log_path, {"type": "ui_error", "traceback": stack})

    loop.set_exception_handler(handle_exception)
    try:
        return await session.app.run_async(set_exception_handler=False)
    finally:
        loop.set_exception_handler(previous_handler)
