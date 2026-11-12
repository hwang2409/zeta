"""Keep unexpected asyncio errors visible inside the terminal UI."""

from __future__ import annotations

import asyncio
import atexit
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
_TUI_STDERR_LOG = "tui-stderr.log"
_TUI_STDERR_MAX_BYTES = 5 * 1024 * 1024
_DIAGNOSTIC_QUEUE: queue.Queue[tuple[Path, dict[str, Any]]] = queue.Queue(maxsize=32)
_DIAGNOSTIC_WORKER_LOCK = threading.Lock()
_DIAGNOSTIC_WORKER_STARTED = False
_STDERR_SETUP_LOCK = threading.Lock()
_STDERR_SETUP = False
_TERMINAL_FD: int | None = None
_PREVIOUS_STDERR: Any = None


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


def terminal_fd() -> int | None:
    """Return the saved terminal for intentional messages after TUI capture."""
    return _TERMINAL_FD


def write_terminal(message: str) -> None:
    if _PREVIOUS_STDERR is not None and not _PREVIOUS_STDERR.isatty():
        print(message, file=_PREVIOUS_STDERR)
        return
    fd = _TERMINAL_FD
    if fd is None:
        print(message, file=sys.stderr)
        return
    try:
        os.write(fd, (message + "\n").encode())
    except OSError:
        pass


def _restore_stderr() -> None:
    fd = _TERMINAL_FD
    if fd is not None:
        try:
            os.dup2(fd, 2)
        except OSError:
            pass


def _open_stderr_log(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() and path.stat().st_size > _TUI_STDERR_MAX_BYTES:
        rotated = path.with_name(path.name + ".1")
        rotated.unlink(missing_ok=True)
        path.replace(rotated)
    return os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )


def restore_stderr_for_cli() -> None:
    if _TERMINAL_FD is not None:
        os.dup2(_TERMINAL_FD, 2)
    if _PREVIOUS_STDERR is not None:
        sys.stderr = _PREVIOUS_STDERR


def _setup_stderr() -> None:
    global _STDERR_SETUP, _TERMINAL_FD, _PREVIOUS_STDERR
    with _STDERR_SETUP_LOCK:
        if not _STDERR_SETUP:
            path = env_home() / "logs" / _TUI_STDERR_LOG
            _PREVIOUS_STDERR = sys.stderr
            terminal_fd = os.dup(2)
            log_fd = _open_stderr_log(path)
            os.dup2(log_fd, 2)
            os.close(log_fd)
            _TERMINAL_FD = terminal_fd
            _STDERR_SETUP = True
            atexit.register(_restore_stderr)
        sys.stderr = open(  # noqa: SIM115 - lives for the process lifetime
            2, "w", buffering=1, encoding="utf-8", errors="backslashreplace", closefd=False
        )
        sys.__stderr__ = sys.stderr


@contextmanager
def stderr_to_log() -> Iterator[None]:
    """Keep TUI stderr on one bounded append-only process log until exit."""
    _setup_stderr()
    try:
        yield
    finally:
        # test runners capture stderr with a non-terminal stream; do not poison
        # the rest of an in-process test suite while real TUI processes persist it.
        if _PREVIOUS_STDERR is not None and not _PREVIOUS_STDERR.isatty():
            global _STDERR_SETUP
            restore_stderr_for_cli()
            _STDERR_SETUP = False


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
