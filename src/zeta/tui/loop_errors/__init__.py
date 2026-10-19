"""Keep unexpected asyncio errors visible inside the terminal UI."""

from __future__ import annotations

import asyncio
import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession

from ...core.session import env_home
from ...diagnostics.tui_stalls import write_tui_diagnostic
from ...protocol.types import ErrorInfo, StreamEvent, StreamEventType
from ..render import render_event

_TUI_ERROR_LOG = "tui-errors.jsonl"
_DIAGNOSTIC_QUEUE: queue.Queue[tuple[Path, dict[str, Any]]] = queue.Queue(maxsize=32)
_DIAGNOSTIC_WORKER_LOCK = threading.Lock()
_DIAGNOSTIC_WORKER_STARTED = False


def _write_diagnostics() -> None:
    while True:
        path, record = _DIAGNOSTIC_QUEUE.get()
        try:
            write_tui_diagnostic(path, record)
        except BaseException:  # noqa: BLE001 - diagnostics must never affect the TUI
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
    except BaseException:  # noqa: BLE001 - diagnostics must never affect the TUI
        return


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
        log_path = env_home() / "logs" / _TUI_ERROR_LOG
        if isinstance(error, BaseException):
            message = str(error).strip() or type(error).__name__
            stack = "".join(traceback.format_exception(error))
        else:
            message = str(context.get("message", "unexpected ui error"))
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
