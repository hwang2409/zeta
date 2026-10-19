"""Keep unexpected asyncio errors visible inside the terminal UI."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prompt_toolkit import PromptSession

from ...protocol.types import ErrorInfo, StreamEvent, StreamEventType
from ..render import render_event

_LOGGER = logging.getLogger(__name__)


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
        if isinstance(error, BaseException):
            message = str(error).strip() or type(error).__name__
            _LOGGER.error("unhandled exception in tui event loop", exc_info=error)
        else:
            message = str(context.get("message", "unexpected ui error"))
            _LOGGER.error("unhandled exception in tui event loop: %s", message)
        app._print_unit(
            render_event(
                StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("ui_error", message),
                )
            )
        )
        app._invalidate_prompt()

    loop.set_exception_handler(handle_exception)
    try:
        return await session.app.run_async(set_exception_handler=False)
    finally:
        loop.set_exception_handler(previous_handler)
