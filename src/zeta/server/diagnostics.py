"""Diagnostics for a wedged serve controller slot.

When a controller ``hello`` is refused with -32001 while the holder's
connection is already gone (EOF / transport lost) but its request task still
runs, :func:`log_stuck_holder` writes one bounded record to stderr naming the
holder's running request and the await chain its run task is stuck on. This is
diagnostic only; it changes no refusal behavior.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connections import _OutputState
    from .server import _Client

logger = logging.getLogger("zeta.server.diagnostics")

# Bound the dump so repeated wedges cannot flood stderr.
_MAX_CHAIN_FRAMES = 40
_MAX_FRAME_CHARS = 200


def _await_chain(task: asyncio.Task[object] | None) -> list[str]:
    """Walk the await chain of a suspended run task, following awaited coroutines.

    ``Task.get_stack()`` shows only the innermost frame of a suspended
    coroutine, so we follow ``cr_await`` (the coroutine a frame is awaiting)
    down the chain instead.
    """
    if task is None:
        return []
    lines: list[str] = []
    coro: object | None = task.get_coro()
    seen: set[int] = set()
    while coro is not None and len(lines) < _MAX_CHAIN_FRAMES:
        if id(coro) in seen:
            break
        seen.add(id(coro))
        frame = (
            getattr(coro, "cr_frame", None)
            or getattr(coro, "gi_frame", None)
            or getattr(coro, "ag_frame", None)
        )
        if frame is not None:
            name = frame.f_code.co_filename.rsplit("/src/", 1)[-1]
            lines.append(f"{name}:{frame.f_lineno} {frame.f_code.co_name}"[:_MAX_FRAME_CHARS])
        nxt = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
        if nxt is coro:
            break
        coro = nxt
    return lines


def _output_summary(output: _OutputState | None) -> str:
    if output is None:
        return "output=none"
    writer = output.writer_task
    if writer is None:
        writer_state = "none"
    elif not writer.done():
        writer_state = "running"
    elif writer.cancelled():
        writer_state = "cancelled"
    else:
        writer_state = f"failed:{output.writer_task.exception()!r}"
    return (
        f"phase={output.phase} queue={len(output.queue)} "
        f"pending_callers={output.pending_callers} writer={writer_state}"
    )


def log_stuck_holder(holder: _Client, output: _OutputState | None, *, reason: str) -> None:
    """Write one warning naming the lost holder's running request and await chain."""
    request = holder._request
    if request is None:
        method, request_id, age = "<none>", None, 0.0
    else:
        method, request_id, started = request
        age = time.monotonic() - started
    chain = _await_chain(holder._run_task)
    logger.warning(
        "serve: %s; holder request method=%s id=%r age=%.1fs; %s; await chain: %s",
        reason,
        method,
        request_id,
        age,
        _output_summary(output),
        " <- ".join(chain) if chain else "<empty>",
    )
