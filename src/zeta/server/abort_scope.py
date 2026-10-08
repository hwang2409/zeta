"""Foreground and session abort request lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

from .protocol import ProtocolError

if TYPE_CHECKING:
    from ..runtime.loop.agent import AgentLoop

AbortScope = Literal["session", "foreground"]


def clear_pending_steering(
    features: set[str], loop: AgentLoop | None
) -> dict[str, int]:
    """Clear queued steering only for clients that negotiated abort scopes."""

    if "abort_scope" not in features:
        raise ProtocolError(
            -32602,
            "clear_steering requires the negotiated abort_scope feature",
        )
    if loop is None:
        raise ProtocolError(-32003, "no active session")
    return {"cleared": loop.clear_pending_steering()}


def parse_abort_scope(params: dict[str, object], features: set[str]) -> AbortScope:
    """Validate an abort request without changing legacy unscoped behavior."""

    scope = params.get("scope", "session")
    if "scope" in params and "abort_scope" not in features:
        raise ProtocolError(-32602, "scope requires the negotiated abort_scope feature")
    if not isinstance(scope, str) or scope not in {"session", "foreground"}:
        raise ProtocolError(-32602, "scope must be 'session' or 'foreground'")
    return scope


async def abort_active_turn(
    task: asyncio.Task[None] | None,
    *,
    scope: AbortScope,
    loop: AgentLoop | None,
    terminate_approvals: Callable[..., Awaitable[None]],
    before_cancel: Callable[[], None],
) -> bool:
    """Cancel the captured turn using the selected ownership scope."""

    if task is None or task.done():
        return False
    foreground_only = scope == "foreground"
    await terminate_approvals(foreground_only=foreground_only)
    if loop is not None:
        loop.abort(foreground_only=foreground_only, steering_drop_reason=None)
    before_cancel()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return True
