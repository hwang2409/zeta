"""Choose one execution owner for a resolved foreground approval."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable

from .approval_grants import ApprovalKey


class ApprovalResumeOwner:
    """Keep MCP reconnect resumes distinct from their suspended live turns."""

    def __init__(self, session_id: Callable[[], str | None]) -> None:
        self._session_id = session_id
        self._mcp_reconnect: set[tuple[str | None, str]] = set()

    def mark_mcp_reconnect(self, keys: Iterable[ApprovalKey]) -> None:
        session_id = self._session_id()
        self._mcp_reconnect = {
            (session_id, key) for key in keys if isinstance(key, str)
        }

    def should_resume(
        self,
        key: ApprovalKey,
        turn_task: asyncio.Task[None] | None,
    ) -> bool:
        marker = (self._session_id(), key) if isinstance(key, str) else None
        reconnected = marker in self._mcp_reconnect
        if marker is not None:
            self._mcp_reconnect.discard(marker)
        return (
            reconnected
            or turn_task is None
            or turn_task.done()
            or turn_task.cancelling() > 0
        )
