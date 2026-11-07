from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MCPCallDeadline:
    """One absolute call deadline anchored before call work starts."""

    expires_at: float
    clock: Callable[[], float]

    @classmethod
    def start(
        cls, seconds: float, *, clock: Callable[[], float] | None = None
    ) -> MCPCallDeadline:
        resolved_clock = clock or asyncio.get_running_loop().time
        return cls(resolved_clock() + seconds, resolved_clock)

    @property
    def expired(self) -> bool:
        return self.clock() >= self.expires_at

    async def wait(self) -> None:
        async with asyncio.timeout_at(self.expires_at):
            await asyncio.Future()


async def finish_call_cleanup(
    call: asyncio.Task[object], *, timeout: float, server_name: str
) -> None:
    """Wait briefly for cancellation while the actor retains call ownership."""

    try:
        await asyncio.wait_for(
            asyncio.shield(asyncio.gather(call, return_exceptions=True)), timeout
        )
    except TimeoutError:
        logger.warning(
            "timed out cleaning up MCP tool call for %s; task remains actor-owned",
            server_name,
        )
