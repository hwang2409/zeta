from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from ..runtime._async_close import close_transport_resources

_MAX_IDLE_ANONYMOUS_CLIENTS = 8


@dataclass(slots=True)
class _ClientEntry:
    client: httpx.AsyncClient
    active: int = 0


class SessionWebClients:
    """Own credential-free HTTP pools for one tool registry."""

    def __init__(self) -> None:
        self._anonymous: OrderedDict[str, _ClientEntry] = OrderedDict()
        self._codex: _ClientEntry | None = None
        self._condition = asyncio.Condition()
        self._retired = False

    @asynccontextmanager
    async def anonymous(self, origin: str) -> AsyncIterator[httpx.AsyncClient]:
        async with self._condition:
            if self._retired:
                raise RuntimeError("web client owner is closed")
            entry = self._anonymous.get(origin)
            if entry is None:
                entry = _ClientEntry(
                    httpx.AsyncClient(
                        timeout=15.0,
                        follow_redirects=False,
                        trust_env=False,
                    )
                )
                self._anonymous[origin] = entry
            entry.active += 1
            self._anonymous.move_to_end(origin)
        try:
            yield entry.client
        finally:
            await self._release_anonymous(origin, entry)

    async def _release_anonymous(self, origin: str, entry: _ClientEntry) -> None:
        closes: list[httpx.AsyncClient] = []
        async with self._condition:
            entry.active -= 1
            if entry.active == 0:
                self._anonymous.move_to_end(origin)
            idle = sum(item.active == 0 for item in self._anonymous.values())
            for key, candidate in tuple(self._anonymous.items()):
                if idle <= _MAX_IDLE_ANONYMOUS_CLIENTS:
                    break
                if candidate.active == 0:
                    del self._anonymous[key]
                    closes.append(candidate.client)
                    idle -= 1
            self._condition.notify_all()
        await self._close_clients(closes)

    @asynccontextmanager
    async def codex(self, timeout: httpx.Timeout) -> AsyncIterator[httpx.AsyncClient]:
        async with self._condition:
            if self._retired:
                raise RuntimeError("web client owner is closed")
            if self._codex is None:
                self._codex = _ClientEntry(httpx.AsyncClient(timeout=timeout))
            self._codex.active += 1
            entry = self._codex
        try:
            yield entry.client
        finally:
            async with self._condition:
                entry.active -= 1
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self._retired = True
            await self._condition.wait_for(self._all_idle)
            clients = [entry.client for entry in self._anonymous.values()]
            if self._codex is not None:
                clients.append(self._codex.client)
            self._anonymous.clear()
            self._codex = None
        await self._close_clients(clients)

    def _all_idle(self) -> bool:
        return all(entry.active == 0 for entry in self._anonymous.values()) and (
            self._codex is None or self._codex.active == 0
        )

    @staticmethod
    async def _close_clients(clients: list[httpx.AsyncClient]) -> None:
        if not clients:
            return
        outcome = await close_transport_resources(
            *(client.aclose() for client in clients)
        )
        if outcome.failures:
            raise outcome.failures[0].error
