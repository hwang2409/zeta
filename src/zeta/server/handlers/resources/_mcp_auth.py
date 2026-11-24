"""Serve-owned headless MCP authorization; only start returns a loopback launch URL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ....mcp.management_privacy import private_management_io
from ...protocol import ErrorDescriptor, closed_error


@dataclass(eq=False, slots=True)
class MCPAuthConnection:
    closed: bool = False


@dataclass
class _Attempt:
    task: asyncio.Task[str] | None = None
    committing: bool = False
    cancelling: bool = False

    @property
    def pending(self) -> bool:
        assert self.task is not None
        return not self.task.done()

    def status(self) -> dict[str, str]:
        task = self.task
        assert task is not None
        if task.cancelled():
            return {"state": "cancelled"}
        return {"state": task.result() if task.done() else "pending"}

    def commit(self) -> None:
        self.committing = True

    async def cancel(self) -> None:
        assert self.task is not None
        if not (self.committing or self.cancelling or self.task.done()):
            self.cancelling = True
            self.task.cancel()
        await asyncio.gather(asyncio.shield(self.task), return_exceptions=True)


class MCPAuth:
    def __init__(self) -> None:
        self._attempts: dict[tuple[MCPAuthConnection | None, str], _Attempt] = {}
        self._closed = False

    def status(self, owner: MCPAuthConnection | None, name: str) -> dict[str, str]:
        attempt = self._attempts.get((owner, name))
        return {"state": "idle"} if attempt is None else attempt.status()

    async def start(self, owner: MCPAuthConnection | None, name: str, login) -> dict[str, str]:
        # Legacy has one serve-wide slot. Multi-controller shares only the name lock.
        if self._closed or (owner is not None and owner.closed) or any(
            attempt.pending and (owner is None or active_name == name)
            for (_, active_name), attempt in self._attempts.items()
        ):
            raise closed_error(ErrorDescriptor.RESOURCE_BUSY)
        key = (owner, name)
        self._attempts.pop(key, None)
        history = [key for key, attempt in self._attempts.items() if key[0] is owner and not attempt.pending]
        if len(history) >= 32:
            del self._attempts[history[0]]
        attempt = _Attempt()
        launch_url = asyncio.get_running_loop().create_future()

        async def run() -> str:
            try:
                with private_management_io():
                    await login(launch_url.set_result, attempt.commit)
            except Exception:  # noqa: BLE001 - never retain or log OAuth errors
                return "failed"
            return "succeeded"

        task = asyncio.create_task(run(), name="mcp-auth")
        attempt.task = task
        self._attempts[key] = attempt
        try:
            await asyncio.wait((launch_url, task), return_when=asyncio.FIRST_COMPLETED)
            if launch_url.done():
                return {"state": "pending", "authorization_url": launch_url.result()}
            return attempt.status()
        except asyncio.CancelledError:
            await attempt.cancel()
            raise

    async def cancel(self, owner: MCPAuthConnection | None, name: str) -> dict[str, str]:
        attempt = self._attempts.get((owner, name))
        if attempt is not None:
            await attempt.cancel()
        return self.status(owner, name)

    def forget(self, owner: MCPAuthConnection | None, name: str) -> None:
        self._attempts.pop((owner, name), None)

    async def close_connection(self, owner: MCPAuthConnection) -> None:
        owner.closed = True
        attempts = {key: attempt for key, attempt in self._attempts.items() if key[0] is owner}
        await asyncio.gather(*(attempt.cancel() for attempt in attempts.values()))
        for key in attempts:
            self._attempts.pop(key, None)

    async def aclose(self) -> None:
        self._closed = True
        for attempt in tuple(self._attempts.values()):
            await attempt.cancel()
        self._attempts.clear()
