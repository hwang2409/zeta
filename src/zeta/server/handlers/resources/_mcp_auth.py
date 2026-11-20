"""Serve-owned headless MCP authorization; only start returns a loopback launch URL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ....mcp.management_privacy import private_management_io
from ...protocol import ErrorDescriptor, closed_error


@dataclass
class _Attempt:
    task: asyncio.Task[str] | None = None
    committing: bool = False
    cancelling: bool = False

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
        self._attempts: dict[str, _Attempt] = {}
        self._closed = False

    def status(self, name: str) -> dict[str, str]:
        attempt = self._attempts.get(name)
        if attempt is None:
            return {"state": "idle"}
        task = attempt.task
        assert task is not None
        if task.cancelled():
            return {"state": "cancelled"}
        return {"state": task.result() if task.done() else "pending"}

    async def start(self, name: str, login) -> dict[str, str]:
        if self._closed or any(self.status(key)["state"] == "pending" for key in self._attempts):
            raise closed_error(ErrorDescriptor.RESOURCE_BUSY)
        self._attempts.pop(name, None)
        if len(self._attempts) >= 32:
            del self._attempts[next(iter(self._attempts))]
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
        self._attempts[name] = attempt
        try:
            await asyncio.wait((launch_url, task), return_when=asyncio.FIRST_COMPLETED)
            if launch_url.done():
                return {"state": "pending", "authorization_url": launch_url.result()}
            return self.status(name)
        except asyncio.CancelledError:
            await attempt.cancel()
            raise

    async def cancel(self, name: str) -> dict[str, str]:
        attempt = self._attempts.get(name)
        if attempt is not None:
            await attempt.cancel()
        return self.status(name)

    def forget(self, name: str) -> None:
        self._attempts.pop(name, None)

    async def aclose(self) -> None:
        self._closed = True
        for name in tuple(self._attempts):
            await self.cancel(name)
        self._attempts.clear()
