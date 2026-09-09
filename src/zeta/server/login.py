"""Connection-owned login tasks; the request loop alone changes the task map."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ..core.login_flow import LoginError, run_login
from ..providers.factory import credential_store
from ..providers.login import build_login_provider, pkce_values
from .protocol import ProtocolError

REQUESTS = ["login_start", "login_status", "login_cancel", "login_providers"]


def failure(code: str, message: str) -> dict:
    return {"state": "failed", "error": {"code": code, "message": message}}


class Logins:
    """One task per provider, bounded across callback and token exchange.

    Completed results remain available until the next start. Disconnect cancels
    all tasks. Tasks own their result; no completion callbacks mutate the map.
    """

    def __init__(self, home: Path, *, timeout: float = 300) -> None:
        self.home = home
        self.timeout = timeout
        self.tasks: dict[str, asyncio.Task[dict]] = {}

    def providers(self) -> dict:
        rows = []
        for provider in ("claude", "codex"):
            store = credential_store(provider, home=self.home)
            try:
                present = (store.read() or store.bootstrap()) is not None
            except (OSError, RuntimeError, ValueError):
                present = False
            rows.append({"provider": provider, "credentials_present": present})
        return {"providers": rows}

    async def request(self, method: str, params: dict) -> dict:
        if method == "login_providers":
            return self.providers()
        provider = params.get("provider")
        if provider not in ("claude", "codex"):
            raise ProtocolError(-32602, "unsupported login provider")
        task = self.tasks.get(provider)
        if method == "login_start":
            if task is not None and not task.done():
                raise ProtocolError(-32005, "login is already pending", {"code": "login_in_progress"})
            url = asyncio.get_running_loop().create_future()
            task = asyncio.create_task(self._run(provider, url))
            self.tasks[provider] = task
            await asyncio.wait((url, task), return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                return task.result()
            return {"state": "pending", "authorization_url": url.result()}
        if task is None:
            return {"state": "idle"}
        if method == "login_cancel" and not task.done():
            if not task.cancelling():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if task.cancelled():
            return {"state": "cancelled"}
        return task.result() if task.done() else {"state": "pending"}

    async def _run(self, provider: str, url: asyncio.Future[str]) -> dict:
        try:
            async with asyncio.timeout(self.timeout):
                await run_login(
                    build_login_provider("anthropic" if provider == "claude" else provider, self.home),
                    pkce_values,
                    # The outer deadline covers the exchange as well as the callback.
                    timeout_seconds=self.timeout + 1,
                    on_authorization_url=url.set_result,
                )
            return {"state": "succeeded"}
        except TimeoutError:
            return failure("login_timeout", "Sign-in timed out. Log in again to retry.")
        except LoginError:
            return failure("login_callback_error", "Sign-in was rejected or the browser callback was invalid. Please try again.")
        except (OSError, RuntimeError, ValueError, httpx.HTTPError):
            # Provider responses can contain tokens or callback codes. Do not
            # forward exception text to the wire or transcript.
            return failure("login_failed", "Could not complete sign-in. Please try again.")

    async def close(self) -> None:
        for task in self.tasks.values():
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()
