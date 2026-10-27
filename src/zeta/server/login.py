"""Connection-owned login tasks; the request loop alone changes the task map."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from ..accounts import AccountCredentialLockTimeoutError, AccountStore
from ..core.login_flow import LoginError, LoginPortInUseError, run_login
from ..providers.factory import credential_store
from ..providers.login import build_login_provider, pkce_values
from .protocol import ProtocolError

REQUESTS = ["login_start", "login_status", "login_cancel", "login_providers"]
LOGIN_PUBLICATION_TIMEOUT_SECONDS = 60.0


def failure(code: str, message: str) -> dict:
    return {"state": "failed", "error": {"code": code, "message": message}}


class _LoginOperation:
    def __init__(self) -> None:
        self.phase = "pending"

    @property
    def committed(self) -> bool:
        return self.phase == "committing"

    def commit(self) -> bool:
        if self.phase != "pending":
            return False
        self.phase = "committing"
        return True

    def cancel(self, task: asyncio.Task[dict]) -> None:
        if self.phase == "pending":
            self.phase = "cancelled"
            task.cancel()

    def finish(self, result: dict) -> None:
        self.phase = result["state"]


async def _wait_for_cleanup(task: asyncio.Task[object]) -> bool:
    cancelled = False
    current = asyncio.current_task()
    assert current is not None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            current.uncancel()
    task.result()
    return cancelled


class Logins:
    """One task per provider, bounded across callback and token exchange.

    Completed results remain available until the next start. Disconnect cancels
    all tasks. Tasks own their result; no completion callbacks mutate the map.
    """

    def __init__(self, home: Path, *, timeout: float = 300) -> None:
        self.home = home
        self.timeout = timeout
        self.tasks: dict[str, asyncio.Task[dict]] = {}
        self.operations: dict[str, _LoginOperation] = {}

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
            return await asyncio.to_thread(self.providers)
        provider = params.get("provider")
        if provider not in ("claude", "codex"):
            raise ProtocolError(-32602, "unsupported login provider")
        task = self.tasks.get(provider)
        if method == "login_start":
            if task is not None and not task.done():
                raise ProtocolError(-32005, "login is already pending", {"code": "login_in_progress"})
            url = asyncio.get_running_loop().create_future()
            operation = _LoginOperation()
            task = asyncio.create_task(self._run(provider, url, operation))
            self.tasks[provider] = task
            self.operations[provider] = operation
            await asyncio.wait((url, task), return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                return task.result()
            return {"state": "pending", "authorization_url": url.result()}
        if task is None:
            return {"state": "idle"}
        if method == "login_cancel" and not task.done():
            self.operations[provider].cancel(task)
            await asyncio.gather(task, return_exceptions=True)
        if task.cancelled():
            return {"state": "cancelled"}
        return task.result() if task.done() else {"state": "pending"}

    async def _attempt(
        self,
        login_provider: str,
        url: asyncio.Future[str],
        operation: _LoginOperation,
    ) -> dict:
        try:
            login = await asyncio.to_thread(
                build_login_provider, login_provider, self.home
            )
            async with asyncio.timeout(self.timeout):
                await run_login(
                    login,
                    pkce_values,
                    # The outer deadline covers the exchange as well as the callback.
                    timeout_seconds=self.timeout + 1,
                    on_authorization_url=url.set_result,
                    publication_timeout_seconds=LOGIN_PUBLICATION_TIMEOUT_SECONDS,
                    on_publication_commit=operation.commit,
                )
            return {"state": "succeeded"}
        except TimeoutError:
            return failure("login_timeout", "Sign-in timed out. Log in again to retry.")
        except AccountCredentialLockTimeoutError:
            return failure(
                "login_lock_timeout",
                "Sign-in could not save credentials because the account is busy. Please try again.",
            )
        except LoginPortInUseError as exc:
            provider_label = exc.provider.capitalize()
            return failure(
                "login_port_in_use",
                f"{provider_label} login cannot start because port {exc.port} is already in use. "
                f"Close any other {provider_label} login and retry.",
            )
        except LoginError:
            return failure("login_callback_error", "Sign-in was rejected or the browser callback was invalid. Please try again.")
        except (OSError, RuntimeError, ValueError, httpx.HTTPError):
            # Provider responses can contain tokens or callback codes. Do not
            # forward exception text to the wire or transcript.
            return failure("login_failed", "Could not complete sign-in. Please try again.")

    async def _run(
        self,
        provider: str,
        url: asyncio.Future[str],
        operation: _LoginOperation | None = None,
    ) -> dict:
        operation = operation or _LoginOperation()
        login_provider = "anthropic" if provider == "claude" else provider
        lock = AccountStore(self.home).provider_login_lock(login_provider)
        locked = False
        acquisition = asyncio.create_task(asyncio.to_thread(lock.__enter__))
        cancelled = False
        result: dict | None = None
        try:
            cancelled = await _wait_for_cleanup(acquisition)
            locked = acquisition.exception() is None
            if cancelled:
                raise asyncio.CancelledError
            result = await self._attempt(login_provider, url, operation)
        except asyncio.CancelledError:
            cancelled = True

        if locked:
            release = asyncio.create_task(
                asyncio.to_thread(lock.__exit__, None, None, None)
            )
            release_cancelled = await _wait_for_cleanup(release)
            cancelled = cancelled or release_cancelled
        if cancelled and not operation.committed:
            operation.phase = "cancelled"
            raise asyncio.CancelledError
        assert result is not None
        operation.finish(result)
        return result

    async def close(self) -> None:
        for provider, task in self.tasks.items():
            if not task.done():
                self.operations[provider].cancel(task)
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()
        self.operations.clear()
