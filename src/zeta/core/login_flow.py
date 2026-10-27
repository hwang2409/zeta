"""Local redirect-server orchestration for provider OAuth logins."""

from __future__ import annotations

import asyncio
import concurrent.futures
import errno
import http.server
import queue
import socketserver
import sys
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TextIO, TypeVar
from urllib.parse import parse_qs, urlsplit

import httpx


TokenT = TypeVar("TokenT")
# Two waits stay within the publication budget while tolerating loop congestion.
_PUBLICATION_DECISION_TIMEOUT_SECONDS = 20.0


class _PublicationArbiter:
    """Move the credential worker's final publication decision onto the loop."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        task: asyncio.Task[object],
        commit: Callable[[], bool],
    ) -> None:
        self._loop = loop
        self._task = task
        self._commit = commit
        self.cancel_requested = False
        self.committed = False
        self.decision_failed = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def is_set(self) -> bool:
        response: concurrent.futures.Future[bool] = concurrent.futures.Future()
        try:
            self._loop.call_soon_threadsafe(self._decide, response)
        except RuntimeError:
            self.decision_failed = True
            return True
        try:
            return not response.result(timeout=_PUBLICATION_DECISION_TIMEOUT_SECONDS)
        except TimeoutError:
            if response.cancel():
                self.decision_failed = True
                return True
            try:
                return not response.result(
                    timeout=_PUBLICATION_DECISION_TIMEOUT_SECONDS
                )
            except TimeoutError:
                self.decision_failed = True
                return True

    def _decide(self, response: concurrent.futures.Future[bool]) -> None:
        if not response.set_running_or_notify_cancel():
            return
        allowed = (
            not self.cancel_requested
            and self._task.cancelling() == 0
            and self._commit()
        )
        self.committed = allowed
        response.set_result(allowed)


async def _wait_bounded_work(
    task: asyncio.Task[object],
    on_cancel: Callable[[], None] | None = None,
) -> bool:
    """Wait for bounded worker work and report whether cancellation arrived."""

    cancelled = False
    current = asyncio.current_task()
    assert current is not None
    baseline = current.cancelling()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            if on_cancel is not None:
                on_cancel()
    while current.cancelling() > baseline:
        current.uncancel()
    task.result()
    return cancelled


async def _publish_credentials(
    provider: LoginProvider[TokenT],
    tokens: TokenT,
    timeout: float | None,
    commit: Callable[[], bool] | None,
) -> _PublicationArbiter | None:
    if timeout is None:
        await asyncio.to_thread(provider.credential_store.save, tokens)
        return None

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    arbiter = _PublicationArbiter(loop, task, commit or (lambda: True))
    publication = asyncio.create_task(
        asyncio.to_thread(
            provider.credential_store.save,
            tokens,
            lock_timeout=timeout,
            cancel_event=arbiter,
        )
    )
    cancelled = await _wait_bounded_work(publication, arbiter.cancel)
    if cancelled and not arbiter.committed:
        raise asyncio.CancelledError
    if arbiter.decision_failed:
        raise LoginPublicationError
    return arbiter


async def _close_listener(
    server: http.server.ThreadingHTTPServer,
    server_thread: threading.Thread,
    publication_arbiter: _PublicationArbiter | None,
) -> None:
    def close() -> None:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    cleanup = asyncio.create_task(asyncio.to_thread(close))
    cancelled = await _wait_bounded_work(cleanup)
    if cancelled and (
        publication_arbiter is None or not publication_arbiter.committed
    ):
        raise asyncio.CancelledError


class CredentialSink(Protocol[TokenT]):
    """The part of a provider credential store used by login."""

    def save(
        self,
        tokens: TokenT,
        *,
        lock_timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Persist tokens."""


ExchangeAuthorizationCode = Callable[
    [httpx.AsyncClient, str, str, str, str], Awaitable[TokenT]
]
BuildAuthorizationURL = Callable[[str, str, str], str]
TokenHandle = Callable[[TokenT], str | None]


@dataclass(frozen=True, slots=True)
class LoginProvider(Generic[TokenT]):
    """Provider operations required by the shared login flow."""

    name: str
    callback_port: int
    callback_path: str
    build_authorization_url: BuildAuthorizationURL
    exchange_authorization_code: ExchangeAuthorizationCode[TokenT]
    credential_store: CredentialSink[TokenT]
    token_handle: TokenHandle[TokenT]


class LoginError(RuntimeError):
    """Raised for user-facing login failures."""


class LoginPublicationError(LoginError):
    """Raised when the event loop cannot authorize credential publication."""

    def __init__(self) -> None:
        super().__init__("Login could not be saved; try again.")


class LoginPortInUseError(LoginError):
    """Raised when a provider's registered callback port is unavailable."""

    def __init__(self, provider: str, port: int) -> None:
        self.provider = provider
        self.port = port
        provider_label = provider.capitalize()
        super().__init__(
            f"port {port} is already in use for {provider} login "
            f"— close any other {provider_label} login and retry"
        )


@dataclass(frozen=True, slots=True)
class _AuthorizationCode:
    code: str
    state: str


class _CallbackReceiver:
    def __init__(
        self,
        expected_state: str,
        expected_path: str,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self.expected_state = expected_state
        self.expected_path = expected_path
        self._notify = notify
        self._result_lock = threading.Lock()
        self.results: queue.Queue[_AuthorizationCode | LoginError] = queue.Queue(maxsize=1)

    def receive(self, path: str) -> tuple[int, str]:
        parsed = urlsplit(path)
        if parsed.path != self.expected_path:
            return 404, "not found"
        query = parse_qs(parsed.query, keep_blank_values=True)
        state = _query_value(query, "state")
        if state != self.expected_state:
            self._set_result(LoginError("OAuth state mismatch"))
            return 400, "login failed: OAuth state mismatch"
        error = _query_value(query, "error")
        if error:
            description = _query_value(query, "error_description")
            message = f"OAuth provider returned {error}"
            if description:
                message += f": {description}"
            self._set_result(LoginError(message))
            return 400, "login failed"
        code = _query_value(query, "code")
        if not code:
            self._set_result(LoginError("OAuth callback did not include a code"))
            return 400, "login failed: callback did not include a code"
        self._set_result(_AuthorizationCode(code, state))
        return 200, "login complete; you may close this window"

    def _set_result(self, result: _AuthorizationCode | LoginError) -> None:
        with self._result_lock:
            if self.results.full():
                return
            self.results.put_nowait(result)
            if self._notify is not None:
                self._notify()


def _query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if values is None or len(values) != 1:
        return ""
    return values[0]


def _handler_for(receiver: _CallbackReceiver) -> type[http.server.BaseHTTPRequestHandler]:
    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            status, message = receiver.receive(self.path)
            body = message.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: str) -> None:
            del format, args

    return CallbackHandler


class _RedirectServer(http.server.ThreadingHTTPServer):
    """The callback server, minus ``HTTPServer.server_bind``'s reverse DNS.

    The stock ``server_bind`` resolves the bind address with
    ``socket.getfqdn``, a reverse lookup that blocks until the resolver gives
    up on hosts with no PTR record for 127.0.0.1 (GitHub's macOS runners;
    ZETA-91), so ``zeta login`` sat silent instead of printing its URL. The
    name is never used: the server only answers the local OAuth redirect.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


def _create_redirect_server(
    handler: type[http.server.BaseHTTPRequestHandler], *, port: int
) -> http.server.ThreadingHTTPServer:
    return _RedirectServer(("127.0.0.1", port), handler)


async def run_login(
    provider: LoginProvider[TokenT],
    build_pkce: Callable[[], tuple[str, str, str]],
    *,
    timeout_seconds: float = 300,
    output: TextIO | None = None,
    on_authorization_url: Callable[[str], None] | None = None,
    publication_timeout_seconds: float | None = None,
    on_publication_commit: Callable[[], bool] | None = None,
) -> str | None:
    """Run one provider login and return its optional account handle."""

    verifier, challenge, state = build_pkce()
    callback_ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    publication_arbiter: _PublicationArbiter | None = None

    def notify() -> None:
        loop.call_soon_threadsafe(callback_ready.set)

    receiver = _CallbackReceiver(state, provider.callback_path, notify)
    try:
        server = _create_redirect_server(
            _handler_for(receiver), port=provider.callback_port
        )
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        raise LoginPortInUseError(provider.name, provider.callback_port) from exc
    redirect_uri = f"http://localhost:{server.server_port}{provider.callback_path}"
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="zeta-login-redirect",
        daemon=True,
    )
    server_thread.start()
    try:
        authorization_url = provider.build_authorization_url(state, challenge, redirect_uri)
        if on_authorization_url is not None:
            on_authorization_url(authorization_url)
        else:
            print(
                f"open this URL to log in with {provider.name}:\n{authorization_url}",
                file=output or sys.stdout,
                flush=True,
            )
        try:
            await asyncio.wait_for(callback_ready.wait(), timeout_seconds)
        except asyncio.TimeoutError:
            raise LoginError("login timed out after 5 minutes")
        try:
            result = receiver.results.get_nowait()
        except queue.Empty:
            raise LoginError("OAuth callback was not received") from None
        if isinstance(result, LoginError):
            raise result
        async with httpx.AsyncClient() as client:
            tokens = await provider.exchange_authorization_code(
                client,
                result.code,
                result.state,
                verifier,
                redirect_uri,
            )
        publication_arbiter = await _publish_credentials(
            provider,
            tokens,
            publication_timeout_seconds,
            on_publication_commit,
        )
        return provider.token_handle(tokens)
    finally:
        # Finish closing the listener before reporting a terminal login state.
        await _close_listener(server, server_thread, publication_arbiter)


__all__ = [
    "LoginError",
    "LoginPortInUseError",
    "LoginProvider",
    "LoginPublicationError",
    "run_login",
]
