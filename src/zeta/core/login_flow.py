"""Local redirect-server orchestration for provider OAuth logins."""

from __future__ import annotations

import asyncio
import http.server
import queue
import sys
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TextIO, TypeVar
from urllib.parse import parse_qs, urlsplit

import httpx


TokenT = TypeVar("TokenT")


class CredentialStore(Protocol[TokenT]):
    """The part of a provider credential store used by login."""

    def save(self, tokens: TokenT) -> None:
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
    build_authorization_url: BuildAuthorizationURL
    exchange_authorization_code: ExchangeAuthorizationCode[TokenT]
    credential_store: CredentialStore[TokenT]
    token_handle: TokenHandle[TokenT]


class LoginError(RuntimeError):
    """Raised for user-facing login failures."""


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


def _create_redirect_server(
    handler: type[http.server.BaseHTTPRequestHandler], *, preferred_port: int = 0
) -> http.server.ThreadingHTTPServer:
    address = ("127.0.0.1", preferred_port)
    try:
        return http.server.ThreadingHTTPServer(address, handler)
    except OSError as exc:
        if preferred_port == 0:
            try:
                return http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            except OSError:
                raise exc from None
        return http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)


async def run_login(
    provider: LoginProvider[TokenT],
    build_pkce: Callable[[], tuple[str, str, str]],
    *,
    timeout_seconds: float = 300,
    output: TextIO | None = None,
) -> str | None:
    """Run one provider login and return its optional account handle."""

    verifier, challenge, state = build_pkce()
    callback_ready = asyncio.Event()
    loop = asyncio.get_running_loop()

    def notify() -> None:
        loop.call_soon_threadsafe(callback_ready.set)

    receiver = _CallbackReceiver(state, "/callback", notify)
    server = _create_redirect_server(_handler_for(receiver))
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="zeta-login-redirect",
        daemon=True,
    )
    server_thread.start()
    try:
        redirect_uri = f"http://127.0.0.1:{server.server_port}/callback"
        authorization_url = provider.build_authorization_url(state, challenge, redirect_uri)
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
        provider.credential_store.save(tokens)
        return provider.token_handle(tokens)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


__all__ = ["LoginError", "LoginProvider", "run_login"]
