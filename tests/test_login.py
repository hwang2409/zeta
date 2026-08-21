from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import httpx
import pytest

import zeta.core.login_flow as login_flow
from zeta.cli import build_parser
from zeta.core.login_flow import LoginError, LoginProvider, run_login
from zeta.providers.auth import OAuthTokens


@dataclass
class _Store:
    saved: OAuthTokens | None = None

    def save(self, tokens: OAuthTokens) -> None:
        self.saved = tokens


def _provider(
    store: _Store,
    build_url: Callable[[str, str, str], str],
    exchange: Callable[
        [httpx.AsyncClient, str, str, str, str], Awaitable[OAuthTokens]
    ],
) -> LoginProvider[OAuthTokens]:
    return LoginProvider(
        name="test",
        build_authorization_url=build_url,
        exchange_authorization_code=exchange,
        credential_store=store,
        token_handle=lambda tokens: tokens.access_token,
    )


def _post_callback(
    redirect_uri: str, values: dict[str, str], expected_status: int = 200
) -> None:
    url = f"{redirect_uri}?{urlencode(values)}"
    try:
        with urlopen(url, timeout=2) as response:
            assert response.status == expected_status
    except HTTPError as exc:
        if exc.code != expected_status:
            raise


def test_login_parser_defaults_to_anthropic() -> None:
    args = build_parser().parse_args(["login"])

    assert args.command == "login"
    assert args.provider == "anthropic"


@pytest.mark.asyncio
async def test_login_stores_tokens_after_callback() -> None:
    store = _Store()
    expected = OAuthTokens("access", "refresh", 4_000_000_000)
    received: dict[str, str] = {}

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        received.update(state=state, challenge=challenge, redirect_uri=redirect_uri)
        threading.Thread(
            target=_post_callback,
            args=(redirect_uri, {"code": "code", "state": state}),
            daemon=True,
        ).start()
        return f"https://authorize.invalid/?state={state}&code_challenge={challenge}"

    async def exchange(
        client: httpx.AsyncClient,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> OAuthTokens:
        del client
        assert (code, state, redirect_uri) == ("code", received["state"], received["redirect_uri"])
        assert verifier
        return expected

    result = await run_login(
        _provider(store, build_url, exchange),
        lambda: ("verifier", "challenge", "state"),
        timeout_seconds=2,
    )

    assert result == "access"
    assert store.saved == expected
    assert urlsplit(received["redirect_uri"]).hostname == "127.0.0.1"
    assert urlsplit(received["redirect_uri"]).port is not None


@pytest.mark.asyncio
async def test_login_rejects_state_mismatch() -> None:
    store = _Store()
    exchanged = False

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        del challenge
        threading.Thread(
            target=_post_callback,
            args=(redirect_uri, {"code": "code", "state": f"{state}-wrong"}, 400),
            daemon=True,
        ).start()
        return "https://authorize.invalid/"

    async def exchange(
        client: httpx.AsyncClient,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> OAuthTokens:
        nonlocal exchanged
        del client, code, state, verifier, redirect_uri
        exchanged = True
        raise AssertionError("state mismatch must stop before exchange")

    with pytest.raises(LoginError, match="state mismatch"):
        await run_login(
            _provider(store, build_url, exchange),
            lambda: ("verifier", "challenge", "state"),
            timeout_seconds=2,
        )

    assert not exchanged
    assert store.saved is None


@pytest.mark.asyncio
async def test_login_times_out() -> None:
    store = _Store()

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        del state, challenge, redirect_uri
        return "https://authorize.invalid/"

    async def exchange(
        client: httpx.AsyncClient,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> OAuthTokens:
        del client, code, state, verifier, redirect_uri
        raise AssertionError("exchange must not run after timeout")

    with pytest.raises(LoginError, match="timed out after 5 minutes"):
        await run_login(
            _provider(store, build_url, exchange),
            lambda: ("verifier", "challenge", "state"),
            timeout_seconds=0.01,
        )


def test_redirect_server_falls_back_to_ephemeral_port(monkeypatch: pytest.MonkeyPatch) -> None:
    real_server = login_flow.http.server.ThreadingHTTPServer
    calls: list[tuple[str, int]] = []

    def server_factory(address, handler):
        calls.append(address)
        if len(calls) == 1:
            raise OSError("address already in use")
        return real_server(address, handler)

    monkeypatch.setattr(login_flow.http.server, "ThreadingHTTPServer", server_factory)
    server = login_flow._create_redirect_server(
        login_flow._handler_for(login_flow._CallbackReceiver("state", "/callback")),
        preferred_port=54321,
    )
    try:
        assert calls == [("127.0.0.1", 54321), ("127.0.0.1", 0)]
        assert server.server_port != 54321
    finally:
        server.server_close()
