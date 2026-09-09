from __future__ import annotations

import http.server
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import urlopen

import httpx
import pytest

from zeta.cli import build_parser
from zeta.core import login_flow
from zeta.core.login_flow import LoginError, LoginProvider, run_login
from zeta.providers import login as provider_login
from zeta.providers.anthropic import AnthropicCredentialStore
from zeta.providers.auth import OAuthTokens
from zeta.providers.codex import CodexCredentialStore
from zeta.providers.codex import (
    build_authorization_url as build_codex_authorization_url,
)


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


@pytest.mark.parametrize("provider", ["anthropic", "codex"])
@pytest.mark.asyncio
async def test_login_dispatches_provider_and_stores_exchange_result(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    if provider == "anthropic":
        store = AnthropicCredentialStore(tmp_path / "anthropic-oauth.json")
        real_builder = provider_login.build_anthropic_authorization_url

        def build_url_for_anthropic(
            state: str, challenge: str, redirect_uri: str
        ) -> str:
            value = real_builder(state, challenge, redirect_uri)
            build_url(state, challenge, redirect_uri)
            return value

        monkeypatch.setattr(provider_login, "build_anthropic_authorization_url", build_url_for_anthropic)
        monkeypatch.setattr(provider_login, "exchange_anthropic_authorization_code", exchange)
    else:
        store = CodexCredentialStore(tmp_path / "codex-oauth.json")
        real_builder = provider_login.build_codex_authorization_url

        def build_url_for_codex(state: str, challenge: str, redirect_uri: str) -> str:
            value = real_builder(state, challenge, redirect_uri)
            build_url(state, challenge, redirect_uri)
            return value

        monkeypatch.setattr(provider_login, "build_codex_authorization_url", build_url_for_codex)
        monkeypatch.setattr(provider_login, "exchange_codex_authorization_code", exchange)
        monkeypatch.setattr(provider_login, "extract_account_id", lambda access_token: "account")

    result = await run_login(
        provider_login.build_login_provider(provider, tmp_path),
        lambda: ("verifier", "challenge", "state"),
        timeout_seconds=2,
        output=StringIO(),
    )

    if provider == "anthropic":
        assert result is None
    else:
        assert result == "account"
    assert store.read() == expected
    assert urlsplit(received["redirect_uri"]).hostname == "localhost"
    assert urlsplit(received["redirect_uri"]).port is not None


def test_codex_authorization_url_includes_upstream_flow_fields() -> None:
    query = parse_qs(
        urlsplit(build_codex_authorization_url("state", "challenge", "http://callback")).query
    )

    assert query["id_token_add_organizations"] == ["true"]
    assert query["codex_cli_simplified_flow"] == ["true"]
    assert query["originator"] == ["codex_cli_rs"]


def test_redirect_server_binds_without_reverse_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZETA-91: binding must never reverse-resolve 127.0.0.1.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn`` on the bind address;
    on a host whose resolver has no answer for it the lookup blocks until the
    resolver times out and ``zeta login`` prints nothing meanwhile.
    """

    def _no_lookup(name: str = "") -> str:
        raise AssertionError(f"reverse lookup attempted for {name!r}")

    monkeypatch.setattr(socket, "getfqdn", _no_lookup)

    server = login_flow._create_redirect_server(http.server.BaseHTTPRequestHandler)
    try:
        assert server.server_name == "localhost"
        assert server.server_port == server.server_address[1]
        assert server.server_port > 0
    finally:
        server.server_close()


def test_login_sigint_closes_callback_server(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["ZETA_HOME"] = str(tmp_path / "zeta-home")
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "from zeta.cli import main; raise SystemExit(main(['login']))",
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    port: int | None = None
    output = bytearray()
    try:
        # Poll-until ceiling, not a budget (ZETA-67 pattern): a cold interpreter
        # on a loaded runner plus ThreadingHTTPServer's reverse-DNS lookup of
        # the bind address blew a 3s deadline on macos-latest (ZETA-90). The
        # loop exits the moment the URL appears, so the ceiling costs nothing
        # on the happy path.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and port is None:
            if process.stdout is None or process.poll() is not None:
                break
            readable, _, _ = select.select([process.stdout], [], [], 0.1)
            if not readable:
                continue
            output.extend(os.read(process.stdout.fileno(), 4096))
            for line in output.splitlines():
                if line.startswith(b"https://"):
                    query = parse_qs(urlsplit(line.decode()).query)
                    port = urlsplit(query["redirect_uri"][0]).port
                    break
        if port is None:
            # Say why: a login that died before printing looks the same as a
            # slow one unless the exit status and stderr are reported.
            process.kill()
            stdout_rest, stderr_rest = process.communicate(timeout=5)
            output.extend(stdout_rest)
            raise AssertionError(
                f"login never printed its URL (exit={process.returncode})\n"
                f"stdout: {output.decode(errors='replace')!r}\n"
                f"stderr: {stderr_rest.decode(errors='replace')!r}"
            )
        process.send_signal(signal.SIGINT)
        shutdown_deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < shutdown_deadline:
            time.sleep(0.01)
        assert process.poll() == 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


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
    real_server = login_flow._RedirectServer
    calls: list[tuple[str, int]] = []

    def server_factory(address, handler):
        calls.append(address)
        if len(calls) == 1:
            raise OSError("address already in use")
        return real_server(address, handler)

    monkeypatch.setattr(login_flow, "_RedirectServer", server_factory)
    server = login_flow._create_redirect_server(
        login_flow._handler_for(login_flow._CallbackReceiver("state", "/callback")),
        preferred_port=54321,
    )
    try:
        assert calls == [("127.0.0.1", 54321), ("127.0.0.1", 0)]
        assert server.server_port != 54321
    finally:
        server.server_close()
