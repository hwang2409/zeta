from __future__ import annotations

import asyncio
import errno
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
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import urlopen

import httpx
import pytest

from zeta.cli.main import build_parser
from zeta.core import login_flow
from zeta.core.login_flow import LoginError, LoginProvider, run_login
from zeta.providers import login as provider_login
from zeta.providers.auth import OAuthTokens
from zeta.providers.codex import (
    build_authorization_url as build_codex_authorization_url,
)
from zeta.providers.factory import credential_store


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
    *,
    name: str = "test",
    callback_port: int = 0,
    callback_path: str = "/callback",
) -> LoginProvider[OAuthTokens]:
    return LoginProvider(
        name=name,
        callback_port=callback_port,
        callback_path=callback_path,
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
        real_builder = provider_login.build_anthropic_authorization_url

        def build_url_for_anthropic(
            state: str, challenge: str, redirect_uri: str
        ) -> str:
            value = real_builder(state, challenge, redirect_uri)
            received["authorization_url"] = value
            build_url(state, challenge, redirect_uri)
            return value

        monkeypatch.setattr(provider_login, "build_anthropic_authorization_url", build_url_for_anthropic)
        monkeypatch.setattr(provider_login, "exchange_anthropic_authorization_code", exchange)
    else:
        real_builder = provider_login.build_codex_authorization_url

        def build_url_for_codex(state: str, challenge: str, redirect_uri: str) -> str:
            value = real_builder(state, challenge, redirect_uri)
            received["authorization_url"] = value
            build_url(state, challenge, redirect_uri)
            return value

        monkeypatch.setattr(provider_login, "build_codex_authorization_url", build_url_for_codex)
        monkeypatch.setattr(provider_login, "exchange_codex_authorization_code", exchange)
        monkeypatch.setattr(provider_login, "extract_account_id", lambda access_token: "account")

    production_provider = await asyncio.to_thread(
        provider_login.build_login_provider, provider, tmp_path
    )
    result = await run_login(
        replace(production_provider, callback_port=0),
        lambda: ("verifier", "challenge", "state"),
        timeout_seconds=2,
        output=StringIO(),
    )

    if provider == "anthropic":
        assert result is None
    else:
        assert result == "account"
    store = await asyncio.to_thread(
        credential_store,
        "claude" if provider == "anthropic" else provider,
        home=tmp_path,
    )
    assert store is not None
    assert await asyncio.to_thread(store.read) == expected
    expected_redirect_uri = {
        "anthropic": "http://localhost:53692/callback",
        "codex": "http://localhost:1455/auth/callback",
    }[provider]
    expected = urlsplit(expected_redirect_uri)
    actual = urlsplit(received["redirect_uri"])
    assert production_provider.callback_port == expected.port
    assert production_provider.callback_path == expected.path
    assert (actual.scheme, actual.hostname, actual.path) == (
        expected.scheme,
        expected.hostname,
        expected.path,
    )
    assert actual.port is not None and actual.port > 0
    assert (
        f"http://localhost:{production_provider.callback_port}{production_provider.callback_path}"
        == expected_redirect_uri
    )
    assert parse_qs(urlsplit(received["authorization_url"]).query)["redirect_uri"] == [
        received["redirect_uri"]
    ]


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

    server = login_flow._create_redirect_server(
        http.server.BaseHTTPRequestHandler, port=0
    )
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
            (
                "import asyncio; from dataclasses import replace; import zeta.cli.main as cli_main; "
                "from zeta.core.login_flow import run_login; "
                "from zeta.providers.login import build_login_provider, pkce_values; "
                "cli_main._run_login = lambda provider, *_args: asyncio.run(run_login("
                "replace(build_login_provider(provider, cli_main.env_home()), callback_port=0), pkce_values)); "
                "raise SystemExit(cli_main.main(['login', '--account', 'default']))"
            ),
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
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.asyncio
async def test_cancelled_login_does_not_publish_before_task_observes_cancellation() -> None:
    tokens = OAuthTokens("access", "refresh", 4_000_000_000)
    save_entered = threading.Event()
    release_save = threading.Event()
    save_finished = threading.Event()

    class Store:
        saved: OAuthTokens | None = None

        def save(
            self,
            value: OAuthTokens,
            *,
            lock_timeout: float,
            cancel_event: threading.Event,
        ) -> None:
            del lock_timeout
            save_entered.set()
            release_save.wait()
            try:
                if not cancel_event.is_set():
                    self.saved = value
            finally:
                save_finished.set()

    store = Store()

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        del challenge
        threading.Thread(
            target=_post_callback,
            args=(redirect_uri, {"code": "code", "state": state}),
            daemon=True,
        ).start()
        return "https://authorize.invalid/"

    async def exchange(*_args: object) -> OAuthTokens:
        return tokens

    task = asyncio.create_task(
        run_login(
            _provider(store, build_url, exchange),  # type: ignore[arg-type]
            lambda: ("verifier", "challenge", "state"),
            timeout_seconds=2,
            publication_timeout_seconds=2,
        )
    )
    assert await asyncio.to_thread(save_entered.wait, 2)

    task.cancel()
    release_save.set()
    assert await asyncio.to_thread(save_finished.wait, 2)

    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.saved is None


@pytest.mark.asyncio
async def test_login_double_cancel_during_publication_reports_success() -> None:
    tokens = OAuthTokens("access", "refresh", 4_000_000_000)
    publication_started = threading.Event()
    release_save = threading.Event()

    class Store:
        saved: OAuthTokens | None = None

        def save(
            self,
            value: OAuthTokens,
            *,
            lock_timeout: float,
            cancel_event: threading.Event,
        ) -> None:
            del lock_timeout
            assert not cancel_event.is_set()
            publication_started.set()
            release_save.wait()
            self.saved = value

    store = Store()

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        del challenge
        threading.Thread(
            target=_post_callback,
            args=(redirect_uri, {"code": "code", "state": state}),
            daemon=True,
        ).start()
        return "https://authorize.invalid/"

    async def exchange(*_args: object) -> OAuthTokens:
        return tokens

    task = asyncio.create_task(
        run_login(
            _provider(store, build_url, exchange),  # type: ignore[arg-type]
            lambda: ("verifier", "challenge", "state"),
            timeout_seconds=2,
            publication_timeout_seconds=2,
        )
    )
    assert await asyncio.to_thread(publication_started.wait, 2)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release_save.set()

    assert await task == "access"
    assert store.saved == tokens


@pytest.mark.asyncio
async def test_login_publication_commit_is_arbitrated_on_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = OAuthTokens("access", "refresh", 4_000_000_000)
    loop_thread = threading.get_ident()
    loop_task = asyncio.current_task()
    assert loop_task is not None

    class TaskProbe:
        def cancelling(self) -> int:
            assert threading.get_ident() == loop_thread
            return loop_task.cancelling()

        def uncancel(self) -> int:
            assert threading.get_ident() == loop_thread
            return loop_task.uncancel()

    monkeypatch.setattr(asyncio, "current_task", lambda: TaskProbe())
    commit_threads: list[int] = []
    save_threads: list[int] = []

    class Store:
        def save(
            self,
            value: OAuthTokens,
            *,
            lock_timeout: float,
            cancel_event: threading.Event,
        ) -> None:
            del value, lock_timeout
            save_threads.append(threading.get_ident())
            assert not cancel_event.is_set()

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        del challenge
        threading.Thread(
            target=_post_callback,
            args=(redirect_uri, {"code": "code", "state": state}),
            daemon=True,
        ).start()
        return "https://authorize.invalid/"

    async def exchange(*_args: object) -> OAuthTokens:
        return tokens

    def commit() -> bool:
        commit_threads.append(threading.get_ident())
        return True

    await run_login(
        _provider(Store(), build_url, exchange),  # type: ignore[arg-type]
        lambda: ("verifier", "challenge", "state"),
        timeout_seconds=2,
        publication_timeout_seconds=2,
        on_publication_commit=commit,
    )

    assert commit_threads == [loop_thread]
    assert save_threads and save_threads[0] != loop_thread


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


@pytest.mark.asyncio
async def test_login_rejects_callback_port_in_use() -> None:
    store = _Store()

    def build_url(state: str, challenge: str, redirect_uri: str) -> str:
        del state, challenge, redirect_uri
        raise AssertionError("authorization must not start when the port is busy")

    async def exchange(
        client: httpx.AsyncClient,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> OAuthTokens:
        del client, code, state, verifier, redirect_uri
        raise AssertionError("exchange must not run when the port is busy")

    provider = _provider(
        store,
        build_url,
        exchange,
        name="codex",
        callback_port=0,
        callback_path="/auth/callback",
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        callback_port = listener.getsockname()[1]
        provider = replace(provider, callback_port=callback_port)
        with pytest.raises(LoginError, match=rf"port {callback_port}.*Codex login"):
            await run_login(
                provider,
                lambda: ("verifier", "challenge", "state"),
            )


@pytest.mark.asyncio
async def test_login_reraises_non_port_bind_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(
        client: httpx.AsyncClient,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> OAuthTokens:
        del client, code, state, verifier, redirect_uri
        raise AssertionError("exchange must not run when the callback server fails")

    def raise_permission_denied(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(login_flow, "_create_redirect_server", raise_permission_denied)
    provider = _provider(_Store(), lambda *_: "https://authorize.invalid/", exchange)

    with pytest.raises(OSError, match="permission denied") as exc_info:
        await run_login(provider, lambda: ("verifier", "challenge", "state"))

    assert exc_info.value.errno == errno.EACCES
