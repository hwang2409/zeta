"""One table for every provider close site that runs behind MustCompleteClose.

Sites: the IteratorProviderAttempt of the real Ollama, Anthropic, and Codex
backends, and the HTTPClientLease behind every backend close. Each row drives
the public close (``attempt.aclose()`` / ``backend.aclose()``) over a real
backend, a real lease, and an ``httpx.MockTransport`` body that the test
controls with events. Time is a manual clock; the bounds are the production
constants.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.support.completion import ManualDeadlineClock
from tests.support.credentials import anthropic_credentials, codex_credentials
from tests.support.owned import await_or_fail
from zeta.oauth import OAuthTokens
from zeta.providers import completion, transport
from zeta.providers.anthropic import AnthropicBackend
from zeta.providers.codex import CodexBackend
from zeta.providers.ollama import OllamaBackend

# loop steps after which a cancel can land: covers every await from the
# close start through the producer drain, the source close, and the
# transport close of a real backend attempt.
CANCEL_STEPS = range(12)


def _codex_token() -> str:
    def encode(value: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    claims = {
        "exp": 4_000_000_000,
        "https://api.openai.com/auth": {"chatgpt_account_id": "account-test"},
    }
    return ".".join((encode({"alg": "none"}), encode(claims), encode({"s": "x"})))


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


FIRST_CHUNK = {
    "ollama": json.dumps({"message": {"role": "assistant", "content": "."}, "done": False}).encode()
    + b"\n",
    "anthropic": _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {"id": "msg-1", "model": "claude-test", "role": "assistant", "usage": {}},
        },
    ),
    "codex": _sse(
        "response.created",
        {"type": "response.created", "response": {"id": "r-1", "model": "codex-test"}},
    ),
}
SOURCE_ERROR = {
    "ollama": "OllamaError",
    "anthropic": "AnthropicStreamError",
    "codex": "CodexHTTPError",
}
CONTENT_TYPE = {
    "ollama": "application/x-ndjson",
    "anthropic": "text/event-stream",
    "codex": "text/event-stream",
}


class _Body(httpx.AsyncByteStream):
    """A provider body: one chunk, then held until the test releases it."""

    def __init__(self, first: bytes) -> None:
        self.first = first
        self.release = asyncio.Event()
        self.read_error: BaseException | None = None
        self.hold_close = False
        self.close_error: BaseException | None = None
        self.close_started = asyncio.Event()
        self.close_cancelled = asyncio.Event()
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        if self.read_error is not None:
            raise self.read_error
        await asyncio.wait_for(self.release.wait(), 10)

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.hold_close:
            try:
                await asyncio.wait_for(self.release.wait(), 10)
            except asyncio.CancelledError:
                self.close_cancelled.set()
                raise
        if self.close_error is not None:
            raise self.close_error


@dataclass
class _Site:
    name: str
    backend: Any
    body: _Body
    clients: list[httpx.AsyncClient] = field(default_factory=list)

    def attempt(self) -> Any:
        if self.name == "ollama":
            return self.backend.create_attempt([], [])
        token = _codex_token() if self.name == "codex" else "token-test"
        return self.backend.create_attempt([], [], token)

    @property
    def borrows(self) -> int:
        return self.backend._lease._active_borrows


_REAL_CLIENT = httpx.AsyncClient


class _ClosingClient(httpx.AsyncClient):
    close_error: BaseException | None = None

    async def aclose(self) -> None:
        await super().aclose()
        if self.close_error is not None:
            raise self.close_error


def _site(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, owned: bool = False
) -> _Site:
    provider = "ollama" if name == "lease" else name
    body = _Body(FIRST_CHUNK[provider])

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": CONTENT_TYPE[provider]}, stream=body
        )

    transport = httpx.MockTransport(respond)
    site = _Site(name, None, body)
    if owned:
        # the lease owns its client: backend close closes it
        def owned_client(**kwargs: object) -> _ClosingClient:
            client = _ClosingClient(transport=transport)
            site.clients.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", owned_client)
        site.backend = OllamaBackend(model="stub", base_url="http://ollama.test")
        return site
    client = _REAL_CLIENT(transport=transport)
    site.clients.append(client)
    if provider == "ollama":
        site.backend = OllamaBackend(model="stub", base_url="http://ollama.test", client=client)
    elif provider == "anthropic":
        store = anthropic_credentials(tmp_path / "zeta.json")
        store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
        site.backend = AnthropicBackend(
            client=client, token_store=store, base_url="https://test.invalid/v1/messages"
        )
    else:
        store = codex_credentials(tmp_path / "codex.json")
        store.save(OAuthTokens(_codex_token(), "refresh-test", 4_000_000_000))
        site.backend = CodexBackend(
            client=client, token_store=store, base_url="https://test.invalid/codex/responses"
        )
    return site


async def _first_event(attempt: Any) -> None:
    while (await await_or_fail(attempt.__anext__(), what="attempt read")).event is None:
        pass


async def _settle(task: asyncio.Task[Any]) -> BaseException | None:
    done = await await_or_fail(
        asyncio.gather(task, return_exceptions=True), what="close task"
    )
    return done[0] if isinstance(done[0], BaseException) else None


async def _steps(count: int) -> None:
    for _ in range(count):
        await asyncio.sleep(0)


async def _closed_backend_failures(site: _Site) -> list[str]:
    outcome = await await_or_fail(site.backend.aclose(), what="backend close")
    return [str(failure.error) for failure in outcome.failures]


# ---- attempt rows: each takes a site whose attempt holds the lease borrow ----


async def _attempt_success(site: _Site, clock: ManualDeadlineClock) -> None:
    attempt = site.attempt()
    await _first_event(attempt)
    await await_or_fail(attempt.aclose(), what="attempt close")
    assert site.body.close_calls == 1
    assert site.borrows == 0, "the closed attempt still holds a borrow"
    assert await _closed_backend_failures(site) == []


def _attempt_cancelled_at(step: int) -> Callable[[_Site, ManualDeadlineClock], Any]:
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing = asyncio.create_task(attempt.aclose())
        await _steps(step)
        closing.cancel()
        await _settle(closing)
        await await_or_fail(attempt.aclose(), what="retried close")
        assert site.borrows == 0, f"cancel at step {step} left the borrow open"
        assert await _closed_backend_failures(site) == []

    return row


async def _attempt_source_failure(site: _Site, clock: ManualDeadlineClock) -> None:
    site.body.read_error = httpx.ReadError("connection reset")
    attempt = site.attempt()
    await _first_event(attempt)
    with pytest.raises(RuntimeError) as raised:
        while True:
            await await_or_fail(attempt.__anext__(), what="attempt read")
    assert type(raised.value).__name__ == SOURCE_ERROR[site.name]
    await await_or_fail(attempt.aclose(), what="attempt close")
    assert site.borrows == 0
    assert await _closed_backend_failures(site) == []


async def _attempt_close_failure(site: _Site, clock: ManualDeadlineClock) -> None:
    site.body.close_error = httpx.ReadError("close failed")
    attempt = site.attempt()
    await _first_event(attempt)
    error = await _settle(asyncio.create_task(attempt.aclose()))
    # the backend reports a stream close failure as a diagnostic, not a raise
    assert error is None
    assert site.body.close_calls == 1
    assert site.borrows == 0
    assert await _closed_backend_failures(site) == []


async def _attempt_duplicate(site: _Site, clock: ManualDeadlineClock) -> None:
    attempt = site.attempt()
    await _first_event(attempt)
    await await_or_fail(
        asyncio.gather(attempt.aclose(), attempt.aclose()), what="two closes"
    )
    await await_or_fail(attempt.aclose(), what="third close")
    assert site.body.close_calls == 1
    assert site.borrows == 0
    assert await _closed_backend_failures(site) == []


async def _attempt_close_after_caller_cancel(
    site: _Site, clock: ManualDeadlineClock
) -> None:
    # the serve and finish-gate shape: the reading task is cancelled, and its
    # cleanup closes the attempt while it is cancelling; the stream close
    # never ends on its own.
    site.body.hold_close = True
    attempt = site.attempt()
    await _first_event(attempt)
    reading = asyncio.Event()

    async def reader() -> None:
        try:
            reading.set()
            while True:
                await attempt.__anext__()
        finally:
            assert asyncio.current_task().cancelling() == 1
            await attempt.aclose()

    task = asyncio.create_task(reader())
    await await_or_fail(reading.wait(), what="reader started")
    await _steps(4)
    task.cancel()
    assert isinstance(await _settle(task), asyncio.CancelledError)
    assert site.borrows == 0, "the cancelled caller's close left the borrow open"
    assert await _closed_backend_failures(site) == []


async def _attempt_never_ends(site: _Site, clock: ManualDeadlineClock) -> None:
    site.body.hold_close = True
    attempt = site.attempt()
    await _first_event(attempt)
    closing = asyncio.create_task(attempt.aclose())
    await await_or_fail(site.body.close_started.wait(), what="stream close started")
    clock.advance(20.0)
    error = await _settle(closing)
    assert str(error) == "provider attempt close did not finish within 20 s"
    assert completion.ATTEMPT_CLOSE_BOUND == 20.0  # the row runs the production bound
    assert site.body.close_cancelled.is_set()
    assert site.borrows == 0
    # a later close waits for the same bounded task and gets the same failure
    later = await _settle(asyncio.create_task(attempt.aclose()))
    assert later is error
    assert await _closed_backend_failures(site) == []


async def _attempt_deadline_abort(site: _Site, clock: ManualDeadlineClock) -> None:
    # the owner of a hard deadline stops waiting: the close still finishes,
    # without waiting on the stream close, and reports no failure.
    site.body.hold_close = True
    attempt = site.attempt()
    await _first_event(attempt)

    async def close_with_deadline(scope_ready: asyncio.Future[asyncio.Timeout]) -> None:
        async with asyncio.timeout(None) as scope:
            scope_ready.set_result(scope)
            await attempt.aclose()

    scope_ready: asyncio.Future[asyncio.Timeout] = asyncio.get_running_loop().create_future()
    closing = asyncio.create_task(close_with_deadline(scope_ready))
    scope = await await_or_fail(scope_ready, what="deadline scope")
    await await_or_fail(site.body.close_started.wait(), what="stream close started")
    scope.reschedule(asyncio.get_running_loop().time())
    assert isinstance(await _settle(closing), TimeoutError)
    await await_or_fail(attempt.aclose(), what="later close")
    assert site.body.close_cancelled.is_set()
    assert site.borrows == 0
    assert await _closed_backend_failures(site) == []


# ---- lease rows: backend.aclose over a lease that owns its client ----


async def _lease_success(site: _Site, clock: ManualDeadlineClock) -> None:
    assert await _closed_backend_failures(site) == []
    assert site.clients[0].is_closed


def _lease_cancelled_at(step: int) -> Callable[[_Site, ManualDeadlineClock], Any]:
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing = asyncio.create_task(site.backend.aclose())
        await _steps(step)
        closing.cancel()
        await _settle(closing)
        await await_or_fail(attempt.aclose(), what="attempt close")
        assert site.borrows == 0
        assert await _closed_backend_failures(site) == []
        assert site.clients[0].is_closed

    return row


async def _lease_client_close_failure(site: _Site, clock: ManualDeadlineClock) -> None:
    site.clients[0].close_error = RuntimeError("client close failed")
    assert await _closed_backend_failures(site) == ["client close failed"]


async def _lease_duplicate(site: _Site, clock: ManualDeadlineClock) -> None:
    attempt = site.attempt()
    await _first_event(attempt)
    closes = [asyncio.create_task(site.backend.aclose()) for _ in range(2)]
    await _steps(2)
    await await_or_fail(attempt.aclose(), what="attempt close")
    outcomes = await await_or_fail(asyncio.gather(*closes), what="two closes")
    assert [outcome.failures for outcome in outcomes] == [(), ()]
    assert await _closed_backend_failures(site) == []
    assert site.clients[0].is_closed


async def _lease_close_after_caller_cancel(
    site: _Site, clock: ManualDeadlineClock
) -> None:
    attempt = site.attempt()
    await _first_event(attempt)
    started = asyncio.Event()

    async def caller() -> None:
        try:
            started.set()
            await asyncio.wait_for(asyncio.Event().wait(), 10)
        finally:
            await site.backend.aclose()

    task = asyncio.create_task(caller())
    await await_or_fail(started.wait(), what="caller started")
    task.cancel()
    assert isinstance(await _settle(task), asyncio.CancelledError)
    await await_or_fail(attempt.aclose(), what="attempt close")
    assert await _closed_backend_failures(site) == []
    assert site.clients[0].is_closed


async def _lease_borrow_never_ends(site: _Site, clock: ManualDeadlineClock) -> None:
    attempt = site.attempt()
    await _first_event(attempt)
    failures: list[str] = []
    closing = asyncio.create_task(
        site.backend.aclose(report_failure=lambda _index, error: failures.append(str(error)))
    )
    await _steps(4)
    clock.advance(30.0)
    outcome = await await_or_fail(closing, what="backend close")
    named = failures + [str(failure.error) for failure in outcome.failures]
    assert named == ["http client closed with borrows still open: ollama completion"]
    assert transport.LEASE_DRAIN_BOUND == 30.0  # the row runs the production bound
    assert site.clients[0].is_closed
    site.body.release.set()
    await await_or_fail(attempt.aclose(), what="attempt close")


ATTEMPT_SITES = ("ollama", "anthropic", "codex")
ATTEMPT_ROWS: list[tuple[str, Callable[[_Site, ManualDeadlineClock], Any]]] = [
    ("success", _attempt_success),
    *((f"cancelled-at-step-{step}", _attempt_cancelled_at(step)) for step in CANCEL_STEPS),
    ("source-failure", _attempt_source_failure),
    ("stream-close-failure", _attempt_close_failure),
    ("duplicate", _attempt_duplicate),
    ("close-after-caller-cancel", _attempt_close_after_caller_cancel),
    ("never-ends-bound", _attempt_never_ends),
    ("hard-deadline-abort", _attempt_deadline_abort),
]
# the lease has no hard-deadline owner: its only deadline is its drain bound.
LEASE_ROWS: list[tuple[str, Callable[[_Site, ManualDeadlineClock], Any]]] = [
    ("success", _lease_success),
    *((f"cancelled-at-step-{step}", _lease_cancelled_at(step)) for step in CANCEL_STEPS),
    ("client-close-failure", _lease_client_close_failure),
    ("duplicate", _lease_duplicate),
    ("close-after-caller-cancel", _lease_close_after_caller_cancel),
    ("never-ends-bound", _lease_borrow_never_ends),
]
TABLE = [
    pytest.param(site, row, id=f"{site}-{name}")
    for site in ATTEMPT_SITES
    for name, row in ATTEMPT_ROWS
] + [pytest.param("lease", row, id=f"lease-{name}") for name, row in LEASE_ROWS]


@pytest.mark.parametrize(("site_name", "row"), TABLE)
@pytest.mark.asyncio
async def test_close_site_releases_its_borrow_and_returns(
    site_name: str,
    row: Callable[[_Site, ManualDeadlineClock], Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = ManualDeadlineClock()
    monkeypatch.setattr(completion, "CLOSE_CLOCK", clock, raising=False)
    site = _site(site_name, tmp_path, monkeypatch, owned=site_name == "lease")
    try:
        await row(site, clock)
    finally:
        site.body.release.set()
        for client in site.clients:
            await await_or_fail(_REAL_CLIENT.aclose(client), what="client close")
