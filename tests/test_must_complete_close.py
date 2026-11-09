"""One table for every provider close site that runs behind MustCompleteClose.

Sites: the IteratorProviderAttempt of the real Ollama, Anthropic, and Codex
backends, and the HTTPClientLease behind every backend close. Each row drives
the public close (``attempt.aclose()`` / ``backend.aclose()``) over a real
backend, a real lease, and an ``httpx.MockTransport`` body that the test
controls with events. Time is a manual clock; the bounds are the production
constants.

Cancellation rows park the close at one named cleanup await (a seam) and
prove it with events at that await, then act there:

- ``paused-producer-drain``: the producer was parked between reads, and
  the attempt close waits for it to unwind (its error put blocks).
- ``producer-drain``: the attempt close waits for its cancelled producer,
  whose stream close outlives the producer's own cancellation.
- ``source-close``: the producer has drained and the attempt close waits on
  the stream close of the source.
- ``drain-wait``: the lease close waits for an attempt's borrow to end.
- ``client-close``: the lease close waits on its owned client's close.
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
from zeta.providers import completion, must_complete_close, transport
from zeta.providers.anthropic import AnthropicBackend
from zeta.providers.codex import CodexBackend
from zeta.providers.ollama import OllamaBackend

ATTEMPT_SEAMS = ("producer-drain", "source-close")
LEASE_SEAMS = ("drain-wait", "client-close")
ATTEMPT_BOUND_FAILURE = "provider attempt close did not finish within 20 s"
LEASE_BOUND_FAILURE = "http client close did not finish within 30 s"
# the test's gates time out later than the 10 s hang guards, so a guard
# reports a hung close (and its teardown releases the gates) first.
GATE_TIMEOUT = 30


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
    """A provider body: one chunk, then held until the test releases it.

    With ``hold_close`` its close also waits for the release; it catches the
    first ``swallow`` cancellations and keeps waiting, like a stream close
    that ignores a cancel. ``cancel_seen[n]`` is set by the (n+1)th cancel.
    """

    def __init__(self, first: bytes) -> None:
        self.first = first
        self.release = asyncio.Event()
        self.read_error: BaseException | None = None
        self.read_parked = asyncio.Event()
        self.hold_close = False
        self.swallow = 0
        self.close_error: BaseException | None = None
        self.close_started = asyncio.Event()
        self.cancel_seen = [asyncio.Event() for _ in range(4)]
        self.close_cancelled = asyncio.Event()
        self.close_calls = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        if self.read_error is not None:
            raise self.read_error
        self.read_parked.set()
        await asyncio.wait_for(self.release.wait(), GATE_TIMEOUT)

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        cancels = 0
        while self.hold_close and not self.release.is_set():
            try:
                await asyncio.wait_for(self.release.wait(), GATE_TIMEOUT)
            except asyncio.CancelledError:
                self.cancel_seen[cancels].set()
                cancels += 1
                if cancels > self.swallow:
                    self.close_cancelled.set()
                    raise
        if self.close_error is not None:
            raise self.close_error


class _GatedQueue(asyncio.Queue):
    """An attempt's signal queue whose error put blocks until ``release``.

    A cancelled producer puts its cancellation here while it unwinds, so
    ``error_put`` proves the attempt close is waiting on the producer drain.
    """

    def __init__(self) -> None:
        super().__init__()
        self.error_put = asyncio.Event()
        self.release = asyncio.Event()

    async def put(self, item: object) -> None:
        if isinstance(item, BaseException):
            self.error_put.set()
            await asyncio.wait_for(self.release.wait(), GATE_TIMEOUT)
        await super().put(item)


class _WatchedEvent(asyncio.Event):
    """The lease's drained event; ``waited`` is set when its close waits on it."""

    def __init__(self) -> None:
        super().__init__()
        self.set()
        self.waited = asyncio.Event()

    async def wait(self) -> bool:
        self.waited.set()
        return await asyncio.wait_for(super().wait(), GATE_TIMEOUT)


@dataclass
class _Site:
    name: str
    backend: Any
    body: _Body
    clients: list[httpx.AsyncClient] = field(default_factory=list)
    attempts: list[Any] = field(default_factory=list)

    def attempt(self) -> Any:
        if self.name == "ollama":
            attempt = self.backend.create_attempt([], [])
        else:
            token = _codex_token() if self.name == "codex" else "token-test"
            attempt = self.backend.create_attempt([], [], token)
        self.attempts.append(attempt)
        return attempt

    @property
    def borrows(self) -> int:
        return self.backend._lease._active_borrows

    @property
    def drained(self) -> _WatchedEvent:
        return self.backend._lease._drained


_REAL_CLIENT = httpx.AsyncClient


class _ClosingClient(httpx.AsyncClient):
    """An owned lease client whose close the test can fail or hold."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.close_error: BaseException | None = None
        self.hold = False
        self.swallow = False
        self.release = asyncio.Event()
        self.close_entered = asyncio.Event()
        self.close_cancelled = asyncio.Event()

    async def aclose(self) -> None:
        await super().aclose()
        self.close_entered.set()
        while self.hold and not self.release.is_set():
            try:
                await asyncio.wait_for(self.release.wait(), GATE_TIMEOUT)
            except asyncio.CancelledError:
                self.close_cancelled.set()
                if not self.swallow:
                    raise
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


async def _drain_left() -> None:
    """Wait for every other task on this test's loop to end."""
    left = asyncio.all_tasks() - {asyncio.current_task()}
    await await_or_fail(
        asyncio.gather(*left, return_exceptions=True), what="tasks the row left"
    )


async def _ended_or_drained(task: asyncio.Task[Any]) -> bool:
    """Wait for a close the row expects to end; return whether it ended.

    On the 10 s hang guard the task is cancelled and drained first, so the
    row's behavior assertion is the reported failure and no close task
    outlives it.
    """
    done, _ = await asyncio.wait({task}, timeout=10)
    if task in done:
        return True
    task.cancel()
    await _settle(task)
    return False


async def _returned_at_bound(task: asyncio.Task[Any], what: str) -> Any:
    """Return the outcome of a close the production bound must have ended.

    The 10 s wait is a hang guard, not the proof: the clock already passed
    the bound, so a close that still runs waits forever.
    """
    ended = await _ended_or_drained(task)
    assert ended, f"the bound expired but the {what} still waits"
    return task.exception() or task.result()


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
    await await_or_fail(site.body.read_parked.wait(), what="producer reading")
    task.cancel()
    assert isinstance(await _settle(task), asyncio.CancelledError)
    assert site.borrows == 0, "the cancelled caller's close left the borrow open"
    assert await _closed_backend_failures(site) == []


async def _read_all(attempt: Any) -> None:
    while True:
        await attempt.__anext__()


async def _park_attempt_close(
    seam: str, site: _Site, attempt: Any
) -> tuple[asyncio.Task[None], asyncio.Task[Any] | None]:
    """Start ``attempt.aclose()`` and return once its cleanup waits at ``seam``."""
    site.body.hold_close = True
    reader = None
    if seam == "producer-drain":
        # the producer is reading when the close cancels it; its stream close
        # outlives that cancel, so the close waits on the producer drain.
        site.body.swallow += 1
        reader = asyncio.create_task(_read_all(attempt))
        await await_or_fail(site.body.read_parked.wait(), what="producer reading")
        closing = asyncio.create_task(attempt.aclose())
        await await_or_fail(site.body.cancel_seen[0].wait(), what="producer cancelled")
        # the stream close runs inside the cancelled producer, which cannot end
        # while that close waits: the attempt close is parked on its drain.
        # test-contract: seam the producer task has no public view
        assert not attempt._producer.done(), "the close is not waiting on its producer drain"
    else:
        closing = asyncio.create_task(attempt.aclose())
        await await_or_fail(site.body.close_started.wait(), what="stream close started")
        # the producer was parked between reads and has ended: the stream close
        # was called by the attempt close itself, on its source.
        # test-contract: seam the producer task has no public view
        assert attempt._producer.done(), "the close is not at its source close"
    assert not closing.done()
    return closing, reader


async def _finish_reader(site: _Site, reader: asyncio.Task[Any] | None) -> None:
    site.body.release.set()
    if reader is not None:
        await await_or_fail(
            asyncio.gather(reader, return_exceptions=True), what="attempt reader"
        )


async def _owner_returns(attempt: Any) -> BaseException | None:
    # a later close waits on the same owner task: it returns only once the
    # owner task has ended. the 10 s wait is a hang guard, not the proof.
    later = asyncio.create_task(attempt.aclose())
    ended = await _ended_or_drained(later)
    assert ended, "the attempt close owner task is still live"
    return later.exception()


async def _attempt_caller_cancelled_at_paused_producer(
    site: _Site, clock: ManualDeadlineClock
) -> None:
    # the bench serve hang: the producer is parked between reads, the close
    # cancels it and waits for it to unwind, and the caller is cancelled
    # there. the close must still close the source and end the borrow.
    attempt = site.attempt()
    await _first_event(attempt)
    queue = _GatedQueue()
    # test-contract: seam the producer's unwind blocks on a named event
    attempt._queue = queue
    closing = asyncio.create_task(attempt.aclose())
    try:
        await await_or_fail(queue.error_put.wait(), what="producer unwinding")
        assert not closing.done()
        closing.cancel()
        assert isinstance(await _settle(closing), asyncio.CancelledError)
        await await_or_fail(attempt.aclose(), what="retried close")
        assert site.borrows == 0, "the cancelled close left the borrow open"
        assert await _closed_backend_failures(site) == []
    finally:
        queue.release.set()
        await _settle(closing)


def _attempt_caller_cancelled_at(
    seam: str,
) -> Callable[[_Site, ManualDeadlineClock], Any]:
    # the caller is cancelled while the cleanup waits at the seam: the abort
    # cancels that await, and the close still releases the borrow by itself.
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing, reader = await _park_attempt_close(seam, site, attempt)
        closing.cancel()
        assert isinstance(await _settle(closing), asyncio.CancelledError)
        assert await _owner_returns(attempt) is None, "an abort is not a failure"
        assert site.body.close_cancelled.is_set()
        assert not site.body.release.is_set()
        assert await _closed_backend_failures(site) == [], f"abort at {seam} kept the borrow"
        await _finish_reader(site, reader)

    return row


def _attempt_bound_at(seam: str) -> Callable[[_Site, ManualDeadlineClock], Any]:
    # nobody aborts: the production bound cancels the cleanup at the seam.
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing, reader = await _park_attempt_close(seam, site, attempt)
        clock.advance(clock.now() + completion.ATTEMPT_CLOSE_BOUND)
        assert str(await _returned_at_bound(closing, "attempt close")) == ATTEMPT_BOUND_FAILURE
        assert completion.ATTEMPT_CLOSE_BOUND == 20.0  # the production bound
        error = await _owner_returns(attempt)
        assert str(error) == ATTEMPT_BOUND_FAILURE
        assert site.body.close_cancelled.is_set()
        assert await _closed_backend_failures(site) == [], f"bound at {seam} kept the borrow"
        await _finish_reader(site, reader)

    return row


def _attempt_abort_swallowed_at(
    seam: str,
) -> Callable[[_Site, ManualDeadlineClock], Any]:
    # the serve two-cancel shape: the caller's abort reaches the stream close,
    # which catches it and stays blocked. the bound must still end the close.
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing, reader = await _park_attempt_close(seam, site, attempt)
        site.body.swallow += 1
        abort_index = 1 if seam == "producer-drain" else 0
        closing.cancel()
        assert isinstance(await _settle(closing), asyncio.CancelledError)
        await await_or_fail(
            site.body.cancel_seen[abort_index].wait(), what="abort reached the stream"
        )
        clock.advance(clock.now() + completion.ATTEMPT_CLOSE_BOUND)
        error = await _owner_returns(attempt)
        assert str(error) == ATTEMPT_BOUND_FAILURE
        assert not site.body.release.is_set()
        # the bound's cancel ended the stream close, so the borrow ended
        assert await _closed_backend_failures(site) == []
        await _finish_reader(site, reader)

    return row


def _attempt_bound_swallowed_at(
    seam: str,
) -> Callable[[_Site, ManualDeadlineClock], Any]:
    # the stream close catches the bound's cancel too: the owner stops
    # waiting anyway, and the lease close names the borrow it still holds.
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing, reader = await _park_attempt_close(seam, site, attempt)
        site.body.swallow += 1
        clock.advance(clock.now() + completion.ATTEMPT_CLOSE_BOUND)
        assert str(await _returned_at_bound(closing, "attempt close")) == ATTEMPT_BOUND_FAILURE
        assert str(await _owner_returns(attempt)) == ATTEMPT_BOUND_FAILURE
        assert not site.body.release.is_set()
        backend_close = asyncio.create_task(site.backend.aclose())
        await await_or_fail(site.drained.waited.wait(), what="lease drain wait")
        clock.advance(clock.now() + transport.LEASE_DRAIN_BOUND)
        outcome = await _returned_at_bound(backend_close, "backend close")
        named = [str(failure.error) for failure in outcome.failures]
        assert named == [f"{LEASE_BOUND_FAILURE}: borrows still open: {site.name} completion"]
        await _finish_reader(site, reader)

    return row


def _attempt_two_aborts_then_bound_at(
    seam: str,
) -> Callable[[_Site, ManualDeadlineClock], Any]:
    # the serve shape: a client disconnect cancels the reading turn (abort 1),
    # then server close cancels the task that closes it (aborts 2 and 3). the
    # stream close catches every cancel and never ends; the production bound
    # alone must end the close, and the lease bound then names the borrow.
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        attempt = site.attempt()
        await _first_event(attempt)
        closing, reader = await _park_attempt_close(seam, site, attempt)
        site.body.swallow = len(site.body.cancel_seen)
        first_abort = 1 if seam == "producer-drain" else 0
        closing.cancel()  # client disconnect
        assert isinstance(await _settle(closing), asyncio.CancelledError)
        await await_or_fail(
            site.body.cancel_seen[first_abort].wait(), what="abort reached the stream"
        )
        serving = asyncio.Event()
        closing_again = asyncio.Event()

        async def server_close() -> None:
            try:
                serving.set()
                await asyncio.wait_for(asyncio.Event().wait(), 10)
            finally:
                closing_again.set()  # the next line parks this task in aclose
                await attempt.aclose()

        server = asyncio.create_task(server_close())
        await await_or_fail(serving.wait(), what="server task started")
        server.cancel()  # server close: its cleanup closes the attempt aborted
        await await_or_fail(closing_again.wait(), what="server close reached aclose")
        server.cancel()  # server close gives up on that close too
        assert isinstance(await _settle(server), asyncio.CancelledError)
        # test-contract: seam no public view of the owner task itself
        assert not attempt._close.task.done(), "the close ended before its bound"
        clock.advance(clock.now() + completion.ATTEMPT_CLOSE_BOUND)
        assert str(await _owner_returns(attempt)) == ATTEMPT_BOUND_FAILURE
        assert not site.body.release.is_set()
        backend_close = asyncio.create_task(site.backend.aclose())
        await await_or_fail(site.drained.waited.wait(), what="lease drain wait")
        clock.advance(clock.now() + transport.LEASE_DRAIN_BOUND)
        outcome = await _returned_at_bound(backend_close, "backend close")
        named = [str(failure.error) for failure in outcome.failures]
        assert named == [f"{LEASE_BOUND_FAILURE}: borrows still open: {site.name} completion"]
        await _finish_reader(site, reader)

    return row


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


async def _lease_client_close_failure(site: _Site, clock: ManualDeadlineClock) -> None:
    site.clients[0].close_error = RuntimeError("client close failed")
    assert await _closed_backend_failures(site) == ["client close failed"]


async def _lease_duplicate(site: _Site, clock: ManualDeadlineClock) -> None:
    attempt = site.attempt()
    await _first_event(attempt)
    closes = [asyncio.create_task(site.backend.aclose()) for _ in range(2)]
    await await_or_fail(site.drained.waited.wait(), what="lease drain wait")
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


async def _park_lease_close(
    seam: str, site: _Site
) -> tuple[asyncio.Task[Any], Any]:
    """Start ``backend.aclose()`` and return once the lease waits at ``seam``."""
    attempt = site.attempt()
    await _first_event(attempt)
    if seam == "client-close":
        site.clients[0].hold = True
        await await_or_fail(attempt.aclose(), what="attempt close")
    closing = asyncio.create_task(site.backend.aclose())
    if seam == "drain-wait":
        await await_or_fail(site.drained.waited.wait(), what="lease drain wait")
        assert site.borrows == 1
    else:
        await await_or_fail(site.clients[0].close_entered.wait(), what="client close")
    assert not closing.done()
    return closing, attempt


def _lease_caller_cancelled_at(
    seam: str,
) -> Callable[[_Site, ManualDeadlineClock], Any]:
    # the lease close has no abort: a cancelled caller leaves it running.
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        closing, attempt = await _park_lease_close(seam, site)
        closing.cancel()
        assert isinstance(await _settle(closing), asyncio.CancelledError)
        if seam == "drain-wait":
            await await_or_fail(attempt.aclose(), what="attempt close")
        else:
            assert not site.clients[0].close_cancelled.is_set()
            site.clients[0].release.set()
        assert await _closed_backend_failures(site) == []
        assert site.clients[0].is_closed

    return row


def _lease_bound_at(seam: str) -> Callable[[_Site, ManualDeadlineClock], Any]:
    async def row(site: _Site, clock: ManualDeadlineClock) -> None:
        closing, attempt = await _park_lease_close(seam, site)
        clock.advance(clock.now() + transport.LEASE_DRAIN_BOUND)
        outcome = await _returned_at_bound(closing, "backend close")
        assert transport.LEASE_DRAIN_BOUND == 30.0  # the production bound
        named = [str(failure.error) for failure in outcome.failures]
        if seam == "drain-wait":
            assert named == [f"{LEASE_BOUND_FAILURE}: borrows still open: ollama completion"]
            site.body.release.set()
            await await_or_fail(attempt.aclose(), what="attempt close")
        else:
            assert named == [LEASE_BOUND_FAILURE]
            assert site.clients[0].close_cancelled.is_set()
        assert site.clients[0].is_closed

    return row


async def _lease_late_failure_after_bound(
    site: _Site, clock: ManualDeadlineClock
) -> None:
    # the client close catches the bound's cancel and fails only later: the
    # backend close returns at the bound, and the late failure still reaches
    # the loop's exception handler instead of vanishing.
    site.clients[0].swallow = True
    site.clients[0].close_error = RuntimeError("late client close failure")
    closing, _ = await _park_lease_close("client-close", site)
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    published = asyncio.Event()

    def report(_loop: object, context: dict[str, Any]) -> None:
        reported.append(context)
        published.set()

    previous = loop.get_exception_handler()
    loop.set_exception_handler(report)
    try:
        clock.advance(clock.now() + transport.LEASE_DRAIN_BOUND)
        outcome = await _returned_at_bound(closing, "backend close")
        assert [str(failure.error) for failure in outcome.failures] == [LEASE_BOUND_FAILURE]
        await await_or_fail(site.clients[0].close_cancelled.wait(), what="bound cancel")
        assert reported == []
        site.clients[0].release.set()
        await await_or_fail(published.wait(), what="late failure report")
        assert [(c["message"], str(c["exception"])) for c in reported] == [
            ("http client close failed", "late client close failure")
        ]
    finally:
        site.clients[0].release.set()
        await _settle(closing)
        await _drain_left()  # a late report lands here, not in a later row
        loop.set_exception_handler(previous)


ATTEMPT_SITES = ("ollama", "anthropic", "codex")
ATTEMPT_ROWS: list[tuple[str, Callable[[_Site, ManualDeadlineClock], Any]]] = [
    ("success", _attempt_success),
    ("source-failure", _attempt_source_failure),
    ("stream-close-failure", _attempt_close_failure),
    ("duplicate", _attempt_duplicate),
    ("close-after-caller-cancel", _attempt_close_after_caller_cancel),
    ("paused-producer-drain-caller-cancelled", _attempt_caller_cancelled_at_paused_producer),
    ("hard-deadline-abort", _attempt_deadline_abort),
    *(
        (f"{seam}-{name}", make(seam))
        for seam in ATTEMPT_SEAMS
        for name, make in (
            ("caller-cancelled", _attempt_caller_cancelled_at),
            ("bound", _attempt_bound_at),
            ("abort-swallowed-bound", _attempt_abort_swallowed_at),
            ("bound-swallowed", _attempt_bound_swallowed_at),
            ("two-aborts-swallowed-bound", _attempt_two_aborts_then_bound_at),
        )
    ),
]
# the lease has no hard-deadline owner: its only deadline is its drain bound.
LEASE_ROWS: list[tuple[str, Callable[[_Site, ManualDeadlineClock], Any]]] = [
    ("success", _lease_success),
    ("client-close-failure", _lease_client_close_failure),
    ("duplicate", _lease_duplicate),
    ("close-after-caller-cancel", _lease_close_after_caller_cancel),
    ("late-failure-after-bound", _lease_late_failure_after_bound),
    *(
        (f"{seam}-{name}", make(seam))
        for seam in LEASE_SEAMS
        for name, make in (
            ("caller-cancelled", _lease_caller_cancelled_at),
            ("bound", _lease_bound_at),
        )
    ),
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
    monkeypatch.setattr(must_complete_close, "CLOSE_CLOCK", clock)
    site = _site(site_name, tmp_path, monkeypatch, owned=site_name == "lease")
    # test-contract: seam observe the lease close parked at its drain wait
    site.backend._lease._drained = _WatchedEvent()
    try:
        await row(site, clock)
    finally:
        site.body.release.set()
        for client in site.clients:
            if isinstance(client, _ClosingClient):
                client.release.set()
        # a failed row can leave attempts, close owners, cleanups, and readers
        # running: with the gates open they end, and are drained here.
        for attempt in site.attempts:
            await await_or_fail(
                asyncio.gather(attempt.aclose(), return_exceptions=True),
                what="attempt close",
            )
        await _drain_left()
        for client in site.clients:
            await await_or_fail(_REAL_CLIENT.aclose(client), what="client close")
