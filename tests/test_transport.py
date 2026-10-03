import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.providers.codex_errors import CodexHTTPError
from zeta.providers.transport import retry_provider_completion, retryable_provider_error


@pytest.mark.parametrize("status_code", [520, 521, 522, 523, 524, 529])
def test_cloudflare_transient_statuses_are_retryable(status_code: int) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)

    assert retryable_provider_error(error)


def test_wrapped_request_timeout_is_retryable() -> None:
    error = CodexHTTPError("request failed")
    error.__cause__ = TimeoutError()

    assert retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [525, 526, 527, 528, 530, 400, 401, 404])
def test_non_retryable_statuses_stay_non_retryable(status_code: int) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)

    assert not retryable_provider_error(error)


async def _unused_retry(_token: str) -> AsyncIterator[StreamEvent]:
    raise AssertionError("auth retry is unreachable")
    yield


async def _unused_refresh() -> str:
    raise AssertionError("auth refresh is unreachable")


def _retry_notice(number: int, delay: float, _error: RuntimeError) -> StreamEvent:
    return StreamEvent(
        StreamEventType.RETRY,
        data={"retry": number, "delay": delay},
    )


@pytest.mark.asyncio
async def test_401_is_not_retried() -> None:
    attempts = 0

    async def fail() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        raise CodexHTTPError("unauthorized", status_code=401)
        yield

    with pytest.raises(CodexHTTPError, match="unauthorized"):
        [
            event
            async for event in retry_provider_completion(
                fail,
                _unused_retry,
                _unused_refresh,
                lambda _error: False,
                lambda error: error,
                lambda event: event.type is StreamEventType.MESSAGE_START,
                retryable_provider_error,
                _retry_notice,
                lambda _error, _retries: None,
            )
        ]

    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_sleep_is_injectable_and_cancellation_interrupts_it() -> None:
    request = httpx.Request("POST", "https://example.invalid")
    sleep_started = asyncio.Event()

    async def fail() -> AsyncIterator[StreamEvent]:
        raise CodexHTTPError("request failed") from httpx.ConnectError(
            "offline", request=request
        )
        yield

    async def blocking_sleep(_delay: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    async def consume() -> None:
        async for _event in retry_provider_completion(
            fail,
            _unused_retry,
            _unused_refresh,
            lambda _error: False,
            lambda error: error,
            lambda event: event.type is StreamEventType.MESSAGE_START,
            retryable_provider_error,
            _retry_notice,
            lambda _error, _retries: None,
            sleep=blocking_sleep,
        ):
            pass

    task = asyncio.create_task(consume())
    await sleep_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_retry_window_uses_injected_clock() -> None:
    request = httpx.Request("POST", "https://example.invalid")
    attempts = 0
    now = 0.0
    sleeps: list[float] = []

    async def fail() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        raise CodexHTTPError("request failed", retry_after=1.0) from httpx.ConnectError(
            "offline", request=request
        )
        yield

    async def advance(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    with pytest.raises(CodexHTTPError, match="request failed"):
        [
            event
            async for event in retry_provider_completion(
                fail,
                _unused_retry,
                _unused_refresh,
                lambda _error: False,
                lambda error: error,
                lambda event: event.type is StreamEventType.MESSAGE_START,
                retryable_provider_error,
                _retry_notice,
                lambda _error, _retries: None,
                sleep=advance,
                clock=lambda: now,
                retry_window_seconds=1.0,
            )
        ]

    assert attempts == 2
    assert len(sleeps) == 1
