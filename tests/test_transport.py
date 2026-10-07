import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

import zeta.providers.retry_policy as retry_policy_module
from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.providers.anthropic_errors import AnthropicStreamError
from zeta.providers.codex_errors import CodexHTTPError
from zeta.providers.ollama import OllamaError
from zeta.providers.retry_policy import ProviderRetryBudget, retryable_provider_error
from zeta.providers.transport import retry_provider_completion


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


def test_http_400_retry_hint_does_not_override_terminal_status() -> None:
    error = CodexHTTPError("overloaded", status_code=400)
    error.retryable = True

    assert not retryable_provider_error(error)


@pytest.mark.parametrize("code", ["auth_error", "permission_denied", "context_length_exceeded"])
def test_terminal_code_ignores_retryable_event_data(code: str) -> None:
    error = RuntimeError("terminal provider error")
    error.code = code

    assert not retryable_provider_error(error, {"retryable": True})


def test_http_429_remains_retryable() -> None:
    error = CodexHTTPError("rate limited", status_code=429)

    assert retryable_provider_error(error)


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
async def test_transport_retries_use_shared_attempt_limit() -> None:
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
                _retry_notice,
                lambda _error, _retries: None,
                sleep=advance,
            )
        ]

    assert attempts == 5
    assert len(sleeps) == 4


@pytest.mark.asyncio
async def test_transport_defers_failures_after_stream_output_to_loop() -> None:
    request = httpx.Request("POST", "https://example.invalid")
    attempts = 0
    now = 0.0
    notices: list[StreamEvent] = []

    class RetryError(CodexHTTPError):
        def __init__(self, *, stall: bool) -> None:
            super().__init__("request failed", retry_after=0.1)
            self.is_stall = stall
            if not stall:
                self.__cause__ = httpx.ReadError("reset", request=request)

    async def fail() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        raise RetryError(stall=attempts <= 2)

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    with pytest.raises(CodexHTTPError, match="request failed"):
        async for event in retry_provider_completion(
            fail,
            _unused_retry,
            _unused_refresh,
            lambda _error: False,
            lambda error: error,
            _retry_notice,
            lambda _error, _retries: None,
            is_stall=lambda error: getattr(error, "is_stall", False),
            stall_notice=lambda number, delay, error: _retry_notice(
                number, delay, error
            ),
            max_stall_retries=2,
            sleep=advance,
        ):
            if event.type is StreamEventType.RETRY:
                notices.append(event)

    assert attempts == 3
    assert len(notices) == 2
    assert now == 0.2


@pytest.mark.asyncio
async def test_stall_retries_ignore_shared_retry_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    now = 0.0

    monkeypatch.setattr(retry_policy_module.time, "monotonic", lambda: now)

    class StallError(CodexHTTPError):
        is_stall = True

    async def fail() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        raise StallError("stalled", retry_after=40.0)
        yield

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    with pytest.raises(StallError, match="stalled"):
        async for _event in retry_provider_completion(
            fail,
            _unused_retry,
            _unused_refresh,
            lambda _error: False,
            lambda error: error,
            _retry_notice,
            lambda _error, _retries: None,
            is_stall=lambda error: getattr(error, "is_stall", False),
            stall_notice=lambda number, delay, error: _retry_notice(
                number, delay, error
            ),
            max_stall_retries=4,
            sleep=advance,
        ):
            pass

    assert attempts == 5
    assert now == 120.0


@pytest.mark.asyncio
async def test_header_stall_is_retried_within_turn_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    now = 0.0
    monkeypatch.setattr(retry_policy_module.time, "monotonic", lambda: now)

    class StallError(CodexHTTPError):
        is_stall = True

    async def first() -> AsyncIterator[StreamEvent]:
        nonlocal attempts, now
        attempts += 1
        if attempts == 1:
            now += 90.0
            raise StallError("headers stalled")
        yield StreamEvent(StreamEventType.MESSAGE_END)

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    events = [event async for event in retry_provider_completion(
        first, _unused_retry, _unused_refresh, lambda _error: False,
        lambda error: error, _retry_notice, lambda _error, _retries: None,
        is_stall=lambda error: getattr(error, "is_stall", False),
        stall_notice=lambda number, delay, error: _retry_notice(number, delay, error),
        sleep=advance,
    )]
    assert attempts == 2
    assert sum(event.type is StreamEventType.RETRY for event in events) == 1


@pytest.mark.asyncio
async def test_stream_stall_before_content_is_retried() -> None:
    attempts = 0

    class StallError(CodexHTTPError):
        is_stall = True

    async def stream() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        if attempts == 1:
            raise StallError("stream stalled")
        yield StreamEvent(StreamEventType.MESSAGE_END)

    events = [event async for event in retry_provider_completion(
        stream, _unused_retry, _unused_refresh, lambda _error: False,
        lambda error: error, _retry_notice, lambda _error, _retries: None,
        is_stall=lambda error: getattr(error, "is_stall", False),
        stall_notice=lambda number, delay, error: _retry_notice(number, delay, error),
        sleep=lambda _delay: asyncio.sleep(0),
    )]
    assert attempts == 2
    assert sum(event.type is StreamEventType.RETRY for event in events) == 1


@pytest.mark.asyncio
async def test_stall_retries_bounded() -> None:
    attempts = 0

    class StallError(CodexHTTPError):
        is_stall = True

    async def stream() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        raise StallError("stream stalled")
        yield

    with pytest.raises(StallError, match="stream stalled"):
        async for _event in retry_provider_completion(
            stream, _unused_retry, _unused_refresh, lambda _error: False,
            lambda error: error, _retry_notice, lambda _error, _retries: None,
            is_stall=lambda error: getattr(error, "is_stall", False),
            max_stall_retries=2, sleep=lambda _delay: asyncio.sleep(0),
        ):
            pass
    assert attempts == 3


@pytest.mark.asyncio
async def test_non_stall_retry_window_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    now = 0.0
    monkeypatch.setattr(retry_policy_module.time, "monotonic", lambda: now)

    async def stream() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        raise CodexHTTPError("rate limited", status_code=429, retry_after=40.0)
        yield

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    with pytest.raises(CodexHTTPError, match="rate limited"):
        async for _event in retry_provider_completion(
            stream, _unused_retry, _unused_refresh, lambda _error: False,
            lambda error: error, _retry_notice, lambda _error, _retries: None,
            sleep=advance,
        ):
            pass
    assert attempts == 2
    assert now == 60.0


@pytest.mark.asyncio
async def test_long_stream_then_stall_respects_wall_window_minus_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(retry_policy_module.time, "monotonic", lambda: now)
    budget = ProviderRetryBudget()
    assert budget.start_attempt("transport")

    class StallError(CodexHTTPError):
        is_stall = True
        stall_seconds = 90.0

    # The request made useful progress for 61 seconds, then waited silently for
    # 90 seconds. Only the silent interval is outside the 60-second window.
    now = 151.0
    plan = budget.plan(
        StallError("stalled", retry_after=0.0),
        owner="transport",
        event_data={"retryable": True},
    )

    excluded_stall_seconds = getattr(budget, "excluded_stall_seconds", 0.0)
    assert now - budget.started_at - excluded_stall_seconds == 61.0
    assert plan is None
    assert budget.records[-1] == {"decision": "budget-exhausted"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_class",
    [
        pytest.param(CodexHTTPError, id="codex"),
        pytest.param(AnthropicStreamError, id="anthropic"),
        pytest.param(OllamaError, id="ollama"),
    ],
)
async def test_stall_exhaustion_recorded_in_budget(error_class) -> None:
    budget = ProviderRetryBudget()
    assert budget.start_attempt("transport")

    class StallError(error_class):
        is_stall = True
        stall_seconds = 90.0

    attempts = 0

    async def stream() -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        error = StallError("stalled")
        error.retry_after = 0.0
        error.is_stall = True
        error.stall_seconds = 90.0
        raise error
        yield

    with (
        retry_policy_module.use_retry_budget(budget),
        pytest.raises(StallError, match="stalled"),
    ):
        async for _event in retry_provider_completion(
            stream,
            _unused_retry,
            _unused_refresh,
            lambda _error: False,
            lambda error: error,
            _retry_notice,
            lambda _error, _retries: None,
            is_stall=lambda error: getattr(error, "is_stall", False),
            max_stall_retries=2,
            sleep=lambda _delay: asyncio.sleep(0),
        ):
            pass

    assert attempts == 3
    assert budget.records[-1] == {"decision": "budget-exhausted"}


@pytest.mark.asyncio
async def test_worst_case_turn_wall_time_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    attempts = 0
    monkeypatch.setattr(retry_policy_module.time, "monotonic", lambda: now)

    class StallError(CodexHTTPError):
        is_stall = True
        stall_seconds = 90.0

    async def stream() -> AsyncIterator[StreamEvent]:
        nonlocal attempts, now
        attempts += 1
        now += 90.0
        raise StallError("terminal stall", retry_after=30.0)
        yield

    async def advance(delay: float) -> None:
        nonlocal now
        now += delay

    with pytest.raises(StallError, match="terminal stall"):
        async for _event in retry_provider_completion(
            stream,
            _unused_retry,
            _unused_refresh,
            lambda _error: False,
            lambda error: error,
            _retry_notice,
            lambda _error, _retries: None,
            is_stall=lambda error: getattr(error, "is_stall", False),
            max_stall_retries=2,
            sleep=advance,
        ):
            pass

    assert attempts == 3
    assert now == 3 * 90.0 + 2 * 30.0
    assert now <= 330.0
