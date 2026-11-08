"""Tests for the ZETA-74 mid-stream stall watchdog.

Covers: the shared ``stall_watchdog`` helper; provider-level integration for
Anthropic and Codex (stall detection, bounded retries, keepalives resetting
the timer, partial-content preservation via ZETA-43); AgentLoop reset on the
stall RETRY event; and headless/settings surface areas.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

import zeta.accounts.store as account_store
from tests.support.completion import ManualDeadlineClock, completion_control
from tests.support.credentials import anthropic_credentials, codex_credentials
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.accounts import AccountCredentials
from zeta.config.settings import Settings, load_settings, resolve
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    FAILED_TURN_MARKER,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from zeta.providers.anthropic import (
    AnthropicBackend,
    OAuthTokens,
)
from zeta.providers.anthropic_errors import AnthropicStreamError
from zeta.providers.codex import CodexBackend
from zeta.providers.completion import ProviderStallError, owned_completion
from zeta.providers.ollama import OllamaBackend
from zeta.runtime.headless import drive_turn
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

pytestmark = pytest.mark.usefixtures("close_direct_test_agent_loops")

ANTHROPIC_SSE = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg-1","model":"claude-test",'
    '"role":"assistant","usage":{"input_tokens":1}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,"content_block":'
    '{"type":"text","text":""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,"delta":'
    '{"type":"text_delta","text":"hi"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":1}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)


class _HangingByteStream(httpx.AsyncByteStream):
    """Emit ``preface`` bytes, then hang forever until the stream is closed."""

    def __init__(self, preface: bytes) -> None:
        self._preface = preface
        self._closed = asyncio.Event()
        self.entered = asyncio.Event()

    async def __aiter__(self):
        if self._preface:
            yield self._preface
        self.entered.set()
        await asyncio.wait_for(self._closed.wait(), 10)

    async def aclose(self) -> None:
        self._closed.set()


class _ClockedByteStream(httpx.AsyncByteStream):
    """Emit controlled chunks while advancing a manual deadline clock."""

    def __init__(
        self,
        clock: ManualDeadlineClock,
        chunks: list[tuple[float, bytes]],
    ) -> None:
        self._clock = clock
        self._chunks = chunks
        self._closed = asyncio.Event()

    async def __aiter__(self):
        for elapsed, chunk in self._chunks:
            self._clock.advance(self._clock.now() + elapsed)
            yield chunk
        await asyncio.wait_for(self._closed.wait(), 10)

    async def aclose(self) -> None:
        self._closed.set()


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _collect_after_expirations(
    completion: AsyncIterator[StreamEvent],
    clock: ManualDeadlineClock,
    gates: list[asyncio.Event],
) -> list[StreamEvent]:
    task = asyncio.create_task(_collect(completion))
    try:
        for gate in gates:
            await asyncio.wait_for(gate.wait(), 10)
            await clock.expire_next()
        return await asyncio.wait_for(task, 10)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)


async def _collect(completion: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in completion]


def _anthropic_store(path: Path) -> AccountCredentials:
    store = anthropic_credentials(path)
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    return store


# ---------------------------- transport helpers ---------------------------- #


class _StubError(RuntimeError):
    def __init__(self, message: str, *, is_stall: bool = False) -> None:
        super().__init__(message)
        self.is_stall = is_stall
















def _no_sleep(sleeps: list[float]):
    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    return sleep


async def _refresh_unreachable() -> str:
    raise AssertionError("refresh unreachable")


async def _unreachable_retry(_token: str) -> AsyncIterator[StreamEvent]:
    raise AssertionError("retry unreachable")
    yield


def _notice_unreachable(*_args, **_kwargs) -> StreamEvent:
    raise AssertionError("notice unreachable")


# ------------------------ Anthropic backend integration ------------------- #


async def test_anthropic_backend_stalls_and_retries_mid_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    partial = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"id":"msg-partial",'
        b'"model":"claude-test","role":"assistant","usage":{"input_tokens":1}}}\n\n'
    )

    hanging = _HangingByteStream(partial)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=hanging,
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=ANTHROPIC_SSE,
            request=request,
        )

    client = _mock_client(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=_anthropic_store(tmp_path / "zeta.json"),
        stall_retries=2,
    )
    clock = ManualDeadlineClock()
    backend._completion_clock = clock
    backend._completion_wait = _no_op_sleep
    loop = AgentLoop(
        backend, ConversationStore(tmp_path / "session"), skill_catalog=SkillCatalog.empty()
    )
    events = await _collect_after_expirations(
        loop.run_turn("hello", origin=MessageOrigin.USER), clock, [hanging.entered]
    )
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    assert len(requests) == 2
    assert len(retries) == 1
    assert not any(event.type is StreamEventType.ASSISTANT_RESET for event in events)
    assert events[-1].type is StreamEventType.AGENT_END
    await client.aclose()


async def test_anthropic_retry_suppresses_truncated_salvage_and_keeps_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    partial = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"id":"msg-partial",'
        b'"model":"claude-test","role":"assistant","usage":{"input_tokens":3}}}\n\n'
        b'event: content_block_start\n'
        b'data: {"type":"content_block_start","index":0,"content_block":'
        b'{"type":"text","text":""}}\n\n'
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":'
        b'{"type":"text_delta","text":"failed partial"}}\n\n'
        b'event: message_delta\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":null},'
        b'"usage":{"output_tokens":7}}\n\n'
    )

    class DisconnectingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield partial
            raise httpx.ReadError("reset", request=requests[-1])

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=DisconnectingStream(),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=ANTHROPIC_SSE,
            request=request,
        )

    client = _mock_client(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=_anthropic_store(tmp_path / "zeta.json"),
    )
    backend._completion_wait = _no_op_sleep
    loop = AgentLoop(
        backend, ConversationStore(tmp_path / "session"), skill_catalog=SkillCatalog.empty()
    )
    events = [event async for event in loop.run_turn("hello", origin=MessageOrigin.USER)]

    message_ends = [
        event for event in events if event.type is StreamEventType.MESSAGE_END
    ]
    retry = next(event for event in events if event.type is StreamEventType.RETRY)
    assert len(requests) == 2
    assert len(message_ends) == 2
    assert message_ends[0].data.get("truncated") is True
    assert message_ends[1].data.get("truncated") is not True
    assert retry.data["usage"] == {"input_tokens": 3, "output_tokens": 7}
    await client.aclose()


async def test_anthropic_backend_stall_budget_exhausts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    streams = [_HangingByteStream(b"") for _ in range(3)]

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=streams[len(requests) - 1],
            request=request,
        )

    client = _mock_client(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=_anthropic_store(tmp_path / "zeta.json"),
        stall_retries=2,
    )
    clock = ManualDeadlineClock()
    backend._completion_clock = clock
    backend._completion_wait = _no_op_sleep
    with pytest.raises(ProviderStallError) as excinfo:
        await _collect_after_expirations(
            backend.complete([], [], control=completion_control()),
            clock,
            [stream.entered for stream in streams],
        )

    assert excinfo.value.is_stall is True
    assert len(requests) == 3
    await client.aclose()






async def _no_op_sleep(_delay: float) -> None:
    return None


async def test_anthropic_backend_ignores_stall_after_message_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-completion silence must NOT trigger a stall retry: the caller
    already has a finished message, so the socket hanging open is a clean EOF
    from our perspective, not a stall that costs another billed attempt."""

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_HangingByteStream(ANTHROPIC_SSE.encode()),
            request=request,
        )

    client = _mock_client(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=_anthropic_store(tmp_path / "zeta.json"),
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [], control=completion_control())]
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    assert len(requests) == 1
    assert retries == []
    assert events[-1].type is StreamEventType.MESSAGE_END
    await client.aclose()


# ------------------------ Codex backend integration ----------------------- #


def _codex_access_token() -> str:
    # Reuse the shared codex test JWT builder to satisfy account extraction.
    import base64

    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = {
        "https://api.openai.com/auth": {
            "chatgpt_account_id": "account-test",
        },
        "exp": 4_000_000_000,
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def _codex_store(path: Path) -> AccountCredentials:
    store = codex_credentials(path)
    store.save(OAuthTokens(_codex_access_token(), "refresh-fixture", 4_000_000_000))
    return store


CODEX_COMPLETE_SSE = (
    'event: response.created\n'
    'data: {"type":"response.created","response":{"id":"r-1","status":"in_progress"}}\n\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"r-1","status":"completed"}}\n\n'
)


@pytest.mark.parametrize("provider", ["anthropic", "codex"])
async def test_provider_stalls_waiting_for_response_headers(
    tmp_path: Path, provider: str
) -> None:
    requests: list[httpx.Request] = []

    headers_waiting = asyncio.Event()

    async def no_headers(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers_waiting.set()
        await await_or_fail(
            asyncio.Event().wait(), what="blocked response headers callback"
        )
        raise AssertionError("unreachable")

    client = _mock_client(no_headers)
    if provider == "anthropic":
        backend = AnthropicBackend(
            client=client,
            token_store=_anthropic_store(tmp_path / "claude.json"),
            stall_retries=0,
        )
    else:
        backend = CodexBackend(
            client=client,
            token_store=_codex_store(tmp_path / "codex.json"),
            base_url="https://test.invalid/codex/responses",
            stall_retries=0,
        )
    try:
        clock = ManualDeadlineClock()
        backend._completion_clock = clock
        backend._completion_wait = _no_op_sleep
        control = completion_control()
        with pytest.raises(ProviderStallError, match="transport stalled for 90s") as caught:
            await _collect_after_expirations(
                backend.complete([], [], control=control), clock, [headers_waiting]
            )
        assert caught.value.is_stall is True
        assert len(requests) == 1
    finally:
        await client.aclose()


@pytest.mark.parametrize("store_type", [anthropic_credentials, codex_credentials])
@pytest.mark.parametrize("method_name", ["access_token", "refresh_token"])
async def test_oauth_refresh_network_wait_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_type,
    method_name: str,
) -> None:
    store = store_type(tmp_path / "oauth.json")
    store.save(OAuthTokens("expired-access", "refresh-fixture", 1))
    monkeypatch.setattr(account_store, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.05)

    async def no_refresh_response(_request: httpx.Request) -> httpx.Response:
        await await_or_fail(
            asyncio.Event().wait(), what="blocked OAuth refresh callback"
        )
        raise AssertionError("unreachable")

    async with _mock_client(no_refresh_response) as client:
        with pytest.raises(store.http_error_type, match="OAuth token refresh stalled"):
            await await_or_fail(
                getattr(store, method_name)(client),
                what="OAuth token refresh stall",
            )
        assert json.loads(store.path.read_text())["access_token"] == "expired-access"

        async def recovered_refresh(_token: str, _client: httpx.AsyncClient) -> OAuthTokens:
            return OAuthTokens("fresh-access", "refresh-fixture", 4_000_000_000)

        monkeypatch.setattr(store, "refresh", recovered_refresh)
        assert (
            await await_or_fail(
                store.access_token(client),
                what="recovered OAuth access token",
            )
            == "fresh-access"
        )


async def test_oauth_credential_lock_wait_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _codex_store(tmp_path / "codex.json")
    monkeypatch.setattr(account_store, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.05)
    async with _mock_client(lambda _request: httpx.Response(500)) as client:
        async with store._async_guard(registry=False):
            with pytest.raises(
                account_store.AccountCredentialLockTimeoutError,
                match=(
                    "cannot access codex:default: another process holds the account "
                    "lock; retry the turn"
                ),
            ):
                await await_or_fail(
                    store.access_token(client),
                    what="account credential lock stall",
                )
        assert (
            await await_or_fail(
                store.access_token(client),
                what="access token after credential lock release",
            )
            == _codex_access_token()
        )


async def test_codex_keepalives_do_not_reset_progress_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = ManualDeadlineClock()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ClockedByteStream(
                clock,
                [
                    (60.0, b'event: keepalive\ndata: {"type":"keepalive"}\n\n')
                    for _ in range(15)
                ],
            ),
            request=request,
        )

    client = _mock_client(handler)
    backend = CodexBackend(
        client=client,
        token_store=_codex_store(tmp_path / "codex.json"),
        base_url="https://test.invalid/codex/responses",
        stall_seconds=90.0,
        stall_retries=0,
    )
    backend._completion_clock = clock
    backend._completion_wait = _no_op_sleep
    control = completion_control()
    with pytest.raises(ProviderStallError, match="model progress stalled for 900s") as excinfo:
        await asyncio.wait_for(_collect(backend.complete([], [], control=control)), 10)

    assert excinfo.value.is_stall is True
    await client.aclose()


async def test_codex_backend_stalls_and_retries_mid_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = (
        b'event: response.created\n'
        b'data: {"type":"response.created","response":{"id":"r-partial",'
        b'"status":"in_progress"}}\n\n'
    )
    requests: list[httpx.Request] = []

    hanging = _HangingByteStream(partial)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=hanging,
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=CODEX_COMPLETE_SSE,
            request=request,
        )

    client = _mock_client(handler)
    backend = CodexBackend(
        client=client,
        token_store=_codex_store(tmp_path / "codex.json"),
        base_url="https://test.invalid/codex/responses",
        stall_retries=2,
    )
    clock = ManualDeadlineClock()
    backend._completion_clock = clock
    backend._completion_wait = _no_op_sleep
    loop = AgentLoop(
        backend, ConversationStore(tmp_path / "session"), skill_catalog=SkillCatalog.empty()
    )
    events = await _collect_after_expirations(
        loop.run_turn("hello", origin=MessageOrigin.USER), clock, [hanging.entered]
    )
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    # The third request is the loop's normal empty-turn nudge after retry success.
    assert len(requests) == 3
    assert len(retries) == 1
    assert not any(event.type is StreamEventType.ASSISTANT_RESET for event in events)
    assert events[-1].type is StreamEventType.AGENT_END
    await client.aclose()


async def test_codex_backend_stall_budget_exhausts_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    streams = [_HangingByteStream(b"") for _ in range(2)]

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=streams[len(requests) - 1],
            request=request,
        )

    client = _mock_client(handler)
    backend = CodexBackend(
        client=client,
        token_store=_codex_store(tmp_path / "codex.json"),
        base_url="https://test.invalid/codex/responses",
        stall_retries=1,
    )
    clock = ManualDeadlineClock()
    backend._completion_clock = clock
    backend._completion_wait = _no_op_sleep
    control = completion_control()
    with pytest.raises(ProviderStallError) as excinfo:
        await _collect_after_expirations(
            backend.complete([], [], control=control),
            clock,
            [stream.entered for stream in streams],
        )

    assert excinfo.value.is_stall is True
    assert len(requests) == 2
    await client.aclose()


async def test_codex_backend_ignores_stall_after_message_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex mirror of the Anthropic post-completion guard."""

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_HangingByteStream(CODEX_COMPLETE_SSE.encode()),
            request=request,
        )

    client = _mock_client(handler)
    backend = CodexBackend(
        client=client,
        token_store=_codex_store(tmp_path / "codex.json"),
        base_url="https://test.invalid/codex/responses",
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [], control=completion_control())]
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    assert len(requests) == 1
    assert retries == []
    assert events[-1].type is StreamEventType.MESSAGE_END
    await client.aclose()


# ---------------------------- AgentLoop integration ------------------------ #


class _StallingBackend:
    """Yield a partial turn, stall, then complete cleanly on the loop retry."""

    def __init__(self) -> None:
        self.attempts = 0

    @owned_completion
    def complete(self, messages, tool_schemas):
        return self._complete(messages, tool_schemas)

    async def _complete(self, _messages, _tool_schemas):
        self.attempts += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        if self.attempts == 1:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="pre-stall text ")
            error = AnthropicStreamError(
                "Anthropic stream stalled", is_stall=True
            )
            error.retry_after = 0.0
            raise error
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="fresh text")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("fresh text")]),
        )


async def test_agent_loop_reset_on_stall_retry_drops_pre_stall_partial(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallingBackend(), store, skill_catalog=SkillCatalog.empty())
    events = [event async for event in loop.run_turn("hi", origin=MessageOrigin.USER)]

    turn_end = next(
        event for event in events if event.type is StreamEventType.TURN_END
    )
    assert turn_end.message is not None
    assert [block for block in turn_end.message.content if isinstance(block, TextContent)] == [
        TextContent("fresh text")
    ]

    persisted = [
        message
        for message in store.messages()
        if message.role is MessageRole.ASSISTANT
    ]
    assert len(persisted) == 1
    assert not persisted[0].metadata.get(FAILED_TURN_MARKER)
    assert any(isinstance(block, TextContent) and block.text == "fresh text" for block in persisted[0].content)
    assert not any(
        isinstance(block, TextContent) and "pre-stall" in block.text
        for block in persisted[0].content
    )


class _StallAfterCompletionBackend:
    """Misbehaving provider: emits a full message, THEN a stall RETRY notice.

    Simulates a broken transport where the post-completion silence somehow
    still produces an ``is_stall`` event; the loop must ignore it rather than
    wipe the already-completed message and bill a fresh attempt."""

    @owned_completion
    def complete(self, messages, tool_schemas):
        return self._complete(messages, tool_schemas)

    async def _complete(self, _messages, _tool_schemas):
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="only text")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("only text")]),
        )
        yield StreamEvent(
            StreamEventType.RETRY,
            data={"is_stall": True, "retry": 1, "delay": 0.0},
        )


async def test_agent_loop_stall_after_message_end_keeps_completed_message(
    tmp_path: Path,
) -> None:
    """Layer B guard: an ``is_stall`` RETRY after MESSAGE_END must not wipe
    the completed message; the store keeps the first message, unmarked."""

    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallAfterCompletionBackend(), store, skill_catalog=SkillCatalog.empty())
    events = [event async for event in loop.run_turn("hi", origin=MessageOrigin.USER)]

    turn_end = next(
        event for event in events if event.type is StreamEventType.TURN_END
    )
    assert turn_end.message is not None
    assert [
        block for block in turn_end.message.content if isinstance(block, TextContent)
    ] == [TextContent("only text")]

    persisted = [
        message
        for message in store.messages()
        if message.role is MessageRole.ASSISTANT
    ]
    assert len(persisted) == 1
    assert not persisted[0].metadata.get(FAILED_TURN_MARKER)
    assert any(
        isinstance(block, TextContent) and block.text == "only text"
        for block in persisted[0].content
    )


class _ExhaustingBackend:
    """Emit a stall RETRY, then raise a stall error to model exhaustion."""

    @owned_completion
    def complete(self, messages, tool_schemas):
        return self._complete(messages, tool_schemas)

    async def _complete(self, _messages, _tool_schemas):
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="partial")
        error = AnthropicStreamError(
            "Anthropic stream stalled for 90s", is_stall=True
        )
        error.retry_after = 0.0
        raise error


async def test_agent_loop_stall_exhaustion_persists_partial_as_failed_turn(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_ExhaustingBackend(), store, skill_catalog=SkillCatalog.empty())
    events = [event async for event in loop.run_turn("hi", origin=MessageOrigin.USER)]

    assert any(event.type is StreamEventType.ERROR for event in events)
    persisted = [
        message
        for message in store.messages()
        if message.role is MessageRole.ASSISTANT
    ]
    assert len(persisted) == 1
    assert persisted[0].metadata.get(FAILED_TURN_MARKER) is True


# ------------------------------ Headless emission --------------------------- #


async def test_headless_text_mode_writes_retry_to_stderr(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallingBackend(), store, skill_catalog=SkillCatalog.empty())
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = await drive_turn(loop, "hi", origin=MessageOrigin.USER, format="text", stdout=stdout, stderr=stderr)

    assert code == 0
    assert stdout.getvalue() == (
        "pre-stall text \n[assistant response restarted]\nfresh text\n"
    )
    assert "provider stalled" in stderr.getvalue()


async def test_headless_json_mode_emits_stall_retry_event(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallingBackend(), store, skill_catalog=SkillCatalog.empty())
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = await drive_turn(loop, "hi", origin=MessageOrigin.USER, format="json", stdout=stdout, stderr=stderr)

    assert code == 0
    events = [json.loads(line) for line in stdout.getvalue().splitlines() if line]
    retry = next(event for event in events if event["type"] == "retry")
    assert retry["is_stall"] is True
    assert "provider stalled" in retry["text"]


# ------------------------------ Settings surface ---------------------------- #


def test_settings_load_reads_stream_stall_keys(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(
        "stream_stall_seconds = 45\nstream_stall_retries = 4\n"
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.settings.stream_stall_seconds == 45
    assert loaded.settings.stream_stall_retries == 4


def test_settings_rejects_bogus_stream_stall_seconds(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(
        "stream_stall_seconds = 0\nstream_stall_retries = -1\n"
    )
    loaded = load_settings(home=home, project_dir=None)
    assert loaded.settings.stream_stall_seconds is None
    assert loaded.settings.stream_stall_retries is None
    joined = " ".join(loaded.notices)
    assert "stream_stall_seconds" in joined
    assert "stream_stall_retries" in joined


def test_project_settings_may_not_grant_stream_stall_overrides(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "settings.toml").write_text("stream_stall_seconds = 45\n")
    (project / "settings.toml").write_text("stream_stall_seconds = 9999\n")
    loaded = load_settings(home=home, project_dir=project)
    assert loaded.settings.stream_stall_seconds == 45
    joined = " ".join(loaded.warnings)
    assert "stream_stall_seconds" in joined


def test_resolve_carries_stream_stall_settings_into_resolved_config() -> None:
    settings = Settings(stream_stall_seconds=45, stream_stall_retries=4)
    resolved = resolve(
        settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    default_provider="codex",
    )
    assert resolved.stream_stall_seconds == 45
    assert resolved.stream_stall_retries == 4


# ------------------------------ Defaults sanity ---------------------------- #




# ------------------------------ FakeBackend guard --------------------------- #


async def test_fake_backend_completion_does_not_trigger_watchdog(tmp_path: Path) -> None:
    """Sanity: existing fake-backend flows are unaffected by the watchdog."""

    backend = FakeBackend([ScriptedTurn(content=[TextContent("hello")])])
    loop = AgentLoop(backend, ConversationStore(tmp_path), skill_catalog=SkillCatalog.empty())
    events = [event async for event in loop.run_turn("hi", origin=MessageOrigin.USER)]
    assert any(event.type is StreamEventType.TURN_END for event in events)


# ------------------------ stall settings reach the owner ------------------- #


def _stalling_backend(provider: str, client: httpx.AsyncClient, tmp_path: Path, **stall):
    if provider == "anthropic":
        return AnthropicBackend(
            client=client, token_store=_anthropic_store(tmp_path / "claude.json"), **stall
        )
    if provider == "codex":
        return CodexBackend(
            client=client,
            token_store=_codex_store(tmp_path / "codex.json"),
            base_url="https://test.invalid/codex/responses",
            **stall,
        )
    return OllamaBackend(client=client, base_url="http://ollama.invalid", **stall)


@pytest.mark.parametrize("provider", ["anthropic", "codex", "ollama"])
@pytest.mark.parametrize(
    ("stall", "idle_seconds", "attempts"),
    [
        pytest.param({}, 90.0, 3, id="production-defaults"),
        pytest.param({"stall_seconds": 30}, 30.0, 3, id="stall_seconds=30"),
        # above the 15-minute model-progress limit, the transport limit still wins
        pytest.param({"stall_seconds": 1200}, 1200.0, 3, id="stall_seconds=1200"),
        pytest.param({"stall_retries": 0}, 90.0, 1, id="stall_retries=0"),
        pytest.param({"stall_retries": 1}, 90.0, 2, id="stall_retries=1"),
    ],
)
async def test_stream_stall_settings_set_owner_deadline_and_retry_count(
    tmp_path: Path, provider: str, stall: dict[str, int], idle_seconds: float, attempts: int
) -> None:
    """stream_stall_seconds / stream_stall_retries, as the backend receives them,
    set the completion owner's transport deadline and stall retry count."""

    streams = [_HangingByteStream(b"") for _ in range(4)]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=streams[sum(s.entered.is_set() for s in streams)])

    client = _mock_client(handler)
    backend = _stalling_backend(provider, client, tmp_path, **stall)
    clock = ManualDeadlineClock()
    backend._completion_clock = clock
    backend._completion_wait = _no_op_sleep
    task = asyncio.create_task(
        _collect(backend.complete([], [], control=completion_control()))
    )
    stalls: list[float] = []
    try:
        for stream in streams:
            entered = asyncio.ensure_future(asyncio.wait_for(stream.entered.wait(), 10))
            try:
                done, _ = await await_or_fail(
                    asyncio.wait({task, entered}, return_when=asyncio.FIRST_COMPLETED),
                    what="next attempt or completion end",
                )
            finally:
                entered.cancel()
            if task in done:
                break
            opened_at = clock.now()
            await clock.expire_next()
            stalls.append(clock.now() - opened_at)
        with pytest.raises(ProviderStallError) as caught:
            await await_or_fail(task, what="stalled completion")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
        await client.aclose()

    assert stalls == [idle_seconds] * attempts
    assert str(caught.value) == f"provider transport stalled for {idle_seconds:.0f}s"

