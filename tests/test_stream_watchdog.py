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

import zeta.providers.anthropic as anthropic_module
import zeta.providers.codex as codex_module
import zeta.providers.transport as transport_module
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.headless import drive_turn
from zeta.loop import AgentLoop
from zeta.providers.anthropic import (
    AnthropicBackend,
    AnthropicCredentialStore,
    OAuthTokens,
)
from zeta.providers.anthropic_errors import AnthropicStreamError
from zeta.providers.codex import CodexBackend, CodexCredentialStore
from zeta.providers.codex_errors import CodexStreamError
from zeta.providers.transport import (
    DEFAULT_STREAM_STALL_RETRIES,
    DEFAULT_STREAM_STALL_SECONDS,
    retry_provider_completion,
    stall_retry_kwargs,
    stall_retry_notice,
    stall_watchdog,
)
from zeta.settings import Settings, load_settings, resolve
from zeta.types import (
    FAILED_TURN_MARKER,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
)

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

    async def __aiter__(self):
        if self._preface:
            yield self._preface
        await self._closed.wait()

    async def aclose(self) -> None:
        self._closed.set()


class _KeepaliveStream(httpx.AsyncByteStream):
    """Emit ``preface``, drip a keepalive line, then emit the tail."""

    def __init__(
        self,
        preface: bytes,
        keepalive: bytes,
        keepalive_after: float,
        tail: bytes,
        tail_after: float,
    ) -> None:
        self._preface = preface
        self._keepalive = keepalive
        self._keepalive_after = keepalive_after
        self._tail = tail
        self._tail_after = tail_after

    async def __aiter__(self):
        if self._preface:
            yield self._preface
        await asyncio.sleep(self._keepalive_after)
        yield self._keepalive
        await asyncio.sleep(self._tail_after)
        yield self._tail


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _anthropic_store(path: Path) -> AnthropicCredentialStore:
    store = AnthropicCredentialStore(path)
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    return store


# ---------------------------- transport helpers ---------------------------- #


class _StubError(RuntimeError):
    def __init__(self, message: str, *, is_stall: bool = False) -> None:
        super().__init__(message)
        self.is_stall = is_stall


async def _drip(items, delays):
    for item, delay in zip(items, delays, strict=True):
        if delay:
            await asyncio.sleep(delay)
        yield item


async def test_stall_watchdog_raises_when_source_is_silent() -> None:
    async def source():
        yield "first"
        await asyncio.Event().wait()

    started = source()
    watched = stall_watchdog(
        started,
        seconds=0.05,
        on_stall=lambda elapsed: _StubError("stalled", is_stall=True),
    )
    aiter = watched.__aiter__()
    assert await aiter.__anext__() == "first"
    with pytest.raises(_StubError) as excinfo:
        await aiter.__anext__()
    assert excinfo.value.is_stall is True
    await watched.aclose()


async def test_stall_watchdog_resets_timer_on_each_item() -> None:
    async def source():
        for value in ("a", "b", "c"):
            await asyncio.sleep(0.03)
            yield value

    watched = stall_watchdog(
        source(),
        seconds=0.1,
        on_stall=lambda elapsed: _StubError("stalled", is_stall=True),
    )
    seen = [item async for item in watched]
    assert seen == ["a", "b", "c"]


async def test_stall_watchdog_disabled_when_seconds_nonpositive() -> None:
    source = _drip(["only"], [0])
    watched = stall_watchdog(
        source,
        seconds=0.0,
        on_stall=lambda elapsed: _StubError("stalled", is_stall=True),
    )
    seen = [item async for item in watched]
    assert seen == ["only"]


def test_stall_retry_notice_has_reset_flag_and_max_retries_text() -> None:
    event = stall_retry_notice(2, 4.0, 3)
    assert event.type is StreamEventType.RETRY
    assert event.data["is_stall"] is True
    assert "reset" not in event.data
    assert event.data["retry"] == 2
    assert event.data["text"] == "provider stalled, retrying (2/3) in 4s"


def test_stall_retry_kwargs_wires_predicate_and_notice() -> None:
    kwargs = stall_retry_kwargs(2)
    assert kwargs["max_stall_retries"] == 2
    predicate = kwargs["is_stall"]
    assert predicate(_StubError("x", is_stall=True)) is True
    assert predicate(_StubError("x", is_stall=False)) is False
    notice = kwargs["stall_notice"](1, 4.0, _StubError("y", is_stall=True))
    assert notice.type is StreamEventType.RETRY
    assert notice.data["is_stall"] is True


async def test_retry_provider_completion_retries_stall_after_stream_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_sleep(sleeps))
    attempts = 0

    async def first():
        nonlocal attempts
        attempts += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        if attempts == 1:
            raise _StubError("stalled", is_stall=True)
        yield StreamEvent(StreamEventType.MESSAGE_END)

    async def _unused(_token: str) -> AsyncIterator[StreamEvent]:
        raise AssertionError("auth retry unreachable")
        yield

    events = [
        event
        async for event in retry_provider_completion(
            first,
            _unused,
            _refresh_unreachable,
            lambda _e: False,
            lambda _e: RuntimeError("auth exhausted"),
            lambda event: event.type is StreamEventType.MESSAGE_START,
            lambda _e: False,
            _notice_unreachable,
            lambda _e, _n: None,
            **stall_retry_kwargs(2),
        )
    ]

    assert attempts == 2
    assert [event.type for event in events] == [
        StreamEventType.MESSAGE_START,
        StreamEventType.RETRY,
        StreamEventType.MESSAGE_START,
        StreamEventType.MESSAGE_END,
    ]
    assert events[1].data["is_stall"] is True
    assert len(sleeps) == 1


async def test_retry_provider_completion_stops_after_stall_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_sleep(sleeps))
    attempts = 0
    exhausted: list[tuple[RuntimeError, int]] = []

    async def first():
        nonlocal attempts
        attempts += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        raise _StubError("stalled", is_stall=True)

    with pytest.raises(_StubError):
        [
            event
            async for event in retry_provider_completion(
                first,
                _unreachable_retry,
                _refresh_unreachable,
                lambda _e: False,
                lambda _e: RuntimeError("auth exhausted"),
                lambda event: event.type is StreamEventType.MESSAGE_START,
                lambda _e: False,
                _notice_unreachable,
                lambda error, retries: exhausted.append((error, retries)),
                **stall_retry_kwargs(2),
            )
        ]

    assert attempts == 3
    assert len(sleeps) == 2
    assert exhausted and exhausted[0][1] == 2


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
    monkeypatch.setattr(anthropic_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
    requests: list[httpx.Request] = []
    partial = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"id":"msg-partial",'
        b'"model":"claude-test","role":"assistant","usage":{"input_tokens":1}}}\n\n'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_HangingByteStream(partial),
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
        stall_seconds=0.05,
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [])]
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    assert len(requests) == 2
    assert len(retries) == 1
    assert retries[0].data["is_stall"] is True
    assert events[-1].type is StreamEventType.MESSAGE_END
    await client.aclose()


async def test_anthropic_backend_stall_budget_exhausts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(anthropic_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_HangingByteStream(b""),
            request=request,
        )

    client = _mock_client(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=_anthropic_store(tmp_path / "zeta.json"),
        stall_seconds=0.05,
        stall_retries=2,
    )
    with pytest.raises(AnthropicStreamError) as excinfo:
        [event async for event in backend.complete([], [])]

    assert excinfo.value.is_stall is True
    assert len(requests) == 3
    await client.aclose()


async def test_anthropic_backend_keepalive_lines_reset_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(anthropic_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
    preface = ANTHROPIC_SSE.encode()[: ANTHROPIC_SSE.encode().index(b"content_block_delta")]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_KeepaliveStream(
                preface,
                b": ping\n\n",
                keepalive_after=0.05,
                tail=ANTHROPIC_SSE.encode()[len(preface):],
                tail_after=0.05,
            ),
            request=request,
        )

    client = _mock_client(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=_anthropic_store(tmp_path / "zeta.json"),
        stall_seconds=0.15,
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [])]

    assert [event for event in events if event.type is StreamEventType.RETRY] == []
    assert events[-1].type is StreamEventType.MESSAGE_END
    await client.aclose()


async def _no_op_sleep(_delay: float) -> None:
    return None


async def test_anthropic_backend_ignores_stall_after_message_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-completion silence must NOT trigger a stall retry: the caller
    already has a finished message, so the socket hanging open is a clean EOF
    from our perspective, not a stall that costs another billed attempt."""

    monkeypatch.setattr(anthropic_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
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
        stall_seconds=0.05,
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [])]
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


def _codex_store(path: Path) -> CodexCredentialStore:
    store = CodexCredentialStore(path)
    store.save(OAuthTokens(_codex_access_token(), "refresh-fixture", 4_000_000_000))
    return store


CODEX_COMPLETE_SSE = (
    'event: response.created\n'
    'data: {"type":"response.created","response":{"id":"r-1","status":"in_progress"}}\n\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"r-1","status":"completed"}}\n\n'
)


async def test_codex_backend_stalls_and_retries_mid_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
    partial = (
        b'event: response.created\n'
        b'data: {"type":"response.created","response":{"id":"r-partial",'
        b'"status":"in_progress"}}\n\n'
    )
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_HangingByteStream(partial),
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
        stall_seconds=0.05,
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [])]
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    assert len(requests) == 2
    assert len(retries) == 1
    assert retries[0].data["is_stall"] is True
    assert events[-1].type is StreamEventType.MESSAGE_END
    await client.aclose()


async def test_codex_backend_stall_budget_exhausts_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(codex_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_HangingByteStream(b""),
            request=request,
        )

    client = _mock_client(handler)
    backend = CodexBackend(
        client=client,
        token_store=_codex_store(tmp_path / "codex.json"),
        base_url="https://test.invalid/codex/responses",
        stall_seconds=0.05,
        stall_retries=1,
    )
    with pytest.raises(CodexStreamError) as excinfo:
        [event async for event in backend.complete([], [])]

    assert excinfo.value.is_stall is True
    assert len(requests) == 2
    await client.aclose()


async def test_codex_backend_ignores_stall_after_message_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex mirror of the Anthropic post-completion guard."""

    monkeypatch.setattr(codex_module.asyncio, "sleep", _no_op_sleep)
    monkeypatch.setattr(transport_module.asyncio, "sleep", _no_op_sleep)
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
        stall_seconds=0.05,
        stall_retries=2,
    )
    events = [event async for event in backend.complete([], [])]
    retries = [event for event in events if event.type is StreamEventType.RETRY]

    assert len(requests) == 1
    assert retries == []
    assert events[-1].type is StreamEventType.MESSAGE_END
    await client.aclose()


# ---------------------------- AgentLoop integration ------------------------ #


class _StallingBackend:
    """Yield a partial turn, emit a stall RETRY event, then complete cleanly."""

    def __init__(self) -> None:
        self.attempts = 0

    def complete(self, messages, tool_schemas):
        return self._complete(messages, tool_schemas)

    async def _complete(self, _messages, _tool_schemas):
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="pre-stall text ")
        yield stall_retry_notice(1, 0.0, 2)
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="fresh text")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("fresh text")]),
        )


async def test_agent_loop_reset_on_stall_retry_drops_pre_stall_partial(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallingBackend(), store)
    events = [event async for event in loop.run_turn("hi")]

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

    def complete(self, messages, tool_schemas):
        return self._complete(messages, tool_schemas)

    async def _complete(self, _messages, _tool_schemas):
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="only text")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("only text")]),
        )
        yield stall_retry_notice(1, 0.0, 2)


async def test_agent_loop_stall_after_message_end_keeps_completed_message(
    tmp_path: Path,
) -> None:
    """Layer B guard: an ``is_stall`` RETRY after MESSAGE_END must not wipe
    the completed message; the store keeps the first message, unmarked."""

    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallAfterCompletionBackend(), store)
    events = [event async for event in loop.run_turn("hi")]

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

    def complete(self, messages, tool_schemas):
        return self._complete(messages, tool_schemas)

    async def _complete(self, _messages, _tool_schemas):
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="partial")
        yield stall_retry_notice(1, 0.0, 1)
        raise AnthropicStreamError(
            "Anthropic stream stalled for 90s", is_stall=True
        )


async def test_agent_loop_stall_exhaustion_persists_partial_as_failed_turn(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_ExhaustingBackend(), store)
    events = [event async for event in loop.run_turn("hi")]

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
    loop = AgentLoop(_StallingBackend(), store)
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = await drive_turn(loop, "hi", format="text", stdout=stdout, stderr=stderr)

    assert code == 0
    assert stdout.getvalue().rstrip() == "fresh text"
    assert "provider stalled" in stderr.getvalue()


async def test_headless_json_mode_emits_stall_retry_event(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    loop = AgentLoop(_StallingBackend(), store)
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = await drive_turn(loop, "hi", format="json", stdout=stdout, stderr=stderr)

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
    )
    assert resolved.stream_stall_seconds == 45
    assert resolved.stream_stall_retries == 4


# ------------------------------ Defaults sanity ---------------------------- #


def test_provider_backends_default_to_shared_stall_constants(tmp_path: Path) -> None:
    codex_backend = CodexBackend(
        token_store=_codex_store(tmp_path / "codex.json"),
        base_url="https://test.invalid/codex/responses",
    )
    anthropic_backend = AnthropicBackend(
        token_store=_anthropic_store(tmp_path / "zeta.json"),
    )
    assert codex_backend.stall_seconds == DEFAULT_STREAM_STALL_SECONDS
    assert codex_backend.stall_retries == DEFAULT_STREAM_STALL_RETRIES
    assert anthropic_backend.stall_seconds == DEFAULT_STREAM_STALL_SECONDS
    assert anthropic_backend.stall_retries == DEFAULT_STREAM_STALL_RETRIES


# ------------------------------ FakeBackend guard --------------------------- #


async def test_fake_backend_completion_does_not_trigger_watchdog(tmp_path: Path) -> None:
    """Sanity: existing fake-backend flows are unaffected by the watchdog."""

    backend = FakeBackend([ScriptedTurn(content=[TextContent("hello")])])
    loop = AgentLoop(backend, ConversationStore(tmp_path))
    events = [event async for event in loop.run_turn("hi")]
    assert any(event.type is StreamEventType.TURN_END for event in events)
