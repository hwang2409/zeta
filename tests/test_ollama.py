from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.factory import build_backend
from zeta.providers.ollama import DEFAULT_OLLAMA_MODEL, OllamaBackend, OllamaError


class _DelayedLinesResponse:
    status_code = 200

    def __init__(self, rows: list[tuple[float, str]]) -> None:
        self.rows = rows

    async def aiter_lines(self):
        for delay, line in self.rows:
            await asyncio.sleep(delay)
            yield line


class _StreamContext:
    def __init__(self, response, header_delay: float = 0) -> None:
        self.response = response
        self.header_delay = header_delay

    async def __aenter__(self):
        await asyncio.sleep(self.header_delay)
        return self.response

    async def __aexit__(self, *args):
        return None


class _ScriptedClient:
    def __init__(self, scripts: list[tuple[float, list[tuple[float, str]]]]) -> None:
        self.scripts = scripts
        self.calls = 0

    def stream(self, *args, **kwargs):
        header_delay, rows = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return _StreamContext(_DelayedLinesResponse(rows), header_delay)


from zeta.server.runtime import ServerRuntime


def test_ollama_endpoint_environment_overrides_home_settings(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "settings.toml").write_text(
        'ollama_base_url = "http://settings.example"\n', encoding="utf-8"
    )
    monkeypatch.setenv("ZETA_OLLAMA_BASE_URL", "http://environment.example/")
    backend, model = build_backend("ollama", None, home=tmp_path)
    assert model == DEFAULT_OLLAMA_MODEL
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://environment.example"


@pytest.mark.asyncio
async def test_ollama_chat_streams_text_and_tool_call() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        rows = [
            {"message": {"role": "assistant", "content": "hi"}},
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": {"command": "pwd"}}}
                    ],
                },
                "done": True,
            },
        ]
        return httpx.Response(
            200, content="\n".join(json.dumps(row) for row in rows).encode()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = OllamaBackend(client=client)
        events = [
            event
            async for event in backend.complete(
                [Message(MessageRole.USER, [TextContent("hello")])],
                [{"name": "bash", "parameters": {"type": "object"}}],
            )
        ]
    assert backend.model == DEFAULT_OLLAMA_MODEL
    assert [event.type for event in events] == [
        StreamEventType.MESSAGE_START,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_END,
    ]
    assert events[-1].message is not None
    assert events[-1].message.content[-1].tool_call.name == "bash"  # type: ignore[union-attr]
    assert seen["tools"][0]["function"]["name"] == "bash"


@pytest.mark.asyncio
async def test_ollama_rejects_malformed_tool_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        row = {
            "message": {
                "tool_calls": [{"function": {"name": "bash", "arguments": "{bad"}}]
            }
        }
        return httpx.Response(200, content=(json.dumps(row) + "\n").encode())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OllamaError, match="arguments are malformed JSON"):
            [event async for event in OllamaBackend(client=client).complete([], [])]


@pytest.mark.asyncio
async def test_ollama_accepts_assistant_tool_use_on_follow_up() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=(
                json.dumps({"message": {"content": "done"}, "done": True}) + "\n"
            ).encode(),
        )

    assistant = Message(
        MessageRole.ASSISTANT,
        [
            TextContent("calling"),
            __import__(
                "zeta.protocol.types", fromlist=["ToolUseContent"]
            ).ToolUseContent(
                __import__("zeta.protocol.types", fromlist=["ToolCall"]).ToolCall(
                    "ollama-0", "bash", {"command": "pwd"}
                )
            ),
        ],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = [
            event
            async for event in OllamaBackend(client=client).complete([assistant], [])
        ]
    assert events[-1].message is not None
    assert seen[0]["messages"][0]["tool_calls"][0]["function"]["name"] == "bash"


def test_ollama_build_uses_resolved_transport_settings(tmp_path) -> None:
    backend, _ = build_backend(
        "ollama",
        None,
        home=tmp_path,
        stall_seconds=7,
        stall_retries=4,
        ollama_base_url="http://resolved.example",
    )
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://resolved.example"
    assert backend.stall_seconds == 7
    assert backend.stall_retries == 4


@pytest.mark.asyncio
async def test_ollama_rejects_premature_eof_and_top_level_error() -> None:
    for row, match in [
        ({"message": {"content": "partial"}}, "ended before"),
        ({"error": "bad model"}, "error"),
    ]:

        def handler(request: httpx.Request, row=row) -> httpx.Response:
            return httpx.Response(200, content=(json.dumps(row) + "\n").encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OllamaError, match=match):
                [event async for event in OllamaBackend(client=client).complete([], [])]


@pytest.mark.asyncio
async def test_ollama_preserves_multiple_frame_calls_and_usage() -> None:
    rows = [
        {"message": {"tool_calls": [{"function": {"name": "one", "arguments": {}}}]}},
        {
            "message": {"tool_calls": [{"function": {"name": "two", "arguments": {}}}]},
            "done": True,
            "done_reason": "tool_calls",
            "prompt_eval_count": 3,
            "eval_count": 4,
        },
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=("\n".join(json.dumps(row) for row in rows) + "\n").encode(),
            )
        )
    ) as client:
        events = [
            event async for event in OllamaBackend(client=client).complete([], [])
        ]
    calls = [event.tool_call for event in events if event.tool_call is not None]
    assert len({call.id for call in calls}) == 2
    assert all(call.id.startswith("ollama-") for call in calls)
    assert events[-1].data["stop_reason"] == "tool_use"
    assert events[-1].data["usage"]["input_tokens"] == 3


@pytest.mark.asyncio
async def test_ollama_retries_connect_error_and_closes_owned_client(monkeypatch) -> None:
    class OwnedClient:
        def __init__(self, **kwargs) -> None:
            self.closed = False

        def stream(self, *args, **kwargs):
            raise httpx.ConnectError("offline", request=httpx.Request("POST", "http://ollama"))

        async def aclose(self) -> None:
            self.closed = True
            clients.append(self)

    clients: list[OwnedClient] = []
    monkeypatch.setattr("zeta.providers.ollama.httpx.AsyncClient", OwnedClient)
    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    with pytest.raises(OllamaError):
        [event async for event in OllamaBackend(stall_retries=1).complete([], [])]
    assert len(clients) == 4  # three ordinary transport retries plus the first attempt
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("done_reason", "expected"),
    [("stop", "end_turn"), ("length", "max_tokens")],
)
async def test_ollama_01211_done_reasons_are_normalized(done_reason, expected) -> None:
    row = {
        "message": {"content": "answer"},
        "done": True,
        "done_reason": done_reason,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=(json.dumps(row) + "\n").encode()
            )
        )
    ) as client:
        events = [
            event async for event in OllamaBackend(client=client).complete([], [])
        ]
    assert events[-1].data["stop_reason"] == expected


def test_server_provider_switch_preserves_home_ollama_url(tmp_path, monkeypatch) -> None:
    (tmp_path / "settings.toml").write_text(
        'provider = "fake"\nollama_base_url = "http://home.example"\n',
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def build(provider, model, home, **kwargs):
        seen.update(kwargs)
        return object(), model or "model"

    monkeypatch.setattr("zeta.server.runtime.default_backend", build)
    runtime = ServerRuntime(tmp_path, cwd=tmp_path)
    runtime.backend_for_model("ollama", "qwen3:4b")
    assert seen["ollama_base_url"] == "http://home.example"


def test_ollama_tool_result_payload_has_tool_name_and_matches_history() -> None:
    call = ToolCall("durable-id", "bash", {"command": "pwd"})
    messages = [
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        Message(
            MessageRole.TOOL_RESULT, [], tool_result=ToolResult("durable-id", "ok")
        ),
    ]
    from zeta.providers.ollama import _messages

    assert _messages(messages)[1] == {
        "role": "tool",
        "tool_name": "bash",
        "content": "ok",
    }


def test_ollama_rejects_orphan_tool_result() -> None:
    from zeta.providers.ollama import _messages

    with pytest.raises(OllamaError, match="does not match"):
        _messages(
            [Message(MessageRole.TOOL_RESULT, [], tool_result=ToolResult("nope", "x"))]
        )


@pytest.mark.asyncio
async def test_ollama_retries_stall_after_headers_before_first_frame(monkeypatch) -> None:
    client = _ScriptedClient([
        (0, [(0.05, '{"message":{"content":"late"},"done":true}')]),
        (0, [(0, '{"message":{"content":"ok"},"done":true}')]),
    ])
    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    events = [
        event
        async for event in OllamaBackend(
            client=client, stall_seconds=0.01, stall_retries=1
        ).complete([], [])
    ]
    assert client.calls == 2
    assert any(event.type is StreamEventType.RETRY for event in events)
    assert events[-1].message.content[0].text == "ok"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ollama_retries_stall_between_frames(monkeypatch) -> None:
    client = _ScriptedClient([
        (0, [
            (0, '{"message":{"content":"first"}}'),
            (0.05, '{"message":{"content":"bad"},"done":true}'),
        ]),
        (0, [(0, '{"message":{"content":"ok"},"done":true}')]),
    ])
    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    events = [
        event
        async for event in OllamaBackend(
            client=client, stall_seconds=0.01, stall_retries=1
        ).complete([], [])
    ]
    assert client.calls == 2
    assert any(event.type is StreamEventType.RETRY for event in events)
    assert events[-1].message.content[0].text == "ok"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ollama_stall_retry_count_and_notice(monkeypatch) -> None:
    client = _ScriptedClient([
        (0, [(0.02, '{"message":{"content":"late"},"done":true}')]),
    ])
    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    with pytest.raises(OllamaError, match="stalled"):
        [
            event
            async for event in OllamaBackend(
                client=client, stall_seconds=0.01, stall_retries=2
            ).complete([], [])
        ]
    assert client.calls == 3


@pytest.mark.asyncio
async def test_ollama_nonpositive_stall_seconds_disables_watchdog() -> None:
    client = _ScriptedClient([
        (0, [(0.02, '{"message":{"content":"ok"},"done":true}')]),
    ])
    events = [event async for event in OllamaBackend(client=client, stall_seconds=0).complete([], [])]
    assert events[-1].message.content[0].text == "ok"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ollama_does_not_transport_retry_after_meaningful_output() -> None:
    class Client(_ScriptedClient):
        def stream(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _StreamContext(_DelayedLinesResponse([
                    (0, '{"message":{"content":"escaped"}}'),
                ]))
            raise AssertionError("ordinary retry after output")

    client = Client([])
    with pytest.raises(OllamaError, match="ended before"):
        [
            event
            async for event in OllamaBackend(
                client=client, stall_seconds=0.01, stall_retries=1
            ).complete([], [])
        ]
    assert client.calls == 1


@pytest.mark.asyncio
async def test_ollama_retries_only_before_events_are_emitted(monkeypatch) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, content=b'{"message":{"content":"partial"}}\n')
        return httpx.Response(
            200, content=b'{"message":{"content":"ok"},"done":true}\n'
        )

    monkeypatch.setattr(
        "zeta.providers.transport.retry_wait_seconds", lambda error, retry: 0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OllamaError, match="ended before"):
            [
                event
                async for event in OllamaBackend(
                    client=client, stall_retries=2
                ).complete([], [])
            ]
    assert attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("close_mode", ["aclose", "cancel"])
async def test_ollama_closes_stream_and_owned_client_on_early_close(close_mode) -> None:
    class Response:
        status_code = 200

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.exited = False

        async def aiter_lines(self):
            yield json.dumps({"message": {"content": "hello"}})
            self.started.set()
            await self.release.wait()

    class Context:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            self.response.exited = True

    class OwnedClient:
        def __init__(self, **kwargs):
            self.response = Response()
            self.closed = False
            clients.append(self)

        def stream(self, *args, **kwargs):
            return Context(self.response)

        async def aclose(self):
            self.closed = True

    clients = []
    from zeta.providers import ollama

    original = ollama.httpx.AsyncClient
    ollama.httpx.AsyncClient = OwnedClient
    try:
        backend = OllamaBackend(stall_seconds=0)
        completion = backend.complete([], [])
        await completion.__anext__()
        if close_mode == "aclose":
            await completion.aclose()
        else:
            async def consume() -> None:
                async for _ in completion:
                    pass

            running = asyncio.create_task(consume())
            await asyncio.sleep(0)
            running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
            await asyncio.sleep(0)
        assert clients and clients[0].response.exited and clients[0].closed
    finally:
        ollama.httpx.AsyncClient = original


@pytest.mark.asyncio
@pytest.mark.parametrize("field, value", [
    ("tool_calls", {}), ("tool_calls", ""), ("tool_calls", 0),
    ("tool_calls", None), ("done", "false"), ("done", 1), ("done", None),
])
async def test_ollama_rejects_wrong_typed_frame_fields(field, value) -> None:
    item = {"message": {}}
    if field == "tool_calls":
        item["message"][field] = value
    else:
        item[field] = value
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=(json.dumps(item) + "\n").encode())
    )) as client:
        with pytest.raises(OllamaError, match=field):
            [event async for event in OllamaBackend(client=client).complete([], [])]


@pytest.mark.asyncio
async def test_ollama_stall_zero_allows_progressing_mock_transport() -> None:
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"message":{"content":"slow"}}\n'
            await asyncio.sleep(0.02)
            yield b'{"message":{"content":"ok"},"done":true}\n'

    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Body())
    )) as client:
        events = [event async for event in OllamaBackend(
            client=client, stall_seconds=0
        ).complete([], [])]
    assert events[-1].type is StreamEventType.MESSAGE_END


@pytest.mark.asyncio
async def test_ollama_watchdog_retries_real_transport_stall(monkeypatch) -> None:
    attempts = 0

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if attempts == 1:
                yield b'{"message":{"content":"partial"}}\n'
                await asyncio.sleep(0.05)
            else:
                yield b'{"message":{"content":"ok"},"done":true}\n'

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=Body())

    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = [event async for event in OllamaBackend(
            client=client, stall_seconds=0.01, stall_retries=1
        ).complete([], [])]
    assert attempts == 2
    assert any(event.type is StreamEventType.RETRY for event in events)
    assert events[-1].message.content[0].text == "ok"  # type: ignore[union-attr]


async def test_ollama_rejects_images() -> None:
    from zeta.protocol.types import ImageContent

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        with pytest.raises(OllamaError, match="images/reasoning are unsupported"):
            [
                event
                async for event in OllamaBackend(client=client).complete(
                    [Message(MessageRole.USER, [ImageContent("a", "image/png")])], []
                )
            ]
