from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from zeta.config.settings import load_settings, resolve
from zeta.core.project_context import ProjectContext
from zeta.core.session import SessionManager
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
from zeta.runtime.cleanup import close_session
from zeta.runtime.composition import compose_runtime
from zeta.server import runtime as server_runtime
from zeta.skills.agent_catalog import AgentCatalog
from zeta.skills.catalog import SkillCatalog
from zeta.tui.bootstrap import build_backend as build_interactive_backend

_ORIGINAL_ASYNC_HTTP_REQUEST = httpx.AsyncHTTPTransport.handle_async_request
_STALL_SECONDS = 0.05
_CLIENT_DEADLINE_SECONDS = 0.75


async def _captured_num_ctx(backend: OllamaBackend) -> int:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=(
                json.dumps({"message": {"content": "ok"}, "done": True}) + "\n"
            ).encode(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend.client = client
        [event async for event in backend.complete([], [])]
    return payloads[0]["options"]["num_ctx"]


class _ClosingStream(httpx.AsyncByteStream):
    def __init__(
        self, stream: httpx.AsyncByteStream, transport: _LoopbackAsyncHTTPTransport
    ) -> None:
        self.stream = stream
        self.transport = transport

    async def __aiter__(self):
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        await self.stream.aclose()
        self.transport.closed_streams += 1


class _LoopbackAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """Use the captured real transport only for this test server."""

    def __init__(self) -> None:
        super().__init__()
        self.closed_streams = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host in {"127.0.0.1", "localhost", "::1"}
        response = await _ORIGINAL_ASYNC_HTTP_REQUEST(self, request)
        response.stream = _ClosingStream(response.stream, self)
        return response


class _TCPHTTPServer:
    def __init__(self) -> None:
        self.scripts: list = []
        self.calls = 0
        self.first_body_sent = asyncio.Event()
        self.server: asyncio.Server | None = None
        self.tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)

    @property
    def url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            if content_length:
                await reader.readexactly(content_length)
            script = self.scripts[min(self.calls, len(self.scripts) - 1)]
            self.calls += 1
            await script(writer, self)
        except (asyncio.IncompleteReadError, ConnectionError, BrokenPipeError):
            pass
        finally:
            self.tasks.discard(task)
            writer.close()

    async def close(self) -> None:
        assert self.server is not None
        self.server.close()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.server.wait_closed()


@pytest.fixture
async def tcp_http_server():
    # The loopback transport captures the real method before conftest's
    # autouse network blocker patches it, without mutating HTTPX module state.
    assert httpx.AsyncHTTPTransport is httpx._transports.default.AsyncHTTPTransport
    server = _TCPHTTPServer()
    await server.start()
    try:
        yield server
    finally:
        await server.close()
        assert httpx.AsyncHTTPTransport is httpx._transports.default.AsyncHTTPTransport


async def _response(
    writer, server, body_chunks, *, status=200, wait=None, wait_before=None
):
    body = b"".join(body_chunks)
    writer.write(
        f"HTTP/1.1 {status} Test\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    if wait_before is not None:
        await wait_before.wait()
    for index, chunk in enumerate(body_chunks):
        writer.write(chunk)
        await writer.drain()
        if index == 0:
            server.first_body_sent.set()
        if wait is not None and index == 0:
            await wait.wait()


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_frame", "match"),
    [
        ({"message": {"content": 1}, "done": False}, "text delta"),
        ({"message": {"tool_calls": {}}, "done": False}, "tool_calls"),
        (
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": "{bad"}}
                    ]
                },
                "done": False,
            },
            "arguments are malformed JSON",
        ),
        ({"message": {"content": "leak"}, "done": "yes"}, "done must"),
        (
            {"message": {"content": "leak"}, "prompt_eval_count": "bad"},
            "prompt_eval_count",
        ),
        (
            {"message": {"content": "leak"}, "prompt_eval_count": -1},
            "prompt_eval_count",
        ),
        (
            {"message": {"content": "leak"}, "eval_count": False},
            "eval_count",
        ),
        (
            {"message": {"content": "leak"}, "eval_count": -1},
            "eval_count",
        ),
    ],
)
async def test_ollama_rejects_malformed_first_frame_before_emitting_events(
    bad_frame, match
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=(json.dumps(bad_frame) + "\n").encode()
            )
        )
    ) as client:
        events = []
        with pytest.raises(OllamaError, match=match):
            async for event in OllamaBackend(client=client).complete([], []):
                events.append(event)
        assert events == []


@pytest.mark.asyncio
async def test_ollama_rejects_later_frame_without_emitting_its_text() -> None:
    rows = [
        {"message": {"content": "good"}},
        {"message": {"content": "bad", "tool_calls": {}}, "done": True},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=("\n".join(json.dumps(row) for row in rows) + "\n").encode(),
            )
        )
    ) as client:
        events = []
        with pytest.raises(OllamaError, match="tool_calls"):
            async for event in OllamaBackend(client=client).complete([], []):
                events.append(event)
        assert [
            event.delta
            for event in events
            if event.type is StreamEventType.MESSAGE_UPDATE
        ] == ["good"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_eval_count", "bad"),
        ("prompt_eval_count", -1),
        ("eval_count", False),
        ("eval_count", -1),
    ],
)
async def test_ollama_rejects_later_malformed_frame_without_emitting_its_text(
    field: str, value: object
) -> None:
    rows = [
        {"message": {"content": "good"}},
        {"message": {"content": "bad"}, field: value, "done": True},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=("\n".join(json.dumps(row) for row in rows) + "\n").encode(),
            )
        )
    ) as client:
        events = []
        with pytest.raises(OllamaError, match=field):
            async for event in OllamaBackend(client=client).complete([], []):
                events.append(event)
        assert [event.delta for event in events if event.type is StreamEventType.MESSAGE_UPDATE] == [
            "good"
        ]


def test_ollama_endpoint_uses_global_settings_but_not_project_settings(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "settings.toml").write_text(
        'ollama_base_url = "http://settings.example"\n', encoding="utf-8"
    )
    project = tmp_path / "project" / ".zeta"
    project.mkdir(parents=True)
    (project / "settings.toml").write_text(
        'ollama_base_url = "http://project.example"\n', encoding="utf-8"
    )
    monkeypatch.delenv("ZETA_OLLAMA_BASE_URL", raising=False)
    backend, _ = build_backend("ollama", None, home=tmp_path)
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://settings.example"


def test_ollama_explicit_endpoint_beats_environment_and_settings(tmp_path, monkeypatch) -> None:
    (tmp_path / "settings.toml").write_text(
        'ollama_base_url = "http://settings.example"\n', encoding="utf-8"
    )
    monkeypatch.setenv("ZETA_OLLAMA_BASE_URL", "http://environment.example")
    backend, _ = build_backend(
        "ollama", None, home=tmp_path, ollama_base_url="http://explicit.example"
    )
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == "http://explicit.example"


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "token_budget", "expected"),
    [
        ("qwen3:4b", None, 40_960),
        ("locally-created-model", None, 8_192),
        ("qwen3:4b", 6_000, 6_000),
    ],
)
async def test_ollama_num_ctx_uses_model_window_and_effective_budget(
    model: str, token_budget: int | None, expected: int
) -> None:
    backend = OllamaBackend(model=model, token_budget=token_budget)
    assert await _captured_num_ctx(backend) == expected


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
async def test_ollama_retries_connect_error_and_closes_owned_client(
    monkeypatch,
) -> None:
    class OwnedClient:
        def __init__(self, **kwargs) -> None:
            self.closed = False

        def stream(self, *args, **kwargs):
            raise httpx.ConnectError(
                "offline", request=httpx.Request("POST", "http://ollama")
            )

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


def _endpoint_sources(
    home: Path,
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_wins: bool,
) -> str:
    home.mkdir(parents=True, exist_ok=True)
    (home / "settings.toml").write_text(
        'ollama_base_url = "http://settings.example"\n', encoding="utf-8"
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "settings.toml").write_text(
        'ollama_base_url = "http://project.example"\n', encoding="utf-8"
    )
    if environment_wins:
        monkeypatch.setenv("ZETA_OLLAMA_BASE_URL", "http://environment.example")
        return "http://environment.example"
    monkeypatch.delenv("ZETA_OLLAMA_BASE_URL", raising=False)
    return "http://settings.example"


@pytest.mark.asyncio
@pytest.mark.parametrize("environment_wins", [True, False])
async def test_interactive_composition_resolves_ollama_endpoint_centrally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_wins: bool
) -> None:
    home = tmp_path / "home"
    project_dir = tmp_path / "project" / ".zeta"
    expected = _endpoint_sources(home, project_dir, monkeypatch, environment_wins)
    config = resolve(
        load_settings(home=home, project_dir=project_dir).settings,
        cli_provider="ollama",
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=6_000,
    )
    builder_kwargs: dict[str, object] = {}

    def backend_builder(provider: str, model: str | None, **kwargs):
        builder_kwargs.update(kwargs)
        return build_interactive_backend(provider, model, **kwargs)

    composition = compose_runtime(
        home=home,
        cwd=project_dir.parent,
        manager=SessionManager(home),
        config=config,
        provider="ollama",
        model=None,
        project_context=ProjectContext("system", ()),
        backend_builder=backend_builder,
        skill_catalog=SkillCatalog.empty(),
        agent_catalog=AgentCatalog.empty(),
    )
    try:
        assert isinstance(composition.loop.backend, OllamaBackend)
        assert composition.loop.backend.base_url == expected
        assert builder_kwargs["token_budget"] == 6_000
        assert await _captured_num_ctx(composition.loop.backend) == 6_000
    finally:
        await close_session(composition.loop)


@pytest.mark.asyncio
@pytest.mark.parametrize("environment_wins", [True, False])
async def test_server_session_creation_resolves_ollama_endpoint_centrally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_wins: bool
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    expected = _endpoint_sources(home, project / ".zeta", monkeypatch, environment_wins)
    with (home / "settings.toml").open("a", encoding="utf-8") as settings:
        settings.write("token_budget = 6_001\n")
    original_default_backend = server_runtime.default_backend
    builder_kwargs: dict[str, object] = {}

    def default_backend(provider: str, model: str | None, path: Path, **kwargs):
        builder_kwargs.update(kwargs)
        return original_default_backend(provider, model, path, **kwargs)

    monkeypatch.setattr(server_runtime, "default_backend", default_backend)
    runtime = ServerRuntime(home, cwd=project, provider="ollama")
    try:
        await runtime.create_session(provider="ollama")
        assert runtime.loop is not None
        assert isinstance(runtime.loop.backend, OllamaBackend)
        assert runtime.loop.backend.base_url == expected
        assert builder_kwargs["token_budget"] == 6_001
        assert await _captured_num_ctx(runtime.loop.backend) == 6_001
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("environment_wins", [True, False])
async def test_server_provider_switch_resolves_ollama_endpoint_centrally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment_wins: bool
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    expected = _endpoint_sources(home, project / ".zeta", monkeypatch, environment_wins)
    runtime = ServerRuntime(home, cwd=project, provider="fake")
    backend = runtime.backend_for_model("ollama", "locally-created-model")
    assert isinstance(backend, OllamaBackend)
    assert backend.base_url == expected
    assert await _captured_num_ctx(backend) == 8_192


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
async def test_ollama_retries_stall_after_headers_before_first_frame(
    monkeypatch,
) -> None:
    client = _ScriptedClient(
        [
            (0, [(0.05, '{"message":{"content":"late"},"done":true}')]),
            (0, [(0, '{"message":{"content":"ok"},"done":true}')]),
        ]
    )
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
    client = _ScriptedClient(
        [
            (
                0,
                [
                    (0, '{"message":{"content":"first"}}'),
                    (0.05, '{"message":{"content":"bad"},"done":true}'),
                ],
            ),
            (0, [(0, '{"message":{"content":"ok"},"done":true}')]),
        ]
    )
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
    client = _ScriptedClient(
        [
            (0, [(0.02, '{"message":{"content":"late"},"done":true}')]),
        ]
    )
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
    client = _ScriptedClient(
        [
            (0, [(0.02, '{"message":{"content":"ok"},"done":true}')]),
        ]
    )
    events = [
        event
        async for event in OllamaBackend(client=client, stall_seconds=0).complete(
            [], []
        )
    ]
    assert events[-1].message.content[0].text == "ok"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_ollama_does_not_transport_retry_after_meaningful_output() -> None:
    class Client(_ScriptedClient):
        def stream(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _StreamContext(
                    _DelayedLinesResponse(
                        [
                            (0, '{"message":{"content":"escaped"}}'),
                        ]
                    )
                )
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
@pytest.mark.parametrize(
    "field, value",
    [
        ("tool_calls", {}),
        ("tool_calls", ""),
        ("tool_calls", 0),
        ("tool_calls", None),
        ("done", "false"),
        ("done", 1),
        ("done", None),
    ],
)
async def test_ollama_rejects_wrong_typed_frame_fields(field, value) -> None:
    item = {"message": {}}
    if field == "tool_calls":
        item["message"][field] = value
    else:
        item[field] = value
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=(json.dumps(item) + "\n").encode()
            )
        )
    ) as client:
        with pytest.raises(OllamaError, match=field):
            [event async for event in OllamaBackend(client=client).complete([], [])]


@pytest.mark.asyncio
async def test_ollama_stall_zero_allows_progressing_real_transport(
    tcp_http_server: _TCPHTTPServer,
) -> None:
    release = asyncio.Event()
    row1 = b'{"message":{"content":"slow"}}\n'
    row2 = b'{"message":{"content":"ok"},"done":true}\n'
    tcp_http_server.scripts.append(
        lambda writer, server: _response(writer, server, [row1, row2], wait=release)
    )
    transport = _LoopbackAsyncHTTPTransport()
    async with httpx.AsyncClient(transport=transport) as client:

        async def collect():
            return [
                event
                async for event in OllamaBackend(
                    client=client, base_url=tcp_http_server.url, stall_seconds=0
                ).complete([], [])
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(
            tcp_http_server.first_body_sent.wait(), timeout=_CLIENT_DEADLINE_SECONDS
        )
        release.set()
        events = await asyncio.wait_for(task, timeout=_CLIENT_DEADLINE_SECONDS)
        assert events[-1].type is StreamEventType.MESSAGE_END
        # The real transport delivered both chunks without a watchdog timeout.
        assert tcp_http_server.calls == 1
        assert transport.closed_streams == 1


@pytest.mark.asyncio
async def test_ollama_watchdog_retries_real_transport_stall(
    tcp_http_server: _TCPHTTPServer, monkeypatch
) -> None:
    partial = b'{"message":{"content":"partial"}}\n'
    ok = b'{"message":{"content":"ok"},"done":true}\n'
    never = asyncio.Event()
    tcp_http_server.scripts.extend(
        [
            lambda writer, server: _response(
                writer, server, [partial], wait_before=never
            ),
            lambda writer, server: _response(writer, server, [ok]),
        ]
    )
    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    transport = _LoopbackAsyncHTTPTransport()
    async with httpx.AsyncClient(transport=transport) as client:

        async def collect():
            return [
                event
                async for event in OllamaBackend(
                    client=client,
                    base_url=tcp_http_server.url,
                    timeout=_STALL_SECONDS / 5,
                    stall_seconds=_STALL_SECONDS,
                    stall_retries=1,
                ).complete([], [])
            ]

        events = await asyncio.wait_for(collect(), timeout=_CLIENT_DEADLINE_SECONDS)
        assert tcp_http_server.calls == 2
        assert any(
            event.type is StreamEventType.RETRY and event.data.get("is_stall") is True
            for event in events
        )
        assert events[-1].message.content[0].text == "ok"  # type: ignore[union-attr]
        assert transport.closed_streams == 2


@pytest.mark.asyncio
async def test_ollama_error_body_stall_retries_and_closes_stream(
    tcp_http_server: _TCPHTTPServer, monkeypatch
) -> None:
    never = asyncio.Event()
    error_chunks = [b"error prefix", b" delayed suffix"]
    final_error = b"final error"
    tcp_http_server.scripts.extend(
        [
            lambda writer, server: _response(
                writer, server, error_chunks, status=500, wait=never
            ),
            lambda writer, server: _response(
                writer, server, [final_error], status=400
            ),
        ]
    )
    monkeypatch.setattr("zeta.providers.transport.retry_wait_seconds", lambda *args: 0)
    transport = _LoopbackAsyncHTTPTransport()
    events = []
    async with httpx.AsyncClient(transport=transport) as client:

        async def collect() -> None:
            async for event in OllamaBackend(
                client=client,
                base_url=tcp_http_server.url,
                stall_seconds=_STALL_SECONDS,
                stall_retries=1,
            ).complete([], []):
                events.append(event)

        with pytest.raises(OllamaError, match="Ollama HTTP 400: final error"):
            await asyncio.wait_for(collect(), timeout=_CLIENT_DEADLINE_SECONDS)
        assert tcp_http_server.calls == 2
        assert any(
            event.type is StreamEventType.RETRY and event.data.get("is_stall") is True
            for event in events
        )
        assert transport.closed_streams == 2


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

@pytest.mark.asyncio
async def test_ollama_payload_sets_num_ctx_to_session_budget() -> None:
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=(json.dumps({"message": {"content": "ok"}, "done": True}) + "\n").encode(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        [event async for event in OllamaBackend(client=client, token_budget=8_192).complete([], [])]
    assert payloads[0]["options"]["num_ctx"] == 8_192
