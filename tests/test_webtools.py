from __future__ import annotations

import asyncio
import gzip
import threading
import zlib
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.codex import CodexLoginRequiredError
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.core.tool_dispatch import dispatch_tool_calls
from zeta.protocol.types import StreamEventType, ToolCall, flatten_tool_content
from zeta.runtime.loop import AgentLoop
from zeta.runtime.loop.agent import _validated_tool_result
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry, websearch
from zeta.tools import fetch as fetch_tool

_ASYNC_CLIENT = httpx.AsyncClient
_ASYNC_HANDLE_REQUEST = httpx.AsyncHTTPTransport.handle_async_request


@pytest.fixture(autouse=True)
async def owned_tool_registries(monkeypatch: pytest.MonkeyPatch):
    """Keep registry spills alive through assertions, then close every owner."""
    registry_type = ToolRegistry
    registries: list[ToolRegistry] = []

    def create_registry(*args, **kwargs):
        registry = registry_type(*args, **kwargs)
        registries.append(registry)
        return registry

    monkeypatch.setitem(globals(), "ToolRegistry", create_registry)
    yield
    for registry in reversed(registries):
        await registry.close()


async def _tick_heartbeat(progressed: threading.Event) -> None:
    progressed.set()


def _release_gate_after_loop_progress(
    loop: asyncio.AbstractEventLoop,
    entered: threading.Event,
    release: threading.Event,
    progressed: threading.Event,
    outcomes: list[bool],
    *required: threading.Event,
) -> None:
    assert entered.wait(timeout=5)
    asyncio.run_coroutine_threadsafe(_tick_heartbeat(progressed), loop)
    all_progressed = progressed.wait(timeout=5)
    for event in required:
        all_progressed = event.wait(timeout=5) and all_progressed
    outcomes.append(all_progressed)
    release.set()


class _ChunkedByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes, *, first_chunk_size: int | None = None) -> None:
        self.content = content
        self.first_chunk_size = first_chunk_size

    async def __aiter__(self):
        split_at = self.first_chunk_size or max(1, len(self.content) // 2)
        yield self.content[:split_at]
        yield self.content[split_at:]

    async def aclose(self) -> None:
        return None


class _KeepAliveServer:
    def __init__(self, body_for_host=None, *, response_gate: asyncio.Event | None = None):
        self.body_for_host = body_for_host or (lambda _host: b"ok")
        self.response_gate = response_gate
        self.accepted = 0
        self.closed = 0
        self.request_started = asyncio.Event()
        self.connection_closed = asyncio.Event()
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.sockets[0].getsockname()[1]

    @property
    def open_connections(self) -> int:
        return self.accepted - self.closed

    def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.accepted += 1
        self._writers.add(writer)
        task = asyncio.create_task(self._serve(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    return
                header_lines = head.decode("latin-1").split("\r\n")
                headers = {
                    key.lower(): value.strip()
                    for line in header_lines[1:]
                    if ":" in line
                    for key, value in [line.split(":", 1)]
                }
                length = int(headers.get("content-length", "0"))
                if length:
                    await reader.readexactly(length)
                self.request_started.set()
                if self.response_gate is not None:
                    await await_or_fail(
                        self.response_gate.wait(), what="HTTP response gate to open"
                    )
                body = self.body_for_host(headers.get("host", ""))
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode()
                    + b"Content-Type: text/html\r\nConnection: keep-alive\r\n\r\n"
                    + body
                )
                await writer.drain()
        finally:
            self._writers.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass
            self.closed += 1
            self.connection_closed.set()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for writer in tuple(self._writers):
            writer.close()
        if self._tasks:
            await await_or_fail(
                asyncio.gather(*self._tasks, return_exceptions=True),
                what="keep-alive server connections to close",
            )


def _route_test_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", _ASYNC_HANDLE_REQUEST
    )
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 0),
            )
        ],
    )


def _raw_deflate(value: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(value) + compressor.flush()


def _mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
    *,
    client_kwargs: dict[str, object] | None = None,
):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    def make_client(**kwargs):
        if client_kwargs is not None:
            client_kwargs.update(kwargs)
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(fetch_tool.httpx, "AsyncClient", make_client)


async def _execute_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    *,
    arguments: dict[str, object] | None = None,
    max_output_chars: int = 50_000,
):
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(
        tmp_path,
        max_output_chars=max_output_chars,
        skill_catalog=SkillCatalog.empty(),
    )
    return await registry.execute(
        ToolCall("fetch-1", "fetch", arguments or {"url": "example.com"})
    )


def test_fetch_schema_documents_max_bytes(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    schema = next(item for item in registry.schemas if item["name"] == "fetch")
    description = schema["parameters"]["properties"]["max_bytes"]["description"]
    tool_description = schema["description"]

    assert "100 MiB" in description
    assert "lower" in description
    assert "received and decompressed" in description
    assert "successful partial result" in description
    assert "safety ceiling" in tool_description


@pytest.mark.asyncio
async def test_fetch_html_extracts_text_and_expands_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<h1>Title</h1><script>bad()</script><p>Hello <a href="/next">there</a>.</p>',
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == (
        "Title\nHello there (https://example.com/next)."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "body"),
    [("text/plain", "plain text"), ("application/json", '{"ok": true}')],
)
async def test_fetch_passes_text_and_json_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    body: str,
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": content_type}, text=body),
    )

    assert result["content"][0]["text"] == body


@pytest.mark.asyncio
async def test_fetch_refuses_binary_and_returns_partial_large_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"pdf"),
    )
    large = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, content=b"123456"),
        arguments={"url": "example.com", "max_bytes": 5},
    )

    assert binary["isError"] is True
    assert "application/pdf" in binary["content"][0]["text"]
    assert large["isError"] is False
    assert "received safety limit" in large["content"][0]["text"]
    assert Path(large["content"][0]["spill_path"]).read_text() == "12345"


@pytest.mark.asyncio
async def test_real_network_connections_are_blocked() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssertionError, match="real network"):
            await client.get("https://example.com")


@pytest.mark.asyncio
async def test_registry_reuses_fetch_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _KeepAliveServer()
    await server.start()
    _route_test_hosts(monkeypatch)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        for index in range(5):
            result = await registry.execute(
                ToolCall(
                    f"fetch-{index}",
                    "fetch",
                    {"url": f"http://fetch.test:{server.port}/"},
                )
            )
            assert result["isError"] is False, result

        assert server.accepted == 1
    finally:
        await registry.close()
        await server.close()


@pytest.mark.asyncio
async def test_registry_reuses_each_duckduckgo_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def body_for_host(host: str) -> bytes:
        if host.startswith("html.test"):
            return b'<link rel="canonical" href="https://duckduckgo.com/">'
        return (
            b'<a class="result-link" href="https://example.com">result</a>'
            b'<td class="result-snippet">snippet</td>'
        )

    server = _KeepAliveServer(body_for_host)
    await server.start()
    _route_test_hosts(monkeypatch)
    monkeypatch.setattr(
        websearch, "DDG_HTML_ENDPOINT", f"http://html.test:{server.port}/html/"
    )
    monkeypatch.setattr(
        websearch, "DDG_LITE_ENDPOINT", f"http://lite.test:{server.port}/lite/"
    )

    async def no_login(*_args, **_kwargs):
        raise CodexLoginRequiredError("no login")

    monkeypatch.setattr(websearch.codex, "search", no_login)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        for index in range(3):
            result = await registry.execute(
                ToolCall(f"search-{index}", "websearch", {"query": "zeta"})
            )
            assert result["structuredContent"]["backend"] == "duckduckgo"

        assert server.accepted == 2
    finally:
        await registry.close()
        await server.close()


@pytest.mark.asyncio
async def test_registry_reuses_codex_search_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        b"data: [DONE]\n\n"
    )
    server = _KeepAliveServer(lambda _host: body)
    await server.start()
    _route_test_hosts(monkeypatch)
    monkeypatch.setattr(
        websearch.codex, "CODEX_API_URL", f"http://codex.test:{server.port}/responses"
    )

    async def access_token(_store, _client):
        return "current-token"

    monkeypatch.setattr(
        websearch.codex.AccountCredentials, "access_token", access_token
    )
    monkeypatch.setattr(
        websearch.codex.AccountCredentials,
        "auth_headers",
        lambda _store, token: {"Authorization": f"Bearer {token}"},
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        for index in range(3):
            result = await registry.execute(
                ToolCall(f"codex-{index}", "websearch", {"query": "zeta"})
            )
            assert result["structuredContent"]["backend"] == "codex", result

        assert server.accepted == 1
    finally:
        await registry.close()
        await server.close()


@pytest.mark.asyncio
async def test_registry_close_drains_in_flight_web_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response_gate = asyncio.Event()
    server = _KeepAliveServer(response_gate=response_gate)
    await server.start()
    _route_test_hosts(monkeypatch)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    request = asyncio.create_task(
        registry.execute(
            ToolCall(
                "fetch-active",
                "fetch",
                {"url": f"http://active.test:{server.port}/"},
            )
        )
    )
    close_task: asyncio.Task[tuple[str, ...]] | None = None
    try:
        await await_or_fail(
            server.request_started.wait(), what="in-flight fetch to reach server"
        )
        close_task = asyncio.create_task(registry.close())
        close_started = asyncio.Event()
        asyncio.get_running_loop().call_soon(close_started.set)
        await await_or_fail(close_started.wait(), what="registry close to start")
        assert not close_task.done(), "registry close did not drain the active request"

        rejected = await registry.execute(
            ToolCall(
                "fetch-rejected",
                "fetch",
                {"url": f"http://active.test:{server.port}/next"},
            )
        )
        assert rejected["isError"] is True
        assert "web client owner is closed" in rejected["content"][0]["text"]

        response_gate.set()
        result = await await_or_fail(request, what="in-flight fetch to finish")
        assert result["isError"] is False
        await await_or_fail(close_task, what="registry web clients to close")
        await await_or_fail(
            server.connection_closed.wait(), what="request connection context to close"
        )
        assert close_task.done()
        assert server.open_connections == 0
    finally:
        response_gate.set()
        if not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        if close_task is not None and not close_task.done():
            await await_or_fail(close_task, what="registry close cleanup")
        await server.close()


@pytest.mark.asyncio
async def test_cloned_registry_owns_separate_web_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _KeepAliveServer()
    await server.start()
    _route_test_hosts(monkeypatch)
    parent = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    child_store = ConversationStore(tmp_path / "child")
    child = parent.clone_for_session(child_store)
    try:
        for registry, call_id in ((parent, "parent"), (child, "child")):
            result = await registry.execute(
                ToolCall(
                    call_id,
                    "fetch",
                    {"url": f"http://clone.test:{server.port}/"},
                )
            )
            assert result["isError"] is False, result
        assert server.accepted == 2

        await parent.close()
        result = await child.execute(
            ToolCall(
                "child-again",
                "fetch",
                {"url": f"http://clone.test:{server.port}/again"},
            )
        )
        assert result["isError"] is False
        assert server.accepted == 2
    finally:
        await parent.close()
        await child.close()
        child_store.close()
        await server.close()


@pytest.mark.asyncio
async def test_registry_bounds_idle_origin_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _KeepAliveServer()
    await server.start()
    _route_test_hosts(monkeypatch)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        for index in range(10):
            result = await registry.execute(
                ToolCall(
                    f"origin-{index}",
                    "fetch",
                    {"url": f"http://origin-{index}.test:{server.port}/"},
                )
            )
            assert result["isError"] is False, result

        assert server.open_connections == 8
    finally:
        await registry.close()
        await server.close()


@pytest.mark.asyncio
async def test_fetch_streams_body_and_ignores_lying_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-length": "5", "content-type": "text/plain"},
            content=b"123456",
        ),
        arguments={"url": "example.com", "max_bytes": 5},
    )

    assert result["isError"] is False
    assert "received safety limit" in result["content"][0]["text"]
    assert Path(result["content"][0]["spill_path"]).read_text() == "12345"


@pytest.mark.asyncio
async def test_fetch_dns_resolution_does_not_block_loop_or_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    progressed = threading.Event()
    cancelled = threading.Event()
    outcomes: list[bool] = []

    def held_getaddrinfo(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ]

    monkeypatch.setattr(fetch_tool.socket, "getaddrinfo", held_getaddrinfo)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    fetch = asyncio.create_task(
        registry.execute(ToolCall("fetch-1", "fetch", {"url": "example.com"}))
    )
    fetch.add_done_callback(lambda _task: cancelled.set())
    worker = threading.Thread(
        target=_release_gate_after_loop_progress,
        args=(
            asyncio.get_running_loop(),
            entered,
            release,
            progressed,
            outcomes,
            cancelled,
        ),
    )
    worker.start()
    await asyncio.to_thread(entered.wait)
    fetch.cancel()

    await asyncio.gather(fetch, return_exceptions=True)
    await asyncio.to_thread(worker.join)

    assert outcomes == [True]
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_fetch_decompression_does_not_block_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    progressed = threading.Event()
    outcomes: list[bool] = []
    original_decompressobj = zlib.decompressobj

    class HeldDecoder:
        def __init__(self, *args, **kwargs) -> None:
            self.decoder = original_decompressobj(*args, **kwargs)

        def decompress(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return self.decoder.decompress(*args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self.decoder, name)

    monkeypatch.setattr(fetch_tool.zlib, "decompressobj", HeldDecoder)
    worker = threading.Thread(
        target=_release_gate_after_loop_progress,
        args=(asyncio.get_running_loop(), entered, release, progressed, outcomes),
    )
    worker.start()
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": "gzip"},
            stream=_ChunkedByteStream(gzip.compress(b"x" * (1024 * 1024))),
        ),
        arguments={"url": "example.com", "max_bytes": 2 * 1024 * 1024},
    )
    await asyncio.to_thread(worker.join)

    assert outcomes == [True]
    assert result["isError"] is False


@pytest.mark.asyncio
async def test_fetch_temp_file_write_does_not_block_loop_and_cleans_up_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    progressed = threading.Event()
    cancelled = threading.Event()
    outcomes: list[bool] = []
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    paths: list[Path] = []

    class HeldWrite:
        def __init__(self, handle) -> None:
            self.handle = handle

        def write(self, value: bytes) -> int:
            entered.set()
            assert release.wait(timeout=5)
            return self.handle.write(value)

        def __getattr__(self, name: str):
            return getattr(self.handle, name)

    @contextmanager
    def temporary_file():
        path = tmp_path / f"held-{len(paths)}.tmp"
        paths.append(path)
        with path.open("w+b") as handle:
            yield HeldWrite(handle)
        path.unlink()

    monkeypatch.setattr(registry.spills, "temporary_file", temporary_file)
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ChunkedByteStream(b"body"),
            request=request,
        ),
    )
    fetch = asyncio.create_task(
        registry.execute(ToolCall("fetch-1", "fetch", {"url": "example.com"}))
    )
    fetch.add_done_callback(lambda _task: cancelled.set())
    worker = threading.Thread(
        target=_release_gate_after_loop_progress,
        args=(
            asyncio.get_running_loop(),
            entered,
            release,
            progressed,
            outcomes,
        ),
    )
    worker.start()
    await asyncio.to_thread(entered.wait)
    fetch.cancel()

    await asyncio.gather(fetch, return_exceptions=True)
    await asyncio.to_thread(worker.join)

    assert outcomes == [True]
    assert cancelled.is_set()
    assert not any(path.exists() for path in paths)


@pytest.mark.asyncio
async def test_fetch_temp_file_read_does_not_block_loop_and_cleans_up_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = threading.Event()
    release = threading.Event()
    progressed = threading.Event()
    cancelled = threading.Event()
    outcomes: list[bool] = []
    cancel_requested = threading.Event()
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    paths: list[Path] = []

    class HeldRead:
        def __init__(self, handle) -> None:
            self.handle = handle

        def read(self, *args):
            entered.set()
            assert release.wait(timeout=15)
            return self.handle.read(*args)

        def __getattr__(self, name: str):
            return getattr(self.handle, name)

    @contextmanager
    def temporary_file():
        path = tmp_path / f"held-{len(paths)}.tmp"
        paths.append(path)
        with path.open("w+b") as handle:
            yield HeldRead(handle) if len(paths) == 3 else handle
        path.unlink()

    monkeypatch.setattr(registry.spills, "temporary_file", temporary_file)
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"body",
            request=request,
        ),
    )
    fetch = asyncio.create_task(
        registry.execute(ToolCall("fetch-1", "fetch", {"url": "example.com"}))
    )
    fetch.add_done_callback(lambda _task: cancelled.set())
    loop = asyncio.get_running_loop()

    def release_after_progress() -> None:
        assert entered.wait(timeout=5)
        asyncio.run_coroutine_threadsafe(_tick_heartbeat(progressed), loop)
        heartbeat_progressed = progressed.wait(timeout=5)
        if heartbeat_progressed:
            assert cancel_requested.wait(timeout=5)
        outcomes.append(heartbeat_progressed)
        release.set()

    worker = threading.Thread(target=release_after_progress)
    worker.start()
    await asyncio.to_thread(entered.wait)
    fetch.cancel()
    cancel_requested.set()

    await asyncio.gather(fetch, return_exceptions=True)
    await asyncio.to_thread(worker.join)

    assert outcomes == [True]
    assert cancelled.is_set()
    assert not any(path.exists() for path in paths)


@pytest.mark.asyncio
async def test_fetch_decodes_gzip_response_from_raw_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(gzip.compress(b"compressed response")),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "compressed response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("encoding", "compressed", "expected"),
    [
        ("gzip", gzip.compress(b"gzip response", mtime=0), "gzip response"),
        ("deflate", zlib.compress(b"zlib response"), "zlib response"),
        (
            "deflate",
            _raw_deflate(b"raw response"),
            "raw response",
        ),
    ],
)
async def test_fetch_decodes_supported_content_encodings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    compressed: bytes,
    expected: str,
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": encoding,
            },
            stream=_ChunkedByteStream(compressed),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == expected


@pytest.mark.asyncio
async def test_fetch_decodes_raw_deflate_with_one_byte_first_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "deflate",
            },
            stream=_ChunkedByteStream(
                _raw_deflate(b"raw response"), first_chunk_size=1
            ),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "raw response"


@pytest.mark.asyncio
async def test_fetch_marks_truncated_gzip_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compressed = gzip.compress(b"truncated response")
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(compressed[:-1]),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["truncated"] is True
    assert "stream ended early" in result["content"][0]["text"]
    assert Path(result["content"][0]["spill_path"]).read_text() == (
        "truncated response"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("content_encoding", ["identity", "x-custom"])
async def test_fetch_preserves_uncompressed_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_encoding: str,
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": content_encoding},
            stream=_ChunkedByteStream(b"identity response"),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "identity response"


@pytest.mark.asyncio
async def test_fetch_rejects_multiple_content_encodings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip, br",
            },
            stream=_ChunkedByteStream(b"body"),
        ),
    )

    assert result["isError"] is True
    assert "multiple content encodings" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_returns_partial_when_decompressed_body_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(gzip.compress(b"x" * 100)),
        ),
        arguments={"url": "example.com", "max_bytes": 32},
    )

    assert result["isError"] is False
    assert result["content"][0]["truncated"] is True
    assert "decompressed safety limit" in result["content"][0]["text"]
    assert Path(result["content"][0]["spill_path"]).read_text() == "x" * 32


@pytest.mark.asyncio
@pytest.mark.parametrize("with_stream", [False, True])
async def test_parallel_fetches_cancel_on_production_dispatch_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_stream: bool,
) -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()
    started_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal started_count
        del request
        started_count += 1
        if started_count == 2:
            started.set()
        await blocked.wait()
        raise AssertionError("blocked transport was not canceled")

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    calls = [
        ToolCall("fetch-a", "fetch", {"url": "example.com/a"}),
        ToolCall("fetch-b", "fetch", {"url": "example.com/b"}),
    ]
    loop = None
    if with_stream:

        async def run_streaming() -> list[object]:
            return await asyncio.gather(
                *(
                    registry.execute(call, _stream_sink=lambda event: None)
                    for call in calls
                )
            )

        task = asyncio.create_task(run_streaming())
    else:
        loop = AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path),
            registry=registry,
            skill_catalog=SkillCatalog.empty(),
        )

        async def run_dispatch():
            return [
                event
                async for event in dispatch_tool_calls(
                    loop, calls, _validated_tool_result
                )
            ]

        task = asyncio.create_task(run_dispatch())

    await asyncio.wait_for(started.wait(), timeout=1)
    registry.abort()
    results = await asyncio.wait_for(task, timeout=1)

    if with_stream:
        assert [result["content"][0]["text"] for result in results] == [
            "tool execution canceled",
            "tool execution canceled",
        ]
    else:
        assert [
            event.tool_result.content
            for event in results
            if event.type is StreamEventType.TOOL_EXECUTION_END
            and event.tool_result is not None
        ] == ["tool execution canceled", "tool execution canceled"]
    if loop is not None:
        await loop.close()
        loop.store.close()


@pytest.mark.asyncio
async def test_fetch_validates_each_redirect_and_uses_final_url_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_hosts: list[str] = []
    requested_addresses: list[str] = []
    responses = {
        "https://example.com/start": httpx.Response(
            302, headers={"location": "http://example.com/final"}
        ),
        "http://example.com/final": httpx.Response(
            200, headers={"content-type": "text/plain"}, text="done"
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requested_addresses.append(request.url.host)
        requested_url = (
            f"{request.url.scheme}://{request.headers['host']}{request.url.path}"
        )
        response = responses[requested_url]
        response.request = request
        return response

    _mock_client(monkeypatch, handler)

    def getaddrinfo(host: str, *args, **kwargs):
        resolved_hosts.append(host)
        address = f"93.184.216.{33 + len(resolved_hosts)}"
        return [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", (address, 0))
        ]

    monkeypatch.setattr(fetch_tool.socket, "getaddrinfo", getaddrinfo)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/start"})
    )

    assert result["isError"] is False
    assert resolved_hosts == ["example.com", "example.com"]
    assert requested_addresses == ["93.184.216.34", "93.184.216.35"]
    assert result["content"][0]["text"].startswith(
        "notice: http URL is not encrypted\n\ndone"
    )


@pytest.mark.asyncio
async def test_get_response_preserves_post_body_on_307_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(307, headers={"location": "/final"}, request=request)
        return httpx.Response(200, text="done", request=request)

    _mock_client(monkeypatch, handler)
    response = await fetch_tool.get_response(
        "https://example.com/start",
        user_agent="test",
        method="POST",
        data={"q": "zeta"},
    )

    assert response.text == "done"
    assert [request.method for request in requests] == ["POST", "POST"]
    assert [request.content for request in requests] == [b"q=zeta", b"q=zeta"]


@pytest.mark.asyncio
async def test_get_response_switches_to_get_without_body_on_303(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(303, headers={"location": "/final"}, request=request)
        return httpx.Response(200, text="done", request=request)

    _mock_client(monkeypatch, handler)
    response = await fetch_tool.get_response(
        "https://example.com/start",
        user_agent="test",
        method="POST",
        data={"q": "zeta"},
    )

    assert response.text == "done"
    assert [request.method for request in requests] == ["POST", "GET"]
    assert requests[0].content == b"q=zeta"
    assert requests[1].content == b""


@pytest.mark.asyncio
async def test_fetch_refuses_non_http_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(302, headers={"location": "file:///tmp/secret"})
        response.request = request
        return response

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/start"})
    )

    assert result["isError"] is True
    assert "http or https" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_caps_manual_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            302,
            headers={"location": "https://example.com/loop"},
            request=request,
        )
        return response

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/start"})
    )

    assert result["isError"] is True
    assert "redirect limit exceeded (5)" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_allows_private_target_with_notice_and_blocks_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def private_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="local", request=request)

    _mock_client(monkeypatch, private_handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("192.168.1.5", 0))
        ],
    )
    allowed = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "http://printer.local"})
    )
    blocked = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-2", "fetch", {"url": "http://169.254.169.254/"})
    )

    assert allowed["isError"] is False
    assert "target resolves to a private or loopback address" in allowed["content"][0]["text"]
    assert blocked["isError"] is True
    assert "cloud metadata" in blocked["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:169.254.169.254]/",
        "http://[::169.254.169.254]/",
    ],
)
async def test_fetch_blocks_mapped_cloud_metadata_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "text/plain"}, text="no"),
        arguments={"url": url},
    )

    assert result["isError"] is True
    assert "cloud metadata" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:192.168.1.5]/",
        "http://[::192.168.1.5]/",
    ],
)
async def test_fetch_notices_mapped_private_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "text/plain"}, text="local"),
        arguments={"url": url},
    )

    assert result["isError"] is False
    assert (
        "target resolves to a private or loopback address"
        in result["content"][0]["text"]
    )


@pytest.mark.asyncio
async def test_fetch_pins_connection_to_first_validated_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups = iter(
        [
            ("192.168.1.5",),
            ("93.184.216.34",),
        ]
    )
    lookup_count = 0

    def getaddrinfo(*args, **kwargs):
        nonlocal lookup_count
        lookup_count += 1
        address = next(lookups)[0]
        return [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                (address, 0),
            )
        ]

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="local", request=request)

    client_kwargs: dict[str, object] = {}
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    _mock_client(monkeypatch, handler, client_kwargs=client_kwargs)
    monkeypatch.setattr(fetch_tool.socket, "getaddrinfo", getaddrinfo)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/"})
    )

    assert result["isError"] is False
    assert lookup_count == 1
    assert client_kwargs["trust_env"] is False
    assert str(requests[0].url) == "https://192.168.1.5/"
    assert requests[0].headers["host"] == "example.com"
    assert requests[0].extensions["sni_hostname"] == "example.com"
    assert (
        "target resolves to a private or loopback address"
        in result["content"][0]["text"]
    )


@pytest.mark.asyncio
async def test_fetch_output_is_paginated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, text="x" * 50_100),
        max_output_chars=10_000,
    )

    block = result["content"][0]
    assert block["truncated"] is True
    assert block["full_size"] == 50_100
    assert block["full_size_chars"] == 50_100
    notice, page = block["text"].split("\n\n", 1)
    assert "full readable content (50100 bytes)" in notice
    assert page == "x" * len(page)
    assert block["next_offset"] == len(page)
    assert Path(block["spill_path"]).read_text() == "x" * 50_100


@pytest.mark.asyncio
async def test_fetch_pagination_reassembles_extracted_readable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = (
        "<article><h1>Title</h1><p>Readable article text. "
        + "word " * 12
        + "</p></article>"
    )
    expected = fetch_tool._readable_content("https://example.com", "text/html", html)
    offset = 0
    pages: list[str] = []

    while True:
        result = await _execute_fetch(
            tmp_path,
            monkeypatch,
            httpx.Response(200, headers={"content-type": "text/html"}, text=html),
            arguments={"url": "example.com", "offset": offset},
            max_output_chars=17,
        )
        block = result["content"][0]
        pages.append(block["text"])
        assert block["full_size"] == len(expected)
        if not block["truncated"]:
            break
        assert block["next_offset"] == offset + len(block["text"])
        offset = block["next_offset"]

    assert len(pages) > 2
    assert "".join(pages) == expected


@pytest.mark.asyncio
async def test_fetch_pagination_overshoot_returns_empty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "readable body"
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "text/plain"}, text=body),
        arguments={"url": "example.com", "offset": len(body) + 100},
    )

    block = result["content"][0]
    assert block["text"] == ""
    assert block["truncated"] is False
    assert block["full_size"] == len(body)
    assert "next_offset" not in block


@pytest.mark.asyncio
async def test_fetch_pagination_repeats_notices_without_advancing_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "readable body " * 10

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=body,
            request=request,
        )

    _mock_client(monkeypatch, handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.5", 0),
            )
        ],
    )
    notice = (
        "notice: http URL is not encrypted\n"
        "notice: target resolves to a private or loopback address\n\n"
    )
    offset = 0
    body_pages: list[str] = []
    registry = ToolRegistry(
        tmp_path, max_output_chars=128, skill_catalog=SkillCatalog.empty()
    )
    try:
        while True:
            result = await registry.execute(
                ToolCall(
                    "fetch-1",
                    "fetch",
                    {"url": "http://printer.local", "offset": offset},
                )
            )
            block = result["content"][0]
            assert block["text"].startswith(notice)
            page = block["text"][len(notice) :]
            body_pages.append(page)
            assert page == body[offset : offset + len(page)]
            if not block["truncated"]:
                break
            assert block["next_offset"] == offset + len(page)
            offset = block["next_offset"]
    finally:
        await registry.close()

    assert "".join(body_pages) == body


@pytest.mark.asyncio
async def test_fetch_pagination_reports_utf8_bytes_and_character_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="éé",
        ),
        arguments={"url": "example.com", "offset": 0},
        max_output_chars=2,
    )

    block = result["content"][0]
    assert block["text"] == "é"
    assert block["full_size"] == 4
    assert block["full_size_chars"] == 2
    assert block["next_offset"] == 1
    assert flatten_tool_content([block]) == (
        "é\n[truncated: full_size_chars=2 chars; next_offset=1]"
    )


@pytest.mark.asyncio
async def test_fetch_small_cap_spills_notice_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="readable body",
            request=request,
        )

    _mock_client(monkeypatch, handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.5", 0),
            )
        ],
    )
    result = await ToolRegistry(tmp_path, max_output_chars=34, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall(
            "fetch-1",
            "fetch",
            {"url": "http://printer.local", "offset": 0},
        )
    )

    block = result["content"][0]
    assert result["isError"] is False
    assert block["truncated"] is True
    assert Path(block["spill_path"]).read_text().startswith("notice: http URL")
    assert "next_offset" not in block


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.TimeoutException("slow"),
        httpx.TooManyRedirects("loop", request=httpx.Request("GET", "https://x.test")),
    ],
)
async def test_fetch_timeout_and_redirect_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise error

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "example.com"})
    )

    assert result["isError"] is True
    expected = "timed out" if isinstance(error, httpx.TimeoutException) else "redirect limit"
    assert expected in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500])
async def test_fetch_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(status, text="failure"),
    )

    assert result["isError"] is True
    assert f"HTTP {status}" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_uses_declared_charset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=iso-8859-1"},
            content="café".encode("iso-8859-1"),
        ),
    )

    assert result["content"][0]["text"] == "café"


@pytest.mark.asyncio
async def test_websearch_parses_saved_duckduckgo_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "duckduckgo.html"
    body = fixture.read_text(encoding="utf-8")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/html/"
        assert request.content == b"q=zeta"
        assert request.headers["accept-language"] == "en-US,en;q=0.9"
        assert request.headers["user-agent"].startswith("Mozilla/5.0")
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "backend": "duckduckgo",
        "results": [
            {
                "title": "First result",
                "url": "https://example.com/one",
                "snippet": "A short description.",
            }
        ]
    }


@pytest.mark.asyncio
async def test_websearch_falls_back_to_lite_for_provider_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = (
        Path(__file__).parent / "fixtures" / "duckduckgo_challenge.html"
    ).read_text(encoding="utf-8")
    lite = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            assert request.headers["host"] == "html.duckduckgo.com"
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=challenge
            )
        assert request.headers["host"] == "lite.duckduckgo.com"
        assert request.method == "POST"
        assert request.content == b"q=zeta"
        return httpx.Response(200, headers={"content-type": "text/html"}, text=lite)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert result["structuredContent"]["results"][0]["title"] == (
        "Login / Sign up - zeta"
    )
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_websearch_falls_back_to_lite_for_parser_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lite = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>unexpected response</body></html>",
            )
        return httpx.Response(200, headers={"content-type": "text/html"}, text=lite)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert calls == 2


def test_websearch_detects_captured_provider_challenge() -> None:
    body = (Path(__file__).parent / "fixtures" / "duckduckgo_challenge.html").read_text(
        encoding="utf-8"
    )

    with pytest.raises(
        websearch.SearchProviderChallengeError,
        match="search provider served a no-results/challenge page",
    ):
        websearch.parse_search_results(body, max_results=8)


def test_websearch_parses_lite_fixture() -> None:
    body = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )

    assert websearch.parse_lite_search_results(body, max_results=1) == [
        {
            "title": "Login / Sign up - zeta",
            "url": "https://zeta-ai.io/en/login",
            "snippet": (
                "The No.1 AI chat! Over 13 hours of weekly use — and it's free. "
                "Not using zeta yet? Everyone else is!"
            ),
        }
    ]


def test_websearch_decodes_wrapped_lite_result_url() -> None:
    body = (
        '<a class="result-link" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fresult">'
        "Wrapped result"
        "</a>"
    )

    assert websearch.parse_lite_search_results(body, max_results=1) == [
        {
            "title": "Wrapped result",
            "url": "https://example.com/result",
            "snippet": "",
        }
    ]


@pytest.mark.asyncio
async def test_websearch_large_output_spills_complete_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '<a class="result__a" href="https://example.com/one">First result</a>'
        '<div class="result__snippet">A short description.</div>'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(
        tmp_path, max_output_chars=64, skill_catalog=SkillCatalog.empty()
    )
    try:
        result = await registry.execute(
            ToolCall("search-1", "websearch", {"query": "zeta"})
        )

        block = result["content"][0]
        spilled = Path(block["spill_path"]).read_text()
        assert block["truncated"] is True
        assert '"url": "https://example.com/one"' in spilled
        assert '"snippet": "A short description."' in spilled
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_websearch_empty_results_and_parse_failure() -> None:
    empty = (Path(__file__).parent / "fixtures" / "duckduckgo_empty.html").read_text(
        encoding="utf-8"
    )
    assert websearch.parse_search_results(empty, max_results=8) == []
    with pytest.raises(ValueError, match="search backend failed"):
        websearch.parse_search_results(
            "<html><body>No results found</body></html>", max_results=8
        )
    with pytest.raises(ValueError, match="search backend failed"):
        websearch.parse_search_results(
            '<div class="no-results__container result__title"><span class="no-results">'
            '<div class="no-results__message"><h1>temporarily blocked</h1></div></span></div>',
            max_results=8,
        )


@pytest.mark.asyncio
async def test_discovery_and_approval_gate_network_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network handler should not run")

    _mock_client(monkeypatch, forbidden)
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(always_deny={"fetch", "websearch"}, store=store)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
skill_catalog=SkillCatalog.empty(),
    )

    try:
        assert {"fetch", "websearch"} <= registry.definitions_by_name.keys()
        fetch_result = await registry.execute(
            ToolCall("fetch-1", "fetch", {"url": "example.com"})
        )
        search_result = await registry.execute(
            ToolCall("search-1", "websearch", {"query": "zeta"})
        )

        assert fetch_result["content"][0]["text"] == "tool execution denied"
        assert search_result["content"][0]["text"] == "tool execution denied"
        assert calls == 0
    finally:
        store.close()


@pytest.mark.asyncio
async def test_get_response_retries_with_identity_on_decode_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/plain",
                    "content-encoding": "gzip",
                },
                stream=_ChunkedByteStream(b"this is not really gzip"),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ChunkedByteStream(b"recovered body"),
            request=request,
        )

    _mock_client(monkeypatch, handler)
    response = await fetch_tool.get_response(
        "https://example.com/data", user_agent="test"
    )

    assert response.text == "recovered body"
    assert len(requests) == 2
    assert requests[0].headers.get("accept-encoding", "").lower() != "identity"
    assert requests[1].headers.get("accept-encoding", "").lower() == "identity"


@pytest.mark.asyncio
async def test_get_response_surfaces_error_when_identity_retry_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(b"still not gzip"),
            request=request,
        )

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="request failed:.*incorrect header check"):
        await fetch_tool.get_response(
            "https://example.com/data", user_agent="test"
        )

    assert len(requests) == 2


@pytest.mark.asyncio
async def test_fetch_tool_recovers_from_bad_encoding_via_identity_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/plain",
                    "content-encoding": "gzip",
                },
                stream=_ChunkedByteStream(b"raw plain body"),
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=_ChunkedByteStream(b"raw plain body"),
            request=request,
        )

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/data"})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "raw plain body"
    assert calls == 2


@pytest.mark.asyncio
async def test_fetch_tool_surfaces_decompression_error_after_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(b"not gzip at all"),
            request=request,
        )

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/data"})
    )

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "request failed:" in text
    assert "incorrect header check" in text
    assert calls == 2


@pytest.mark.asyncio
async def test_get_response_skips_identity_retry_when_caller_pins_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(b"not gzip"),
            request=request,
        )

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="request failed"):
        await fetch_tool.get_response(
            "https://example.com/data",
            user_agent="test",
            headers={"Accept-Encoding": "identity"},
        )

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_websearch_parser_failure_error_carries_status_and_body_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "<html><body>the backend was temporarily blocked, please try later</body></html>"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=body,
            request=request,
        )

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    with pytest.raises(websearch.SearchParserError) as excinfo:
        await websearch._ddg_search("zeta", 1, registry.web_clients)

    message = str(excinfo.value)
    assert "search backend failed" in message
    assert "HTTP 200" in message
    assert "temporarily blocked" in message


@pytest.mark.asyncio
async def test_websearch_tool_surface_includes_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "<html><body>captcha challenge please solve</body></html>"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=body,
            request=request,
        )

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("search-1", "websearch", {"query": "zeta"})
    )

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "search backend failed" in text
    assert "HTTP 200" in text
    assert "captcha challenge" in text
