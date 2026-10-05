from __future__ import annotations

import gzip
import shlex
import stat
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from zeta.core.store import ConversationStore
from zeta.mcp.client import translate_call_result
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools import fetch as fetch_tool

_REAL_ASYNC_HTTP_HANDLER = httpx.AsyncHTTPTransport.handle_async_request


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def _store(tmp_path: Path, name: str = "session") -> ConversationStore:
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    return ConversationStore(
        tmp_path / "sessions", session_id=name, cwd=cwd, bash_cwd=cwd
    )


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        return None


@contextmanager
def _serve(body: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/large"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _mock_fetch(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
) -> None:
    transport = httpx.MockTransport(lambda request: response)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 0),
            )
        ],
    )
    monkeypatch.setattr(
        fetch_tool.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=transport, **kwargs),
    )


@pytest.mark.asyncio
async def test_large_text_result_spills_full_content(tmp_path: Path) -> None:
    value = "0123456789" * 5_000
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=1_000,
            register_builtin=False,
            skill_catalog=SkillCatalog.empty(),
        )
        registry.register("large", lambda arguments: value)

        result = await registry.execute(ToolCall("large-call", "large", {}))
        block = result["content"][0]
        spill_path = Path(block["spill_path"])

        assert block["truncated"] is True
        assert block["full_size"] == len(value.encode())
        assert len(block["text"]) <= 1_000
        assert block["text"].startswith(value[:100])
        assert block["text"].endswith(value[-100:])
        assert str(spill_path) in block["text"]
        assert spill_path.is_absolute()
        assert spill_path.is_relative_to(store.session_dir / "spill")
        assert spill_path.read_text() == value
        assert stat.S_IMODE(spill_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(spill_path.parent.stat().st_mode) == 0o700
        await registry.close()


@pytest.mark.asyncio
async def test_spill_readable_under_restricted_policy(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    with _store(tmp_path) as store:
        producer = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=512,
            register_builtin=False,
            skill_catalog=SkillCatalog.empty(),
        )
        producer.register("large", lambda arguments: "secret" * 2_000)
        produced = await producer.execute(ToolCall("large-call", "large", {}))
        spill_path = produced["content"][0]["spill_path"]

        restricted = ToolRegistry(
            store.cwd,
            session_store=store,
            tool_allow=["read"],
            skill_catalog=SkillCatalog.empty(),
        )
        readable = await restricted.execute(
            ToolCall("read-spill", "read", {"path": spill_path})
        )
        denied = await restricted.execute(
            ToolCall("read-outside", "read", {"path": str(outside)})
        )

        assert readable["isError"] is False
        assert readable["content"][0]["text"].startswith("secret")
        assert denied["isError"] is True
        assert "path escaped" in denied["content"][0]["text"]
        await restricted.close()
        await producer.close()


@pytest.mark.asyncio
async def test_fetch_large_page_succeeds_and_spills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"a" * (5 * 1024 * 1024)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        _REAL_ASYNC_HTTP_HANDLER,
    )
    with _serve(body) as url, _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=2_000,
            skill_catalog=SkillCatalog.empty(),
        )
        result = await registry.execute(
            ToolCall("fetch-large", "fetch", {"url": url})
        )
        block = result["content"][0]
        spilled = Path(block["spill_path"])

        assert result["isError"] is False
        assert spilled.read_bytes() == body
        assert block["next_offset"] > 0
        page = await registry.execute(
            ToolCall(
                "fetch-page-2",
                "fetch",
                {"url": url, "offset": block["next_offset"]},
            )
        )
        assert page["isError"] is False
        assert "full readable content" in page["content"][0]["text"]
        assert page["content"][0]["text"].endswith("a")
        await registry.close()


@pytest.mark.asyncio
async def test_fetch_decompressed_large_page_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"z" * (5 * 1024 * 1024)
    response = httpx.Response(
        200,
        headers={"content-type": "text/plain", "content-encoding": "gzip"},
        stream=_ByteStream(gzip.compress(body)),
    )
    _mock_fetch(monkeypatch, response)
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=2_000,
            skill_catalog=SkillCatalog.empty(),
        )
        result = await registry.execute(
            ToolCall(
                "fetch-gzip-large",
                "fetch",
                {"url": "https://example.com", "max_bytes": 3 * 1024 * 1024},
            )
        )
        block = result["content"][0]
        spilled = Path(block["spill_path"])

        assert result["isError"] is False
        assert spilled.read_bytes() == body[: 3 * 1024 * 1024]
        assert "stopped at" in block["text"]
        assert "decompressed safety limit" in block["text"]
        await registry.close()


@pytest.mark.asyncio
async def test_fetch_truncated_gzip_returns_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"partial body " * 10_000
    compressed = gzip.compress(body)[:-8]
    response = httpx.Response(
        200,
        headers={"content-type": "text/plain", "content-encoding": "gzip"},
        stream=_ByteStream(compressed),
    )
    _mock_fetch(monkeypatch, response)
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd, session_store=store, skill_catalog=SkillCatalog.empty()
        )
        result = await registry.execute(
            ToolCall("fetch-truncated", "fetch", {"url": "https://example.com"})
        )

        assert result["isError"] is False
        assert "partial body" in result["content"][0]["text"]
        assert "compressed stream ended early" in result["content"][0]["text"]
        await registry.close()


@pytest.mark.asyncio
async def test_bash_large_output_spills(tmp_path: Path) -> None:
    output = "line\n" * 200_000
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=2_000,
            skill_catalog=SkillCatalog.empty(),
        )
        result = await registry.execute(
            ToolCall(
                "bash-large",
                "bash",
                {"command": _python_command("print('line\\n' * 200000, end='')")},
            )
        )
        block = result["content"][0]
        spilled = Path(block["spill_path"])

        assert result["isError"] is False
        assert spilled.read_text() == f"stdout:\n{output}\nstderr:\n"
        assert result["structuredContent"]["stdout"].startswith("line\n")
        await registry.close()


@pytest.mark.asyncio
async def test_mcp_large_result_spills(tmp_path: Path) -> None:
    value = "mcp" * 20_000
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=1_000,
            register_builtin=False,
            skill_catalog=SkillCatalog.empty(),
        )
        registry.register_mcp(
            "server__large",
            lambda arguments: translate_call_result(
                {"content": [{"type": "text", "text": value}], "isError": False}
            ),
            owner=object(),
            generation=1,
        )

        result = await registry.execute(
            ToolCall("mcp-large", "server__large", {})
        )

        assert Path(result["content"][0]["spill_path"]).read_text() == value
        await registry.close()


@pytest.mark.asyncio
async def test_spill_dir_bounded_evicts_oldest(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=64,
            register_builtin=False,
            skill_catalog=SkillCatalog.empty(),
        )
        registry.spills.max_bytes = 150
        registry.register("large", lambda arguments: arguments["value"])

        first = await registry.execute(
            ToolCall("first", "large", {"value": "a" * 100})
        )
        second = await registry.execute(
            ToolCall("second", "large", {"value": "b" * 100})
        )
        first_path = Path(first["content"][0]["spill_path"])
        second_path = Path(second["content"][0]["spill_path"])

        assert not first_path.exists()
        assert second_path.read_text() == "b" * 100
        assert sum(path.stat().st_size for path in second_path.parent.iterdir()) <= 150
        await registry.close()


@pytest.mark.asyncio
async def test_ephemeral_session_spill_cleanup(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path,
        max_output_chars=64,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    registry.register("large", lambda arguments: "x" * 1_000)

    result = await registry.execute(ToolCall("ephemeral", "large", {}))
    spill_path = Path(result["content"][0]["spill_path"])
    spill_root = spill_path.parent

    assert spill_path.exists()
    assert spill_root.parent != tmp_path
    await registry.close()
    assert not spill_root.exists()
