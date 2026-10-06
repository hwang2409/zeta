from __future__ import annotations

import ast
import asyncio
import gzip
import io
import shlex
import stat
import sys
import threading
import tracemalloc
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
from zeta.tools import browser as browser_tool
from zeta.tools import fetch as fetch_tool
from zeta.tools._spill import SpillStore

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


class _InterleavedReader(io.BytesIO):
    def __init__(self, started: threading.Event, resume: threading.Event) -> None:
        super().__init__(b"a" * 100)
        self._started = started
        self._resume = resume
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        result = super().read(size)
        if self._reads == 1:
            return result
        self._started.set()
        assert self._resume.wait(timeout=5)
        return result


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


def test_no_handler_pretruncates() -> None:
    roots = [Path("src/zeta/tools"), Path("src/zeta/mcp")]
    capped_calls: list[str] = []
    for root in roots:
        for source_path in root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (
                    isinstance(node.func, ast.Name) and node.func.id == "text_block"
                ):
                    continue
                if any(keyword.arg == "cap" for keyword in node.keywords):
                    capped_calls.append(f"{source_path}:{node.lineno}")

    assert capped_calls == []


@pytest.mark.asyncio
async def test_browser_large_result_spills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = "browser output " * 10_000

    class FakeProxy:
        denial_count = 0

        def denials_since(self, _start: int):
            return []

    class FakePage:
        url = "https://example.com"

        async def title(self) -> str:
            return "Large"

        async def aria_snapshot(self, **_kwargs: object) -> str:
            return snapshot

    class FakeSession:
        def __init__(self) -> None:
            self.lock = asyncio.Lock()
            self.proxy = FakeProxy()
            self.page = FakePage()

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(browser_tool, "_BrowserSession", FakeSession)
    monkeypatch.setenv("ZETA_BROWSER", "1")
    registry = ToolRegistry(
        tmp_path,
        max_output_chars=1_000,
        skill_catalog=SkillCatalog.empty(),
    )
    try:
        result = await registry.execute(
            ToolCall("browser-large", "browser", {"action": "snapshot"})
        )
        block = result["content"][0]
        spill_path = Path(block["spill_path"])

        assert result["isError"] is False
        assert spill_path.read_text().endswith(snapshot)
        assert block["full_size"] == len(spill_path.read_bytes())
    finally:
        await registry.close()


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
async def test_restricted_spill_reads_are_session_anchored(tmp_path: Path) -> None:
    with _store(tmp_path, "parent") as parent, _store(tmp_path, "other") as other:
        producer = ToolRegistry(
            parent.cwd,
            session_store=parent,
            max_output_chars=64,
            register_builtin=False,
            skill_catalog=SkillCatalog.empty(),
        )
        producer.register("large", lambda arguments: "parent secret" * 100)
        produced = await producer.execute(ToolCall("large", "large", {}))
        spill_path = Path(produced["content"][0]["spill_path"])

        restricted_cwd = tmp_path / "restricted-cwd"
        restricted_cwd.mkdir()
        restricted = ToolRegistry(
            restricted_cwd,
            session_store=parent,
            tool_allow=["read"],
            skill_catalog=SkillCatalog.empty(),
        )
        own = await restricted.execute(
            ToolCall("own", "read", {"path": str(spill_path)})
        )
        dotdot = await restricted.execute(
            ToolCall(
                "dotdot",
                "read",
                {
                    "path": str(spill_path.parent / ".." / "conversation.jsonl")
                },
            )
        )

        symlink_path = spill_path.parent / "symlink.txt"
        symlink_path.symlink_to(tmp_path / "outside.txt")
        symlink = await restricted.execute(
            ToolCall("symlink", "read", {"path": str(symlink_path)})
        )

        outside = tmp_path / "hardlink-source.txt"
        outside.write_text("outside")
        hardlink_path = spill_path.parent / "hardlink.txt"
        hardlink_path.hardlink_to(outside)
        hardlink = await restricted.execute(
            ToolCall("hardlink", "read", {"path": str(hardlink_path)})
        )

        other_registry = ToolRegistry(
            other.cwd,
            session_store=other,
            max_output_chars=64,
            register_builtin=False,
            skill_catalog=SkillCatalog.empty(),
        )
        other_registry.register("large", lambda arguments: "other secret" * 100)
        other_result = await other_registry.execute(ToolCall("other", "large", {}))
        other_spill = other_result["content"][0]["spill_path"]
        other_session = await restricted.execute(
            ToolCall("other-session", "read", {"path": other_spill})
        )

        child_root = parent.session_dir / "agents"
        with ConversationStore(
            child_root,
            session_id="child",
            cwd=restricted_cwd,
            bash_cwd=restricted_cwd,
        ) as child:
            child_registry = ToolRegistry(
                restricted_cwd,
                session_store=child,
                tool_allow=["read"],
                skill_catalog=SkillCatalog.empty(),
            )
            child_to_parent = await child_registry.execute(
                ToolCall("parent-spill", "read", {"path": str(spill_path)})
            )
            await child_registry.close()

        assert own["isError"] is False
        assert "parent secret" in own["content"][0]["text"]
        for name, denied in {
            "dotdot": dotdot,
            "symlink": symlink,
            "hardlink": hardlink,
            "other-session": other_session,
            "child-to-parent": child_to_parent,
        }.items():
            assert denied["isError"] is True, (name, denied)
        await other_registry.close()
        await restricted.close()
        await producer.close()


@pytest.mark.asyncio
async def test_fetch_large_body_bounded_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body_size = 24 * 1024 * 1024
    body = b"m" * body_size
    response = httpx.Response(
        200,
        headers={"content-type": "text/plain"},
        stream=_ByteStream(body),
    )
    _mock_fetch(monkeypatch, response)
    with _store(tmp_path) as store:
        registry = ToolRegistry(
            store.cwd,
            session_store=store,
            max_output_chars=2_000,
            skill_catalog=SkillCatalog.empty(),
        )
        tracemalloc.start()
        try:
            result = await registry.execute(
                ToolCall(
                    "fetch-memory",
                    "fetch",
                    {"url": "https://example.com", "max_bytes": body_size},
                )
            )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        block = result["content"][0]
        assert result["isError"] is False
        assert Path(block["spill_path"]).stat().st_size == body_size
        assert peak < body_size // 2
        await registry.close()


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


def test_concurrent_spill_stores_do_not_break_each_other(tmp_path: Path) -> None:
    with _store(tmp_path) as store:
        first = SpillStore(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            max_bytes=50,
        )
        second = SpillStore(
            session_dir=store.session_dir,
            directory_fd=store.directory_fd,
            max_bytes=50,
        )
        started = threading.Event()
        resume = threading.Event()
        failures: list[OSError] = []

        def write_first() -> None:
            try:
                first.write_parts(
                    "first", "call", 0, [_InterleavedReader(started, resume)]
                )
            except OSError as exc:
                failures.append(exc)

        second_paths: list[Path] = []

        def write_second() -> None:
            try:
                second_paths.append(
                    second.write_bytes("second", "call", 0, b"b" * 100)
                )
            except OSError as exc:
                failures.append(exc)

        first_thread = threading.Thread(target=write_first)
        second_thread = threading.Thread(target=write_second)
        first_thread.start()
        assert started.wait(timeout=5)
        second_thread.start()
        try:
            assert second_thread.is_alive()
        finally:
            resume.set()
            first_thread.join(timeout=5)
            second_thread.join(timeout=5)
            first.close()
            second.close()

        assert not first_thread.is_alive()
        assert not second_thread.is_alive()
        assert failures == []
        assert len(second_paths) == 1
        assert second_paths[0].name.endswith(".txt")


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
    spill_root = registry.spills.root

    assert not spill_root.exists()
    result = await registry.execute(ToolCall("ephemeral", "large", {}))
    spill_path = Path(result["content"][0]["spill_path"])

    assert spill_path.parent == spill_root
    assert spill_path.exists()
    assert spill_root.parent != tmp_path
    await registry.close()
    assert not spill_root.exists()
