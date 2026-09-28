"""One live page check for the opt-in browser tool."""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import zeta.tools.browser as browser_module
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


@pytest.mark.asyncio
async def test_browser_proxy_uses_first_validated_address_after_dns_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(
        [
            (ip_address("203.0.113.10"),),
            (ip_address("203.0.113.11"),),
        ]
    )
    calls = 0

    def resolve(_url: str):
        nonlocal calls
        calls += 1
        return next(answers)

    connected: list[tuple[str, int]] = []

    async def connect(address: str, port: int):
        connected.append((address, port))
        return object(), object()

    proxy = browser_module._PinnedProxy()
    monkeypatch.setattr(browser_module, "_target_addresses", resolve)
    monkeypatch.setattr(browser_module.asyncio, "open_connection", connect)
    await proxy._connect("example.test", 443, "https")
    await proxy._connect("example.test", 443, "https")

    assert calls == 1
    assert connected == [("203.0.113.10", 443)] * 2


@pytest.mark.asyncio
async def test_browser_proxy_blocks_metadata_address_in_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = browser_module._PinnedProxy()
    monkeypatch.setattr(
        browser_module,
        "_target_addresses",
        lambda _url: (ip_address("169.254.169.254"),),
    )
    await proxy.start()
    assert proxy.server is not None and proxy.server.sockets
    proxy_port = proxy.server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    try:
        writer.write(
            b"GET http://metadata.test/ HTTP/1.1\r\nHost: metadata.test\r\n\r\n"
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=3)
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.close()

    assert response.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert proxy.get("metadata.test", 80) is None


@pytest.mark.asyncio
async def test_browser_proxy_forwards_plain_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[bytes] = []

    async def upstream(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n"
                b"hello proxy"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    upstream_server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    assert upstream_server.sockets
    upstream_port = upstream_server.sockets[0].getsockname()[1]
    proxy = browser_module._PinnedProxy()
    monkeypatch.setattr(
        browser_module,
        "_target_addresses",
        lambda _url: (ip_address("127.0.0.1"),),
    )
    await proxy.start()
    assert proxy.server is not None and proxy.server.sockets
    proxy_port = proxy.server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    try:
        writer.write(
            f"GET http://origin.test:{upstream_port}/path?query=1 HTTP/1.1\r\n"
            f"Host: origin.test:{upstream_port}\r\n\r\n".encode()
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=3)
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.close()
        upstream_server.close()
        await upstream_server.wait_closed()

    assert response.endswith(b"hello proxy")
    assert requests == [
        f"GET /path?query=1 HTTP/1.1\r\nHost: origin.test:{upstream_port}\r\n\r\n".encode()
    ]


@pytest.mark.asyncio
async def test_browser_proxy_validates_cross_origin_https_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[bytes] = []

    async def upstream(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            requests.append(request)
            if b"GET /start " in request:
                response = (
                    b"HTTP/1.1 302 Found\r\n"
                    + f"Location: https://secure.test:{upstream_port}/secure\r\n".encode()
                    + b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            else:
                response = (
                    b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n"
                    b"Connection: close\r\n\r\nsecure hello"
                )
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    upstream_server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    assert upstream_server.sockets
    upstream_port = upstream_server.sockets[0].getsockname()[1]
    proxy = browser_module._PinnedProxy()
    resolved: list[str] = []

    def resolve(url: str):
        resolved.append(url)
        return (ip_address("127.0.0.1"),)

    monkeypatch.setattr(browser_module, "_target_addresses", resolve)
    await proxy.start()
    assert proxy.server is not None and proxy.server.sockets
    proxy_port = proxy.server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(
            f"GET http://redirect.test:{upstream_port}/start HTTP/1.1\r\n"
            f"Host: redirect.test:{upstream_port}\r\n\r\n".encode()
        )
        await writer.drain()
        redirect = await reader.read()
        writer.close()
        await writer.wait_closed()

        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(
            f"CONNECT secure.test:{upstream_port} HTTP/1.1\r\n"
            f"Host: secure.test:{upstream_port}\r\n\r\n".encode()
        )
        await writer.drain()
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(
            b"HTTP/1.1 200 Connection Established"
        )
        writer.write(
            f"GET /secure HTTP/1.1\r\nHost: secure.test:{upstream_port}\r\n\r\n".encode()
        )
        await writer.drain()
        secure = await reader.read()
        writer.close()
        await writer.wait_closed()
    finally:
        await proxy.close()
        upstream_server.close()
        await upstream_server.wait_closed()

    assert b"Location: https://secure.test:" in redirect
    assert secure.endswith(b"secure hello")
    assert resolved == [
        f"http://redirect.test:{upstream_port}/",
        f"https://secure.test:{upstream_port}/",
    ]
    assert b"GET /start HTTP/1.1" in requests[0]
    assert b"GET /secure HTTP/1.1" in requests[1]


@pytest.mark.asyncio
async def test_browser_proxy_validates_first_use_websocket_in_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[bytes] = []

    async def upstream(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 101 Switching Protocols\r\n"
                b"Connection: Upgrade\r\nUpgrade: websocket\r\n\r\n"
            )
            await writer.drain()
            writer.write(await reader.readexactly(5))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    upstream_server = await asyncio.start_server(upstream, "127.0.0.1", 0)
    assert upstream_server.sockets
    upstream_port = upstream_server.sockets[0].getsockname()[1]
    proxy = browser_module._PinnedProxy()
    resolved: list[str] = []

    def resolve(url: str):
        resolved.append(url)
        return (ip_address("127.0.0.1"),)

    monkeypatch.setattr(browser_module, "_target_addresses", resolve)
    await proxy.start()
    assert proxy.server is not None and proxy.server.sockets
    proxy_port = proxy.server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    try:
        writer.write(
            (
                f"GET ws://socket.test:{upstream_port}/chat HTTP/1.1\r\n"
                f"Host: socket.test:{upstream_port}\r\n"
                "Connection: Upgrade\r\nUpgrade: websocket\r\n\r\n"
            ).encode()
        )
        await writer.drain()
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(
            b"HTTP/1.1 101 Switching Protocols"
        )
        writer.write(b"hello")
        await writer.drain()
        assert await reader.readexactly(5) == b"hello"
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.close()
        upstream_server.close()
        await upstream_server.wait_closed()

    assert resolved == [f"ws://socket.test:{upstream_port}/"]
    assert b"GET /chat HTTP/1.1" in requests[0]


def test_browser_is_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZETA_BROWSER", raising=False)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    assert "browser" not in registry.registered_names
    monkeypatch.setenv("ZETA_BROWSER", "1")
    enabled = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    assert "browser" in enabled.registered_names


@pytest.mark.asyncio
async def test_browser_clicks_live_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("playwright.async_api")
    monkeypatch.setenv("ZETA_BROWSER", "1")

    def resolve(url: str):
        if urlsplit(url).hostname in {"blocked.test", "169.254.169.254"}:
            return (ip_address("169.254.169.254"),)
        return (ip_address("127.0.0.1"),)

    monkeypatch.setattr(browser_module, "_target_addresses", resolve)

    class Page(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = (
                b"""
                <article aria-label="First">
                  <button onclick="setTimeout(() => document.getElementById('answer').textContent='clicked', 50)">Reveal</button>
                </article>
                <script src="http://blocked.test/blocked.js"></script>
                <p id="answer"></p>
                <article aria-label="Second">
                  <button onclick="document.getElementById('answer').textContent='second'">Reveal</button>
                </article>
                <input aria-label="Entry" onkeydown="if (event.key === 'Enter') document.getElementById('answer').textContent='submitted'">
                <a href="#target">Jump</a>
            """
                + b"<div>"
                + b"noise " * 3000
                + b"</div><section aria-label='Target area'><h3 id='target'>Target section</h3><p>Expected detail</p></section>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Page)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    try:
        blocked = await registry.execute(
            ToolCall(
                "blocked",
                "browser",
                {"action": "open", "url": "http://169.254.169.254/"},
            )
        )
        if (
            blocked["isError"]
            and "Executable doesn't exist" in blocked["content"][0]["text"]
        ):
            pytest.skip("Playwright Chromium binary is not installed")
        assert blocked["isError"]
        assert "metadata" in blocked["content"][0]["text"]
        opened = await registry.execute(
            ToolCall(
                "open",
                "browser",
                {"action": "open", "url": f"http://127.0.0.1:{server.server_port}/"},
            )
        )
        if (
            opened["isError"]
            and "Executable doesn't exist" in opened["content"][0]["text"]
        ):
            pytest.skip("Playwright Chromium binary is not installed")
        assert not opened["isError"], opened
        assert 'button "Reveal"' in opened["content"][0]["text"]
        assert "Expected detail" not in opened["content"][0]["text"]
        assert "Proxy denials:" in opened["content"][0]["text"]
        assert "refusing cloud metadata target" in opened["content"][0]["text"]
        found = await registry.execute(
            ToolCall("find", "browser", {"action": "find", "text": "Expected detail"})
        )
        assert not found["isError"], found
        assert 'region "Target area"' in found["content"][0]["text"]
        assert "Expected detail" in found["content"][0]["text"]
        assert len(found["content"][0]["text"]) < 2000
        ambiguous = await registry.execute(
            ToolCall(
                "ambiguous",
                "browser",
                {"action": "click", "role": "button", "name": "Reveal"},
            )
        )
        assert ambiguous["isError"]
        assert "index" in ambiguous["content"][0]["text"]
        clicked = await registry.execute(
            ToolCall(
                "click",
                "browser",
                {"action": "click", "role": "button", "name": "Reveal", "index": 0},
            )
        )
        assert not clicked["isError"], clicked
        assert "clicked" in clicked["content"][0]["text"]
        scoped = await registry.execute(
            ToolCall(
                "scoped",
                "browser",
                {
                    "action": "click",
                    "within_role": "article",
                    "within_name": "Second",
                    "role": "button",
                    "name": "Reveal",
                },
            )
        )
        assert not scoped["isError"], scoped
        assert "second" in scoped["content"][0]["text"]
        ambiguous_scope = await registry.execute(
            ToolCall(
                "ambiguous-scope",
                "browser",
                {
                    "action": "click",
                    "within_role": "article",
                    "role": "button",
                    "name": "Reveal",
                },
            )
        )
        assert ambiguous_scope["isError"]
        assert "one matching container" in ambiguous_scope["content"][0]["text"]
        uppercase_enter = await registry.execute(
            ToolCall(
                "uppercase-enter",
                "browser",
                {"action": "press", "role": "textbox", "name": "Entry", "key": "ENTER"},
            )
        )
        assert not uppercase_enter["isError"], uppercase_enter
        assert "submitted" in uppercase_enter["content"][0]["text"]
        batched = await registry.execute(
            ToolCall(
                "batch",
                "browser",
                {
                    "action": "batch",
                    "url": f"http://127.0.0.1:{server.server_port}/",
                    "steps": [
                        {
                            "action": "click",
                            "role": "button",
                            "name": "Reveal",
                            "index": 0,
                        },
                        {
                            "action": "click",
                            "role": "button",
                            "name": "Reveal",
                            "index": 1,
                        },
                    ],
                },
            )
        )
        assert not batched["isError"], batched
        assert "second" in batched["content"][0]["text"]
        jumped = await registry.execute(
            ToolCall(
                "jump", "browser", {"action": "click", "role": "link", "name": "Jump"}
            )
        )
        assert not jumped["isError"], jumped
        assert (
            "Anchor section: Target section | Expected detail"
            in jumped["content"][0]["text"]
        )
        found_after_jump = await registry.execute(
            ToolCall(
                "find-after-jump",
                "browser",
                {"action": "find", "text": "Expected detail"},
            )
        )
        assert (
            "Anchor section: Target section | Expected detail"
            in found_after_jump["content"][0]["text"]
        )
    finally:
        await registry.close()
        server.shutdown()
        server.server_close()
        thread.join()
