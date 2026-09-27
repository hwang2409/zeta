"""One live page check for the opt-in browser tool."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


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

    class Page(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"""
                <article aria-label="First">
                  <button onclick="setTimeout(() => document.getElementById('answer').textContent='clicked', 50)">Reveal</button>
                </article>
                <p id="answer"></p>
                <article aria-label="Second">
                  <button onclick="document.getElementById('answer').textContent='second'">Reveal</button>
                </article>
            """
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
    finally:
        await registry.close()
        server.shutdown()
        server.server_close()
        thread.join()
