from __future__ import annotations

import httpx
import pytest


@pytest.fixture(autouse=True)
def block_real_http_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_sync(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        del self, request
        raise AssertionError("real network connections are forbidden in tests")

    async def blocked_async(
        self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        del self, request
        raise AssertionError("real network connections are forbidden in tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked_sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
