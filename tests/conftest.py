from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from rich.console import Console


LIVE_ZETA_HOME = Path.home() / ".zeta"


@pytest.fixture(scope="session", autouse=True)
def isolate_zeta_home(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Keep every test away from the developer's real zeta home."""

    isolated_home = tmp_path_factory.mktemp("zeta-home")
    fake_home = isolated_home.parent / "home"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("ZETA_HOME", str(isolated_home))
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        yield
        assert Path(os.environ["ZETA_HOME"]) != LIVE_ZETA_HOME
        assert Path.home() == fake_home


def test_test_home_is_isolated() -> None:
    assert Path(os.environ["ZETA_HOME"]) != LIVE_ZETA_HOME


@pytest.fixture(autouse=True)
def stable_terminal_defaults(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Keep Rich output stable unless a test explicitly probes the environment."""
    if request.node.get_closest_marker("environment_sensitive") is not None:
        yield
        return

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert os.environ["TERM"] == "xterm-256color"
    assert os.environ["COLORTERM"] == "truecolor"

    original_init = Console.__init__

    def init_with_stable_defaults(self: Console, *args: object, **kwargs: object) -> None:
        if kwargs.get("force_terminal") is not False:
            kwargs.setdefault("force_terminal", True)
            kwargs.setdefault("color_system", "truecolor")
            kwargs.setdefault("no_color", False)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Console, "__init__", init_with_stable_defaults)

    yield


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
