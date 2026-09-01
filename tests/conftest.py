from __future__ import annotations

import os
from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from rich.console import Console


LIVE_ZETA_HOME = Path.home() / ".zeta"


def _persistence_snapshot() -> dict[Path, tuple[object, ...]]:
    paths = [LIVE_ZETA_HOME / "history"]
    sessions_dir = LIVE_ZETA_HOME / "sessions"
    if sessions_dir.is_dir():
        paths.extend(sessions_dir.glob("*/draft"))
    snapshot: dict[Path, tuple[object, ...]] = {}
    for path in paths:
        try:
            metadata = path.stat()
            content = sha256(path.read_bytes()).digest()
        except FileNotFoundError:
            continue
        snapshot[path] = (
            metadata.st_mode,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            content,
        )
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def isolate_zeta_home(tmp_path_factory: pytest.TempPathFactory) -> Generator[None, None, None]:
    """Keep every test away from the developer's real zeta home."""

    live_persistence = _persistence_snapshot()
    isolated_home = tmp_path_factory.mktemp("zeta-home")
    fake_home = isolated_home.parent / "home"
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("ZETA_HOME", str(isolated_home))
        monkeypatch.setenv("HOME", str(fake_home))
        yield
        assert Path(os.environ["ZETA_HOME"]) != LIVE_ZETA_HOME
        assert _persistence_snapshot() == live_persistence, (
            "tests changed live zeta history or draft state"
        )


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
