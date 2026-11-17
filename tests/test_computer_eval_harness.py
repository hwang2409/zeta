from __future__ import annotations

import asyncio
import json
import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from evals.computer_macos.fixture import FixtureServer, trial_state
from evals.computer_macos.reset import reset
from evals.computer_macos.run import (
    EXPECTED_MODEL,
    EXPECTED_PROVIDER,
    phase0,
    pilot,
    prepare_home,
)
from zeta.config.settings import ComputerUseConfig
from zeta.media.image_policy import ANTHROPIC_IMAGE_POLICY
from zeta.tools.computer import run_action
from zeta.tools.computer.backend import ScreenFrame, ScreenState

_image = BytesIO()
Image.new("RGB", (1280, 800), "white").save(_image, format="PNG")
_PNG = _image.getvalue()


class FakeBackend:
    def __init__(self, url: str | None = None) -> None:
        self.closed = False
        self.url = url
        self.actions: list[str] = []

    async def screenshot(self, display=None):
        self.actions.append("screenshot")
        return ScreenFrame(_PNG, "image/png", 1280, 800, 1.0, 0, 0)

    async def cursor_position(self):
        return (12, 34)

    async def state(self):
        return ScreenState(False, False, "com.apple.Safari")

    async def click(self, x, y, button, count):
        self.actions.append("click")
        data = json.dumps({"nonce": self.nonce, "x": x, "y": y, "viewport": [1280, 800], "device_pixel_ratio": 1}).encode()
        import urllib.request
        request = urllib.request.Request(self.url + "/event", data=data, method="POST", headers={"content-type": "application/json"})
        await asyncio.to_thread(urllib.request.urlopen, request)

    def close(self):
        self.closed = True


class FixtureHandle:
    def __init__(self, state_dir: Path, trial: int) -> None:
        self.server = FixtureServer(("127.0.0.1", 0), state_dir, trial)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        trial_state(state_dir, trial)
        self.closed = False

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.closed = True


@pytest.mark.asyncio
async def test_phase0_has_exact_capture_counts_and_no_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeBackend()
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "old"))
    result = await phase0(tmp_path, backend)
    assert len(result["events"]) == 120
    assert result["input_events"] == []
    assert backend.closed


@pytest.mark.asyncio
async def test_pilot_runs_public_computer_tool_loop_three_trials_and_writes_truth(tmp_path: Path) -> None:
    backends: list[FakeBackend] = []
    fixtures: list[FixtureHandle] = []

    def make_fixture(state_dir: Path, trial: int) -> FixtureHandle:
        fixture = FixtureHandle(state_dir, trial)
        fixtures.append(fixture)
        return fixture

    def make_backend() -> FakeBackend:
        backend = FakeBackend()
        backends.append(backend)
        return backend

    async def scripted_provider(backend: FakeBackend, state: dict, trial: int) -> dict:
        backend.url = fixtures[trial - 1].url
        backend.nonce = state["nonce"]
        registry = SimpleNamespace(computer_enabled=True, computer_backend=backend, computer_config=ComputerUseConfig(enabled=True, default_app_access="allow"), computer_geometry=None, image_policy=ANTHROPIC_IMAGE_POLICY)
        screenshot = await run_action(registry, "screenshot", {})
        assert not screenshot["isError"], screenshot
        target = state["target"]
        click = await run_action(registry, "left_click", {"coordinate": [target["x"] + 1, target["y"] + 1]})
        assert not click["isError"]
        return {"provider": EXPECTED_PROVIDER, "model": EXPECTED_MODEL, "provider_events": 2, "usage_tokens": 7, "computer_actions": ["screenshot", "left_click"]}

    result = await pilot(tmp_path, 3, provider_runner=scripted_provider, backend_factory=make_backend, fixture_factory=make_fixture)
    assert result["status"] == "complete"
    assert len(result["trials"]) == 3
    assert all(trial["grade"]["pass"] for trial in result["trials"])
    assert all(backend.actions == ["screenshot", "click", "screenshot"] for backend in backends)
    assert all(backend.closed for backend in backends)
    assert all(fixture.closed for fixture in fixtures)
    assert json.loads((tmp_path / "pilot.json").read_text())["cleanup"]["pass"]


@pytest.mark.asyncio
async def test_pilot_writes_truth_on_provider_failure_and_cancellation(tmp_path: Path) -> None:
    fixture = FixtureHandle(tmp_path / "fixture", 1)
    backend = FakeBackend()

    async def failing(*_args: object) -> dict:
        raise RuntimeError("scripted provider failure")

    with pytest.raises(RuntimeError, match="scripted provider failure"):
        await pilot(tmp_path, 3, provider_runner=failing, backend_factory=lambda: backend, fixture_factory=lambda *_: fixture)
    artifact = json.loads((tmp_path / "pilot.json").read_text())
    assert artifact["status"] == "failed"
    assert artifact["error"] == "scripted provider failure"
    assert backend.closed and fixture.closed


@pytest.mark.asyncio
async def test_pilot_writes_truth_on_cancellation_and_partial_startup(tmp_path: Path) -> None:
    fixture = FixtureHandle(tmp_path / "fixture", 1)

    async def cancelled(*_args: object) -> dict:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await pilot(tmp_path, 3, provider_runner=cancelled, backend_factory=FakeBackend, fixture_factory=lambda *_: fixture)
    assert json.loads((tmp_path / "pilot.json").read_text())["status"] == "cancelled"
    assert json.loads((tmp_path / "cleanup.json").read_text())["pass"]

    partial = tmp_path / "partial"
    def no_fixture(*_args: object) -> object:
        raise RuntimeError("fixture startup failed")
    with pytest.raises(RuntimeError, match="fixture startup failed"):
        await pilot(partial, 3, fixture_factory=no_fixture)
    artifact = json.loads((partial / "pilot.json").read_text())
    assert artifact["status"] == "failed"
    assert artifact["error"] == "fixture startup failed"


def test_reset_removes_only_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "outer-home"))
    home = prepare_home(tmp_path)
    (home / "settings.toml").write_text("safe")
    result = reset(tmp_path, verify=True)
    assert result["pass"]
    assert not home.exists()
