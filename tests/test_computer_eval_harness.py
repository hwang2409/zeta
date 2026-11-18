from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from evals.computer_macos import grade as grade_module
from evals.computer_macos import run as run_module
from evals.computer_macos.fixture import FixtureServer, trial_state
from evals.computer_macos.reset import reset
from evals.computer_macos.run import EXPECTED_MODEL, EXPECTED_PROVIDER, phase0, pilot
from tests.support.fake_backend import FakeBackend as ProviderBackend
from tests.support.fake_backend import ScriptedTurn
from zeta.protocol.types import TextContent, ToolCall
from zeta.tools.computer.backend import ScreenFrame, ScreenState
from zeta.tools.computer.fake import _solid_png


class ComputerBackend:
    def __init__(self, url: str | None = None) -> None:
        self.url = url
        self.nonce = ""
        self.closed = False
        self.actions: list[str] = []

    async def screenshot(self, display=None):
        self.actions.append("screenshot")
        return ScreenFrame(_solid_png(1280, 800), "image/png", 1280, 800, 1.0, 0, 0)

    async def state(self):
        self.actions.append("state")
        return ScreenState(False, False, "com.apple.Safari")

    async def click(self, x, y, button, count):
        self.actions.append("click")
        import urllib.request
        payload = json.dumps({"nonce": self.nonce, "x": x, "y": y, "viewport": [1280, 800], "device_pixel_ratio": 1}).encode()
        request = urllib.request.Request(self.url + "/event", data=payload, method="POST", headers={"content-type": "application/json"})
        response = await asyncio.to_thread(urllib.request.urlopen, request)
        response.close()

    async def cursor_position(self):
        return (0, 0)

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

    async def close_and_verify(self) -> bool:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.closed = True
        return not self.thread.is_alive()


class Browser:
    def __init__(self):
        self.before = {"frontmost": "com.apple.Safari", "tabs": [{"wid": "1", "tid": "1", "u": "about:blank"}]}
        self.after = dict(self.before)
        self.cleaned = False

    async def snapshot(self):
        return self.before if not self.cleaned else self.after

    async def open_owned(self, url):
        self.after = {"frontmost": "com.apple.Safari", "tabs": [*self.before["tabs"], {"wid": "2", "tid": "2", "u": url}]}

    async def cleanup_owned(self, before):
        self.cleaned = True
        self.after = before
        return {"owned_browser_state_removed": True, "prior_state_restored": self.after == before}


def provider(x: int = 301, y: int = 201) -> ProviderBackend:
    return ProviderBackend([
        ScriptedTurn(tool_calls=[ToolCall("shot", "computer", {"action": "screenshot"})]),
        ScriptedTurn(tool_calls=[ToolCall("click", "computer", {"action": "left_click", "coordinate": [x + 1, y + 1]})]),
        ScriptedTurn(content=[TextContent("done")]),
        ScriptedTurn(content=[TextContent("done")]),
    ])


@pytest.mark.asyncio
async def test_phase0_has_exact_capture_counts_and_no_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ComputerBackend()
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "old"))
    result = await phase0(tmp_path, backend)
    assert len(result["events"]) == 120
    assert result["input_events"] == []
    assert backend.closed


@pytest.mark.asyncio
async def test_fake_provider_crosses_public_loop_three_trials_and_cleans_each_trial(tmp_path: Path) -> None:
    backends: list[ComputerBackend] = []
    fixtures: list[FixtureHandle] = []
    browsers: list[Browser] = []

    def make_fixture(state_dir, trial):
        fixture = FixtureHandle(state_dir, trial)
        fixtures.append(fixture)
        return fixture

    def make_backend(_trial):
        backend = ComputerBackend()
        backend.nonce = ""
        backends.append(backend)
        return backend

    def make_provider(trial):
        target = trial_state(tmp_path / "fixture", trial)["target"]
        adapter = provider(target["x"], target["y"])
        adapter.provider = EXPECTED_PROVIDER
        adapter.model = EXPECTED_MODEL
        return adapter

    result = await pilot(tmp_path, provider_factory=make_provider, backend_factory=make_backend, fixture_factory=make_fixture, browser_factory=lambda: browsers.append(Browser()) or browsers[-1])
    assert result["status"] == "complete"
    assert len(result["trials"]) == 3
    assert all(row["grade"]["pass"] for row in result["trials"])
    assert all(row["provider"] == EXPECTED_PROVIDER and row["model"] == EXPECTED_MODEL for row in result["trials"])
    assert all(backend.actions.count("click") == 1 for backend in backends)
    assert all(backend.closed for backend in backends)
    assert all(fixture.closed for fixture in fixtures)
    assert all(browser.cleaned for browser in browsers)
    assert json.loads((tmp_path / "pilot.json").read_text())["cleanup"]["disposable_home_removed"]


@pytest.mark.asyncio
async def test_provider_failure_writes_terminal_truth_and_cleanup(tmp_path: Path) -> None:
    fixture = FixtureHandle(tmp_path / "fixture", 1)
    backend = ComputerBackend()

    def failing(_trial):
        adapter = ProviderBackend([])
        adapter.provider = EXPECTED_PROVIDER
        adapter.model = EXPECTED_MODEL
        async def complete(*args, **kwargs):
            raise RuntimeError("provider failed")
            yield
        adapter.complete = complete
        return adapter

    with pytest.raises(RuntimeError):
        await pilot(tmp_path, provider_factory=failing, backend_factory=lambda _trial: backend, fixture_factory=lambda *_: fixture, browser_factory=lambda: Browser())
    artifact = json.loads((tmp_path / "pilot.json").read_text())
    assert artifact["status"] == "failed"
    assert artifact["cleanup"]["disposable_home_removed"]
    assert backend.closed and fixture.closed


def test_reset_removes_only_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "outer-home"))
    from evals.computer_macos.run import prepare_home
    home = prepare_home(tmp_path)
    (home / "settings.toml").write_text("safe")
    result = reset(tmp_path, verify=True)
    assert result["pass"]
    assert not home.exists()


def test_pilot_cli_accepts_only_exact_three_without_running_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    def fake_pilot(*args, **kwargs):
        captured.append(kwargs["trials"])
        async def done():
            return {}
        return done()

    monkeypatch.setattr(run_module, "pilot", fake_pilot)
    monkeypatch.setattr(run_module.asyncio, "run", lambda coroutine: coroutine.close())
    monkeypatch.setattr("sys.argv", ["run", "pilot", "--run-dir", "/tmp/run", "--trials", "3"])
    run_module.main()
    assert captured == [3]
    monkeypatch.setattr("sys.argv", ["run", "pilot", "--run-dir", "/tmp/run", "--trials", "2"])
    with pytest.raises(SystemExit):
        run_module.main()


def test_grade_preserves_phase0_and_fails_incomplete_or_bad_pilot(tmp_path: Path) -> None:
    (tmp_path / "events.jsonl").write_text(json.dumps({"action": "screenshot"}) + "\n")
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    state = trial_state(fixture, 1)
    (fixture / "trial-1.json").write_text(json.dumps(state))
    result = grade_module.grade(tmp_path)
    assert result["phase0"]["pass"]
    assert result["pilot"]["pass"] is False
    assert result["pilot"]["trials"][0]["pass"] is False
