from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

import pytest

from evals.computer_macos.fixture import FixtureServer, trial_state
from evals.computer_macos.reset import reset
from evals.computer_macos.run import (
    EXPECTED_MODEL,
    EXPECTED_PROVIDER,
    MacSafari,
    SafariOwnership,
    _cleanup_trial,
    phase0,
    pilot,
)
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
        self.open_tokens = []
        self.cleanup_tokens = []

    async def open_owned(self, url):
        self.after = {"frontmost": "com.apple.Safari", "tabs": [*self.before["tabs"], {"wid": "2", "tid": "2", "u": url}]}
        token = SafariOwnership("2", "2")
        self.open_tokens.append(token)
        return token

    async def cleanup_owned(self, ownership):
        self.cleaned = True
        self.cleanup_tokens.append(ownership)
        self.after = self.before
        return {"owned_browser_state_removed": True, "owned_window_missing": False, "owned_window_tabs_changed": False, "close_failed": False, "other_app_frontmost": False, "error": None}


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
async def test_phase0_restores_environment_when_artifact_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "outer-home"))
    monkeypatch.delenv("ZETA_ACCOUNT_HOME", raising=False)
    before = dict(os.environ)

    class FailingBackend:
        async def screenshot(self):
            raise OSError("artifact write failed")

        def close(self):
            return None

    with pytest.raises(OSError, match="artifact write failed"):
        await phase0(tmp_path, FailingBackend())
    assert dict(os.environ) == before


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
    assert all(browser.cleanup_tokens == browser.open_tokens for browser in browsers)
    artifact = json.loads((tmp_path / "pilot.json").read_text())
    assert artifact["cleanup"]["disposable_home_removed"]
    assert artifact["cleanup"]["owned_browser_state_removed"]
    assert artifact["cleanup"]["owned_browser_state_removed"]


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


def test_prepare_home_restores_set_and_unset_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.computer_macos.run import prepare_home

    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "outer-home"))
    monkeypatch.delenv("ZETA_ACCOUNT_HOME", raising=False)
    before = dict(os.environ)
    try:
        with prepare_home(tmp_path) as home:
            assert os.environ["ZETA_HOME"] == str(home)
            assert os.environ["ZETA_ACCOUNT_HOME"] == str(Path.home() / ".zeta")
            raise OSError("artifact write failed")
    except OSError as exc:
        assert str(exc) == "artifact write failed"
    except TypeError:
        pytest.fail("prepare_home must be an exception-safe environment context manager")
    assert dict(os.environ) == before

    monkeypatch.setenv("ZETA_ACCOUNT_HOME", str(tmp_path / "outer-account"))
    before = dict(os.environ)
    with pytest.raises(OSError, match="artifact write failed"), prepare_home(tmp_path):
        assert os.environ["ZETA_HOME"] == str(tmp_path / "zeta-home")
        assert os.environ["ZETA_ACCOUNT_HOME"] == str(Path.home() / ".zeta")
        raise OSError("artifact write failed")
    assert dict(os.environ) == before



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "observation", "failure", "expected"),
    [
        ("closed", "owned\tcreated\n", None, {"owned_browser_state_removed": True}),
        ("gone", "\t\n", None, {"owned_window_missing": True}),
        ("changed", "other\tcreated\n", None, {"owned_window_missing": True}),
        ("extra-tab", "owned\tcreated,user\n", None, {"owned_window_tabs_changed": True}),
        ("close-fails", "owned\tcreated\n", RuntimeError("close failed"), {"close_failed": True, "error": "close failed"}),
    ],
)
async def test_safari_cleanup_production_path(name, observation, failure, expected, monkeypatch):
    token = SafariOwnership("owned", "created")
    calls = iter(["true\tcom.apple.Safari\n", observation, failure] if failure else ["true\tcom.apple.Safari\n", observation, ""])
    scripts = []

    async def fake_osascript(source):
        scripts.append(source)
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    # test-contract: seam replaces the subprocess boundary; no live Safari is allowed.
    monkeypatch.setattr(MacSafari, "_osascript", staticmethod(fake_osascript))
    result = await MacSafari().cleanup_owned(token)
    # test-contract: seam through MacSafari cleanup evidence.
    assert all(result[key] == value for key, value in expected.items()), name
    expected_calls = 3 if name in {"closed", "close-fails"} else 2
    assert len(scripts) == expected_calls
    if name == "close-fails":
        assert result["error"] == "close failed"


@pytest.mark.asyncio
async def test_safari_cleanup_executor_observes_once_then_one_call_per_action(monkeypatch: pytest.MonkeyPatch) -> None:
    token = SafariOwnership("42", "99")
    scripts: list[str] = []
    responses = iter(["true\tcom.apple.Safari\n", "42\t99\n", ""])

    async def fake_osascript(source: str) -> str:
        scripts.append(source)
        return next(responses)

    # test-contract: seam replaces the subprocess boundary; no live Safari is allowed.
    monkeypatch.setattr(MacSafari, "_osascript", staticmethod(fake_osascript))
    result = await MacSafari().cleanup_owned(token)
    assert result["owned_browser_state_removed"]
    assert len(scripts) == 3
    assert "tell application \"Safari\"" not in scripts[0]
    assert "close" not in scripts[1]
    assert all(source.count("tell application") == 1 for source in scripts[2:])


@pytest.mark.asyncio
async def test_safari_not_running_is_reported_without_targeting_safari(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_model = {"safari_targeted": False}

    async def fake_osascript(source: str) -> str:
        calls.append(source)
        if len(calls) > 1:
            fake_model["safari_targeted"] = True
        return "false\tcom.apple.Terminal\n"

    # test-contract: seam replaces the subprocess boundary; no live Safari is allowed.
    monkeypatch.setattr(MacSafari, "_osascript", staticmethod(fake_osascript))
    result = await MacSafari().cleanup_owned(SafariOwnership("42", "99"))
    assert result["owned_window_missing"]
    assert not result["owned_browser_state_removed"]
    assert len(calls) == 1
    assert not fake_model["safari_targeted"]


@pytest.mark.asyncio
async def test_cleanup_trial_preserves_browser_close_error() -> None:
    class BrowserWithCloseError:
        async def cleanup_owned(self, ownership):
            return {
                "owned_browser_state_removed": False,
                "owned_window_missing": False,
                "owned_window_tabs_changed": False,
                "close_failed": True,
                "other_app_frontmost": False,
                "error": "close failed",
            }

    class Backend:
        closed = False

        def close(self):
            self.closed = True

    evidence = await _cleanup_trial(Backend(), None, BrowserWithCloseError(), object())
    assert evidence["error"] == "close failed"
    assert evidence["close_failed"]


@pytest.mark.asyncio
async def test_timeout_persists_false_terminal_cleanup_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class HangingProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.killed = False
            self.waited = False
            self.released = asyncio.Event()

        async def communicate(self):
            await asyncio.wait_for(self.released.wait(), 10)
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.released.set()

        async def wait(self):
            self.waited = True

    process = HangingProcess()

    async def create(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    safari = MacSafari()

    class ProductionBrowser:
        async def open_owned(self, url):
            return SafariOwnership("42", "99")

        async def cleanup_owned(self, ownership):
            return await safari.cleanup_owned(ownership)

    fixture = FixtureHandle(tmp_path / "fixture", 1)
    backend = ComputerBackend()
    target = trial_state(tmp_path / "fixture", 1)["target"]
    adapter = provider(target["x"], target["y"])
    adapter.provider = EXPECTED_PROVIDER
    adapter.model = EXPECTED_MODEL
    calls = 0

    def timeout_factory(_seconds):
        nonlocal calls
        calls += 1
        return asyncio.timeout(5 if calls == 1 else 0.1)

    with pytest.raises((RuntimeError, asyncio.TimeoutError)):
        await pilot(tmp_path, provider_factory=lambda _: adapter, backend_factory=lambda _: backend, fixture_factory=lambda *_: fixture, browser_factory=ProductionBrowser, timeout_factory=timeout_factory)
    artifact = json.loads((tmp_path / "pilot.json").read_text())
    assert process.killed and process.waited and process.returncode is not None
    assert artifact["status"] in {"failed", "timeout"}
    assert artifact["cleanup"]["owned_browser_state_removed"] is False
    assert artifact["cleanup"]["errors"]


def test_reset_removes_only_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "outer-home"))
    from evals.computer_macos.run import prepare_home
    with prepare_home(tmp_path) as home:
        (home / "settings.toml").write_text("safe")
    result = reset(tmp_path, verify=True)
    assert result["pass"]
    assert not home.exists()
