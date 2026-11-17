from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from evals.computer_macos.fixture import trial_state
from evals.computer_macos.grade import grade
from evals.computer_macos.reset import reset
from evals.computer_macos.run import phase0, prepare_home
from zeta.tools.computer.backend import ScreenFrame, ScreenState


class FakeBackend:
    def __init__(self) -> None:
        self.closed = False

    async def screenshot(self, display=None):
        return ScreenFrame(b"png", "image/png", 100, 80, 2.0, 0, 0)

    async def cursor_position(self):
        return (12, 34)

    async def state(self):
        return ScreenState(False, False, "com.apple.Safari")

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_phase0_has_exact_capture_counts_and_no_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeBackend()
    original_home = os.environ["ZETA_HOME"]
    result = await phase0(tmp_path, backend)
    os.environ["ZETA_HOME"] = original_home
    os.environ.pop("ZETA_ACCOUNT_HOME", None)
    assert len(result["events"]) == 120
    assert result["input_events"] == []
    assert backend.closed


def test_fixture_and_grader_require_one_real_target_click(tmp_path: Path) -> None:
    state = trial_state(tmp_path / "fixture", 1)
    state["status"] = "PASS"
    target = state["target"]
    state["clicks"] = [{"nonce": state["nonce"], "x": target["x"] + 1, "y": target["y"] + 1}]
    (tmp_path / "fixture" / "trial-1.json").write_text(json.dumps(state))
    result = grade(tmp_path)
    assert result["pilot"]["pass"]


class FakeFixture:
    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pilot_runs_three_trials_through_provider_and_cleans_up(tmp_path: Path) -> None:
    from evals.computer_macos.run import pilot

    backends: list[FakeBackend] = []
    fixtures: list[FakeFixture] = []

    def make_backend() -> FakeBackend:
        backend = FakeBackend()
        backends.append(backend)
        return backend

    def make_fixture(_state_dir: Path, _trial: int) -> FakeFixture:
        fixture = FakeFixture()
        fixtures.append(fixture)
        return fixture

    async def fake_provider(_backend: FakeBackend, state: dict, trial: int) -> dict[str, int]:
        target = state["target"]
        state["clicks"] = [{"nonce": state["nonce"], "x": target["x"] + 1, "y": target["y"] + 1}]
        state["status"] = "PASS"
        (tmp_path / "fixture" / f"trial-{trial}.json").write_text(json.dumps(state))
        return {"provider_time_ns": 1, "actions": 1}

    result = await pilot(tmp_path, 3, provider_runner=fake_provider, backend_factory=make_backend, fixture_factory=make_fixture)
    assert result["status"] == "complete"
    assert len(result["trials"]) == 3
    assert all(backend.closed for backend in backends)
    assert all(fixture.closed for fixture in fixtures)
    assert grade(tmp_path)["pilot"]["pass"]
    assert json.loads((tmp_path / "cleanup.json").read_text())["pass"]


@pytest.mark.asyncio
async def test_pilot_cleans_up_when_provider_fails(tmp_path: Path) -> None:
    from evals.computer_macos.run import pilot

    backend = FakeBackend()
    fixture = FakeFixture()

    async def failing_provider(*_args: object) -> dict[str, int]:
        raise RuntimeError("scripted provider failure")

    with pytest.raises(RuntimeError, match="scripted provider failure"):
        await pilot(tmp_path, 3, provider_runner=failing_provider, backend_factory=lambda: backend, fixture_factory=lambda *_: fixture)
    assert backend.closed
    assert fixture.closed
    assert json.loads((tmp_path / "cleanup.json").read_text())["pass"]


def test_reset_removes_only_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "outer-home"))
    home = prepare_home(tmp_path)
    (home / "settings.toml").write_text("safe")
    result = reset(tmp_path, verify=True)
    os.environ["ZETA_HOME"] = str(tmp_path / "outer-home")
    assert result["pass"]
    assert not home.exists()
