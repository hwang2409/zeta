"""Unit tests for opt-in computer-use features."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "prototypes/computer-mcp"


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = _module("backend", "backend.py")
features = _module("features", "features.py")
server = _module("computer_feature_server", "server.py")


class FakeBackend:
    def __init__(self, fail_at: int | None = None) -> None:
        self.actions = []
        self.fail_at = fail_at
        self.screenshot_options = []
        self.settle_calls = 0

    def start(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def destroy(self) -> None:
        pass

    def input(self, action, arguments) -> None:
        if len(self.actions) == self.fail_at:
            raise RuntimeError("input failed")
        self.actions.append((action, arguments))

    def screenshot(self, *, crop=None, cursor=False):
        self.screenshot_options.append((crop, cursor))
        return backend.Screenshot(b"frame", "image/jpeg")

    def observe(self):
        return {
            "active_window": {
                "title": "Editor",
                "bounds": {"x": 1, "y": 2, "w": 3, "h": 4},
            },
            "windows": [],
            "focused_widget": None,
            "mouse": {"x": 12, "y": 34},
        }

    def settle(self) -> float:
        self.settle_calls += 1
        return 0.125


def test_default_tool_list_is_exactly_legacy_behavior(monkeypatch) -> None:
    monkeypatch.delenv("ZETA_COMPUTER_FEATURES", raising=False)
    computer = server.ComputerServer(FakeBackend())
    assert [tool["name"] for tool in computer.tools] == [
        "computer_screenshot",
        "computer_click",
        "computer_double_click",
        "computer_drag",
        "computer_type",
        "computer_key",
        "computer_scroll",
        "computer_wait",
    ]


def test_batch_validates_every_coordinate_before_any_action() -> None:
    fake = FakeBackend()
    computer = server.ComputerServer(fake, frozenset({"batch"}))
    result = computer.call(
        "computer_batch",
        {
            "actions": [
                {"type": "click", "x": 10, "y": 20},
                {"type": "drag", "x1": 0, "y1": 0, "x2": 1024, "y2": 20},
            ]
        },
    )
    assert result["isError"] is True
    assert "actions[1]" in result["content"][0]["text"]
    assert fake.actions == []
    assert fake.screenshot_options == []


def test_batch_stops_on_runtime_error_and_returns_one_final_screenshot() -> None:
    fake = FakeBackend(fail_at=1)
    result = server.ComputerServer(fake, frozenset({"batch"})).call(
        "computer_batch",
        {
            "actions": [
                {"type": "click", "x": 1, "y": 2},
                {"type": "type", "text": "hello"},
                {"type": "key", "keys": "Return"},
            ]
        },
    )
    assert result["isError"] is True
    batch = json.loads(result["content"][1]["text"].removeprefix("Batch results: "))
    assert [item["status"] for item in batch] == ["ok", "error"]
    assert fake.actions == [("click", {"x": 1, "y": 2})]
    assert fake.screenshot_options == [(None, False)]


def test_zoom_coordinate_math_and_global_coordinate_legend() -> None:
    arguments = {"x": 256, "y": 160, "w": 512, "h": 320}
    assert features.model_crop(arguments) == (320, 200, 640, 400)
    legend = features.zoom_legend(arguments)
    assert "GLOBAL" in legend
    assert "global_x=256+u*512/1024" in legend
    with pytest.raises(ValueError, match="must fit"):
        features.model_crop({"x": 900, "y": 0, "w": 200, "h": 10})


def test_zoom_and_cursor_are_passed_to_screenshot_backend() -> None:
    fake = FakeBackend()
    result = server.ComputerServer(fake, frozenset({"zoom", "cursor"})).call(
        "computer_zoom", {"x": 256, "y": 160, "w": 512, "h": 320}
    )
    assert result["isError"] is False
    assert fake.screenshot_options == [((320, 200, 640, 400), True)]


def test_observation_format_tracks_screen_hash_without_clipboard() -> None:
    raw = FakeBackend().observe()
    first, first_hash = features.format_observation(raw, b"first", None)
    second, second_hash = features.format_observation(raw, b"second", first_hash)
    first_value, second_value = json.loads(first), json.loads(second)
    assert first_value["screen_change"]["changed"] is None
    assert second_value["screen_change"] == {
        "changed": True,
        "hash": second_hash,
        "previous_hash": first_hash,
    }
    assert second_value["active_window"]["title"] == "Editor"
    assert "clipboard" not in second


def test_settle_stops_after_two_near_identical_frames() -> None:
    frames = iter((bytes([0, 0]), bytes([255, 255]), bytes([255, 254])))
    clock = iter((0.0, 0.0, 0.1, 0.2))
    sleeps = []
    elapsed = features.wait_for_stable(
        lambda: next(frames),
        monotonic=lambda: next(clock),
        sleep=sleeps.append,
        interval=0.1,
        timeout=2,
        threshold=0.01,
    )
    assert elapsed == pytest.approx(0.2)
    assert sleeps == [0.1, 0.1]


def test_settle_feature_reports_time_after_action() -> None:
    fake = FakeBackend()
    result = server.ComputerServer(fake, frozenset({"settle"})).call(
        "computer_key", {"keys": "Return"}
    )
    assert result["isError"] is False
    assert fake.settle_calls == 1
    assert result["content"][1]["text"] == "Screen settled in 0.125 seconds."


def test_unknown_feature_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("ZETA_COMPUTER_FEATURES", "batch,typo")
    with pytest.raises(ValueError, match="typo"):
        server.ComputerServer(FakeBackend())
