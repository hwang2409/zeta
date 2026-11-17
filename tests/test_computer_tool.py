"""Tests for the opt-in ``computer`` tool and its backend seam (slice 1)."""

from __future__ import annotations

import base64
import tomllib
from contextlib import asynccontextmanager
from pathlib import Path
from textwrap import dedent

import pytest

from zeta.agent.tool_results import validated_tool_result
from zeta.config.settings import (
    SETTINGS_FILENAME,
    ComputerUseConfig,
    load_settings,
    resolve,
)
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, ToolCall
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex import build_responses_payload
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.computer.backend import ComputerError
from zeta.tools.computer.fake import FakeBackend


@asynccontextmanager
async def _computer_registry(
    tmp_path: Path, *, enabled: bool = True, backend: FakeBackend | None = None
):
    registry = ToolRegistry(
        tmp_path, skill_catalog=SkillCatalog.empty(), computer=ComputerUseConfig(enabled=enabled, default_app_access="allow")
    )
    if backend is not None:
        registry.computer_backend = backend
    try:
        yield registry
    finally:
        await registry.close()


def _ops(backend: FakeBackend) -> list[tuple]:
    return [action for action in backend.actions if action[0] != "state"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["success", "input_policy", "capture_policy", "capture_error", "input_error", "killed", "locked", "secure_input"])
async def test_action_capture_result_table(tmp_path: Path, case: str) -> None:
    class CaseBackend(FakeBackend):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.capture_count = 0

        async def state(self):
            self.actions.append(("state",))
            if case == "capture_policy" and getattr(self, "input_done", False):
                self.frontmost_bundle_id = "com.denied.app"
            return type(await super().state())(self.locked, self.secure_input, self.frontmost_bundle_id)

        async def screenshot(self, display=None):
            self.capture_count += 1
            if case == "capture_error" and self.capture_count > 1:
                raise ComputerError("not_permitted", "screen recording denied")
            return await super().screenshot(display)

        async def click(self, x, y, button, count):
            if case == "input_error":
                raise ComputerError("internal", "input failed")
            if case == "killed":
                raise ComputerError("killed", "kill switch engaged")
            self.input_done = True
            return await super().click(x, y, button, count)

    backend = CaseBackend(locked=case == "locked", secure_input=case == "secure_input")
    config = ComputerUseConfig(enabled=True, default_app_access="deny", bundle_ids={"com.example.app": "allow"})
    if case == "input_policy":
        config = ComputerUseConfig(enabled=True)
    async with _computer_registry(tmp_path, backend=backend) as registry:
        registry.computer_config = config
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        result = await registry.execute(ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]}))
    if case == "success":
        assert result["isError"] is False and result["content"][-1]["type"] == "image"
    elif case in {"capture_policy", "capture_error"}:
        assert result["isError"] is False and "screenshot failed" in result["content"][0]["text"]
    else:
        assert result["isError"] is True
    if case == "input_policy":
        assert not any(action[0] == "click" for action in backend.actions)


@pytest.mark.asyncio
async def test_wait_is_guarded_and_returns_screenshot(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, backend=FakeBackend()) as registry:
        result = await registry.execute(ToolCall("w", "computer", {"action": "wait"}))
    assert result["isError"] is False
    assert result["content"][-1]["type"] == "image"


def _tool_names(registry: ToolRegistry) -> set[str]:
    return {schema["name"] for schema in registry.schemas}


@pytest.mark.asyncio
async def test_computer_tool_is_opt_in(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, enabled=False) as off:
        assert "computer" not in _tool_names(off)
    async with _computer_registry(tmp_path, enabled=True) as on:
        assert "computer" in _tool_names(on)


@pytest.mark.asyncio
async def test_screenshot_returns_image_and_records_capture(tmp_path: Path) -> None:
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "screenshot"})
        )

    assert result["isError"] is False
    assert [block["type"] for block in result["content"]] == ["text", "image"]
    image = result["content"][1]
    assert image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"]) == FakeBackend().image
    assert _ops(backend) == [("screenshot", None)]


@pytest.mark.asyncio
async def test_screen_changing_actions_record_op_and_return_screenshot(
    tmp_path: Path,
) -> None:
    cases = [
        ({"action": "left_click", "coordinate": [10, 20]}, ("click", 10, 20, "left", 1)),
        ({"action": "right_click", "coordinate": [1, 2]}, ("click", 1, 2, "right", 1)),
        ({"action": "double_click", "coordinate": [3, 4]}, ("click", 3, 4, "left", 2)),
        ({"action": "triple_click", "coordinate": [5, 6]}, ("click", 5, 6, "left", 3)),
        ({"action": "mouse_move", "coordinate": [7, 8]}, ("move", 7, 8)),
        ({"action": "left_mouse_down", "coordinate": [9, 1]}, ("mouse_down", 9, 1, "left")),
        ({"action": "left_mouse_up", "coordinate": [2, 3]}, ("mouse_up", 2, 3, "left")),
        (
            {"action": "left_click_drag", "start_coordinate": [0, 0], "coordinate": [4, 5]},
            ("drag", [(0, 0), (4, 5)]),
        ),
        ({"action": "type", "text": "hi"}, ("type_text", "hi")),
        ({"action": "key", "text": "cmd+s"}, ("key", "cmd+s")),
        (
            {"action": "scroll", "coordinate": [1, 1], "scroll_direction": "down", "scroll_amount": 3},
            ("scroll", 1, 1, 0, 3),
        ),
    ]
    for arguments, expected in cases:
        backend = FakeBackend()
        async with _computer_registry(tmp_path, backend=backend) as registry:
            await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
            result = await registry.execute(ToolCall("c", "computer", arguments))
        assert result["isError"] is False, arguments
        # every screen-changing action returns a fresh screenshot last
        assert result["content"][-1]["type"] == "image", arguments
        assert expected in _ops(backend), arguments
        assert _ops(backend)[-1] == ("screenshot", None), arguments


@pytest.mark.asyncio
async def test_action_done_is_not_error_when_followup_screenshot_fails(
    tmp_path: Path,
) -> None:
    class ScreenshotFailureBackend(FakeBackend):
        async def screenshot(self, display: int | None = None):
            if any(action[0] == "click" for action in self.actions):
                raise ComputerError("not_permitted", "screen recording denied")
            return await super().screenshot(display)

    backend = ScreenshotFailureBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        result = await registry.execute(
            ToolCall(
                "c",
                "computer",
                {"action": "left_click", "coordinate": [10, 20]},
            )
        )

    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "left_click done" in text
    assert "screenshot failed" in text
    assert "screen recording denied" in text
    assert ("click", 10, 20, "left", 1) in _ops(backend)


@pytest.mark.asyncio
async def test_cursor_position_returns_text_without_capture(tmp_path: Path) -> None:
    backend = FakeBackend(cursor=(42, 7))
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "cursor_position"})
        )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "cursor_position x=42 y=7"
    assert all(block["type"] != "image" for block in result["content"])
    assert _ops(backend)[-1] == ("cursor_position",)


@pytest.mark.asyncio
async def test_cursor_position_edge_can_be_reused_for_click(tmp_path: Path) -> None:
    backend = FakeBackend(width=1280, height=800, cursor=(1280, 800))
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        position = await registry.execute(
            ToolCall("p", "computer", {"action": "cursor_position"})
        )
        point = position["structuredContent"]
        result = await registry.execute(
            ToolCall(
                "c",
                "computer",
                {"action": "left_click", "coordinate": [point["x"], point["y"]]},
            )
        )

    assert position["isError"] is False
    assert {key: point[key] for key in ("x", "y")} == {"x": 1279, "y": 799}
    assert result["isError"] is False
    assert ("click", 1279, 799, "left", 1) in _ops(backend)


@pytest.mark.asyncio
async def test_missing_coordinate_is_a_clear_error(tmp_path: Path) -> None:
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click"})
        )

    assert result["isError"] is True
    assert "coordinate" in result["content"][0]["text"]
    assert _ops(backend) == [("screenshot", None)]


@pytest.mark.asyncio
async def test_child_registry_does_not_inherit_parent_screenshot(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    (tmp_path / "parent").mkdir()
    parent = ToolRegistry(
        tmp_path / "parent", skill_catalog=SkillCatalog.empty(), computer_enabled=True
    )
    child_store = ConversationStore(tmp_path / "child", cwd=tmp_path)
    child: ToolRegistry | None = None
    try:
        parent.computer_backend = backend
        await parent.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        child = parent.clone_for_session(child_store)
        child_backend = FakeBackend()
        child.computer_backend = child_backend
        result = await child.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]})
        )
        assert result["isError"] is True
        assert "computer" in result["content"][0]["text"]
        assert _ops(child_backend) == []
    finally:
        if child is not None:
            await child.close()
        await parent.close()
        child_store.close()


@pytest.mark.asyncio
async def test_coordinate_action_before_screenshot_is_a_clear_error(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]})
        )

    assert result["isError"] is True
    assert "screenshot" in result["content"][0]["text"]
    assert _ops(backend) == []


@pytest.mark.asyncio
async def test_enabled_without_backend_reports_clearly(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path) as registry:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "screenshot"})
        )

    assert result["isError"] is True
    assert "no computer backend" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_screenshot_reaches_provider_payloads_as_image(tmp_path: Path) -> None:
    async with _computer_registry(tmp_path, backend=FakeBackend()) as registry:
        raw = await registry.execute(
            ToolCall("shot", "computer", {"action": "screenshot"})
        )
    tool_result = validated_tool_result(raw, "shot")
    message = Message(MessageRole.TOOL_RESULT, tool_result=tool_result)

    anthropic = build_messages_payload(
        [message], [], model="claude-test", max_tokens=4096, thinking_budget=2048
    )
    content = anthropic["messages"][0]["content"][0]["content"]
    image = next(block for block in content if block["type"] == "image")
    assert image["source"]["type"] == "base64"
    assert base64.b64decode(image["source"]["data"]) == FakeBackend().image

    codex = build_responses_payload([message], [], model="codex-test")
    codex_image = codex["input"][1]["content"][0]
    assert codex_image["type"] == "input_image"
    assert codex_image["image_url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_click_maps_through_downscaled_retina_frame(tmp_path: Path) -> None:
    # A 2x retina display downscaled to the WXGA box; the physical/image ratio
    # (2.0) and the point scale (2.0) cancel, so model coords land at origin+xy.
    backend = FakeBackend(width=2560, height=1600, scale=2.0, origin_x=100, origin_y=50)
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        geometry = registry.computer_geometry
        assert (geometry.image_width, geometry.image_height) == (1280, 800)
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [640, 400]})
        )

    assert result["isError"] is False
    assert ("click", 740, 450, "left", 1) in _ops(backend)


@pytest.mark.asyncio
async def test_out_of_bounds_coordinate_is_rejected(tmp_path: Path) -> None:
    backend = FakeBackend(width=1280, height=800)
    async with _computer_registry(tmp_path, backend=backend) as registry:
        await registry.execute(ToolCall("s", "computer", {"action": "screenshot"}))
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [2000, 10]})
        )

    assert result["isError"] is True
    assert "outside" in result["content"][0]["text"]
    assert all(action[0] != "click" for action in _ops(backend))


def _write_settings(base: Path, body: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / SETTINGS_FILENAME).write_text(dedent(body).lstrip(), encoding="utf-8")


@pytest.mark.asyncio
async def test_denied_app_guidance_is_parseable_and_allows_app(tmp_path: Path) -> None:
    backend = FakeBackend(frontmost_bundle_id="com.apple.Safari")
    async with _computer_registry(tmp_path, backend=backend) as registry:
        registry.computer_config = ComputerUseConfig(enabled=True)
        denied = await registry.execute(ToolCall("c", "computer", {"action": "screenshot"}))
        line = denied["content"][0]["text"].split("add ", 1)[1]
        parsed = tomllib.loads(line)
        registry.computer_config = ComputerUseConfig(enabled=True, bundle_ids={"com.apple.Safari": "allow"})
        allowed = await registry.execute(ToolCall("c", "computer", {"action": "screenshot"}))
    assert parsed["computer"]["macos"]["bundle_ids"]["com.apple.Safari"] == "allow"
    assert allowed["isError"] is False


def _resolved_computer_enabled(home: Path, project: Path | None = None) -> bool:
    settings = load_settings(home=home, project_dir=project).settings
    return resolve(
        settings,
        cli_provider="claude",
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    ).computer_enabled


def test_computer_defaults_disabled(tmp_path: Path) -> None:
    assert _resolved_computer_enabled(tmp_path / "empty") is False


def test_home_can_enable_computer(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_settings(home, "[computer]\nenabled = true\n")
    assert _resolved_computer_enabled(home) is True


def test_project_cannot_enable_computer(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_settings(home, 'provider = "claude"\n')
    _write_settings(project, "[computer]\nenabled = true\n")
    assert _resolved_computer_enabled(home, project) is False
