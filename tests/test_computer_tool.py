"""Tests for the opt-in ``computer`` tool and its backend seam (slice 1)."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from textwrap import dedent

import pytest

from zeta.agent.tool_results import validated_tool_result
from zeta.config.settings import SETTINGS_FILENAME, load_settings, resolve
from zeta.protocol.types import Message, MessageRole, ToolCall
from zeta.providers.anthropic import build_messages_payload
from zeta.providers.codex import build_responses_payload
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.computer.fake import FakeBackend


@asynccontextmanager
async def _computer_registry(
    tmp_path: Path, *, enabled: bool = True, backend: FakeBackend | None = None
):
    registry = ToolRegistry(
        tmp_path, skill_catalog=SkillCatalog.empty(), computer_enabled=enabled
    )
    if backend is not None:
        registry.computer_backend = backend
    try:
        yield registry
    finally:
        await registry.close()


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
    assert backend.actions == [("screenshot", None)]


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
            result = await registry.execute(ToolCall("c", "computer", arguments))
        assert result["isError"] is False, arguments
        # every screen-changing action returns a fresh screenshot last
        assert result["content"][-1]["type"] == "image", arguments
        assert expected in backend.actions, arguments
        assert backend.actions[-1] == ("screenshot", None), arguments


@pytest.mark.asyncio
async def test_cursor_position_returns_text_without_capture(tmp_path: Path) -> None:
    backend = FakeBackend(cursor=(42, 7))
    async with _computer_registry(tmp_path, backend=backend) as registry:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "cursor_position"})
        )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "cursor_position x=42 y=7"
    assert all(block["type"] != "image" for block in result["content"])
    assert backend.actions == [("cursor_position",)]


@pytest.mark.asyncio
async def test_missing_coordinate_is_a_clear_error(tmp_path: Path) -> None:
    backend = FakeBackend()
    async with _computer_registry(tmp_path, backend=backend) as registry:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click"})
        )

    assert result["isError"] is True
    assert "coordinate" in result["content"][0]["text"]
    assert backend.actions == []


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


def _write_settings(base: Path, body: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / SETTINGS_FILENAME).write_text(dedent(body).lstrip(), encoding="utf-8")


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
