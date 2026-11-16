"""MacBackend translation of the helper protocol, and the tool's per-action error messages."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry
from zeta.tools.computer.backend import Button, ComputerError, ScreenFrame, ScreenState
from zeta.tools.computer.mac import MacBackend

FAKE = Path(__file__).parent / "fixtures" / "computer_fake_helper.py"


def make_backend() -> MacBackend:
    return MacBackend([sys.executable, str(FAKE)])


@pytest.mark.asyncio
async def test_screenshot_decodes_frame() -> None:
    backend = make_backend()
    try:
        frame = await backend.screenshot()
    finally:
        backend.close()
    assert isinstance(frame, ScreenFrame)
    assert frame.media_type == "image/png"
    assert frame.width == 1 and frame.height == 1 and frame.scale == 2.0
    assert frame.image.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_state_and_cursor_round_trip() -> None:
    backend = make_backend()
    try:
        state = await backend.state()
        cursor = await backend.cursor_position()
    finally:
        backend.close()
    assert state == ScreenState(False, False, "dev.zeta.fake")
    assert cursor == (12, 34)


@pytest.mark.asyncio
async def test_helper_error_maps_to_computer_error() -> None:
    backend = make_backend()
    try:
        with pytest.raises(ComputerError) as excinfo:
            await backend.screenshot(99)  # fixture sentinel -> secure_input error
    finally:
        backend.close()
    assert excinfo.value.code == "secure_input"


class _RefusingBackend:
    """A backend whose click is refused, to check the tool's per-action message."""

    async def screenshot(self, display: int | None = None) -> ScreenFrame:
        raise AssertionError("not called")

    async def click(self, x: int, y: int, button: Button, count: int) -> None:
        raise ComputerError("locked", "screen is locked")


@pytest.mark.asyncio
async def test_tool_reports_action_specific_refusal(tmp_path: Path) -> None:
    registry = ToolRegistry(
        tmp_path, skill_catalog=SkillCatalog.empty(), computer_enabled=True
    )
    registry.computer_backend = _RefusingBackend()  # type: ignore[assignment]
    try:
        result = await registry.execute(
            ToolCall("c", "computer", {"action": "left_click", "coordinate": [1, 2]})
        )
    finally:
        await registry.close()
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "left_click failed" in text
    assert "screen is locked" in text
