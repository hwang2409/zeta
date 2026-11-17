"""Keep native-computer screenshots live without leaking them into history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ...protocol.types import Message, ToolContentBlock, ToolResult


def _computer_screenshot(result: ToolResult | None) -> bool:
    if result is None or not isinstance(result.structured_content, Mapping):
        return False
    computer = result.structured_content.get("computer")
    return isinstance(computer, Mapping) and computer.get("has_screenshot") is True


def _images(result: ToolResult) -> list[ToolContentBlock]:
    return [
        block
        for block in result.content_blocks or ()
        if block.get("type") == "image"
    ]


def _image_stub(block: Mapping[str, Any]) -> dict[str, Any]:
    stub = {
        key: block[key]
        for key in ("type", "mimeType", "caption", "width", "height", "size")
        if key in block
    }
    data = block.get("data")
    if "size" not in stub and isinstance(data, str):
        padding = len(data) - len(data.rstrip("="))
        stub["size"] = len(data) * 3 // 4 - padding
    stub["data_omitted"] = True
    return stub


def _without_images(result: ToolResult) -> ToolResult:
    if not _computer_screenshot(result):
        return result
    blocks = list(result.content_blocks or ())
    images = [block for block in blocks if block.get("type") == "image"]
    if not images:
        return result
    structured = dict(result.structured_content or {})
    computer = dict(structured["computer"])
    computer["screenshot"] = _image_stub(images[-1])
    structured["computer"] = computer
    return replace(
        result,
        content_blocks=[block for block in blocks if block.get("type") != "image"],
        structured_content=structured,
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _RetainedScreenshot:
    result: ToolResult


class ComputerScreenshotRetention:
    """Own durable screenshot stripping and latest-only provider projection."""

    def __init__(self, *, persist_screenshots: bool = False) -> None:
        self.persist_screenshots = persist_screenshots
        self._latest: _RetainedScreenshot | None = None

    def durable(self, result: ToolResult) -> ToolResult:
        """Return the result shape allowed in the conversation store."""

        return result if self.persist_screenshots else _without_images(result)

    def durable_message(self, message: Message) -> Message:
        """Return a message safe for durable compaction views."""

        if message.tool_result is None:
            return message
        result = self.durable(message.tool_result)
        return message if result is message.tool_result else replace(message, tool_result=result)

    def remember(self, result: ToolResult) -> None:
        """Replace the live screenshot after its durable row commits."""

        if (
            _computer_screenshot(result)
            and not result.is_error
            and not result.is_canceled
            and _images(result)
        ):
            self._latest = _RetainedScreenshot(result)

    def project(self, messages: Sequence[Message]) -> list[Message]:
        """Return provider messages with at most the newest computer image."""

        newest = next(
            (
                message.tool_result
                for message in reversed(messages)
                if _computer_screenshot(message.tool_result)
            ),
            None,
        )
        if newest is None:
            return list(messages)
        latest = self._latest.result if self._latest is not None else None
        if latest is None or latest.tool_call_id != newest.tool_call_id:
            latest = newest if _images(newest) else None
        projected: list[Message] = []
        for message in messages:
            result = message.tool_result
            if not _computer_screenshot(result):
                projected.append(message)
                continue
            replacement = (
                latest
                if latest is not None and result.tool_call_id == newest.tool_call_id
                else _without_images(result)
            )
            projected.append(replace(message, tool_result=replacement))
        return projected


def public_tool_result(result: ToolResult) -> dict[str, Any]:
    """Project a computer result for live transports without screenshot bytes."""

    value = result.to_dict()
    if not _computer_screenshot(result):
        return value
    value["content_blocks"] = [
        _image_stub(block) if block.get("type") == "image" else dict(block)
        for block in result.content_blocks or ()
    ]
    return value
