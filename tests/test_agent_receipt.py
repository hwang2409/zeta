import pytest

from zeta.agent.receipt import (
    MAX_AGENT_RESULT_BYTES,
    _append_tool_result,
    build_explicit_finish_receipt,
    encode_json,
)
from zeta.core.retention.computer import ComputerScreenshotRetention
from zeta.protocol.types import ToolCall, ToolResult


def test_failed_screenshot_append_preserves_previous_retained_owner() -> None:
    class FailingSecondStore:
        def __init__(self) -> None:
            self.messages = []

        def append_message(self, message):  # type: ignore[no-untyped-def]
            if self.messages:
                raise OSError("append failed")
            self.messages.append(message)
            return message

    class Owner:
        store = FailingSecondStore()
        computer_screenshot_retention = ComputerScreenshotRetention()
        tool_registry = None

    first_marker = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHnOcQAAAAABJRU5ErkJggg=="
    second_marker = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg=="
    owner = Owner()
    first = ToolResult(
        "shot-1",
        "screenshot",
        content_blocks=[{"type": "image", "data": first_marker, "mimeType": "image/png"}],
        structured_content={"computer": {"has_screenshot": True}},
    )
    second = ToolResult(
        "shot-2",
        "screenshot",
        content_blocks=[{"type": "image", "data": second_marker, "mimeType": "image/png"}],
        structured_content={"computer": {"has_screenshot": True}},
    )

    _append_tool_result(
        owner, ToolCall("shot-1", "computer", {"action": "screenshot"}), first
    )
    first_owner = owner.computer_screenshot_retention._latest
    with pytest.raises(OSError, match="append failed"):
        _append_tool_result(
            owner, ToolCall("shot-2", "computer", {"action": "screenshot"}), second
        )

    projected = owner.computer_screenshot_retention.project(owner.store.messages)
    assert owner.computer_screenshot_retention._latest is first_owner
    assert len(owner.store.messages) == 1
    assert projected[0].tool_result.content_blocks[-1]["data"] == first_marker


def test_explicit_finish_multibyte_summary_fits_exact_envelope_limit() -> None:
    report = {
        "status": "done",
        "summary": "界" * 8000,
        "head_sha": "abcdef0",
        "verdict": "pass",
    }
    receipt = build_explicit_finish_receipt(
        report, {}, tool_call_id="finish-1", max_bytes=MAX_AGENT_RESULT_BYTES
    )
    summary = receipt["structuredContent"]["summary"]
    assert summary
    assert receipt["content"][0]["text"].startswith("done: " + summary)
    assert len(encode_json(receipt)) <= MAX_AGENT_RESULT_BYTES
    longer = {**report, "summary": summary + "界"}
    longer_receipt = build_explicit_finish_receipt(
        longer, {}, tool_call_id="finish-1", max_bytes=MAX_AGENT_RESULT_BYTES
    )
    assert longer_receipt["structuredContent"]["summary"] == summary
