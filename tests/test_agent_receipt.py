import pytest

from zeta.agent.receipt import (
    MAX_AGENT_RESULT_BYTES,
    _append_tool_result,
    build_explicit_finish_receipt,
    encode_json,
)
from zeta.core.retention.computer import ComputerScreenshotRetention
from zeta.protocol.types import ToolCall, ToolResult


def test_computer_screenshot_is_not_retained_when_durable_append_fails() -> None:
    class FailingStore:
        def append_message(self, message):  # type: ignore[no-untyped-def]
            raise OSError("append failed")

    class Owner:
        store = FailingStore()
        computer_screenshot_retention = ComputerScreenshotRetention()
        tool_registry = None

    marker = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAHnOcQAAAAABJRU5ErkJggg=="
    result = ToolResult(
        "shot",
        "screenshot",
        content_blocks=[{"type": "image", "data": marker, "mimeType": "image/png"}],
        structured_content={"computer": {"has_screenshot": True}},
    )
    owner = Owner()

    with pytest.raises(OSError, match="append failed"):
        _append_tool_result(
            owner, ToolCall("shot", "computer", {"action": "screenshot"}), result
        )

    assert owner.computer_screenshot_retention._latest is None


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
