"""Explicit completion for opted-in background children."""

from __future__ import annotations

from typing import Any

from ...agent.explicit_finish import owned_work_descriptions
from ...agent.terminal_fence import TerminalTransitionFence
from ..registry import AbortSignal, ToolExecutionContext, ToolRegistry, text_block

_STATUSES = {"done", "blocked", "failed"}
_VERDICTS = {"pass", "fail", "inconclusive"}


def register(registry: ToolRegistry, *, child_loop: Any = None) -> None:
    """Register the finish tool on one eligible child loop."""
    if child_loop is None:
        return

    notification_wake = child_loop.notification_wake
    fence = TerminalTransitionFence(
        registry._session_store,
        notification_pending=lambda: (
            notification_wake.pending_message() is not None
            or getattr(notification_wake, "has_undelivered_claim", False)
        ),
        persisted_notification_ids=lambda: getattr(
            notification_wake, "persisted_ids", frozenset()
        ),
        owned_work=lambda: owned_work_descriptions(child_loop),
    )
    child_loop._terminal_transition_fence = fence

    registry.set_execution_guard(
        lambda _name: (
            "child is finishing"
            if child_loop._explicit_finish_report is not None
            else None
        )
    )

    async def finish(
        _registry: ToolRegistry,
        arguments: dict[str, Any],
        _abort_signal: AbortSignal,
        _stream_publisher: object = None,
        _execution_context: ToolExecutionContext | None = None,
    ) -> dict[str, object]:
        status = arguments.get("status")
        summary = arguments.get("summary")
        if status not in _STATUSES:
            return _error("status must be done, blocked, or failed")
        if type(summary) is not str or not summary.strip() or len(summary) > 8_000:
            return _error("summary must be a nonempty string of at most 8000 characters")
        head_sha = arguments.get("head_sha")
        if head_sha is not None and (
            type(head_sha) is not str
            or not 7 <= len(head_sha) <= 64
            or any(char not in "0123456789abcdef" for char in head_sha)
        ):
            return _error("head_sha must be 7 to 64 lowercase hexadecimal characters")
        verdict = arguments.get("verdict")
        if verdict is not None and verdict not in _VERDICTS:
            return _error("verdict must be pass, fail, or inconclusive")
        if child_loop._explicit_finish_report is not None:
            return _error("child is finishing")
        report = {"status": status, "summary": summary.strip()}
        if head_sha is not None:
            report["head_sha"] = head_sha
        if verdict is not None:
            report["verdict"] = verdict
        refusal = await fence.accept_finish(report)
        if refusal is not None:
            return _error("finish refused; " + refusal)
        child_loop._explicit_finish_report = report
        return {
            "content": [text_block("finish accepted; no more work or prose is needed")],
            "isError": False,
            "structuredContent": report,
        }

    registry.register_session_tool(
        "finish",
        finish,
        description="Declare that this background agent contract is complete.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"enum": sorted(_STATUSES)},
                "summary": {"type": "string", "minLength": 1, "maxLength": 8_000},
                "head_sha": {"type": "string", "pattern": "^[0-9a-f]{7,64}$"},
                "verdict": {"enum": sorted(_VERDICTS)},
            },
            "required": ["status", "summary"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=False,
    )


def _error(message: str) -> dict[str, object]:
    return {
        "content": [text_block("finish error: " + message)],
        "isError": True,
        "structuredContent": None,
    }
