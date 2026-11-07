from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from rich.console import Console

from zeta.agent.wakes import WakeActivation, WakeRecord, wake_notice
from zeta.protocol.types import StreamEvent, StreamEventType, ToolCall, ToolResult
from zeta.server.control import ServeControl
from zeta.tui.render import render_event


def _plain(event: StreamEvent) -> str:
    renderable = render_event(event)
    assert renderable is not None
    console = Console(record=True, width=100, color_system=None)
    console.print(renderable)
    return console.export_text()


@pytest.mark.parametrize(
    ("action", "arguments", "content"),
    (
        (
            "schedule",
            {"after_seconds": 300, "note": "check staging"},
            "scheduled wake_1 for 2026-01-01T00:05:00+00:00",
        ),
        ("list", {"action": "list"}, "1 pending wake(s)"),
        (
            "cancel",
            {"action": "cancel", "wake_id": "wake_1"},
            "canceled wake_1",
        ),
    ),
)
def test_tui_real_render_path_shows_each_wake_tool_card(
    action: str, arguments: dict[str, object], content: str
) -> None:
    """Mutation: hide generic wake cards; the wake-name assertion fails."""

    call = ToolCall(f"wake-{action}", "wake", arguments)
    start = _plain(StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call))
    end = _plain(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, content),
        )
    )

    assert "wake" in start
    assert action in start or action == "schedule"
    assert "wake" in end
    assert content in end


def test_tui_real_render_path_marks_wake_limit_errors() -> None:
    """Mutation: clear ToolResult.is_error; the failed-state assertion fails."""

    call = ToolCall(
        "wake-overflow",
        "wake",
        {"after_seconds": 10, "note": "one too many"},
    )
    rendered = _plain(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "wake error: at most 8 wakes may be pending",
                is_error=True,
            ),
        )
    )

    assert "· error" in rendered
    assert "at most 8 wakes" in rendered


@pytest.mark.asyncio
async def test_serve_real_event_path_maps_wake_tool_with_negotiated_fidelity() -> None:
    """Mutation: omit structured content in tool_end; the reason assertion fails."""

    published: list[tuple[str, str | None, dict[str, object]]] = []

    class State:
        session_id = "session"

        def tool_started(self) -> None:
            pass

        def tool_finished(self) -> None:
            pass

    class Connections:
        @staticmethod
        def controller_metadata() -> SimpleNamespace:
            return SimpleNamespace(
                features=frozenset({"session_history_fidelity"}),
                protocol_version="1.2",
            )

    server = SimpleNamespace(
        runtime=SimpleNamespace(state=State(), opened=None, policy=None),
        connections=Connections(),
    )
    control = ServeControl(server)

    async def publish(
        event: str, session_id: str | None = None, **fields: object
    ) -> bool:
        published.append((event, session_id, fields))
        return True

    async def no_stale_approvals(_session_id: str | None) -> None:
        return None

    control._notify = publish  # type: ignore[method-assign]
    control._end_stale_approvals = no_stale_approvals  # type: ignore[method-assign]
    call = ToolCall(
        "wake-limit",
        "wake",
        {"after_seconds": 10, "note": "retry deploy"},
    )
    result = ToolResult(
        call.id,
        "autonomous wake disabled: top_level_chain_limit",
        is_error=True,
        structured_content={
            "wake_id": "wake_1",
            "autonomous": False,
            "reason": "top_level_chain_limit",
        },
    )

    await control._event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    )
    await control._event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=result,
        )
    )
    scheduled_at = datetime(2026, 1, 1, tzinfo=UTC)
    notice = wake_notice(
        WakeActivation(
            passive=(
                WakeRecord(
                    wake_id="wake_1",
                    note="retry deploy",
                    scheduled_at_utc=scheduled_at,
                    due_at_utc=scheduled_at,
                    session_class="top_level",
                    chain_id="chain_1",
                    generation=3,
                    state="passive",
                    passive_reason="top_level_chain_limit",
                ),
            )
        )
    )
    await control._event(
        StreamEvent(StreamEventType.MESSAGE_END, message=notice),
        session_id="session",
        background=True,
    )

    assert [item[0] for item in published] == [
        "tool_start",
        "tool_end",
        "assistant_message",
    ]
    assert published[0][2]["tool_call"] == {
        "id": "wake-limit",
        "name": "wake",
        "arguments": {"after_seconds": 10, "note": "retry deploy"},
        "arguments_truncated": False,
    }
    wire_result = published[1][2]["tool_result"]
    assert isinstance(wire_result, dict)
    assert wire_result["structured_content"] == {
        "wake_id": "wake_1",
        "autonomous": False,
        "reason": "top_level_chain_limit",
    }
    assert wire_result["structured_content_truncated"] is None
    wire_notice = published[2][2]["message"]
    assert isinstance(wire_notice, dict)
    assert wire_notice["role"] == "system"
    assert wire_notice["content"][0]["text"].startswith("[zeta] wake reminder:")
