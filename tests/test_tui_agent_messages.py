from __future__ import annotations

import importlib
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from zeta.protocol.types import StreamEvent, StreamEventType, ToolCall, ToolResult
from zeta.tui import theme
from zeta.tui.render import render_event
from zeta.tui.transcript import TranscriptPresenter, TranscriptWidget

HANDLE = "86e3d775a40b42ceaa76ca571e27ff1b:159"
DESCRIPTION = "durable-write planner"
QUESTION_ID = "q-7f2a"


def _render(
    event: StreamEvent,
    width: int,
    *,
    markers: dict[str, dict[str, object]] | None = None,
) -> str:
    """Use the shipping event renderer, with the new index when it exists."""

    agent_message = importlib.util.find_spec("zeta.tui.cards.agent_message")
    transcript = TranscriptWidget()
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=True,
        color_system="truecolor",
        theme=theme.RICH_THEME,
    )
    presenter = (
        TranscriptPresenter(
            transcript,
            console,
            lambda: True,
            transcript.append,
            lambda: markers or {},
        )
        if agent_message is not None
        else TranscriptPresenter(transcript, console, lambda: True, transcript.append)
    )
    if event.tool_call is not None and event.tool_result is not None:
        presenter.replay_tool_result(event)
    else:
        rendered = (
            presenter.render_event(event)
            if agent_message is not None
            else render_event(event)
        )
        assert rendered is not None
        presenter.print_unit(rendered)
    return Text.from_ansi(transcript.render(width)).plain


def _agent_send(message: str, *, handle: str = HANDLE) -> StreamEvent:
    call = ToolCall(
        "send-1",
        "agent_send",
        {"child_instance_id": handle, "message": message},
    )
    return StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            f"queued a follow-up for {handle}; a running child receives it after "
            "its current tool call, before its next model call",
            structured_content={"child_instance_id": handle},
        ),
    )


def _markers() -> dict[str, dict[str, object]]:
    return {
        HANDLE: {
            "description": DESCRIPTION,
            "tool_call": {"arguments": {"model": "gpt-5.6-luna"}},
            "agent_type": "planner",
        }
    }


@pytest.mark.parametrize("width", [80, 160])
def test_agent_send_card_names_target_and_wraps_message(width: int) -> None:
    message = " ".join(f"message-word-{index}" for index in range(18))

    output = _render(_agent_send(message), width, markers=_markers())

    assert DESCRIPTION in output
    assert ":159" in output
    assert all(word in output for word in message.split())
    assert "queued" in output
    assert HANDLE not in output
    assert "a running child receives it" not in output


@pytest.mark.parametrize("width", [80, 160])
def test_agent_send_to_unknown_handle_falls_back_to_short_handle(width: int) -> None:
    unknown = "86e3d775a40b42ceaa76ca571e27ff1b:160:1"

    output = _render(_agent_send("check the shutdown path", handle=unknown), width)

    assert "child :160:1" in output
    assert unknown not in output


@pytest.mark.parametrize("width", [80, 160])
def test_ask_parent_round_trip_is_readable_in_parent_and_child_views(
    width: int,
) -> None:
    question = (
        "Should cancellation wait for the writer or return after ownership transfer?"
    )
    options = ["wait for writer", "return after transfer"]
    ask_call = ToolCall(
        "ask-1", "ask_parent", {"question": question, "options": options}
    )
    child_call = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=ask_call,
        tool_result=ToolResult(
            ask_call.id,
            f"question sent (id {QUESTION_ID})",
            structured_content={"question_id": QUESTION_ID},
        ),
    )
    parent_question = StreamEvent(
        StreamEventType.AGENT_NOTIFICATION,
        data={
            "kind": "child_question",
            "child_instance_id": HANDLE,
            "question_id": QUESTION_ID,
            "question": question,
            "options": options,
        },
    )
    answer = _agent_send(f"Re {QUESTION_ID}: wait for the writer")
    delivered = StreamEvent(
        StreamEventType.CHILD_MESSAGE_DELIVERED,
        data={"preview": f"Re {QUESTION_ID}: wait for the writer"},
    )
    withdrawn = StreamEvent(
        StreamEventType.AGENT_NOTIFICATION,
        data={
            "kind": "child_question_withdrawn",
            "child_instance_id": HANDLE,
            "question_id": QUESTION_ID,
        },
    )

    child_output = _render(child_call, width)
    parent_output = _render(parent_question, width, markers=_markers())
    answer_output = _render(answer, width, markers=_markers())
    delivered_output = _render(delivered, width)
    withdrawn_output = _render(withdrawn, width, markers=_markers())

    assert question in child_output
    assert all(option in child_output for option in options)
    assert QUESTION_ID in child_output
    assert "sent" in child_output
    assert DESCRIPTION in parent_output
    assert ":159" in parent_output
    assert question in parent_output
    assert all(option in parent_output for option in options)
    assert QUESTION_ID in parent_output
    assert "answer to" in answer_output
    assert QUESTION_ID in answer_output
    assert "answer to" in delivered_output
    assert QUESTION_ID in delivered_output
    assert DESCRIPTION in withdrawn_output
    assert QUESTION_ID in withdrawn_output


@pytest.mark.parametrize("width", [80, 160])
def test_long_agent_message_is_bounded(width: int) -> None:
    message = "\n".join(f"line {index}" for index in range(200))

    output = _render(_agent_send(message), width, markers=_markers())

    assert "line 0" in output
    assert "line 14" in output
    assert "line 15" not in output
    assert "… +185 lines" in output
