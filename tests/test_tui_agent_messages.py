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


def _render_sequence(
    events: list[StreamEvent],
    width: int,
    *,
    markers: dict[str, dict[str, object]] | None = None,
) -> str:
    transcript = TranscriptWidget()
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=True,
        color_system="truecolor",
        theme=theme.RICH_THEME,
    )
    presenter = TranscriptPresenter(
        transcript,
        console,
        lambda: True,
        transcript.append,
        lambda: markers or {},
    )
    for event in events:
        if event.tool_call is not None and event.tool_result is not None:
            presenter.replay_tool_result(event)
        else:
            rendered = presenter.render_event(event)
            assert rendered is not None
            presenter.print_unit(rendered)
    return Text.from_ansi(transcript.render(width)).plain


def _tool_result_event(
    name: str,
    arguments: dict[str, object],
    content: str,
    structured_content: dict[str, object],
    *,
    is_error: bool = False,
) -> StreamEvent:
    call = ToolCall(f"{name}-1", name, arguments)
    return StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            content,
            is_error=is_error,
            structured_content=structured_content,
        ),
    )


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


@pytest.mark.parametrize("width", [80, 160])
def test_agent_notifications_keep_receipts_compact_and_show_idle_report(width: int) -> None:
    summary = "the shutdown path preserves writer ownership until close completes"
    idle_report = "waiting for the parent to choose the storage cutover window"
    failure_reason = "child transcript ended before the required review completed"
    completion = StreamEvent(
        StreamEventType.AGENT_NOTIFICATION,
        data={
            "kind": "agent_completion",
            "child_instance_id": HANDLE,
            "description": DESCRIPTION,
            "status": "completed",
            "text": "done",
            "summary": summary,
            "verdict": "merge-ready",
            "stats": {"elapsed": 42.1, "turns_used": 3},
        },
    )
    failed = StreamEvent(
        StreamEventType.AGENT_NOTIFICATION,
        data={
            "kind": "agent_completion",
            "child_instance_id": HANDLE,
            "description": DESCRIPTION,
            "status": "error",
            "text": failure_reason,
            "stats": {
                "elapsed": 4.2,
                "turns_used": 1,
                "error": True,
                "canceled": False,
            },
        },
    )
    idle = StreamEvent(
        StreamEventType.AGENT_NOTIFICATION,
        data={
            "kind": "child_idle",
            "child_instance_id": HANDLE,
            "description": DESCRIPTION,
            "status": "child_idle",
            "text": idle_report,
        },
    )

    completion_output = _render(completion, width, markers=_markers())
    failed_output = _render(failed, width, markers=_markers())
    idle_output = _render(idle, width, markers=_markers())

    assert f"⏺ {DESCRIPTION} · completed · 42.1s · 3 turns" in completion_output
    assert summary not in completion_output
    assert f"⏺ {DESCRIPTION} · failed · 4.2s · 1 turns" in failed_output
    assert "reason:" in failed_output
    assert all(
        value in idle_output
        for value in (DESCRIPTION, ":159", "idle", idle_report)
    )
    assert "{" not in completion_output + failed_output + idle_output


@pytest.mark.parametrize("width", [80, 160])
def test_agent_control_cards_render_named_structured_rows(width: int) -> None:
    nested_handle = f"{HANDLE}:1"
    status = _tool_result_event(
        "agent_status",
        {},
        "raw child status serialization",
        {
            "children": [
                {
                    "handle": HANDLE,
                    "description": DESCRIPTION,
                    "agent_type": "planner",
                    "model": "gpt-5.6-luna",
                    "state": "running",
                },
                {
                    "handle": "root:160",
                    "description": "shutdown reviewer",
                    "agent_type": "reviewer",
                    "model": "claude-sonnet-5",
                    "state": "idle",
                },
                {
                    "handle": nested_handle,
                    "description": "nested verifier",
                    "agent_type": "worker",
                    "model": "gpt-5.6-luna",
                    "state": "completed",
                },
            ],
            "total": 3,
        },
    )
    output = _tool_result_event(
        "agent_output",
        {"handle": nested_handle, "offset": 0},
        "assistant: nested verification passed",
        {
            "handle": nested_handle,
            "offset": 0,
            "total": 37,
            "truncated": False,
        },
    )
    cancel = _tool_result_event(
        "agent_cancel",
        {"handle": nested_handle},
        f"agent cancellation confirmed: {nested_handle}",
        {"handle": nested_handle, "status": "cancellation_confirmed"},
    )

    controls_output = _render_sequence([status, output, cancel], width)

    assert all(
        value in controls_output
        for value in (
            f"{DESCRIPTION} :159 · planner · gpt-5.6-luna · running",
            "shutdown reviewer :160 · reviewer · claude-sonnet-5 · idle",
            "nested verifier :159:1 · worker · gpt-5.6-luna · completed",
            "output ← nested verifier :159:1",
            "page 1-37 of 37",
            "assistant: nested verification passed",
            "cancel → nested verifier :159:1",
            "canceled",
        )
    )
    assert "raw child status serialization" not in controls_output
    assert "{'children'" not in controls_output


def test_unknown_identity_snapshot_is_cached_until_observed() -> None:
    from zeta.tui.cards.agent_message import AgentIdentityIndex

    snapshots = 0

    def marker_snapshot() -> dict[str, dict[str, object]]:
        nonlocal snapshots
        snapshots += 1
        return {}

    index = AgentIdentityIndex(marker_snapshot)

    assert index.resolve("root:404").label == "child :404"
    assert index.resolve("root:404").label == "child :404"
    assert snapshots == 1

    observed = _tool_result_event(
        "agent_status",
        {},
        "status",
        {
            "children": [
                {
                    "handle": "root:404",
                    "description": "late child",
                    "state": "running",
                }
            ]
        },
    )
    index.observe(observed)

    assert index.resolve("root:404").label == "late child :404"
    assert snapshots == 1
