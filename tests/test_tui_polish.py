"""Visible tool and message surfaces for the TUI polish arc."""

from __future__ import annotations

from io import StringIO

from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.text import Text

from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.tui import theme
from zeta.tui.cards.agent import AgentCard
from zeta.tui.render import render_event
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.user import user_message


def _plain(renderable: object) -> str:
    output = StringIO()
    Console(file=output, width=80, force_terminal=False).print(renderable)
    return output.getvalue()


def test_gruvbox_fills_user_and_tool_surfaces() -> None:
    previous = theme.active_palette()
    theme.set_active_palette(theme.GRUVBOX_DARK)
    try:
        output = StringIO()
        console = Console(file=output, width=60, force_terminal=True, color_system="truecolor")
        console.print(user_message(Text("▌ build the maze", style=theme.BODY)))
        user_ansi = output.getvalue()
        assert "\x1b[48;2;59;54;64m" in user_ansi

        call = ToolCall("read-1", "read", {"path": "maze.c"})
        rendered = render_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=ToolResult(call.id, "int main(void) { return 0; }"),
            )
        )
        assert rendered is not None
        assert str(rendered.style) == theme.READ_BG
    finally:
        theme.set_active_palette(previous)


def test_bash_card_shows_command_output_and_exit_state() -> None:
    call = ToolCall("bash-1", "bash", {"command": "make test"})
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "stdout:\n1 passed\nstderr:\nwarning",
                is_error=True,
                structured_content={
                    "stdout": "1 passed",
                    "stderr": "warning",
                    "exit_code": 1,
                    "cwd_after": "/tmp/project",
                },
            ),
        )
    )
    assert rendered is not None
    plain = _plain(rendered)
    assert "make test" in plain
    assert "1 passed" in plain
    assert "warning" in plain
    assert "exit 1" in plain
    assert "exit_code:" not in plain


def test_exec_card_uses_shell_status_and_bounded_output() -> None:
    call = ToolCall("exec-1", "exec", {"command": "cargo test"})
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "exit_code: 0\nstdout:\n" + "running 0 tests\n" * 30,
                structured_content={"exit_code": 0},
            ),
        )
    )
    assert rendered is not None
    plain = _plain(rendered)
    assert "cargo test" in plain
    assert "exit 0" in plain
    assert "exit_code:" not in plain
    assert plain.count("running 0 tests") <= 15


def test_todo_and_skill_cards_show_useful_summaries() -> None:
    todo = ToolCall("todo-1", "todo", {})
    todo_card = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=todo,
            tool_result=ToolResult(
                todo.id,
                "todo list: 1 pending, 1 in progress, 2 completed, 0 canceled",
                structured_content={
                    "counts": {"pending": 1, "in_progress": 1, "completed": 2},
                    "items": [{"content": "test the solver", "status": "in_progress"}],
                },
            ),
        )
    )
    skill = ToolCall("skill-1", "skill", {"name": "review"})
    skill_card = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=skill,
            tool_result=ToolResult(skill.id, "long skill prompt"),
        )
    )
    assert todo_card is not None and skill_card is not None
    assert "test the solver" in _plain(todo_card)
    assert "skill review" in _plain(skill_card)
    assert "long skill prompt" not in _plain(skill_card)


def test_expanded_agent_shows_tool_status_without_raw_result(tmp_path) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="child")
    command = ToolCall("exec-1", "exec", {"command": "cargo test"})
    child.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(command)]))
    child.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("stdout:\nrunning 0 tests\n" * 10)],
            tool_result=ToolResult(
                command.id,
                "stdout:\nrunning 0 tests\n" * 10,
                structured_content={"exit_code": 0},
            ),
        )
    )
    agent = ToolCall("agent-1", "agent", {"description": "review tests"})

    rendered = AgentCard.render_expanded(
        agent,
        elapsed_seconds=1.0,
        turns_used=1,
        child_session_path=str(child.session_dir),
    )

    assert rendered is not None
    plain = _plain(rendered)
    assert "exec: exit 0" in plain
    assert "running 0 tests" not in plain
    assert "tool_result:" not in plain


def test_gruvbox_transcript_stays_readable_at_narrow_widths() -> None:
    previous = theme.active_palette()
    theme.set_active_palette(theme.GRUVBOX_DARK)
    try:
        transcript = TranscriptWidget()
        transcript.append(user_message(Text("▌ inspect the project")))
        call = ToolCall("exec-1", "exec", {"command": "cargo test --all-targets"})
        transcript.append(
            render_event(
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_END,
                    tool_call=call,
                    tool_result=ToolResult(
                        call.id,
                        "exit_code: 0\nstdout:\nall tests passed",
                        structured_content={"exit_code": 0},
                    ),
                )
            )
        )
        for width in (40, 80):
            ansi = transcript.render(width)
            plain = Text.from_ansi(ansi).plain
            assert "inspect the project" in plain
            assert "all tests passed" in plain
            assert "\x1b[48;2;" in ansi
            assert all(get_cwidth(line) <= width for line in plain.splitlines())
    finally:
        theme.set_active_palette(previous)
