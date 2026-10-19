"""Visible tool and message surfaces for the TUI polish arc."""

from __future__ import annotations

from io import StringIO

from prompt_toolkit.application import create_app_session
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput
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
from zeta.tui.layout import (
    CONTENT_MARGIN,
    QueuedPanel,
    TranscriptMargins,
    content_width,
    queued_lines,
)
from zeta.tui.render import render_event
from zeta.tui.transcript import TranscriptPresenter, TranscriptWidget
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
        console.print(user_message(Text("build the maze", style=theme.BODY)))
        user_ansi = output.getvalue()
        assert "\x1b[48;2;60;56;54m" in user_ansi

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


def test_sent_messages_are_padded_bands_with_one_separator() -> None:
    previous = theme.active_palette()
    theme.set_active_palette(theme.GRUVBOX_DARK)
    try:
        transcript = TranscriptWidget()
        presenter = TranscriptPresenter(
            transcript,
            Console(file=StringIO(), force_terminal=False),
            lambda: True,
            transcript.append,
        )
        presenter.print_user(user_message(Text("first")))
        presenter.print_user(user_message(Text("second")))

        for width in (12, 40):
            ansi_lines = transcript.render(width).splitlines()
            lines = [Text.from_ansi(line).plain for line in ansi_lines]
            assert [line.strip() for line in lines] == [
                "", "first", "", "", "", "second", "",
            ]
            band_rows = (0, 1, 2, 4, 5, 6)
            assert all(len(lines[row]) == width for row in band_rows)
            assert all("\x1b[48;2;60;56;54m" in ansi_lines[row] for row in band_rows)
            assert lines[3] == ""
    finally:
        theme.set_active_palette(previous)


def test_long_sent_message_wraps_inside_the_band() -> None:
    output = StringIO()
    Console(file=output, width=20, force_terminal=True, color_system="truecolor").print(
        user_message(Text("one two three four five six seven", style=theme.BODY))
    )
    ansi_lines = output.getvalue().splitlines()
    lines = [Text.from_ansi(line).plain for line in ansi_lines]

    assert len(lines) > 3
    assert lines[0].strip() == "" and lines[-1].strip() == ""
    assert " ".join(" ".join(lines).split()) == "one two three four five six seven"
    assert all(len(line) == 20 and line.startswith(" ") for line in lines)
    fill = theme.active_palette().composer_fill.lstrip("#")
    band = "48;2;" + ";".join(str(int(fill[i : i + 2], 16)) for i in (0, 2, 4))
    assert all(band in line for line in ansi_lines)


def test_sent_message_band_bleeds_into_margins_but_tool_cards_do_not() -> None:
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        Console(file=StringIO(), force_terminal=False),
        lambda: True,
        transcript.append,
    )
    presenter.print_user(user_message(Text("hello", style=theme.BODY)))
    call = ToolCall("read-1", "read", {"path": "maze.c"})
    card = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "int main(void) { return 0; }"),
        )
    )
    presenter.print_unit(card)
    width, height = 30, 12
    lines = transcript.lines(content_width(width))
    margins = TranscriptMargins(
        [
            Window(width=CONTENT_MARGIN, char=" "),
            transcript.window(),
            Window(width=CONTENT_MARGIN, char=" "),
        ]
    )
    screen = Screen()
    with create_app_session(output=DummyOutput()):
        margins.write_to_screen(
            screen, MouseHandlers(), WritePosition(0, 0, width, height), "", True, None
        )

    band = f"bg:{theme.active_palette().composer_fill}"
    top = height - len(lines)
    edges = (0, 1, width - 2, width - 1)
    for y in range(top, top + 3):
        assert all(band in screen.data_buffer[y][x].style for x in edges)
    for y in range(top + 3, height):
        assert not any(band in screen.data_buffer[y][x].style for x in edges)
    assert "".join(screen.data_buffer[top + 1][x].char for x in range(width)).strip() == "hello"


def test_queued_lines_list_each_message_on_one_truncated_row() -> None:
    def rows(
        messages: tuple[str, ...], width: int = 40, *, label: str = "main", main: int = 0
    ) -> list[str]:
        panel = QueuedPanel(label, messages, main)
        return "".join(text for _style, text in queued_lines(panel, width)).split("\n")

    assert rows(("fix the\nbuild",)) == ["  queued · main · 1", "  › fix the build"]
    many = rows(("a", "b", "c", "a much longer message that cannot fit on one row"))
    assert many == ["  queued · main · 4", "  › a", "  › b", "  › c", "    +1 more"]
    child = rows(("go on",), label="root:worker", main=2)
    assert child == ["  queued · root:worker · 1", "  › go on", "  main · 2 queued"]
    assert rows((), label="root:worker", main=1) == ["  main · 1 queued"]
    long = rows(("x" * 100,), width=30)[1]
    assert get_cwidth(long) == 2 + 2 + content_width(30) - 2
    assert long.endswith("...")


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
        transcript.append(user_message(Text("inspect the project")))
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
