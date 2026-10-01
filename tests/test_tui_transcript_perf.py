from __future__ import annotations

from unittest.mock import Mock

import pytest
from rich.markdown import Markdown
from rich.text import Text

from zeta.protocol.types import Message, MessageRole, ToolResult
from zeta.tui import checkpoints as checkpoints_module
from zeta.tui import render as render_module
from zeta.tui import theme
from zeta.tui.composer import TurnConsumerMixin
from zeta.tui.render import render_markdown
from zeta.tui.transcript import TranscriptPresenter, TranscriptWidget


def _streaming_transcript() -> tuple[TranscriptWidget, TranscriptPresenter]:
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        Mock(),
        lambda: True,
        transcript.append,
    )
    return transcript, presenter


def _transcript(messages: int) -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(messages):
        transcript.append(
            Markdown(
                f"message {index}\n\n"
                "A paragraph with **markdown** and `code` that stays visible."
            )
        )
        transcript.append_blank()
    return transcript


def test_follow_tail_redraw_does_not_rebuild_location_map() -> None:
    counts: list[int] = []
    for size in (100, 2_000):
        transcript = _transcript(size)
        original = transcript._locations
        locations = Mock(wraps=original)
        transcript._locations = locations
        transcript.create_content(100, 30)
        transcript.create_content(100, 30)
        counts.append(locations.call_count)
    assert counts == [0, 0]


def test_follow_tail_rendered_output_remains_available() -> None:
    transcript = _transcript(8)
    transcript.create_content(100, 30)
    assert "message 7" in "\n".join(transcript.lines(100))
    assert "markdown" in "\n".join(transcript.lines(100))


def _content_text(transcript: TranscriptWidget, width: int, height: int) -> str:
    content = transcript.create_content(width, height)
    return "\n".join(
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    )


def test_resume_renders_only_visible_units_eagerly() -> None:
    render_counts: list[int] = []
    for size in (500, 2_000):
        transcript = _transcript(size)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        transcript.create_content(100, 30)
        render_counts.append(rendered.call_count)

    assert render_counts[0] == render_counts[1]
    assert render_counts[0] < 30


def test_older_units_render_on_scroll() -> None:
    transcript = _transcript(500)
    rendered = Mock(wraps=transcript._render_unit)
    transcript._render_unit = rendered
    transcript.create_content(100, 30)
    eager_count = rendered.call_count

    transcript.page_up()
    transcript.create_content(100, 30)

    assert rendered.call_count > eager_count
    assert rendered.call_count >= 500


def test_resume_defers_markdown_parsing_outside_viewport(
    monkeypatch,
) -> None:
    original_parse = render_module._MARKDOWN.parse
    parse = Mock(wraps=original_parse)
    monkeypatch.setattr(render_module._MARKDOWN, "parse", parse)
    transcript = TranscriptWidget()
    for index in range(500):
        transcript.append(render_markdown(f"message **{index}**"))
        transcript.append_blank()

    transcript.create_content(72, 24)

    assert parse.call_count < 24


def test_resume_does_not_sanitize_entire_tool_result_before_bounded_render(
    monkeypatch,
) -> None:
    sanitize = Mock(wraps=checkpoints_module.strip_terminal_controls)
    monkeypatch.setattr(checkpoints_module, "strip_terminal_controls", sanitize)
    message = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("call", "output\n" * 10_000),
    )

    sanitized = checkpoints_module._sanitize_replayed_message(message)

    assert sanitized.tool_result is message.tool_result
    sanitize.assert_not_called()


def test_resume_output_matches_eager_render() -> None:
    transcript = _transcript(500)
    eager_lines = transcript._parsed_lines(72)
    expected = "\n".join(
        "".join(fragment[1] for fragment in line) for line in eager_lines[-24:]
    )

    lazy = _transcript(500)
    assert _content_text(lazy, 72, 24) == expected
    lazy.page_up()
    assert lazy.lines(72) == transcript.lines(72)


def _line_transcript() -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(200):
        unit = transcript.append(Text(f"line {index}"))
        if index % 50 == 0:
            transcript.mark_user(unit)
    return transcript


def test_lazy_tail_previous_user_message_navigates_from_tail() -> None:
    transcript = _line_transcript()
    transcript.create_content(80, 10)

    assert transcript.previous_user_message()
    assert transcript.scroll_offset == 150


def test_lazy_tail_next_user_message_stays_at_tail() -> None:
    transcript = _line_transcript()
    transcript.create_content(80, 10)

    assert not transcript.next_user_message()
    assert transcript.follow_tail


def test_lazy_tail_page_up_continues_from_visible_position() -> None:
    transcript = _line_transcript()
    transcript.create_content(80, 10)

    transcript.page_up()
    content = transcript.create_content(80, 10)

    assert transcript.scroll_offset == 180
    assert "line 180" in "".join(text for _, text in content.get_line(180))


@pytest.mark.parametrize("size", [2_000, 20_000])
def test_streaming_assistant_paint_work_is_bounded_per_delta(size: int) -> None:
    transcript, presenter = _streaming_transcript()
    delta = "abcdefghij" * 10
    rendered_characters = 0
    original = transcript._render_unit

    def counted_render(unit, width):
        nonlocal rendered_characters
        value = unit.value
        if isinstance(value, Text):
            rendered_characters += len(value.plain)
        return original(unit, width)

    transcript._render_unit = counted_render
    for _ in range(size // len(delta)):
        presenter.append_assistant(delta)
        transcript.create_content(80, 24)

    assert rendered_characters <= (size // len(delta)) * 500


def test_streaming_assistant_paint_work_is_history_independent() -> None:
    counts: list[int] = []
    for history_size in (100, 2_000, 10_000):
        transcript = _transcript(history_size)
        presenter = TranscriptPresenter(
            transcript,
            Mock(),
            lambda: True,
            transcript.append,
        )
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        for _ in range(20):
            presenter.append_assistant("abcdefghij" * 10)
            transcript.create_content(80, 24)
        counts.append(rendered.call_count)

    assert max(counts) - min(counts) <= 2


@pytest.mark.parametrize("width", [7, 13, 40, 80])
@pytest.mark.parametrize(
    "value",
    [
        "plain words wrap exactly like rich text does",
        "wide: 界🙂 café e\u0301 and tabs\there",
        "markup [bold]is literal[/bold] and ansi \x1b[31m stays literal",
        "first line\n\nsecond line with trailing spaces   \nlast",
    ],
)
def test_streaming_assistant_live_output_matches_text(value: str, width: int) -> None:
    expected = TranscriptWidget()
    expected.append(Text(value, style=theme.BODY))
    transcript, presenter = _streaming_transcript()
    for start in range(0, len(value), 3):
        presenter.append_assistant(value[start : start + 3])
        transcript.create_content(width, 40)

    assert _content_text(transcript, width, 40) == _content_text(expected, width, 40)
    assert transcript.lines(width) == expected.lines(width)


def test_streaming_thinking_uses_the_incremental_unit() -> None:
    transcript, presenter = _streaming_transcript()
    rendered = Mock(wraps=transcript._render_unit)
    transcript._render_unit = rendered
    for _ in range(200):
        presenter.append_thinking("reasoning delta ")
        transcript.create_content(80, 24)

    assert rendered.call_count == 0
    assert "reasoning delta" in _content_text(transcript, 80, 24)


@pytest.mark.asyncio
async def test_streaming_repaints_are_coalesced_and_final_paint_is_immediate() -> None:
    consumer = TurnConsumerMixin()
    paints = Mock()
    consumer._invalidate_prompt = paints

    for _ in range(1_000):
        consumer._invalidate_stream_prompt()

    assert paints.call_count == 1
    consumer._finish_stream_invalidation()
    assert paints.call_count == 2


def test_lazy_tail_disabled_when_max_lines_set() -> None:
    marker = "[custom older output]"

    def limited_transcript() -> TranscriptWidget:
        transcript = TranscriptWidget(max_lines=5)
        transcript.set_line_limit_marker(marker)
        for index in range(128):
            transcript.append(Text(f"line {index}"))
        return transcript

    expected = limited_transcript()._parsed_lines(80)
    transcript = limited_transcript()
    content = transcript.create_content(80, 10)
    actual = [content.get_line(index) for index in range(content.line_count)]

    assert not transcript._lazy_viewport
    assert actual == ([[]] * (10 - len(expected))) + expected
    assert marker in "".join(text for _, text in expected[0])
    assert content.line_count == 10
