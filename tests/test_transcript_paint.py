"""Per-unit paint caches: a streaming token must not re-parse the whole transcript."""

from __future__ import annotations

import re
from io import StringIO

from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from rich.console import Console
from rich.text import Text

from zeta.tui import theme
from zeta.tui.render import render_event, render_markdown
from zeta.tui.transcript import TranscriptWidget, _ToolUnit, _TranscriptUnit
from zeta.tui.transcript_presenter import TranscriptPresenter
from zeta.types import ErrorInfo, StreamEvent, StreamEventType, ToolCall, ToolResult

MARKDOWN = (
    "## Answer\n\nSome **bold** text and `code` here.\n\n"
    "| name | value |\n|---|---|\n| a | 1 |\n\n```python\nprint('hi')\n```\n"
    "- item one\n- item two\n"
)


def _mixed_transcript() -> TranscriptWidget:
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        Console(file=StringIO(), force_terminal=False),
        lambda: True,
        transcript.append,
    )
    transcript.append_blank()  # leading separator, trimmed by the paint
    presenter.print_user(Text.assemble(("▌ ", theme.USER_ROLE), ("question one", theme.BODY)))
    presenter.print_unit(render_markdown(MARKDOWN))
    call = ToolCall("call-1", "bash", {"command": "ls"})
    presenter.handle_tool_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call), aborted=False
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "README.md\npyproject.toml"),
        ),
        aborted=False,
    )
    presenter.print_unit(
        render_event(
            StreamEvent(StreamEventType.ERROR, error=ErrorInfo("http_error", "boom " * 30))
        )
    )
    presenter.print_unit(Text("streaming reply " * 5))
    transcript.append_blank()  # trailing separator, trimmed by the paint
    return transcript


def _reference_locations(
    transcript: TranscriptWidget, width: int
) -> list[tuple[_TranscriptUnit | None, int]]:
    """The pre-cache algorithm, kept verbatim as the oracle for the per-unit version."""

    raw_lines: list[tuple[str, _TranscriptUnit | None, int]] = []
    for unit in transcript._units:
        rendered = transcript._render_unit(unit, width)
        rendered_lines = transcript._plain_lines(rendered)
        renderable = (
            unit.value.renderable if isinstance(unit.value, _ToolUnit) else unit.value
        )
        source = getattr(renderable, "plain", None)
        if not isinstance(source, str):
            source = "\n".join(transcript._strip_padding(line) for line in rendered_lines)
        source_offset = 0
        for line in rendered_lines:
            content = transcript._strip_padding(line)
            offset = source.find(content, source_offset)
            matched_length = len(content)
            if offset < 0:
                match = re.search(r"[\w]+(?:[-'][\w]+)*", content)
                if match is not None:
                    offset = source.find(match.group(), source_offset)
                    matched_length = len(match.group())
                if offset < 0:
                    offset = source_offset
            raw_lines.append((line, unit, offset))
            source_offset = offset + matched_length
    while raw_lines and not raw_lines[0][0].strip():
        raw_lines.pop(0)
    return [(unit, text_offset) for _, unit, text_offset in raw_lines]


def _without_empty_fragments(
    lines: list[list[tuple[str, str]]],
) -> list[list[tuple[str, str]]]:
    # The whole-string parse leaves zero-width ('', '') fragments around each
    # newline; they paint nothing and occupy no column, so they are noise here.
    return [[fragment for fragment in line if fragment[1]] for line in lines]


def test_assembled_lines_match_the_whole_string_parse() -> None:
    transcript = _mixed_transcript()

    for width in (40, 72, 120):
        expected = list(split_lines(to_formatted_text(ANSI(transcript.render(width))))) or [[]]
        assert _without_empty_fragments(transcript._parsed_lines(width)) == (
            _without_empty_fragments(expected)
        )
        assert transcript._locations(width) == _reference_locations(transcript, width)
        # A trailing blank unit is trimmed by the paint but kept by the location
        # table (pre-existing); every line that is painted has a location.
        assert len(transcript._parsed_lines(width)) <= len(transcript._locations(width))


def test_streaming_token_reparses_only_the_changed_unit() -> None:
    transcript = TranscriptWidget()
    units = [transcript.append(Text(f"line {index}")) for index in range(40)]
    streaming = units[-1]
    transcript.create_content(60, 20)
    lines_before = {key: id(entry[1]) for key, entry in transcript._unit_lines_cache.items()}
    locations_before = {
        key: id(entry[2]) for key, entry in transcript._unit_locations_cache.items()
    }
    assert len(lines_before) == 40

    transcript.replace(streaming, Text("line 39 plus a token"))
    transcript.create_content(60, 20)

    lines_after = {key: id(entry[1]) for key, entry in transcript._unit_lines_cache.items()}
    locations_after = {
        key: id(entry[2]) for key, entry in transcript._unit_locations_cache.items()
    }
    changed_lines = {key for key in lines_before if lines_before[key] != lines_after[key]}
    changed_locations = {
        key for key in locations_before if locations_before[key] != locations_after[key]
    }
    assert changed_lines == {streaming.key}
    assert changed_locations == {streaming.key}
    assert "".join(f[1] for f in transcript._parsed_lines(60)[-1]) == "line 39 plus a token"


def test_removed_and_cleared_units_leave_the_per_unit_caches() -> None:
    transcript = TranscriptWidget()
    kept = transcript.append(Text("kept"))
    gone = transcript.append(Text("gone"))
    transcript.create_content(60, 10)
    assert gone.key in transcript._unit_lines_cache

    transcript.remove(gone)

    assert gone.key not in transcript._unit_lines_cache
    assert gone.key not in transcript._unit_locations_cache
    assert kept.key in transcript._unit_lines_cache

    transcript.clear()

    assert not transcript._unit_lines_cache
    assert not transcript._unit_locations_cache
    assert transcript._keyed_cache is None


def test_search_highlight_path_still_paints_matches() -> None:
    transcript = _mixed_transcript()
    transcript.create_content(72, 20)

    transcript.begin_search()
    transcript.update_search("README")
    lines = transcript._parsed_lines(72)

    def style_at(line: list[tuple[str, str]], column: int) -> str:
        position = 0
        for style, text in line:
            if position <= column < position + len(text):
                return style
            position += len(text)
        raise AssertionError("column past the end of the line")

    plain = ["".join(fragment[1] for fragment in line) for line in lines]
    match_line = next(index for index, text in enumerate(plain) if "README" in text)
    column = plain[match_line].index("README")
    assert "bg:" in style_at(lines[match_line], column)

    transcript.end_search()

    assert "bg:" not in style_at(transcript._parsed_lines(72)[match_line], column)
    assert transcript._parsed_lines(72) == transcript._assembled_lines(72)
