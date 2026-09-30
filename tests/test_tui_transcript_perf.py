from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text


class _FakeClock:
    def __init__(self, tick: float = 0.0) -> None:
        self.now = 0.0
        self.tick = tick

    def __call__(self) -> float:
        value = self.now
        self.now += self.tick
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ManualScheduler:
    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []

    def __call__(self, callback: Callable[[], None]) -> None:
        self.pending.append(callback)

    def step(self) -> None:
        self.pending.pop(0)()

    def drain(self) -> None:
        while self.pending:
            self.step()

from zeta.protocol.types import Message, MessageRole, ToolResult
from zeta.tui import checkpoints as checkpoints_module
from zeta.tui import render as render_module
from zeta.tui.render import render_markdown
from zeta.tui.transcript import TranscriptWidget


class _RecordingRenderable:
    def __init__(self) -> None:
        self.calls = 0

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        del console, options
        self.calls += 1
        yield Text("expensive renderable")


def _prewarm_with_oldest(renderable: object) -> tuple[TranscriptWidget, _ManualScheduler]:
    scheduler = _ManualScheduler()
    transcript = TranscriptWidget(prewarm_scheduler=scheduler)
    transcript.append(renderable)  # type: ignore[arg-type]
    for index in range(127):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 10)
    scheduler.drain()
    return transcript, scheduler


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


def _line_transcript(
    *,
    scheduler: _ManualScheduler | None = None,
    chunk_size: int = 16,
    clock: Callable[[], float] | None = None,
    time_budget: float = 0.008,
) -> TranscriptWidget:
    transcript = TranscriptWidget(
        prewarm_scheduler=scheduler,
        prewarm_chunk_size=chunk_size,
        prewarm_clock=clock,
        prewarm_time_budget=time_budget,
    )
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


def test_prewarm_callback_respects_time_budget() -> None:
    scheduler = _ManualScheduler()
    clock = _FakeClock()
    transcript = TranscriptWidget(
        prewarm_scheduler=scheduler,
        prewarm_chunk_size=200,
        prewarm_time_budget=0.008,
        prewarm_clock=clock,
    )
    for index in range(200):
        transcript.append(Text(f"line {index}"))
    original = transcript._render_unit

    def timed_render(*args, **kwargs):
        clock.advance(0.003)
        return original(*args, **kwargs)

    rendered = Mock(side_effect=timed_render)
    transcript._render_unit = rendered
    transcript.create_content(80, 10)
    before = rendered.call_count

    scheduler.step()

    assert rendered.call_count - before <= 3
    assert scheduler.pending


def test_prewarm_skips_fenced_code_markdown() -> None:
    document = render_markdown("before\n\n```python\nprint('slow')\n```\n")
    transcript, _ = _prewarm_with_oldest(document)

    assert not document._parsed
    assert 80 not in transcript._parsed_cache


def test_prewarm_skips_markdown_table() -> None:
    document = render_markdown("| name | value |\n| --- | --- |\n| a | b |\n")
    transcript, _ = _prewarm_with_oldest(document)

    assert not document._parsed
    assert 80 not in transcript._parsed_cache


def test_prewarm_skips_direct_and_nested_syntax() -> None:
    direct = Syntax("print('slow')", "python")
    nested = Group(Syntax("print('nested')", "python"))

    for renderable in (direct, nested):
        transcript, _ = _prewarm_with_oldest(renderable)
        unit = transcript._units[0]
        assert unit is not None
        assert unit.key not in transcript._render_cache
        assert 80 not in transcript._parsed_cache


def test_prewarm_skips_unknown_renderable() -> None:
    renderable = _RecordingRenderable()
    transcript, _ = _prewarm_with_oldest(renderable)

    assert renderable.calls == 0
    assert 80 not in transcript._parsed_cache


def test_prewarm_still_warms_plain_and_small_markdown() -> None:
    scheduler = _ManualScheduler()
    transcript = TranscriptWidget(prewarm_scheduler=scheduler)
    unknown = _RecordingRenderable()
    transcript.append(unknown)
    plain = transcript.append(Text("plain text"))
    document = render_markdown("A small paragraph with **emphasis**.")
    markdown = transcript.append(document)
    for index in range(125):
        transcript.append(Text(f"line {index}"))

    transcript.create_content(80, 10)
    scheduler.drain()

    assert unknown.calls == 0
    assert document._parsed
    assert plain.key in transcript._render_cache
    assert markdown.key in transcript._render_cache


def test_prewarm_cost_guard_rejects_expensive_renderable() -> None:
    renderable = _RecordingRenderable()
    _, scheduler = _prewarm_with_oldest(renderable)

    assert renderable.calls == 0
    assert not scheduler.pending


def test_skipped_units_render_on_scroll() -> None:
    renderable = _RecordingRenderable()
    transcript, _ = _prewarm_with_oldest(renderable)

    assert renderable.calls == 0
    transcript.page_up()
    transcript.create_content(80, 10)

    assert renderable.calls == 1
    assert "expensive renderable" in transcript.lines(80)


def test_prewarm_skips_or_defers_oversized_unit() -> None:
    scheduler = _ManualScheduler()
    clock = _FakeClock()
    transcript = TranscriptWidget(
        prewarm_scheduler=scheduler,
        prewarm_chunk_size=200,
        prewarm_time_budget=0.008,
        prewarm_clock=clock,
        prewarm_max_unit_chars=64,
    )
    oversized = transcript.append(Text("x" * 1_000))
    for index in range(127):
        transcript.append(Text(f"line {index}"))
    rendered = Mock(wraps=transcript._render_unit)
    transcript._render_unit = rendered

    transcript.create_content(80, 10)
    scheduler.drain()

    assert all(call.args[0] is not oversized for call in rendered.call_args_list)
    assert 80 not in transcript._parsed_cache


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


def test_prewarm_final_assembly_is_incremental() -> None:
    scheduler = _ManualScheduler()
    clock = _FakeClock(tick=0.003)
    transcript = _line_transcript(
        scheduler=scheduler,
        chunk_size=200,
        clock=clock,
        time_budget=0.008,
    )
    for unit in transcript._units:
        assert unit is not None
        transcript._unit_parsed_lines(unit, 80)
        transcript._unit_locations(unit, 80)

    transcript.create_content(80, 10)
    while transcript._prewarm.phase == "render":
        scheduler.step()
    assert transcript._prewarm.phase == "assemble"

    scheduler.step()
    scheduler.step()

    assert transcript._prewarm.phase == "assemble"
    scheduler.drain()
    assert len(transcript._parsed_cache[80]) == 200
    assert len(transcript._locations_cache[80][1]) == 200


def test_prewarm_materializes_history_in_bounded_chunks() -> None:
    scheduler = _ManualScheduler()
    transcript = _line_transcript(scheduler=scheduler, chunk_size=7)
    rendered = Mock(wraps=transcript._render_unit)
    transcript._render_unit = rendered
    transcript.create_content(80, 10)
    eager_count = rendered.call_count

    while scheduler.pending:
        before = rendered.call_count
        scheduler.step()
        assert rendered.call_count - before <= 7

    assert rendered.call_count > eager_count
    assert len(transcript._parsed_cache[80]) == 200


def test_prewarm_cancels_on_resize_and_restarts() -> None:
    scheduler = _ManualScheduler()
    transcript = _line_transcript(scheduler=scheduler, chunk_size=7)
    transcript.create_content(80, 10)
    old_step = scheduler.pending.pop(0)

    transcript.create_content(40, 10)
    old_step()
    scheduler.drain()

    assert 40 in transcript._parsed_cache
    assert 80 not in transcript._parsed_cache
    assert all(cache[0] == 40 for cache in transcript._render_cache.values())


def test_scroll_back_after_prewarm_does_no_synchronous_full_materialization() -> None:
    scheduler = _ManualScheduler()
    transcript = _line_transcript(scheduler=scheduler, chunk_size=7)
    transcript.create_content(80, 10)
    scheduler.drain()
    rendered = Mock(wraps=transcript._render_unit)
    transcript._render_unit = rendered

    transcript.page_up()
    transcript.create_content(80, 10)

    rendered.assert_not_called()
