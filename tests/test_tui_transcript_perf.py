from __future__ import annotations

from unittest.mock import Mock

from rich.markdown import Markdown

from zeta.tui.transcript import TranscriptWidget


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
    assert counts == [1, 1]


def test_follow_tail_rendered_output_remains_available() -> None:
    transcript = _transcript(8)
    transcript.create_content(100, 30)
    assert "message 7" in "\n".join(transcript.lines(100))
    assert "markdown" in "\n".join(transcript.lines(100))
