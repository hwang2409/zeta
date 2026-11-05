"""Composer history finder and in-memory session history behavior."""

from __future__ import annotations

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from zeta.tui.session_prompt_history import SessionPromptHistory
from zeta.tui.transcript import FinderControl
from zeta.tui.transcript.history_finder import HistoryFinder, build_history_candidates


def _finder(history: SessionPromptHistory, path: str, *, on_accept=None) -> HistoryFinder:
    return HistoryFinder(lambda: None, lambda: history.snapshot(path), on_accept=on_accept)


async def _open(finder: HistoryFinder, buffer: Buffer) -> None:
    request = finder.open_finder(buffer)
    assert request is not None
    snapshot = await finder.build_finder_candidates(request)
    assert finder.finder_publish_candidates(request, snapshot)
    result = finder.finder_rank()
    assert result is not None
    assert finder.finder_publish(result)


def test_candidates_are_newest_first_deduplicated_and_multiline() -> None:
    snapshot = build_history_candidates(("older match", "two\nlines", "older match   ", "newest match"))
    assert [candidate.text for candidate in snapshot.candidates] == [
        "newest match", "older match", "two ⏎ lines"
    ]
    assert snapshot.candidates[2].preview == ("two", "lines")


def test_session_history_is_scoped_and_navigation_restores_draft() -> None:
    history = SessionPromptHistory()
    history.replace("a", ("a one", "a two"))
    history.replace("b", ("b one",))
    assert history.older("a", Document("draft")).text == "a two"
    assert history.older("a", Document("a two")).text == "a one"
    assert history.older("b", Document("draft")).text == "b one"
    assert history.newer("b").text == "draft"


@pytest.mark.asyncio
async def test_finder_uses_viewed_session_snapshot() -> None:
    history = SessionPromptHistory()
    history.replace("a", ("a prompt",))
    history.replace("b", ("b prompt",))
    finder = _finder(history, "b")
    await _open(finder, Buffer())
    state = finder.finder_state()
    assert state is not None
    assert [row.candidate.text for row in state.rows] == ["b prompt"]


@pytest.mark.asyncio
async def test_accept_and_cancel_preserve_draft() -> None:
    history = SessionPromptHistory()
    history.replace("a", ("selected prompt",))
    buffer = Buffer()
    buffer.text = ""
    finder = _finder(history, "a")
    await _open(finder, buffer)
    assert finder.finder_accept()
    assert buffer.text == "selected prompt"


@pytest.mark.asyncio
async def test_empty_history_renders_empty_state() -> None:
    finder = _finder(SessionPromptHistory(), "missing")
    await _open(finder, Buffer())
    content = FinderControl(finder.finder_state).create_content(80, 20)
    rendered = "".join(
        fragment[1]
        for line in range(content.line_count)
        for fragment in content.get_line(line)
    )
    assert "no history" in rendered
