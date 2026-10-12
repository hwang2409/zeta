"""Composer history finder behavior and worker-thread coverage."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from prompt_toolkit.buffer import Buffer

from zeta.runtime.loop.persistence import history_for
from zeta.tui.transcript import FinderControl
from zeta.tui.transcript import history_finder as history_module
from zeta.tui.transcript.history_finder import HistoryFinder, load_history_candidates


def _write_history(path: Path, *entries: str) -> None:
    history = history_for(path)
    for entry in entries:
        history.append_string(entry)


async def _open(finder: HistoryFinder, buffer: Buffer) -> None:
    request = finder.open_finder(buffer)
    assert request is not None
    snapshot = await finder.build_finder_candidates(request)
    assert finder.finder_publish_candidates(request, snapshot)
    result = finder.finder_rank()
    assert result is not None
    assert finder.finder_publish(result)


def test_candidates_are_newest_first_deduplicated_and_multiline(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "older match", "two\nlines", "older match   ", "newest match")

    snapshot = load_history_candidates(path)

    assert [candidate.text for candidate in snapshot.candidates] == [
        "newest match",
        "older match",
        "two ⏎ lines",
    ]
    assert snapshot.candidates[2].preview == ("two", "lines")


def test_matching_text_and_preview_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "x" * 20_000)

    candidate = load_history_candidates(path).candidates[0]

    assert len(candidate.text) == 2_048
    assert sum(map(len, candidate.preview)) <= 12_000


@pytest.mark.asyncio
async def test_fuzzy_ranking_prefers_newer_duplicate_free_matches(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "run pytest old", "other", "run pytest newest")
    finder = HistoryFinder(path)
    await _open(finder, Buffer())

    finder.finder_set_query("rpt")
    result = finder.finder_rank()
    assert result is not None and finder.finder_publish(result)

    state = finder.finder_state()
    assert state is not None
    assert [row.candidate.text for row in state.rows] == [
        "run pytest newest",
        "run pytest old",
    ]


@pytest.mark.asyncio
async def test_accept_inserts_without_submitting_and_remains_editable(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "selected prompt")
    buffer = Buffer()
    buffer.text = "draft"
    finder = HistoryFinder(path)
    await _open(finder, buffer)
    finder.finder_set_query("")
    result = finder.finder_rank()
    assert result is not None and finder.finder_publish(result)

    assert finder.finder_accept()
    assert buffer.text == "selected prompt"
    assert buffer.cursor_position == len("selected prompt")
    buffer.insert_text(" edited")
    assert buffer.text == "selected prompt edited"
    buffer.undo()
    assert buffer.text == "draft"


@pytest.mark.asyncio
async def test_attachment_tokens_return_as_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "reuse [Image #1]")
    accepted: list[str] = []
    buffer = Buffer()
    finder = HistoryFinder(path, on_accept=lambda: accepted.append(buffer.text))
    await _open(finder, buffer)

    assert finder.finder_accept()
    assert accepted == [""]
    assert buffer.text == "reuse [Image #1]"


@pytest.mark.asyncio
async def test_cancel_restores_draft_and_cursor_exactly(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "selected prompt")
    buffer = Buffer()
    buffer.text = "draft text"
    buffer.cursor_position = 3
    finder = HistoryFinder(path)
    await _open(finder, buffer)
    buffer.text = "changed while open"

    finder.finder_cancel()

    assert buffer.text == "draft text"
    assert buffer.cursor_position == 3


@pytest.mark.asyncio
async def test_open_and_cancel_do_not_write_history(tmp_path: Path) -> None:
    path = tmp_path / "history"
    _write_history(path, "saved prompt")
    before = path.read_bytes()
    finder = HistoryFinder(path)
    await _open(finder, Buffer())

    finder.finder_cancel()

    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_history_loading_runs_off_the_event_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "history"
    _write_history(path, *(f"prompt {index}" for index in range(1_000)))
    loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    original = history_module.load_history_candidates

    def recording_loader(path: str | Path, limit: int):
        worker_threads.append(threading.get_ident())
        return original(path, limit)

    monkeypatch.setattr(history_module, "load_history_candidates", recording_loader)
    finder = HistoryFinder(path)
    request = finder.open_finder(Buffer())
    assert request is not None

    await finder.build_finder_candidates(request)

    assert worker_threads and worker_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_empty_history_has_empty_state_and_does_not_crash(tmp_path: Path) -> None:
    finder = HistoryFinder(tmp_path / "missing-history")
    await _open(finder, Buffer())
    state = finder.finder_state()
    assert state is not None and state.rows == ()

    content = FinderControl(finder.finder_state).create_content(80, 20)
    rendered = "".join(
        fragment[1]
        for line in range(content.line_count)
        for fragment in content.get_line(line)
    )
    assert "no history" in rendered
