"""Composer history finder behavior and worker-thread coverage."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest
from prompt_toolkit.buffer import Buffer

from zeta.runtime.loop.persistence import BoundedFileHistory, history_for
from zeta.tui.transcript import FinderControl
from zeta.tui.transcript.history_finder import (
    HistoryFinder,
    build_history_candidates,
)


def _history(path: Path, *entries: str) -> BoundedFileHistory:
    history = history_for(path)
    for entry in entries:
        history.append_string(entry)
    return history


def _finder(
    history: BoundedFileHistory, *, on_accept=None
) -> HistoryFinder:
    return HistoryFinder(
        history.load_from_disk_once,
        history.snapshot,
        on_accept=on_accept,
    )


async def _open(finder: HistoryFinder, buffer: Buffer) -> None:
    request = finder.open_finder(buffer)
    assert request is not None
    snapshot = await finder.build_finder_candidates(request)
    assert finder.finder_publish_candidates(request, snapshot)
    result = finder.finder_rank()
    assert result is not None
    assert finder.finder_publish(result)


def test_candidates_are_newest_first_deduplicated_and_multiline(tmp_path: Path) -> None:
    history = _history(
        tmp_path / "history",
        "older match",
        "two\nlines",
        "older match   ",
        "newest match",
    )

    snapshot = build_history_candidates(history.snapshot())

    assert [candidate.text for candidate in snapshot.candidates] == [
        "newest match",
        "older match",
        "two ⏎ lines",
    ]
    assert snapshot.candidates[2].preview == ("two", "lines")


def test_matching_text_and_preview_are_bounded(tmp_path: Path) -> None:
    history = _history(tmp_path / "history", "x" * 20_000)

    candidate = build_history_candidates(history.snapshot()).candidates[0]

    assert len(candidate.text) == 2_048
    assert sum(map(len, candidate.preview)) <= 12_000


@pytest.mark.asyncio
async def test_fuzzy_ranking_prefers_newer_duplicate_free_matches(tmp_path: Path) -> None:
    history = _history(
        tmp_path / "history", "run pytest old", "other", "run pytest newest"
    )
    finder = _finder(history)
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
    history = _history(tmp_path / "history", "selected prompt")
    buffer = Buffer()
    buffer.text = "draft"
    finder = _finder(history)
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
    history = _history(tmp_path / "history", "reuse [Image #1]")
    accepted: list[str] = []
    buffer = Buffer()
    finder = _finder(history, on_accept=lambda: accepted.append(buffer.text))
    await _open(finder, buffer)

    assert finder.finder_accept()
    assert accepted == [""]
    assert buffer.text == "reuse [Image #1]"


@pytest.mark.asyncio
async def test_cancel_restores_draft_and_cursor_exactly(tmp_path: Path) -> None:
    history = _history(tmp_path / "history", "selected prompt")
    buffer = Buffer()
    buffer.text = "draft text"
    buffer.cursor_position = 3
    finder = _finder(history)
    await _open(finder, buffer)
    buffer.text = "changed while open"

    finder.finder_cancel()

    assert buffer.text == "draft text"
    assert buffer.cursor_position == 3


@pytest.mark.asyncio
async def test_open_and_cancel_do_not_write_history(tmp_path: Path) -> None:
    path = tmp_path / "history"
    history = _history(path, "saved prompt")
    await history.flush_pending()
    before = path.read_bytes()
    finder = _finder(history)
    await _open(finder, Buffer())

    finder.finder_cancel()

    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_fresh_finder_loads_persisted_history_once_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history"
    history_for(path).store_string("saved prompt")
    history = history_for(path)
    loop_thread = threading.get_ident()
    load_threads: list[int] = []
    original = history.load_history_strings

    def recording_loader() -> list[str]:
        load_threads.append(threading.get_ident())
        return original()

    monkeypatch.setattr(history, "load_history_strings", recording_loader)
    finder = _finder(history)
    await _open(finder, Buffer())
    loaded_for_prompt = [entry async for entry in history.load()]

    state = finder.finder_state()
    assert state is not None
    assert [row.candidate.text for row in state.rows] == ["saved prompt"]
    assert loaded_for_prompt == ["saved prompt"]
    assert len(load_threads) == 1
    assert load_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_unreadable_history_is_terminal_and_keeps_in_memory_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    history = _history(tmp_path / "history", "in-memory prompt")
    await history.flush_pending()
    notices: list[str] = []
    reads = 0

    def unreadable() -> list[str]:
        nonlocal reads
        reads += 1
        raise PermissionError("history is unreadable")

    monkeypatch.setattr(history, "load_history_strings", unreadable)
    history.set_notice_sink(notices.append)
    with caplog.at_level(logging.WARNING):
        await asyncio.gather(
            asyncio.to_thread(history.load_from_disk_once),
            asyncio.to_thread(history.load_from_disk_once),
        )
        loaded_for_prompt = [entry async for entry in history.load()]
        finder = _finder(history)
        await _open(finder, Buffer())

    state = finder.finder_state()
    assert state is not None
    assert loaded_for_prompt == ["in-memory prompt"]
    assert [row.candidate.text for row in state.rows] == ["in-memory prompt"]
    assert reads == 1
    assert notices == ["composer history could not be read: history is unreadable"]
    assert [
        record.message
        for record in caplog.records
        if "history is unreadable" in record.message
    ] == ["composer history could not be read: history is unreadable"]


@pytest.mark.asyncio
async def test_history_does_not_start_loading_after_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = history_for(tmp_path / "history")
    reads = 0

    def recording_loader() -> list[str]:
        nonlocal reads
        reads += 1
        return []

    monkeypatch.setattr(history, "load_history_strings", recording_loader)
    await history.shutdown_pending()
    await asyncio.to_thread(history.load_from_disk_once)
    assert [entry async for entry in history.load()] == []

    assert reads == 0


@pytest.mark.asyncio
async def test_empty_history_has_empty_state_and_does_not_crash(tmp_path: Path) -> None:
    finder = _finder(history_for(tmp_path / "missing-history"))
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
