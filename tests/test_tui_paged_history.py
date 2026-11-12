from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from prompt_toolkit.document import Document
from rich.text import Text

from zeta.tui.transcript.paged_history import (
    HistoryCursor,
    PagedTranscriptHistory,
    SessionPromptHistory,
    TUIHistoryPage,
    TUIHistoryUnit,
    TUIPaintSegment,
)
from zeta.tui.transcript.transcript import TranscriptWidget


class FakeSource:
    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def page(self, cursor=None, *, direction="older"):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        number = self.calls
        return TUIHistoryPage(
            (TUIPaintSegment((TUIHistoryUnit(str(number), Text(str(number))),),),),
            older=HistoryCursor(str(number + 1)),
            endpoint_id="endpoint",
        )

    async def reveal(self, entry_id):
        return TUIHistoryPage(
            (TUIPaintSegment((TUIHistoryUnit(entry_id, Text(entry_id)),),),),
            endpoint_id="endpoint",
        )

    async def prompt_page(self, cursor=None, *, direction="older"):
        return ((("id", "prompt"),),), None, None


@pytest.mark.asyncio
async def test_edge_loads_coalesce_and_resident_pages_are_bounded() -> None:
    source = FakeSource()
    pager = PagedTranscriptHistory(source)
    tasks = [asyncio.create_task(pager.load()) for _ in range(3)]
    await asyncio.wait_for(source.started.wait(), 10)
    assert source.calls == 1
    source.release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 10)
    assert len(pager.pages) == 1
    for _ in range(8):
        await asyncio.wait_for(pager.load(), 10)
    assert len(pager.pages) <= 4
    assert len(pager.descriptors) <= 4
    assert len(pager.scalar_tokens) == 6


def test_widget_bulk_adopt_and_remove_uses_real_render_path() -> None:
    widget = TranscriptWidget()
    units = widget.adopt_units((Text("old"), Text("new")))
    assert "old" in "\n".join(widget.lines(80))
    widget.create_content(80, 5)
    widget.remove_units(units)
    assert widget.units == ()


def test_prompt_owner_keeps_only_two_pages_and_exact_draft() -> None:
    owner = SessionPromptHistory()
    owner.bind(Path("session"))
    owner.append("one", "a")
    owner.start(Document("draft", cursor_position=2))
    assert owner.select("one") == Document("a", cursor_position=1)
    for index in range(300):
        owner.append(str(index), "x")
    assert len(owner.pages) <= 2
    owner.clear()
    assert owner.pages == ()
