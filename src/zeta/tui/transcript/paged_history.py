"""Bounded, asynchronous state for paged transcript and prompt history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Literal

from prompt_toolkit.document import Document


@dataclass(frozen=True, slots=True)
class HistoryCursor:
    value: str


@dataclass(frozen=True, slots=True)
class TUIHistoryUnit:
    entry_id: str
    value: Any
    unit_ordinal: int = 0
    text_offset: int = 0
    is_user: bool = False


@dataclass(frozen=True, slots=True)
class TUIPaintSegment:
    units: tuple[TUIHistoryUnit, ...]
    first_entry_id: str | None = None
    last_entry_id: str | None = None
    decoded_bytes: int = 0

    def __post_init__(self) -> None:
        if self.first_entry_id is None and self.units:
            object.__setattr__(self, "first_entry_id", self.units[0].entry_id)
        if self.last_entry_id is None and self.units:
            object.__setattr__(self, "last_entry_id", self.units[-1].entry_id)


@dataclass(frozen=True, slots=True)
class TUIHistoryPage:
    segments: tuple[TUIPaintSegment, ...]
    older: HistoryCursor | None = None
    newer: HistoryCursor | None = None
    endpoint_id: str | None = None
    repair_generation: str | None = None

    @property
    def units(self) -> tuple[TUIHistoryUnit, ...]:
        return tuple(unit for segment in self.segments for unit in segment.units)


class TUIHistorySource(Protocol):
    async def page(
        self,
        cursor: HistoryCursor | None = None,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> TUIHistoryPage: ...

    async def reveal(self, entry_id: str) -> TUIHistoryPage: ...

    async def prompt_page(
        self,
        cursor: HistoryCursor | None = None,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> tuple[tuple[tuple[str, str], ...], HistoryCursor | None, HistoryCursor | None]: ...


@dataclass(frozen=True, slots=True)
class PageDescriptor:
    key: str
    page: TUIHistoryPage


class PagedTranscriptHistory:
    """Own at most four transcript pages and coalesce edge reads."""

    max_pages = 4
    max_scalar_tokens = 6

    def __init__(self, source: TUIHistorySource) -> None:
        self.source = source
        self.generation = 0
        self.endpoint_id: str | None = None
        self._pages: dict[str, PageDescriptor] = {}
        self._older: HistoryCursor | None = None
        self._newer: HistoryCursor | None = None
        self._frozen_endpoint: str | None = None
        self._newest_tail: HistoryCursor | None = None
        self._anchor: tuple[str, int, int] | None = None
        self._reveal_target: str | None = None
        self._loads: dict[tuple[int, str, str | None], asyncio.Task[TUIHistoryPage]] = {}

    @property
    def pages(self) -> tuple[TUIHistoryPage, ...]:
        return tuple(descriptor.page for descriptor in self._pages.values())

    @property
    def descriptors(self) -> tuple[PageDescriptor, ...]:
        return tuple(self._pages.values())

    @property
    def scalar_tokens(self) -> tuple[object | None, ...]:
        return (
            self._older,
            self._newer,
            self._frozen_endpoint,
            self._newest_tail,
            self._anchor,
            self._reveal_target,
        )

    @property
    def anchor(self) -> tuple[str, int, int] | None:
        return self._anchor

    def set_anchor(self, entry_id: str, unit_ordinal: int, text_offset: int) -> None:
        self._anchor = (entry_id, unit_ordinal, text_offset)

    def reset(self, *, endpoint_id: str | None = None) -> None:
        self.generation += 1
        for task in self._loads.values():
            task.cancel()
        self._loads.clear()
        self._pages.clear()
        self._older = self._newer = self._newest_tail = None
        self._frozen_endpoint = endpoint_id
        self.endpoint_id = endpoint_id
        self._anchor = None
        self._reveal_target = None

    def _key(self, page: TUIHistoryPage, cursor: HistoryCursor | None) -> str:
        if page.units:
            return page.units[0].entry_id
        return f"{page.endpoint_id}:{cursor.value if cursor else 'initial'}"

    async def load(self, direction: Literal["older", "newer"] = "older") -> TUIHistoryPage:
        cursor = self._older if direction == "older" else self._newer
        key = (self.generation, direction, cursor.value if cursor else None)
        task = self._loads.get(key)
        if task is None:
            generation = self.generation
            task = asyncio.create_task(self.source.page(cursor, direction=direction))
            self._loads[key] = task
            try:
                page = await task
            finally:
                self._loads.pop(key, None)
            if generation != self.generation:
                return TUIHistoryPage(())
            self.adopt(page, direction=direction, cursor=cursor)
            return page
        return await task

    def adopt(
        self,
        page: TUIHistoryPage,
        *,
        direction: Literal["older", "newer"] = "older",
        cursor: HistoryCursor | None = None,
    ) -> None:
        if page.endpoint_id is not None:
            self.endpoint_id = page.endpoint_id
            self._frozen_endpoint = page.endpoint_id
        key = self._key(page, cursor)
        self._pages[key] = PageDescriptor(key, page)
        if direction == "older":
            self._older = page.older
        else:
            self._newer = page.newer
        if page.newer is None:
            self._newest_tail = cursor
        while len(self._pages) > self.max_pages:
            # Keep the newest insertion and discard the oldest resident page.
            self._pages.pop(next(iter(self._pages)))

    def evict(self, key: str) -> None:
        self._pages.pop(key, None)

    async def reveal(self, entry_id: str) -> TUIHistoryPage:
        self._reveal_target = entry_id
        generation = self.generation
        page = await self.source.reveal(entry_id)
        if generation == self.generation:
            self.adopt(page)
        return page


class SessionPromptHistory:
    """Bounded prompt navigation over one active viewed session."""

    max_pages = 2
    max_prompts = 128
    max_text_bytes = 256 * 1024

    def __init__(self, source: TUIHistorySource | None = None) -> None:
        self.source = source
        self.session_path: Path | None = None
        self.generation = 0
        self._pages: list[tuple[tuple[str, str], ...]] = []
        self._older: HistoryCursor | None = None
        self._newer: HistoryCursor | None = None
        self._selected: str | None = None
        self._draft: Document | None = None

    @property
    def pages(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        return tuple(self._pages)

    def bind(self, session_path: Path, source: TUIHistorySource | None = None) -> None:
        if source is not None:
            self.source = source
        if session_path == self.session_path:
            return
        self.generation += 1
        self.session_path = session_path
        self._pages.clear()
        self._older = self._newer = None
        self._selected = None
        self._draft = None

    async def load(self, direction: Literal["older", "newer"] = "older") -> None:
        if self.source is None:
            return
        cursor = self._older if direction == "older" else self._newer
        rows, older, newer = await self.source.prompt_page(cursor, direction=direction)
        if direction == "older":
            self._older = older
        else:
            self._newer = newer
        self._pages.append(rows)
        while len(self._pages) > self.max_pages:
            self._pages.pop(0 if direction == "older" else -1)

    def append(self, prompt_id: str, text: str) -> None:
        if not text:
            return
        if not self._pages:
            self._pages.append(())
        page = (*self._pages[-1], (prompt_id, text))
        if len(page) > self.max_prompts or sum(len(value.encode()) for _, value in page) > self.max_text_bytes:
            self._pages.append(((prompt_id, text),))
        else:
            self._pages[-1] = page
        while len(self._pages) > self.max_pages:
            self._pages.pop(0)
        self._selected = None

    def start(self, draft: Document) -> None:
        self._draft = draft
        self._selected = None

    def select(self, prompt_id: str) -> Document | None:
        for page in reversed(self._pages):
            for candidate_id, text in reversed(page):
                if candidate_id == prompt_id:
                    self._selected = candidate_id
                    return Document(text, cursor_position=len(text))
        return None

    def clear(self) -> None:
        self.generation += 1
        self._pages.clear()
        self._older = self._newer = None
        self._selected = None
        self._draft = None
