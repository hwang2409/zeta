"""Bounded, asynchronous state for paged transcript and prompt history."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Protocol


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

    async def adjacent_visible_user(
        self, entry_id: str, *, direction: Literal["older", "newer"]
    ) -> str | None: ...

    async def search(self, query: str, *, limit: int = 200) -> tuple[str, ...]: ...


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
        self._loads: dict[
            tuple[int, str | None, str, str | None], asyncio.Task[TUIHistoryPage]
        ] = {}

    @property
    def pages(self) -> tuple[TUIHistoryPage, ...]:
        def position(descriptor: PageDescriptor) -> int:
            cursor = descriptor.page.older
            return int(cursor.value) if cursor is not None else -1

        return tuple(
            descriptor.page
            for descriptor in sorted(self._pages.values(), key=position)
        )

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

    async def close(self) -> None:
        tasks = tuple(self._loads.values())
        self.reset()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
        key = (
            self.generation,
            self.endpoint_id,
            direction,
            cursor.value if cursor else None,
        )
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
            candidates = [
                candidate
                for candidate, descriptor in self._pages.items()
                if candidate != key and descriptor.page.newer is not None
            ]
            if not candidates:
                candidates = [candidate for candidate in self._pages if candidate != key]
            self._pages.pop(candidates[0])

    def evict(self, key: str) -> None:
        self._pages.pop(key, None)

    async def reveal(self, entry_id: str) -> TUIHistoryPage:
        self._reveal_target = entry_id
        generation = self.generation
        page = await self.source.reveal(entry_id)
        if generation == self.generation:
            self.adopt(page)
        return page

    async def navigate_user(
        self, entry_id: str, direction: Literal["older", "newer"]
    ) -> str | None:
        target = await self.source.adjacent_visible_user(
            entry_id, direction=direction
        )
        if target is not None:
            await self.reveal(target)
        return target

    async def search(self, query: str) -> tuple[str, ...]:
        return await self.source.search(query)
