"""Bounded, asynchronous state for paged transcript and prompt history."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HistoryCursor:
    value: str


@dataclass(frozen=True, slots=True)
class HistorySearchResult:
    entry_id: str
    unit_ordinal: int = 0


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

    def prompt_pages(
        self,
    ) -> AsyncIterator[tuple[tuple[str, str], ...]]: ...

    async def adjacent_visible_user(
        self, entry_id: str, *, direction: Literal["older", "newer"]
    ) -> str | None: ...

    async def search(
        self, query: str, *, limit: int = 200
    ) -> tuple[HistorySearchResult, ...]: ...


@dataclass(frozen=True, slots=True)
class PageDescriptor:
    key: str
    page: TUIHistoryPage


class PagedTranscriptHistory:
    """Own one contiguous, bounded deque of transcript paint segments."""

    max_pages = 4
    max_scalar_tokens = 6
    max_segment_rows = 128
    max_segment_bytes = 1024 * 1024

    def __init__(self, source: TUIHistorySource) -> None:
        self.source: TUIHistorySource | None = source
        self.generation = 0
        self.endpoint_id: str | None = None
        self._segments: deque[PageDescriptor] = deque()
        self._older: HistoryCursor | None = None
        self._newer: HistoryCursor | None = None
        self._frozen_endpoint: str | None = None
        self._newest_tail: HistoryCursor | None = None
        self._anchor: tuple[str, int, int] | None = None
        self._reveal_target: str | None = None
        self._loads: dict[
            tuple[int, str | None, str, str | None], asyncio.Task[TUIHistoryPage]
        ] = {}
        self._live_sequence = 0

    @property
    def pages(self) -> tuple[TUIHistoryPage, ...]:
        seen: set[tuple[str, int]] = set()
        pages: list[TUIHistoryPage] = []
        for descriptor in self._segments:
            units: list[TUIHistoryUnit] = []
            for unit in descriptor.page.units:
                identity = (unit.entry_id, unit.unit_ordinal)
                if identity not in seen:
                    seen.add(identity)
                    units.append(unit)
            if units:
                pages.append(
                    TUIHistoryPage(
                        (TUIPaintSegment(tuple(units), decoded_bytes=descriptor.page.segments[0].decoded_bytes),),
                        older=descriptor.page.older,
                        newer=descriptor.page.newer,
                        endpoint_id=descriptor.page.endpoint_id,
                        repair_generation=descriptor.page.repair_generation,
                    )
                )
        return tuple(pages)

    @property
    def descriptors(self) -> tuple[PageDescriptor, ...]:
        return tuple(self._segments)

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

    @property
    def can_load_older(self) -> bool:
        return self._older is not None

    @property
    def can_load_newer(self) -> bool:
        return self._newer is not None

    def set_anchor(self, entry_id: str, unit_ordinal: int, text_offset: int) -> None:
        self._anchor = (entry_id, unit_ordinal, text_offset)

    async def close(self) -> None:
        """Invalidate publication without waiting for executor-backed reads."""

        self.reset()
        self.source = None
        await asyncio.sleep(0)

    def reset(self, *, endpoint_id: str | None = None) -> None:
        self.generation += 1
        for task in self._loads.values():
            task.cancel()
        self._loads.clear()
        self._segments.clear()
        self._older = self._newer = self._newest_tail = None
        self._frozen_endpoint = endpoint_id
        self.endpoint_id = endpoint_id
        self._anchor = None
        self._reveal_target = None

    def _refresh_edges(self) -> None:
        if not self._segments:
            self._older = self._newer = None
            return
        self._older = self._segments[0].page.older
        newest_durable = next(
            (
                descriptor
                for descriptor in reversed(self._segments)
                if not descriptor.key.startswith("live:")
            ),
            None,
        )
        self._newer = (
            newest_durable.page.newer if newest_durable is not None else None
        )

    def _descriptor(self, page: TUIHistoryPage, index: int = 0) -> PageDescriptor:
        segment = page.segments[index]
        key = segment.first_entry_id or f"{page.endpoint_id}:{index}:{self.generation}"
        return PageDescriptor(
            key,
            TUIHistoryPage(
                (segment,),
                older=page.older if index == 0 else None,
                newer=page.newer if index == len(page.segments) - 1 else None,
                endpoint_id=page.endpoint_id,
                repair_generation=page.repair_generation,
            ),
        )

    def append_live(
        self, entry_id: str, value: Any, *, unit_ordinal: int = 0
    ) -> None:
        """Rotate provisional tail rows through the same bounded segment deque."""

        size = len(str(value).encode("utf-8", errors="replace"))
        unit = TUIHistoryUnit(entry_id, value, unit_ordinal=unit_ordinal)
        for index, descriptor in enumerate(self._segments):
            units = list(descriptor.page.units)
            for unit_index, existing in enumerate(units):
                if (existing.entry_id, existing.unit_ordinal) == (
                    entry_id,
                    unit_ordinal,
                ):
                    units[unit_index] = unit
                    self._segments[index] = PageDescriptor(
                        descriptor.key,
                        TUIHistoryPage(
                            (
                                TUIPaintSegment(
                                    tuple(units),
                                    decoded_bytes=descriptor.page.segments[0].decoded_bytes,
                                ),
                            ),
                            older=descriptor.page.older,
                            newer=descriptor.page.newer,
                            endpoint_id=descriptor.page.endpoint_id,
                            repair_generation=descriptor.page.repair_generation,
                        ),
                    )
                    return
        tail = self._segments[-1] if self._segments else None
        if (
            tail is not None
            and tail.key.startswith("live:")
            and len(tail.page.units) < self.max_segment_rows
            and tail.page.segments[0].decoded_bytes + size <= self.max_segment_bytes
        ):
            units = (*tail.page.units, unit)
            self._segments[-1] = PageDescriptor(
                tail.key,
                TUIHistoryPage(
                    (TUIPaintSegment(units, decoded_bytes=tail.page.segments[0].decoded_bytes + size),),
                    older=tail.page.older,
                    newer=None,
                    endpoint_id=self.endpoint_id,
                ),
            )
        else:
            self._live_sequence += 1
            self._segments.append(
                PageDescriptor(
                    f"live:{self._live_sequence}",
                    TUIHistoryPage(
                        (TUIPaintSegment((unit,), decoded_bytes=size),),
                        older=tail.page.newer if tail is not None else self._older,
                        newer=None,
                        endpoint_id=self.endpoint_id,
                    ),
                )
            )
        while len(self._segments) > self.max_pages:
            self._segments.popleft()
        self._refresh_edges()

    async def load(self, direction: Literal["older", "newer"] = "older") -> TUIHistoryPage:
        cursor = self._older if direction == "older" else self._newer
        key = (self.generation, self.endpoint_id, direction, cursor.value if cursor else None)
        task = self._loads.get(key)
        if task is None:
            generation = self.generation
            source = self.source
            if source is None:
                return TUIHistoryPage(())
            task = asyncio.create_task(source.page(cursor, direction=direction))
            self._loads[key] = task
            try:
                page = await task
            finally:
                self._loads.pop(key, None)
            if generation != self.generation:
                return TUIHistoryPage(())
            if cursor != (self._older if direction == "older" else self._newer):
                # The edge moved while this page loaded, so it no longer
                # touches the resident pages: drop it and load the new edge.
                return await self.load(direction)
            if not self.adopt(page, direction=direction):
                return TUIHistoryPage(())
            return page
        return await task

    def adopt(
        self,
        page: TUIHistoryPage,
        *,
        direction: Literal["older", "newer"] = "older",
    ) -> bool:
        """Attach ``page`` at one edge; False when it was empty or not adjacent."""

        if not page.segments:
            return False
        if page.endpoint_id is not None:
            self.endpoint_id = page.endpoint_id
            self._frozen_endpoint = page.endpoint_id
        durable_units = {
            (unit.entry_id, unit.unit_ordinal) for unit in page.units
        }
        reconciled: deque[PageDescriptor] = deque()
        for descriptor in self._segments:
            if not descriptor.key.startswith("live:"):
                reconciled.append(descriptor)
                continue
            retained = tuple(
                unit
                for unit in descriptor.page.units
                if (unit.entry_id, unit.unit_ordinal) not in durable_units
            )
            if not retained:
                continue
            decoded_bytes = sum(
                len(str(unit.value).encode("utf-8", errors="replace"))
                for unit in retained
            )
            reconciled.append(
                PageDescriptor(
                    descriptor.key,
                    TUIHistoryPage(
                        (TUIPaintSegment(retained, decoded_bytes=decoded_bytes),),
                        older=descriptor.page.older,
                        newer=descriptor.page.newer,
                        endpoint_id=descriptor.page.endpoint_id,
                        repair_generation=descriptor.page.repair_generation,
                    ),
                )
            )
        descriptors = [self._descriptor(page, index) for index in range(len(page.segments))]
        if direction == "older":
            for descriptor in reversed(descriptors):
                reconciled.appendleft(descriptor)
            while len(reconciled) > self.max_pages:
                reconciled.pop()
        else:
            live_index = next(
                (
                    index
                    for index, descriptor in enumerate(reconciled)
                    if descriptor.key.startswith("live:")
                ),
                len(reconciled),
            )
            for descriptor in descriptors:
                reconciled.insert(live_index, descriptor)
                live_index += 1
            while len(reconciled) > self.max_pages:
                reconciled.popleft()
        # Resident durable pages must stay adjacent and in order. A page that
        # breaks that is a bug; drop it so the resident pages stay usable.
        durable = [d for d in reconciled if not d.key.startswith("live:")]
        for older, newer in pairwise(durable):
            if older.page.newer != newer.page.older:
                logger.error(
                    "history pages not adjacent: %s ends at %s but %s starts at %s",
                    older.key,
                    older.page.newer,
                    newer.key,
                    newer.page.older,
                )
                return False
        self._segments = reconciled
        self._refresh_edges()
        return True

    async def reveal(self, entry_id: str) -> TUIHistoryPage:
        self._reveal_target = entry_id
        source = self.source
        if source is None:
            return TUIHistoryPage(())
        # The latest reveal or reset wins; loads started before it belong to
        # the pages it replaces.
        self.generation += 1
        generation = self.generation
        page = await self._read(("reveal", entry_id), source.reveal(entry_id))
        # The revealed page ends at the target; rows after it let the view put
        # the target on the top row. One huge row (a compaction) can fill a
        # whole page, so read until a page of rows follows. All reads finish
        # before any page is replaced, so keys pressed meanwhile page the old
        # view and never mix with the revealed pages.
        pages = [page] if page.segments else []
        while (
            pages
            and generation == self.generation
            and pages[-1].newer is not None
            and sum(len(read.segments) for read in pages) < self.max_pages
            and sum(len(read.units) for read in pages[1:]) < self.max_segment_rows
        ):
            cursor = pages[-1].newer
            newer = await self._read(
                ("newer", cursor.value), source.page(cursor, direction="newer")
            )
            if not newer.segments:
                break
            pages.append(newer)
        if not pages or generation != self.generation:
            # A later reveal or reset owns the view now.
            return TUIHistoryPage(())
        self._segments.clear()
        for read in pages:
            self.adopt(read, direction="newer")
        return page

    async def _read(
        self, what: tuple[str, str], read: Coroutine[Any, Any, TUIHistoryPage]
    ) -> TUIHistoryPage:
        """Run one source read as a task that a reset cancels."""

        key = (self.generation, self.endpoint_id, *what)
        task = asyncio.create_task(read)
        self._loads[key] = task
        try:
            return await task
        finally:
            if self._loads.get(key) is task:
                del self._loads[key]

    async def navigate_user(
        self, entry_id: str, direction: Literal["older", "newer"]
    ) -> str | None:
        source = self.source
        if source is None:
            return None
        target = await source.adjacent_visible_user(entry_id, direction=direction)
        if target is not None:
            await self.reveal(target)
        return target

    async def search(self, query: str) -> tuple[HistorySearchResult, ...]:
        source = self.source
        return () if source is None else await source.search(query)
