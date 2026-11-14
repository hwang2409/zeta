"""Bounded prompt history scoped to the session currently shown."""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Callable, Sequence
from pathlib import Path

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from .transcript.fuzzy import match_query, parse_query
from .transcript.paged_history import HistoryCursor, TUIHistorySource


class SessionPromptHistory:
    """Own at most two prompt pages for one active viewed session."""

    max_pages = 2

    def __init__(self, source: TUIHistorySource | None = None) -> None:
        self.source = source
        self._entries: list[tuple[tuple[str, str], ...]] = []
        self._page_edges: list[tuple[HistoryCursor | None, HistoryCursor | None]] = []
        self._session: Path | None = None
        self._branch_id: str | None = None
        self._older: HistoryCursor | None = None
        self._newer: HistoryCursor | None = None
        self._index: int | None = None
        self._draft: Document | None = None
        self._load: asyncio.Task[None] | None = None
        self._finder_scan: asyncio.Task[None] | None = None
        self._generation = 0
        self._navigation_revision = 0
        self._applying_document = False

    @property
    def navigating(self) -> bool:
        return self._index is not None

    @property
    def navigation_revision(self) -> int:
        return self._navigation_revision

    def navigation_is_current(self, revision: int) -> bool:
        return revision == self._navigation_revision

    @property
    def pages(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        return tuple(self._entries)

    @property
    def finder_task(self) -> asyncio.Task[None] | None:
        return self._finder_scan

    def bind(
        self,
        session_path: Path,
        source: TUIHistorySource | None = None,
        branch_id: str | None = None,
    ) -> None:
        if session_path == self._session and (
            branch_id is None or branch_id == self._branch_id
        ):
            if source is not None:
                self.source = source
            return
        self.clear()
        if source is not None:
            self.source = source
        self._session = session_path
        self._branch_id = branch_id

    async def load_older(self) -> None:
        if self.source is None:
            return
        if self._load is not None:
            await self._load
            return

        generation = self._generation

        async def read() -> None:
            cursor = self._older
            while True:
                rows, older, newer = await self.source.prompt_page(
                    cursor, direction="older"
                )
                if generation != self._generation:
                    return
                if rows or older is None:
                    break
                cursor = older
            if rows:
                self._entries.insert(0, rows)
                self._page_edges.insert(0, (older, newer))
                del self._entries[self.max_pages:]
                del self._page_edges[self.max_pages:]
                self._older = self._page_edges[0][0]
                self._newer = self._page_edges[-1][1]
                if self._index is not None:
                    self._index += len(rows)

        self._load = asyncio.create_task(read())
        try:
            await self._load
        finally:
            self._load = None

    async def load_newer(self) -> None:
        if self.source is None or self._newer is None:
            return
        if self._load is not None:
            await self._load
            return

        generation = self._generation

        async def read() -> None:
            rows, older, newer = await self.source.prompt_page(
                self._newer, direction="newer"
            )
            if generation != self._generation:
                return
            if rows:
                self._entries.append(rows)
                self._page_edges.append((older, newer))
                self._entries = self._entries[-self.max_pages:]
                self._page_edges = self._page_edges[-self.max_pages:]
                self._older = self._page_edges[0][0]
                self._newer = self._page_edges[-1][1]
                if self._index is not None:
                    retained = sum(len(page) for page in self._entries)
                    self._index = max(0, retained - len(rows) - 1)

        self._load = asyncio.create_task(read())
        try:
            await self._load
        finally:
            self._load = None

    def replace(self, session_path: Path, prompts: Sequence[str]) -> None:
        """Adopt a bounded child snapshot until child paging lands in 2b-2."""

        self.bind(session_path)
        rows = tuple((str(index), value) for index, value in enumerate(prompts[-256:]))
        self._entries = [rows[index:index + 128] for index in range(0, len(rows), 128)]
        self._page_edges = [(None, None)] * len(self._entries)

    def append(self, session_path: Path, prompt: str) -> None:
        if not prompt:
            return
        self.bind(session_path)
        if not self._entries:
            self._entries.append(())
            self._page_edges.append((None, None))
        self._entries[-1] = (*self._entries[-1], (f"live:{id(prompt)}", prompt))
        if len(self._entries[-1]) > 128:
            overflow = self._entries[-1][128:]
            self._entries[-1] = self._entries[-1][:128]
            self._entries.append(overflow)
            self._page_edges.append((None, None))
        self._entries = self._entries[-self.max_pages:]
        self._page_edges = self._page_edges[-self.max_pages:]
        self.reset_navigation()

    def snapshot(self, session_path: Path) -> tuple[str, ...]:
        if session_path != self._session:
            return ()
        return tuple(text for page in self._entries for _entry_id, text in page)

    def start(self, draft: Document) -> None:
        self._draft = draft
        self._index = None

    def select(self, prompt_id: str) -> Document | None:
        for page in reversed(self._entries):
            for candidate_id, text in reversed(page):
                if candidate_id == prompt_id:
                    return Document(text, cursor_position=len(text))
        return None

    def older(self, session_path: Path, draft: Document) -> Document | None:
        self.bind(session_path)
        entries = self.snapshot(session_path)
        if not entries:
            return None
        if self._index is None:
            self._draft = draft
            self._index = len(entries) - 1
        elif self._index > 0:
            self._index -= 1
        elif self._older is not None:
            return None
        return Document(entries[self._index], cursor_position=len(entries[self._index]))

    def newer(self, session_path: Path) -> Document | None:
        if self._session != session_path or self._index is None:
            return None
        entries = self.snapshot(session_path)
        if self._index < len(entries) - 1:
            self._index += 1
            value = entries[self._index]
            return Document(value, cursor_position=len(value))
        if self._newer is not None:
            return None
        draft = self._draft
        self.reset_navigation()
        return draft

    def apply(self, buffer: Buffer, document: Document) -> None:
        self._applying_document = True
        try:
            buffer.set_document(document, bypass_readonly=True)
        finally:
            self._applying_document = False

    def detach(self) -> None:
        if not self._applying_document:
            self.reset_navigation()

    def reset_navigation(self) -> None:
        self._navigation_revision += 1
        self._index = None
        self._draft = None

    def start_finder_scan(
        self,
        query: str,
        publish: Callable[[tuple[str, ...]], None],
        *,
        limit: int = 200,
    ) -> None:
        """Own and publish one branch-scoped finder scan."""

        if self._finder_scan is not None:
            self._finder_scan.cancel()
        generation = self._generation
        self._finder_scan = asyncio.create_task(
            self._scan_finder(query, publish, limit=limit, generation=generation)
        )

    async def _scan_finder(
        self,
        query: str,
        publish: Callable[[tuple[str, ...]], None],
        *,
        limit: int,
        generation: int,
    ) -> None:
        source = self.source
        if source is None:
            values = self.snapshot(self._session) if self._session is not None else ()
            if generation == self._generation:
                publish(values[-limit:])
            return
        parsed = parse_query(query)
        ranked: list[tuple[int, int, str]] = []
        retained_values: set[str] = set()
        newest_ordinal = 0
        published = False
        async for rows in source.prompt_pages():
            if generation != self._generation:
                return

            pending: list[tuple[int, str]] = []
            pending_values: set[str] = set()
            for _entry_id, raw in reversed(rows):
                value = raw.rstrip()
                ordinal = newest_ordinal
                newest_ordinal += 1
                if (
                    value
                    and value not in retained_values
                    and value not in pending_values
                ):
                    pending.append((ordinal, value))
                    pending_values.add(value)

            def score_page(
                candidates: tuple[tuple[int, str], ...],
            ) -> list[tuple[int, int, str]]:
                scored: list[tuple[int, int, str]] = []
                for ordinal, value in candidates:
                    match = match_query(parsed, value)
                    if match is not None:
                        scored.append((match.score, -ordinal, value))
                return scored

            candidates = await asyncio.to_thread(score_page, tuple(pending))
            if generation != self._generation:
                return
            for candidate in candidates:
                if len(ranked) < limit:
                    heapq.heappush(ranked, candidate)
                    retained_values.add(candidate[2])
                elif candidate > ranked[0]:
                    removed = heapq.heapreplace(ranked, candidate)
                    retained_values.remove(removed[2])
                    retained_values.add(candidate[2])
            publish(
                tuple(
                    value
                    for _score, _recency, value in sorted(
                        ranked, key=lambda item: item[1]
                    )
                )
            )
            published = True
        if not published and generation == self._generation:
            publish(())

    async def close(self) -> None:
        self.clear()
        await asyncio.sleep(0)

    def clear(self) -> None:
        self._generation += 1
        if self._load is not None:
            self._load.cancel()
        if self._finder_scan is not None:
            self._finder_scan.cancel()
        self._load = None
        self._finder_scan = None
        self._entries.clear()
        self._page_edges.clear()
        self._older = self._newer = None
        self.reset_navigation()
        self._session = None
        self._branch_id = None
