"""Bounded prompt history scoped to the session currently shown."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from prompt_toolkit.document import Document

from .transcript.paged_history import HistoryCursor, TUIHistorySource


class SessionPromptHistory:
    """Own at most two prompt pages for one active viewed session."""

    max_pages = 2

    def __init__(self, source: TUIHistorySource | None = None) -> None:
        self.source = source
        self._entries: list[tuple[tuple[str, str], ...]] = []
        self._session: Path | None = None
        self._older: HistoryCursor | None = None
        self._newer: HistoryCursor | None = None
        self._index: int | None = None
        self._draft: Document | None = None
        self._load: asyncio.Task[None] | None = None

    @property
    def pages(self) -> tuple[tuple[tuple[str, str], ...], ...]:
        return tuple(self._entries)

    def bind(self, session_path: Path, source: TUIHistorySource | None = None) -> None:
        if source is not None:
            self.source = source
        if session_path == self._session:
            return
        self.clear()
        self._session = session_path

    async def load_older(self) -> None:
        if self.source is None:
            return
        if self._load is not None:
            await self._load
            return

        async def read() -> None:
            rows, older, newer = await self.source.prompt_page(
                self._older, direction="older"
            )
            self._older, self._newer = older, newer
            if rows:
                self._entries.insert(0, rows)
                del self._entries[self.max_pages:]
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

        async def read() -> None:
            rows, older, newer = await self.source.prompt_page(
                self._newer, direction="newer"
            )
            self._older, self._newer = older, newer
            if rows:
                self._entries.append(rows)
                self._entries = self._entries[-self.max_pages:]
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

    def append(self, session_path: Path, prompt: str) -> None:
        if not prompt:
            return
        self.bind(session_path)
        if not self._entries:
            self._entries.append(())
        self._entries[-1] = (*self._entries[-1], (f"live:{id(prompt)}", prompt))
        if len(self._entries[-1]) > 128:
            overflow = self._entries[-1][128:]
            self._entries[-1] = self._entries[-1][:128]
            self._entries.append(overflow)
        self._entries = self._entries[-self.max_pages:]
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

    def reset_navigation(self) -> None:
        self._index = None
        self._draft = None

    async def close(self) -> None:
        task = self._load
        self.clear()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    def clear(self) -> None:
        if self._load is not None:
            self._load.cancel()
        self._load = None
        self._entries.clear()
        self._older = self._newer = None
        self._index = None
        self._draft = None
        self._session = None
