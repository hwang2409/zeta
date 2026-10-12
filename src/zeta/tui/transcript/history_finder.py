"""Cross-session composer history finder."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from .finder_overlay import FinderState
from .message_finder import Candidate, MessageFinder, RankingResult, Role

_MATCH_TEXT_LIMIT = 2_048
_PREVIEW_LINE_LIMIT = 60
_PREVIEW_TEXT_LIMIT = 12_000


@dataclass(frozen=True, slots=True)
class HistoryCandidateRequest:
    generation: int


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    candidates: tuple[Candidate, ...]
    entries: tuple[tuple[int, str], ...]


def build_history_candidates(history: Sequence[str]) -> HistorySnapshot:
    """Prepare newest unique history entries from an oldest-first snapshot."""

    unique: list[str] = []
    seen: set[str] = set()
    for value in reversed(history):
        normalized = value.rstrip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)

    count = len(unique)
    candidates: list[Candidate] = []
    entries: list[tuple[int, str]] = []
    for newest_index, value in enumerate(unique):
        key = newest_index
        index = count - newest_index
        preview_text = value[:_PREVIEW_TEXT_LIMIT]
        preview = tuple(preview_text.splitlines()[:_PREVIEW_LINE_LIMIT])
        flattened = " ⏎ ".join(value.splitlines())[:_MATCH_TEXT_LIMIT]
        candidates.append(
            Candidate(
                key=key,
                index=index,
                role=Role.USER,
                marker=f"#{index}",
                text=flattened,
                preview=preview,
            )
        )
        entries.append((key, value))
    return HistorySnapshot(tuple(candidates), tuple(entries))


class HistoryFinder:
    """Own history loading, ranking, and draft restoration behind finder methods."""

    def __init__(
        self,
        load_history: Callable[[], list[str]],
        history_snapshot: Callable[[], tuple[str, ...]],
        *,
        on_accept: Callable[[], None] | None = None,
    ) -> None:
        self._load_history = load_history
        self._history_snapshot = history_snapshot
        self._history_loaded = False
        self._on_accept = on_accept
        self._finder: MessageFinder | None = None
        self._generation = 0
        self._buffer: Buffer | None = None
        self._draft: Document | None = None
        self._entries: dict[int, str] = {}
        self._preview = True

    @property
    def finder_active(self) -> bool:
        return self._finder is not None

    def open_finder(self, buffer: Buffer) -> HistoryCandidateRequest | None:
        if self._finder is not None:
            return None
        self._buffer = buffer
        self._draft = buffer.document
        self._finder = MessageFinder(())
        self._finder.set_query(buffer.text)
        self._generation += 1
        return HistoryCandidateRequest(self._generation)

    async def build_finder_candidates(
        self, request: HistoryCandidateRequest
    ) -> HistorySnapshot:
        del request
        return await asyncio.to_thread(self._load_snapshot)

    def _load_snapshot(self) -> HistorySnapshot:
        if not self._history_loaded:
            self._load_history()
            self._history_loaded = True
        return build_history_candidates(self._history_snapshot())

    def finder_publish_candidates(
        self, request: HistoryCandidateRequest, snapshot: HistorySnapshot
    ) -> bool:
        if self._finder is None or request.generation != self._generation:
            return False
        self._entries = dict(snapshot.entries)
        self._finder.load_candidates(snapshot.candidates)
        return True

    def finder_set_query(self, query: str) -> int | None:
        return None if self._finder is None else self._finder.set_query(query)

    def finder_rank(self) -> RankingResult | None:
        return None if self._finder is None else self._finder.rank()

    def finder_publish(self, result: RankingResult) -> bool:
        return self._finder is not None and self._finder.publish(result)

    def finder_move(self, delta: int) -> None:
        if self._finder is not None:
            self._finder.move(delta)

    def finder_toggle_preview(self) -> None:
        self._preview = not self._preview

    def finder_accept(self) -> bool:
        if self._finder is None or self._buffer is None:
            return False
        row = self._finder.selected
        if row is None:
            return False
        value = self._entries.get(row.candidate.key)
        if value is None:
            return False
        buffer = self._buffer
        buffer.save_to_undo_stack()
        if self._on_accept is not None:
            self._on_accept()
        buffer.set_document(Document(value, cursor_position=len(value)), bypass_readonly=True)
        self._close()
        return True

    def finder_cancel(self) -> None:
        if self._buffer is not None and self._draft is not None:
            self._buffer.set_document(self._draft, bypass_readonly=True)
        self._close()

    def finder_state(self) -> FinderState | None:
        if self._finder is None:
            return None
        return FinderState(
            query=self._finder.query,
            rows=self._finder.rows,
            selected=self._finder.selected_index,
            preview=self._finder.preview(),
            preview_visible=self._preview,
            total=self._finder.candidate_count,
            complete=self._finder.complete,
            title="history",
            empty_text="no history",
        )

    def _close(self) -> None:
        self._finder = None
        self._buffer = None
        self._draft = None
        self._entries = {}


__all__ = [
    "HistoryCandidateRequest",
    "HistoryFinder",
    "HistorySnapshot",
    "build_history_candidates",
]
