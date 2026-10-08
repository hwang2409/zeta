"""Ranking model for the transcript message finder.

:class:`MessageFinder` owns the finder's candidates, current query, ranked
results, and selection. It is deliberately free of terminal dependencies.
Ranking is a pure, potentially expensive operation that callers run off the UI
thread; generation-checked publication prevents stale results from replacing a
newer query.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from .fuzzy import Query, match_query, parse_query


class Role(Enum):
    """The kind of transcript unit a candidate came from."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    RESULT = "result"
    NOTICE = "notice"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One searchable transcript message.

    ``key`` is the unit's stable transcript identity. ``index`` is its
    snapshot position, used only for result ordering. ``text`` is the bounded,
    flattened text used for matching and excerpts. ``preview`` keeps the
    original bounded lines for the preview pane.
    """

    key: int
    index: int
    role: Role
    marker: str
    text: str
    preview: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FinderRow:
    """A ranked result ready to render: an excerpt with highlight columns."""

    candidate: Candidate
    score: int
    excerpt: str
    highlights: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Rows computed for one query generation, ready for safe publication."""

    generation: int
    rows: tuple[FinderRow, ...]


def _excerpt(text: str, positions: Sequence[int], width: int) -> tuple[str, tuple[int, ...]]:
    """Return a one-line excerpt no wider than ``width`` around the first match."""

    if len(text) <= width:
        return text, tuple(positions)
    first = positions[0] if positions else 0
    lead = width // 4
    start = max(0, first - lead)
    start = min(start, len(text) - width)
    body = text[start : start + width]
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(text) else ""
    shift = start - len(prefix)
    mapped = tuple(
        position - shift
        for position in positions
        if start <= position < start + width
    )
    return prefix + body + suffix, mapped


class MessageFinder:
    """Rank transcript messages without doing scoring on the caller's thread."""

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        max_results: int = 200,
        excerpt_width: int = 160,
    ) -> None:
        self._lock = Lock()
        self._candidates = tuple(candidates)
        self._max_results = max_results
        self._excerpt_width = excerpt_width
        self._query_text = ""
        self._query = parse_query("")
        self._generation = 0
        self._rows: tuple[FinderRow, ...] = ()
        self._selected = 0
        self._complete = True
        self._rebuild_empty_rows()

    @property
    def query(self) -> str:
        return self._query_text

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def complete(self) -> bool:
        return self._complete

    def set_query(self, query: str) -> int:
        """Start a query and return its generation for off-thread ranking."""

        with self._lock:
            self._generation += 1
            self._query_text = query
            self._query = parse_query(query)
            self._reset_visible_rows()
            return self._generation

    def load_candidates(self, candidates: Sequence[Candidate]) -> int:
        """Replace candidates while preserving the current query.

        Candidate extraction can finish after the overlay opens. Loading its
        result starts a new generation so ranking against the empty initial
        collection cannot be published over the real candidates.
        """

        with self._lock:
            self._generation += 1
            self._candidates = tuple(candidates)
            self._reset_visible_rows()
            return self._generation

    def rank(self) -> RankingResult:
        """Compute the current generation without mutating visible state.

        The caller must run this method away from the event-loop thread, then
        pass its result to :meth:`publish` on the owner thread.
        """

        with self._lock:
            generation = self._generation
            query = self._query
            candidates = self._candidates
        if query.is_empty:
            rows = self._empty_rows(candidates)
        else:
            rows = self._rank_rows(query, candidates)
        return RankingResult(generation, rows)

    def publish(self, result: RankingResult) -> bool:
        """Publish ``result`` only if it belongs to the current generation."""

        with self._lock:
            if result.generation != self._generation:
                return False
            self._rows = result.rows
            self._complete = True
            self._clamp_selection()
            return True

    def rank_all(self) -> None:
        """Rank the current query synchronously (only for tests/non-UI callers)."""

        self.publish(self.rank())

    @property
    def rows(self) -> tuple[FinderRow, ...]:
        return self._rows

    @property
    def selected_index(self) -> int:
        return self._selected

    @property
    def selected(self) -> FinderRow | None:
        if not self._rows:
            return None
        return self._rows[self._selected]

    def move(self, delta: int) -> None:
        """Move the selection by ``delta`` rows, clamped to the result list."""

        if not self._rows:
            self._selected = 0
            return
        self._selected = max(0, min(len(self._rows) - 1, self._selected + delta))

    def select(self, index: int) -> None:
        if not self._rows:
            self._selected = 0
            return
        self._selected = max(0, min(len(self._rows) - 1, index))

    def preview(self) -> tuple[str, ...]:
        """Return the selected message's original lines for the preview pane."""

        row = self.selected
        return row.candidate.preview if row is not None else ()

    def _reset_visible_rows(self) -> None:
        self._selected = 0
        if self._query.is_empty:
            self._rebuild_empty_rows()
            self._complete = True
        else:
            self._rows = ()
            self._complete = False

    def _empty_rows(
        self, candidates: Sequence[Candidate] | None = None
    ) -> tuple[FinderRow, ...]:
        source = self._candidates if candidates is None else candidates
        newest = sorted(source, key=lambda item: item.index, reverse=True)
        return tuple(
            FinderRow(candidate, 0, *_excerpt(candidate.text, (), self._excerpt_width))
            for candidate in newest[: self._max_results]
        )

    def _rebuild_empty_rows(self) -> None:
        self._rows = self._empty_rows()
        self._clamp_selection()

    def _rank_rows(
        self, query: Query, candidates: Sequence[Candidate]
    ) -> tuple[FinderRow, ...]:
        heap: list[tuple[int, int, int]] = []
        matches: dict[int, tuple[Candidate, int, tuple[int, ...]]] = {}
        for scan_index, candidate in enumerate(candidates):
            result = match_query(query, candidate.text)
            if result is None:
                continue
            key = (result.score, candidate.index, scan_index)
            matches[scan_index] = (candidate, result.score, result.positions)
            if len(heap) < self._max_results:
                heapq.heappush(heap, key)
            elif key > heap[0]:
                evicted = heapq.heapreplace(heap, key)
                matches.pop(evicted[2], None)
            else:
                matches.pop(scan_index, None)

        rows: list[FinderRow] = []
        for score, _candidate_index, scan_index in sorted(heap, reverse=True):
            candidate, _score, positions = matches[scan_index]
            excerpt, highlights = _excerpt(
                candidate.text, positions, self._excerpt_width
            )
            rows.append(FinderRow(candidate, score, excerpt, highlights))
        return tuple(rows)

    def _clamp_selection(self) -> None:
        if not self._rows:
            self._selected = 0
        else:
            self._selected = max(0, min(len(self._rows) - 1, self._selected))


__all__ = [
    "Candidate",
    "FinderRow",
    "MessageFinder",
    "RankingResult",
    "Role",
]
