"""Ranking model for the transcript message finder.

:class:`MessageFinder` owns the finder's state: the candidate messages, the
current query, the ranked results, and the selection. It is deliberately free
of any terminal or prompt-toolkit dependency so the ranking can be unit tested
in isolation and driven in bounded slices from the event-loop ticker.

The transcript builds :class:`Candidate` rows once (each a flattened, single
line of searchable text plus the original lines for the preview pane) and hands
them to the finder. Every keystroke calls :meth:`set_query`; the actual scoring
runs in bounded slices via :meth:`rank_more`, so a long session never stalls the
UI between the 10 ms redraw ticks.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from .fuzzy import Query, match_query, parse_query


class Role(Enum):
    """The kind of transcript unit a candidate came from.

    The overlay maps each role to a short label and a theme colour; the finder
    only needs the distinction for display and never styles anything itself.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    RESULT = "result"
    NOTICE = "notice"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One searchable transcript message.

    ``index`` is the unit's position in the transcript; the finder jumps to it
    and uses it as the newest-first tiebreak. ``text`` is a single flattened
    line used for both matching and the excerpt, so match positions map
    straight onto the rendered excerpt. ``preview`` keeps the original lines for
    the preview pane.
    """

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


# Secondary scan budget: the number of candidates scored per bounded slice.
# One slice over this many short messages stays well under a redraw tick.
_DEFAULT_SLICE = 400


@dataclass(slots=True)
class _ScanState:
    query: Query
    cursor: int = 0
    heap: list[tuple[int, int, int]] = field(default_factory=list)
    scratch: dict[int, tuple[Candidate, int, tuple[int, ...]]] = field(
        default_factory=dict
    )
    complete: bool = False


def _excerpt(text: str, positions: Sequence[int], width: int) -> tuple[str, tuple[int, ...]]:
    """Return a one-line excerpt no wider than ``width`` around the first match.

    The window slides to include the first matched character with a little lead
    context, trimmed with ellipses. Highlight columns are remapped into the
    returned excerpt and any that fall outside the window are dropped.
    """

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
    """Rank transcript messages against an fzf-style query, in bounded slices."""

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        max_results: int = 200,
        excerpt_width: int = 160,
    ) -> None:
        self._candidates = tuple(candidates)
        self._max_results = max_results
        self._excerpt_width = excerpt_width
        self._query_text = ""
        self._rows: tuple[FinderRow, ...] = ()
        self._selected = 0
        self._scan = _ScanState(parse_query(""))
        self._rebuild_empty_rows()

    # -- query lifecycle ---------------------------------------------------

    @property
    def query(self) -> str:
        return self._query_text

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    def set_query(self, query: str) -> None:
        """Reset ranking for ``query`` and run the first bounded slice."""

        self._query_text = query
        parsed = parse_query(query)
        self._scan = _ScanState(parsed)
        if parsed.is_empty:
            self._rebuild_empty_rows()
            self._scan.complete = True
            return
        self._rows = ()
        self.rank_more()

    @property
    def complete(self) -> bool:
        """True once every candidate has been scored for the current query."""

        return self._scan.complete

    def rank_more(self, budget: int = _DEFAULT_SLICE) -> bool:
        """Score up to ``budget`` more candidates; return :attr:`complete`."""

        scan = self._scan
        if scan.complete:
            return True
        if scan.query.is_empty:
            scan.complete = True
            return True
        end = min(len(self._candidates), scan.cursor + max(1, budget))
        for index in range(scan.cursor, end):
            candidate = self._candidates[index]
            result = match_query(scan.query, candidate.text)
            if result is None:
                continue
            key = (result.score, candidate.index, index)
            scan.scratch[index] = (candidate, result.score, result.positions)
            if len(scan.heap) < self._max_results:
                heapq.heappush(scan.heap, key)
            elif key > scan.heap[0]:
                evicted = heapq.heapreplace(scan.heap, key)
                scan.scratch.pop(evicted[2], None)
        scan.cursor = end
        scan.complete = scan.cursor >= len(self._candidates)
        self._publish_rows()
        return scan.complete

    def rank_all(self) -> None:
        """Rank every candidate now (used for small sessions and tests)."""

        while not self.rank_more():
            pass

    # -- results and selection --------------------------------------------

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

    # -- internals ---------------------------------------------------------

    def _rebuild_empty_rows(self) -> None:
        rows = [
            FinderRow(
                candidate,
                0,
                *_excerpt(candidate.text, (), self._excerpt_width),
            )
            for candidate in sorted(
                self._candidates, key=lambda item: item.index, reverse=True
            )[: self._max_results]
        ]
        self._rows = tuple(rows)
        self._clamp_selection()

    def _publish_rows(self) -> None:
        scan = self._scan
        ordered = sorted(scan.heap, reverse=True)
        rows: list[FinderRow] = []
        for score, _candidate_index, scan_index in ordered:
            entry = scan.scratch.get(scan_index)
            if entry is None:
                continue
            candidate, _score, positions = entry
            excerpt, highlights = _excerpt(
                candidate.text, positions, self._excerpt_width
            )
            rows.append(FinderRow(candidate, score, excerpt, highlights))
        self._rows = tuple(rows)
        self._clamp_selection()

    def _clamp_selection(self) -> None:
        if not self._rows:
            self._selected = 0
        else:
            self._selected = max(0, min(len(self._rows) - 1, self._selected))


__all__ = ["Candidate", "FinderRow", "MessageFinder", "Role"]
