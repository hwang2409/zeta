"""The transcript side of the fuzzy message finder.

:class:`TranscriptFinderMixin` adds the finder lifecycle to the transcript
widget: snapshotting candidate sources for off-thread extraction and ranking,
and jumping to a chosen message. It relies on the widget for unit storage,
scroll state, and the existing substring highlight
that the post-jump next/previous keys reuse.

Kept in its own module so the main transcript stays within the module-size
budget; it is a base class of :class:`TranscriptWidget`, so it reads the
widget's private state through ``self`` the same way the virtual mixin does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text

from .finder_overlay import FinderState
from .fuzzy import highlight_literal
from .message_finder import Candidate, MessageFinder, RankingResult, Role
from .streaming_text import StreamingText

_FINDER_TEXT_LIMIT = 2_000
_FINDER_PREVIEW_LINES = 60
_FINDER_PREVIEW_TEXT_LIMIT = 12_000


@dataclass(frozen=True, slots=True)
class _FinderRestore:
    """The scroll view captured when the finder opens, restored if it cancels."""

    follow_tail: bool
    scroll_offset: int
    anchor: tuple[Any, int] | None
    virtual_start: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class _FinderCandidateSource:
    """Immutable plain data for one off-thread candidate extraction."""

    key: int
    index: int
    role: str
    marker: str
    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinderCandidateRequest:
    """An immutable plain-data snapshot for off-thread candidate extraction."""

    generation: int
    sources: tuple[_FinderCandidateSource, ...]


def _is_tool_unit(value: Any) -> bool:
    """Duck-type a tool unit without importing it (would cycle with transcript)."""

    return hasattr(value, "call") and hasattr(value, "output")


class TranscriptFinderMixin:
    """Finder lifecycle, candidate extraction, and jump-to-message for the widget."""

    @property
    def finder_active(self) -> bool:
        return self._finder is not None

    def open_finder(self) -> FinderCandidateRequest | None:
        """Open immediately and return bounded extraction work for a worker."""

        if self._finder is not None:
            return None
        self._finder_restore = _FinderRestore(
            self._follow_tail,
            self._scroll_offset,
            self._anchor,
            self._virtual_start,
        )
        finder = MessageFinder(())
        self._finder = finder
        self._finder_generation += 1
        user_unit_ids = frozenset(id(unit) for unit in self._user_units)
        sources: list[_FinderCandidateSource] = []
        turn = 0
        for index, unit in enumerate(self._units):
            if unit is None or unit.value is None:
                continue
            role = self._finder_role(unit, user_unit_ids)
            if role is Role.USER:
                turn += 1
            sources.append(
                _FinderCandidateSource(
                    key=unit.key,
                    index=index,
                    role=role.value,
                    marker=f"#{turn}" if turn else "#0",
                    parts=self._finder_source_parts(unit),
                )
            )
        return FinderCandidateRequest(
            generation=self._finder_generation,
            sources=tuple(sources),
        )

    def build_finder_candidates(
        self, request: FinderCandidateRequest
    ) -> tuple[Candidate, ...]:
        """Extract bounded candidates from ``request`` away from the UI thread."""

        return tuple(self._build_finder_candidates(request))

    def finder_publish_candidates(
        self,
        request: FinderCandidateRequest,
        candidates: tuple[Candidate, ...],
    ) -> bool:
        """Load candidates only if their finder overlay is still active."""

        if self._finder is None or self._finder_generation != request.generation:
            return False
        self._finder.load_candidates(candidates)
        return True

    def finder_set_query(self, query: str) -> int | None:
        if self._finder is None:
            return None
        return self._finder.set_query(query)

    def finder_rank(self) -> RankingResult | None:
        """Compute the current query result; callers run this off the UI thread."""

        if self._finder is None:
            return None
        return self._finder.rank()

    def finder_publish(self, result: RankingResult) -> bool:
        """Publish a current result and reject stale query generations."""

        if self._finder is None:
            return False
        return self._finder.publish(result)

    def finder_move(self, delta: int) -> None:
        if self._finder is not None:
            self._finder.move(delta)

    def finder_toggle_preview(self) -> None:
        self._finder_preview = not self._finder_preview

    def finder_cancel(self) -> None:
        """Close the finder and restore the scroll position from before it opened."""

        restore = self._finder_restore
        self._finder = None
        self._finder_restore = None
        if restore is not None:
            self._follow_tail = restore.follow_tail
            self._scroll_offset = restore.scroll_offset
            self._anchor = restore.anchor
            self._virtual_start = restore.virtual_start

    def finder_accept(self) -> bool:
        """Jump to the selected message and highlight the match for next/prev keys.

        The longest contiguous run of the fuzzy match drives the transcript's
        existing substring highlight, so once the overlay closes the familiar
        next/previous match keys continue to work from the landing position.
        """

        if self._finder is None:
            return False
        row = self._finder.selected
        query = self._finder.query
        if row is None:
            return False
        unit_index = self._resolve_finder_unit(row.candidate)
        if unit_index is None:
            return False
        self._finder = None
        self._finder_restore = None
        self.jump_to_index(unit_index)
        literal = highlight_literal(query, row.candidate.text) if query else None
        if literal:
            self.begin_search()
            self.update_search(literal)
            self._focus_search_on_unit(unit_index)
        return True

    def finder_state(self) -> FinderState | None:
        if self._finder is None:
            return None
        return FinderState(
            query=self._finder.query,
            rows=self._finder.rows,
            selected=self._finder.selected_index,
            preview=self._finder.preview(),
            preview_visible=self._finder_preview,
            total=self._finder.candidate_count,
            complete=self._finder.complete,
        )

    def _finder_role(self, unit: Any, user_unit_ids: frozenset[int]) -> Role:
        if id(unit) in user_unit_ids:
            return Role.USER
        value = unit.value
        if _is_tool_unit(value):
            return Role.TOOL
        if isinstance(value, StreamingText):
            return Role.ASSISTANT
        return Role.NOTICE

    def _finder_source_parts(self, unit: Any) -> tuple[str, ...]:
        """Snapshot only immutable text before candidate work leaves the UI loop."""

        value = unit.value
        if _is_tool_unit(value):
            argument_parts: list[str] = []
            argument_chars = 0
            for argument in value.call.arguments.values():
                if not isinstance(argument, (str, int, float)):
                    continue
                remaining = _FINDER_TEXT_LIMIT - argument_chars
                if remaining <= 0:
                    break
                part = str(argument)[:remaining]
                argument_parts.append(part)
                argument_chars += len(part) + 1
            summary = f"{value.call.name} {' '.join(argument_parts)}".strip()
            parts = [summary]
            remaining = _FINDER_PREVIEW_TEXT_LIMIT - len(summary)
            for chunk in value.output:
                if remaining <= 0:
                    break
                part = str(chunk)[:remaining]
                parts.append(part)
                remaining -= len(part)
            return tuple(parts)
        plain = getattr(value, "plain", None)
        if not isinstance(plain, str):
            plain = Text.from_ansi(
                self._searchable_text(unit, self._content_width)
            ).plain
        return (plain[:_FINDER_PREVIEW_TEXT_LIMIT],)

    @staticmethod
    def _finder_text(source: _FinderCandidateSource) -> tuple[str, tuple[str, ...]]:
        """Return bounded matching text and preview lines from plain snapshot data."""

        chunks: list[str] = []
        remaining = _FINDER_PREVIEW_TEXT_LIMIT
        for part in source.parts:
            if remaining <= 0:
                break
            chunks.append(part[:remaining])
            remaining -= len(chunks[-1])
        plain = "\n".join(chunks)
        lines = [line.rstrip() for line in plain.splitlines() if line.strip()]
        if not lines:
            return "", ()
        flat = " ".join(lines)
        if len(flat) > _FINDER_TEXT_LIMIT:
            flat = flat[:_FINDER_TEXT_LIMIT]
        return flat, tuple(lines[:_FINDER_PREVIEW_LINES])

    def _build_finder_candidates(
        self, request: FinderCandidateRequest
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for source in request.sources:
            text, preview = self._finder_text(source)
            if not text:
                continue
            candidates.append(
                Candidate(
                    key=source.key,
                    index=source.index,
                    role=Role(source.role),
                    marker=source.marker,
                    text=text,
                    preview=preview,
                )
            )
        return candidates

    def _resolve_finder_unit(self, candidate: Candidate) -> int | None:
        """Resolve a stable key, or choose its next logical surviving neighbour."""

        previous: int | None = None
        for index, unit in enumerate(self._units):
            if unit is None or unit.value is None:
                continue
            if unit.key == candidate.key:
                return index
            if unit.key > candidate.key:
                return index
            previous = index
        return previous

    def _focus_search_on_unit(self, unit_index: int) -> None:
        """Make the match inside ``unit_index`` the current one, if any exists."""

        if unit_index >= len(self._units):
            return
        unit = self._units[unit_index]
        matches = self._search_matches()
        if not matches:
            return
        if self._uses_virtual_history():
            for index, occurrence in enumerate(self._virtual_search_occurrences):
                if occurrence.unit is unit:
                    self._search_index = index
                    break
            self._focus_search_match()
            return
        locations = self._locations(self._content_width)
        for index, match in enumerate(matches):
            line = match.first_line
            if 0 <= line < len(locations) and locations[line][0] is unit:
                self._search_index = index
                break
        self._refresh_search_render_cache()
        self._focus_search_match()

    def jump_to_index(self, unit_index: int) -> bool:
        """Scroll so the unit at ``unit_index`` is at the top of the viewport."""

        if unit_index < 0 or unit_index >= len(self._units):
            return False
        unit = self._units[unit_index]
        if unit is None:
            return False
        if self._uses_virtual_history():
            self._virtual_start = (unit_index, 0)
            self._virtual_start_needs_clamp = True
            self._anchor = (unit, 0)
            self._scroll_offset = self._estimated_prefix(
                self._content_width, unit_index, 0
            )
            self._follow_tail = False
            return True
        self._materialize_for_interaction()
        if self._locations_revision != self._revision:
            self.create_content(self._content_width, self._viewport_height)
        locations = self._locations(self._content_width)
        for line, (located, _offset) in enumerate(locations):
            if located is unit:
                self._set_scroll_offset(line, allow_follow_tail=False)
                return True
        return False
