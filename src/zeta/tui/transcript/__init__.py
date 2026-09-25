"""Transcript rendering, presentation, and search."""

from .transcript import TranscriptWidget, _ToolUnit, _TranscriptUnit, stream_key
from .transcript_presenter import TranscriptPresenter
from .transcript_search import (
    AnchoredSelection,
    Cell,
    HighlightCache,
    SearchMatch,
    Selection,
    SelectionAnchor,
    find_matches,
    highlight,
    highlight_fragments,
    resolve_anchor,
)

__all__ = [
    "AnchoredSelection",
    "Cell",
    "HighlightCache",
    "SearchMatch",
    "Selection",
    "SelectionAnchor",
    "TranscriptPresenter",
    "TranscriptWidget",
    "_ToolUnit",
    "_TranscriptUnit",
    "find_matches",
    "highlight",
    "highlight_fragments",
    "resolve_anchor",
    "stream_key",
]
