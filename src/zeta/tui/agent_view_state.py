"""Value objects and pure helpers for agent-view navigation state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.document import Document


@dataclass(frozen=True, slots=True)
class AgentEntry:
    """One session shown in the current agent list."""

    path: Path
    label: str
    agent_type: str
    state: str


@dataclass(frozen=True, slots=True)
class BoundedAgentMessages:
    """The bounded message tail and its locked truncation marker."""

    messages: tuple[dict[str, Any], ...]
    marker: str | None


@dataclass(frozen=True, slots=True)
class InactiveViewState:
    anchor: tuple[str, int, int] | None
    draft: str
    draft_cursor: int
    prompt_cursor: str | None
    prompt_text: str | None
    finder_query: str
    finder_cursor: int


@dataclass(frozen=True, slots=True)
class PreparedNavigation:
    entries: tuple[AgentEntry, ...]


@dataclass(frozen=True, slots=True)
class NavigationSnapshot:
    current_path: Path
    path_stack: tuple[Path, ...]
    breadcrumb_labels: tuple[str, ...]
    entries: tuple[AgentEntry, ...]
    selected_index: int
    focused: Any


def capture_inactive_view(
    *,
    buffer: Any,
    prompt_history: Any,
    transcript: Any,
    anchor: tuple[str, int, int] | None,
    search_index: int,
) -> InactiveViewState | None:
    """Capture the bounded composer and transcript state for one inactive view."""

    document = buffer.document if buffer is not None else Document()
    if len(document.text.encode("utf-8")) > 256 * 1024:
        return None
    prompt_cursor = prompt_history.cursor_id() if prompt_history is not None else None
    draft = prompt_history.draft_document(document) if prompt_history is not None else document
    return InactiveViewState(
        anchor,
        draft.text,
        draft.cursor_position,
        prompt_cursor,
        document.text if prompt_cursor is not None else None,
        getattr(transcript, "search_query", "") if transcript is not None else "",
        search_index,
    )


def short_agent_value(value: object, limit: int = 2_000) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
