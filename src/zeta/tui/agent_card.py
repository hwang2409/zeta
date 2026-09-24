"""Compatibility exports for the focused TUI card package."""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension

from ..core.checkpoints import ConversationIntegrityError, load_session_json
from ..core.session_files import SessionError, open_session_file, session_directory
from ..core.store import ConversationStore
from .cards.agent import (
    AgentCard,
    AgentRunCommandMixin,
    _read_lifecycle,
    render_agent_expanded,
    render_agent_progress,
    render_agent_receipt,
)
from .cards.shared import (
    MAX_CARD_COLUMNS,
    MAX_CARD_LINES,
    infer_language,
)
from .cards.shared import (
    BoundedToolOutput as _BoundedToolOutput,
)
from .cards.shared import (
    scan_tool_output as _scan_tool_output,
)
from .cards.tool import TOOL_CARD_REGISTRY, register_tool_card

__all__ = [
    "MAX_AGENT_VIEW_LINES",
    "MAX_CARD_COLUMNS",
    "MAX_CARD_LINES",
    "TOOL_CARD_REGISTRY",
    "AgentCard",
    "AgentEntry",
    "AgentNavigation",
    "AgentRunCommandMixin",
    "_BoundedToolOutput",
    "_read_lifecycle",
    "_scan_tool_output",
    "infer_language",
    "read_agent_transcript",
    "register_tool_card",
    "render_agent_expanded",
    "render_agent_progress",
    "render_agent_receipt",
    "time",
]


MAX_AGENT_VIEW_LINES = 240
MAX_AGENT_LINE_CHARS = 2_000
MAX_AGENT_LIST_ROWS = 7
MAX_AGENT_SCAN_BYTES = MAX_AGENT_VIEW_LINES * (MAX_AGENT_LINE_CHARS + 256)
MAX_AGENT_ACCOUNTING_ROWS = MAX_AGENT_VIEW_LINES + 16
_TRUNCATION_MARKER = "[older lines omitted]"


@dataclass(frozen=True, slots=True)
class AgentEntry:
    """One session shown in the current agent list."""

    path: Path
    label: str
    agent_type: str
    state: str


def _short(value: object, limit: int = MAX_AGENT_LINE_CHARS) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = load_session_json(path)
    except (ConversationIntegrityError, OSError):
        return {}
    return value if type(value) is dict else {}


def _agent_metadata(path: Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    lifecycle = _read_json(path / "agent_lifecycle.json")
    if lifecycle:
        return lifecycle
    state = _read_json(path / "session_state.json")
    parent = state.get("agent_parent")
    result = dict(fallback or {})
    if isinstance(parent, dict):
        result.setdefault("agent_type", parent.get("agent_type"))
    return result


def _text_lines(text: str, tail: int | None = None) -> Iterator[str]:
    if not text:
        yield ""
        return
    if tail is not None:
        yield from text.splitlines()[-tail:]
        return
    start = 0
    while start < len(text):
        ends = [
            index
            for index in (text.find("\n", start), text.find("\r", start))
            if index >= 0
        ]
        end = min(ends) if ends else -1
        if end < 0:
            yield text[start:]
            return
        yield text[start:end]
        start = end + 1
        if text[end] == "\r" and start < len(text) and text[start] == "\n":
            start += 1


def _message_lines(
    message: dict[str, Any], *, tail: int | None = None
) -> Iterator[str]:
    role = _short(message.get("role", "message"), 32)
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                for line in _text_lines(block["text"], tail):
                    yield f"{role}: {_short(line)}"
            elif block.get("type") == "tool_use" and isinstance(block.get("tool_call"), dict):
                call = block["tool_call"]
                name = call.get("name", "tool")
                arguments = call.get("arguments", {})
                if isinstance(arguments, dict):
                    args = " ".join(
                        f"{key}={_short(value, 160)}"
                        for key, value in sorted(arguments.items())
                    )
                    yield f"tool: {_short(name, 64)} {_short(args, 400)}".rstrip()
    result = message.get("tool_result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            for line in _text_lines(content, tail):
                yield f"tool result: {_short(line)}"


def _read_partial_row_tail(handle: Any, limit: int) -> bytes:
    chunk = handle.read(limit)
    line, separator, remainder = chunk.partition(b"\n")
    if separator and remainder:
        handle.seek(-len(remainder), os.SEEK_CUR)
    return line if separator else chunk


def _count_rendered_lines_before(handle: Any, end: int) -> int | None:
    """Count rendered lines in a bounded, complete prefix of the log."""

    handle.seek(0)
    count = 0
    rows = 0
    while handle.tell() < end:
        raw_line = handle.readline()
        if not raw_line or handle.tell() > end:
            handle.seek(end)
            return None
        rows += 1
        if rows > MAX_AGENT_ACCOUNTING_ROWS:
            handle.seek(end)
            return None
        try:
            row = load_session_json(raw_line)
        except ConversationIntegrityError:
            handle.seek(end)
            return None
        if not isinstance(row, dict) or row.get("type") != "message":
            continue
        data = row.get("data")
        message = data.get("message") if isinstance(data, dict) else None
        if isinstance(message, dict):
            count += sum(1 for _ in _message_lines(message))
    handle.seek(end)
    return count if handle.tell() == end else None


def _oversized_message(raw_tail: bytes) -> dict[str, Any] | None:
    """Recover the final text value from a row whose prefix was bounded away."""

    backslashes = 0
    closing_quote: int | None = None
    for index, value in enumerate(raw_tail):
        if value == 92:
            backslashes += 1
            continue
        if value == 34 and backslashes % 2 == 0:
            closing_quote = index
            break
        backslashes = 0
    if closing_quote is None:
        return None
    encoded_tail = raw_tail[:closing_quote]
    for offset in range(min(8, len(encoded_tail))):
        try:
            text = load_session_json(b'"' + encoded_tail[offset:] + b'"')
        except ConversationIntegrityError:
            continue
        if isinstance(text, str):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
    return None


def read_agent_transcript(path: Path, limit: int = MAX_AGENT_VIEW_LINES) -> list[str]:
    """Read one agent's direct transcript, without nested child sessions."""

    lines: deque[str] = deque(maxlen=limit)
    byte_omitted = False
    partial_row_recovered = False
    omitted_line_count: int | None = 0
    overflow_count = 0

    def append_message_lines(message: dict[str, Any], tail: int | None = None) -> None:
        nonlocal overflow_count
        for line in _message_lines(message, tail=tail):
            if len(lines) == limit:
                overflow_count += 1
            lines.append(line)

    try:
        with session_directory(path.parent, path.name) as (_, directory_fd), os.fdopen(
            open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY), "rb"
        ) as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            start = max(0, file_size - MAX_AGENT_SCAN_BYTES)
            byte_omitted = start > 0
            handle.seek(start)
            if start:
                handle.seek(start - 1)
                at_line_start = handle.read(1) == b"\n"
                handle.seek(start)
                if at_line_start:
                    omitted_line_count = _count_rendered_lines_before(handle, start)
                if not at_line_start:
                    omitted_line_count = None
                    raw_tail = _read_partial_row_tail(handle, MAX_AGENT_SCAN_BYTES)
                    message = _oversized_message(raw_tail)
                    if message is not None:
                        append_message_lines(message, tail=limit)
                        partial_row_recovered = True
            for raw_line in handle:
                try:
                    row = load_session_json(raw_line)
                except ConversationIntegrityError:
                    continue
                if not isinstance(row, dict) or row.get("type") != "message":
                    continue
                data = row.get("data")
                message = data.get("message") if isinstance(data, dict) else None
                if isinstance(message, dict):
                    append_message_lines(message)
    except (OSError, SessionError):
        return []
    result = list(lines)
    marker: str | None = None
    if byte_omitted or overflow_count:
        if partial_row_recovered or omitted_line_count is None:
            marker = _TRUNCATION_MARKER
        else:
            omitted_count = omitted_line_count + overflow_count
            if omitted_count:
                marker = f"[{omitted_count} older lines omitted]"
    if marker is not None:
        result.insert(0, marker)
    return result or ["transcript unavailable"]


class AgentListControl(UIControl):
    """Focusable, compact list of the current agent and its children."""

    def __init__(self, navigator: AgentNavigation) -> None:
        self.navigator = navigator

    @property
    def is_focusable(self) -> bool:
        return True

    def create_content(self, width: int, height: int | None) -> UIContent:
        del width, height
        self.navigator.refresh()
        entries = self.navigator.entries

        def get_line(index: int) -> list[tuple[str, str]]:
            entry = entries[index]
            selected = index == self.navigator.selected_index
            marker = ">" if selected else " "
            label = f"{entry.label} · {entry.agent_type} · {entry.state}"
            style = "class:agent-list.selected" if selected else "class:agent-list"
            return [(style, f"{marker} {label}")]

        return UIContent(
            get_line=get_line,
            line_count=len(entries),
            cursor_position=Point(x=0, y=self.navigator.selected_index),
            show_cursor=False,
        )


class AgentTranscriptControl(UIControl):
    """Scrollable bounded transcript for one session directory."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.offset = 0
        self.viewport_height = 1

    @property
    def is_focusable(self) -> bool:
        return True

    def load(self, path: Path) -> None:
        self.lines = read_agent_transcript(path)
        self.offset = max(0, len(self.lines) - self.viewport_height)

    def _clamp(self) -> None:
        self.offset = min(self.offset, max(0, len(self.lines) - self.viewport_height))
        self.offset = max(0, self.offset)

    def scroll(self, amount: int) -> None:
        self.offset += amount
        self._clamp()

    def half_page(self, amount: int) -> None:
        self.scroll(amount * max(1, self.viewport_height // 2))

    def top(self) -> None:
        self.offset = 0

    def bottom(self) -> None:
        self.offset = max(0, len(self.lines) - self.viewport_height)

    def create_content(self, width: int, height: int | None) -> UIContent:
        del width
        self.viewport_height = max(1, height or 1)
        self._clamp()
        lines = self.lines or ["transcript unavailable"]

        def get_line(index: int) -> list[tuple[str, str]]:
            return [("class:agent-view", lines[index])]

        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=self.offset),
            show_cursor=False,
        )

    def vertical_scroll(self, window: Window) -> int:
        del window
        return self.offset


class AgentNavigation:
    """Own the current session path, list selection, and child transcript."""

    def __init__(self, store: ConversationStore) -> None:
        self.store = store
        self.root_path = store.session_dir
        self.current_path = self.root_path
        self._path_stack: list[Path] = [self.root_path]
        self._breadcrumb_labels = ["main"]
        self.entries: list[AgentEntry] = []
        self.selected_index = 0
        self.list_control = AgentListControl(self)
        self.transcript_control = AgentTranscriptControl()
        self.list_window = Window(
            content=self.list_control,
            height=Dimension(min=1, max=MAX_AGENT_LIST_ROWS),
            wrap_lines=False,
        )
        self.transcript_window = Window(
            content=self.transcript_control,
            wrap_lines=False,
            get_vertical_scroll=self.transcript_control.vertical_scroll,
        )
        self.breadcrumb_window = Window(
            content=FormattedTextControl(
                lambda: [("class:agent-breadcrumb", " > ".join(self._breadcrumb_labels))]
            ),
            height=1,
            wrap_lines=False,
        )
        self.view_container = HSplit([self.breadcrumb_window, self.transcript_window])
        self._layout: Any = None
        self._composer_buffer: Any = None
        self._transcript_layout: Any = None
        self._main_transcript: Any = None
        self.refresh()

    @property
    def child_view_active(self) -> bool:
        return self.current_path != self.root_path

    def child_view_focused(self) -> bool:
        return self._layout is not None and self._layout.has_focus(self.transcript_window)

    @property
    def list_visible(self) -> bool:
        self.refresh()
        return bool(self.entries)

    def bind_layout(self, layout: Any, composer_buffer: Any) -> None:
        self._layout = layout
        self._composer_buffer = composer_buffer

    def bind_transcript_layout(self, layout: Any, main_transcript: Any) -> None:
        self._transcript_layout = layout
        self._main_transcript = main_transcript

    def _switch_transcript(self) -> None:
        if self._transcript_layout is None:
            return
        self._transcript_layout.children[0] = (
            self.view_container if self.child_view_active else self._main_transcript
        )

    def refresh(self) -> None:
        selected_path = self.entries[self.selected_index].path if self.entries else None
        fallback: dict[Path, dict[str, Any]] = {}
        if self.current_path == self.root_path:
            for marker in self.store.agent_children().values():
                child_path = marker.get("child_session_path")
                if isinstance(child_path, str):
                    fallback[Path(child_path)] = marker
        children = self._children(self.current_path, fallback)
        self.entries = children
        if selected_path is not None:
            self.selected_index = next(
                (index for index, entry in enumerate(self.entries) if entry.path == selected_path),
                0,
            )
        else:
            self.selected_index = min(self.selected_index, max(0, len(self.entries) - 1))

    @staticmethod
    def _children(path: Path, fallback: dict[Path, dict[str, Any]]) -> list[AgentEntry]:
        agents = path / "agents"
        try:
            candidates = sorted(
                (child for child in agents.iterdir() if child.is_dir() and not child.is_symlink()),
                key=lambda child: (not child.name.isdigit(), child.name),
            )
        except OSError:
            return []
        entries: list[AgentEntry] = []
        for child in candidates:
            metadata = _agent_metadata(child, fallback.get(child))
            entries.append(
                AgentEntry(
                    child,
                    str(metadata.get("description") or f"child {child.name}"),
                    str(metadata.get("agent_type") or "child"),
                    str(metadata.get("state") or "running"),
                )
            )
        return entries

    def focus_composer(self) -> None:
        if self._layout is not None and self._composer_buffer is not None:
            self._layout.focus(self._composer_buffer)

    def list_focused(self) -> bool:
        return self._layout is not None and self._layout.has_focus(self.list_window)

    def focus_list(self) -> None:
        self.refresh()
        if self.list_visible and self._layout is not None:
            self._layout.focus(self.list_window)
        else:
            self.focus_composer()

    def list_back(self) -> None:
        if self.child_view_active and self._layout is not None:
            self._layout.focus(self.transcript_window)
        else:
            self.focus_composer()

    def exit_navigation(self) -> None:
        self.current_path = self.root_path
        self._path_stack[:] = [self.root_path]
        self._breadcrumb_labels[:] = ["main"]
        self.selected_index = 0
        self.refresh()
        self._switch_transcript()
        self.focus_composer()

    def focus_child_list(self) -> None:
        self.refresh()
        if self.list_visible and self._layout is not None:
            self._layout.focus(self.list_window)

    def move_selection(self, amount: int) -> None:
        if not self.entries:
            return
        self.selected_index = (self.selected_index + amount) % len(self.entries)

    def open_selected(self) -> None:
        if not self.entries:
            return
        entry = self.entries[self.selected_index]
        if entry.path == self.current_path:
            return
        self.current_path = entry.path
        self._path_stack.append(entry.path)
        self._breadcrumb_labels.append(entry.label)
        self.transcript_control.load(entry.path)
        self.refresh()
        self._switch_transcript()
        if self._layout is not None:
            self._layout.focus(self.transcript_window)

    def back_to_parent(self) -> None:
        if not self.child_view_active:
            self.focus_composer()
            return
        self._leave_current_view()

    def _leave_current_view(self) -> None:
        child_path = self.current_path
        self._path_stack.pop()
        self.current_path = self._path_stack[-1]
        self._breadcrumb_labels.pop()
        self.refresh()
        self.selected_index = next(
            (
                index
                for index, entry in enumerate(self.entries)
                if entry.path == child_path
            ),
            0,
        )
        if self.child_view_active:
            self.transcript_control.load(self.current_path)
        self._switch_transcript()
        if self.child_view_active and self._layout is not None:
            self._layout.focus(self.transcript_window)
        else:
            self.focus_composer()

    def child_scroll(self, amount: int) -> None:
        self.transcript_control.scroll(amount)

    def child_half_page(self, amount: int) -> None:
        self.transcript_control.half_page(amount)

    def child_top(self) -> None:
        self.transcript_control.top()

    def child_bottom(self) -> None:
        self.transcript_control.bottom()
