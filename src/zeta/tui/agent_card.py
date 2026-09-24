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
from rich.console import Console
from rich.text import Text

from ..core.checkpoints import ConversationIntegrityError, load_session_json
from ..core.session_files import SessionError, open_session_file, session_directory
from ..core.store import ConversationStore
from ..types import Message, TextContent, ToolCall
from . import theme
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
from .checkpoints import render_replayed_message

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
_TRUNCATION_MARKER = "[older lines omitted]"


@dataclass(frozen=True, slots=True)
class BoundedAgentMessages:
    """The bounded message tail and its locked truncation marker."""

    messages: tuple[dict[str, Any], ...]
    marker: str | None


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


def _bounded_text(text: str) -> str:
    return "\n".join(_short(line) for line in _text_lines(text))


def _bounded_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep replayed text within the transcript's per-line bound."""

    result = dict(message)
    content = message.get("content")
    if isinstance(content, list):
        bounded_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            bounded_block = dict(block)
            if (
                block.get("type") in {"text", "thinking"}
                and isinstance(block.get("text"), str)
            ):
                bounded_block["text"] = _bounded_text(block["text"])
            bounded_content.append(bounded_block)
        result["content"] = bounded_content
    tool_result = message.get("tool_result")
    if isinstance(tool_result, dict) and isinstance(tool_result.get("content"), str):
        bounded_result = dict(tool_result)
        bounded_content = _bounded_text(tool_result["content"])
        bounded_result["content"] = bounded_content
        if bounded_content != tool_result["content"]:
            bounded_result.pop("content_blocks", None)
        result["tool_result"] = bounded_result
    return result


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
            elif block.get("type") == "thinking":
                yield f"{role}: thought"
                if isinstance(block.get("text"), str):
                    for line in _text_lines(block["text"], tail):
                        yield f"{role}: {_short(line)}"
            elif block.get("type") == "redacted_thinking":
                yield f"{role}: thought: redacted"
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


def _message_line_count(message: dict[str, Any]) -> int:
    """Count persisted rows without synthetic thought headers."""

    count = sum(1 for _ in _message_lines(message))
    content = message.get("content")
    if isinstance(content, list):
        count -= sum(block.get("type") == "thinking" for block in content if isinstance(block, dict))
    return count


def _read_partial_row_tail(handle: Any, limit: int) -> bytes:
    chunk = handle.read(limit)
    line, separator, remainder = chunk.partition(b"\n")
    if separator and remainder:
        handle.seek(-len(remainder), os.SEEK_CUR)
    return line if separator else chunk


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


def _tail_message(message: dict[str, Any], limit: int) -> dict[str, Any]:
    """Keep a renderable tail when one message exceeds the row bound."""

    if limit <= 0:
        return {**message, "content": []}
    rendered_lines = list(_message_lines(message))
    if len(rendered_lines) <= limit:
        return message

    result = dict(message)
    content = message.get("content")
    skipped = len(rendered_lines) - limit
    if isinstance(content, list):
        retained_content: list[dict[str, Any]] = []
        seen = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type in {"text", "thinking"} and isinstance(
                block.get("text"), str
            ):
                lines = list(_text_lines(block["text"]))
                header_lines = 1 if block_type == "thinking" else 0
                block_lines = header_lines + len(lines)
                block_start = max(0, skipped - seen)
                block_end = min(block_lines, limit + skipped - seen)
                if block_start < block_end:
                    retained = dict(block)
                    text_start = max(0, block_start - header_lines)
                    text_end = min(len(lines), block_end - header_lines)
                    if text_start or text_end < len(lines):
                        retained["text"] = "\n".join(lines[text_start:text_end])
                    retained_content.append(retained)
                seen += block_lines
            elif block_type == "redacted_thinking" or (
                block_type == "tool_use" and isinstance(block.get("tool_call"), dict)
            ):
                if skipped <= seen < limit + skipped:
                    retained_content.append(block)
                seen += 1
            else:
                retained_content.append(block)
        result["content"] = retained_content

    tool_result = message.get("tool_result")
    if isinstance(tool_result, dict) and isinstance(tool_result.get("content"), str):
        lines = list(_text_lines(tool_result["content"]))
        block_start = max(0, skipped - seen)
        block_end = min(len(lines), limit + skipped - seen)
        if block_start < block_end:
            retained_result = dict(tool_result)
            if block_start or block_end < len(lines):
                retained_result["content"] = "\n".join(lines[block_start:block_end])
                # A clipped result must use the normal text-card path. Keeping
                # full content_blocks here would bypass the bounded content.
                retained_result.pop("content_blocks", None)
            result["tool_result"] = retained_result

    return result


def _tool_call_only(message: dict[str, Any], call_id: str) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, list):
        return None
    tool_blocks = [
        block
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance(block.get("tool_call"), dict)
        and block["tool_call"].get("id") == call_id
    ]
    if not tool_blocks:
        return None
    return {"role": "assistant", "content": tool_blocks[:1]}


def _has_tool_call(message: dict[str, Any], call_id: str) -> bool:
    content = message.get("content")
    return (
        isinstance(content, list)
        and any(
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and isinstance(block.get("tool_call"), dict)
            and block["tool_call"].get("id") == call_id
            for block in content
        )
    )


def _read_bounded_messages(
    path: Path, limit: int = MAX_AGENT_VIEW_LINES
) -> BoundedAgentMessages:
    """Read direct message rows with the existing bounded-tail accounting."""

    messages: deque[tuple[dict[str, Any], int]] = deque()
    retained_lines = 0
    byte_omitted = False
    overflow_count = 0
    tool_calls: dict[str, dict[str, Any]] = {}

    def append_message(message: dict[str, Any]) -> None:
        nonlocal overflow_count, retained_lines
        message = _bounded_message(message)
        rendered_lines = _message_line_count(message)
        paired_call: dict[str, Any] | None = None
        tool_result = message.get("tool_result")
        if isinstance(tool_result, dict):
            call_id = tool_result.get("tool_call_id")
            if isinstance(call_id, str):
                paired_call = tool_calls.get(call_id)
        if rendered_lines > limit:
            pair_lines = 1 if paired_call is not None else 0
            retained_limit = max(0, limit - pair_lines)
            retained_message = _tail_message(message, retained_limit)
            overflow_count += (
                retained_lines + rendered_lines - retained_limit - pair_lines
            )
            messages.clear()
            retained_lines = retained_limit
            if paired_call is not None:
                messages.append((paired_call, pair_lines))
            messages.append((retained_message, retained_limit))
            retained_lines += pair_lines
            return
        if paired_call is not None and not any(
            _has_tool_call(candidate, tool_result["tool_call_id"])
            for candidate, _ in messages
        ):
            messages.append((paired_call, 1))
            retained_lines += 1
        messages.append((message, rendered_lines))
        retained_lines += rendered_lines
        while retained_lines > limit and messages:
            oldest, dropped = messages[0]
            excess = retained_lines - limit
            if dropped > excess:
                retained = _tail_message(oldest, dropped - excess)
                messages[0] = (retained, dropped - excess)
                retained_lines -= excess
                overflow_count += excess
                break
            messages.popleft()
            retained_lines -= dropped
            overflow_count += dropped

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
                if not at_line_start:
                    raw_tail = _read_partial_row_tail(handle, MAX_AGENT_SCAN_BYTES)
                    message = _oversized_message(raw_tail)
                    if message is not None:
                        append_message(message)
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
                    append_message(message)
                    content = message.get("content")
                    if isinstance(content, list):
                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            call = block.get("tool_call")
                            if (
                                block.get("type") == "tool_use"
                                and isinstance(call, dict)
                                and isinstance(call.get("id"), str)
                            ):
                                tool_calls[call["id"]] = _tool_call_only(
                                    message, call["id"]
                                ) or message
    except (OSError, SessionError):
        return BoundedAgentMessages((), None)
    marker: str | None = None
    if byte_omitted or overflow_count:
        marker = _TRUNCATION_MARKER
    return BoundedAgentMessages(tuple(message for message, _ in messages), marker)


def read_agent_messages(
    path: Path, limit: int = MAX_AGENT_VIEW_LINES
) -> BoundedAgentMessages:
    """Read a bounded, renderable tail of one agent's direct messages."""

    return _read_bounded_messages(path, limit)


def read_agent_transcript(path: Path, limit: int = MAX_AGENT_VIEW_LINES) -> list[str]:
    """Read one agent's direct transcript, without nested child sessions."""

    bounded = _read_bounded_messages(path, limit)
    result = [
        line
        for message in bounded.messages
        for line in _message_lines(message)
    ]
    if bounded.marker is not None:
        result.insert(0, bounded.marker)
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
    """Scrollable bounded transcript rendered by the shared presenter."""

    def __init__(self) -> None:
        from .transcript import TranscriptWidget
        from .transcript_presenter import TranscriptPresenter

        self.lines: list[str] = []
        self.transcript = TranscriptWidget(max_lines=MAX_AGENT_VIEW_LINES)
        self.presenter = TranscriptPresenter(
            self.transcript,
            Console(),
            lambda: True,
            self.transcript.append,
        )
        self._tool_calls: dict[str, ToolCall] = {}

    @property
    def offset(self) -> int:
        return self.transcript.scroll_offset

    @property
    def is_focusable(self) -> bool:
        return True

    def load(self, path: Path) -> None:
        bounded = read_agent_messages(path)
        self.lines = [
            line
            for message in bounded.messages
            for line in _message_lines(message)
        ]
        if bounded.marker is not None:
            self.lines.insert(0, bounded.marker)
        self.transcript.clear()
        self.presenter.clear()
        self._tool_calls.clear()
        self.transcript.set_line_limit_marker(bounded.marker)
        for raw_message in bounded.messages:
            try:
                message = Message.from_dict(raw_message)
            except (TypeError, ValueError):
                continue
            render_replayed_message(
                message,
                presenter=self.presenter,
                print_user=self._print_user,
                print_unit=self.presenter.print_unit,
                tool_calls=self._tool_calls,
                include_thoughts=True,
                replay_tool_results=True,
                replay_tool_starts=True,
            )

    def _print_user(self, message: Message) -> None:
        prompt = next(
            (
                block.text
                for block in message.content
                if isinstance(block, TextContent) and block.path is None
            ),
            "",
        )
        self.presenter.print_user(
            Text.assemble(("▌ ", theme.USER_ROLE), (prompt, theme.BODY))
        )

    def scroll(self, amount: int) -> None:
        self.transcript._set_scroll_offset(self.offset + amount)

    def half_page(self, amount: int) -> None:
        self.scroll(amount * max(1, self.transcript._viewport_height // 2))

    def top(self) -> None:
        self.transcript._set_scroll_offset(0, allow_follow_tail=False)

    def bottom(self) -> None:
        self.transcript._set_scroll_offset(
            len(self.transcript.lines(self.transcript._content_width)),
        )

    def toggle_latest_agent(self) -> bool:
        return self.transcript.toggle_latest_agent()

    def create_content(self, width: int, height: int | None) -> UIContent:
        return self.transcript.create_content(width, height)

    def vertical_scroll(self, window: Window) -> int:
        del window
        return self.transcript.scroll_offset


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
        if self.current_path == self.root_path and not children:
            self.entries = []
        else:
            self.entries = [self._root_entry(), *children]
        if selected_path is not None:
            self.selected_index = next(
                (index for index, entry in enumerate(self.entries) if entry.path == selected_path),
                1 if children else 0,
            )
        else:
            default_index = 1 if children else 0
            self.selected_index = min(
                max(self.selected_index, default_index),
                max(0, len(self.entries) - 1),
            )

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

    def _root_entry(self) -> AgentEntry:
        metadata = _agent_metadata(self.root_path)
        return AgentEntry(
            self.root_path,
            "main",
            str(metadata.get("agent_type") or "main"),
            str(metadata.get("state") or "running"),
        )

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
        self.back_to_parent()

    def exit_navigation(self, *, preselect_path: Path | None = None) -> None:
        previous_path = self.current_path
        self.current_path = self.root_path
        self._path_stack[:] = [self.root_path]
        self._breadcrumb_labels[:] = ["main"]
        self.selected_index = 0
        self.refresh()
        target = preselect_path
        if target is None and previous_path != self.root_path:
            target = previous_path
        if target is not None:
            self.selected_index = next(
                (
                    index
                    for index, entry in enumerate(self.entries)
                    if entry.path == target
                ),
                self.selected_index,
            )
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
        if entry.path == self.root_path:
            if self.child_view_active:
                self.exit_navigation(preselect_path=self.current_path)
            return
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

    def toggle_latest_agent(self) -> bool:
        if self.child_view_active:
            return self.transcript_control.toggle_latest_agent()
        return False

    def child_scroll(self, amount: int) -> None:
        self.transcript_control.scroll(amount)

    def child_half_page(self, amount: int) -> None:
        self.transcript_control.half_page(amount)

    def child_top(self) -> None:
        self.transcript_control.top()

    def child_bottom(self) -> None:
        self.transcript_control.bottom()
