"""Compatibility exports for the focused TUI card package."""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import monotonic as _monotonic
from typing import Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIContent, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseEvent
from rich.console import Console
from rich.text import Text

from ..core.checkpoints import ConversationIntegrityError, load_session_json
from ..core.session_files import SessionError, open_session_file, session_directory
from ..core.store import ConversationStore
from ..protocol.types import Message, ToolCall
from . import theme
from .cards.agent import (
    AgentCard,
    AgentRunCommandMixin,
    _read_lifecycle,
    render_agent_expanded,
    render_agent_progress,
    render_agent_receipt,
)
from .cards.agent_sync import (
    AgentTranscriptSnapshot,
    refresh_agent_transcript,
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
from .user import displayed_user_text, user_message

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
    "agent_navigation_style_rules",
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
AGENT_LIST_PAGE_SIZE = 5
MAX_AGENT_LIST_ROWS = AGENT_LIST_PAGE_SIZE + 1
MAX_AGENT_SCAN_BYTES = MAX_AGENT_VIEW_LINES * (MAX_AGENT_LINE_CHARS + 256)
_TRUNCATION_MARKER = "[older lines omitted]"
_TERMINAL_AGENT_STATES = frozenset({"completed", "failed", "canceled"})
_AGENT_REFRESH_INTERVAL_SECONDS = 1.0
_AGENT_IDLE_REFRESH_INTERVAL_SECONDS = 4.0


def agent_navigation_style_rules() -> dict[str, str]:
    """Return the prompt-toolkit style rules for the subagent navigator.

    Defined next to the navigator it styles so the composition root keeps a
    single owner for these class names. Subagents read as green identity in
    both the list and the breadcrumb; the main agent stays blue; red is left
    for real failures.
    """

    return {
        "agent-list": f"fg:{theme.AGENT_CHILD}",
        "agent-list.selected": f"fg:{theme.AGENT_CHILD} bold",
        "agent-breadcrumb": f"fg:{theme.CHROME}",
        "agent-breadcrumb.main": f"fg:{theme.AGENT_MAIN} bold",
        "agent-breadcrumb.child": f"fg:{theme.AGENT_CHILD} bold",
        "agent-view": f"fg:{theme.BODY}",
    }


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
    """Read a bounded, renderable tail of one agent's direct messages."""

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
                    metadata = message.get("metadata")
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("zeta_event") == "empty_turn_nudge"
                    ):
                        continue
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

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any,
    ) -> int:
        del width, max_available_height, wrap_lines, get_line_prefix
        self.navigator.refresh()
        return self.navigator.list_height

    def create_content(self, width: int, height: int | None) -> UIContent:
        del height
        self.navigator.refresh()
        entries = self.navigator.entries
        page_index = self.navigator.page_index
        page_start = page_index * AGENT_LIST_PAGE_SIZE
        page_entries = entries[page_start : page_start + AGENT_LIST_PAGE_SIZE]
        has_pager = len(entries) > AGENT_LIST_PAGE_SIZE
        list_focused = self.navigator.list_focused()

        def get_line(index: int) -> list[tuple[str, str]]:
            if index < len(page_entries):
                entry_index = page_start + index
                entry = page_entries[index]
                selected = list_focused and entry_index == self.navigator.selected_index
                marker = ">" if selected else " "
                label = f"{entry.label} · {entry.agent_type} · {entry.state}"
                style = "class:agent-list.selected" if selected else "class:agent-list"
                return [(style, f"{marker} {label}")]

            page_count = self.navigator.page_count
            pager = f"page {page_index + 1}/{page_count} · {len(entries)} agents"
            return [("class:agent-list", pager.rjust(width))]

        return UIContent(
            get_line=get_line,
            line_count=len(page_entries) + int(has_pager),
            cursor_position=Point(x=0, y=self.navigator.selected_index - page_start),
            show_cursor=False,
        )


class AgentTranscriptControl(UIControl):
    """Scrollable child transcript synchronized through a read-only store."""

    _RENDER_BATCH_SIZE = 4

    def __init__(self) -> None:
        from .transcript import TranscriptPresenter, TranscriptWidget

        self.transcript = TranscriptWidget(max_lines=None)
        self.presenter = TranscriptPresenter(
            self.transcript,
            Console(),
            lambda: True,
            self.transcript.append,
        )
        self._tool_calls: dict[str, ToolCall] = {}
        self._path: Path | None = None
        self._store: ConversationStore | None = None
        self._snapshot: AgentTranscriptSnapshot | None = None
        self._entry_units: dict[str, list[Any]] = {}
        self._unit_entries: dict[Any, str] = {}
        self._sync_task: asyncio.Task[bool] | None = None
        self._pending_path: Path | None = None

    @property
    def offset(self) -> int:
        return self.transcript.scroll_offset

    @property
    def is_focusable(self) -> bool:
        return True

    async def load(self, path: Path) -> bool:
        """Load a selected child without doing storage work on the event loop."""

        if path != self._path:
            self._reset_rendered(path)
        if self._store is None:
            try:
                await asyncio.to_thread(self._replace_store, path)
            except (ConversationIntegrityError, OSError, ValueError):
                return False
        return await self.sync(path)

    async def sync(self, path: Path) -> bool:
        """Apply one active-branch snapshot, yielding between render batches."""

        if path != self._path or self._store is None:
            return await self.load(path)
        try:
            snapshot = await asyncio.to_thread(refresh_agent_transcript, self._store)
        except (ConversationIntegrityError, OSError, ValueError):
            return False
        previous = self._snapshot
        if snapshot == previous:
            return False
        extends = (
            previous is not None
            and snapshot.entry_ids[: len(previous.entry_ids)] == previous.entry_ids
            and snapshot.messages[: len(previous.messages)] == previous.messages
        )
        if extends:
            await self._replay_batches(snapshot.messages[len(previous.messages) :], path)
        else:
            await self._rebuild(snapshot, path)
        self._snapshot = snapshot
        return True

    def request_sync(
        self, path: Path, invalidate: Callable[[], None] | None = None
    ) -> bool:
        """Coalesce a UI refresh onto one asynchronous synchronization task."""

        self._pending_path = path
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._pending_path = None
            return asyncio.run(self.load(path))
        if self._sync_task is not None and not self._sync_task.done():
            return False

        async def run_pending() -> bool:
            changed = False
            while self._pending_path is not None:
                requested = self._pending_path
                self._pending_path = None
                changed = await self.load(requested) or changed
            if changed and invalidate is not None:
                invalidate()
            return changed

        self._sync_task = loop.create_task(run_pending())
        return False

    def _reset_rendered(self, path: Path) -> None:
        self.presenter.clear()
        self._tool_calls.clear()
        self.transcript.set_line_limit_marker(None)
        self._entry_units.clear()
        self._unit_entries.clear()
        self._path = path
        self._snapshot = None

    def _replace_store(self, path: Path) -> None:
        previous, self._store = self._store, None
        if previous is not None:
            previous.close()
        self._store = ConversationStore(
            path.parent,
            session_id=path.name,
            _read_only=True,
            _must_exist=True,
        )

    async def _rebuild(
        self, snapshot: AgentTranscriptSnapshot, path: Path
    ) -> None:
        anchor_id, anchor_unit_index, anchor_offset, follow_tail = self._capture_anchor()
        old_entry_ids = (
            self._snapshot.entry_ids if self._snapshot is not None else ()
        )
        self.presenter.clear()
        self._tool_calls.clear()
        self.transcript.set_line_limit_marker(None)
        self._entry_units.clear()
        self._unit_entries.clear()
        await self._replay_batches(snapshot.messages, path)
        if not follow_tail:
            target = self._nearest_entry(
                anchor_id, old_entry_ids, snapshot.entry_ids
            )
            units = self._entry_units.get(target or "", ())
            if units:
                unit_index = min(anchor_unit_index, len(units) - 1)
                self.transcript.restore_scroll_anchor(units[unit_index], anchor_offset)

    async def _replay_batches(
        self, messages: list[tuple[str, dict[str, Any]]] | tuple[tuple[str, dict[str, Any]], ...], path: Path
    ) -> None:
        for start in range(0, len(messages), self._RENDER_BATCH_SIZE):
            for entry_id, raw_message in messages[start : start + self._RENDER_BATCH_SIZE]:
                self._replay(entry_id, raw_message, path)
            await asyncio.sleep(0.001)

    def _replay(
        self, entry_id: str, raw_message: dict[str, Any], path: Path
    ) -> None:
        try:
            message = Message.from_dict(_bounded_message(raw_message))
        except (TypeError, ValueError):
            return
        unit_start = len(self.transcript._units)
        render_replayed_message(
            message,
            presenter=self.presenter,
            print_user=self._print_user,
            print_unit=self.presenter.print_unit,
            tool_calls=self._tool_calls,
            include_thoughts=True,
            replay_tool_results=True,
            replay_tool_starts=True,
            session_path=path,
        )
        units = self.transcript._units[unit_start:]
        self._entry_units.setdefault(entry_id, []).extend(units)
        self._unit_entries.update((unit, entry_id) for unit in units)

    def _capture_anchor(self) -> tuple[str | None, int, int, bool]:
        unit, offset = self.transcript.scroll_anchor
        entry_id = self._unit_entries.get(unit)
        units = self._entry_units.get(entry_id or "", ())
        unit_index = units.index(unit) if unit in units else 0
        return entry_id, unit_index, offset, self.transcript.follow_tail

    @staticmethod
    def _nearest_entry(
        anchor_id: str | None,
        old_version: tuple[str, ...],
        new_version: tuple[str, ...],
    ) -> str | None:
        if anchor_id is None:
            return None
        if anchor_id in new_version:
            return anchor_id
        try:
            anchor_index = old_version.index(anchor_id)
        except ValueError:
            return None
        new_ids = set(new_version)
        candidates = (
            (abs(index - anchor_index), entry_id)
            for index, entry_id in enumerate(old_version)
            if entry_id in new_ids
        )
        return min(candidates, default=(0, None))[1]

    def _print_user(self, message: Message) -> None:
        self.presenter.print_user(
            user_message(Text(displayed_user_text(message), style=theme.BODY))
        )

    def scroll(self, amount: int) -> None:
        self.transcript.scroll_lines(amount)

    def half_page(self, amount: int) -> None:
        self.transcript.scroll_lines(
            amount * max(1, self.transcript._viewport_height // 2)
        )

    def top(self) -> None:
        self.transcript.scroll_to_top()

    def bottom(self) -> None:
        self.transcript.scroll_to_bottom()

    def toggle_latest_agent(self) -> bool:
        return self.transcript.toggle_latest_agent()

    def create_content(self, width: int, height: int | None) -> UIContent:
        return self.transcript.create_content(width, height)

    def vertical_scroll(self, window: Window) -> int:
        return self.transcript.vertical_scroll(window)

    def mouse_handler(self, mouse_event: MouseEvent):
        return self.transcript.mouse_handler(mouse_event)


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
            content=FormattedTextControl(self._breadcrumb_fragments),
            height=1,
            wrap_lines=False,
        )
        self.view_container = HSplit([self.breadcrumb_window, self.transcript_window])
        self._layout: Any = None
        self._composer_buffer: Any = None
        self._transcript_layout: Any = None
        self._main_transcript: Any = None
        self._main_transcript_parent: Any = None
        self._main_transcript_index: int | None = None
        self._todo_store: ConversationStore | None = None
        self._todo_store_path: Path | None = None
        self._metadata_cache: dict[Path, tuple[tuple[int, int, int], dict[str, Any]]] = {}
        self._refresh_signature: tuple[object, ...] | None = None
        self._last_refresh_at = float("-inf")
        self._refresh_handle: asyncio.TimerHandle | None = None
        self._invalidate: Callable[[], None] | None = None
        self.refresh(force=True)

    def _breadcrumb_fragments(self) -> list[tuple[str, str]]:
        """Style the breadcrumb so the main agent and its subagents differ.

        The first crumb is the main agent (blue); every deeper crumb is a
        subagent (green). Separators stay in the dim chrome colour.
        """

        fragments: list[tuple[str, str]] = []
        for index, label in enumerate(self._breadcrumb_labels):
            if index:
                fragments.append(("class:agent-breadcrumb", " > "))
            role = (
                "class:agent-breadcrumb.main"
                if index == 0
                else "class:agent-breadcrumb.child"
            )
            fragments.append((role, label))
        return fragments

    @property
    def child_view_active(self) -> bool:
        return self.current_path != self.root_path

    def todo_store(self) -> ConversationStore | None:
        """Return the selected session's read-only TODO store.

        Child stores are deliberately leased only while their transcript is
        selected. A failed refresh is isolated to that child and never falls
        back to the parent store.
        """
        if not self.child_view_active:
            return self.store
        if self._todo_store_path != self.current_path:
            self._close_todo_store()
            self._todo_store_path = self.current_path
            try:
                self._todo_store = ConversationStore(
                    self.current_path.parent,
                    session_id=self.current_path.name,
                    cwd=self.store.cwd,
                    _read_only=True,
                    _must_exist=True,
                )
            except (ConversationIntegrityError, OSError, ValueError):
                self._todo_store = None
        if self._todo_store is None:
            return None
        try:
            self._todo_store.refresh()
        except (ConversationIntegrityError, OSError, ValueError):
            self._close_todo_store()
            return None
        return self._todo_store

    def _close_todo_store(self) -> None:
        if self._todo_store is not None:
            self._todo_store.close()
        self._todo_store = None
        self._todo_store_path = None

    def child_view_focused(self) -> bool:
        return self._layout is not None and self._layout.has_focus(self.transcript_window)

    @property
    def list_visible(self) -> bool:
        self.refresh()
        return bool(self.entries)

    @property
    def page_index(self) -> int:
        return self.selected_index // AGENT_LIST_PAGE_SIZE

    @property
    def page_count(self) -> int:
        return max(
            1,
            (len(self.entries) + AGENT_LIST_PAGE_SIZE - 1) // AGENT_LIST_PAGE_SIZE,
        )

    @property
    def list_height(self) -> int:
        if not self.entries:
            return 0
        page_start = self.page_index * AGENT_LIST_PAGE_SIZE
        page_rows = min(AGENT_LIST_PAGE_SIZE, len(self.entries) - page_start)
        return page_rows + int(len(self.entries) > AGENT_LIST_PAGE_SIZE)

    def bind_layout(
        self,
        layout: Any,
        composer_buffer: Any,
        invalidate: Callable[[], None] | None = None,
    ) -> None:
        self._layout = layout
        self._composer_buffer = composer_buffer
        self._invalidate = invalidate
        self._schedule_refresh(_AGENT_REFRESH_INTERVAL_SECONDS)

    def unbind_layout(self) -> None:
        if self._refresh_handle is not None:
            self._refresh_handle.cancel()
            self._refresh_handle = None
        self._invalidate = None
        self._layout = None
        self._composer_buffer = None

    def bind_transcript_layout(self, layout: Any, main_transcript: Any) -> None:
        self._transcript_layout = layout
        self._main_transcript = main_transcript
        self._main_transcript_parent = None
        self._main_transcript_index = None
        layout_children = getattr(layout, "children", [])
        if not layout_children:
            return
        parent = layout_children[0]
        for index, child in enumerate(getattr(parent, "children", [])):
            if child is main_transcript:
                self._main_transcript_parent = parent
                self._main_transcript_index = index
                return

    def _switch_transcript(self) -> None:
        if self._transcript_layout is None:
            return
        replacement = self.view_container if self.child_view_active else self._main_transcript
        if self._main_transcript_parent is not None and self._main_transcript_index is not None:
            self._main_transcript_parent.children[self._main_transcript_index] = replacement
        else:
            self._transcript_layout.children[0] = replacement

    @staticmethod
    def _path_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _agents_directory_signature(self) -> tuple[object, ...]:
        agents = self.current_path / "agents"
        signature = self._path_signature(agents)
        try:
            entry_count = sum(1 for _ in os.scandir(agents))
        except OSError:
            entry_count = 0
        return signature, entry_count

    def _agent_tree_signature(self) -> tuple[object, ...]:
        """Detect tree changes while statting lifecycle only for live children."""

        running = tuple(
            (entry.path, self._path_signature(entry.path / "agent_lifecycle.json"))
            for entry in self.entries
            if entry.state not in _TERMINAL_AGENT_STATES
            and (entry.path != self.root_path or self.child_view_active)
        )
        return (
            self.current_path,
            self.selected_index,
            self._agents_directory_signature(),
            running,
        )

    def _schedule_refresh(self, delay: float) -> None:
        if self._invalidate is None or self._refresh_handle is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refresh_handle = loop.call_later(delay, self._refresh_after_debounce)

    def request_refresh(self) -> None:
        """Re-arm discovery promptly after an event that can create a child."""

        if self._refresh_handle is not None:
            self._refresh_handle.cancel()
            self._refresh_handle = None
        self._schedule_refresh(0)

    def _refresh_after_debounce(self) -> None:
        self._refresh_handle = None
        if self._invalidate is None:
            return
        if self.refresh():
            self._invalidate()
        delay = (
            _AGENT_REFRESH_INTERVAL_SECONDS
            if any(
                entry.state not in _TERMINAL_AGENT_STATES
                for entry in self.entries
            )
            else _AGENT_IDLE_REFRESH_INTERVAL_SECONDS
        )
        self._schedule_refresh(delay)

    def refresh(self, *, force: bool = False) -> bool:
        """Refresh at most once per second unless fresh state is required."""

        now = _monotonic()
        elapsed = now - self._last_refresh_at
        if not force and elapsed < _AGENT_REFRESH_INTERVAL_SECONDS:
            self._schedule_refresh(_AGENT_REFRESH_INTERVAL_SECONDS - elapsed)
            self._resize_list_window()
            return False
        self._last_refresh_at = now
        transcript_changed = (
            self.transcript_control.request_sync(self.current_path, self._invalidate)
            if self.child_view_active
            else False
        )
        signature = self._agent_tree_signature()
        if not force and signature == self._refresh_signature:
            return transcript_changed
        was_list_focused = bool(self.entries) and self.list_focused()
        selected_path = self.entries[self.selected_index].path if self.entries else None
        previous_index = self.selected_index
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
                min(previous_index, max(0, len(self.entries) - 1)),
            )
        else:
            default_index = 1 if children else 0
            self.selected_index = min(
                max(previous_index, default_index),
                max(0, len(self.entries) - 1),
            )
        self._resize_list_window()
        # Keep the pre-read signature: mutations during _children() must be
        # visible to the next refresh rather than being paired with stale data.
        self._refresh_signature = signature
        if was_list_focused and not self.entries:
            self.focus_composer()
        return True

    def _resize_list_window(self) -> None:
        list_height = min(self.list_height, MAX_AGENT_LIST_ROWS)
        self.list_window.height = Dimension(
            min=list_height,
            preferred=list_height,
            max=MAX_AGENT_LIST_ROWS,
        )

    def _children(self, path: Path, fallback: dict[Path, dict[str, Any]]) -> list[AgentEntry]:
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
            metadata = self._cached_agent_metadata(child, fallback.get(child))
            entries.append(
                AgentEntry(
                    child,
                    str(metadata.get("description") or f"child {child.name}"),
                    str(metadata.get("agent_type") or "child"),
                    str(metadata.get("state") or "running"),
                )
            )
        candidate_set = set(candidates)
        self._metadata_cache = {
            child: cached
            for child, cached in self._metadata_cache.items()
            if child == self.root_path or child in candidate_set
        }
        return [
            entry for entry in entries if entry.state not in _TERMINAL_AGENT_STATES
        ]

    def _cached_agent_metadata(
        self, path: Path, fallback: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        lifecycle_path = path / "agent_lifecycle.json"
        try:
            stat = lifecycle_path.stat()
            key = (1, stat.st_mtime_ns, stat.st_size)
        except OSError:
            try:
                stat = (path / "session_state.json").stat()
                key = (0, stat.st_mtime_ns, stat.st_size)
            except OSError:
                key = (0, 0, 0)
        cached = self._metadata_cache.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        metadata = _agent_metadata(path, fallback)
        self._metadata_cache[path] = (key, metadata)
        return metadata

    def _root_entry(self) -> AgentEntry:
        metadata = self._cached_agent_metadata(self.root_path)
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
        self.refresh(force=True)
        if self.list_visible and self._layout is not None:
            self._layout.focus(self.list_window)
        else:
            self.focus_composer()

    def list_back(self) -> None:
        self.back_to_parent()

    def exit_navigation(self, *, preselect_path: Path | None = None) -> None:
        previous_path = self.current_path
        self._close_todo_store()
        self.current_path = self.root_path
        self._path_stack[:] = [self.root_path]
        self._breadcrumb_labels[:] = ["main"]
        self.selected_index = 0
        self.refresh(force=True)
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
        self.refresh(force=True)
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
        self._close_todo_store()
        self._path_stack.append(entry.path)
        self._breadcrumb_labels.append(entry.label)
        self.transcript_control.request_sync(entry.path, self._invalidate)
        self.refresh(force=True)
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
        self._close_todo_store()
        self._path_stack.pop()
        self.current_path = self._path_stack[-1]
        self._breadcrumb_labels.pop()
        self.refresh(force=True)
        self.selected_index = next(
            (
                index
                for index, entry in enumerate(self.entries)
                if entry.path == child_path
            ),
            0,
        )
        if self.child_view_active:
            self.transcript_control.request_sync(self.current_path, self._invalidate)
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
