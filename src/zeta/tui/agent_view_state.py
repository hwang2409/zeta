"""Value objects and pure helpers for agent-view navigation state."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit.document import Document

from ..core.agent_progress import live_agent_lifecycle
from ..core.checkpoints import ConversationIntegrityError, load_session_json
from ..core.session_files import SessionError, open_session_file, session_directory
from ..core.transcript_recovery.runtime import repair_command_for_child_path
from ..protocol.types import is_passive_harness_message
from . import theme


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


MAX_AGENT_VIEW_LINES = 240
MAX_AGENT_LINE_CHARS = 2_000
MAX_AGENT_SCAN_BYTES = MAX_AGENT_VIEW_LINES * (MAX_AGENT_LINE_CHARS + 256)
_TRUNCATION_MARKER = "[older lines omitted]"
_TERMINAL_AGENT_STATES = frozenset({"completed", "failed", "canceled"})
def agent_navigation_style_rules() -> dict[str, str]:
    """Return the prompt-toolkit style rules for the subagent navigator."""

    return {
        "agent-list": f"fg:{theme.AGENT_CHILD}",
        "agent-list.selected": f"fg:{theme.AGENT_CHILD} bold",
        "agent-breadcrumb": f"fg:{theme.CHROME}",
        "agent-breadcrumb.main": f"fg:{theme.AGENT_MAIN} bold",
        "agent-breadcrumb.child": f"fg:{theme.AGENT_CHILD} bold",
        "agent-view": f"fg:{theme.BODY}",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = load_session_json(path)
    except (ConversationIntegrityError, OSError):
        return {}
    return value if type(value) is dict else {}


def _agent_metadata(path: Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    lifecycle_path = path / "agent_lifecycle.json"
    lifecycle = live_agent_lifecycle(lifecycle_path) or _read_json(lifecycle_path)
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
    # Each search runs again only once start passes its hit: one pass, not a
    # scan to the end of the text for every line.
    newline = text.find("\n")
    carriage = text.find("\r")
    while start < len(text):
        if 0 <= newline < start:
            newline = text.find("\n", start)
        if 0 <= carriage < start:
            carriage = text.find("\r", start)
        ends = [index for index in (newline, carriage) if index >= 0]
        end = min(ends) if ends else -1
        if end < 0:
            yield text[start:]
            return
        yield text[start:end]
        start = end + 1
        if text[end] == "\r" and start < len(text) and text[start] == "\n":
            start += 1


def _bounded_text(text: str) -> str:
    return "\n".join(short_agent_value(line) for line in _text_lines(text))


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
    role = short_agent_value(message.get("role", "message"), 32)
    metadata = message.get("metadata")
    if (
        role == "user"
        and isinstance(metadata, dict)
        and metadata.get("zeta.origin") == "agent_send"
    ):
        role = "parent message"
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                for line in _text_lines(block["text"], tail):
                    yield f"{role}: {short_agent_value(line)}"
            elif block.get("type") == "thinking":
                yield f"{role}: thought"
                if isinstance(block.get("text"), str):
                    for line in _text_lines(block["text"], tail):
                        yield f"{role}: {short_agent_value(line)}"
            elif block.get("type") == "redacted_thinking":
                yield f"{role}: thought: redacted"
            elif block.get("type") == "tool_use" and isinstance(block.get("tool_call"), dict):
                call = block["tool_call"]
                name = call.get("name", "tool")
                arguments = call.get("arguments", {})
                if isinstance(arguments, dict):
                    args = " ".join(
                        f"{key}={short_agent_value(value, 160)}"
                        for key, value in sorted(arguments.items())
                    )
                    yield f"tool: {short_agent_value(name, 64)} {short_agent_value(args, 400)}".rstrip()
    result = message.get("tool_result")
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str):
            for line in _text_lines(content, tail):
                yield f"tool result: {short_agent_value(line)}"


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
                    if is_passive_harness_message(message):
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
    if result:
        return result
    command = repair_command_for_child_path(path)
    return [f"transcript corrupt; run {command}" if command else "transcript unavailable"]
