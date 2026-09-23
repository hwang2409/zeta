"""Per-tool TUI cards."""

from __future__ import annotations

import re
from collections.abc import Callable
from difflib import unified_diff
from typing import Any

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text

from ...types import StreamEvent, ToolCall, flatten_tool_content
from .. import theme
from .base import (
    strip_terminal_controls,
    tool_card,
    tool_content,
    tool_panel,
)
from .shared import (
    MAX_CARD_COLUMNS,
    MAX_CARD_LINES,
    infer_language,
)

ToolCardRenderer = Callable[[StreamEvent, bool], RenderableType]
TOOL_CARD_REGISTRY: dict[str, ToolCardRenderer] = {}
DIFF_CONTEXT_LINES = 3


def register_tool_card(*tool_names: str) -> Callable[[ToolCardRenderer], ToolCardRenderer]:
    """Register one renderer for one or more normalized tool names."""

    def register(renderer: ToolCardRenderer) -> ToolCardRenderer:
        for tool_name in tool_names:
            TOOL_CARD_REGISTRY[tool_name.strip().lower()] = renderer
        return renderer

    return register


def card_path(call: ToolCall, result: Any | None = None) -> str:
    arguments_path = call.arguments.get("path")
    if isinstance(arguments_path, str) and arguments_path:
        return arguments_path
    structured = getattr(result, "structured_content", None)
    result_path = structured.get("path") if isinstance(structured, dict) else None
    return result_path if isinstance(result_path, str) else "<unknown>"


def read_content(event: StreamEvent) -> str:
    """Return text for a read card while preserving image read receipts."""

    result = event.tool_result
    blocks = result.content_blocks if result is not None else None
    structured = result.structured_content if result is not None else None
    if (
        blocks
        and any(block.get("type") == "image" for block in blocks)
        and isinstance(structured, dict)
        and structured.get("format") in {"png", "jpeg", "gif", "webp"}
    ):
        return flatten_tool_content([block for block in blocks if block.get("type") == "text"])
    return tool_content(event)


def card_line_count(value: str) -> int:
    if not value:
        return 0
    return value.count("\n") + (0 if value.endswith("\n") else 1)


def bounded_card_lines(value: str) -> tuple[list[str], int]:
    visible: list[str] = []
    start = 0
    while start < len(value) and len(visible) < MAX_CARD_LINES:
        end = value.find("\n", start)
        if end < 0:
            line = value[start:]
            start = len(value)
        else:
            line = value[start:end]
            start = end + 1
        visible.append(line.removesuffix("\r")[:MAX_CARD_COLUMNS])
    total = card_line_count(value)
    return visible, max(0, total - len(visible))


def read_header(call: ToolCall, content: str, result: Any | None) -> Text:
    path = card_path(call, result)
    offset = call.arguments.get("offset", 0)
    line_start = offset + 1 if type(offset) is int and offset >= 0 else 1
    line_count = max(1, card_line_count(content))
    line_end = line_start + line_count - 1
    return Text.assemble(
        (call.name, theme.COMMAND),
        (f" {path}", theme.BODY),
        (f" · lines {line_start}-{line_end}", theme.DIM),
    )


def read_tool_card(event: StreamEvent, running: bool) -> RenderableType:
    call = event.tool_call
    if call is None:
        return tool_card(event, running=running)
    if running or event.tool_result is None:
        return Text(
            f"⏺ {call.name} {card_path(call)} · running",
            style=theme.RECEIPT,
            overflow="ellipsis",
            no_wrap=True,
        )
    result = event.tool_result
    image_read = (
        result.content_blocks
        and any(block.get("type") == "image" for block in result.content_blocks)
        and isinstance(result.structured_content, dict)
        and result.structured_content.get("format") in {"png", "jpeg", "gif", "webp"}
    )
    if result.is_error or (
        any(block.get("type") != "text" for block in result.content_blocks or [])
        and not image_read
    ):
        return tool_card(event)
    content = read_content(event)
    raw_visible, omitted = bounded_card_lines(content)
    visible = [strip_terminal_controls(line) for line in raw_visible]
    syntax = Syntax(
        "\n".join(visible),
        infer_language(card_path(call, result)),
        theme=theme.CODE_THEME,
        word_wrap=True,
        background_color="default",
    )
    body: RenderableType = syntax
    if omitted:
        body = Group(syntax, Text(f"… +{omitted} lines", style=theme.AFFORDANCE))
    return tool_panel(call, body, header=read_header(call, content, result))


def cap_diff_lines(lines: list[str]) -> tuple[list[str], int]:
    visible = [line[:MAX_CARD_COLUMNS] for line in lines[:MAX_CARD_LINES]]
    return visible, max(0, len(lines) - len(visible))


def iter_card_lines(value: str, *, reverse: bool = False):
    if not reverse:
        start = 0
        while start < len(value):
            end = value.find("\n", start)
            if end < 0:
                end = len(value)
            yield value[start:end].removesuffix("\r")
            start = end + 1
        return

    end = len(value)
    if end and value[end - 1] == "\n":
        end -= 1
    while end > 0:
        start = value.rfind("\n", 0, end)
        yield value[start + 1 : end].removesuffix("\r")
        if start < 0:
            return
        end = start


def common_line_prefix(old: str, new: str, limit: int) -> int:
    count = 0
    for old_line, new_line in zip(iter_card_lines(old), iter_card_lines(new), strict=False):
        if old_line != new_line:
            break
        count += 1
        if count >= limit:
            break
    return count


def common_line_suffix(old: str, new: str, limit: int) -> int:
    if limit <= 0:
        return 0
    count = 0
    for old_line, new_line in zip(
        iter_card_lines(old, reverse=True),
        iter_card_lines(new, reverse=True),
        strict=False,
    ):
        if old_line != new_line:
            break
        count += 1
        if count >= limit:
            break
    return count


def card_line_window(value: str, start: int, end: int) -> list[str]:
    lines: list[str] = []
    for index, line in enumerate(iter_card_lines(value)):
        if index >= end:
            break
        if index >= start:
            lines.append(line[:MAX_CARD_COLUMNS])
            if len(lines) >= MAX_CARD_LINES:
                break
    return lines


def diff_window_bounds(total: int, change_start: int, change_end: int) -> tuple[int, int]:
    if total == 0:
        return 0, 0
    start = max(0, change_start - DIFF_CONTEXT_LINES)
    preview_end = min(change_end, start + MAX_CARD_LINES - DIFF_CONTEXT_LINES)
    end = min(total, max(start + 1, preview_end + DIFF_CONTEXT_LINES))
    return start, min(end, start + MAX_CARD_LINES)


HUNK_RE = re.compile(
    r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@(.*)$"
)


def offset_hunk_header(line: str, old_start: int, new_start: int) -> str:
    match = HUNK_RE.match(line)
    if match is None:
        return line
    old_line = int(match.group(1)) + old_start
    new_line = int(match.group(3)) + new_start
    return (
        f"@@ -{old_line}{match.group(2) or ''} +{new_line}"
        f"{match.group(4) or ''} @@{match.group(5)}"
    )


def bounded_unified_diff(old: str, new: str, path: str) -> tuple[list[str], int]:
    if old == new:
        return [], 0
    old_count = card_line_count(old)
    new_count = card_line_count(new)
    prefix = common_line_prefix(old, new, min(old_count, new_count))
    suffix_limit = min(old_count - prefix, new_count - prefix)
    suffix = common_line_suffix(old, new, suffix_limit)
    old_change_end = old_count - suffix
    new_change_end = new_count - suffix
    if prefix == old_change_end == new_change_end:
        return [], 0

    old_start, old_end = diff_window_bounds(old_count, prefix, old_change_end)
    new_start, new_end = diff_window_bounds(new_count, prefix, new_change_end)
    old_window = card_line_window(old, old_start, old_end)
    new_window = card_line_window(new, new_start, new_end)
    lines = list(
        unified_diff(old_window, new_window, fromfile=path, tofile=path, lineterm="")
    )
    lines = [
        offset_hunk_header(line, old_start, new_start) if line.startswith("@@") else line
        for line in lines
    ]
    if old_start or new_start:
        lines.insert(3, "  … unchanged lines omitted")
    if old_end < old_change_end or new_end < new_change_end or suffix:
        lines.append("  … unchanged lines omitted")
    return cap_diff_lines(lines)


def diff_from_structured(result: Any, path: str) -> tuple[list[str], str, int] | None:
    structured = result.structured_content
    if not isinstance(structured, dict):
        return None
    for key in ("diff", "unified_diff"):
        value = structured.get(key)
        if isinstance(value, str):
            lines, omitted = bounded_card_lines(value)
            return lines, "structured diff", omitted
    old = next((structured.get(key) for key in ("old_content", "before", "pre_image")), None)
    new = next((structured.get(key) for key in ("new_content", "after")), None)
    if isinstance(old, str) and isinstance(new, str):
        visible, omitted = bounded_unified_diff(old, new, path)
        return visible, "structured pre-image", omitted
    return None


def diff_lines(event: StreamEvent, path: str) -> tuple[list[str], str, int]:
    call = event.tool_call
    result = event.tool_result
    assert call is not None and result is not None
    if call.name.strip().lower() == "edit":
        old = call.arguments.get("old_string")
        new = call.arguments.get("new_string")
        if isinstance(old, str) and isinstance(new, str):
            visible, omitted = bounded_unified_diff(old, new, path)
            return visible, "", omitted
    structured_diff = diff_from_structured(result, path)
    if structured_diff is not None:
        return structured_diff
    content = call.arguments.get("content")
    if not isinstance(content, str):
        return [], "", 0
    content_lines, content_omitted = bounded_card_lines(content)
    if not content_lines:
        return [], "", 0
    total = len(content_lines) + content_omitted
    lines = [
        "--- /dev/null",
        f"+++ {path}",
        f"@@ -0,0 +1,{total} @@",
        *(f"+{line}" for line in content_lines),
    ]
    visible = [line[:MAX_CARD_COLUMNS] for line in lines[:MAX_CARD_LINES]]
    omitted = total + 3 - len(visible)
    return visible, "new file · pre-image unavailable", omitted


def render_diff(lines: list[str], note: str, omitted: int = 0) -> Text:
    rendered = Text(overflow="ellipsis", no_wrap=True)
    for index, line in enumerate(lines):
        if index:
            rendered.append("\n")
        if line.startswith("+"):
            style = theme.DIFF_ADD
        elif line.startswith("-"):
            style = theme.DIFF_REMOVE
        elif line.startswith(("@@", " ")):
            style = theme.DIFF_CONTEXT
        else:
            style = theme.DIM
        rendered.append(strip_terminal_controls(line)[:MAX_CARD_COLUMNS], style=style)
    if omitted > 0:
        if rendered:
            rendered.append("\n")
        rendered.append(f"… +{omitted} diff lines", style=theme.AFFORDANCE)
    if note:
        if rendered:
            rendered.append("\n")
        rendered.append(note, style=theme.DIM)
    return rendered


@register_tool_card("write", "edit")
def write_edit_tool_card(event: StreamEvent, running: bool) -> RenderableType:
    call = event.tool_call
    if call is None:
        return tool_card(event, running=running)
    if running or event.tool_result is None:
        return tool_panel(call, Text("running…", style=theme.DIM))
    result = event.tool_result
    if result.is_error:
        return tool_card(event)
    path = card_path(call, result)
    lines, note, omitted = diff_lines(event, path)
    if not lines:
        return tool_card(event)
    header = Text.assemble(
        (call.name, theme.COMMAND),
        (f" {path}", theme.BODY),
        (" · diff", theme.DIM),
    )
    return tool_panel(call, render_diff(lines, note, omitted), header=header)


@register_tool_card("read")
def registered_read_tool_card(event: StreamEvent, running: bool) -> RenderableType:
    return read_tool_card(event, running)
