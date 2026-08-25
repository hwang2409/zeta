"""Pure Rich renderers for provider-neutral zeta events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from rich.cells import cell_len
from rich.console import RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..types import (
    flatten_tool_content,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolUseContent,
)
from .theme import (
    ACCENT,
    AFFORDANCE,
    BODY,
    CARD_BG,
    CARD_BORDER,
    CHROME,
    CODE_BG,
    CODE_THEME,
    COMMAND,
    DIM,
    ERROR,
    RECEIPT,
    THOUGHT,
    VIM_STATE,
)


MAX_ARGUMENTS = 140
MAX_RESULT = 180
MAX_TOOL_LINES = 15
SPINNER_FRAMES = ("·", "•", "●", "•")
RECEIPT_TOOLS = frozenset(
    {"read", "glob", "grep", "search", "find", "list", "websearch"}
)
SUMMARY_TOOLS = frozenset({"glob", "grep", "search", "find", "websearch"})
ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "etc",
        "mr",
        "mrs",
        "ms",
        "dr",
        "vs",
        "no",
        "fig",
        "prof",
        "sr",
        "jr",
    }
)
OSC_RE = re.compile(r"(?:\x1b\]|\x9d)[^\x07\x1b]*(?:\x07|\x1b\\)")
ESC_RE = re.compile(r"\x1b(?:[PX^_].*?\x1b\\|\][^\x07]*(?:\x07|\x1b\\))")
CSI_UNSUPPORTED_RE = re.compile(
    r"(?:\x1b\[|\x9b)[0-?]*[ -/]*(?!m)[@-~]"
)
ToolRenderMode = Literal["card", "receipt"]


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _arguments(arguments: dict[str, Any]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return _truncate(encoded, MAX_ARGUMENTS)


def _tool_content(event: StreamEvent) -> str:
    result = event.tool_result
    if result is None:
        return ""
    blocks = result.content_blocks or []
    return flatten_tool_content(blocks) if blocks else result.content


def tool_render_mode(event: StreamEvent) -> ToolRenderMode:
    """Choose the one display mode for completed tool results."""

    call = event.tool_call
    result = event.tool_result
    if call is None or result is None or call.name.lower() not in RECEIPT_TOOLS:
        return "card"
    if result.is_error or any(
        block.get("type") != "text" for block in result.content_blocks or []
    ):
        return "card"
    content = _tool_content(event)
    line_count = len(content.splitlines()) or 1
    if any(cell_len(_strip_terminal_controls(line)) > MAX_RESULT for line in content.splitlines()):
        return "card"
    if (
        line_count < 3
        or call.name.lower() in SUMMARY_TOOLS
    ):
        return "receipt"
    return "card"


def _command(arguments: dict[str, Any]) -> str:
    value = arguments.get("cmd", arguments.get("command", ""))
    return str(value) if value else _arguments(arguments)


def _tool_header(call: ToolCall) -> Text:
    if call.name.lower() == "bash":
        return Text.assemble(("$ ", COMMAND), (_command(call.arguments), COMMAND))
    return Text.assemble((call.name, COMMAND), (f" {_arguments(call.arguments)}", DIM))


def _receipt_arguments(call: ToolCall, content: str) -> str:
    arguments = call.arguments
    name = call.name.lower()
    if name == "read":
        label = str(arguments.get("path", arguments.get("file", "")))
        limit = arguments.get("limit")
        return f"{label} [limit={limit}]" if limit is not None else label
    if name in SUMMARY_TOOLS:
        pattern = arguments.get(
            "pattern", arguments.get("query", arguments.get("path", ""))
        )
        if not content:
            return f'"{pattern}"'
        matches = re.search(r"(\d+)\s+matches?", content, re.IGNORECASE)
        count = (
            matches.group(1)
            if matches
            else str(sum(bool(line.strip()) for line in content.splitlines()))
        )
        location = arguments.get("path", arguments.get("cwd", "."))
        return f'"{pattern}" in {location} · {count} matches'
    return _arguments(arguments)


def _strip_terminal_controls(value: str) -> str:
    """Remove terminal controls that are unsafe in transcript scrollback."""

    value = OSC_RE.sub("", value)
    value = ESC_RE.sub("", value)
    value = CSI_UNSUPPORTED_RE.sub("", value)
    return "".join(
        character
        for character in value
        if character in {"\n", "\t"} or ord(character) >= 0x20
    )


def _safe_text(value: str, *, style: str) -> Text:
    return Text.from_ansi(
        _strip_terminal_controls(value),
        style=style,
        overflow="ellipsis",
        no_wrap=True,
    )


def _tool_receipt(event: StreamEvent) -> Text:
    call = event.tool_call
    assert call is not None
    result = event.tool_result
    assert result is not None
    prefix = "⏺ "
    if result.is_error:
        prefix += "failed · "
    suffix = _receipt_arguments(call, _tool_content(event))
    return Text(
        f"{prefix}{call.name}{f' {suffix}' if suffix else ''}",
        style=RECEIPT,
        overflow="ellipsis",
        no_wrap=True,
    )


def _render_tool_output(content: str, extra_lines: list[str] | None = None) -> Text:
    lines = content.splitlines() or ["empty"]
    truncated = len(lines) > MAX_TOOL_LINES
    visible = lines[:MAX_TOOL_LINES]
    if extra_lines:
        visible.extend(extra_lines)
    rendered = Text(style=BODY, overflow="ellipsis", no_wrap=True)
    for index, line in enumerate(visible):
        if index:
            rendered.append("\n")
        rendered.append(_safe_text(line, style=BODY))
    if truncated:
        rendered.append(f"\n… +{len(lines) - MAX_TOOL_LINES} lines", style=AFFORDANCE)
    return rendered


def _tool_body(event: StreamEvent) -> Text:
    content = _tool_content(event) or "empty"
    result = event.tool_result
    extra_lines: list[str] = []
    if result is not None and result.content_blocks:
        sizes = [
            block["full_size"]
            for block in result.content_blocks
            if block.get("type") == "text" and block.get("truncated")
        ]
        if sizes:
            extra_lines.append(f"[truncated; full_size={max(sizes)}]")
    return _render_tool_output(content, extra_lines)


def _tool_card(event: StreamEvent, *, running: bool = False) -> Panel:
    call = event.tool_call or ToolCall(
        event.tool_result.tool_call_id if event.tool_result is not None else "unknown",
        "tool",
        {},
    )
    body = Text("running…", style=DIM) if running else _tool_body(event)
    return _tool_panel(
        call,
        body,
        error=bool(event.tool_result and event.tool_result.is_error),
    )


def _tool_panel(call: ToolCall, body: Text, *, error: bool = False) -> Panel:
    content = Text.assemble(_tool_header(call), "\n", body)
    content.no_wrap = True
    content.overflow = "ellipsis"
    return Panel(
        content,
        border_style=ERROR if error else CARD_BORDER,
        style=CARD_BG,
        padding=(0, 1),
        expand=True,
    )


def render_tool_progress(call: ToolCall, content: str) -> Panel:
    """Render streamed tool output inside the same card surface."""

    body = (
        Text("running…", style=DIM)
        if not content
        else _render_tool_output(content)
    )
    return _tool_panel(call, body)


def collapse_thought(value: str) -> str:
    """Keep the first sentence of provider reasoning for the transcript."""

    normalized = " ".join(value.replace("\n", " ").split())
    if not normalized:
        return ""
    for index, character in enumerate(normalized):
        if character not in ".!?":
            continue
        if character == ".":
            token = normalized[: index + 1].rsplit(" ", 1)[-1]
            normalized_token = token.rstrip(".,!?;:").lower()
            if normalized_token in ABBREVIATIONS or re.fullmatch(
                r"(?:[a-z]\.){2,}", token.lower()
            ):
                continue
        end = index + 1
        while end < len(normalized) and normalized[end] in "\"'”’)]}":
            end += 1
        if end == len(normalized) or normalized[end].isspace():
            return _truncate(normalized[:end], MAX_RESULT)
    return _truncate(normalized, MAX_RESULT)


def format_thought(value: str, duration: float | None = None) -> Text:
    summary = collapse_thought(value)
    parts = ["✱ thought"]
    if summary:
        parts.append(summary)
    if duration is not None:
        parts.append(f"{duration:.1f}s")
    return Text(" · ".join(parts), style=THOUGHT)


def _duration(data: dict[str, Any]) -> float | None:
    for key in ("duration", "elapsed_seconds", "thinking_duration"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for key in ("duration_ms", "elapsed_ms", "thinking_duration_ms"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 1000
    return None


_INLINE_MARKER = re.compile(r"(\*\*|__|`|(?<!\*)\*(?!\*)|(?<!_)_(?!_))")


def render_line(value: str) -> Text:
    """Keep one model line intact while styling common inline markdown."""

    rendered = Text(style=BODY)
    markers = list(_INLINE_MARKER.finditer(value))
    cursor = 0
    marker_stack: list[tuple[str, str]] = []
    for marker in markers:
        if marker.start() > cursor:
            style = marker_stack[-1][1] if marker_stack else BODY
            rendered.append(value[cursor : marker.start()], style=style)
        token = marker.group()
        if marker_stack and marker_stack[-1][0] == token:
            marker_stack.pop()
        elif token == "`":
            marker_stack.append((token, BODY))
        elif token in {"**", "__"}:
            marker_stack.append((token, f"bold {BODY}"))
        else:
            marker_stack.append((token, f"italic {BODY}"))
        cursor = marker.end()
    if marker_stack:
        return Text(value, style=BODY)
    if cursor < len(value):
        style = marker_stack[-1][1] if marker_stack else BODY
        rendered.append(value[cursor:], style=style)
    return rendered


def render_code(value: str, language: str = "text") -> Syntax:
    """Render a complete code block with syntax highlighting."""

    return Syntax(
        value,
        language or "text",
        theme=CODE_THEME,
        word_wrap=True,
        background_color=CODE_BG,
    )


@dataclass(slots=True)
class MarkdownStream:
    """Turn committed lines into scrollback renderables.

    Fenced code switches to a syntax-aware line renderer as soon as its opener
    arrives. Every completed line reaches scrollback without waiting for the
    closing fence.
    """

    language: str | None = None
    fence_char: str | None = None
    fence_length: int = 0
    table_lines: list[str] | None = None

    @staticmethod
    def _fence(line: str) -> tuple[str, int, str] | None:
        stripped = line.strip()
        if not stripped or stripped[0] not in "`~":
            return None
        char = stripped[0]
        length = len(stripped) - len(stripped.lstrip(char))
        if length < 3:
            return None
        return char, length, stripped[length:]

    @staticmethod
    def _table_cells(line: str) -> list[str] | None:
        stripped = line.strip()
        if not stripped.startswith("|"):
            return None
        body = stripped[1:]
        if body.endswith("|"):
            body = body[:-1]
        return [cell.strip() for cell in body.split("|")]

    @classmethod
    def _is_table_separator(cls, line: str) -> bool:
        cells = cls._table_cells(line)
        return bool(cells) and all(
            re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
            for cell in cells
        )

    def _render_table(self) -> list[RenderableType]:
        lines = self.table_lines
        self.table_lines = None
        if lines is None:
            return []

        def render_lines() -> list[RenderableType]:
            return [render_line(line) if line else Text("") for line in lines]

        separator_index = next(
            (index for index, line in enumerate(lines) if self._is_table_separator(line)),
            None,
        )
        if separator_index != 1:
            return render_lines()
        header = self._table_cells(lines[0])
        if header is None:
            return render_lines()
        table = Table(show_header=True, header_style=ACCENT, expand=True)
        for cell in header:
            table.add_column(cell)
        for line in lines[separator_index + 1 :]:
            cells = self._table_cells(line)
            if cells is None:
                return render_lines()
            table.add_row(*(cells + [""] * len(header))[: len(header)])
        return [table]

    def _consume_plain(self, line: str) -> list[RenderableType]:
        fence = self._fence(line)
        if fence is not None:
            char, length, language = fence
            self.language = language.strip() or "text"
            self.fence_char = char
            self.fence_length = length
            return [Text(line, style=DIM)]
        if self.table_lines is not None:
            if self._table_cells(line) is not None:
                self.table_lines.append(line)
                return []
            result = self._render_table()
            result.extend(self._consume_plain(line))
            return result
        if self._table_cells(line) is not None:
            self.table_lines = [line]
            return []
        return [render_line(line) if line else Text("")]

    def consume(self, line: str) -> list[RenderableType]:
        if self.language is not None:
            fence = self._fence(line)
            if (
                fence is not None
                and fence[0] == self.fence_char
                and fence[1] >= self.fence_length
                and not fence[2].strip()
            ):
                result: list[RenderableType] = [Text(line, style=DIM)]
                self.language = None
                self.fence_char = None
                self.fence_length = 0
                return result
            return [render_code(line, self.language)]

        return self._consume_plain(line)

    def flush(self) -> list[RenderableType]:
        self.language = None
        self.fence_char = None
        self.fence_length = 0
        return self._render_table()


def render_event(event: StreamEvent) -> RenderableType | None:
    """Render one event that belongs in scrollback.

    Text deltas return None. The app owns their newline buffer and status bar.
    """

    if event.type is StreamEventType.MESSAGE_END and event.data.get("truncated"):
        dropped = event.data.get("dropped_tool_calls")
        if type(dropped) is int and dropped:
            noun = "tool call" if dropped == 1 else "tool calls"
            return Text(
                f"response truncated (stream ended early; dropped {dropped} incomplete {noun})",
                style=DIM,
            )
        return Text("response truncated (stream ended early)", style=DIM)
    if event.type is StreamEventType.TOOL_EXECUTION_START and event.tool_call:
        if event.tool_call.name.lower() in RECEIPT_TOOLS:
            suffix = _receipt_arguments(event.tool_call, "")
            return Text(
                f"⏺ {event.tool_call.name}{f' {suffix}' if suffix else ''} · running",
                style=RECEIPT,
            )
        return _tool_card(event, running=True)
    if event.type is StreamEventType.TOOL_EXECUTION_UPDATE and event.delta is not None:
        stream = event.data.get("stream")
        label = f"[{stream}] " if stream in {"stdout", "stderr"} else ""
        return _safe_text(f"  ↳ {label}{event.delta}", style=DIM)
    if event.type is StreamEventType.TOOL_EXECUTION_END and event.tool_result:
        if tool_render_mode(event) == "receipt":
            return _tool_receipt(event)
        return _tool_card(event)
    if event.type is StreamEventType.ERROR:
        message = event.error.message if event.error else "unknown error"
        return Text(f"[error] {message}", style=ERROR)
    if event.type in {
        StreamEventType.AGENT_END,
        StreamEventType.COMPACTION_START,
        StreamEventType.COMPACTION_END,
    }:
        return None
    if event.type is StreamEventType.TURN_START:
        return None
    if event.type is StreamEventType.MESSAGE_UPDATE:
        if isinstance(event.content, ThinkingContent):
            return format_thought(event.content.text, _duration(event.data))
        if isinstance(event.content, RedactedThinkingContent):
            return format_thought("redacted", _duration(event.data))
        if isinstance(event.content, ToolUseContent):
            return None
        if isinstance(event.content, TextContent) or event.delta is not None:
            return None
    return None


def format_status(
    provider: str,
    model: str,
    loop_state: str,
    usage: dict[str, Any] | None = None,
    partial: str = "",
    *,
    session_id: str | None = None,
    token_count: int | None = None,
    retained_tail: int | None = None,
    streaming: bool = False,
    width: int | None = None,
    spinner_frame: int = 0,
    spinner_active: bool | None = None,
    model_window: int | None = None,
    vim_state: str | None = None,
) -> Text:
    """Format the compact status bar shown below the composer."""

    show_spinner = streaming if spinner_active is None else spinner_active
    usage = usage or {}
    del provider, model, partial, retained_tail
    context_tokens = token_count
    if context_tokens is None:
        context_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    context_tokens = context_tokens or 0
    window = model_window or 200_000
    percent = round((context_tokens / window) * 100) if window else 0
    if context_tokens >= 1000:
        value = f"{context_tokens / 1000:.1f}K".removesuffix(".0K")
    else:
        value = str(context_tokens)
    context_text = f"{value} ({percent}%)"

    state = loop_state if loop_state in {
        "streaming",
        "tool-running",
        "approval",
        "idle",
        "interrupted",
        "compacting",
    } else "streaming"
    if show_spinner and state not in {"interrupted", "compacting"}:
        state_text = f"{SPINNER_FRAMES[spinner_frame % len(SPINNER_FRAMES)]} {state}"
    else:
        state_text = state
    state_segment = f"{state_text}  {context_text}"
    left_segments = [state_segment]
    if vim_state:
        left_segments.insert(0, vim_state)
    left = "  ".join(left_segments)
    right_segments = ["/status", "ctrl+c interrupt", "ctrl+d quit"]
    if session_id:
        right_segments.append(session_id[:8])
    if width is None:
        value = f"{left}  {' · '.join(right_segments)}"
    else:
        value = left
        candidates = (left, state_segment) if vim_state else (left,)
        for candidate_left in candidates:
            value = candidate_left
            for start in range(len(right_segments)):
                right = " · ".join(right_segments[start:])
                gap = width - cell_len(candidate_left) - cell_len(right)
                if gap >= 2:
                    value = f"{candidate_left}{' ' * gap}{right}"
                    break
            if value != candidate_left or candidate_left == state_segment:
                break
        if cell_len(value) > width:
            fitted = Text(value, no_wrap=True, overflow="ellipsis")
            fitted.truncate(width, overflow="ellipsis")
            value = fitted.plain.rstrip(" ·")
    rendered = Text(value, style=CHROME)
    if vim_state and value.startswith(vim_state):
        rendered.stylize(VIM_STATE, 0, len(vim_state))
    return rendered
