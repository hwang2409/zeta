"""Pure Rich renderers for provider-neutral zeta events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from rich.console import RenderableType
from rich.markdown import Markdown
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
    COMMAND,
    DIM,
    ERROR,
    OK,
    RECEIPT,
    THOUGHT,
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


def _summary_appropriate(event: StreamEvent) -> bool:
    if event.data.get("summary_appropriate") is True:
        return True
    result = event.tool_result
    if result is None or not result.content_blocks:
        return False
    return any(
        block.get("annotations", {}).get("summary_appropriate") is True
        for block in result.content_blocks
        if block.get("type") == "text"
    )


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
    line_count = len(_tool_content(event).splitlines()) or 1
    if (
        line_count < 3
        or call.name.lower() in SUMMARY_TOOLS
        or _summary_appropriate(event)
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
        return f'"{pattern}" [{count} matches]'
    return _arguments(arguments)


def _tool_receipt(event: StreamEvent) -> Text:
    call = event.tool_call
    assert call is not None
    result = event.tool_result
    assert result is not None
    prefix = "failed · " if result.is_error else ""
    suffix = _receipt_arguments(call, _tool_content(event))
    return Text(
        f"{prefix}{call.name}{f' {suffix}' if suffix else ''}",
        style=RECEIPT,
    )


def _render_tool_output(content: str, extra_lines: list[str] | None = None) -> Text:
    lines = content.splitlines() or ["empty"]
    truncated = len(lines) > MAX_TOOL_LINES
    visible = lines[:MAX_TOOL_LINES]
    if extra_lines:
        visible.extend(extra_lines)
    rendered = Text(
        "\n".join(_truncate(line, MAX_RESULT) for line in visible),
        style=BODY,
    )
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
    return Panel(
        Text.assemble(_tool_header(call), "\n", body),
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
            token = token.rstrip(".,!?;:").lower()
            if token in ABBREVIATIONS:
                continue
        if index + 1 == len(normalized) or normalized[index + 1].isspace():
            return _truncate(normalized[: index + 1], MAX_RESULT)
    return _truncate(normalized, MAX_RESULT)


def format_thought(value: str, duration: float | None = None) -> Text:
    label = "thought"
    if duration is not None:
        label += f" · {duration:.1f}s"
    summary = collapse_thought(value)
    return Text.assemble((label, THOUGHT), (f"  {summary}" if summary else "", THOUGHT))


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


def render_markdown(value: str) -> RenderableType:
    """Render assistant text with Rich markdown and fenced-code highlighting."""

    return Markdown(
        value,
        code_theme="monokai",
        hyperlinks=False,
        inline_code_theme="monokai",
        style=BODY,
    )


def render_code(value: str, language: str = "text") -> Syntax:
    """Render a complete code block with syntax highlighting."""

    return Syntax(
        value,
        language or "text",
        theme="monokai",
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
            return [render_markdown(line) if line else Text("") for line in lines]

        separator_index = next(
            (index for index, line in enumerate(lines) if self._is_table_separator(line)),
            None,
        )
        if separator_index != 1:
            return render_lines()
        header = self._table_cells(lines[0])
        if header is None:
            return render_lines()
        table = Table(show_header=True, header_style=ACCENT)
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
        return [render_markdown(line) if line else Text("")]

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

    if event.type is StreamEventType.TOOL_EXECUTION_START and event.tool_call:
        if event.tool_call.name.lower() in RECEIPT_TOOLS:
            suffix = _receipt_arguments(event.tool_call, "")
            return Text(
                f"{event.tool_call.name}{f' {suffix}' if suffix else ''} · running",
                style=RECEIPT,
            )
        return _tool_card(event, running=True)
    if event.type is StreamEventType.TOOL_EXECUTION_UPDATE and event.delta is not None:
        stream = event.data.get("stream")
        label = f"[{stream}] " if stream in {"stdout", "stderr"} else ""
        return Text(f"  ↳ {label}{event.delta}", style=DIM)
    if event.type is StreamEventType.TOOL_EXECUTION_END and event.tool_result:
        if tool_render_mode(event) == "receipt":
            return _tool_receipt(event)
        return _tool_card(event)
    if event.type is StreamEventType.ERROR:
        message = event.error.message if event.error else "unknown error"
        return Text(f"[error] {message}", style=ERROR)
    if event.type is StreamEventType.AGENT_END:
        return Text("done", style=OK)
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
) -> Text:
    """Format the persistent status line shown beneath the composer."""

    show_spinner = streaming if spinner_active is None else spinner_active
    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if input_tokens is not None and output_tokens is not None:
        token_text = f"tok {input_tokens}/{output_tokens}"
    elif input_tokens is not None or output_tokens is not None:
        token_text = f"tok in={input_tokens or 0} out={output_tokens or 0}"
    elif usage.get("total_tokens") is not None:
        token_text = f"tok {usage['total_tokens']}"
    elif token_count is not None:
        token_text = f"tok ~{token_count}"
    else:
        token_text = "tok ?"

    state = loop_state if loop_state in {"streaming", "idle", "interrupted", "compacting"} else "streaming"
    session_text = (session_id or "session")[:8]
    left = f"{session_text} · {model} · {state}"
    right_segments = [token_text]
    if retained_tail is not None:
        right_segments.append(f"tail {retained_tail}")
    cache_read = usage.get("cache_read_input_tokens", usage.get("cache_read"))
    cache_write = usage.get("cache_creation_input_tokens", usage.get("cache_creation"))
    if cache_read is not None or cache_write is not None:
        right_segments.append(f"cache {cache_read or 0}/{cache_write or 0}")
    right = " · ".join(right_segments)
    if show_spinner:
        right = f"{SPINNER_FRAMES[spinner_frame % len(SPINNER_FRAMES)]} {right}"
    if partial and width is None:
        right += f" · {_truncate(partial.replace(chr(10), ' '), 32)}"
    if width is not None and len(left) + len(right) + 2 > width:
        right = " · ".join(right_segments)
    if width is not None and len(left) + len(right) + 2 > width:
        right = token_text
    if width is not None and len(left) + len(right) + 2 > width:
        left = _truncate(left, max(1, width - len(right) - 2))
    if width is not None and len(left) + len(right) < width:
        return Text(f"{left}{' ' * (width - len(left) - len(right))}{right}", style=CHROME)
    return Text(f"{left}  {right}", style=CHROME)
