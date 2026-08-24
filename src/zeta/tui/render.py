"""Pure Rich renderers for provider-neutral zeta events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
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
    ToolUseContent,
)
from .theme import (
    ACCENT,
    BODY,
    CHROME,
    CODE_BG,
    DIM,
    ERROR,
    OK,
)


MAX_ARGUMENTS = 140
MAX_RESULT = 180
SPINNER_FRAMES = ("·", "•", "●", "•")


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _arguments(arguments: dict[str, Any]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return _truncate(encoded, MAX_ARGUMENTS)


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
        return Text.assemble(
            ("▸ ", CHROME),
            (event.tool_call.name, ACCENT),
            (f"({_arguments(event.tool_call.arguments)})", CHROME),
        )
    if event.type is StreamEventType.TOOL_EXECUTION_UPDATE and event.delta is not None:
        stream = event.data.get("stream")
        label = f"[{stream}] " if stream in {"stdout", "stderr"} else ""
        return Text(f"  ↳ {label}{event.delta}", style=DIM)
    if event.type is StreamEventType.TOOL_EXECUTION_END and event.tool_result:
        style = ERROR if event.tool_result.is_error else OK
        marker = "[tool error]" if event.tool_result.is_error else "[tool result]"
        content_blocks = event.tool_result.content_blocks or []
        truncated_sizes = [
            block["full_size"]
            for block in content_blocks
            if block["type"] == "text" and block["truncated"]
        ]
        if truncated_sizes:
            marker = f"{marker} [truncated; full_size={max(truncated_sizes)}]"
        content = (
            flatten_tool_content(content_blocks)
            if content_blocks
            else event.tool_result.content
        )
        lines = content.split("\n")
        if not content:
            lines = ["empty"]
        rendered = Text()
        for index, line in enumerate(lines):
            if index:
                rendered.append("\n")
            rendered.append("  ↳ ", style=DIM)
            if index == 0:
                rendered.append(f"{marker} ", style=style)
            rendered.append(_truncate(line, MAX_RESULT), style=DIM)
        return rendered
    if event.type is StreamEventType.ERROR:
        message = event.error.message if event.error else "unknown error"
        return Text(f"[error] {message}", style=ERROR)
    if event.type is StreamEventType.AGENT_END:
        return Text("[done]", style=OK)
    if event.type is StreamEventType.TURN_START:
        turn = event.data.get("turn", "?")
        return Text(f"[turn {turn}]", style=CHROME)
    if event.type is StreamEventType.MESSAGE_UPDATE:
        if isinstance(event.content, ThinkingContent):
            return Text(f"[thinking] {event.content.text}", style=DIM)
        if isinstance(event.content, RedactedThinkingContent):
            return Text("[thinking] redacted", style=DIM)
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
    elif usage.get("total_tokens") is not None:
        token_text = f"tok {usage['total_tokens']}"
    elif token_count is not None:
        token_text = f"tok ~{token_count}"
    else:
        token_text = "tok ?"

    if (
        session_id is None
        and retained_tail is None
        and token_count is None
        and width is None
        and not show_spinner
    ):
        line = f" {provider}/{model}  {loop_state}"
        if input_tokens is not None or output_tokens is not None:
            line += f"  tokens in={input_tokens or 0} out={output_tokens or 0}"
        if partial:
            line += f"  |  {partial}"
        return Text(line, style=CHROME)

    session_text = f"s:{(session_id or '')[:5]}" if session_id else "s:?"
    tail_text = f"tail {retained_tail}" if retained_tail is not None else "tail ?"
    segments = [f"mode {loop_state}", token_text]
    if show_spinner:
        segments.append(SPINNER_FRAMES[spinner_frame % len(SPINNER_FRAMES)])
    optional = [f"{provider}/{model}", session_text, tail_text]
    separator = "  |  " if (width or 0) >= 160 else " | "
    for candidate in optional:
        proposed = separator.join([*segments, candidate])
        if width is None or len(proposed) <= max(1, width):
            segments.append(candidate)
    if partial:
        partial_text = _truncate(partial.replace("\n", " "), 32)
        proposed = separator.join([*segments, partial_text])
        if width is None or len(proposed) <= max(1, width):
            segments.append(partial_text)
    return Text(separator.join(segments), style=CHROME)
