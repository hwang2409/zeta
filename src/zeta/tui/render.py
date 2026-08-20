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
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolUseContent,
)
from .theme import (
    ACCENT,
    ACCENT_DIM,
    ASSISTANT_BODY,
    CHROME,
    CODE_BG,
    ERROR,
    OK,
    THINKING,
    TOOL_RESULT,
)


MAX_ARGUMENTS = 140
MAX_RESULT = 180


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _arguments(arguments: dict[str, Any]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return _truncate(encoded, MAX_ARGUMENTS)


def _result_summary(value: str) -> str:
    first_line = value.strip().splitlines()[0] if value.strip() else "empty"
    return _truncate(first_line, MAX_RESULT)


def render_markdown(value: str) -> RenderableType:
    """Render assistant text with Rich markdown and fenced-code highlighting."""

    return Markdown(
        value,
        code_theme="monokai",
        hyperlinks=False,
        inline_code_theme="monokai",
        style=ASSISTANT_BODY,
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
        table = Table(show_header=True, header_style="bold")
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
            return [Text(line, style="dim")]
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
                result: list[RenderableType] = [Text(line, style="dim")]
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
            ("▸ ", ACCENT_DIM),
            (event.tool_call.name, ACCENT),
            (f"({_arguments(event.tool_call.arguments)})", CHROME),
        )
    if event.type is StreamEventType.TOOL_EXECUTION_END and event.tool_result:
        style = ERROR if event.tool_result.is_error else OK
        marker = "[tool error]" if event.tool_result.is_error else "[tool result]"
        return Text.assemble(
            ("  ↳ ", TOOL_RESULT),
            (f"{marker} ", style),
            (_result_summary(event.tool_result.content), TOOL_RESULT),
        )
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
            return Text(f"[thinking] {event.content.text}", style=THINKING)
        if isinstance(event.content, RedactedThinkingContent):
            return Text("[thinking] redacted", style=THINKING)
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
) -> Text:
    """Format the persistent status line shown beneath the composer."""

    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    usage_text = ""
    if token_count is not None:
        usage_text = f"tokens {token_count}"
    elif input_tokens is not None or output_tokens is not None:
        usage_text = f"tokens in={input_tokens or 0} out={output_tokens or 0}"

    if session_id is None and retained_tail is None and token_count is None:
        line = f" {provider}/{model}  {loop_state}"
        if usage_text:
            line += f"  {usage_text}"
    else:
        session_text = session_id or "--------"
        tail_text = f"tail {retained_tail}" if retained_tail is not None else "tail ?"
        line = (
            f" session {session_text}  |  {provider}/{model}  |  "
            f"{usage_text or 'tokens ?'}  |  {tail_text}  |  mode {loop_state}"
        )
        if streaming:
            line += "  •"
    if partial:
        line += f"  |  {partial}"
    return Text(line, style=ACCENT_DIM)
