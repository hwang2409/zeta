"""Pure Rich renderers for provider-neutral zeta events."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from ..types import (
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolUseContent,
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

    return Markdown(value, code_theme="monokai", hyperlinks=False)


def render_code(value: str, language: str = "text") -> Syntax:
    """Render a complete code block with syntax highlighting."""

    return Syntax(value, language or "text", theme="monokai", word_wrap=True)


@dataclass(slots=True)
class MarkdownStream:
    """Turn committed lines into scrollback renderables.

    Fenced code is held until its closing fence so Rich can highlight it as one
    block. Ordinary lines commit immediately.
    """

    language: str | None = None
    code_lines: list[str] = field(default_factory=list)

    def consume(self, line: str) -> list[RenderableType]:
        stripped = line.strip()
        if self.language is not None:
            if stripped.startswith("```"):
                result: list[RenderableType] = [
                    render_code("\n".join(self.code_lines), self.language),
                    Text(line, style="dim"),
                ]
                self.language = None
                self.code_lines.clear()
                return result
            self.code_lines.append(line)
            return []

        if stripped.startswith("```"):
            self.language = stripped[3:].strip() or "text"
            return [Text(line, style="dim")]
        return [render_markdown(line) if line else Text("")]

    def flush(self) -> list[RenderableType]:
        if self.language is None or not self.code_lines:
            return []
        result = [render_code("\n".join(self.code_lines), self.language)]
        self.language = None
        self.code_lines.clear()
        return result


def render_event(event: StreamEvent) -> RenderableType | None:
    """Render one event that belongs in scrollback.

    Text deltas return None. The app owns their newline buffer and status bar.
    """

    if event.type is StreamEventType.TOOL_EXECUTION_START and event.tool_call:
        return Text(
            f"[tool] {event.tool_call.name} {_arguments(event.tool_call.arguments)}",
            style="yellow",
        )
    if event.type is StreamEventType.TOOL_EXECUTION_END and event.tool_result:
        style = "red" if event.tool_result.is_error else "green"
        marker = "[tool error]" if event.tool_result.is_error else "[tool result]"
        return Text(
            f"{marker} {_result_summary(event.tool_result.content)}",
            style=style,
        )
    if event.type is StreamEventType.ERROR:
        message = event.error.message if event.error else "unknown error"
        return Text(f"[error] {message}", style="bold red")
    if event.type is StreamEventType.AGENT_END:
        return Text("[done]", style="green")
    if event.type is StreamEventType.TURN_START:
        turn = event.data.get("turn", "?")
        return Text(f"[turn {turn}]", style="dim")
    if event.type is StreamEventType.MESSAGE_UPDATE:
        if isinstance(event.content, ThinkingContent):
            return Text(f"[thinking] {event.content.text}", style="dim italic")
        if isinstance(event.content, RedactedThinkingContent):
            return Text("[thinking] redacted", style="dim italic")
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
) -> Text:
    """Format the persistent status line shown beneath the composer."""

    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    usage_text = ""
    if input_tokens is not None or output_tokens is not None:
        usage_text = f"  tokens in={input_tokens or 0} out={output_tokens or 0}"
    line = f" {provider}/{model}  {loop_state}{usage_text}"
    if partial:
        line += f"  |  {partial}"
    return Text(line, style="green")
