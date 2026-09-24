"""Neutral rendering support shared by the TUI card implementations."""

from __future__ import annotations

import re
from typing import Any

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from ...types import StreamEvent, ToolCall, flatten_tool_content
from .. import theme
from .shared import BoundedToolOutput, scan_tool_output

MAX_TOOL_LINES = 15
OSC_RE = re.compile(r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)")
ESC_RE = re.compile(r"\x1b(?:[PX^_].*?\x1b\\|\][^\x07]*(?:\x07|\x1b\\))")
C1_DCS_RE = re.compile(r"\x90.*?(?:\x9c|\x1b\\)", re.DOTALL)
CSI_UNSUPPORTED_RE = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*(?!m)[@-~]")
CSI_SGR_RE = re.compile(r"\x1b\[[0-?]*[ -/]*m")


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def readable_argument(value: Any) -> str:
    if isinstance(value, dict):
        pairs = " ".join(
            f"{key}={readable_argument(nested)}"
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        )
        return "{" + pairs + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(readable_argument(item) for item in value) + "]"
    return str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def arguments(arguments: dict[str, Any]) -> str:
    parts = [f"{key}={readable_argument(arguments[key])}" for key in sorted(arguments)]
    return truncate(" ".join(parts), 140)


def command(arguments: dict[str, Any]) -> str | None:
    for key in ("command", "cmd"):
        if key in arguments:
            return str(arguments[key])
    return None


def shell_syntax(value: str) -> Syntax:
    return Syntax(
        value,
        "bash",
        theme=theme.CODE_THEME,
        word_wrap=True,
        background_color="default",
    )


def tool_content(event: StreamEvent) -> str:
    result = event.tool_result
    if result is None:
        return ""
    blocks = result.content_blocks or []
    if not blocks:
        return result.content
    tool_name = event.tool_call.name if event.tool_call is not None else "tool"
    return flatten_tool_content(blocks, detailed_images=True, tool_name=tool_name)


def strip_terminal_controls(value: str) -> str:
    """Remove terminal controls that are unsafe in transcript scrollback."""

    value = value.replace("\x9b", "\x1b[")
    value = OSC_RE.sub("", value)
    value = C1_DCS_RE.sub("", value)
    value = ESC_RE.sub("", value)
    value = CSI_UNSUPPORTED_RE.sub("", value)
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\x1b":
            match = CSI_SGR_RE.match(value, index)
            if match is not None:
                result.append(match.group())
                index = match.end()
                continue
            index += 1
            continue
        character = value[index]
        if (
            character in {"\n", "\t"}
            or ord(character) >= 0x20
            and not 0x80 <= ord(character) <= 0x9F
        ):
            result.append(character)
        index += 1
    return "".join(result)


def safe_text(value: str, *, style: str, wrap: bool = False) -> Text:
    return Text.from_ansi(
        strip_terminal_controls(value),
        style=style,
        overflow="fold" if wrap else "ellipsis",
        no_wrap=not wrap,
    )


def tool_header(call: ToolCall) -> RenderableType:
    value = command(call.arguments)
    if value is not None:
        return Columns(
            [Text.assemble((call.name, theme.COMMAND)), shell_syntax(value)],
            padding=(0, 1),
            expand=False,
        )
    return Text.assemble((call.name, theme.COMMAND), (f" {arguments(call.arguments)}", theme.DIM))


def render_tool_output(
    content: str,
    extra_lines: list[str] | None = None,
    *,
    scan: BoundedToolOutput | None = None,
) -> Text:
    scan = scan_tool_output(content) if scan is None else scan
    visible = list(scan.lines[:MAX_TOOL_LINES])
    omitted = max(0, scan.total_lines - MAX_TOOL_LINES) if scan.total_lines is not None else None
    if extra_lines:
        visible.extend(extra_lines)
    rendered = Text(style=theme.BODY, overflow="ellipsis", no_wrap=True)
    for index, line in enumerate(visible):
        if index:
            rendered.append("\n")
        style = theme.DIM if line.startswith("[image block]") else theme.BODY
        rendered.append(safe_text(line, style=style))
    if omitted:
        rendered.append(f"\n… +{omitted} lines", style=theme.AFFORDANCE)
    elif scan.truncated:
        rendered.append("\n… more lines", style=theme.AFFORDANCE)
    return rendered


def split_tool_output(
    content: str,
    *,
    scan: BoundedToolOutput | None = None,
) -> tuple[list[tuple[str, str]], str]:
    """Split standard command receipts without hiding generic tool output."""

    scan = scan_tool_output(content) if scan is None else scan
    lines = scan.lines
    section_labels = {"stdout:", "stderr:", "result:"}
    if not any(line in section_labels for line in lines):
        generic = "\n".join(line for line in lines if not line.startswith("exit_code:"))
        return [], generic

    sections: list[tuple[str, str]] = []
    generic_lines: list[str] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_label is not None:
            sections.append((current_label, "\n".join(current_lines)))

    for line in lines:
        if line in section_labels:
            flush()
            current_label = line[:-1]
            current_lines = []
        elif line.startswith("exit_code:"):
            continue
        elif current_label is not None:
            current_lines.append(line)
        else:
            generic_lines.append(line)
    flush()
    return sections, "\n".join(generic_lines)


def tool_body(
    event: StreamEvent,
    *,
    scan: BoundedToolOutput | None = None,
) -> Text | None:
    content = tool_content(event)
    scan = scan_tool_output(content) if scan is None else scan
    sections, generic = split_tool_output(content, scan=scan)
    result = event.tool_result
    extra_lines: list[str] = []
    if result is not None and result.content_blocks:
        char_sizes = [
            block["full_size_chars"]
            for block in result.content_blocks
            if block.get("type") == "text" and block.get("truncated") and "full_size_chars" in block
        ]
        byte_sizes = [
            block["full_size"]
            for block in result.content_blocks
            if block.get("type") == "text" and block.get("truncated") and "full_size_chars" not in block
        ]
        if char_sizes:
            extra_lines.append(f"[truncated; full_size_chars={max(char_sizes)} chars]")
        if byte_sizes:
            extra_lines.append(f"[truncated; full_size={max(byte_sizes)} bytes]")
    rendered = Text(style=theme.BODY, overflow="ellipsis", no_wrap=True)
    visible_sections = [(label, value) for label, value in sections if value.strip()]
    if generic.strip():
        rendered.append_text(render_tool_output(generic, extra_lines))
        extra_lines = []
    if sections:
        for label, value in visible_sections:
            if rendered:
                rendered.append("\n\n")
            rendered.append(f"{label}:\n", style=theme.DIM)
            rendered.append_text(render_tool_output(value, extra_lines))
            extra_lines = []
    return rendered if rendered else None


def tool_panel(
    call: ToolCall,
    body: RenderableType | None,
    *,
    error: bool = False,
    header: RenderableType | None = None,
) -> Panel:
    header = tool_header(call) if header is None else header
    if body is None:
        content: RenderableType = header
    elif isinstance(header, Text) and isinstance(body, Text):
        content = Text.assemble(header, "\n", body)
        content.no_wrap = True
        content.overflow = "ellipsis"
    else:
        content = Group(header, body)
    return Panel(
        content,
        border_style=theme.ERROR if error else theme.CARD_BORDER,
        style=theme.CARD_BG,
        padding=(0, 1),
        expand=True,
    )


def tool_card(
    event: StreamEvent,
    *,
    running: bool = False,
    scan: BoundedToolOutput | None = None,
) -> Panel:
    call = event.tool_call or ToolCall(
        event.tool_result.tool_call_id if event.tool_result is not None else "unknown",
        "tool",
        {},
    )
    body = Text("running…", style=theme.DIM) if running else tool_body(event, scan=scan)
    return tool_panel(call, body, error=bool(event.tool_result and event.tool_result.is_error))


def compact_tool_card(rendered: RenderableType) -> RenderableType:
    if not isinstance(rendered, Panel):
        return rendered
    content = rendered.renderable
    if isinstance(content, Group) and content.renderables:
        header = content.renderables[0]
    elif isinstance(content, Text):
        header = content.split("\n")[0]
    else:
        return rendered
    if isinstance(header, Text):
        header = header.copy()
        header.append(" · expand: ctrl+x ctrl+o", style=theme.DIM)
    else:
        header = Group(header, Text("expand: ctrl+x ctrl+o", style=theme.DIM))
    return Panel(
        header,
        border_style=rendered.border_style,
        style=rendered.style,
        padding=rendered.padding,
        expand=rendered.expand,
    )
