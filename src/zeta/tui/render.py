"""Pure Rich renderers for provider-neutral zeta events."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from markdown_it import MarkdownIt
from rich.cells import cell_len
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.table import Table
from rich import box
from mdit_py_plugins.tasklists import tasklists_plugin

from ..types import (
    ErrorInfo,
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
from .agent_card import AgentCard


MAX_ARGUMENTS = 140
MAX_RESULT = 180
MAX_TOOL_LINES = 15
MAX_ERROR_REASON = 400
SPINNER_FRAMES = ("·", "•", "●", "•")
RECEIPT_TOOLS = frozenset(
    {"read", "glob", "grep", "search", "find", "list", "websearch"}
)
SUMMARY_TOOLS = frozenset({"glob", "grep", "search", "find", "websearch"})
OSC_RE = re.compile(r"(?:\x1b\]|\x9d)[^\x07\x1b]*(?:\x07|\x1b\\)")
ESC_RE = re.compile(r"\x1b(?:[PX^_].*?\x1b\\|\][^\x07]*(?:\x07|\x1b\\))")
CSI_UNSUPPORTED_RE = re.compile(
    r"(?:\x1b\[|\x9b)[0-?]*[ -/]*(?!m)[@-~]"
)
ToolRenderMode = Literal["card", "receipt"]

# Keep the renderer imports used by callers stable while dispatch stays in AgentCard.
render_agent_expanded = AgentCard.render_expanded
render_agent_progress = AgentCard.render_progress
render_agent_receipt = AgentCard.render_receipt


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _readable_argument(value: Any) -> str:
    if isinstance(value, dict):
        pairs = " ".join(
            f"{key}={_readable_argument(nested)}"
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        )
        return "{" + pairs + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_readable_argument(item) for item in value) + "]"
    return str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _arguments(arguments: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(arguments):
        parts.append(f"{key}={_readable_argument(arguments[key])}")
    return _truncate(" ".join(parts), MAX_ARGUMENTS)


def _tool_content(event: StreamEvent) -> str:
    result = event.tool_result
    if result is None:
        return ""
    blocks = result.content_blocks or []
    if not blocks:
        return result.content
    tool_name = event.tool_call.name if event.tool_call is not None else "tool"
    return flatten_tool_content(blocks, detailed_images=True, tool_name=tool_name)


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


def _command(arguments: dict[str, Any]) -> str | None:
    for key in ("command", "cmd"):
        if key in arguments:
            return str(arguments[key])
    return None


def _shell_syntax(command: str) -> Syntax:
    return Syntax(
        command,
        "bash",
        theme=CODE_THEME,
        word_wrap=True,
        background_color="default",
    )


def _tool_header(call: ToolCall) -> RenderableType:
    command = _command(call.arguments)
    if command is not None:
        return Columns(
            [
                Text.assemble((call.name, COMMAND)),
                _shell_syntax(command),
            ],
            padding=(0, 1),
            expand=False,
        )
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


def render_error_card(event: StreamEvent) -> Panel:
    """Render a bounded error with a retry affordance when supported."""

    error = event.error
    code = error.code if error is not None and error.code else "backend_error"
    raw_reason = error.message if error is not None else "unknown error"
    reason = _strip_terminal_controls(raw_reason).strip()
    reason = reason or "unknown error"
    try:
        is_json_payload = json.loads(reason) is not None
    except (json.JSONDecodeError, TypeError):
        is_json_payload = False
    reason = _truncate(reason, MAX_ERROR_REASON)
    retryable = is_retryable_error(error)
    title = "provider failure" if retryable else "error"
    content: list[RenderableType] = [Text(f"{title} · {code}", style=ERROR)]
    if not is_json_payload:
        content.append(_safe_text(f"reason: {reason}", style=BODY))
    else:
        content.extend(
            (
                Text("payload · json", style=DIM),
                Syntax(
                    reason,
                    "json",
                    theme=CODE_THEME,
                    word_wrap=True,
                    background_color="default",
                ),
            )
        )
    if retryable:
        content.append(Text("retry: ctrl+y", style=AFFORDANCE))
    return Panel(
        Group(*content),
        border_style=ERROR,
        style=CARD_BG,
        padding=(0, 1),
        expand=True,
    )


def is_retryable_error(error: ErrorInfo | None) -> bool:
    """Return whether an error can succeed when the provider is retried."""

    return error is not None and error.code in {
        "auth_error",
        "backend_error",
        "http_error",
        "stream_error",
        "timeout",
        "transport_error",
    }


def render_approval_card(
    tool_name: str,
    arguments: dict[str, object],
    *,
    label: str | None = None,
    key: str | None = None,
    shortcut: bool = True,
) -> Panel:
    """Render an inline permission-request card styled like Claude/Codex.

    `shortcut` marks the request the y/n keys answer: the rest have to be
    named, so they show their key instead of an affordance they do not have.
    """

    header = Text.assemble(
        ("allow ", DIM),
        (label or tool_name, COMMAND),
        ("?", DIM),
    )
    if key is not None:
        header.append(f"  [{key}]", style=DIM)
    arg_line = _arguments(arguments)
    body_parts: list[RenderableType] = [header]
    if arg_line:
        body_parts.append(Text(arg_line, style=DIM, overflow="ellipsis", no_wrap=True))
    if shortcut:
        affordance = "y approve · n deny"
    else:
        affordance = f"approve {key} · deny {key}" if key is not None else "approve · deny"
    body_parts.append(Text(affordance, style=AFFORDANCE))
    return Panel(
        Group(*body_parts),
        border_style=ACCENT,
        style=CARD_BG,
        padding=(0, 1),
        expand=True,
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
    lines = content.splitlines()
    truncated = len(lines) > MAX_TOOL_LINES
    visible = lines[:MAX_TOOL_LINES]
    if extra_lines:
        visible.extend(extra_lines)
    rendered = Text(style=BODY, overflow="ellipsis", no_wrap=True)
    for index, line in enumerate(visible):
        if index:
            rendered.append("\n")
        style = DIM if line.startswith("[image block]") else BODY
        rendered.append(_safe_text(line, style=style))
    if truncated:
        rendered.append(f"\n… +{len(lines) - MAX_TOOL_LINES} lines", style=AFFORDANCE)
    return rendered


def _split_tool_output(
    content: str,
) -> tuple[list[tuple[str, str]], str]:
    """Split standard command receipts without hiding generic tool output."""

    lines = content.splitlines()
    section_labels = {"stdout:", "stderr:", "result:"}
    if not any(line in section_labels for line in lines):
        generic = "\n".join(
            line for line in lines if not line.startswith("exit_code:")
        )
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


def _tool_body(event: StreamEvent) -> Text | None:
    content = _tool_content(event)
    sections, generic = _split_tool_output(content)
    result = event.tool_result
    extra_lines: list[str] = []
    if result is not None and result.content_blocks:
        char_sizes = [
            block["full_size_chars"]
            for block in result.content_blocks
            if (
                block.get("type") == "text"
                and block.get("truncated")
                and "full_size_chars" in block
            )
        ]
        byte_sizes = [
            block["full_size"]
            for block in result.content_blocks
            if (
                block.get("type") == "text"
                and block.get("truncated")
                and "full_size_chars" not in block
            )
        ]
        if char_sizes:
            extra_lines.append(
                f"[truncated; full_size_chars={max(char_sizes)} chars]"
            )
        if byte_sizes:
            extra_lines.append(f"[truncated; full_size={max(byte_sizes)} bytes]")
    rendered = Text(style=BODY, overflow="ellipsis", no_wrap=True)

    visible_sections = [(label, value) for label, value in sections if value.strip()]
    if generic.strip():
        rendered.append_text(_render_tool_output(generic, extra_lines))
        extra_lines = []
    if sections:
        for label, value in visible_sections:
            if rendered:
                rendered.append("\n\n")
            rendered.append(f"{label}:\n", style=DIM)
            rendered.append_text(_render_tool_output(value, extra_lines))
            extra_lines = []

    return rendered if rendered else None


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


def _tool_panel(
    call: ToolCall,
    body: Text | None,
    *,
    error: bool = False,
) -> Panel:
    header = _tool_header(call)
    if body is None:
        content: RenderableType = header
    elif isinstance(header, Text):
        content = Text.assemble(header, "\n", body)
        content.no_wrap = True
        content.overflow = "ellipsis"
    else:
        content = Group(header, body)
    return Panel(
        content,
        border_style=ERROR if error else CARD_BORDER,
        style=CARD_BG,
        padding=(0, 1),
        expand=True,
    )


def render_tool_progress(
    call: ToolCall,
    content: str,
    *,
    elapsed_seconds: float = 0.0,
    turns_used: int | None = None,
) -> RenderableType:
    """Render streamed tool output for the transcript."""

    agent_render = AgentCard.render_progress(
        call,
        content,
        elapsed_seconds=elapsed_seconds,
        turns_used=turns_used,
    )
    if agent_render is not None:
        return agent_render
    body = Text("running…", style=DIM) if not content else _render_tool_output(content)
    return _tool_panel(call, body)


def format_thought(duration: float | None = None) -> Text:
    parts = ["✱ thought"]
    if duration is not None:
        parts.append(f"{duration:.1f}s")
    return Text(" · ".join(parts), style=THOUGHT)


def render_thought(value: str, duration: float | None = None) -> Text:
    """Render a complete thinking block with its full trace."""

    if value == "redacted":
        return Text(
            f"✱ thought · redacted"
            f"{f' · {duration:.1f}s' if duration is not None else ''}",
            style=THOUGHT,
        )
    trace = Text(
        _strip_terminal_controls(value),
        style=THOUGHT,
    )
    rendered = Text.assemble(format_thought(duration), "\n", trace)
    return rendered


def render_thought_live(value: str) -> Text:
    """Render the in-progress thinking trace without its completion header."""

    if value == "redacted":
        return render_thought(value)
    return Text(
        _strip_terminal_controls(value),
        style=THOUGHT,
    )


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


_MARKDOWN = (
    MarkdownIt("commonmark")
    .enable(("table", "strikethrough"))
    .use(tasklists_plugin)
)
_MAX_MARKDOWN_SECONDS = 1.0
_MAX_MARKDOWN_TABLE_ROWS = 1_000


@dataclass(slots=True)
class _MarkdownNode:
    token: Any
    children: list[_MarkdownNode]


def _token_tree(tokens: Iterable[Any]) -> list[_MarkdownNode]:
    roots: list[_MarkdownNode] = []
    stack: list[list[_MarkdownNode]] = [roots]
    for token in tokens:
        if token.nesting == 1:
            node = _MarkdownNode(token, [])
            stack[-1].append(node)
            stack.append(node.children)
        elif token.nesting == -1:
            if len(stack) == 1:
                raise ValueError("unbalanced markdown token stream")
            stack.pop()
        else:
            stack[-1].append(_MarkdownNode(token, []))
    if len(stack) != 1:
        raise ValueError("unbalanced markdown token stream")
    return roots


def _inline_style(active: set[str], *, link: bool = False) -> str:
    styles: list[str] = [] if "strike" in active else [BODY]
    if link:
        styles.append("underline")
    styles.extend(sorted(active))
    return " ".join(styles)


def _render_inline_tokens(tokens: Iterable[Any]) -> Text:
    rendered = Text()
    active: set[str] = set()
    link_href: list[str] = []
    strike_seen = False

    def append(value: str, *, style: str | None = None) -> None:
        if value:
            rendered.append(
                value,
                style=style or _inline_style(active, link=bool(link_href)),
            )

    for token in tokens:
        token_type = token.type
        if token_type == "text":
            append(token.content)
        elif token_type == "code_inline":
            append(token.content, style=BODY)
        elif token_type == "softbreak":
            append(" ")
        elif token_type == "hardbreak":
            append("\n")
        elif token_type == "strong_open":
            active.add("bold")
        elif token_type == "strong_close":
            active.discard("bold")
        elif token_type == "em_open":
            active.add("italic")
        elif token_type == "em_close":
            active.discard("italic")
        elif token_type == "s_open":
            strike_seen = True
            active.add("strike")
        elif token_type == "s_close":
            active.discard("strike")
        elif token_type == "link_open":
            href = token.attrGet("href") or ""
            link_href.append(href)
        elif token_type == "link_close":
            href = link_href.pop() if link_href else ""
            if href:
                append(f" ({href})", style=CHROME)
        elif token_type == "image":
            src = token.attrGet("src") or ""
            append(f"![{token.content}]({src})")
        elif token_type == "html_inline":
            if token.content.startswith("<input class=\"task-list-item-checkbox\""):
                append("[x]" if "checked=\"checked\"" in token.content else "[ ]")
            else:
                append(token.content)
        else:
            append(token.content)
    if not strike_seen:
        rendered.style = BODY
    return rendered


def _inline_child(node: _MarkdownNode) -> Text:
    inline = next(
        (child for child in node.children if child.token.type == "inline"),
        None,
    )
    return _render_inline_tokens(inline.token.children or []) if inline else Text()


def render_line(value: str) -> Text:
    """Render one inline markdown value through markdown-it."""

    token = _MARKDOWN.parseInline(value)[0]
    return _render_inline_tokens(token.children or [])


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
class _Prefixed:
    renderable: RenderableType
    prefix: str
    style: str

    def __rich_console__(self, console: Console, options: Any) -> Iterable[Text]:
        inner_options = options.update_width(
            max(1, options.max_width - cell_len(self.prefix))
        )
        for line in console.render_lines(self.renderable, inner_options):
            text = Text()
            for segment in line:
                text.append(segment.text, style=segment.style)
            yield Text.assemble((self.prefix, self.style), text)


def _with_blank_lines(rendered: list[RenderableType]) -> list[RenderableType]:
    result: list[RenderableType] = []
    for index, item in enumerate(rendered):
        if index:
            result.append(Text(""))
        result.append(item)
    return result


def _wrapped_list_item(
    console: Console,
    content: Text,
    prefix: str,
    width: int,
) -> Text:
    lines = content.wrap(console, max(1, width - cell_len(prefix)), overflow="fold")
    result = Text()
    for index, line in enumerate(lines):
        if index:
            result.append("\n" + " " * cell_len(prefix))
        else:
            result.append(prefix)
        result.append_text(line)
    return result


def _render_list(
    node: _MarkdownNode,
    console: Console,
    width: int,
    depth: int = 0,
    deadline: float | None = None,
) -> Text:
    ordered = node.token.type == "ordered_list_open"
    start = int(node.token.attrGet("start") or 1)
    items = [child for child in node.children if child.token.type == "list_item_open"]
    loose = any(
        child.token.type == "paragraph_open" and not child.token.hidden
        for item in items
        for child in item.children
    )
    rendered = Text()
    indent = "  " * depth
    for item_index, item in enumerate(items):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("markdown painting exceeded its time budget")
        paragraphs = [
            child for child in item.children if child.token.type == "paragraph_open"
        ]
        marker = f"{start + item_index}. " if ordered else "- "
        first = _inline_child(paragraphs[0]) if paragraphs else Text()
        first_prefix = indent + marker
        if rendered:
            rendered.append("\n\n" if loose else "\n")
        rendered.append_text(
            _wrapped_list_item(console, first, first_prefix, width)
        )
        for paragraph in paragraphs[1:]:
            rendered.append("\n")
            rendered.append_text(
                _wrapped_list_item(
                    console,
                    _inline_child(paragraph),
                    indent + "  ",
                    width,
                )
            )
        for child in item.children:
            if child.token.type in {"bullet_list_open", "ordered_list_open"}:
                rendered.append("\n")
                rendered.append_text(
                    _render_list(child, console, width, depth + 1, deadline)
                )
    return rendered


def _table_rows(
    node: _MarkdownNode,
    deadline: float | None = None,
) -> tuple[list[_MarkdownNode], list[list[_MarkdownNode]]]:
    headers: list[_MarkdownNode] = []
    body: list[list[_MarkdownNode]] = []
    for section in node.children:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("markdown painting exceeded its time budget")
        if section.token.type == "thead_open":
            rows = [child for child in section.children if child.token.type == "tr_open"]
            if rows:
                headers = [
                    cell
                    for cell in rows[0].children
                    if cell.token.type == "th_open"
                ]
        elif section.token.type == "tbody_open":
            for row in section.children:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("markdown painting exceeded its time budget")
                if row.token.type == "tr_open":
                    if len(body) >= _MAX_MARKDOWN_TABLE_ROWS:
                        raise TimeoutError("markdown table exceeded its time budget")
                    body.append(
                        [
                            cell
                            for cell in row.children
                            if cell.token.type == "td_open"
                        ]
                    )
    return headers, body


def _render_table(node: _MarkdownNode, deadline: float | None = None) -> Table:
    headers, body = _table_rows(node, deadline)
    if len(body) > _MAX_MARKDOWN_TABLE_ROWS:
        raise TimeoutError("markdown table exceeded its time budget")
    table = Table(
        box=box.SQUARE,
        border_style=CHROME,
        header_style=f"bold {BODY}",
        style=CARD_BG,
        pad_edge=True,
        show_lines=False,
    )
    for cell in headers:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("markdown painting exceeded its time budget")
        align = (cell.token.attrGet("style") or "").split(":")[-1]
        table.add_column(
            header=_inline_child(cell),
            justify=align if align in {"left", "center", "right"} else "left"
        )
    for row in body:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("markdown painting exceeded its time budget")
        cells = [_inline_child(cell) for cell in row]
        cells.extend(Text() for _ in range(len(headers) - len(cells)))
        table.add_row(*cells)
    return table


def _render_blocks(
    nodes: list[_MarkdownNode],
    console: Console,
    width: int,
    deadline: float | None = None,
) -> list[RenderableType]:
    rendered: list[RenderableType] = []
    for node in nodes:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("markdown painting exceeded its time budget")
        token_type = node.token.type
        if token_type == "paragraph_open":
            rendered.append(_inline_child(node))
        elif token_type == "heading_open":
            heading = _inline_child(node)
            level = int(node.token.tag.removeprefix("h") or 1)
            heading.stylize(
                {
                    1: f"bold underline {BODY}",
                    2: f"bold {BODY}",
                    3: f"bold {BODY}",
                    4: f"underline {BODY}",
                    5: BODY,
                    6: f"italic {BODY}",
                }.get(level, BODY)
            )
            rendered.append(heading)
        elif token_type in {"bullet_list_open", "ordered_list_open"}:
            rendered.append(_render_list(node, console, width, deadline=deadline))
        elif token_type == "blockquote_open":
            inner = _with_blank_lines(
                _render_blocks(node.children, console, width, deadline)
            )
            rendered.append(
                _Prefixed(Group(*inner), "│ " * 1, f"dim {DIM}")
            )
        elif token_type == "fence":
            language = (node.token.info.strip() or "text").split()[0]
            rendered.append(render_code(node.token.content, language))
        elif token_type in {"code_block", "html_block"}:
            rendered.append(
                Text(_strip_terminal_controls(node.token.content), style=BODY)
            )
        elif token_type == "hr":
            rendered.append(Text("─" * max(1, width), style=DIM, overflow="crop"))
        elif token_type == "table_open":
            rendered.append(_render_table(node, deadline))
        else:
            raise ValueError(f"unhandled markdown block: {token_type}")
    return rendered


@dataclass(slots=True)
class MarkdownDocument:
    """Parsed markdown that paints at the transcript's current width."""

    source: str
    nodes: list[_MarkdownNode] | None

    @property
    def plain(self) -> str:
        return self.source

    def __rich_console__(self, console: Console, options: Any) -> Iterable[RenderableType]:
        if self.nodes is None:
            yield Text(_strip_terminal_controls(self.source), style=BODY)
            return
        width = max(1, options.max_width)
        started = time.monotonic()
        try:
            blocks = _render_blocks(
                self.nodes,
                console,
                width,
                started + _MAX_MARKDOWN_SECONDS,
            )
        except Exception:
            yield Text(_strip_terminal_controls(self.source), style=BODY)
            return
        for rendered in _with_blank_lines(blocks):
            yield rendered


def render_markdown(value: str) -> MarkdownDocument:
    """Parse one completed assistant message into a width-independent document."""

    started = time.monotonic()
    try:
        tokens = _MARKDOWN.parse(value)
        if time.monotonic() - started > _MAX_MARKDOWN_SECONDS:
            raise TimeoutError("markdown rendering exceeded its time budget")
        return MarkdownDocument(value, _token_tree(tokens))
    except Exception:
        return MarkdownDocument(value, None)


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
    if event.type is StreamEventType.RETRY:
        text = event.data.get("text")
        return Text(text if type(text) is str else "retrying", style=DIM)
    if event.type is StreamEventType.AGENT_NOTIFICATION:
        description = event.data.get("description")
        status = event.data.get("status")
        text = event.data.get("text")
        path = event.data.get("child_session_path")
        if not all(type(value) is str for value in (description, status, text, path)):
            return Text("background agent notification unavailable", style=ERROR)
        style = ERROR if status in {"error", "canceled"} else RECEIPT
        return Text(
            f"background · {description} · {status} · {text} · {path}",
            style=style,
            overflow="ellipsis",
        )
    if event.type is StreamEventType.TOOL_EXECUTION_START and event.tool_call:
        agent_render = AgentCard.render_start(event)
        if agent_render is not None:
            return agent_render
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
        agent_render = AgentCard.render_receipt(event)
        if agent_render is not None:
            return agent_render
        if tool_render_mode(event) == "receipt":
            return _tool_receipt(event)
        return _tool_card(event)
    if event.type is StreamEventType.ERROR:
        return render_error_card(event)
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
            return render_thought(event.content.text, _duration(event.data))
        if isinstance(event.content, RedactedThinkingContent):
            return render_thought("redacted", _duration(event.data))
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
    background_count: int = 0,
    undo_available: bool = False,
    transcript_navigation: bool = False,
    transcript_search: str | None = None,
    transcript_match: tuple[int, int] | None = None,
    transcript_position: str | None = None,
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
    if background_count > 0:
        left_segments.append(f"bg {background_count}")
    if transcript_position:
        left_segments.append(transcript_position)
    if transcript_search is not None:
        current, total = transcript_match or (0, 0)
        left_segments.insert(
            0,
            f'find "{transcript_search}" {current}/{total}',
        )
    left = "  ".join(left_segments)
    right_segments = ["/status", "ctrl+c interrupt", "ctrl+d quit"]
    if transcript_navigation:
        right_segments.extend(("ctrl+f find", "ctrl+up/down users"))
    if transcript_search is not None:
        right_segments.extend(("enter/n next", "N prev", "esc close"))
    if undo_available:
        right_segments.append("ctrl+u undo")
    if session_id:
        right_segments.append(session_id[:8])
    if width is None:
        value = f"{left}  {' · '.join(right_segments)}"
    else:
        value = left
        if transcript_search is not None:
            search_current, search_total = transcript_match or (0, 0)
            search_prefix = 'find "'
            search_suffix = f'" {search_current}/{search_total}'
            minimum_search_width = cell_len(f'{search_prefix}…{search_suffix}')

            def search_segment(max_width: int) -> str:
                if max_width < minimum_search_width:
                    return ""
                full = f'{search_prefix}{transcript_search}{search_suffix}'
                if cell_len(full) <= max_width:
                    return full
                available = max_width - cell_len(search_prefix) - cell_len(search_suffix)
                query = Text(
                    transcript_search,
                    no_wrap=True,
                    overflow="ellipsis",
                )
                query.truncate(max(1, available), overflow="ellipsis")
                return f"{search_prefix}{query.plain}{search_suffix}"

            navigation_candidates = (
                (state_segment, transcript_position),
                (state_text, transcript_position),
                ("", transcript_position),
                (state_segment, ""),
                (state_text, ""),
                ("", ""),
            )
            for state, position in navigation_candidates:
                fixed = cell_len(state) + cell_len(position)
                gaps = 2 * (bool(state) + bool(position))
                if fixed + gaps >= width:
                    continue
                search = search_segment(width - fixed - gaps)
                if not search:
                    continue
                parts = [search, state, position]
                candidate = "  ".join(part for part in parts if part)
                if cell_len(candidate) <= width:
                    value = candidate
                    break
        candidates = (value,)
        if transcript_search is None:
            candidates = (left, state_segment)
            if vim_state and background_count > 0:
                candidates = (left, f"{vim_state}  {state_segment}", state_segment)
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
