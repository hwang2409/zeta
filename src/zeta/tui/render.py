"""Pure Rich renderers for provider-neutral zeta events."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin
from rich import box
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..agent.receipt import terminal_state
from ..protocol.types import (
    ErrorInfo,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolUseContent,
)
from ..tools.exec import MacroDisplay
from . import theme
from .cards.agent import AgentCard
from .cards.base import (
    MAX_TOOL_LINES,
)
from .cards.base import (
    arguments as _arguments,
)
from .cards.base import (
    command as _command,
)
from .cards.base import (
    render_tool_output as _render_tool_output,
)
from .cards.base import (
    safe_text as _safe_text,
)
from .cards.base import (
    strip_terminal_controls as _strip_terminal_controls,
)
from .cards.base import (
    tool_card as _tool_card,
)
from .cards.base import (
    tool_content as _tool_content,
)
from .cards.base import (
    tool_panel as _tool_panel,
)
from .cards.base import (
    truncate as _truncate,
)
from .cards.shared import (
    BoundedToolOutput as _BoundedToolOutput,
)
from .cards.shared import (
    infer_language as _infer_language,
)
from .cards.shared import (
    scan_tool_output as _scan_tool_output,
)
from .cards.tool import TOOL_CARD_REGISTRY

MAX_RESULT = 180
MAX_ERROR_REASON = 400
MAX_AGENT_NOTIFICATION_NAME = 64
MAX_AGENT_NOTIFICATION_REASON = 80
MAX_AGENT_NOTIFICATION_LINE = 78
SPINNER_FRAMES = ("·", "•", "●", "•")
RECEIPT_TOOLS = frozenset(
    {"read", "glob", "grep", "search", "find", "list", "websearch"}
)
SUMMARY_TOOLS = frozenset({"glob", "grep", "search", "find", "websearch"})
ToolRenderMode = Literal["card", "receipt"]


def infer_language(path: str) -> str:
    return _infer_language(path)

render_agent_expanded = AgentCard.render_expanded
render_agent_progress = AgentCard.render_progress
render_agent_receipt = AgentCard.render_receipt


def tool_render_mode(
    event: StreamEvent,
    *,
    scan: _BoundedToolOutput | None = None,
) -> ToolRenderMode:
    """Choose the one display mode for completed tool results."""

    if event.data.get("macro"):
        return "receipt"
    call = event.tool_call
    result = event.tool_result
    if call is None or result is None or call.name.lower() not in RECEIPT_TOOLS:
        return "card"
    if result.is_error or any(
        block.get("type") != "text" for block in result.content_blocks or []
    ):
        return "card"
    content = _tool_content(event)
    scan = _scan_tool_output(content) if scan is None else scan
    line_count = scan.total_lines if scan.total_lines is not None else MAX_TOOL_LINES + 1
    if any(cell_len(_strip_terminal_controls(line)) > MAX_RESULT for line in scan.lines):
        return "card"
    if (
        line_count < 3
        or call.name.lower() in SUMMARY_TOOLS
    ):
        return "receipt"
    return "card"


def _receipt_arguments(
    call: ToolCall,
    content: str,
    *,
    scan: _BoundedToolOutput | None = None,
) -> str:
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
            else str(
                sum(
                    bool(line.strip())
                    for line in (scan or _scan_tool_output(content)).lines
                )
            )
        )
        location = arguments.get("path", arguments.get("cwd", "."))
        return f'"{pattern}" in {location} · {count} matches'
    return _arguments(arguments)


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
    content: list[RenderableType] = [Text(f"{title} · {code}", style=theme.ERROR)]
    if not is_json_payload:
        content.append(_safe_text(f"reason: {reason}", style=theme.BODY, wrap=True))
    else:
        content.extend(
            (
                Text("payload · json", style=theme.DIM),
                Syntax(
                    reason,
                    "json",
                    theme=theme.CODE_THEME,
                    word_wrap=True,
                    background_color="default",
                ),
            )
        )
    if retryable:
        content.append(Text("retry: ctrl+y", style=theme.AFFORDANCE))
    return Panel(
        Group(*content),
        border_style=theme.ERROR,
        style=theme.CARD_BG,
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
    trusted_display: MacroDisplay | None = None,
) -> Panel:
    """Render an inline permission-request card styled like Claude/Codex.

    `shortcut` marks the request the y/n keys answer: the rest have to be
    named, so they show their key instead of an affordance they do not have.

    Display strings (`trusted_display`) are harness-side only; the arguments
    dict is provider-visible and can never override what the card shows.
    """

    header = Text.assemble(
        ("allow ", theme.DIM),
        (label or tool_name, theme.COMMAND),
        ("?", theme.DIM),
    )
    if key is not None:
        header.append(f"  [{key}]", style=theme.DIM)
    body_parts: list[RenderableType] = [header]
    if trusted_display is not None:
        command = trusted_display.command
        argv: tuple[str, ...] = trusted_display.argv
    else:
        raw_command = _command(arguments)
        command = str(raw_command) if raw_command is not None else None
        argv = ()
    if command is not None:
        body_parts.append(Text(f"command={command}", style=theme.DIM, overflow="fold"))
        if argv:
            body_parts.append(Text("argv:", style=theme.DIM))
            for index, value in enumerate(argv, 1):
                body_parts.append(
                    Text(f"  [{index}] {value}", style=theme.DIM, overflow="fold")
                )
    else:
        arg_line = _arguments(arguments)
        if arg_line:
            body_parts.append(Text(arg_line, style=theme.DIM, overflow="ellipsis", no_wrap=True))
    if shortcut:
        affordance = "y approve · n deny"
    else:
        affordance = f"approve {key} · deny {key}" if key is not None else "approve · deny"
    body_parts.append(Text(affordance, style=theme.AFFORDANCE))
    return Panel(
        Group(*body_parts),
        border_style=theme.ACCENT,
        style=theme.CARD_BG,
        padding=(0, 1),
        expand=True,
    )


def _tool_receipt(
    event: StreamEvent,
    scan: _BoundedToolOutput | None = None,
) -> Text:
    call = event.tool_call
    assert call is not None
    result = event.tool_result
    assert result is not None
    macro = event.data.get("macro")
    if isinstance(macro, str) and macro:
        structured = result.structured_content or {}
        if result.is_canceled:
            status = "canceled"
        elif result.content.startswith("tool execution denied"):
            status = "denied"
        elif structured.get("timed_out") is True:
            status = "timeout"
        elif structured.get("status") == "running":
            status = "running"
        else:
            exit_code = structured.get("exit_code")
            status = f"exit {exit_code}" if exit_code is not None else "failed"
        log_path = structured.get("log_path")
        suffix = f"/{macro} · {status}"
        if isinstance(log_path, str) and log_path:
            suffix += f" · log {log_path}"
        return Text(
            f"⏺ {suffix}",
            style=theme.ERROR if result.is_error else theme.RECEIPT,
            overflow="ellipsis",
            no_wrap=True,
        )
    prefix = "⏺ "
    if result.is_error:
        prefix += "failed · "
    suffix = _receipt_arguments(call, _tool_content(event), scan=scan)
    return Text(
        f"{prefix}{call.name}{f' {suffix}' if suffix else ''}",
        style=theme.RECEIPT,
        overflow="ellipsis",
        no_wrap=True,
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
    body = Text("running…", style=theme.DIM) if not content else _render_tool_output(content)
    return _tool_panel(call, body)


def format_thought(duration: float | None = None) -> Text:
    parts = ["✱ thought"]
    if duration is not None:
        parts.append(f"{duration:.1f}s")
    return Text(" · ".join(parts), style=theme.THOUGHT)


def render_thought(value: str, duration: float | None = None) -> Text:
    """Render a complete thinking block with its full trace."""

    if value == "redacted":
        return Text(
            f"✱ thought · redacted"
            f"{f' · {duration:.1f}s' if duration is not None else ''}",
            style=theme.THOUGHT,
        )
    trace = Text(
        _strip_terminal_controls(value),
        style=theme.THOUGHT,
    )
    rendered = Text.assemble(format_thought(duration), "\n", trace)
    return rendered


def render_thought_live(value: str) -> Text:
    """Render the in-progress thinking trace without its completion header."""

    if value == "redacted":
        return render_thought(value)
    return Text(
        _strip_terminal_controls(value),
        style=theme.THOUGHT,
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
    styles: list[str] = [] if "strike" in active else [theme.BODY]
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
            append(token.content, style=theme.BODY)
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
                append(f" ({href})", style=theme.CHROME)
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
        rendered.style = theme.BODY
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
        theme=theme.CODE_THEME,
        word_wrap=True,
        background_color=theme.CODE_BG,
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
        border_style=theme.CHROME,
        header_style=f"bold {theme.BODY}",
        style=theme.CARD_BG,
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
                    1: f"bold underline {theme.BODY}",
                    2: f"bold {theme.BODY}",
                    3: f"bold {theme.BODY}",
                    4: f"underline {theme.BODY}",
                    5: theme.BODY,
                    6: f"italic {theme.BODY}",
                }.get(level, theme.BODY)
            )
            rendered.append(heading)
        elif token_type in {"bullet_list_open", "ordered_list_open"}:
            rendered.append(_render_list(node, console, width, deadline=deadline))
        elif token_type == "blockquote_open":
            inner = _with_blank_lines(
                _render_blocks(node.children, console, width, deadline)
            )
            rendered.append(
                _Prefixed(Group(*inner), "│ " * 1, f"dim {theme.DIM}")
            )
        elif token_type == "fence":
            language = (node.token.info.strip() or "text").split()[0]
            rendered.append(render_code(node.token.content, language))
        elif token_type in {"code_block", "html_block"}:
            rendered.append(
                Text(_strip_terminal_controls(node.token.content), style=theme.BODY)
            )
        elif token_type == "hr":
            rendered.append(Text("─" * max(1, width), style=theme.DIM, overflow="crop"))
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
            yield Text(_strip_terminal_controls(self.source), style=theme.BODY)
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
        except Exception:  # noqa: BLE001 - fall back to plain text
            yield Text(_strip_terminal_controls(self.source), style=theme.BODY)
            return
        yield from _with_blank_lines(blocks)


def render_markdown(value: str) -> MarkdownDocument:
    """Parse one completed assistant message into a width-independent document."""

    started = time.monotonic()
    try:
        tokens = _MARKDOWN.parse(value)
        if time.monotonic() - started > _MAX_MARKDOWN_SECONDS:
            raise TimeoutError("markdown rendering exceeded its time budget")
        return MarkdownDocument(value, _token_tree(tokens))
    except Exception:  # noqa: BLE001 - return an unparsed document
        return MarkdownDocument(value, None)


def _compact_notification_text(value: str, limit: int) -> str:
    """Return one bounded, terminal-safe notification fragment."""

    single_line = " ".join(_strip_terminal_controls(value).split())
    return _truncate(single_line, limit)


def _notification_stats(data: dict[str, Any]) -> list[str]:
    stats = data.get("stats")
    if type(stats) is not dict:
        return []
    parts: list[str] = []
    elapsed = stats.get("elapsed")
    if type(elapsed) in {int, float} and elapsed >= 0:
        seconds = float(elapsed)
        if seconds >= 60:
            minutes, remainder = divmod(int(seconds), 60)
            parts.append(f"{minutes}m{remainder:02d}s")
        else:
            parts.append(f"{seconds:.1f}s")
    turns = stats.get("turns_used")
    if type(turns) is int and turns >= 0:
        parts.append(f"{turns} turns")
    return parts


def render_agent_notification(event: StreamEvent) -> Text:
    """Render a background completion as one compact receipt line."""

    description = event.data.get("description")
    status = event.data.get("status")
    if type(description) is not str or type(status) is not str or not description:
        return Text("background agent notification unavailable", style=theme.ERROR)
    try:
        state = terminal_state(status=status)
    except ValueError:
        return Text("background agent notification unavailable", style=theme.ERROR)

    stats_parts = _notification_stats(event.data)
    status_parts = [state, *stats_parts]
    fixed_suffix = " · ".join(status_parts)
    name_limit = min(
        MAX_AGENT_NOTIFICATION_NAME,
        MAX_AGENT_NOTIFICATION_LINE
        - cell_len("⏺ ")
        - cell_len(" · ")
        - cell_len(fixed_suffix),
    )
    parts = [
        f"⏺ {_compact_notification_text(description, name_limit)}",
        *status_parts,
    ]
    if state in {"failed", "canceled"}:
        text = event.data.get("text")
        if type(text) is str and text:
            reason_limit = min(
                MAX_AGENT_NOTIFICATION_REASON,
                MAX_AGENT_NOTIFICATION_LINE
                - cell_len(" · ".join(parts))
                - cell_len(" · reason: "),
            )
            if reason_limit > 0:
                reason = _compact_notification_text(text, reason_limit)
                parts.append(f"reason: {reason}")
    rendered = Text(
        " · ".join(parts),
        style=theme.ERROR if state != "completed" else theme.RECEIPT,
        overflow="ellipsis",
        no_wrap=True,
    )
    rendered.truncate(MAX_AGENT_NOTIFICATION_LINE, overflow="ellipsis")
    return rendered


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
                style=theme.DIM,
            )
        return Text("response truncated (stream ended early)", style=theme.DIM)
    if event.type is StreamEventType.RETRY:
        text = event.data.get("text")
        return Text(text if type(text) is str else "retrying", style=theme.DIM)
    if event.type is StreamEventType.AGENT_NOTIFICATION:
        return render_agent_notification(event)
    if event.type is StreamEventType.TOOL_EXECUTION_START and event.tool_call:
        agent_render = AgentCard.render_start(event)
        if agent_render is not None:
            return agent_render
        macro = event.data.get("macro")
        if isinstance(macro, str) and macro:
            return Text(
                f"⏺ /{macro} · running",
                style=theme.RECEIPT,
            )
        card_renderer = TOOL_CARD_REGISTRY.get(event.tool_call.name.strip().lower())
        if card_renderer is not None:
            return card_renderer(event, True)
        if event.tool_call.name.lower() in RECEIPT_TOOLS:
            suffix = _receipt_arguments(event.tool_call, "")
            return Text(
                f"⏺ {event.tool_call.name}{f' {suffix}' if suffix else ''} · running",
                style=theme.RECEIPT,
            )
        return _tool_card(event, running=True)
    if event.type is StreamEventType.TOOL_EXECUTION_UPDATE and event.delta is not None:
        stream = event.data.get("stream")
        label = f"[{stream}] " if stream in {"stdout", "stderr"} else ""
        return _safe_text(f"  ↳ {label}{event.delta}", style=theme.DIM)
    if event.type is StreamEventType.TOOL_EXECUTION_END and event.tool_result:
        agent_render = AgentCard.render_receipt(event)
        if agent_render is not None:
            return agent_render
        if event.tool_call is not None:
            card_renderer = TOOL_CARD_REGISTRY.get(
                event.tool_call.name.strip().lower()
            )
            if card_renderer is not None:
                return card_renderer(event, False)
        content = _tool_content(event)
        scan = _scan_tool_output(content)
        if tool_render_mode(event, scan=scan) == "receipt":
            return _tool_receipt(event, scan)
        return _tool_card(event, scan=scan)
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
    plan_state: str | None = None,
    background_count: int = 0,
    undo_available: bool = False,
    transcript_navigation: bool = False,
    transcript_search: str | None = None,
    transcript_match: tuple[int, int] | None = None,
    transcript_position: str | None = None,
    copy_notice: str | None = None,
    approval_mode: str | None = None,
    cwd: str | Path | None = None,
) -> Text:
    """Format the compact status bar shown below the composer."""

    show_spinner = streaming if spinner_active is None else spinner_active
    usage = usage or {}
    del partial, retained_tail
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
    model_segment: str | None = None
    approval_segment = approval_mode
    cwd_segment: str | None = None
    if approval_mode is not None or cwd is not None:
        model_segment = f"{provider}/{model}"
        if cwd is not None:
            path = Path(cwd).expanduser()
            try:
                path = path.relative_to(Path.home())
                cwd_text = f"~/{path}" if str(path) != "." else "~"
            except ValueError:
                cwd_text = str(path)
            if len(cwd_text) > 20:
                cwd_text = f"{'~/' if cwd_text.startswith('~/') else ''}…/{path.name}"
            cwd_segment = cwd_text

    def build_left(
        *,
        include_model: bool = True,
        include_approval: bool = True,
        include_cwd: bool = True,
        include_vim: bool = True,
    ) -> str:
        segments: list[str] = []
        if plan_state:
            segments.append(plan_state)
        if include_vim and vim_state:
            segments.append(vim_state)
        if include_model and model_segment:
            segments.append(model_segment)
        if include_approval and approval_segment:
            segments.append(approval_segment)
        if include_cwd and cwd_segment:
            segments.append(cwd_segment)
        segments.append(state_segment)
        if background_count > 0:
            segments.append(f"bg {background_count}")
        if transcript_position:
            segments.append(transcript_position)
        if copy_notice:
            segments.append(copy_notice)
        return "  ".join(segments)

    left = build_left()
    if transcript_search is not None:
        current, total = transcript_match or (0, 0)
        left = f'find "{transcript_search}" {current}/{total}  {left}'
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

            position_segment = transcript_position or ""
            navigation_candidates = (
                (state_segment, position_segment),
                (state_text, position_segment),
                ("", position_segment),
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
        if transcript_search is None:
            candidates = (
                (True, True, True, True),
                (True, True, False, True),
                (True, False, False, True),
                (False, False, False, True),
                (False, False, False, False),
            )
            context_present = any(
                (model_segment, approval_segment, cwd_segment)
            )
            selected: str | None = None
            for candidate_index, (
                include_model,
                include_approval,
                include_cwd,
                include_vim,
            ) in enumerate(candidates):
                starts = (
                    range(len(right_segments) + 1)
                    if candidate_index == 0
                    else (len(right_segments),)
                )
                candidate_left = build_left(
                    include_model=include_model,
                    include_approval=include_approval,
                    include_cwd=include_cwd,
                    include_vim=include_vim,
                )
                if cell_len(candidate_left) > width:
                    continue
                for start in starts:
                    right = " · ".join(right_segments[start:])
                    gap = width - cell_len(candidate_left) - cell_len(right)
                    if gap < 2 and right:
                        continue
                    if not right and include_vim and not context_present:
                        continue
                    selected = (
                        f"{candidate_left}{' ' * gap}{right}"
                        if right
                        else candidate_left
                    )
                    break
                if selected is not None:
                    value = selected
                    break
            if selected is None:
                value = build_left(
                    include_model=False,
                    include_approval=False,
                    include_cwd=False,
                    include_vim=False,
                )
        if cell_len(value) > width:
            fitted = Text(value, no_wrap=True, overflow="ellipsis")
            fitted.truncate(width, overflow="ellipsis")
            value = fitted.plain.rstrip(" ·")
    rendered = Text(value, style=theme.CHROME)
    offset = 0
    if plan_state and value.startswith(plan_state):
        rendered.stylize(theme.PLAN_STATE, 0, len(plan_state))
        offset = len(plan_state) + 2
    if vim_state and value[offset:].startswith(vim_state):
        rendered.stylize(theme.VIM_STATE, offset, offset + len(vim_state))
    return rendered
