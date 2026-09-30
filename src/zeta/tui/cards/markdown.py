"""Markdown renderables used by the TUI transcript."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin
from rich import box
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .. import theme
from .base import strip_terminal_controls as _strip_terminal_controls

_MARKDOWN = (
    MarkdownIt("commonmark").enable(("table", "strikethrough")).use(tasklists_plugin)
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
            if token.content.startswith('<input class="task-list-item-checkbox"'):
                append("[x]" if 'checked="checked"' in token.content else "[ ]")
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
        rendered.append_text(_wrapped_list_item(console, first, first_prefix, width))
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
            rows = [
                child for child in section.children if child.token.type == "tr_open"
            ]
            if rows:
                headers = [
                    cell for cell in rows[0].children if cell.token.type == "th_open"
                ]
        elif section.token.type == "tbody_open":
            for row in section.children:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("markdown painting exceeded its time budget")
                if row.token.type == "tr_open":
                    if len(body) >= _MAX_MARKDOWN_TABLE_ROWS:
                        raise TimeoutError("markdown table exceeded its time budget")
                    body.append(
                        [cell for cell in row.children if cell.token.type == "td_open"]
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
            justify=align if align in {"left", "center", "right"} else "left",
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
            rendered.append(_Prefixed(Group(*inner), "│ " * 1, f"dim {theme.DIM}"))
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
    """Markdown source that parses and paints when its unit becomes visible."""

    source: str
    nodes: list[_MarkdownNode] | None = None
    _parsed: bool = False

    @property
    def plain(self) -> str:
        return self.source

    def _parse(self) -> None:
        if self._parsed:
            return
        self._parsed = True
        started = time.monotonic()
        try:
            tokens = _MARKDOWN.parse(self.source)
            if time.monotonic() - started > _MAX_MARKDOWN_SECONDS:
                raise TimeoutError("markdown rendering exceeded its time budget")
            self.nodes = _token_tree(tokens)
        except Exception:  # noqa: BLE001 - paint unparsed source as plain text
            self.nodes = None

    def __rich_console__(self, console: Console, options: Any) -> Iterable[RenderableType]:
        self._parse()
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
    """Return a width-independent document that parses on its first paint."""

    return MarkdownDocument(value)


