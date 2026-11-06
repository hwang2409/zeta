"""Shared bounded helpers for TUI cards."""

from __future__ import annotations

from pathlib import PurePath
from typing import NamedTuple

from prompt_toolkit.utils import get_cwidth

MAX_RESULT = 180
MAX_CARD_LINES = 15
MAX_CARD_COLUMNS = 240
MAX_TOOL_SCAN_LINES = MAX_CARD_LINES * 4
MAX_TOOL_SCAN_BYTES = 64 * 1024

LANGUAGE_BY_EXTENSION = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".ini": "ini",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "bash",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "bash",
}


class BoundedToolOutput(NamedTuple):
    lines: tuple[str, ...]
    total_lines: int | None
    truncated: bool


def scan_tool_output(content: str) -> BoundedToolOutput:
    """Read a bounded prefix of tool output without building line lists."""

    lines: list[str] = []
    start = 0
    scan_end = min(len(content), MAX_TOOL_SCAN_BYTES)
    truncated = False
    while start < len(content) and len(lines) < MAX_TOOL_SCAN_LINES:
        newline = content.find("\n", start, scan_end)
        if newline < 0:
            end = scan_end
            line = content[start:end]
            lines.append(line.removesuffix("\r")[: MAX_RESULT + 1])
            if end < len(content):
                truncated = True
            start = len(content)
            break
        lines.append(content[start:newline].removesuffix("\r")[: MAX_RESULT + 1])
        start = newline + 1
    if start < len(content):
        truncated = True
    return BoundedToolOutput(
        tuple(lines),
        None if truncated else len(lines),
        truncated,
    )


def infer_language(path: str) -> str:
    """Return the syntax lexer for a path, with text as the safe fallback."""

    return LANGUAGE_BY_EXTENSION.get(PurePath(path).suffix.lower(), "text")


def _cell_prefix(text: str, width: int) -> int:
    cells = 0
    for index, character in enumerate(text):
        step = get_cwidth(character)
        if cells + step > width:
            return max(1, index)
        cells += step
    return len(text)


def wrap_by_cells(text: str, width: int) -> list[str]:
    """Wrap text by terminal cells, splitting words that exceed the width."""

    width = max(1, width)
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split() or [""]:
            piece = word if not current else f"{current} {word}"
            if get_cwidth(piece) <= width:
                current = piece
                continue
            if current:
                lines.append(current)
            while get_cwidth(word) > width:
                split = _cell_prefix(word, width)
                lines.append(word[:split])
                word = word[split:]
            current = word
        lines.append(current)
    return lines or [""]
