"""Shared bounded helpers for TUI cards."""

from __future__ import annotations

from pathlib import PurePath
from typing import NamedTuple

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
