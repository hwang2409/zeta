from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

PATH_COMPLETION_LIMIT = 50
PATH_SCAN_LIMIT = 1000
_PATH_TOKEN_RE = re.compile(
    r'(?<!\S)@(?:"(?P<quoted>[^"\n]*)|(?P<unquoted>[^\s]*))$'
)


@dataclass(frozen=True, slots=True)
class PathCompletion:
    text: str
    start_position: int
    display: str
    kind: str


def complete_paths(
    text_before_cursor: str,
    base_dir: str | Path,
    *,
    cancel_event: threading.Event | None = None,
) -> tuple[list[PathCompletion], bool]:
    """Complete one filesystem directory level for the active ``@`` token."""

    match = _PATH_TOKEN_RE.search(text_before_cursor)
    if match is None:
        return [], False
    quoted = match.group("quoted") is not None
    path_text = match.group("quoted") or match.group("unquoted") or ""
    base_path = Path(base_dir)
    directory, display_prefix, basename = _completion_context(path_text, base_path)
    if _escapes_base_through_symlink(directory, base_path):
        return [], False
    matches, scan_truncated = _matching_entries(
        directory, basename, basename.startswith("."), cancel_event
    )
    matches.sort(key=lambda item: (not item[0], item[1]))
    truncated = scan_truncated or len(matches) > PATH_COMPLETION_LIMIT
    result = []
    for is_directory, name in matches[:PATH_COMPLETION_LIMIT]:
        candidate = _candidate(display_prefix, name, is_directory)
        completion_text = _quoted_candidate(candidate, quoted, is_directory)
        result.append(
            PathCompletion(
                text=completion_text,
                start_position=-len(path_text),
                display=f"@{completion_text}",
                kind="directory" if is_directory else "file",
            )
        )
    return result, truncated


def _matching_entries(
    directory: Path,
    basename: str,
    show_hidden: bool,
    cancel_event: threading.Event | None,
) -> tuple[list[tuple[bool, str]], bool]:
    try:
        entries = os.scandir(directory)
    except OSError:
        return [], False
    scan_truncated = False
    matches: list[tuple[bool, str]] = []
    with entries:
        for index, entry in enumerate(entries):
            if cancel_event is not None and cancel_event.is_set():
                break
            if index >= PATH_SCAN_LIMIT:
                scan_truncated = True
                break
            name = entry.name
            if not name.startswith(basename) or (
                name.startswith(".") and not show_hidden
            ):
                continue
            try:
                is_directory = entry.is_dir()
            except OSError:
                continue
            matches.append((is_directory, name))
    return matches, scan_truncated


def _escapes_base_through_symlink(directory: Path, base_dir: Path) -> bool:
    # Only reject an actual symlink component escaping the session cwd; lexical
    # ``..`` and ``~`` paths retain the TUI's existing completion behavior.
    try:
        relative = directory.relative_to(base_dir)
    except ValueError:
        return False
    resolved_base = base_dir.resolve()
    current = base_dir
    for component in relative.parts:
        current /= component
        if not current.is_symlink():
            continue
        try:
            current.resolve().relative_to(resolved_base)
        except ValueError:
            return True
    return False


def _completion_context(path_text: str, base_dir: Path) -> tuple[Path, str, str]:
    if path_text == "~":
        return Path.home(), "~/", ""
    slash = path_text.rfind("/")
    if slash < 0:
        return base_dir, "", path_text
    display_prefix = path_text[: slash + 1]
    directory_text = path_text[:slash] or "/"
    directory = Path.home() if directory_text == "~" else Path(directory_text).expanduser()
    if not directory.is_absolute():
        directory = base_dir / directory
    return directory, display_prefix, path_text[slash + 1 :]


def _candidate(display_prefix: str, name: str, is_directory: bool) -> str:
    candidate = f"/{name}" if display_prefix == "/" else f"{display_prefix}{name}"
    return f"{candidate}/" if is_directory else candidate


def _quoted_candidate(candidate: str, quoted: bool, is_directory: bool) -> str:
    if quoted:
        closing_quote = "" if is_directory else '"'
        return f"{candidate}{closing_quote}"
    if not any(character.isspace() for character in candidate):
        return candidate
    closing_quote = "" if is_directory else '"'
    return f'"{candidate}{closing_quote}'


__all__ = ["PathCompletion", "complete_paths"]
