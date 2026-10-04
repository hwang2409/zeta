"""Session labels, previews, and relative timestamps."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime

from rich.cells import cell_len

from .core.session_files import SessionError

SESSION_NAME_MAX_LENGTH = 60
_ANSI_SEQUENCE = re.compile(
    r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x9b[0-?]*[ -/]*[@-~])"
    r"|(?:\x1b\][^\x07]*(?:\x07|\x1b\\)|\x9d[^\x07]*(?:\x07|\x1b\\))"
    r"|\x1b[ -/]*[@-~]"
)
_PREVIEW_CODEPOINT_LIMIT = 512
_PREVIEW_STRIPPED_CHARACTERS = frozenset(
    chr(codepoint)
    for start, end in ((0x200B, 0x200D), (0x202A, 0x202E), (0x2066, 0x2069))
    for codepoint in range(start, end + 1)
) | {"\ufeff"}


def normalize_session_name(value: str) -> str:
    """Return a validated session label or raise ``SessionError``."""

    cleaned = _ANSI_SEQUENCE.sub("", value)
    cleaned = "".join(
        character
        for character in cleaned
        if character not in _PREVIEW_STRIPPED_CHARACTERS
        and unicodedata.category(character) != "Cc"
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        raise SessionError("session name must be a nonempty label")
    if cell_len(cleaned) > SESSION_NAME_MAX_LENGTH:
        raise SessionError(
            f"session name is too long (max {SESSION_NAME_MAX_LENGTH} cells)"
        )
    return cleaned


def preview_text(value: str, *, limit: int = 80) -> str:
    clean = _ANSI_SEQUENCE.sub("", value)
    clean = "".join(
        character
        for character in clean
        if (
            character not in _PREVIEW_STRIPPED_CHARACTERS
            and (character in "\t\n\r" or unicodedata.category(character) != "Cc")
        )
    )
    clean = clean[:_PREVIEW_CODEPOINT_LIMIT]
    clean = " ".join(clean.split())
    if cell_len(clean) <= limit:
        return clean
    suffix = "..."
    available = max(0, limit - cell_len(suffix))
    result = ""
    for character in clean:
        candidate = result + character
        if cell_len(candidate) > available:
            break
        result = candidate
    return result + suffix


def format_relative_age(updated_at: str, *, now: datetime | None = None) -> str:
    """Return a compact human-readable age such as ``2h ago``."""

    try:
        parsed = datetime.fromisoformat(updated_at)
    except ValueError:
        return "unknown"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    delta_seconds = int((reference - parsed).total_seconds())
    if delta_seconds < 5:
        return "just now"
    if delta_seconds < 60:
        return f"{delta_seconds}s ago"
    minutes = delta_seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months < 12:
        return f"{months}mo ago"
    return f"{days // 365}y ago"
