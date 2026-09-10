"""Plain-text export of the visible transcript for ``/copy``.

The full-screen transcript lives on prompt-toolkit's alternate screen with
mouse reporting on, so a terminal drag-select either scrolls or grabs card
borders, side margins, and ellipsized error reasons along with the words.
``/copy`` walks the same logical units the transcript paints and writes them
out as border-free text, then hands that to the system clipboard, or to a
file in the session directory when no clipboard tool is available.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.panel import Panel

from ...core.process_env import subprocess_env
from ..theme import RICH_THEME
from ..transcript import _ToolUnit

USER_PREFIX = "▌ "
EXPORT_WIDTH = 100
CLIPBOARD_TIMEOUT = 5.0
_BLANK_RUN = re.compile(r"\n{3,}")

ExportEntry = tuple["RenderableType | _ToolUnit | None", bool]


class ClipboardError(Exception):
    """Raised when no clipboard tool is available or the copy fails."""


def plain_transcript(
    entries: Sequence[ExportEntry],
    *,
    header: str = "",
    last_turn: bool = False,
    width: int = EXPORT_WIDTH,
) -> str:
    """Render transcript units as plain text with no borders, margins, or color.

    ``entries`` pairs each unit with whether it is a user message; blank
    separator units arrive as ``None``. ``last_turn`` keeps only the newest
    user message and everything after it. Returns ``""`` when nothing has
    been shown yet, so callers can tell an empty chat from a copied one.
    """

    selected = list(entries)
    if last_turn:
        starts = [index for index, (_, is_user) in enumerate(selected) if is_user]
        if starts:
            selected = selected[starts[-1] :]
    blocks = [
        _quote_user(_entry_text(value, width)) if is_user else _entry_text(value, width)
        for value, is_user in selected
    ]
    body = _BLANK_RUN.sub("\n\n", "\n".join(blocks)).strip("\n")
    if not body:
        return ""
    if header:
        body = f"{header}\n\n{body}"
    return body + "\n"


def _entry_text(value: RenderableType | _ToolUnit | None, width: int) -> str:
    """Flatten one unit to text, preferring each renderable's own plain form.

    Panels are unwrapped rather than drawn so their borders never reach the
    clipboard; a renderable that carries ``plain_export`` (the error card,
    which truncates on screen) supplies its full text directly.
    """

    if value is None:
        return ""
    if isinstance(value, _ToolUnit):
        value = value.renderable
    exported = getattr(value, "plain_export", None)
    if isinstance(exported, str):
        return exported
    plain = getattr(value, "plain", None)
    if isinstance(plain, str):
        return plain
    code = getattr(value, "code", None)
    if isinstance(code, str):
        return code
    if isinstance(value, (Panel, Padding)):
        return _entry_text(value.renderable, width)
    children = getattr(value, "renderables", None)
    if children is not None:
        return "\n".join(_entry_text(child, width) for child in children)
    return _render_plain(value, width)


def _render_plain(value: RenderableType, width: int) -> str:
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=False,
        no_color=True,
        color_system=None,
        highlight=False,
        markup=False,
        emoji=False,
        legacy_windows=False,
        theme=RICH_THEME,
    )
    console.print(value, soft_wrap=True)
    return "\n".join(line.rstrip() for line in output.getvalue().splitlines())


def _quote_user(text: str) -> str:
    """Swap the composer's bar for a markdown quote so the role stays visible."""

    lines = text.removeprefix(USER_PREFIX).splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def _which(tool: str, *arguments: str) -> list[str] | None:
    path = shutil.which(tool)
    return [path, *arguments] if path else None


def clipboard_command() -> list[str] | None:
    """Return the argv of the local clipboard writer, or None when there is none."""

    system = platform.system()
    if system == "Darwin":
        return _which("pbcopy")
    if system == "Windows":
        return _which("clip")
    if os.environ.get("WAYLAND_DISPLAY"):
        found = _which("wl-copy")
        if found is not None:
            return found
    return _which("xclip", "-selection", "clipboard", "-in") or _which(
        "xsel", "--clipboard", "--input"
    )


def copy_to_clipboard(text: str) -> str:
    """Put ``text`` on the system clipboard and return the tool that did it."""

    command = clipboard_command()
    if command is None:
        raise ClipboardError("no clipboard tool found: pbcopy, wl-copy, xclip, or xsel")
    name = Path(command[0]).name
    try:
        result = subprocess.run(
            command,
            input=text.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=CLIPBOARD_TIMEOUT,
            env=subprocess_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClipboardError(f"{name} failed: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        suffix = f": {detail}" if detail else ""
        raise ClipboardError(f"{name} exited {result.returncode}{suffix}")
    return name


def write_transcript_file(session_dir: str | Path, text: str) -> Path:
    """Save the export beside the session so it survives a missing clipboard."""

    directory = Path(session_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = directory / f"transcript-{stamp}.txt"
    counter = 1
    while path.exists():
        counter += 1
        path = directory / f"transcript-{stamp}-{counter}.txt"
    path.write_text(text, encoding="utf-8")
    return path


__all__ = [
    "ClipboardError",
    "clipboard_command",
    "copy_to_clipboard",
    "plain_transcript",
    "write_transcript_file",
]
