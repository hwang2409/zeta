"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

import asyncio
import base64
import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.cursor_shapes import CursorShape, CursorShapeConfig
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, vi_insert_mode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.vi import load_vi_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from rich.text import Text

from ..types import (
    ImageContent,
    Message,
    MessageRole,
    TextContent,
    image_signature_matches,
)
from .theme import BODY, USER_ROLE

SHIFT_ENTER_SEQUENCES = frozenset(
    {
        "\x1b[27;2;13~",
        "\x1b[27;5;13~",
        "\x1b[27;6;13~",
    }
)
ATTACHMENT_MAX_TEXT_BYTES = 200 * 1024
ATTACHMENT_TOKEN_RE = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|([^\s]+))')


class AttachmentError(ValueError):
    """Raised when a composer attachment cannot be read or decoded."""


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    token: str
    path: Path


def attachment_refs(value: str, base_dir: str | Path) -> tuple[AttachmentRef, ...]:
    """Parse quoted or path-like local ``@`` references.

    Bare words such as ``@user`` and ``@dataclass`` remain prompt text.
    """

    base = Path(base_dir)
    refs: list[AttachmentRef] = []
    for match in ATTACHMENT_TOKEN_RE.finditer(value):
        raw_path = match.group(1) or match.group(2)
        if raw_path is None:
            continue
        quoted = match.group(1) is not None
        if not quoted and not (
            "/" in raw_path
            or raw_path.startswith(("./", "../", "~/"))
        ):
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base / path
        refs.append(AttachmentRef(match.group(0), path.resolve()))
    return tuple(refs)


def _image_media_type(data: bytes) -> str | None:
    if (
        len(data) >= 16
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
        and data[12:16] in {b"VP8 ", b"VP8L", b"VP8X"}
    ):
        return "image/webp"
    candidates = (
        ("image/png", data.startswith(b"\x89PNG\r\n\x1a\n")),
        ("image/jpeg", data.startswith(b"\xff\xd8\xff")),
        ("image/gif", data.startswith((b"GIF87a", b"GIF89a"))),
    )
    for media_type, matches in candidates:
        if matches and image_signature_matches(media_type, data):
            return media_type
    return None


def _read_attachment(path: Path) -> TextContent | ImageContent:
    if not path.exists():
        raise AttachmentError(f"file does not exist: {path}")
    if not path.is_file():
        raise AttachmentError(f"directory attachments are not supported: {path}")
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(64)
    except OSError as exc:
        raise AttachmentError(f"cannot read {path}: {exc}") from exc
    media_type = _image_media_type(prefix)
    try:
        if media_type is not None:
            data = path.read_bytes()
            if not image_signature_matches(media_type, data):
                raise AttachmentError(f"binary file is not an image: {path}")
            return ImageContent(
                base64.b64encode(data).decode("ascii"),
                media_type,
                str(path),
                size,
            )
        if size > ATTACHMENT_MAX_TEXT_BYTES:
            raise AttachmentError(
                f"text file is {size} bytes; limit is {ATTACHMENT_MAX_TEXT_BYTES} bytes: {path}"
            )
        data = path.read_bytes()
    except OSError as exc:
        raise AttachmentError(f"cannot read {path}: {exc}") from exc
    if b"\x00" in data:
        raise AttachmentError(f"binary file is not an image: {path}")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise AttachmentError(f"binary file is not an image: {path}") from None
    labeled = f"[file: {path} · {size} bytes]\n{text}"
    return TextContent(labeled, str(path), size)


def build_user_message(
    value: str,
    base_dir: str | Path,
    pending_paths: tuple[Path, ...] = (),
) -> Message:
    """Resolve references into one message, deduplicating resolved paths."""

    paths: list[Path] = []
    for ref in attachment_refs(value, base_dir):
        if ref.path not in paths:
            paths.append(ref.path)
    for path in pending_paths:
        resolved = path.resolve()
        if resolved not in paths:
            paths.append(resolved)
    blocks = [TextContent(value)]
    blocks.extend(_read_attachment(path) for path in paths)
    return Message(MessageRole.USER, blocks)


def paste_image(session_dir: str | Path) -> Path:
    """Save a macOS clipboard image in the session directory."""

    if platform.system() != "Darwin":
        raise AttachmentError("image paste is only available on macOS")
    destination = Path(session_dir) / f"clipboard-{uuid4().hex}.png"
    pngpaste = shutil.which("pngpaste")
    if pngpaste is not None:
        result = subprocess.run(
            [pngpaste, str(destination)],
            capture_output=True,
            check=False,
        )
    else:
        script = """
use framework "AppKit"
on run argv
    set destination to item 1 of argv
    set imageData to current application's NSPasteboard's generalPasteboard()'s dataForType:(current application's NSPasteboardTypePNG)
    if imageData is missing value then return "empty"
    imageData's writeToFile:destination atomically:true
    return "ok"
end run
"""
        result = subprocess.run(
            ["osascript", "-e", script, str(destination)],
            capture_output=True,
            check=False,
        )
    if result.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        raise AttachmentError("clipboard does not contain an image")
    try:
        if not destination.read_bytes():
            raise AttachmentError("clipboard does not contain an image")
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise AttachmentError(f"cannot read clipboard image: {exc}") from exc
    return destination


class ComposerAttachmentMixin:
    """Attachment behavior shared by the TUI composition root."""

    @staticmethod
    def _display_attachment_path(path: Path) -> str:
        value = str(path)
        if len(value) <= 80:
            return value
        return f".../{path.name}"[-80:]

    def slash_paste(self, args: str) -> str:
        if args.strip():
            return "paste unavailable: /paste does not take arguments"
        try:
            path = paste_image(self.loop.store.session_dir)
            attachment = build_user_message(
                "paste", self.loop.store.cwd, (path,)
            ).content[1]
        except AttachmentError as exc:
            return f"paste unavailable: {exc}"
        if not isinstance(attachment, ImageContent):
            return "paste unavailable: clipboard image could not be decoded"
        token = f"[Image #{self._next_image_token}]"
        self._next_image_token += 1
        self._pending_attachments.append(path)
        self._pending_attachment_tokens[token] = path
        return token

    def _delete_staged_attachment(self, path: Path) -> None:
        if (
            path.parent == self.loop.store.session_dir
            and path.name.startswith("clipboard-")
            and path.suffix == ".png"
        ):
            path.unlink(missing_ok=True)

    def _pending_paths_for(self, value: str) -> list[Path]:
        mapped_paths = set(self._pending_attachment_tokens.values())
        cancelled_paths: set[Path] = set()
        for token, path in tuple(self._pending_attachment_tokens.items()):
            if token not in value:
                del self._pending_attachment_tokens[token]
                cancelled_paths.add(path)
        remaining_mapped_paths = set(self._pending_attachment_tokens.values())
        for path in cancelled_paths - remaining_mapped_paths:
            self._delete_staged_attachment(path)
        self._pending_attachments[:] = [
            path
            for path in self._pending_attachments
            if path not in mapped_paths or path in remaining_mapped_paths
        ]

        token_paths = sorted(
            (
                (value.index(token), path)
                for token, path in self._pending_attachment_tokens.items()
            ),
            key=lambda item: item[0],
        )
        selected_paths = [path for _, path in token_paths]
        selected_paths.extend(
            path
            for path in self._pending_attachments
            if path not in remaining_mapped_paths
        )
        return selected_paths

    def _prepare_user_message(self, value: str) -> Message | None:
        try:
            message = build_user_message(value, self.loop.store.cwd)
        except AttachmentError as exc:
            self._print_system(f"attachment rejected: {exc}")
            return None

        valid_pending: list[Path] = []
        for path in self._pending_paths_for(value):
            try:
                build_user_message(value, self.loop.store.cwd, (path,))
            except AttachmentError as exc:
                self._print_system(f"pending attachment dropped: {exc}")
            else:
                valid_pending.append(path)
        self._pending_attachments[:] = valid_pending
        if not valid_pending:
            return message
        return build_user_message(value, self.loop.store.cwd, tuple(valid_pending))

    def _clear_pending_attachments(self) -> None:
        mapped_paths = set(self._pending_attachment_tokens.values())
        for path in self._pending_attachments:
            if path not in mapped_paths:
                self._delete_staged_attachment(path)
        self._pending_attachments.clear()
        self._pending_attachment_tokens.clear()
        self._next_image_token = 1

    def _print_user(self, user: str | Message) -> None:
        self._presenter.reset_assistant_unit()
        if isinstance(user, str):
            self._print_unit(Text.assemble(("▌ ", USER_ROLE), (user, BODY)))
            return
        prompt = next(
            (
                block.text
                for block in user.content
                if isinstance(block, TextContent) and block.path is None
            ),
            "",
        )
        rendered = Text.assemble(("▌ ", USER_ROLE), (prompt, BODY))
        for block in user.content:
            if isinstance(block, TextContent) and block.path is not None:
                label = self._display_attachment_path(Path(block.path))
                rendered.append(
                    f"\n  file · {label} · {block.size or 0} bytes",
                    style="dim",
                )
        self._print_unit(rendered)

    def _replay_attachment_messages(self) -> None:
        if self._replay_rendered:
            return
        self._replay_rendered = True
        for entry in self.loop.store.replay():
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            if message.role is MessageRole.USER and any(
                isinstance(block, (ImageContent, TextContent)) and block.path is not None
                for block in message.content
            ):
                self._print_user(message)

    def _start_queued_turn(self) -> None:
        if not self._queued:
            return
        user_message = self._queued.popleft()
        user_text = next(
            block.text
            for block in user_message.content
            if isinstance(block, TextContent) and block.path is None
        )
        self._print_user(user_message)
        self._print(Text("[queued]", style="dim"))
        self._start_turn(user_text, user_message=user_message)

    def _start_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
    ) -> None:
        self._active_task = asyncio.create_task(
            self._consume_turn(user_text, user_message=user_message)
        )


class VimCursorShapeConfig(CursorShapeConfig):
    """Use a beam in insert mode and a block in every other vi mode."""

    def get_cursor_shape(self, application: Application[Any]) -> CursorShape:
        if getattr(application, "editing_mode", None) is not EditingMode.VI:
            return CursorShape._NEVER_CHANGE
        if getattr(application.vi_state, "input_mode", None) in {
            InputMode.INSERT,
            InputMode.INSERT_MULTIPLE,
        }:
            return CursorShape.BEAM
        return CursorShape.BLOCK


def vim_state_label(vim_mode: bool) -> str | None:
    """Return the native prompt-toolkit vi state for the footer."""

    if not vim_mode:
        return None
    try:
        app = get_app()
    except RuntimeError:
        return "INSERT"
    if getattr(app, "editing_mode", EditingMode.VI) is not EditingMode.VI:
        return None
    buffer = getattr(app, "current_buffer", None)
    if buffer is not None and buffer.selection_state is not None:
        return "VISUAL"
    mode = getattr(getattr(app, "vi_state", None), "input_mode", None)
    return "NORMAL" if mode is InputMode.NAVIGATION else "INSERT"


def status_formatted_text(status: Text) -> FormattedText:
    """Convert Rich status spans into prompt-toolkit fragments."""

    fragments: list[tuple[str, str]] = []
    boundaries = {0, len(status.plain)}
    for span in status.spans:
        boundaries.update((span.start, span.end))
    ordered_boundaries = sorted(boundaries)
    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        styles = ["class:status-bar"]
        if status.style:
            styles.append(str(status.style))
        styles.extend(
            str(span.style)
            for span in status.spans
            if span.start <= start and end <= span.end
        )
        fragments.append((" ".join(styles), status.plain[start:end]))
    return FormattedText(fragments)


def parse_input(value: str) -> str | None:
    """Return a usable user turn, or None for blank input."""

    stripped = value.strip()
    return stripped or None


def build_key_bindings(
    *,
    on_interrupt: Callable[[], None],
    on_exit: Callable[[], None],
    on_submit: Callable[[str], None] | None = None,
    on_paste: Callable[[KeyPressEvent], None] | None = None,
    on_page_up: Callable[[], None] | None = None,
    on_page_down: Callable[[], None] | None = None,
    on_toggle_agent: Callable[[], None] | None = None,
) -> KeyBindings:
    """Build the small key map used by the full-screen composer."""

    bindings = KeyBindings()
    escape_chord_pending = False
    escape_chord_cursor_position: int | None = None

    @Condition
    def vi_insert_history_navigation() -> bool:
        app = get_app()
        buffer = app.current_buffer
        return vi_insert_mode() and (
            not buffer.text
            or buffer.working_index < len(buffer._working_lines) - 1
        )

    @Condition
    def full_screen_mode() -> bool:
        return get_app().full_screen

    def insert_newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("enter")
    def submit(event: KeyPressEvent) -> None:
        nonlocal escape_chord_cursor_position, escape_chord_pending
        if escape_chord_pending:
            escape_chord_pending = False
            if escape_chord_cursor_position is not None:
                event.current_buffer.cursor_position = escape_chord_cursor_position
            escape_chord_cursor_position = None
            event.app.vi_state.input_mode = InputMode.INSERT
            insert_newline(event)
            return
        if event.data in SHIFT_ENTER_SEQUENCES:
            insert_newline(event)
            return
        if on_submit is not None:
            event.current_buffer.append_to_history()
            on_submit(event.current_buffer.text)
            event.current_buffer.reset()
        else:
            event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def newline(event: KeyPressEvent) -> None:
        insert_newline(event)

    if on_paste is not None:

        @bindings.add("c-v")
        def paste(event: KeyPressEvent) -> None:
            on_paste(event)

    @bindings.add("escape", "enter", filter=~full_screen_mode)
    def alt_enter(event: KeyPressEvent) -> None:
        insert_newline(event)

    native_escape = next(
        binding
        for binding in load_vi_bindings().bindings
        if binding.keys == (Keys.Escape,)
    )

    bindings.add(Keys.Escape, filter=native_escape.filter & ~full_screen_mode)(native_escape)

    @bindings.add(Keys.Escape, filter=native_escape.filter & full_screen_mode, eager=True)
    def escape(event: KeyPressEvent) -> None:
        nonlocal escape_chord_cursor_position, escape_chord_pending
        escape_chord_cursor_position = event.current_buffer.cursor_position
        native_escape.call(event)
        next_key = next(iter(event.key_processor.input_queue), None)
        escape_chord_pending = next_key is not None and next_key.key == Keys.Enter
        if not escape_chord_pending:
            escape_chord_cursor_position = None

    @bindings.add("up", filter=vi_insert_history_navigation)
    def history_up(event: KeyPressEvent) -> None:
        event.current_buffer.auto_up()

    @bindings.add("down", filter=vi_insert_history_navigation)
    def history_down(event: KeyPressEvent) -> None:
        event.current_buffer.auto_down()

    @bindings.add("c-c")
    def interrupt(event: KeyPressEvent) -> None:
        on_interrupt()
        event.current_buffer.reset()

    @bindings.add("c-d")
    def exit_prompt(event: KeyPressEvent) -> None:
        on_exit()
        event.app.exit(exception=EOFError())

    if on_page_up is not None:

        @bindings.add("pageup")
        def page_up(event: KeyPressEvent) -> None:
            del event
            on_page_up()

    if on_page_down is not None:

        @bindings.add("pagedown")
        def page_down(event: KeyPressEvent) -> None:
            del event
            on_page_down()

    if on_toggle_agent is not None:

        @bindings.add("c-x", "c-o")
        def toggle_agent(event: KeyPressEvent) -> None:
            del event
            on_toggle_agent()

    return bindings


def history_for(path: str | Path) -> FileHistory:
    """Create a persistent history object and its parent directory."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(history_path))
