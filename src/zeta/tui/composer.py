"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cursor_shapes import CursorShape, CursorShapeConfig
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.filters import Condition, is_searching, vi_insert_mode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.vi import load_vi_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from rich.text import Text

from ..types import (
    ErrorInfo,
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    image_signature_matches,
)
from .render import is_retryable_error, render_event
from .theme import BODY, CHROME, DIM, ERROR, USER_ROLE

SHIFT_ENTER_SEQUENCES = frozenset(
    {
        "\x1b[27;2;13~",
        "\x1b[27;5;13~",
        "\x1b[27;6;13~",
    }
)
ATTACHMENT_MAX_TEXT_BYTES = 200 * 1024
ATTACHMENT_TOKEN_RE = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|([^\s]+))')
SPINNER_INTERVAL = 0.2
HISTORY_LIMIT = 1000
DRAFT_WRITE_DELAY = 0.2


class TurnConsumerMixin:
    """Consume loop events and preserve failed-turn recovery state."""

    async def _pulse_spinner(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._spinner_reset.wait(), timeout=SPINNER_INTERVAL
                )
            except asyncio.TimeoutError:
                if self._spinner_active:
                    self._spinner_frame += 1
                    if self._presenter.has_active_agent:
                        self._presenter.refresh_active_agents()
                    self._invalidate_prompt()
            else:
                self._spinner_reset.clear()

    async def _consume_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
    ) -> None:
        self._abort_requested = False
        self._turn_had_visible_output = False
        self._loop_state = "streaming"
        self._streaming = True
        self._spinner_active = True
        spinner_task = asyncio.create_task(self._pulse_spinner())
        retry_message = user_message or Message(
            role=MessageRole.USER,
            content=[TextContent(user_text)],
        )
        turn_failed = False
        try:
            async for event in self.loop.run_turn(
                user_text,
                user_message=user_message,
                persist_user_message=persist_user_message,
            ):
                self._update_usage(event)
                self._usage_tracker.record(event.type, self.model)
                self._prepare_stream_event(event)
                stop_after_tool = self._handle_tool_event(event)
                if self.verbose:
                    self._print(Text(json.dumps(event.to_dict(), sort_keys=True), style=DIM))
                if event.type is StreamEventType.MESSAGE_UPDATE:
                    self._consume_text(event)
                    self._invalidate_prompt()
                    continue
                if event.type is StreamEventType.TURN_START:
                    self._spinner_frame = 0
                    self._spinner_reset.set()
                    self._streaming = True
                    self._compaction_shown = False
                elif event.type is StreamEventType.COMPACTION_START:
                    self._compaction_shown = True
                    self._loop_state = "compacting"
                elif event.type is StreamEventType.COMPACTION_END:
                    self._loop_state = "streaming"
                elif event.type is StreamEventType.MESSAGE_END:
                    self._finish_message(event)
                    self._streaming = False
                elif event.type is StreamEventType.AGENT_END:
                    self._reset_stream_state()
                    self._loop_state = "idle"
                    if not turn_failed:
                        self._failed_turn = None
                    if not self._turn_had_visible_output:
                        self._print_unit(Text("no response", style=CHROME))
                        self._turn_had_visible_output = True
                if event.type not in {
                    StreamEventType.TOOL_APPROVAL_START,
                    StreamEventType.TOOL_APPROVAL_END,
                    StreamEventType.TOOL_EXECUTION_UPDATE,
                    StreamEventType.TOOL_EXECUTION_END,
                    StreamEventType.TOOL_EXECUTION_START,
                }:
                    rendered = render_event(event)
                    if rendered is not None:
                        self._turn_had_visible_output |= (
                            event.type is StreamEventType.MESSAGE_END
                            and bool(event.data.get("truncated"))
                        )
                        if event.type in {
                            StreamEventType.AGENT_END,
                            StreamEventType.ERROR,
                        }:
                            self._print_unit(rendered)
                            if event.type is StreamEventType.ERROR and is_retryable_error(
                                event.error
                            ):
                                turn_failed = True
                                self._failed_turn = (user_text, retry_message)
                                self._turn_had_visible_output = True
                        else:
                            self._print(rendered)
                self._invalidate_prompt()
                if event.type is StreamEventType.TOOL_EXECUTION_END and stop_after_tool:
                    break
        except asyncio.CancelledError:
            self._flush_stream_kind(preserve_inline=True)
            self._presenter.reset_assistant_unit()
            self._reset_stream_state()
            self._loop_state = "interrupted"
            self._print_unit(Text("[aborted]", style=ERROR))
            raise
        except Exception as exc:
            self._flush_stream_kind(preserve_inline=True)
            self._presenter.reset_assistant_unit()
            self._reset_stream_state()
            self._loop_state = "idle"
            try:
                message = str(exc).strip() or type(exc).__name__
            except Exception:
                message = "unexpected ui error"
            self._print_unit(
                render_event(
                    StreamEvent(
                        StreamEventType.ERROR,
                        error=ErrorInfo("ui_error", message),
                    )
                )
            )
        finally:
            self._usage_tracker.record_compaction(self.model)
            self._presenter.clear_active_tool_calls()
            self._discard_tool_region()
            self._presenter.reset_assistant_message()
            self._abort_requested = False
            self._streaming = False
            self._spinner_active = False
            spinner_task.cancel()
            await asyncio.gather(spinner_task, return_exceptions=True)
            self._invalidate_prompt()


class AttachmentError(ValueError):
    """Raised when a composer attachment cannot be read or decoded."""


class BoundedFileHistory(FileHistory):
    """Store recent composer entries without retaining attachment payloads."""

    def __init__(self, filename: str | Path, *, limit: int = HISTORY_LIMIT) -> None:
        if limit < 1:
            raise ValueError("history limit must be positive")
        self.limit = limit
        super().__init__(str(filename))

    def load_history_strings(self) -> list[str]:
        return list(super().load_history_strings())[: self.limit]

    def append_string(self, string: str) -> None:
        if not self._loaded:
            self._loaded_strings = list(self.load_history_strings())
            self._loaded = True
        if self._loaded_strings and self._loaded_strings[0] == string:
            return
        self._loaded_strings.insert(0, string)
        del self._loaded_strings[self.limit :]
        self._write_entries(self._loaded_strings[::-1])

    def store_string(self, string: str) -> None:
        """Keep direct history writes bounded as well."""

        if self._loaded:
            entries = [string, *self._loaded_strings]
        else:
            entries = [string, *self.load_history_strings()]
        self._write_entries(entries[: self.limit][::-1])

    def _write_entries(self, entries: list[str]) -> None:
        history_path = Path(self.filename)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=history_path.parent,
                prefix=f".{history_path.name}.",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                for entry in entries:
                    handle.write(b"\n# zeta composer history\n")
                    for line in entry.split("\n"):
                        handle.write(f"+{line}\n".encode("utf-8", errors="replace"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, history_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)


class DraftPersistence:
    """Persist the current composer text after a short idle delay."""

    def __init__(self, path: str | Path, *, delay: float = DRAFT_WRITE_DELAY) -> None:
        self.path = Path(path)
        self.delay = delay
        self._pending_text: str | None = None
        self._pending_revision: int | None = None
        self._revision = 0
        self._persisted_revision: int | None = None
        self._submitted_revision: int | None = None
        self._scheduled: asyncio.TimerHandle | None = None

    def load(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError:
            return ""

    def schedule(self, text: str) -> None:
        self._revision += 1
        self._pending_text = text
        self._pending_revision = self._revision
        if self._scheduled is not None:
            self._scheduled.cancel()
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.flush()
            return
        self._scheduled = event_loop.call_later(self.delay, self.flush)

    def flush(self) -> None:
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        text = self._pending_text
        revision = self._pending_revision
        self._pending_text = None
        self._pending_revision = None
        if text is None:
            return
        if not text:
            self.path.unlink(missing_ok=True)
            self._persisted_revision = revision
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._persisted_revision = revision
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)

    def clear(self) -> None:
        self._pending_text = None
        self._pending_revision = None
        self._submitted_revision = None
        if self._scheduled is not None:
            self._scheduled.cancel()
            self._scheduled = None
        self.path.unlink(missing_ok=True)
        self._persisted_revision = None

    def mark_submitted(self) -> None:
        """Remember the draft revision submitted by the prompt callback."""

        self._submitted_revision = self._revision

    def clear_submitted(self) -> bool:
        """Clear only the revision captured by the most recent submission."""

        submitted_revision = self._submitted_revision
        self._submitted_revision = None
        if submitted_revision is None:
            return False
        if (
            self._pending_revision is not None
            and self._pending_revision <= submitted_revision
        ):
            self._pending_text = None
            self._pending_revision = None
            if self._scheduled is not None:
                self._scheduled.cancel()
                self._scheduled = None
        if (
            self._persisted_revision is not None
            and self._persisted_revision <= submitted_revision
        ):
            self.path.unlink(missing_ok=True)
            self._persisted_revision = None
        return True

    def attach(self, buffer: Buffer) -> None:
        draft = self.load()
        if draft:
            self._persisted_revision = self._revision
        if draft and not buffer.text:
            buffer.set_document(Document(draft, len(draft)))

        def changed(_buffer: Buffer) -> None:
            self.schedule(buffer.text)

        buffer.on_text_changed += changed


@dataclass(frozen=True, slots=True)
class UndoCandidate:
    """Keep submitted text and attachment references available for undo."""

    text: str
    attachment_paths: tuple[Path, ...] = ()
    attachment_tokens: tuple[tuple[str, Path], ...] = ()

    @classmethod
    def from_message(
        cls,
        text: str,
        message: Message,
        attachment_tokens: Mapping[str, Path],
    ) -> UndoCandidate:
        paths = tuple(
            dict.fromkeys(
                Path(block.path).resolve()
                for block in message.content
                if isinstance(block, (TextContent, ImageContent))
                and block.path is not None
            )
        )
        path_set = set(paths)
        tokens = tuple(
            (token, path.resolve())
            for token, path in attachment_tokens.items()
            if path.resolve() in path_set
        )
        return cls(text, paths, tokens)


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

    def _undo_candidate_for_message(
        self, text: str, message: Message
    ) -> UndoCandidate:
        return UndoCandidate.from_message(
            text, message, self._pending_attachment_tokens
        )

    def _restore_composer(self, value: str) -> None:
        session = self._active_session or self._session
        if session is None:
            return
        session.app.current_buffer.set_document(Document(value, len(value)))

    def undo_sent_turn(self) -> None:
        """Abort the current turn and restore its submitted text once."""

        candidate = self._undo_candidate
        if (
            candidate is None
            or not self.active
            or self._loop_state
            not in {"streaming", "compacting", "tool-running", "approval"}
        ):
            self._print_unit(Text("undo unavailable: turn already completed", style=DIM))
            return
        self._undo_candidate = None
        self.abort_active()
        if isinstance(candidate, str):
            candidate = UndoCandidate(candidate)
        session = self._active_session or self._session
        buffer = session.app.current_buffer if session is not None else None
        if buffer is not None and buffer.text:
            self._print_system(
                f"undo kept the current draft; sent text: {candidate.text}"
            )
            self._draft.schedule(buffer.text)
            return
        self._restore_composer(candidate.text)
        self._pending_attachments[:] = list(candidate.attachment_paths)
        self._pending_attachment_tokens.clear()
        self._pending_attachment_tokens.update(candidate.attachment_tokens)
        self._next_image_token = len(self._pending_attachment_tokens) + 1
        self._draft.schedule(candidate.text)

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

    def _start_queued_turn(self) -> None:
        if not self._queued:
            return
        user_message, candidate = self._queued.popleft()
        user_text = next(
            block.text
            for block in user_message.content
            if isinstance(block, TextContent) and block.path is None
        )
        self._print_user(user_message)
        self._print(Text("[queued]", style="dim"))
        self._undo_candidate = candidate
        self._start_turn(user_text, user_message=user_message)

    def _start_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
    ) -> None:
        self._loop_state = "streaming"
        self._active_task = asyncio.create_task(
            self._consume_turn(
                user_text,
                user_message=user_message,
                persist_user_message=persist_user_message,
            )
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
    on_retry: Callable[[], None] | None = None,
    retry_available: Callable[[], bool] | None = None,
    on_undo: Callable[[], None] | None = None,
    append_history: bool = True,
) -> KeyBindings:
    """Build the small key map used by the full-screen composer."""

    bindings = KeyBindings()
    escape_chord_pending = False
    escape_chord_cursor_position: int | None = None
    history_navigation_active = False
    history_navigation_buffer: Buffer | None = None
    suppress_history_detach = False

    def track_history_buffer(buffer: Buffer) -> None:
        nonlocal history_navigation_active, history_navigation_buffer
        if history_navigation_buffer is buffer:
            return
        history_navigation_buffer = buffer

        def detach_history_navigation(_buffer: Buffer) -> None:
            nonlocal history_navigation_active
            if not suppress_history_detach:
                history_navigation_active = False

        buffer.on_text_changed += detach_history_navigation

    @Condition
    def vi_insert_history_navigation() -> bool:
        app = get_app()
        buffer = app.current_buffer
        track_history_buffer(buffer)
        return vi_insert_mode()

    @Condition
    def emacs_history_navigation() -> bool:
        app = get_app()
        track_history_buffer(app.current_buffer)
        return app.editing_mode is EditingMode.EMACS

    @Condition
    def full_screen_mode() -> bool:
        return get_app().full_screen

    @Condition
    def retry_ready() -> bool:
        return (
            on_retry is not None
            and (retry_available is None or retry_available())
        )

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
            if append_history:
                event.current_buffer.append_to_history()
            on_submit(event.current_buffer.text)
            event.current_buffer.reset()
        else:
            event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def newline(event: KeyPressEvent) -> None:
        insert_newline(event)

    @bindings.add("enter", filter=is_searching, eager=True)
    def accept_history_search(event: KeyPressEvent) -> None:
        del event
        from prompt_toolkit.search import accept_search

        accept_search()

    if on_retry is not None:

        @bindings.add("c-r", filter=retry_ready, eager=True)
        def retry(event: KeyPressEvent) -> None:
            del event
            on_retry()

    if on_undo is not None:

        @bindings.add("c-u", eager=True)
        def undo(event: KeyPressEvent) -> None:
            del event
            on_undo()

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

    @bindings.add(
        "up", filter=vi_insert_history_navigation | emacs_history_navigation
    )
    def history_up(event: KeyPressEvent) -> None:
        nonlocal history_navigation_active, suppress_history_detach
        buffer = event.current_buffer
        if not history_navigation_active and buffer.text:
            if buffer.document.cursor_position_row > 0:
                buffer.auto_up()
            return
        suppress_history_detach = True
        try:
            buffer.history_backward()
        finally:
            suppress_history_detach = False
        history_navigation_active = buffer.text != ""

    @bindings.add(
        "down", filter=vi_insert_history_navigation | emacs_history_navigation
    )
    def history_down(event: KeyPressEvent) -> None:
        nonlocal history_navigation_active, suppress_history_detach
        buffer = event.current_buffer
        if not history_navigation_active:
            if buffer.document.cursor_position_row < buffer.document.line_count - 1:
                buffer.auto_down()
            return
        suppress_history_detach = True
        try:
            buffer.history_forward()
            buffer.cursor_position = len(buffer.text)
        finally:
            suppress_history_detach = False
        history_navigation_active = buffer.text != ""

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


def history_for(path: str | Path) -> BoundedFileHistory:
    """Create a persistent history object and its parent directory."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return BoundedFileHistory(history_path)
