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
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding.vi_state import InputMode
from rich.text import Text

from ..core.abort import AbortSignal
from ..core.approval import ApprovalRequest
from ..core.process_env import subprocess_env
from ..core.session_files import (
    child_directory,
    open_session_file,
    session_root,
    write_session_file,
)
from ..core.slash import SlashCommandRegistry
from ..core.store import ConversationStore
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
from . import theme
from .key_bindings import (
    FullScreenPromptSession,
    VimCursorShapeConfig,
    build_key_bindings,
)
from .render import is_retryable_error, render_event

ATTACHMENT_MAX_TEXT_BYTES = 200 * 1024
ATTACHMENT_TOKEN_RE = re.compile(r'(?<!\S)@(?:"([^"\n]+)"|([^\s]+))')
SPINNER_INTERVAL = 0.2

__all__ = [
    "FullScreenPromptSession",
    "SlashCompleter",
    "VimCursorShapeConfig",
    "build_key_bindings",
]


class TurnConsumerMixin:
    """Consume loop events and preserve failed-turn recovery state."""

    async def _pulse_spinner(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._spinner_reset.wait(), timeout=SPINNER_INTERVAL
                )
            except TimeoutError:
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
        abort_signal: Any | None = None,
    ) -> None:
        user_text, user_message = self._consume_macro_receipts(
            user_text, user_message
        )
        self._abort_requested = False
        self._turn_had_visible_output = False
        self._loop_state = "streaming"
        self._streaming = True
        self._spinner_active = True
        turn_abort_signal = abort_signal or self.loop.tool_registry.abort_signal.registry.new_generation()
        self._standalone_abort_signal = turn_abort_signal
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
                abort_signal=turn_abort_signal,
            ):
                self._update_usage(event)
                self._usage_tracker.record(event.type, self.model)
                self._prepare_stream_event(event)
                stop_after_tool = self._handle_tool_event(event)
                if self.verbose:
                    self._print(Text(json.dumps(event.to_dict(), sort_keys=True), style=theme.DIM))
                if event.type is StreamEventType.MESSAGE_UPDATE:
                    self._consume_text(event)
                    self._invalidate_prompt()
                    continue
                if event.type is StreamEventType.TURN_START:
                    self._todo_widget.turn_boundary()
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
                        self._print_unit(Text("no response", style=theme.CHROME))
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
            self._print_unit(Text("[aborted]", style=theme.ERROR))
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
            if self._standalone_abort_signal is turn_abort_signal:
                self._standalone_abort_signal = None
            spinner_task.cancel()
            await asyncio.gather(spinner_task, return_exceptions=True)
            self._invalidate_prompt()


class AttachmentError(ValueError):
    """Raised when a composer attachment cannot be read or decoded."""


class SlashCompleter(Completer):
    """Complete slash commands with descriptions and custom-source badges."""

    def __init__(self, registry: SlashCommandRegistry) -> None:
        self.registry = registry

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        del complete_event
        before_cursor = document.text_before_cursor
        if not before_cursor.startswith("/") or any(
            character.isspace() for character in before_cursor
        ):
            return
        prefix = before_cursor[1:]
        for name, description, source in self.registry.completion_entries:
            if not name.startswith(prefix):
                continue
            meta = description
            if source:
                meta = f"[{source}] {description}".strip()
            yield Completion(
                name,
                start_position=-len(prefix),
                display=f"/{name}",
                display_meta=meta,
            )


@dataclass(frozen=True, slots=True)
class UndoCandidate:
    """Keep submitted text and attachment references available for undo."""

    text: str
    attachment_paths: tuple[Path, ...] = ()
    attachment_tokens: tuple[tuple[str, Path], ...] = ()
    next_image_token: int = 1
    submission_id: int | None = None

    @classmethod
    def from_message(
        cls,
        text: str,
        message: Message,
        attachment_tokens: Mapping[str, Path],
        next_image_token: int,
        *,
        submission_id: int | None = None,
    ) -> UndoCandidate:
        paths = tuple(
            dict.fromkeys(
                _attachment_path(Path(block.path))
                for block in message.content
                if isinstance(block, (TextContent, ImageContent))
                and block.path is not None
            )
        )
        path_set = set(paths)
        tokens = tuple(
            (token, _attachment_path(path))
            for token, path in attachment_tokens.items()
            if _attachment_path(path) in path_set
        )
        return cls(text, paths, tokens, next_image_token, submission_id)


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
        refs.append(AttachmentRef(match.group(0), _attachment_path(path)))
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


def _attachment_path(path: Path) -> Path:
    # Preserve session components so link validation happens at descriptor opens.
    return path.absolute() if "sessions" in path.parts else path.resolve()


def _read_attachment(path: Path, session_store: ConversationStore | None = None) -> TextContent | ImageContent:
    try:
        with ExitStack() as cleanup:
            if session_store is not None and path.is_relative_to(session_store.session_dir):
                directory_fd = session_store.directory_fd
                for component in path.parent.relative_to(session_store.session_dir).parts:
                    directory_fd = cleanup.enter_context(child_directory(directory_fd, component))
                handle = cleanup.enter_context(os.fdopen(open_session_file(directory_fd, path.name, os.O_RDONLY), "rb"))
            elif "sessions" in path.parts:
                directory_fd = cleanup.enter_context(session_root(path.parent))
                handle = cleanup.enter_context(os.fdopen(open_session_file(directory_fd, path.name, os.O_RDONLY), "rb"))
            else:
                if not path.exists():
                    raise AttachmentError(f"file does not exist: {path}")
                if not path.is_file():
                    raise AttachmentError(f"directory attachments are not supported: {path}")
                handle = cleanup.enter_context(path.open("rb"))
            size = os.fstat(handle.fileno()).st_size
            prefix = handle.read(64)
            if _image_media_type(prefix) is None and size > ATTACHMENT_MAX_TEXT_BYTES:
                raise AttachmentError(
                    f"text file is {size} bytes; limit is {ATTACHMENT_MAX_TEXT_BYTES} bytes: {path}"
                )
            data = prefix + handle.read()
    except FileNotFoundError as exc:
        raise AttachmentError(f"file does not exist: {path}") from exc
    except IsADirectoryError as exc:
        raise AttachmentError(f"directory attachments are not supported: {path}") from exc
    except AttachmentError:
        raise
    except (OSError, ValueError) as exc:
        raise AttachmentError(f"cannot read {path}: {exc}") from exc
    media_type = _image_media_type(prefix)
    if media_type is not None:
        if not image_signature_matches(media_type, data):
            raise AttachmentError(f"binary file is not an image: {path}")
        return ImageContent(
            base64.b64encode(data).decode("ascii"),
            media_type,
            str(path),
            size,
        )
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
    *,
    attachment_value: str | None = None,
    session_store: ConversationStore | None = None,
) -> Message:
    """Resolve references into one message, deduplicating resolved paths."""

    paths: list[Path] = []
    source = value if attachment_value is None else attachment_value
    for ref in attachment_refs(source, base_dir):
        if ref.path not in paths:
            paths.append(ref.path)
    for path in pending_paths:
        resolved = _attachment_path(path)
        if resolved not in paths:
            paths.append(resolved)
    blocks = [TextContent(value)]
    blocks.extend(_read_attachment(path, session_store) for path in paths)
    return Message(MessageRole.USER, blocks)


def paste_image(session_dir: str | Path, *, directory_fd: int | None = None) -> Path:
    """Save a macOS clipboard image in the session directory."""

    if platform.system() != "Darwin":
        raise AttachmentError("image paste is only available on macOS")
    with tempfile.TemporaryDirectory(prefix="zeta-clipboard-") as temporary:
        destination = Path(temporary) / "clipboard.png"
        pngpaste = shutil.which("pngpaste")
        if pngpaste is not None:
            result = subprocess.run(
                [pngpaste, str(destination)],
                capture_output=True,
                check=False,
                env=subprocess_env(),
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
                env=subprocess_env(),
            )
        if result.returncode != 0 or not destination.is_file():
            raise AttachmentError("clipboard does not contain an image")
        try:
            data = destination.read_bytes()
        except OSError as exc:
            raise AttachmentError(f"cannot read clipboard image: {exc}") from exc
        if not data:
            raise AttachmentError("clipboard does not contain an image")
    name = f"clipboard-{uuid4().hex}.png"
    directory = nullcontext(directory_fd) if directory_fd is not None else session_root(Path(session_dir))
    try:
        with directory as pinned_fd:
            write_session_file(pinned_fd, name, data)
    except (OSError, ValueError) as exc:
        raise AttachmentError(f"cannot save clipboard image: {exc}") from exc
    return Path(session_dir) / name


class ComposerAttachmentMixin:
    """Attachment behavior shared by the TUI composition root."""

    def abort_active(self, submission_id: int | None = None) -> None:
        if submission_id is None and not self._submissions.active:
            task = self._active_task
            if task is None or task.done():
                return
            signal = getattr(self, "_standalone_abort_signal", None)
            if signal is not None:
                signal.abort()
            for request in self.pending_approvals:
                if self._approval_policy is not None:
                    self._approval_policy.abort(request.key)
                self.loop.finalize_canceled(request.request_id)
            if self._loop_state == "tool-running" and not self._resuming_tool:
                self._abort_requested = True
            else:
                task.cancel()
            self._loop_state = "interrupted"
            self._invalidate_prompt()
            return
        self._submissions.abort(submission_id)

    def _restore_draft_state(self, draft: Any) -> None:
        self._pending_attachment_tokens = {
            token: _attachment_path(path) for token, path in draft.attachment_tokens
        }
        self._pending_attachments[:] = list(
            dict.fromkeys(self._pending_attachment_tokens.values())
        )
        self._next_image_token = draft.next_image_token

    def _capture_pending_attachment_state(
        self,
    ) -> tuple[tuple[Path, ...], tuple[tuple[str, Path], ...], int]:
        state = (
            tuple(self._pending_attachments),
            tuple(self._pending_attachment_tokens.items()),
            self._next_image_token,
        )
        self._pending_attachments.clear()
        self._pending_attachment_tokens.clear()
        self._next_image_token = 1
        return state

    def _restore_pending_attachment_state(
        self,
        paths: tuple[Path, ...],
        tokens: tuple[tuple[str, Path], ...],
        next_image_token: int,
    ) -> None:
        self._pending_attachments[:] = paths
        self._pending_attachment_tokens = dict(tokens)
        self._next_image_token = next_image_token

    def _release_attachment_paths(self, paths: tuple[Path, ...]) -> None:
        for path in paths:
            self._delete_staged_attachment(path)

    def _attach_draft_state(self, buffer: Buffer, draft: Any) -> None:
        self._restore_draft_state(draft)
        self._draft.attach(
            buffer,
            state_provider=lambda: (
                self._pending_attachment_tokens,
                self._next_image_token,
            ),
        )

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
            path = paste_image(self.loop.store.session_dir, directory_fd=self.loop.store.directory_fd)
            attachment = build_user_message(
                "paste", self.loop.store.cwd, (path,), session_store=self.loop.store
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
            try:
                os.unlink(path.name, dir_fd=self.loop.store.directory_fd)
            except FileNotFoundError:
                pass

    def _pending_paths_for(
        self,
        value: str,
        *,
        pending_attachments: list[Path] | None = None,
        pending_attachment_tokens: dict[str, Path] | None = None,
    ) -> list[Path]:
        attachments = (
            self._pending_attachments
            if pending_attachments is None
            else pending_attachments
        )
        tokens = (
            self._pending_attachment_tokens
            if pending_attachment_tokens is None
            else pending_attachment_tokens
        )
        mapped_paths = set(tokens.values())
        cancelled_paths: set[Path] = set()
        for token, path in tuple(tokens.items()):
            if token not in value:
                del tokens[token]
                cancelled_paths.add(path)
        remaining_mapped_paths = set(tokens.values())
        for path in cancelled_paths - remaining_mapped_paths:
            self._delete_staged_attachment(path)
        attachments[:] = [
            path
            for path in attachments
            if path not in mapped_paths or path in remaining_mapped_paths
        ]

        token_paths = sorted(
            (
                (value.index(token), path)
                for token, path in tokens.items()
            ),
            key=lambda item: item[0],
        )
        selected_paths = [path for _, path in token_paths]
        selected_paths.extend(
            path
            for path in attachments
            if path not in remaining_mapped_paths
        )
        return selected_paths

    def _prepare_user_message(
        self,
        value: str,
        *,
        pending_attachments: list[Path] | None = None,
        pending_attachment_tokens: dict[str, Path] | None = None,
        attachment_value: str | None = None,
    ) -> Message | None:
        source = value if attachment_value is None else attachment_value
        try:
            message = build_user_message(
                value,
                self.loop.store.cwd,
                attachment_value=source,
                session_store=self.loop.store,
            )
        except AttachmentError as exc:
            self._print_system(f"attachment rejected: {exc}")
            return None

        valid_pending: list[Path] = []
        for path in self._pending_paths_for(
            source,
            pending_attachments=pending_attachments,
            pending_attachment_tokens=pending_attachment_tokens,
        ):
            try:
                build_user_message(
                    value,
                    self.loop.store.cwd,
                    (path,),
                    attachment_value=source,
                    session_store=self.loop.store,
                )
            except AttachmentError as exc:
                self._print_system(f"pending attachment dropped: {exc}")
            else:
                valid_pending.append(path)
        attachments = (
            self._pending_attachments
            if pending_attachments is None
            else pending_attachments
        )
        attachments[:] = valid_pending
        if not valid_pending:
            return message
        return build_user_message(
            value,
            self.loop.store.cwd,
            tuple(valid_pending),
            attachment_value=source,
            session_store=self.loop.store,
        )

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
            text,
            message,
            self._pending_attachment_tokens,
            self._next_image_token,
        )

    def _restore_composer(self, value: str) -> None:
        session = self._active_session or self._session
        if session is None:
            return
        session.app.current_buffer.set_document(Document(value, len(value)))

    def _restore_pending_submission(self, submission: Any) -> None:
        session = self._active_session or self._session
        buffer = session.app.current_buffer if session is not None else None
        self._draft.clear_submitted(submission.draft_revision)
        if buffer is not None and buffer.text:
            self._release_attachment_paths(submission.attachment_paths)
            self._print_system(
                f"undo kept the current draft; sent text: {submission.text}"
            )
            self._draft.schedule(buffer.text)
            return
        self._restore_composer(submission.text)
        self._restore_pending_attachment_state(
            submission.attachment_paths,
            submission.attachment_tokens,
            submission.next_image_token,
        )
        self._draft.schedule(
            submission.text,
            attachment_tokens=dict(submission.attachment_tokens),
            next_image_token=submission.next_image_token,
        )

    def undo_sent_turn(self) -> None:
        """Abort the current turn and restore its submitted text once."""

        if not self._submissions.has_pending:
            candidate = self._undo_candidate
            if (
                candidate is None
                or self._active_task is None
                or self._active_task.done()
                or self._loop_state
                not in {"streaming", "compacting", "tool-running", "approval"}
            ):
                self._print_unit(
                    Text("undo unavailable: turn already completed", style=theme.DIM)
                )
                return
            self._undo_candidate = None
            self.abort_active(candidate.submission_id)
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
            self._next_image_token = candidate.next_image_token
            self._draft.schedule(candidate.text)
            return
        self._submissions.undo()

    def _print_user(self, user: str | Message) -> None:
        self._presenter.reset_assistant_unit()
        if isinstance(user, str):
            self._presenter.print_user(
                Text.assemble(("▌ ", theme.USER_ROLE), (user, theme.BODY))
            )
            return
        prompt = next(
            (
                block.text
                for block in user.content
                if isinstance(block, TextContent) and block.path is None
            ),
            "",
        )
        rendered = Text.assemble(("▌ ", theme.USER_ROLE), (prompt, theme.BODY))
        for block in user.content:
            if isinstance(block, TextContent) and block.path is not None:
                label = self._display_attachment_path(Path(block.path))
                rendered.append(
                    f"\n  file · {label} · {block.size or 0} bytes",
                    style="dim",
                )
        self._presenter.print_user(rendered)

    def _consume_macro_receipts(
        self, user_text: str, user_message: Message | None
    ) -> tuple[str, Message | None]:
        if not self._macro_receipts:
            return user_text, user_message
        receipts = "\n".join(self._macro_receipts)
        self._macro_receipts.clear()
        user_text = f"{receipts}\n\n{user_text}"
        if user_message is None:
            return user_text, None
        content = [
            replace(block, text=user_text)
            if isinstance(block, TextContent) and block.path is None
            else block
            for block in user_message.content
        ]
        return user_text, replace(user_message, content=content)

    def _start_turn(
        self,
        user_text: str,
        *,
        user_message: Message | None = None,
        persist_user_message: bool = True,
        submission_id: int | None = None,
        abort_signal: Any | None = None,
    ) -> asyncio.Task[None]:
        self._loop_state = "streaming"
        self._active_turn_submission_id = submission_id
        task = asyncio.create_task(
            self._consume_turn(
                user_text,
                user_message=user_message,
                persist_user_message=persist_user_message,
                abort_signal=abort_signal,
            )
        )
        self._active_task = task
        return task


class SubmissionMixin:
    """Translate composer callbacks into pipeline messages."""

    async def slash_mcp(self, args: str) -> str:
        return await self.loop.slash_mcp(args)

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str:
        return await self.loop.slash_mcp_prompt(name, arguments)

    def _submit_input(self, value: str) -> None:
        if (
            self._submissions._approval_action_for(value) is not None
            and not self._submissions.active
        ):
            task = asyncio.create_task(self._handle_approval_input(value))
            self._active_task = task
            task.add_done_callback(self._clear_approval_task)
            return
        text, steer, passthrough = parse_submission(value)
        if passthrough is not None:
            self._draft.mark_submitted()
            task = asyncio.create_task(self._run_shell_passthrough(passthrough))
            self._active_task = task
            task.add_done_callback(self._clear_approval_task)
            return
        if text is None:
            return
        draft_revision = self._draft.mark_submitted()
        paths, tokens, next_image_token = self._capture_pending_attachment_state()
        self._submissions.submit(
            text,
            draft_revision=draft_revision,
            attachment_paths=paths,
            attachment_tokens=dict(tokens),
            next_image_token=next_image_token,
            steer=steer,
        )

    def _clear_approval_task(self, task: asyncio.Task[Any]) -> None:
        if self._active_task is task:
            self._active_task = None

    async def _handle_prompt_value(self, value: str) -> None:
        if self._input_loop_active:
            self._submit_input(value)
            await asyncio.sleep(0)
            return
        text, steer, passthrough = parse_submission(value)
        if passthrough is not None:
            self._draft.mark_submitted()
            await self._run_shell_passthrough(passthrough)
            return
        if text is None:
            return
        draft_revision = self._draft.mark_submitted()
        paths, tokens, next_image_token = self._capture_pending_attachment_state()
        await self._submissions.submit_text(
            text,
            draft_revision=draft_revision,
            attachment_paths=paths,
            attachment_tokens=dict(tokens),
            next_image_token=next_image_token,
            steer=steer,
        )

    async def _handle_approval_input(self, value: str) -> bool:
        action = self._submissions._approval_action_for(value)
        if action is None:
            return False
        decision, requested_key = action
        await self._submissions.approval_action_wait(decision, requested_key)
        return True

    def _pending_approvals_for_submission(
        self, submission_id: int
    ) -> tuple[ApprovalRequest, ...]:
        owners = self._submissions.approval_owners
        return tuple(
            request
            for request in self.pending_approvals
            if owners.get(request.key) == submission_id
        )

    def _preprocessing_wait_set(self) -> set[asyncio.Task[Any]]:
        return set(self._submissions.preprocessing_tasks.values())

    def _drain_preprocessing(self, done: set[asyncio.Task[Any]]) -> None:
        del done

    def _set_pipeline_task(self, task: asyncio.Task[Any]) -> None:
        self._active_task = task

    def _record_macro_receipt(self, receipt: str) -> None:
        self._macro_receipts.append(receipt)

    def _handle_slash_output(self, output: str) -> None:
        if self._fork_rebuilt:
            self._fork_rebuilt = False
        elif output.startswith("[Image #"):
            self._insert_paste_token(output)
        elif output:
            self._print_system(output)

    @property
    def _preprocessing_tasks(self) -> dict[int, asyncio.Task[Any]]:
        return self._submissions.preprocessing_tasks

    @property
    def _preprocessing_task(self) -> asyncio.Task[Any] | None:
        return self._submissions.preprocessing_task

    @_preprocessing_task.setter
    def _preprocessing_task(self, value: asyncio.Task[Any] | None) -> None:
        del value

    @property
    def _inline_abort_signals(self) -> dict[int, AbortSignal]:
        return self._submissions.inline_abort_signals

    @property
    def _approval_owners(self) -> dict[str | tuple[str, str], int]:
        return self._submissions.approval_owners

    @property
    def _approval_queue(self) -> tuple[tuple[Any, UndoCandidate], ...]:
        return self._submissions.approval_queue

    @property
    def _queued(self) -> tuple[tuple[Any, UndoCandidate], ...]:
        return self._submissions.queued

    def _set_undo_candidate(self, candidate: UndoCandidate) -> None:
        self._undo_candidate = candidate

    def _restore_undo_candidate(self, candidate: UndoCandidate) -> None:
        session = self._active_session or self._session
        buffer = session.app.current_buffer if session is not None else None
        if buffer is not None and buffer.text:
            self._release_attachment_paths(candidate.attachment_paths)
            self._print_system(
                f"undo kept the current draft; sent text: {candidate.text}"
            )
            self._draft.schedule(buffer.text)
            return
        self._restore_composer(candidate.text)
        self._pending_attachments[:] = list(candidate.attachment_paths)
        self._pending_attachment_tokens.clear()
        self._pending_attachment_tokens.update(candidate.attachment_tokens)
        self._next_image_token = candidate.next_image_token
        self._draft.schedule(candidate.text)

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


# --- composer submission intents ------------------------------------------
#
# Steering is the default: while a turn is running, ``Enter`` injects the
# next composer message at the next tool boundary rather than starting a
# fresh turn once the current one ends. A leading backslash marks a message
# as follow-up instead — deliver it after the current turn ends. The prefix
# is stripped from the model-visible text, so ``\\hello`` sends ``hello`` as
# a queued follow-up. Two backslashes at the start escape to a literal
# backslash (``\\\\path`` → ``\\path``). This is the per-message toggle
# called out in the ZETA-82 contract.
FOLLOW_UP_PREFIX = "\\"


def parse_submission(value: str) -> tuple[str | None, bool, str | None]:
    """Classify one composer submission.

    Returns ``(text, steer, passthrough)``:

    * ``text`` — the composer text to submit through the pipeline, or ``None``
      when the input is blank or is a passthrough command (which never
      reaches the model). Whitespace is preserved so undo restores exactly
      what the user typed; the submission pipeline strips through
      :func:`parse_input` before sending to the model.
    * ``steer`` — whether the message should inject at the next tool boundary
      when a turn is running (``True`` is the default); ``False`` restores
      the pre-ZETA-82 "deliver after turn end" behavior.
    * ``passthrough`` — a shell command to run as a one-off receipt (no model
      turn), or ``None`` when this is a normal submission.
    """

    stripped = value.strip()
    if not stripped:
        return None, True, None
    if stripped.startswith("!"):
        rest = stripped[1:]
        if rest.startswith("!"):
            return None, True, rest[1:].strip()
        return None, True, rest.strip()
    if stripped.startswith(FOLLOW_UP_PREFIX):
        # Two backslashes escape to a literal single-backslash message.
        if stripped.startswith(FOLLOW_UP_PREFIX * 2):
            return value.lstrip()[1:], True, None
        # Strip only the marker + one trailing whitespace character so the
        # remaining composer text stays as the user typed it. That keeps undo
        # restore faithful and lets follow-up messages carry their own
        # indentation.
        prefix_index = value.find(FOLLOW_UP_PREFIX)
        rest = value[prefix_index + len(FOLLOW_UP_PREFIX):]
        rest = rest.removeprefix(" ")
        return (rest or None), False, None
    return value, True, None
