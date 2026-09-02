"""Prompt-toolkit composer setup and input parsing."""

from __future__ import annotations

import asyncio
import base64
import json
import platform
import re
import shutil
import subprocess
from collections.abc import Iterator, Mapping
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

from ..core.slash import CustomCommand, SlashCommandRegistry
from ..tools.exec import (
    forget_macro_display,
    register_macro_display,
    run_exec_macro,
    run_inline_shell_batch,
)
from ..types import (
    ErrorInfo,
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    image_signature_matches,
)
from .key_bindings import (
    FullScreenPromptSession,
    VimCursorShapeConfig,
    build_key_bindings,
)
from .render import is_retryable_error, render_event
from .theme import BODY, CHROME, DIM, ERROR, USER_ROLE

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
        user_text, user_message = self._consume_macro_receipts(
            user_text, user_message
        )
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

    @classmethod
    def from_message(
        cls,
        text: str,
        message: Message,
        attachment_tokens: Mapping[str, Path],
        next_image_token: int,
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
        return cls(text, paths, tokens, next_image_token)


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
    *,
    attachment_value: str | None = None,
) -> Message:
    """Resolve references into one message, deduplicating resolved paths."""

    paths: list[Path] = []
    source = value if attachment_value is None else attachment_value
    for ref in attachment_refs(source, base_dir):
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

    def abort_active(self) -> None:
        if self._inline_abort_signal is not None:
            self._inline_abort_signal.abort()
            for request in self.pending_approvals:
                self._abort_approval(request.key)
            self._invalidate_prompt()
            return
        if self._active_task is not None and not self._active_task.done():
            macro_running = self._macro_abort_signal is not None
            if macro_running:
                self._abort_macro()
            else:
                self.loop.abort()
                for request in self.pending_approvals:
                    self._abort_approval(request.key)
                    self.loop.finalize_canceled(request.request_id)
            if macro_running or (
                self._loop_state == "tool-running" and not self._resuming_tool
            ):
                self._abort_requested = True
            else:
                self._active_task.cancel()
            self._loop_state = "interrupted"
            self._invalidate_prompt()
        elif self.loop.background_children_running:
            self.loop.abort()
            self._invalidate_prompt()

    def _restore_draft_state(self, draft: Any) -> None:
        self._pending_attachment_tokens = {
            token: path.resolve() for token, path in draft.attachment_tokens
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
        )

    def _clear_pending_attachments(self) -> None:
        mapped_paths = set(self._pending_attachment_tokens.values())
        for path in self._pending_attachments:
            if path not in mapped_paths:
                self._delete_staged_attachment(path)
        self._pending_attachments.clear()
        self._pending_attachment_tokens.clear()
        self._next_image_token = 1

    async def _resolve_inline_shell(self, commands: tuple[str, ...]) -> tuple[str, ...]:
        """Resolve template shell spans through the exec tool path."""

        signal = self.loop.tool_registry.abort_signal.registry.new_generation()
        self._inline_abort_signal = signal

        def lifecycle_sink(kind: str, call: ToolCall) -> None:
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
            }.get(kind)
            if event_type is None:
                return
            self._handle_tool_event(
                StreamEvent(
                    event_type,
                    tool_call=call,
                    data={"inline_shell": True},
                )
            )
            if kind == "approval_start":
                asyncio.get_running_loop().call_soon(self._present_pending_approvals)
            self._invalidate_prompt()

        try:
            return await run_inline_shell_batch(
                self.loop.tool_registry,
                commands,
                lifecycle_sink=lifecycle_sink,
                abort_signal=signal,
            )
        finally:
            self._inline_abort_signal = None

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

    def undo_sent_turn(self) -> None:
        """Abort the current turn and restore its submitted text once."""

        pending_submission = self._submissions.cancel_current()
        if pending_submission is not None:
            session = self._active_session or self._session
            buffer = session.app.current_buffer if session is not None else None
            self._draft.clear_submitted(pending_submission.draft_revision)
            if buffer is not None and buffer.text:
                self._release_attachment_paths(pending_submission.attachment_paths)
                self._print_system(
                    f"undo kept the current draft; sent text: {pending_submission.text}"
                )
                self._draft.schedule(buffer.text)
                return
            self._restore_composer(pending_submission.text)
            self._restore_pending_attachment_state(
                pending_submission.attachment_paths,
                pending_submission.attachment_tokens,
                pending_submission.next_image_token,
            )
            self._draft.schedule(
                pending_submission.text,
                attachment_tokens=dict(pending_submission.attachment_tokens),
                next_image_token=pending_submission.next_image_token,
            )
            return

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

    def _print_user(self, user: str | Message) -> None:
        self._presenter.reset_assistant_unit()
        if isinstance(user, str):
            self._presenter.print_user(Text.assemble(("▌ ", USER_ROLE), (user, BODY)))
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
        self._presenter.print_user(rendered)

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
    ) -> None:
        self._loop_state = "streaming"
        self._active_task = asyncio.create_task(
            self._consume_turn(
                user_text,
                user_message=user_message,
                persist_user_message=persist_user_message,
            )
        )

    async def slash_exec_macro(self, command: CustomCommand, args: str) -> str:
        """Run one custom shell macro through the normal exec safety path."""

        if self.active:
            return "macro unavailable while a turn is running"
        call = ToolCall(
            f"macro-{uuid4().hex}",
            "exec",
            {
                "command": command.render_exec(args),
                "timeout": command.timeout,
            },
        )
        register_macro_display(
            call.id,
            command=command.render(args),
            argv=tuple(args.split()),
        )
        abort_signal = self.loop.tool_registry.abort_signal.registry.new_generation()
        self._macro_abort_signal = abort_signal
        self._macro_call_id = call.id
        task = asyncio.create_task(self._run_exec_macro(command, call))
        self._active_task = task
        self._loop_state = "tool-running"
        if not self._input_loop_active:
            try:
                await asyncio.shield(task)
            finally:
                if self._active_task is task:
                    self._active_task = None
        return ""

    def _abort_macro(self) -> None:
        signal = self._macro_abort_signal
        if signal is None:
            return
        signal.abort()
        for request in self.pending_approvals:
            if request.tool_call.id == self._macro_call_id:
                self._abort_approval(request.key)

    async def _run_exec_macro(self, command: CustomCommand, call: ToolCall) -> None:
        log_path = self.loop.store.session_dir / f"macro-{call.id[6:]}.log"

        def lifecycle_sink(kind: str) -> None:
            event_type = {
                "approval_start": StreamEventType.TOOL_APPROVAL_START,
                "approval_end": StreamEventType.TOOL_APPROVAL_END,
                "execution_start": StreamEventType.TOOL_EXECUTION_START,
            }.get(kind)
            if event_type is None:
                return
            self._handle_tool_event(
                StreamEvent(
                    event_type,
                    tool_call=call,
                    data={"macro": command.name},
                )
            )
            if kind == "approval_start":
                asyncio.get_running_loop().call_soon(self._present_pending_approvals)
            self._invalidate_prompt()

        def stream_sink(event: StreamEvent) -> None:
            event.data["macro"] = command.name
            self._handle_tool_event(event)
            self._invalidate_prompt()

        canceled = False
        try:
            result = await run_exec_macro(
                self.loop.tool_registry,
                call,
                log_path,
                stream_sink=stream_sink,
                lifecycle_sink=lifecycle_sink,
                abort_signal=self._macro_abort_signal,
                background=command.background,
            )
        except asyncio.CancelledError:
            result = ToolResult(call.id, "tool execution canceled", True)
            canceled = True
        finally:
            if self._approval_policy is not None:
                self._approval_policy.forget_ephemeral(call.id)
            forget_macro_display(call.id)
        self._handle_tool_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=result,
                data={"macro": command.name},
            )
        )
        structured = result.structured_content or {}
        if result.content == "tool execution canceled":
            status = "canceled"
        elif result.content == "tool execution denied":
            status = "denied"
        elif structured.get("status") == "running":
            status = "running"
        elif structured.get("timed_out") is True:
            status = "timeout"
        else:
            exit_code = (
                result.structured_content.get("exit_code")
                if result.structured_content is not None
                else None
            )
            status = f"exit {exit_code}" if exit_code is not None else "failed"
        if status == "running":
            task_id = structured.get("task_id")
            if isinstance(task_id, str):
                self._watch_background_macro(command, call, task_id, log_path)
        else:
            self._macro_receipts.append(f"ran /{command.name}, {status}")
        if self._macro_call_id == call.id:
            self._macro_abort_signal = None
            self._macro_call_id = None
        self._loop_state = "idle"
        self._streaming = False
        if canceled:
            raise asyncio.CancelledError

    def _watch_background_macro(
        self,
        command: CustomCommand,
        call: ToolCall,
        task_id: str,
        log_path: Path,
    ) -> None:
        instance_id = f"macro:{call.id}"

        async def watch() -> None:
            try:
                status_data = await self.loop.tool_registry.background_tasks.wait(task_id)
                note = status_data.get("note")
                exit_code = status_data.get("exit_code")
                if isinstance(note, str) and note.startswith("task killed"):
                    status = "canceled"
                    text = f"background macro /{command.name} canceled"
                elif exit_code == 0:
                    status = "completed"
                    text = f"background macro /{command.name} completed"
                else:
                    status = "error"
                    text = f"background macro /{command.name} failed with exit {exit_code}"
                self.loop.store.append_agent_notification(
                    instance_id,
                    child_session_path=str(log_path),
                    description=f"/{command.name}",
                    status=status,
                    text=f"{text}; log {log_path}",
                )
            finally:
                self.loop._background_owner.unregister(instance_id)

        watcher = asyncio.create_task(watch())
        self.loop._background_owner.register(
            instance_id,
            lambda: asyncio.create_task(
                self.loop.tool_registry.background_tasks.kill(task_id)
            ),
            watcher,
        )


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
