"""Composition root for the inline zeta terminal UI."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console, RenderableType
from rich.text import Text

from ..approval import ApprovalPolicy, ApprovalRequest
from ..anthropic import AnthropicBackend
from ..anthropic import AnthropicCredentialStore
from ..codex import CodexBackend
from ..codex import CodexCredentialStore
from ..loop import AgentLoop
from ..session import SessionError, SessionManager, SessionMetadata, env_home
from ..types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
)
from .composer import build_key_bindings, history_for, parse_input
from .render import MarkdownStream, format_status, render_event


DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.4"


def _zeta_home() -> Path:
    return env_home()


class FakeInteractiveBackend(CompletionBackend):
    """Small streaming backend for offline CLI smoke tests."""

    def __init__(self, *, delay: float = 0.03) -> None:
        self.delay = delay
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        prompt = ""
        for message in reversed(messages):
            if message.role is MessageRole.USER:
                prompt = "".join(
                    block.text for block in message.content if isinstance(block, TextContent)
                )
                break
        response = (
            f"you said: {prompt}\n\n"
            "the fake provider is streaming this response offline.\n"
            "try queueing another message while this turn runs."
        )
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for chunk in _chunks(response, 9):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=chunk)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(response)]),
            data={"usage": {"input_tokens": len(prompt), "output_tokens": len(response)}},
        )


def _chunks(value: str, size: int) -> list[str]:
    return [value[index : index + size] for index in range(0, len(value), size)]


def build_backend(
    provider: str,
    model: str | None,
    *,
    home: str | Path | None = None,
) -> tuple[CompletionBackend, str]:
    """Build the selected provider without loading network credentials for fake."""

    auth_home = Path(home) if home is not None else _zeta_home()
    if provider == "fake":
        return FakeInteractiveBackend(), model or "offline"
    if provider == "claude":
        selected_model = model or DEFAULT_CLAUDE_MODEL
        return AnthropicBackend(
            model=selected_model,
            token_store=AnthropicCredentialStore(auth_home / "anthropic-oauth.json"),
        ), selected_model
    if provider == "codex":
        selected_model = model or DEFAULT_CODEX_MODEL
        return CodexBackend(
            model=selected_model,
            token_store=CodexCredentialStore(auth_home / "codex-oauth.json"),
        ), selected_model
    raise ValueError(f"unsupported provider: {provider}")


class _LineBuffer:
    def __init__(self) -> None:
        self.value = ""

    def feed(self, value: str) -> list[str]:
        self.value += value
        lines = self.value.split("\n")
        self.value = lines.pop()
        return lines

    def flush(self) -> list[str]:
        if not self.value:
            return []
        line = self.value
        self.value = ""
        return [line]


class TUIApp:
    """Inline transcript, persistent composer, and follow-up queue."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        provider: str,
        model: str,
        verbose: bool = False,
        console: Console | None = None,
        session: PromptSession[str] | None = None,
        history_path: str | Path | None = None,
        session_manager: SessionManager | None = None,
        session_metadata: SessionMetadata | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self.loop = loop
        self.provider = provider
        self.model = model
        self.verbose = verbose
        self.console = console or Console()
        self._active_task: asyncio.Task[None] | None = None
        self._queued: deque[str] = deque()
        self._exit_requested = False
        self._loop_state = "idle"
        self._usage: dict[str, Any] = {}
        self._assistant_lines = _LineBuffer()
        self._thinking_lines = _LineBuffer()
        self._markdown_stream = MarkdownStream()
        self._stream_kind: str | None = None
        self._partial = ""
        self._session = session
        self._history_path = Path(history_path) if history_path else _zeta_home() / "history"
        self._session_manager = session_manager
        self._session_metadata = session_metadata
        self._approval_policy = approval_policy

    @property
    def queued_messages(self) -> tuple[str, ...]:
        return tuple(self._queued)

    @property
    def active(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    @property
    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        if self._approval_policy is None:
            return ()
        return tuple(self._approval_policy.pending_requests())

    def _present_pending_approvals(self) -> None:
        for request in self.pending_approvals:
            arguments = json.dumps(request.tool_call.arguments, sort_keys=True)
            self._print(
                Text(
                    f"[approval pending] {request.request_id}: "
                    f"{request.tool_call.name} {arguments}; "
                    f"type approve {request.request_id} or deny {request.request_id}",
                    style="yellow",
                )
            )

    async def _handle_approval_input(self, value: str) -> bool:
        parts = value.split(maxsplit=1)
        if not parts or parts[0] not in {"approve", "deny"}:
            return False
        pending = self.pending_approvals
        if not pending:
            self._print(Text("[approval] no pending requests", style="dim"))
            return True
        if len(parts) != 2:
            self._print(Text(f"[approval] use {parts[0]} <request-id>", style="yellow"))
            return True
        request_id = parts[1].strip()
        if request_id not in {request.request_id for request in pending}:
            self._print(Text(f"[approval] unknown request: {request_id}", style="yellow"))
            return True
        resolved = (
            self._approval_policy.approve(request_id)
            if parts[0] == "approve"
            else self._approval_policy.deny(request_id)
        )
        if resolved:
            self._print(Text(f"[approval] {parts[0]}d {request_id}", style="green"))
            resume_task = asyncio.create_task(
                self.loop.resume_pending_tool(request_id)
            )
            self._active_task = resume_task
            try:
                result = await resume_task
            except asyncio.CancelledError:
                self._print(Text("[aborted]", style="yellow"))
                return True
            finally:
                if self._active_task is resume_task:
                    self._active_task = None
            request = next(
                request for request in pending if request.request_id == request_id
            )
            if result is not None:
                self._print(
                    render_event(
                        StreamEvent(
                            StreamEventType.TOOL_EXECUTION_END,
                            tool_call=request.tool_call,
                            tool_result=result,
                        )
                    )
                )
        self._present_pending_approvals()
        return True

    def _make_session(self) -> PromptSession[str]:
        bindings = build_key_bindings(
            on_interrupt=self.abort_active,
            on_exit=self.request_exit,
        )
        return PromptSession(
            history=history_for(self._history_path),
            key_bindings=bindings,
            multiline=True,
            style=Style.from_dict(
                {
                    "prompt": "#c6ff4a bold",
                    "bottom-toolbar": "#0b0c0a #c6ff4a",
                }
            ),
        )

    def request_exit(self) -> None:
        self._exit_requested = True
        self.abort_active()

    def abort_active(self) -> None:
        if self._active_task is not None and not self._active_task.done():
            self.loop.abort()
            self._active_task.cancel()

    def _status_toolbar(self) -> FormattedText:
        status = format_status(
            self.provider,
            self.model,
            self._loop_state,
            self._usage,
            self._partial,
        )
        return FormattedText([("class:bottom-toolbar", status.plain)])

    def _print(self, renderable: RenderableType | None) -> None:
        if renderable is not None:
            self.console.print(renderable)

    def _print_committed(self, lines: list[str], *, thinking: bool = False) -> None:
        if thinking:
            for line in lines:
                self._print(Text(f"[thinking] {line}", style="dim italic"))
            return
        for line in lines:
            for renderable in self._markdown_stream.consume(line):
                self._print(renderable)

    def _update_usage(self, event: StreamEvent) -> None:
        usage = event.data.get("usage")
        if isinstance(usage, dict):
            self._usage.update(usage)

    @staticmethod
    def _invalidate_prompt() -> None:
        get_app().invalidate()

    def _flush_stream_kind(self) -> None:
        if self._stream_kind == "thinking":
            self._print_committed(self._thinking_lines.flush(), thinking=True)
        elif self._stream_kind == "assistant":
            self._print_committed(self._assistant_lines.flush())
        self._stream_kind = None
        self._partial = ""

    def _flush_markdown(self) -> None:
        for renderable in self._markdown_stream.flush():
            self._print(renderable)

    def _flush_pending_stream(self) -> None:
        self._flush_stream_kind()
        self._flush_markdown()

    def _consume_text(self, event: StreamEvent) -> None:
        value = event.delta
        thinking = False
        if isinstance(event.content, TextContent):
            value = event.content.text
        elif isinstance(event.content, ThinkingContent):
            value = event.content.text
            thinking = True
        if not value:
            return
        stream_kind = "thinking" if thinking else "assistant"
        if self._stream_kind is not None and self._stream_kind != stream_kind:
            self._flush_pending_stream()
        self._stream_kind = stream_kind
        buffer = self._thinking_lines if thinking else self._assistant_lines
        committed = buffer.feed(value)
        self._partial = buffer.value
        self._print_committed(committed, thinking=thinking)

    def _finish_stream(self) -> None:
        self._flush_pending_stream()
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._reset_stream_buffers()
        self._partial = ""

    def _reset_stream_buffers(self) -> None:
        self._assistant_lines.value = ""
        self._thinking_lines.value = ""

    def _print_user(self, user_text: str) -> None:
        self.console.print(Text(f"[user] {user_text}", style="bold"))

    def _start_queued_turn(self) -> None:
        if self._queued:
            user_text = self._queued.popleft()
            self._print_user(user_text)
            self.console.print(Text("[queued]", style="dim"))
            self._start_turn(user_text)

    def _prepare_stream_event(self, event: StreamEvent) -> None:
        if event.type is not StreamEventType.MESSAGE_UPDATE:
            self._flush_pending_stream()
            return
        if isinstance(event.content, ThinkingContent):
            incoming_kind = "thinking"
        elif isinstance(event.content, TextContent) or event.delta is not None:
            incoming_kind = "assistant"
        else:
            incoming_kind = None
        if incoming_kind != self._stream_kind and (
            incoming_kind is not None or self._stream_kind is not None
        ):
            self._flush_pending_stream()

    async def _consume_turn(self, user_text: str) -> None:
        self._loop_state = "streaming"
        try:
            async for event in self.loop.run_turn(user_text):
                self._update_usage(event)
                self._prepare_stream_event(event)
                if self.verbose:
                    self._print(Text(json.dumps(event.to_dict(), sort_keys=True), style="dim"))
                if event.type is StreamEventType.MESSAGE_UPDATE:
                    self._consume_text(event)
                    self._invalidate_prompt()
                    continue
                if event.type is StreamEventType.TOOL_EXECUTION_START:
                    self._reset_stream_state()
                    self._loop_state = "tool-running"
                    self._present_pending_approvals()
                elif event.type is StreamEventType.TOOL_EXECUTION_END:
                    self._loop_state = "streaming"
                elif event.type is StreamEventType.AGENT_END:
                    self._reset_stream_state()
                    self._loop_state = "idle"
                self._print(render_event(event))
                self._invalidate_prompt()
        except asyncio.CancelledError:
            self._finish_stream()
            self._loop_state = "idle"
            self._print(Text("[aborted]", style="yellow"))
            raise
        except Exception as exc:
            self._finish_stream()
            self._loop_state = "idle"
            self._print(Text(f"[error] {exc}", style="bold red"))

    async def _read_prompt(self, session: PromptSession[str]) -> str | None:
        try:
            value = await session.prompt_async(
                "you > ",
                bottom_toolbar=self._status_toolbar,
            )
        except EOFError:
            return None
        return value

    async def run(self, session: PromptSession[str] | None = None) -> None:
        """Run until Ctrl-D or an exit request."""

        session = session or self._session or self._make_session()
        self._present_pending_approvals()
        prompt_task: asyncio.Task[str | None] | None = asyncio.create_task(
            self._read_prompt(session)
        )
        try:
            while prompt_task is not None and not self._exit_requested:
                wait_for: set[asyncio.Task[Any]] = {prompt_task}
                if self._active_task is not None:
                    wait_for.add(self._active_task)
                done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

                if self._active_task is not None and self._active_task in done:
                    try:
                        await self._active_task
                    except asyncio.CancelledError:
                        pass
                    self._active_task = None
                    if self._queued:
                        self._start_queued_turn()

                if prompt_task in done:
                    value = await prompt_task
                    if value is None:
                        break
                    parsed = parse_input(value)
                    if parsed is not None and not self._exit_requested:
                        if await self._handle_approval_input(parsed):
                            pass
                        elif self.pending_approvals:
                            self._present_pending_approvals()
                        elif self.active:
                            self._queued.append(parsed)
                        else:
                            self._print_user(parsed)
                            self._start_turn(parsed)
                    if self._exit_requested:
                        break
                    prompt_task = asyncio.create_task(self._read_prompt(session))
        finally:
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()
                await asyncio.gather(prompt_task, return_exceptions=True)
            if self._active_task is not None and not self._active_task.done():
                self._active_task.cancel()
                await asyncio.gather(self._active_task, return_exceptions=True)

    def _start_turn(self, user_text: str) -> None:
        self._active_task = asyncio.create_task(self._consume_turn(user_text))


def create_app(args: argparse.Namespace) -> TUIApp:
    home = _zeta_home()
    manager = SessionManager(home)
    continue_session = getattr(args, "continue_session", False)
    resume_id = getattr(args, "resume", None)
    force_provider = getattr(args, "force_provider", False)
    resuming = continue_session or resume_id is not None
    if force_provider and not resuming:
        raise SessionError("--force-provider requires --continue or --resume")
    if force_provider and args.model is None:
        raise SessionError("--force-provider requires --model")

    if resuming:
        if resume_id is not None:
            opened = manager.open(resume_id)
        else:
            recent = manager.find_most_recent(cwd=Path.cwd())
            opened = manager.open(recent.session_id)
        metadata = opened.metadata
        provider_override = args.provider
        model_override = args.model
        mismatches = []
        if provider_override is not None and provider_override != metadata.provider:
            mismatches.append(
                f"provider {provider_override!r} does not match {metadata.provider!r}"
            )
        if model_override is not None and model_override != metadata.model:
            mismatches.append(
                f"model {model_override!r} does not match {metadata.model!r}"
            )
        if mismatches and not force_provider:
            raise SessionError(
                f"session override rejected: {'; '.join(mismatches)}; "
                "use --force-provider to override"
            )
        if mismatches and metadata.override_audit:
            raise SessionError(
                "session override already committed by a prior resume: "
                f"{metadata.override_audit[-1]}"
            )
        provider = provider_override or metadata.provider
        model = model_override or metadata.model
        store = opened.store
    else:
        provider = args.provider or "fake"
        backend, selected_model = build_backend(provider, args.model, home=home)
        opened = manager.create(
            provider=provider,
            model=selected_model,
            cwd=Path.cwd(),
        )
        metadata = opened.metadata
        store = opened.store
    if resuming:
        backend, selected_model = build_backend(provider, model, home=home)
    approval_policy = ApprovalPolicy(store=store)
    pending_override = None
    if resuming and mismatches:
        pending_override = (
            provider if provider != metadata.provider else None,
            model if model != metadata.model else None,
        )

    def completion_success() -> None:
        nonlocal pending_override
        if pending_override is not None:
            manager.record_override(
                metadata,
                provider=pending_override[0],
                model=pending_override[1],
            )
            pending_override = None
            return
        manager.touch(metadata)

    loop = AgentLoop(
        backend,
        store,
        approval_policy=approval_policy,
        token_budget=metadata.compaction_budget,
        retained_tail=metadata.retained_tail,
        on_completion_success=completion_success,
    )
    return TUIApp(
        loop,
        provider=provider,
        model=selected_model,
        verbose=args.verbose,
        history_path=home / "history",
        session_manager=manager,
        session_metadata=metadata,
        approval_policy=approval_policy,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="chat with the zeta harness")
    parser.add_argument(
        "--provider",
        choices=("fake", "claude", "codex"),
        help="completion provider",
    )
    parser.add_argument("--model", help="provider model override")
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--continue",
        "-c",
        dest="continue_session",
        action="store_true",
        help="resume the most recent session in this directory",
    )
    session_group.add_argument("--resume", help="resume a session by id")
    parser.add_argument(
        "--force-provider",
        action="store_true",
        help="allow provider or model overrides during resume",
    )
    parser.add_argument("--verbose", action="store_true", help="show raw stream events")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force_provider and args.model is None:
        parser.error("--force-provider requires --model")
    try:
        app = create_app(args)
    except SessionError as exc:
        parser.error(str(exc))
    with patch_stdout(raw=True):
        asyncio.run(app.run())
    return 0


__all__ = [
    "FakeInteractiveBackend",
    "TUIApp",
    "build_backend",
    "build_parser",
    "create_app",
    "main",
]
