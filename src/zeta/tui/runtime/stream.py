"""stream event handling for the full-screen terminal UI."""
from __future__ import annotations

import time

from prompt_toolkit.application import get_app
from rich.text import Text

from ...protocol.types import (
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    assistant_text,
)
from .. import render, theme
from ..bootstrap import background_notice
from ..render import render_markdown, render_thought, render_thought_live_delta
from ..transcript import stream_key


class StreamEventMixin:
    """Render model, tool, and background stream events."""

    def _handle_tool_event(self, event: StreamEvent) -> bool:
        self._refresh_decisions_for_event(event)
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            if event.tool_call is not None and not event.data.get("inline_shell"):
                self._submissions.notify_approval_started(
                    event.tool_call,
                    event.data.get("submission_id", self._active_turn_submission_id),
                )
            if self._agent_navigation.child_view_active:
                self._agent_navigation.exit_navigation()
            self._reset_stream_state()
            self._loop_state = "approval"
            self._repaint_accounts_picker()
            self._present_pending_approvals()
            return False
        if event.type is StreamEventType.TOOL_APPROVAL_END:
            if event.tool_call is not None and not event.data.get("inline_shell"):
                self._submissions.notify_approval_finished(event.tool_call)
            self._loop_state = "streaming"
            self._repaint_accounts_picker()
            self._present_pending_approvals()
            return False
        if event.data.get("agent_instance_id") is not None:
            return False
        if event.type is StreamEventType.TOOL_EXECUTION_START:
            self._agent_navigation.request_refresh()
            self._reset_stream_state()
            self._loop_state = "tool-running"
            presentation = self._presenter.handle_tool_event(
                event,
                aborted=False,
            )
            if presentation is not None and presentation.visible_output:
                self._turn_had_visible_output = True
            return False
        if event.type is StreamEventType.TOOL_EXECUTION_UPDATE:
            presentation = self._presenter.handle_tool_event(
                event,
                aborted=False,
            )
            if presentation is not None and presentation.visible_output:
                self._turn_had_visible_output = True
            return False
        if event.type is not StreamEventType.TOOL_EXECUTION_END:
            return False

        aborted = self._abort_requested or self._loop_state == "interrupted"
        self._loop_state = (
            "interrupted"
            if aborted
            else ("idle" if self._resuming_tool else "streaming")
        )
        presentation = self._presenter.handle_tool_event(
            event,
            aborted=aborted,
        )
        if presentation is not None and presentation.visible_output:
            self._turn_had_visible_output = True
        stop_after_tool = presentation is not None and presentation.stop_after_tool
        self._abort_requested = False
        return stop_after_tool

    def _handle_background_notice(self, notice: object) -> None:
        self._agent_navigation.request_refresh()
        if not self._defer_startup_notice(notice):
            background_notice(self, notice)

    def _handle_background_event(self, event: StreamEvent) -> None:
        if not self._defer_startup_runtime_event(event):
            self._handle_background_event_now(event)

    def _handle_background_event_now(self, event: StreamEvent) -> None:
        """Render child progress while keeping completion notices at turn boundaries."""
        self._refresh_decisions_for_event(event)
        if event.type in render.BACKGROUND_EVENTS_RENDERED_IMMEDIATELY:
            rendered = self._presenter.render_event(event)
            if rendered is not None:
                self._print_unit(rendered)
                self._invalidate_background_prompt()
            return
        if event.type in {
            StreamEventType.TOOL_APPROVAL_START,
            StreamEventType.TOOL_APPROVAL_END,
        }:
            self._handle_tool_event(event)
            self._finish_background_invalidation()
            return
        if event.data.get("agent_instance_id") is not None:
            return
        if event.type in {
            StreamEventType.TOOL_EXECUTION_START,
            StreamEventType.TOOL_EXECUTION_UPDATE,
            StreamEventType.TOOL_EXECUTION_END,
        }:
            if event.type is StreamEventType.TOOL_EXECUTION_START:
                self._agent_navigation.request_refresh()
            presentation = self._presenter.handle_tool_event(event, aborted=False)
            if presentation is not None and presentation.visible_output:
                if event.type is StreamEventType.TOOL_EXECUTION_END:
                    self._finish_background_invalidation()
                else:
                    self._invalidate_background_prompt()

    def _schedule_background_wake(self) -> None:
        if self._closed:
            return
        if self._startup_replay_active:
            self._startup_wake_pending = True
            return
        self._submissions.wake()

    def _discard_tool_region(self) -> None:
        self._presenter.discard_tool_region()

    @property
    def _tool_region(self):
        return self._presenter.tool_region

    def _print_committed(self, lines: list[str], *, thinking: bool = False) -> None:
        value = "\n".join(lines)
        if thinking:
            if value:
                self._presenter.finish_thinking(
                    render_thought(
                        value, self._thinking_duration, provider=self.provider
                    )
                )
                self._turn_had_visible_output = True
            return
        if value:
            self._assistant_chunks.append(value)
            self._presenter.append_assistant(value)
            self._turn_had_visible_output |= bool(value.strip())

    def _update_usage(self, event: StreamEvent) -> None:
        usage = event.data.get("usage")
        if isinstance(usage, dict):
            self._usage.update(usage)

    def _invalidate_prompt(self) -> None:
        # Suspended behind a discussion fork: keep working but do not paint over
        # the fork that owns the terminal (output rebuilds from the store later).
        if self._fork_controller is not None and not self._fork_controller.is_visible(
            self
        ):
            return
        get_app().invalidate()

    def _flush_stream_kind(self, *, preserve_inline: bool = False) -> None:
        if (
            self._stream_kind in {"thinking", "redacted-thinking"}
            and self._thinking_chunks
        ):
            self._print_committed(["".join(self._thinking_chunks)], thinking=True)
        elif self._stream_kind == "assistant" and self._assistant_chunks:
            self._presenter.finish_assistant(
                Text("".join(self._assistant_chunks), style=theme.BODY),
                preserve_inline=preserve_inline,
            )
        self._stream_kind = self._stream_identity = None
        self._partial = ""
        self._thinking_chunks.clear()
        self._thinking_duration = self._thinking_started_at = None

    def _flush_markdown(self) -> None:
        if self._assistant_chunks:
            self._presenter.finish_assistant(
                render_markdown("".join(self._assistant_chunks))
            )

    def _finish_message(self, event: StreamEvent) -> None:
        if self._stream_kind in {"thinking", "redacted-thinking"}:
            self._flush_stream_kind()
        value = (
            assistant_text(event.message)
            if event.message is not None
            else "".join(self._assistant_chunks)
        )
        self._presenter.finish_assistant_message(
            render_markdown(value) if value else None
        )
        if value:
            self._turn_had_visible_output |= bool(value.strip())
        self._stream_kind = self._stream_identity = None
        self._reset_stream_buffers()

    def _flush_pending_stream(self) -> None:
        self._flush_stream_kind()
        self._presenter.reset_assistant_unit()

    def _consume_text(self, event: StreamEvent) -> None:
        incoming_kind, incoming_identity = stream_key(event)
        if incoming_kind is None:
            return
        if self._stream_kind is not None and (incoming_kind, incoming_identity) != (
            self._stream_kind,
            self._stream_identity,
        ):
            self._flush_pending_stream()
        redacted = incoming_kind == "redacted-thinking"
        thinking = redacted
        value = "redacted" if thinking else event.delta
        if isinstance(event.content, TextContent):
            value = event.content.text
        elif isinstance(event.content, ThinkingContent):
            value = event.content.text
            thinking = True
        if not value:
            return
        self._streaming = True
        stream_kind = incoming_kind
        assert stream_kind is not None
        self._stream_kind = stream_kind
        self._stream_identity = incoming_identity
        if thinking:
            if self._thinking_started_at is None:
                self._thinking_started_at = time.monotonic()
            self._thinking_chunks.append(value)
            self._thinking_duration = max(
                0.0, time.monotonic() - self._thinking_started_at
            )
            rendered, self._partial = render_thought_live_delta(
                value, self._partial, provider=self.provider
            )
            if rendered.plain:
                self._presenter.append_thinking(rendered.plain)
            return
        self._assistant_chunks.append(value)
        self._presenter.append_assistant(value)

    def _reset_stream_state(self) -> None:
        self._stream_kind = self._stream_identity = None
        self._partial = ""
        self._thinking_chunks.clear()
        self._thinking_duration = self._thinking_started_at = None
        self._streaming = False

    def _reset_stream_buffers(self) -> None:
        self._assistant_chunks.clear()
        self._thinking_chunks.clear()

    def _prepare_stream_event(self, event: StreamEvent) -> None:
        if event.type is StreamEventType.MESSAGE_START:
            self._flush_pending_stream()
            self._presenter.reset_assistant_message()
            self._assistant_chunks.clear()
            return
        if event.type is StreamEventType.MESSAGE_END:
            return
        if event.type is StreamEventType.ERROR:
            self._flush_stream_kind(preserve_inline=True)
            self._presenter.reset_assistant_unit()
            return
        if event.type is StreamEventType.ASSISTANT_RESET:
            self._presenter.discard_assistant_message()
            self._reset_stream_state()
            self._reset_stream_buffers()
            return
        if event.type is StreamEventType.RETRY:
            return
        if event.type is not StreamEventType.MESSAGE_UPDATE:
            self._flush_pending_stream()
            return
        incoming_kind, incoming_identity = stream_key(event)
        if incoming_kind is None:
            return
        current = self._stream_kind, self._stream_identity
        if (incoming_kind, incoming_identity) != current:
            self._flush_pending_stream()
