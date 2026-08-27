"""Checkpoint commands and transcript rebuilding for the terminal UI."""

from __future__ import annotations

from datetime import UTC, datetime

from ..types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolUseContent,
    assistant_text,
)
from .render import render_event, render_markdown


def _age(created_at: str) -> str:
    try:
        seconds = max(
            0,
            int(
                (datetime.now(UTC) - datetime.fromisoformat(created_at)).total_seconds()
            ),
        )
    except ValueError:
        return "age unknown"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3_600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3_600}h ago"
    return f"{seconds // 86_400}d ago"


class CheckpointTranscriptMixin:
    """Add checkpoint commands and active-branch transcript rebuilding."""

    def slash_checkpoint(self, args: str) -> str:
        if self.active or self.loop.store.turn_in_flight():
            return "checkpoint unavailable while a turn is running"
        try:
            entry = self.loop.store.append_checkpoint(args.strip() or None)
        except ValueError as exc:
            return f"checkpoint failed: {exc}"
        return f"checkpoint '{entry.data['label']}' at seq {entry.seq}"

    async def slash_compact(self) -> str:
        """Force one compaction through the context assembler."""

        if self.active:
            return "compact unavailable while a turn is running"
        before = self.loop.store.compaction_marker_count()
        try:
            context = await self.loop.context_assembler.assemble_context(
                backend=self.loop.backend,
                force=True,
            )
        except Exception as exc:
            return f"compact failed: {exc}"
        after = self.loop.store.compaction_marker_count()
        if after == before:
            return "compact: nothing to compact"
        marker = next(
            entry
            for entry in reversed(self.loop.store.replay())
            if entry.type == "compaction"
        )
        return (
            "compacted entries "
            f"{marker.data['source_seq_start']}–{marker.data['source_seq_end']}; "
            f"tokens after: {context.token_count}"
        )

    def slash_fork(self, args: str) -> str:
        if self.active or self.loop.store.turn_in_flight():
            return "fork unavailable while a turn is running"
        selector = args.strip()
        if not selector:
            checkpoints = self.loop.store.list_checkpoints()
            if not checkpoints:
                return "no checkpoints on the active branch; use /checkpoint [label]"
            lines = ["checkpoints on the active branch:"]
            for entry, preview in checkpoints:
                next_message = preview or "(no message after checkpoint)"
                lines.append(
                    f"seq {entry.seq} · {entry.data['label']} · "
                    f"{_age(entry.data['created_at'])} · {next_message}"
                )
            return "\n".join(lines)
        try:
            entry = self.loop.store.append_fork(selector)
        except ValueError as exc:
            return f"fork failed: {exc}"
        self._rebuild_transcript()
        self._fork_rebuilt = True
        return f"forked to checkpoint '{entry.data['label']}' at seq {entry.data['from_seq']}"

    def _rebuild_transcript(self) -> None:
        """Re-render the visible transcript from the active durable branch."""

        self._presenter.clear()
        tool_calls: dict[str, ToolCall] = {}
        for entry in self.loop.store.replay():
            if entry.type == "checkpoint":
                self._print_system(
                    f"checkpoint '{entry.data['label']}' at seq {entry.seq}"
                )
                continue
            if entry.type == "fork":
                self._print_system(
                    f"forked to checkpoint '{entry.data['label']}' at seq "
                    f"{entry.data['from_seq']}"
                )
                continue
            if entry.type == "compaction":
                self._print_system(
                    "[compaction marker: entries "
                    f"{entry.data['source_seq_start']}–{entry.data['source_seq_end']}]"
                )
                continue
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            if message.role is MessageRole.USER:
                self._print_user(message)
            elif message.role is MessageRole.ASSISTANT:
                text = assistant_text(message)
                if text:
                    self._print_unit(render_markdown(text))
                for block in message.content:
                    if isinstance(block, ToolUseContent):
                        tool_calls[block.tool_call.id] = block.tool_call
            elif (
                message.role is MessageRole.TOOL_RESULT
                and message.tool_result is not None
            ):
                self._print_unit(
                    render_event(
                        StreamEvent(
                            StreamEventType.TOOL_EXECUTION_END,
                            tool_call=tool_calls.get(message.tool_result.tool_call_id),
                            tool_result=message.tool_result,
                        )
                    )
                )
