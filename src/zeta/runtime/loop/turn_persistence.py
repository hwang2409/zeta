"""Persist completed and interrupted agent-turn state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...agent.durable import durable_message
from ...agent.receipt import finalize_agent_results
from ...protocol.types import (
    ASSISTANT_RESPONSE_ABORTED,
    ASSISTANT_RESPONSE_FAILED,
    ASSISTANT_RESPONSE_STATE,
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    ContentBlock,
    ErrorInfo,
    Message,
    MessageRole,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from ...providers.stream_diagnostics import fd_diagnostics
from .empty_turn import annotate_turn_metadata

if TYPE_CHECKING:
    from .agent import AgentLoop


class TurnPersistenceMixin:
    """Own provider metadata and partial-turn durability."""

    def _finalize_tool_results(
        self: AgentLoop,
        calls: Sequence[ToolCall],
        slots: Sequence[ToolResult | None],
    ) -> list[ToolResult]:
        return finalize_agent_results(self, calls, slots)

    def _annotate_current_turn(self: AgentLoop, message: Message) -> Message:
        """Apply provider metadata before any assistant message is persisted."""
        annotated = annotate_turn_metadata(
            message,
            stop_reason=self._turn_stop_reason,
            output_tokens=self._turn_output_tokens,
        )
        account_context = getattr(self, "account_context", None)
        if account_context is not None and account_context.active_provider is not None:
            metadata = dict(annotated.metadata)
            metadata["account_provider"] = account_context.active_provider
            metadata["account_switch_sequence"] = account_context.switch_sequence
            annotated = Message(
                annotated.role,
                annotated.content,
                tool_result=annotated.tool_result,
                metadata=metadata,
            )
        if not self._turn_provider_retry_records:
            return annotated
        metadata = dict(annotated.metadata)
        metadata["provider_retries"] = [
            dict(record) for record in self._turn_provider_retry_records
        ]
        return Message(
            annotated.role,
            annotated.content,
            tool_result=annotated.tool_result,
            metadata=metadata,
        )

    def _persist_partial(
        self: AgentLoop,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        if assistant_message is None:
            durable_blocks = [
                block
                for block in partial_blocks
                if not isinstance(block, ThinkingContent)
                or not block.text
                or block.signature
            ]
            if not durable_blocks and failure is None:
                return
            assistant_message = Message(MessageRole.ASSISTANT, durable_blocks)
        assistant_message = self._annotate_current_turn(assistant_message)
        metadata = dict(assistant_message.metadata)
        metadata[ASSISTANT_RESPONSE_STATE] = (
            ASSISTANT_RESPONSE_FAILED
            if failure is not None
            else ASSISTANT_RESPONSE_ABORTED
        )
        if failure is not None:
            metadata[FAILED_TURN_MARKER] = True
            metadata[FAILED_TURN_ERROR] = failure.to_dict()
            metadata["fd_diagnostics"] = fd_diagnostics()
        assistant_message = Message(
            assistant_message.role,
            assistant_message.content,
            tool_result=assistant_message.tool_result,
            metadata=metadata,
        )
        self.store.append_message(durable_message(assistant_message))

    def _persist_partial_with_cancelled_tools(
        self: AgentLoop,
        partial_blocks: list[ContentBlock],
        assistant_message: Message | None,
        *,
        failure: ErrorInfo | None = None,
    ) -> None:
        self._persist_partial(partial_blocks, assistant_message, failure=failure)
        blocks = (
            assistant_message.content
            if assistant_message is not None
            else partial_blocks
        )
        calls = [
            block.tool_call for block in blocks if isinstance(block, ToolUseContent)
        ]
        self._finalize_tool_results(calls, [None] * len(calls))
