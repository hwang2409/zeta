"""Tool-call preparation before approval, persistence, and execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..core.approval import ApprovalAbortPolicy, ApprovalRule
from ..protocol.types import Message, ToolCall, ToolUseContent

if TYPE_CHECKING:
    from .registry import ToolDefinition


class _CallPreparationMixin:
    _tools: dict[str, ToolDefinition]
    approval_policy: ApprovalAbortPolicy | None
    project_registry: Any
    resolve_approval_rule: Callable[[ApprovalRule], ApprovalRule]

    async def prepare_tool_calls(
        self, assistant_message: Message
    ) -> tuple[Message, list[ToolCall], set[str]]:
        calls = [
            block.tool_call
            for block in assistant_message.content
            if isinstance(block, ToolUseContent)
        ]
        prepared_calls: list[ToolCall] = []
        preparation_errors: set[str] = set()
        try:
            for tool_call in calls:
                try:
                    prepared = await self.prepare_tool_call(tool_call)
                except (KeyError, TypeError, ValueError):
                    prepared = tool_call
                    preparation_errors.add(tool_call.id)
                prepared_calls.append(prepared)
        except BaseException:
            self.forget_approval_labels(calls)
            raise
        if prepared_calls != calls:
            prepared_by_id = {call.id: call for call in prepared_calls}
            assistant_message = replace(
                assistant_message,
                content=[
                    ToolUseContent(prepared_by_id[block.tool_call.id])
                    if isinstance(block, ToolUseContent)
                    else block
                    for block in assistant_message.content
                ],
            )
        return assistant_message, prepared_calls, preparation_errors

    async def prepare_tool_call(self, tool_call: ToolCall) -> ToolCall:
        """Prepare one call before approval, persistence, and execution."""
        definition = self._tools.get(tool_call.name)
        if definition is None:
            return tool_call
        if definition.call_preparer is None:
            return tool_call
        if self.approval_policy is not None:
            await self.approval_policy.prepare_rules(self._prepare_approval_rule)
        prepared = await definition.call_preparer(tool_call)
        if (
            self.approval_policy is not None
            and tool_call.name == "inbox"
            and tool_call.arguments.get("action") == "send"
            and isinstance(tool_call.arguments.get("project"), str)
            and prepared.arguments.get("project") != tool_call.arguments["project"]
        ):
            self.approval_policy.set_display_label(
                tool_call.id, f"{tool_call.arguments['project']} ({prepared.arguments['project']})"
            )
        return prepared

    def forget_approval_labels(self, calls: Iterable[ToolCall]) -> None:
        """Drop display labels once their calls settle; the policy owns them."""
        if self.approval_policy is not None:
            for call in calls:
                self.approval_policy.forget_display_label(call.id)

    async def _prepare_approval_rule(self, rule: ApprovalRule) -> ApprovalRule:
        resolved = self.resolve_approval_rule(rule)
        subject = resolved.subject_pattern
        if resolved.tool != "inbox" or resolved.action != "send" or not isinstance(subject, str):
            return resolved
        from ..project_registry import resolve_project_ref
        from ..project_schema import PROJECT_ID_PATTERN
        if PROJECT_ID_PATTERN.fullmatch(subject) or self.project_registry is None:
            return resolved
        project_id = await asyncio.to_thread(resolve_project_ref, self.project_registry, subject)
        return resolved if project_id is None else replace(resolved, subject_pattern=project_id)
