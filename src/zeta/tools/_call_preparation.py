"""Tool-call preparation before approval, persistence, and execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
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
        for tool_call in calls:
            try:
                prepared = await self.prepare_tool_call(tool_call)
            except (KeyError, TypeError, ValueError):
                prepared = tool_call
                preparation_errors.add(tool_call.id)
            prepared_calls.append(prepared)
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
        return await definition.call_preparer(tool_call)

    async def approval_label(self, tool_call: ToolCall) -> str | None:
        """Name an inbox send's canonical target project, looked up off the loop."""
        from ..project_errors import ProjectRegistryError
        from ..project_schema import PROJECT_ID_PATTERN
        project = tool_call.arguments.get("project")
        if (
            tool_call.name != "inbox"
            or tool_call.arguments.get("action") != "send"
            or self.project_registry is None
            or not isinstance(project, str)
            or not PROJECT_ID_PATTERN.fullmatch(project)
        ):
            return None
        try:
            found = await asyncio.to_thread(self.project_registry.show_project, project_id=project)
        except (ProjectRegistryError, OSError):
            # the label is display-only: any registry read error yields no label.
            return None
        return f"{found.name} ({project})"

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
