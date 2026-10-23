"""Bounded result payloads produced by child-agent loops."""

from __future__ import annotations

from collections.abc import Sequence

from ..protocol.types import ToolResult
from ..tools.agent import MAX_AGENT_RESULT_BYTES, agent_result
from .receipt import TerminalState, terminal_state
from .tool_results import validated_tool_result


class ChildResultPayloadMixin:
    """Build canonical progress and terminal payloads for child tools."""

    def _canceled_agent_result(
        self,
        tool_call_id: str,
        *,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        child_instance_id: str | None = None,
    ) -> ToolResult:
        return validated_tool_result(
            self._child_result_payload(
                tool_call_id,
                "tool execution canceled",
                state="canceled",
                child_session_path=child_session_path,
                turns_used=turns_used,
                agent_type=agent_type,
                child_instance_id=child_instance_id,
            ),
            tool_call_id,
        )

    def _child_result_payload(
        self,
        tool_call_id: str,
        content: str,
        *,
        state: TerminalState | None = None,
        error: bool | None = None,
        child_session_path: str | None = None,
        turns_used: int | None = None,
        agent_type: str | None = None,
        child_instance_id: str | None = None,
        status: str | None = None,
        description: str | None = None,
        depth: int | None = None,
        stats: dict[str, object] | None = None,
        include_stats: bool = True,
        canceled: bool = False,
        notice: str | None = None,
        notice_items: Sequence[str] | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, object]:
        child_store = self._agent_child_stores.get(tool_call_id)
        path = (
            child_session_path
            if child_session_path is not None
            else str(child_store.session_dir)
            if child_store is not None
            else ""
        )
        turns = (
            turns_used
            if turns_used is not None
            else self._agent_child_turns.get(tool_call_id, 0)
        )
        if child_instance_id is None and child_store is not None:
            child_instance_id = child_store.agent_handle()
        result_status = (
            "running"
            if status == "running"
            else state
            or terminal_state(
                error=error is True,
                canceled=canceled,
                status=status,
            )
        )
        return agent_result(
            content,
            tool_call_id=tool_call_id,
            error=result_status == "failed",
            turns_used=turns,
            child_session_path=path,
            agent_type=agent_type if isinstance(agent_type, str) else None,
            status=status if status == "running" else None,
            child_instance_id=child_instance_id,
            description=description,
            depth=depth,
            stats=stats,
            include_stats=include_stats,
            canceled=result_status == "canceled",
            notice=notice,
            notice_items=notice_items,
            max_bytes=(
                max_bytes
                if max_bytes is not None
                else getattr(
                    getattr(self, "tool_registry", None),
                    "max_output_chars",
                    MAX_AGENT_RESULT_BYTES,
                )
            ),
        )
