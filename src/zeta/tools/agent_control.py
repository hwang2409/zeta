"""Model-facing cancellation of child agents."""

from __future__ import annotations

import asyncio
from typing import Any

from .registry import AbortSignal, ToolExecutionContext, ToolRegistry, text_block


def _owner(registry: ToolRegistry) -> Any:
    owner = getattr(registry, "_agent_owner", None)
    if owner is None:
        raise TypeError("child control is unavailable outside an agent session")
    return owner


def _status(registry: ToolRegistry, handles: list[str]) -> set[str]:
    from .agent import _read_agent_status

    rows = _read_agent_status(registry.session_store)
    known = {row.get("handle") for row in rows}
    return set(handles) - known


async def agent_cancel(
    registry: ToolRegistry, arguments: dict[str, Any], abort_signal: AbortSignal,
    stream_publisher: object = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del stream_publisher, execution_context
    from .agent import _agent_error, _bounded_agent_result

    max_bytes = registry.max_output_chars
    handle = arguments.get("handle")
    if type(handle) is not str or not handle:
        return _agent_error("handle must be a nonempty string", max_bytes)
    try:
        if abort_signal.is_set():
            raise asyncio.CancelledError
        owner = _owner(registry)
        if _status(registry, [handle]):
            return _agent_error("unknown child handle", max_bytes)
        outcome = await owner.cancel_and_wait(handle)
        status = {
            "confirmed": "cancellation_confirmed",
            "unconfirmed": "cancellation_unconfirmed",
            "failed": "cancellation_failed",
            "already_finished": "already_finished",
        }[outcome.status]
        text = {
            "confirmed": "agent cancellation confirmed: ",
            "unconfirmed": "agent cancellation not confirmed within 1 second: ",
            "failed": "agent cancellation failed: ",
            "already_finished": "agent already finished: ",
        }[outcome.status] + handle
        if outcome.error is not None:
            text += ": " + outcome.error
        structured_content = {"handle": handle, "status": status}
        if outcome.error is not None:
            structured_content["confirmation_error"] = outcome.error
        result = {
            "content": [text_block(text)],
            "isError": outcome.status in {"unconfirmed", "failed"},
            "structuredContent": structured_content,
        }
        return _bounded_agent_result(result, max_bytes)
    except (TypeError, ValueError) as exc:
        return _agent_error(str(exc), max_bytes)


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "agent_cancel", agent_cancel,
        description=("Cancel one running child agent owned by this agent tree. "
                     "The request is idempotent and never affects another session; "
                     "completion remains announced normally. A finished child returns "
                     "already_finished."),
        parameters={"type": "object", "properties": {"handle": {"type": "string", "minLength": 1}},
                    "required": ["handle"], "additionalProperties": False},
        parallel_safe=True, requires_approval=False,
    )
