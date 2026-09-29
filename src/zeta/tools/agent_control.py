"""Model-facing cancellation and event-driven child waiting."""

from __future__ import annotations

import asyncio
from typing import Any

from .registry import AbortSignal, ToolExecutionContext, ToolRegistry, text_block


def _handles(arguments: dict[str, Any]) -> list[str] | str:
    values = arguments.get("handles")
    if type(values) is not list or not values:
        return "handles must be a nonempty array of strings"
    if any(type(value) is not str or not value for value in values):
        return "handles must be a nonempty array of strings"
    if len(set(values)) != len(values):
        return "handles must not contain duplicates"
    return values


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
        requested = owner.cancel(handle)
        result = {
            "content": [text_block(
                "agent cancellation requested: " + handle
                if requested else "agent already finished: " + handle
            )],
            "isError": False,
            "structuredContent": {
                "handle": handle,
                "status": "cancellation_requested" if requested else "already_finished",
            },
        }
        return _bounded_agent_result(result, max_bytes)
    except (TypeError, ValueError) as exc:
        return _agent_error(str(exc), max_bytes)


async def agent_wait(
    registry: ToolRegistry, arguments: dict[str, Any], abort_signal: AbortSignal,
    stream_publisher: object = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del stream_publisher, execution_context
    from .agent import _agent_error, _bounded_agent_result, _read_agent_status

    max_bytes = registry.max_output_chars
    handles = _handles(arguments)
    if isinstance(handles, str):
        return _agent_error(handles, max_bytes)
    timeout = arguments.get("timeout", 30.0)
    if type(timeout) not in {int, float} or isinstance(timeout, bool) or timeout < 0 or timeout > 300:
        return _agent_error("timeout must be a number between 0 and 300 seconds", max_bytes)
    try:
        if abort_signal.is_set():
            raise asyncio.CancelledError
        owner = _owner(registry)
        if _status(registry, handles):
            return _agent_error("unknown child handle", max_bytes)
        running = {handle for handle in handles if owner.owns_running(handle)}
        timed_out = False
        if running:
            wait_task = asyncio.create_task(owner.wait_for(running, float(timeout)))
            abort_task = asyncio.create_task(abort_signal.wait())
            try:
                done, pending = await asyncio.wait(
                    {wait_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                # Only successful completion wins a same-turn abort race.
                if wait_task in done:
                    completed = bool(wait_task.result())
                    if completed:
                        timed_out = False
                    elif abort_signal.is_set():
                        raise asyncio.CancelledError
                    else:
                        timed_out = True
                elif abort_task in done and abort_signal.is_set():
                    wait_task.cancel()
                    await asyncio.gather(wait_task, return_exceptions=True)
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                wait_task.cancel()
                abort_task.cancel()
                await asyncio.gather(wait_task, abort_task, return_exceptions=True)
                raise
        rows = _read_agent_status(registry.session_store)
        by_handle = {row.get("handle"): row for row in rows}
        if set(handles) - set(by_handle):
            return _agent_error("unknown child handle", max_bytes)
        result = {
            "content": [text_block(
                "agent wait timed out" if timed_out else "agent children finished"
            )],
            "isError": False,
            "structuredContent": {
                "children": [by_handle[handle] for handle in handles],
                "timed_out": timed_out,
            },
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
    registry.register_session_tool(
        "agent_wait", agent_wait,
        description=("Wait for one or more owned child agents to finish. This is "
                     "event-driven (not polling), returns finished snapshots, and "
                     "has a bounded timeout of 30 seconds by default (maximum 300)."),
        parameters={"type": "object", "properties": {
            "handles": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
            "timeout": {"type": "number", "minimum": 0, "maximum": 300,
                        "description": "Maximum wait in seconds; 30 by default, 300 maximum."},
        }, "required": ["handles"], "additionalProperties": False},
        parallel_safe=True, requires_approval=False,
    )
