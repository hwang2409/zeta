"""Model-facing process-local wake tool."""

from __future__ import annotations

from typing import Any

from ...agent.wakes import WakeError, WakeRecord, WakeScheduler
from ..registry import ToolRegistry, text_block

WAKE_TOOL_NAME = "wake"
_RESTART_WARNING = "pending wakes are lost when this zeta session process restarts"


def register(_registry: ToolRegistry) -> None:
    """Wake is registered explicitly only for an eligible resident loop."""


def register_wake(registry: ToolRegistry, scheduler: WakeScheduler) -> None:
    async def wake(
        _registry: ToolRegistry,
        arguments: dict[str, Any],
    ) -> dict[str, object]:
        try:
            return _execute(scheduler, arguments)
        except WakeError as exc:
            return _error(str(exc))

    registry.register_session_tool(
        WAKE_TOOL_NAME,
        wake,
        description=(
            "Schedule one process-local reminder for this session, list pending "
            "reminders, or cancel one. Pending reminders are lost when this zeta "
            "session process restarts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["schedule", "cancel", "list"],
                    "default": "schedule",
                },
                "after_seconds": {"type": "number"},
                "note": {"type": "string"},
                "wake_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
        validate_arguments=False,
        requires_approval=False,
        parallel_safe=False,
    )


def _execute(scheduler: WakeScheduler, arguments: dict[str, Any]) -> dict[str, object]:
    action = arguments.get("action", "schedule")
    if action == "schedule":
        if set(arguments) - {"action", "after_seconds", "note"}:
            raise WakeError("schedule accepts only after_seconds and note")
        if "after_seconds" not in arguments or "note" not in arguments:
            raise WakeError("schedule requires after_seconds and note")
        result = scheduler.schedule(
            after_seconds=arguments["after_seconds"], note=arguments["note"]
        )
        structured = {
            **_record_data(result.record),
            "autonomous": result.autonomous,
            "reason": result.reason,
            "durability": "process_local",
            "warning": _RESTART_WARNING,
            "admissions_used": scheduler.admission_count,
        }
        text = (
            f"scheduled {result.record.wake_id} for "
            f"{result.record.due_at_utc.isoformat()}; {_RESTART_WARNING}"
        )
        if result.reason is not None:
            text += f"; autonomous wake disabled: {result.reason}"
        return {
            "content": [text_block(text)],
            "isError": result.reason is not None,
            "structuredContent": structured,
        }
    if action == "cancel":
        if set(arguments) - {"action", "wake_id"}:
            raise WakeError("cancel accepts only wake_id")
        wake_id = arguments.get("wake_id")
        if type(wake_id) is not str or not wake_id:
            raise WakeError("cancel requires wake_id")
        record = scheduler.cancel(wake_id)
        return {
            "content": [text_block(f"canceled {record.wake_id}")],
            "isError": False,
            "structuredContent": {
                "action": "cancel",
                "wake_id": record.wake_id,
                "canceled": True,
            },
        }
    if action == "list":
        if set(arguments) != {"action"} and arguments:
            raise WakeError("list accepts no operation fields")
        records = scheduler.list_pending()
        return {
            "content": [
                text_block(f"{len(records)} pending wake(s); {_RESTART_WARNING}")
            ],
            "isError": False,
            "structuredContent": {
                "action": "list",
                "wakes": [_record_data(record) for record in records],
                "count": len(records),
                "durability": "process_local",
                "warning": _RESTART_WARNING,
            },
        }
    raise WakeError("action must be schedule, cancel, or list")


def _record_data(record: WakeRecord) -> dict[str, object]:
    return {
        "wake_id": record.wake_id,
        "note": record.note,
        "scheduled_at": record.scheduled_at_utc.isoformat(),
        "due_at": record.due_at_utc.isoformat(),
        "session_class": record.session_class,
        "chain_id": record.chain_id,
        "generation": record.generation,
        "state": record.state,
        "reason": record.passive_reason,
    }


def _error(message: str) -> dict[str, object]:
    return {
        "content": [text_block("wake error: " + message)],
        "isError": True,
        "structuredContent": {
            "error": {
                "kind": "invalid_wake_operation",
                "message": message,
            }
        },
    }
