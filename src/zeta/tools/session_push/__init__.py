"""Approval-gated upload of the current durable session."""

from __future__ import annotations

import asyncio
from typing import Any

from ...protocol.types import StructuredToolResult
from ...remote_sync import RemoteSyncError, push_session, resolve_transport
from ..registry import ToolRegistry, _success_result, text_block


async def _push(
    registry: ToolRegistry, arguments: dict[str, Any]
) -> StructuredToolResult:
    host = arguments["host"]
    try:
        store = registry.session_store
        home = store.root_dir.parent
        transport = resolve_transport(home, host)
        result = await asyncio.to_thread(
            push_session,
            home,
            transport,
            session_id=store.session_id,
            force=bool(arguments.get("force", False)),
        )
    except (RemoteSyncError, OSError, ValueError) as exc:
        return _error(str(exc))
    return _success_result(
        text_block(
            f"uploaded session {result.session_id} to {host}; "
            f"last sequence {result.last_seq}"
        ),
        structured_content={
            "session_id": result.session_id,
            "host": host,
            "last_seq": result.last_seq,
            "digest": result.digest,
            "resume_notice": result.resume_notice,
        },
    )


def _error(message: str) -> StructuredToolResult:
    return {
        "content": [text_block(message)],
        "isError": True,
        "structuredContent": {
            "error": {"kind": "session_push", "message": message}
        },
    }


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "session_push",
        _push,
        approval_subject="host",
        description=(
            "Upload the current session and linked project memory through SSH. "
            "This sends transcripts, child-agent history, background-output state, "
            "and spill files to a trusted remote machine. Network access requires approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Configured remote alias or explicit SSH host.",
                },
                "force": {
                    "type": "boolean",
                    "description": "Replace divergent remote state.",
                },
            },
            "required": ["host"],
            "additionalProperties": False,
        },
        requires_approval=True,
    )


__all__ = ["register"]
