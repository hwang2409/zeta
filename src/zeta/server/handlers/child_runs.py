"""Child-run list and safe-point follow-up request bodies."""

from __future__ import annotations

from typing import Any

from ...agent.pending_delivery import send_to_run
from ...core.child_runs import child_run_rows
from ...protocol.types import MessageOrigin
from ..protocol import ErrorDescriptor, ProtocolError
from . import validation

_MAX_PAGE_SIZE = 100


def _page(params: dict[str, Any]) -> tuple[int, int]:
    offset = params.get("offset", 0)
    limit = params.get("limit", _MAX_PAGE_SIZE)
    if type(offset) is not int or offset < 0:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "offset must be a nonnegative integer"
        )
    if type(limit) is not int or not 1 <= limit <= _MAX_PAGE_SIZE:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            f"limit must be an integer from 1 to {_MAX_PAGE_SIZE}",
        )
    return offset, limit


async def list_runs(client: Any, params: dict[str, Any]) -> dict[str, object]:
    runtime = client.server.runtime
    store = validation.active(runtime, params)
    offset, limit = _page(params)
    loop = runtime.loop
    running_handles = (
        loop._background_owner.active_instance_ids
        if loop is not None
        else frozenset()
    )
    rows = await client.connections.run_blocking(
        lambda: child_run_rows(store, running_handles=running_handles)
    )
    if not isinstance(rows, list):
        raise ProtocolError(
            ErrorDescriptor.STATE_UNAVAILABLE, "child run state is unavailable"
        )
    page = rows[offset : offset + limit]
    result: dict[str, object] = {"runs": page}
    next_offset = offset + len(page)
    if next_offset < len(rows):
        result["next_offset"] = next_offset
    return result


async def send(client: Any, params: dict[str, Any]) -> dict[str, object]:
    runtime = client.server.runtime
    store = validation.active(runtime, params)
    run_id = params.get("run_id")
    text = params.get("text")
    if type(run_id) is not str or not run_id:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "run_id must be a nonempty string"
        )
    if type(text) is not str or not text.strip():
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "text must be a nonempty string"
        )
    loop = runtime.loop
    if loop is None:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
    error = await send_to_run(
        store,
        run_id,
        text,
        owner=loop._background_owner,
        channel=loop._background_owner.conversation_channel,
        origin=MessageOrigin.USER,
    )
    if error is not None:
        raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, error)
    return {"accepted": True, "run_id": run_id}
