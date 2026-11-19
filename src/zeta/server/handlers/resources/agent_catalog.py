"""Observer-safe aggregate of durable child-run rows."""

from __future__ import annotations

import hashlib
import json

from ....core import agent_run_catalog as catalog
from ....core.store._child_runs import parse_child_run_id
from ....read_page import Key, TooLarge, page_result, resume
from ...resource_protocol import Access, ResourceMethod, require_params, run_owner

FEATURE = "agent_run_catalog_v1"
_METHOD = "agent_run_catalog"
_ORDER = (Key("mtime_ns", int, descending=True), Key("session_id", str), Key("child_path", str))


def _public_page(page, start, fits):
    rows, keys, scanned = page

    def shape(items, cursor):
        data = {"items": items}
        if cursor is not None:
            data["next_cursor"] = cursor
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return {"revision": hashlib.sha256(raw).hexdigest(), "data": data}

    try:
        return page_result(
            rows, keys.__getitem__, start, more=scanned is not None, total=None,
            fits=fits, shape=shape, scan_end=scanned,
        )
    except TooLarge:
        raise catalog.CatalogReadError("result_too_large") from None


async def agent_run_catalog(client, request_id, params):
    from ...protocol import ErrorDescriptor, invalid_resource_params

    require_params(params, optional=("cursor", "limit", "state", "root_session_id"))
    limit = params.get("limit", 50)
    state = params.get("state")
    root_id = params.get("root_session_id")
    if (
        type(limit) is not int or not 1 <= limit <= 100
        or ("state" in params and (type(state) is not str or state not in catalog.STATES))
        or ("root_session_id" in params and type(root_id) is not str)
        or ("cursor" in params and type(params["cursor"]) is not str)
    ):
        raise invalid_resource_params()
    try:
        if root_id is not None:
            catalog.validate_root_id(root_id)
        start = resume(params.get("cursor"), (_METHOD, root_id, state), (), _ORDER)
        if start.position is not None:
            stamp, sid, path = start.position
            catalog.validate_root_id(sid)
            if root_id is not None and (stamp != 0 or sid != root_id):
                raise ValueError("invalid scoped cursor")
            if path:
                parse_child_run_id(sid, f"{sid}:{path}")
    except (ValueError, RecursionError):
        raise invalid_resource_params() from None

    return await run_owner(
        client, lambda: catalog.read_page(
            client.server.home, root_id=root_id, state=state, after=start.position, limit=limit,
        ),
        family="agent_run_catalog",
        project=lambda page: _public_page(page, start, lambda result: client.codec.response_fits(request_id, result)),
        owner_error=catalog.CatalogReadError,
        errors={
            "state_unavailable": ErrorDescriptor.STATE_READ_UNAVAILABLE,
            "result_too_large": ErrorDescriptor.RESULT_TOO_LARGE,
        },
    )


METHODS = (ResourceMethod(_METHOD, Access.READ, agent_run_catalog),)
