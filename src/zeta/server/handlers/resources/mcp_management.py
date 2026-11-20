"""Serve projection of the durable MCP configuration, not the running mount."""

from __future__ import annotations

import asyncio

from ....mcp.management import ManagedServer, MCPManagementError, MCPManagementService
from ....mcp.management_privacy import private_management_io
from ...resource_protocol import (
    Access,
    ResourceMethod,
    owner_errors,
    require_params,
    require_revision,
    restart_report,
    run_owner,
)

FEATURE = "mcp_management_v1"
_SCOPES = ("user", "project", "effective")


def _owner(client) -> MCPManagementService:
    runtime = client.server.runtime
    project = runtime.loop._mcp_project_dir_value if runtime.loop is not None else runtime.cwd
    return MCPManagementService(home=client.server.home, project_dir=project)


def _error_rules():
    from ...protocol import ErrorDescriptor

    return {
        "family": "mcp",
        "owner_error": MCPManagementError,
        "errors": {"busy": ErrorDescriptor.RESOURCE_BUSY, "state_unavailable": ErrorDescriptor.STATE_READ_UNAVAILABLE},
    }


def _data(entries: tuple[ManagedServer, ...], scope: str) -> dict:
    rows = {}
    for entry in entries:
        if scope not in (entry.scope, "effective"):
            continue
        row = entry.as_json()
        row["config"] = MCPManagementService.redact(row["config"], wire=True)
        row["auth"] = entry.auth if entry.auth in ("none", "unauthorized") else "none"
        rows[entry.name] = row  # The project definition shadows the user definition.
    return {"items": [rows[name] for name in sorted(rows)]}


async def mcp_server_list(client, request_id, params):
    from ...protocol import invalid_resource_params

    require_params(params, optional=("scope",))
    scope = params.get("scope", "effective")
    if scope not in _SCOPES:
        raise invalid_resource_params()
    owner = _owner(client)
    def read():
        with private_management_io():
            return owner.read_disk(require_project=scope == "project")

    return await run_owner(
        client, read,
        project=lambda result: {"revision": result[0], "data": _data(result[1], scope)},
        **_error_rules(),
    )


def _string(value: object) -> bool:
    return isinstance(value, str) and len(value) <= 8192 and "\0" not in value


def _text(value: object) -> bool:
    return _string(value) and value != ""


def _string_map(value: object) -> bool:
    return isinstance(value, dict) and len(value) <= 128 and all(
        _text(key) and _string(item) for key, item in value.items()
    )


def _args(value: object) -> bool:
    return isinstance(value, list) and len(value) <= 128 and all(map(_string, value))


def _boolean(value: object) -> bool:
    return type(value) is bool


_ADD_FIELDS = {
    "command": _text,
    "args": _args,
    "url": _text,
    "env": _string_map,
    "headers": _string_map,
    "oauth": _boolean,
    "enabled": _boolean,
    "client_id": _text,
    "client_secret": _text,
}


def _add_params(mutation: dict) -> None:
    from ...protocol import invalid_resource_params

    if any(key in mutation and not validate(mutation[key]) for key, validate in _ADD_FIELDS.items()):
        raise invalid_resource_params()
    if (
        ("command" in mutation) == ("url" in mutation)
        or ("command" in mutation and ("headers" in mutation or mutation.get("oauth")))
        or ("url" in mutation and "args" in mutation)
        or (not mutation.get("oauth") and ("client_id" in mutation or "client_secret" in mutation))
        or ("client_secret" in mutation and "client_id" not in mutation)
    ):
        raise invalid_resource_params()


def _write(action: str, *keys: str, optional: tuple[str, ...] = ()):
    async def handle(client, request_id, params):
        from ...protocol import invalid_resource_params

        require_params(params, "revision", "mutation")
        require_revision(params["revision"])
        mutation = params["mutation"]
        require_params(mutation, "name", *keys, optional=optional)
        name = mutation["name"]
        scope = mutation.get("scope", "project")
        if not _text(name) or len(name) > 256 or scope not in _SCOPES[:2]:
            raise invalid_resource_params()
        if action == "add":
            _add_params(mutation)
        elif any(key in mutation and not _boolean(mutation[key]) for key in ("enabled", "trusted")):
            raise invalid_resource_params()

        owner = _owner(client)
        with owner_errors(**_error_rules()), private_management_io():
            try:
                view = await getattr(owner, action)(**mutation, revision=params["revision"])
                restart = await client.connections.run_blocking(lambda: restart_report(
                    client.server.home, serve=False, changed=view.revision != params["revision"],
                ))
                return {"revision": view.revision, "data": _data(view.entries, scope), "restart": restart}
            finally:
                # Join the worker before unwinding a cancelled request.
                await owner.aclose()

    return handle


def _target(params, *, scoped=True):
    from ...protocol import invalid_resource_params

    require_params(params, "name", optional=("scope",) if scoped else ())
    name = params["name"]
    scope = params.get("scope", "effective")
    if not _text(name) or len(name) > 256 or scope not in _SCOPES:
        raise invalid_resource_params()
    return name, scope


def _registry(client):
    loop = client.server.runtime.loop
    return loop.tool_registry if loop is not None else None


async def mcp_server_test(client, request_id, params):
    name, scope = _target(params)
    with owner_errors(**_error_rules()), private_management_io():
        result = await _owner(client).test(name, scope=scope, wire=True, registry=_registry(client))
    return {"data": {"status": result["status"], "tool_count": result.get("tool_count", 0)}}


async def mcp_auth_start(client, request_id, params):
    name, scope = _target(params)
    owner = _owner(client)
    registry = _registry(client)
    return {"data": await client.server._mcp_auth.start(name, lambda ready, commit: owner.login(
        name, scope=scope, launch_ready=ready, before_commit=commit, registry=registry,
    ))}


async def mcp_auth_status(client, request_id, params):
    name, _ = _target(params, scoped=False)
    return {"data": client.server._mcp_auth.status(name)}


async def mcp_auth_cancel(client, request_id, params):
    name, _ = _target(params, scoped=False)
    return {"data": await client.server._mcp_auth.cancel(name)}


async def mcp_auth_logout(client, request_id, params):
    name, scope = _target(params)
    auth = client.server._mcp_auth
    await auth.cancel(name)
    with owner_errors(**_error_rules()), private_management_io():
        work = asyncio.create_task(_owner(client).logout(name, scope=scope))
        cancelled = False
        # Every cancellation waits for the credential delete before unwinding.
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            if not work.cancelled():
                work.exception()  # Mark it retrieved; a cancelled request never relays it.
            raise asyncio.CancelledError
        work.result()
    auth.forget(name)
    return {"data": {"state": "unauthorized"}}


METHODS = (
    ResourceMethod("mcp_server_list", Access.READ, mcp_server_list),
    ResourceMethod("mcp_server_add", Access.CONTROLLER_WRITE, _write("add", "scope", optional=tuple(_ADD_FIELDS))),
    ResourceMethod("mcp_server_remove", Access.CONTROLLER_WRITE, _write("remove", "scope")),
    ResourceMethod("mcp_server_set_enabled", Access.CONTROLLER_WRITE, _write("set_enabled", "scope", "enabled")),
    ResourceMethod("mcp_server_set_trusted", Access.CONTROLLER_WRITE, _write("set_trusted", "trusted")),
    ResourceMethod("mcp_server_test", Access.CONTROLLER_WRITE, mcp_server_test),
    ResourceMethod("mcp_auth_start", Access.CONTROLLER_WRITE, mcp_auth_start),
    ResourceMethod("mcp_auth_status", Access.READ, mcp_auth_status),
    ResourceMethod("mcp_auth_cancel", Access.CONTROLLER_WRITE, mcp_auth_cancel),
    ResourceMethod("mcp_auth_logout", Access.CONTROLLER_WRITE, mcp_auth_logout),
)
