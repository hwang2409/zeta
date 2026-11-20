"""Shared wire rules for adapters over existing resource owners."""

from __future__ import annotations

import errno
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..core.session_execution import execution_owner_state
from ..core.session_files import SessionError, child_directory

RESTART_SCAN_SECONDS = 1.0


class Access(StrEnum):
    READ = "read"
    CONTROLLER_READ = "controller_read"
    SESSION_MUTATION = "session_mutation"
    DELIVERY = "delivery"
    CONTROLLER_WRITE = "controller_write"


@dataclass(frozen=True)
class ResourceMethod:
    wire: str
    access: Access
    handler: Callable[..., Awaitable[Any]]
    since: str | None = None
    extends_legacy: bool = False


@dataclass(frozen=True)
class ResourceDescriptor:
    wire: str
    since: str | None
    feature: str
    access: Access
    handler: Callable[..., Awaitable[Any]]
    extends_legacy: bool


def _close_handler_errors(handler: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    async def handle(client: Any, request_id: Any, params: Any) -> Any:
        with owner_errors():
            return await handler(client, request_id, params)

    return handle


def collect_families() -> tuple[ResourceDescriptor, ...]:
    """Collect explicitly registered resource families."""
    # Family modules import the wire rules defined here.
    from .handlers.resources import FAMILIES

    descriptors = []
    flags = set()
    for family in FAMILIES:
        if family.FEATURE in flags or not family.METHODS:
            raise RuntimeError("duplicate or empty serve resource family")
        flags.add(family.FEATURE)
        for method in family.METHODS:
            if method.access not in {Access.READ, Access.CONTROLLER_READ, Access.CONTROLLER_WRITE}:
                raise RuntimeError("serve resource methods must declare read, controller read, or controller write access")
            descriptors.append(ResourceDescriptor(
                method.wire, method.since, family.FEATURE, method.access, _close_handler_errors(method.handler),
                method.extends_legacy,
            ))
    return tuple(descriptors)


def require_params(params: object, *required: str, optional: tuple[str, ...] = ()) -> None:
    """Bound the object to declared fields; adapters still validate field values and sizes."""
    from .protocol import invalid_resource_params

    if not isinstance(params, dict) or not set(required) <= params.keys() <= {*required, *optional}:
        raise invalid_resource_params()


def require_revision(value: object, *, prefix: str = "") -> None:
    from .protocol import invalid_resource_params

    if not isinstance(value, str) or re.fullmatch(re.escape(prefix) + r"[0-9a-f]{64}", value) is None:
        raise invalid_resource_params()


def require_access(role: str, method: str, access: Access | None, *, resource: bool) -> None:
    if role == "observer" and access is not Access.READ:
        from .protocol import ErrorDescriptor, ProtocolError

        data = {"method": method}
        if resource:
            data["reason"] = "observer_read_only"
        raise ProtocolError(ErrorDescriptor.OBSERVER_READ_ONLY, "method is unavailable to read-only observers", data)


async def run_owner(
    client: Any,
    call: Callable[[], Any],
    *,
    family: str,
    project: Callable[[Any], Any],
    owner_error: type[Exception] | tuple[type[Exception], ...],
    errors: Mapping[str, Any],
) -> Any:
    """Run the owner and an explicit public projection off-loop. Never relay owner error details.

    Owners must compute revisions and compare them inside their own write transaction. For MCP,
    the adapter's projector must use MCPManagementService.redact, not serialize raw owner data.
    """
    def run():
        with owner_errors(family=family, owner_error=owner_error, errors=errors):
            return project(call())

    return await client.connections.run_blocking(run)


@contextmanager
def owner_errors(
    *,
    family: str = "",
    owner_error: type[Exception] | tuple[type[Exception], ...] = (),
    errors: Mapping[str, Any] | None = None,
):
    """Close unexpected failures and optionally map an owner's known errors.

    Owner detail never crosses the wire, with one exception: ``telemetry_unavailable`` keeps the
    owner's ``data`` (``code`` and ``detail``). Its detail is one of the fixed telemetry preflight
    messages that tells the user how to fix the install or environment; it names no path or value.
    """
    from .protocol import (
        ErrorDescriptor,
        ProtocolError,
        closed_error,
        invalid_resource_params,
        stale_resource_revision,
    )

    try:
        yield
    except ProtocolError:
        raise
    except Exception as exc:  # noqa: BLE001 - owner paths and request values must not cross the wire
        descriptor = ErrorDescriptor.INTERNAL_ERROR
        if isinstance(exc, owner_error):
            code = getattr(exc, "code", None)
            if code == "stale_revision":
                raise stale_resource_revision(family) from None
            if code == "invalid_params":
                raise invalid_resource_params() from None
            if errors is not None:
                descriptor = errors.get(code, descriptor)
            if descriptor is ErrorDescriptor.TELEMETRY_UNAVAILABLE:
                raise ProtocolError(descriptor, descriptor.meaning, getattr(exc, "data", None) or None) from None
        raise closed_error(descriptor) from None


def restart_report(home: Path, *, serve: bool, changed: bool) -> dict[str, Any]:
    """Find every affected top-level session by its execution lease, off-loop."""
    if not changed:
        return {"serve": False, "sessions": [], "complete": True}
    try:
        home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return {"serve": serve, "sessions": [], "complete": False}
    try:
        return _restart_report(home_fd, serve=serve)
    finally:
        os.close(home_fd)


def _restart_report(home_fd: int, *, serve: bool) -> dict[str, Any]:
    """Every top-level session a live process owns, by the kernel lease lock; never by a pid hint.

    No row cap and no recency filter; a scan that hits its deadline or an unreadable directory
    reports ``complete: false`` with the owners it confirmed. Serve itself only needs to restart for
    telemetry, because its receiver is configured once at startup.
    """
    deadline = time.monotonic() + RESTART_SCAN_SECONDS
    sessions: list[str] = []
    complete = True
    try:
        with child_directory(home_fd, "sessions") as root_fd:
            for name in os.listdir(root_fd):
                if time.monotonic() >= deadline:
                    complete = False
                    break
                try:
                    with child_directory(root_fd, name) as session_fd:
                        if execution_owner_state(session_fd) is not None:
                            sessions.append(name)
                except OSError as exc:
                    if exc.errno not in (errno.ENOTDIR, errno.ELOOP, errno.ENOENT):
                        complete = False
                except SessionError:
                    complete = False
    except FileNotFoundError:
        pass
    except OSError:
        complete = False
    return {"serve": serve, "sessions": sorted(sessions), "complete": complete}
