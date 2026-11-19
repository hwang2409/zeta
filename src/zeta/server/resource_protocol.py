"""Shared wire rules for adapters over existing resource owners."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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


@dataclass(frozen=True)
class ResourceDescriptor:
    wire: str
    since: str | None
    feature: str
    access: Access
    handler: Callable[..., Awaitable[Any]]


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
                method.wire, method.since, family.FEATURE, method.access, method.handler,
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
    owner_error: type[Exception],
    errors: Mapping[str, Any],
) -> Any:
    """Run the owner and an explicit public projection off-loop. Never relay owner error details.

    Owners must compute revisions and compare them inside their own write transaction. For MCP,
    the adapter's projector must use MCPManagementService.redact, not serialize raw owner data.
    """
    from .protocol import (
        ErrorDescriptor,
        closed_error,
        invalid_resource_params,
        stale_resource_revision,
    )

    def run():
        try:
            return project(call())
        except owner_error as exc:
            code = getattr(exc, "code", None)
            if code == "stale_revision":
                raise stale_resource_revision(family) from None
            if code == "invalid_params":
                raise invalid_resource_params() from None
            raise closed_error(errors.get(code, ErrorDescriptor.INTERNAL_ERROR)) from None
        except Exception:  # noqa: BLE001 - owner paths and request values must not cross the wire
            raise closed_error(ErrorDescriptor.INTERNAL_ERROR) from None

    return await client.connections.run_blocking(run)
