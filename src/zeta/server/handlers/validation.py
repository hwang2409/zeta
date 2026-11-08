"""The one definition of each shared serve request validator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..protocol import ErrorDescriptor, ProtocolError
from ..runtime import ServerRuntime


def require_feature(self, feature: str, param: str) -> None:
    if feature not in self.features:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            f"{param} requires the negotiated {feature} feature",
        )


def required_string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, f"{name} must be a nonempty string"
        )
    return value


def integer(
    params: dict[str, Any], name: str, default: int | None, *, minimum: int
) -> int | None:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            f"{name} must be an integer of at least {minimum}",
        )
    return value


def directory(params: dict[str, Any], name: str) -> Path:
    value = params.get(name)
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, f"{name} must be an absolute directory path"
        )
    path = Path(value).resolve()
    if not path.is_dir():
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            f"{name} is not an existing directory: {value}",
        )
    return path


def optional_string(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is not None and (not isinstance(value, str) or not value):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, f"{name} must be a nonempty string"
        )
    return value


def active(runtime: ServerRuntime, params: dict):
    session_id = params.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "session_id must be a nonempty string"
        )
    if runtime.opened is None or session_id != runtime.session_id:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
    return runtime.opened.store


def require_mutable(
    runtime: ServerRuntime, has_pending_approvals: bool = False
) -> None:
    loop = runtime.loop
    if loop is not None and (
        loop.background_children_running
        or loop.store.has_outstanding_tool_calls(loop.store.replay())
        or has_pending_approvals
    ):
        raise ProtocolError(
            ErrorDescriptor.OPERATION_BUSY,
            "finish pending tools and background agents first",
        )
