"""Actor-serialized MCP resource requests.

Resource list/read calls run through the owning actor's message queue exactly
like tool calls: the generation is validated on dispatch and again on
completion, a timeout degrades the actor (unpublishing its tools) and schedules
a bounded, non-blocking transport close, and a result that completes under a
stale generation is rejected instead of being returned as if it were live.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..core.abort import AbortSignal
from .client import MCPClient, MCPProtocolError, MCPTransportError
from .prompt_actor import cancel_request, set_exception, set_result

if TYPE_CHECKING:
    from .server_actor import MCPServerActor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResourceRequest:
    """One resource list/read sharing the actor request lifecycle."""

    identifier: int
    kind: Literal["list", "read"]
    uri: str | None
    generation: int
    result: asyncio.Future[object]
    task: asyncio.Task[object] | None = None
    client: MCPClient | None = None
    abort_signal: AbortSignal | None = None


@dataclass(frozen=True, slots=True)
class ResourceFinished:
    request: ResourceRequest
    task: asyncio.Task[object]
    result: object | None
    error: BaseException | None


def resource_unavailable(name: str) -> str:
    return f"MCP server {name} is unavailable"


def resource_timed_out(name: str) -> str:
    return f"MCP server {name} resource request timed out"


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask the failure
        return type(error).__name__


def _resource_timeout_seconds() -> float:
    # Read lazily so tests can monkeypatch server_actor's module-level bound and
    # so this module does not import server_actor at load time (it would cycle).
    from . import server_actor

    return server_actor.RESOURCE_REQUEST_TIMEOUT_SECONDS


async def request_resource(
    actor: MCPServerActor,
    kind: Literal["list", "read"],
    uri: str | None,
    *,
    generation: int,
    abort_signal: AbortSignal | None = None,
) -> object:
    """Queue one resource request with actor-owned cancellation."""

    future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    actor._next_identifier += 1
    request = ResourceRequest(
        actor._next_identifier, kind, uri, generation, future, abort_signal=abort_signal
    )
    if actor.is_terminal:
        raise RuntimeError(resource_unavailable(actor.name))
    actor._queue.put_nowait(request)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await cancel_request(actor, request.identifier)
        raise


def handle_resource(actor: MCPServerActor, request: ResourceRequest) -> None:
    if (
        actor._closed
        or request.generation != actor._generation
        or actor._client is None
        or actor._status.state != "mounted"
    ):
        set_exception(request.result, RuntimeError(resource_unavailable(actor.name)))
        return
    _dispatch_resource(actor, request, actor._client)


def _dispatch_resource(
    actor: MCPServerActor, request: ResourceRequest, client: MCPClient
) -> None:
    request.client = client
    request.task = asyncio.create_task(_invoke_resource(client, request))
    actor._resource_requests[request.identifier] = request
    actor._children.add(request.task)
    request.task.add_done_callback(
        lambda done, call=request: _queue_resource_result(actor, done, call)
    )


async def _invoke_resource(client: MCPClient, request: ResourceRequest) -> object:
    timeout = _resource_timeout_seconds()
    if request.kind == "list":
        return await asyncio.wait_for(
            client.list_resources(request.abort_signal), timeout
        )
    return await asyncio.wait_for(
        client.read_resource(request.uri or "", request.abort_signal), timeout
    )


def _queue_resource_result(
    actor: MCPServerActor,
    task: asyncio.Task[object],
    request: ResourceRequest,
) -> None:
    if task.cancelled():
        actor._queue.put_nowait(
            ResourceFinished(request, task, None, asyncio.CancelledError())
        )
        return
    try:
        result = task.result()
    except BaseException as exc:  # noqa: BLE001 - routed back through the actor
        actor._queue.put_nowait(ResourceFinished(request, task, None, exc))
    else:
        actor._queue.put_nowait(ResourceFinished(request, task, result, None))


def handle_resource_finished(actor: MCPServerActor, message: ResourceFinished) -> None:
    actor._children.discard(message.task)
    request = actor._resource_requests.pop(
        message.request.identifier, message.request
    )
    error = message.error
    if isinstance(error, asyncio.CancelledError):
        set_exception(request.result, RuntimeError(resource_unavailable(actor.name)))
        return
    current = (
        not actor._closed
        and actor._client is request.client
        and request.generation == actor._generation
        and actor._status.state == "mounted"
    )
    if error is not None:
        # Degrade only on failures that compromise the transport (timeout,
        # transport failure, or malformed protocol state). Request-level errors
        # such as "resource not found" are returned while staying mounted,
        # mirroring how the prompt/tool paths distinguish them.
        degrades = isinstance(error, (MCPTransportError, MCPProtocolError, TimeoutError))
        if current and degrades:
            timed_out = isinstance(error, TimeoutError)
            actor._degrade_current(
                resource_timed_out(actor.name) if timed_out else _error_text(error)
            )
            text = (
                resource_timed_out(actor.name)
                if timed_out
                else resource_unavailable(actor.name)
            )
            set_exception(request.result, RuntimeError(text))
        elif current:
            set_exception(request.result, error)
        else:
            set_exception(request.result, RuntimeError(resource_unavailable(actor.name)))
        return
    if not current:
        set_exception(request.result, RuntimeError(resource_unavailable(actor.name)))
        return
    set_result(request.result, message.result)


def resolve_resource_message(message: object, server_name: str) -> bool:
    """Resolve a queued resource message after actor termination."""

    if isinstance(message, ResourceRequest):
        set_exception(message.result, RuntimeError(resource_unavailable(server_name)))
    elif isinstance(message, ResourceFinished):
        set_exception(
            message.request.result, RuntimeError(resource_unavailable(server_name))
        )
    else:
        return False
    return True


def cancel_terminated_resources(actor: MCPServerActor) -> None:
    """Cancel and fail every in-flight resource request while terminating."""

    for request in actor._resource_requests.values():
        if request.task is not None:
            request.task.cancel()
        set_exception(request.result, RuntimeError(resource_unavailable(actor.name)))
    actor._resource_requests.clear()
