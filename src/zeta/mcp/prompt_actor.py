"""Prompt request handling owned by one MCP server actor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from .client import MCPClient, MCPPrompt

if TYPE_CHECKING:
    from .server_actor import MCPServerActor


@dataclass(slots=True)
class PromptRequest:
    identifier: int
    prompt_name: str
    arguments: dict[str, str]
    generation: int
    result: asyncio.Future[str]
    task: asyncio.Task[str] | None = None
    client: MCPClient | None = None


@dataclass(slots=True)
class CallRequest:
    """One tool request sharing the actor request lifecycle."""

    identifier: int
    tool_name: str
    arguments: dict[str, object]
    abort_signal: AbortSignal
    generation: int
    result: asyncio.Future[StructuredToolResult]
    task: asyncio.Task[StructuredToolResult] | None = None
    client: MCPClient | None = None


@dataclass(frozen=True, slots=True)
class PromptFinished:
    request: PromptRequest
    task: asyncio.Task[str]
    result: str | None
    error: BaseException | None


@dataclass(frozen=True, slots=True)
class CallFinished:
    request: CallRequest
    task: asyncio.Task[StructuredToolResult]
    result: StructuredToolResult | None
    error: BaseException | None


@dataclass(frozen=True, slots=True)
class CancelRequest:
    """A cancellation message shared by all actor request types."""

    identifier: int
    acknowledged: asyncio.Future[None] | None = None


async def cancel_request(actor: "MCPServerActor", identifier: int) -> None:
    """Cancel one queued actor request and wait for ownership cleanup."""

    acknowledged = asyncio.get_running_loop().create_future()
    if actor._task is not None and not actor._task.done():
        actor._queue.put_nowait(CancelRequest(identifier, acknowledged))
        await asyncio.shield(acknowledged)


async def get_prompt(
    actor: "MCPServerActor",
    prompt_name: str,
    arguments: dict[str, str],
    *,
    generation: int,
) -> str:
    """Queue one prompt request with actor-owned cancellation."""

    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    actor._next_identifier += 1
    request = PromptRequest(
        actor._next_identifier,
        prompt_name,
        dict(arguments),
        generation,
        future,
    )
    if actor.is_terminal:
        raise RuntimeError(prompt_unavailable(actor.name))
    actor._queue.put_nowait(request)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await cancel_request(actor, request.identifier)
        raise


def handle_prompt(actor: "MCPServerActor", request: PromptRequest) -> None:
    if actor._closed or request.generation != actor._generation:
        set_exception(request.result, RuntimeError(prompt_unavailable(actor.name)))
        return
    if actor._status.state != "mounted" or actor._client is None:
        set_exception(request.result, RuntimeError(prompt_unavailable(actor.name)))
        return
    request.client = actor._client
    request.task = asyncio.create_task(
        invoke_prompt(actor, request, actor._client)
    )
    actor._requests[request.identifier] = request
    actor._children.add(request.task)
    request.task.add_done_callback(
        lambda done, prompt=request: queue_prompt_result(actor, done, prompt)
    )


async def invoke_prompt(
    actor: "MCPServerActor",
    request: PromptRequest,
    client: MCPClient,
) -> str:
    return await asyncio.wait_for(
        client.get_prompt(request.prompt_name, request.arguments),
        timeout=actor._setup_timeout,
    )


def queue_prompt_result(
    actor: "MCPServerActor",
    task: asyncio.Task[str],
    request: PromptRequest,
) -> None:
    if task.cancelled():
        actor._queue.put_nowait(
            PromptFinished(request, task, None, asyncio.CancelledError())
        )
        return
    try:
        result = task.result()
    except BaseException as exc:  # noqa: BLE001 - preserve task failure
        actor._queue.put_nowait(PromptFinished(request, task, None, exc))
    else:
        actor._queue.put_nowait(PromptFinished(request, task, result, None))


def handle_prompt_finished(
    actor: "MCPServerActor", message: PromptFinished
) -> None:
    actor._children.discard(message.task)
    request = actor._requests.pop(message.request.identifier, message.request)
    if isinstance(message.error, asyncio.CancelledError):
        return
    current = (
        not actor._closed
        and actor._client is request.client
        and request.generation == actor._generation
        and actor._status.state == "mounted"
    )
    if message.error is not None:
        if current:
            actor._degrade_current(_error_text(message.error))
        set_exception(request.result, message.error)
        return
    if not current:
        set_exception(
            request.result,
            RuntimeError(prompt_unavailable(actor.name)),
        )
        return
    if message.result is not None:
        set_result(request.result, message.result)


async def discover_prompts(
    client: MCPClient,
    server_name: str,
    timeout: float,
    notice: Callable[[str], None],
) -> tuple[MCPPrompt, ...]:
    capabilities = getattr(client, "capabilities", None)
    if type(capabilities) is not dict or "prompts" not in capabilities:
        if capabilities is not None:
            notice(
                f"mcp · {server_name} has no prompts capability; tools only"
            )
        return ()
    try:
        return tuple(
            await asyncio.wait_for(client.list_prompts(), timeout=timeout)
        )
    except Exception as exc:  # noqa: BLE001 - discovery stays tools-only
        notice(f"mcp · {server_name} prompts unavailable: {_error_text(exc)}")
        return ()


def prompt_unavailable(name: str) -> str:
    return f"MCP prompt server '{name}' is unavailable. Use /mcp reconnect {name}."


def set_result(future: asyncio.Future[object] | None, value: object) -> None:
    if future is not None and not future.done():
        future.set_result(value)


def set_exception(future: asyncio.Future[object], error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)


def resolve_prompt_message(message: object, server_name: str) -> bool:
    """Resolve a queued prompt message after actor termination."""

    if isinstance(message, PromptRequest):
        set_exception(message.result, RuntimeError(prompt_unavailable(server_name)))
    elif isinstance(message, PromptFinished):
        set_exception(
            message.request.result,
            RuntimeError(prompt_unavailable(server_name)),
        )
    else:
        return False
    return True


def cancel_prompt_request(request: PromptRequest, server_name: str) -> None:
    """Cancel one prompt task while closing the actor."""

    if request.task is not None:
        request.task.cancel()
    set_exception(request.result, RuntimeError(prompt_unavailable(server_name)))


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask failure
        return type(error).__name__


__all__ = [
    "CancelRequest",
    "CallFinished",
    "CallRequest",
    "cancel_request",
    "cancel_prompt_request",
    "PromptFinished",
    "PromptRequest",
    "discover_prompts",
    "prompt_unavailable",
    "resolve_prompt_message",
    "set_exception",
]
