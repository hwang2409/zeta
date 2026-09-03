"""Prompt request handling owned by one MCP server actor."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .client import MCPClient, MCPPrompt


@dataclass(slots=True)
class PromptRequest:
    identifier: int
    prompt_name: str
    arguments: dict[str, str]
    generation: int
    result: asyncio.Future[str]
    task: asyncio.Task[str] | None = None
    client: MCPClient | None = None


@dataclass(frozen=True, slots=True)
class PromptFinished:
    request: PromptRequest
    task: asyncio.Task[str]
    result: str | None
    error: BaseException | None


class MCPPromptActorMixin:
    """Add serialized prompt calls to an MCP server actor."""

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str],
        *,
        generation: int,
    ) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._next_identifier += 1
        request = PromptRequest(
            self._next_identifier,
            prompt_name,
            dict(arguments),
            generation,
            future,
        )
        if self.is_terminal:
            raise RuntimeError(prompt_unavailable(self.name))
        self._queue.put_nowait(request)
        return await asyncio.shield(future)

    def _handle_prompt(self, request: PromptRequest) -> None:
        if self._closed or request.generation != self._generation:
            set_exception(request.result, RuntimeError(prompt_unavailable(self.name)))
            return
        if self._status.state != "mounted" or self._client is None:
            set_exception(request.result, RuntimeError(prompt_unavailable(self.name)))
            return
        self._dispatch_prompt(request, self._client)

    def _dispatch_prompt(
        self,
        request: PromptRequest,
        client: MCPClient,
    ) -> None:
        request.client = client
        request.task = asyncio.create_task(
            client.get_prompt(request.prompt_name, request.arguments)
        )
        self._prompt_calls[request.identifier] = request
        self._children.add(request.task)
        request.task.add_done_callback(
            lambda done, prompt=request: self._queue_prompt_result(done, prompt)
        )

    def _queue_prompt_result(
        self,
        task: asyncio.Task[str],
        request: PromptRequest,
    ) -> None:
        if task.cancelled():
            self._queue.put_nowait(
                PromptFinished(request, task, None, asyncio.CancelledError())
            )
            return
        try:
            result = task.result()
        except BaseException as exc:  # noqa: BLE001 - preserve task failure
            self._queue.put_nowait(PromptFinished(request, task, None, exc))
        else:
            self._queue.put_nowait(PromptFinished(request, task, result, None))

    def _handle_prompt_finished(self, message: PromptFinished) -> None:
        self._children.discard(message.task)
        self._prompt_calls.pop(message.request.identifier, None)
        current = (
            not self._closed
            and self._client is message.request.client
            and message.request.generation == self._generation
            and self._status.state == "mounted"
        )
        if message.error is not None:
            if current:
                self._degrade_current(_error_text(message.error))
            set_exception(message.request.result, message.error)
            return
        if not current:
            set_exception(
                message.request.result,
                RuntimeError(prompt_unavailable(self.name)),
            )
            return
        if message.result is not None:
            set_result(message.request.result, message.result)


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


def set_result(future: asyncio.Future[object], value: object) -> None:
    if not future.done():
        future.set_result(value)


def set_exception(future: asyncio.Future[object], error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)


def cancel_prompt_calls(calls: dict[int, PromptRequest], server_name: str) -> None:
    for request in calls.values():
        if request.task is not None:
            request.task.cancel()
        set_exception(request.result, RuntimeError(prompt_unavailable(server_name)))
    calls.clear()


def resolve_prompt_message(message: object, server_name: str) -> bool:
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


def _error_text(error: BaseException) -> str:
    try:
        return str(error).strip() or type(error).__name__
    except Exception:  # noqa: BLE001 - error reporting must not mask failure
        return type(error).__name__


__all__ = [
    "MCPPromptActorMixin",
    "PromptFinished",
    "PromptRequest",
    "cancel_prompt_calls",
    "discover_prompts",
    "prompt_unavailable",
    "resolve_prompt_message",
    "set_exception",
]
