"""Shared async transport cleanup and exception precedence."""

from __future__ import annotations

import asyncio
from typing import TypeVar

import httpx

ErrorT = TypeVar("ErrorT", bound=RuntimeError)


def task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def is_control_exception(value: BaseException) -> bool:
    return isinstance(value, (asyncio.CancelledError, GeneratorExit))


def control_priority(value: BaseException) -> int:
    if isinstance(value, asyncio.CancelledError):
        return 2
    if isinstance(value, GeneratorExit):
        return 1
    return 0


def request_error[ErrorT: RuntimeError](
    cause: httpx.HTTPError, error_type: type[ErrorT]
) -> ErrorT:
    error = error_type("request failed")
    error.__cause__ = cause
    return error


def merge_exception(
    primary: BaseException | None,
    cleanup: BaseException,
) -> BaseException:
    if primary is None:
        return cleanup
    primary_is_control = is_control_exception(primary)
    cleanup_is_control = is_control_exception(cleanup)
    if cleanup_is_control and not primary_is_control:
        cleanup.__cause__ = primary
        cleanup.__context__ = primary
        return cleanup
    if (
        cleanup_is_control
        and primary_is_control
        and control_priority(cleanup) > control_priority(primary)
    ):
        return cleanup
    if primary_is_control and not cleanup_is_control:
        primary.__context__ = cleanup
    return primary


async def cleanup_transport[ErrorT: RuntimeError](
    *,
    stream_context: object | None,
    entered: bool,
    client: object,
    owns_client: bool,
    primary_exception: BaseException | None,
    http_error_type: type[ErrorT],
) -> BaseException | None:
    if entered and stream_context is not None:
        try:
            await stream_context.__aexit__(  # type: ignore[attr-defined]
                type(primary_exception) if primary_exception else None,
                primary_exception,
                primary_exception.__traceback__ if primary_exception else None,
            )
        except BaseException as exc:
            cleanup = (
                request_error(exc, http_error_type)
                if isinstance(exc, httpx.HTTPError)
                else exc
            )
            primary_exception = merge_exception(primary_exception, cleanup)
    if owns_client:
        try:
            await client.aclose()  # type: ignore[attr-defined]
        except BaseException as exc:
            cleanup = (
                request_error(exc, http_error_type)
                if isinstance(exc, httpx.HTTPError)
                else exc
            )
            primary_exception = merge_exception(primary_exception, cleanup)
    if task_is_cancelling():
        primary_exception = merge_exception(
            primary_exception, asyncio.CancelledError()
        )
    return primary_exception
