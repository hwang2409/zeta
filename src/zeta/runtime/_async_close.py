from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..providers.must_complete_close import task_is_cancelling

_BACKGROUND_CLOSE_TASKS: set[asyncio.Task[object]] = set()
CloseFailureReporter = Callable[[int, BaseException], None]


@dataclass(frozen=True, slots=True)
class CloseFailure:
    index: int
    error: BaseException


@dataclass(frozen=True, slots=True)
class CloseOutcome:
    """Failures from one aggregate close; an empty tuple means success."""

    failures: tuple[CloseFailure, ...]


def _exception_type_name(error: BaseException) -> str:
    name = f"{type(error).__module__}.{type(error).__qualname__}"
    return name.encode()[:128].decode(errors="ignore")


def _primary_outcome(primary_exception: BaseException | None) -> str:
    if isinstance(primary_exception, asyncio.CancelledError):
        return "cancelled"
    return "error" if primary_exception is not None else "success"


def cleanup_failure_context(
    error: BaseException,
    *,
    stage: str,
    index: int,
    primary_exception: BaseException | None = None,
) -> dict[str, object]:
    return {
        "exception_type": _exception_type_name(error),
        "stage": stage,
        "index": index,
        "primary_outcome": _primary_outcome(primary_exception),
    }


def report_loop_cleanup_failure(
    error: BaseException,
    *,
    stage: str,
    index: int,
    primary_exception: BaseException | None = None,
) -> None:
    asyncio.get_running_loop().call_exception_handler(
        cleanup_failure_context(
            error,
            stage=stage,
            index=index,
            primary_exception=primary_exception,
        )
    )


def is_control_exception(value: BaseException) -> bool:
    return isinstance(value, (asyncio.CancelledError, GeneratorExit))


def control_priority(value: BaseException) -> int:
    if isinstance(value, asyncio.CancelledError):
        return 2
    if isinstance(value, GeneratorExit):
        return 1
    return 0


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


def _release_background_close(
    task: asyncio.Task[object],
    index: int,
    report_failure: CloseFailureReporter | None,
) -> None:
    _BACKGROUND_CLOSE_TASKS.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is None or is_control_exception(error):
        return
    if report_failure is None:
        report_loop_cleanup_failure(error, stage="background_close", index=index)
    else:
        with contextlib.suppress(BaseException):
            report_failure(index, error)


def retain_background_closes(
    tasks: tuple[asyncio.Task[object], ...],
    report_failure: CloseFailureReporter | None,
) -> None:
    for index, task in enumerate(tasks):
        _BACKGROUND_CLOSE_TASKS.add(task)
        task.add_done_callback(
            lambda done, index=index: _release_background_close(
                done, index, report_failure
            )
        )


async def close_transport_resources(
    *closes: Awaitable[object],
    report_failure: CloseFailureReporter | None = None,
) -> CloseOutcome:
    """Close every owned resource and return each ordinary failure."""

    created: list[asyncio.Task[object]] = []
    try:
        for close in closes:
            created.append(asyncio.create_task(close))
        tasks = tuple(created)
        if task_is_cancelling():
            await asyncio.sleep(0)
            for task in tasks:
                task.cancel()
        results = tuple(await asyncio.gather(*tasks, return_exceptions=True))
        cancellation = next(
            (result for result in results if isinstance(result, asyncio.CancelledError)),
            None,
        )
        if cancellation is not None:
            raise cancellation
        return CloseOutcome(
            tuple(
                CloseFailure(index, result)
                for index, result in enumerate(results)
                if isinstance(result, BaseException)
            )
        )
    finally:
        tasks = tuple(created)
        pending = tuple(task for task in tasks if not task.done())
        for task in pending:
            task.cancel()
        if task_is_cancelling():
            retain_background_closes(tasks, report_failure)
        elif pending:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                retain_background_closes(tasks, report_failure)
                raise


async def close_resource(resource: object) -> None:
    """Close one async resource through the shared cancellation-safe owner."""

    outcome = await close_transport_resources(
        resource.aclose()  # type: ignore[attr-defined]
    )
    if outcome.failures:
        raise outcome.failures[0].error
