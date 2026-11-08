"""One owner cancels a tool handler task: every route sends it at most one cancel."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.owned import await_or_fail
from zeta.core.abort import AbortSignal
from zeta.core.approval import ApprovalPolicy
from zeta.core.session import SessionManager
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.runtime import execution
from zeta.runtime.loop import AgentLoop
from zeta.server.abort_scope import abort_active_turn
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry


async def settle() -> None:
    # Loop order, not time: the ready queue is FIFO and no I/O is pending, so
    # 50 turns let every already-scheduled cancel request reach its target.
    for _ in range(50):
        await asyncio.sleep(0)


@dataclass
class Probe:
    """A tool handler that records every cancel request and delivery it gets."""

    mode: str = "wait"
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    exited: asyncio.Event = field(default_factory=asyncio.Event)
    forever: asyncio.Event = field(default_factory=asyncio.Event)
    delivered: int = 0
    requests: int | None = None
    caller: asyncio.Task[Any] | None = None

    async def __call__(
        self, arguments: dict[str, Any], abort_signal: AbortSignal
    ) -> str:
        del arguments
        task = asyncio.current_task()
        assert task is not None
        self.started.set()
        try:
            if self.mode == "value":
                return "done"
            if self.mode == "value_then_cancel_caller":
                assert self.caller is not None
                self.caller.cancel()
                return "done"
            if self.mode == "raises":
                raise ValueError("boom")
            if self.mode == "tool_canceled":
                raise execution._ToolCanceled()
            if self.mode == "cooperative":
                # Start waiting after the owner's signal watcher, as bash does.
                await asyncio.sleep(0)
                await asyncio.wait_for(abort_signal.wait(), 30)
                return "stopped on signal"
            # wait_for, not await_or_fail: a cancel here must not leave an inner
            # task pending. 30 s, so the row-level 10 s guards fire first.
            await asyncio.wait_for(self.forever.wait(), 30)
            return "unreachable"
        except asyncio.CancelledError:
            self.delivered += 1
            self.cleanup_started.set()
            while not self.release.is_set():
                try:
                    await asyncio.wait_for(self.release.wait(), 30)
                except asyncio.CancelledError:
                    self.delivered += 1
            raise
        finally:
            # Task.cancelling() counts every cancel() request, also the ones
            # that coalesce into one delivered CancelledError.
            self.requests = task.cancelling()
            self.exited.set()


@dataclass(frozen=True)
class Outcome:
    requests: tuple[int | None, ...]
    delivered: tuple[int, ...]
    result: tuple[str, ...]


def summarize(result: object) -> str:
    if isinstance(result, ToolResult):
        if result.is_canceled:
            return "canceled"
        return f"error:{result.content}" if result.is_error else str(result.content)
    return str(result)


class Harness:
    def __init__(self, tmp_path: Path, probes: list[Probe]) -> None:
        self.tmp_path = tmp_path
        self.probes = probes
        self.tasks: list[asyncio.Task[Any]] = []

    def outcome(self, result: tuple[str, ...]) -> Outcome:
        return Outcome(
            tuple(probe.requests for probe in self.probes),
            tuple(probe.delivered for probe in self.probes),
            result,
        )

    async def release_all(self) -> None:
        await settle()
        for probe in self.probes:
            if probe.started.is_set():
                probe.release.set()
                await await_or_fail(probe.exited.wait(), what="probe exit")

    def turn(self, *, parallel: bool = False) -> tuple[AgentLoop, asyncio.Task[Any]]:
        registry = ToolRegistry(
            self.tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty()
        )
        calls = []
        for index, probe in enumerate(self.probes):
            registry.register(
                f"probe{index}",
                probe,
                parallel_safe=parallel,
                requires_approval=False,
            )
            calls.append(ToolCall(f"call-{index}", f"probe{index}", {}))
        backend = FakeBackend(
            [ScriptedTurn([], calls), ScriptedTurn([TextContent("after")])]
        )
        self.store = ConversationStore(self.tmp_path)
        loop = AgentLoop(
            backend, self.store, registry=registry, skill_catalog=SkillCatalog.empty()
        )
        task = asyncio.create_task(self._consume(loop))
        self.tasks.append(task)
        return loop, task

    @staticmethod
    async def _consume(loop: AgentLoop) -> None:
        async for _event in loop.run_turn("start", origin=MessageOrigin.USER):
            pass

    async def all_started(self) -> None:
        for probe in self.probes:
            await await_or_fail(probe.started.wait(), what="probe start")

    async def all_in_cleanup(self) -> None:
        for probe in self.probes:
            await await_or_fail(probe.cleanup_started.wait(), what="probe cleanup")

    async def turn_end(self, task: asyncio.Task[Any]) -> str:
        await await_or_fail(
            asyncio.wait({task}), what="turn end"
        )
        return "turn cancelled" if task.cancelled() else "turn done"

    def transcript(self) -> tuple[str, ...]:
        return tuple(
            summarize(message.tool_result)
            for message in self.store.messages()
            if message.tool_result is not None
        )

    async def close(self) -> None:
        for probe in self.probes:
            probe.release.set()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await await_or_fail(
                asyncio.wait(self.tasks), what="harness task cleanup"
            )


async def owner_call(
    probe: Probe,
    *,
    cancel_first: bool = False,
    signal_first: bool = False,
) -> str:
    """Call the owner directly from a caller task, as registry.execute does."""

    signal = AbortSignal()
    caller = asyncio.current_task()
    assert caller is not None
    probe.caller = caller
    if signal_first:
        signal.abort()
    if cancel_first:
        caller.cancel()
    try:
        result = await execution.run_handler_with_abort(
            probe, {}, signal, None, "call-0"
        )
    except asyncio.CancelledError:
        return "raised CancelledError"
    return summarize(result)


Scenario = Callable[[Harness], Awaitable[Outcome]]


def direct(mode: str, **kwargs: bool) -> Scenario:
    async def run(h: Harness) -> Outcome:
        task = asyncio.create_task(owner_call(h.probes[0], **kwargs))
        h.tasks.append(task)
        if mode == "wait":
            await h.all_in_cleanup()
            await h.release_all()
        return h.outcome((await await_or_fail(task, what="owner call"),))

    run.mode = mode  # type: ignore[attr-defined]
    return run


async def outer_cancel_at_each_loop_turn(h: Harness) -> Outcome:
    """Signal, then cancel the caller k loop turns later, for each k.

    Across k the outer cancel lands in the owner's wait, in its two yields
    after the signal, and after its cancel was sent.
    """
    seen = set()
    for turns in range(8):
        probe = Probe()
        h.probes = [probe]
        signal = AbortSignal()
        task = asyncio.create_task(
            execution.run_handler_with_abort(probe, {}, signal, None, "call-0")
        )
        h.tasks.append(task)
        await await_or_fail(probe.started.wait(), what="probe start")
        signal.abort()
        for _ in range(turns):
            await asyncio.sleep(0)
        task.cancel()
        await h.release_all()
        result = await await_or_fail(task, what="owner result")
        seen.add((probe.requests, probe.delivered, summarize(result)))
    assert len(seen) == 1, f"(requests, delivered, result) differ by turn: {sorted(seen, key=str)}"
    requests, delivered, result = seen.pop()
    return Outcome((requests,), (delivered,), (result,))


async def cooperative_handler_sees_signal_first(h: Harness) -> Outcome:
    """A handler that stops on the signal itself (bash, agent) gets no cancel."""
    signal = AbortSignal()
    task = asyncio.create_task(
        execution.run_handler_with_abort(h.probes[0], {}, signal, None, "call-0")
    )
    h.tasks.append(task)
    await h.all_started()
    signal.abort()
    return h.outcome((summarize(await await_or_fail(task, what="owner result")),))


cooperative_handler_sees_signal_first.mode = "cooperative"  # type: ignore[attr-defined]


async def signal_only(h: Harness) -> Outcome:
    loop, task = h.turn()
    await h.all_started()
    loop.abort()
    await h.release_all()
    end = await h.turn_end(task)
    return h.outcome((*h.transcript(), end))


async def turn_first_then_signal(h: Harness) -> Outcome:
    loop, task = h.turn()
    await h.all_started()
    task.cancel()
    await h.all_in_cleanup()
    loop.abort()
    await h.release_all()
    end = await h.turn_end(task)
    return h.outcome((*h.transcript(), end))


def serve_late_turn(scope: str, *, parallel: bool = False) -> Scenario:
    """Signal first; the server's turn cancel arrives while cleanup runs."""

    async def run(h: Harness) -> Outcome:
        loop, task = h.turn(parallel=parallel)
        await h.all_started()

        async def terminate_approvals(**_: object) -> None:
            await h.all_in_cleanup()

        abort = asyncio.create_task(
            abort_active_turn(
                task,
                scope=scope,  # type: ignore[arg-type]
                loop=loop,
                terminate_approvals=terminate_approvals,
                before_cancel=lambda: None,
            )
        )
        h.tasks.append(abort)
        await h.all_in_cleanup()
        await h.release_all()
        aborted = await await_or_fail(abort, what="abort result")
        end = await h.turn_end(task)
        return h.outcome((*h.transcript(), f"aborted={aborted}", end))

    return run


async def serve_abort_then_disconnect(h: Harness) -> Outcome:
    """A session abort, then a second turn cancel (disconnect) while it drains."""
    loop, task = h.turn()
    await h.all_started()

    async def terminate_approvals(**_: object) -> None:
        return None

    abort = asyncio.create_task(
        abort_active_turn(
            task,
            scope="session",
            loop=loop,
            terminate_approvals=terminate_approvals,
            before_cancel=lambda: None,
        )
    )
    h.tasks.append(abort)
    await h.all_in_cleanup()
    await settle()
    task.cancel()
    await h.release_all()
    aborted = await await_or_fail(abort, what="abort result")
    end = await h.turn_end(task)
    return h.outcome((*h.transcript(), f"aborted={aborted}", end))


async def outer_only_twice(h: Harness) -> Outcome:
    _loop, task = h.turn()
    await h.all_started()
    task.cancel()
    await h.all_in_cleanup()
    task.cancel()
    await h.release_all()
    end = await h.turn_end(task)
    return h.outcome((*h.transcript(), end))


async def shutdown_during_cleanup(h: Harness) -> Outcome:
    loop, task = h.turn()
    await h.all_started()
    task.cancel()
    await h.all_in_cleanup()
    close = asyncio.create_task(loop.close())
    h.tasks.append(close)
    await h.release_all()
    await await_or_fail(close, what="loop close")
    end = await h.turn_end(task)
    return h.outcome((*h.transcript(), end))


async def durable_inline(h: Harness) -> Outcome:
    """The pipeline's durable abort: loop.abort(), then cancel the resume task."""
    manager = SessionManager(h.tmp_path / "zeta-home")
    opened = await asyncio.to_thread(
        manager.create, provider="codex", model="offline", cwd=h.tmp_path
    )
    policy = ApprovalPolicy(store=opened.store)
    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"probe0": h.probes[0]},
        approval_policy=policy,
        skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("call-0", "probe0", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    assert policy.approve(call.id)
    task = asyncio.create_task(loop.resume_pending_tool(call.id))
    h.tasks.append(task)
    await h.all_started()
    loop.abort()
    loop.finalize_canceled(call.id)
    task.cancel()
    await h.release_all()
    await await_or_fail(asyncio.wait({task}), what="durable task end")
    results = tuple(
        summarize(message.tool_result)
        for message in opened.store.messages()
        if message.tool_result is not None
    )
    return h.outcome(results)


async def hung_cleanup_is_bounded(h: Harness) -> Outcome:
    """The abort returns while cleanup still runs; the detached task ends later."""
    loop, task = h.turn()
    await h.all_started()

    async def terminate_approvals(**_: object) -> None:
        return None

    aborted = await await_or_fail(
        abort_active_turn(
            task,
            scope="session",
            loop=loop,
            terminate_approvals=terminate_approvals,
            before_cancel=lambda: None,
        ),
        what="abort result",
    )
    still_in_cleanup = not h.probes[0].exited.is_set()
    await h.release_all()
    return h.outcome(
        (*h.transcript(), f"aborted={aborted}", f"returned in cleanup={still_in_cleanup}")
    )


async def next_turn_after_abort(h: Harness) -> Outcome:
    """After an aborted turn, the next turn runs the handler to its value."""
    loop, task = h.turn()
    await h.all_started()
    loop.abort()
    await h.release_all()
    first = await h.turn_end(task)
    probe = Probe(mode="value")
    h.probes.append(probe)
    loop.tool_registry.register("probe1", probe, requires_approval=False)
    loop.backend = FakeBackend(
        [
            ScriptedTurn([], [ToolCall("call-1", "probe1", {})]),
            ScriptedTurn([TextContent("after")]),
        ]
    )
    second = asyncio.create_task(Harness._consume(loop))
    h.tasks.append(second)
    end = await h.turn_end(second)
    return h.outcome((*h.transcript(), first, end))


ROWS: list[Any] = [
    # The owner's own mapping, called the way registry.execute calls it.
    pytest.param(direct("value"), 1, False, Outcome((0,), (0,), ("done",)), id="success"),
    pytest.param(
        direct("value_then_cancel_caller"), 1, False,
        Outcome((0,), (0,), ("done",)),
        id="outer-cancel-after-handler-returned",
    ),
    pytest.param(direct("raises"), 1, False, Outcome((0,), (0,), ("error:boom",)), id="body-raises"),
    pytest.param(direct("tool_canceled"), 1, False, Outcome((0,), (0,), ("canceled",)), id="tool-canceled"),
    pytest.param(
        direct("wait", cancel_first=True), 1, False,
        Outcome((1,), (1,), ("raised CancelledError",)),
        id="outer-cancel-before-owner-first-await",
    ),
    pytest.param(
        direct("wait", cancel_first=True, signal_first=True), 1, False,
        Outcome((1,), (1,), ("canceled",)),
        id="outer-cancel-before-owner-first-await-with-signal",
    ),
    pytest.param(
        cooperative_handler_sees_signal_first, 1, False,
        Outcome((0,), (0,), ("stopped on signal",)),
        id="cooperative-handler-sees-signal-first",
    ),
    pytest.param(outer_cancel_at_each_loop_turn, 1, False, Outcome((1,), (1,), ("canceled",)), id="outer-cancel-at-each-owner-await"),
    # Every route that reaches a running handler, through the real loop.
    pytest.param(signal_only, 1, False, Outcome((1,), (1,), ("canceled", "turn done")), id="signal-only"),
    pytest.param(turn_first_then_signal, 1, False, Outcome((1,), (1,), ("canceled", "turn cancelled")), id="turn-cancel-first-then-signal"),
    pytest.param(
        serve_late_turn("session"), 1, False,
        Outcome((1,), (1,), ("canceled", "aborted=True", "turn cancelled")),
        id="serve-session-signal-first-late-turn-cancel",
    ),
    pytest.param(
        serve_late_turn("foreground"), 1, False,
        Outcome((1,), (1,), ("canceled", "aborted=True", "turn cancelled")),
        id="serve-foreground-signal-first-late-turn-cancel",
    ),
    pytest.param(
        serve_late_turn("session", parallel=True), 2, False,
        Outcome((1, 1), (1, 1), ("canceled", "canceled", "aborted=True", "turn cancelled")),
        id="parallel-signal-first-late-turn-cancel",
    ),
    pytest.param(
        serve_abort_then_disconnect, 1, False,
        Outcome((1,), (1,), ("canceled", "aborted=True", "turn cancelled")),
        id="duplicate-turn-cancel-while-draining",
    ),
    pytest.param(outer_only_twice, 1, False, Outcome((1,), (1,), ("canceled", "turn cancelled")), id="duplicate-outer-cancel-no-signal"),
    pytest.param(shutdown_during_cleanup, 1, False, Outcome((1,), (1,), ("canceled", "turn cancelled")), id="shutdown-during-cleanup"),
    pytest.param(durable_inline, 1, False, Outcome((1,), (1,), ("canceled",)), id="durable-resume-abort"),
    pytest.param(
        hung_cleanup_is_bounded, 1, True,
        Outcome((1,), (1,), ("canceled", "aborted=True", "returned in cleanup=True")),
        id="hung-cleanup-bounded",
    ),
    pytest.param(
        next_turn_after_abort, 1, False,
        Outcome((1, 0), (1, 0), ("canceled", "done", "turn done", "turn done")),
        id="next-turn-after-abort",
    ),
]


@pytest.mark.parametrize(("scenario", "probe_count", "hang", "expected"), ROWS)
async def test_one_owner_cancels_the_handler_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    scenario: Scenario,
    probe_count: int,
    hang: bool,
    expected: Outcome,
) -> None:
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    mode = getattr(scenario, "mode", "wait")
    harness = Harness(tmp_path, [Probe(mode=mode) for _ in range(probe_count)])
    if hang:
        # raising=False: the unchanged base has no bound, so its row times out.
        monkeypatch.setattr(
            execution, "HANDLER_CANCEL_DRAIN_SECONDS", 0.01, raising=False
        )
    try:
        with caplog.at_level(logging.WARNING, logger=execution.__name__):
            outcome = await scenario(harness)
        assert outcome == expected
        detached_logs = [r.getMessage() for r in caplog.records if "detached" in r.getMessage()]
        assert detached_logs == (
            ["tool call-0: cleanup still running 0 s after its cancel; detached"] if hang else []
        )
    finally:
        await harness.close()
        gc.collect()
        loop.set_exception_handler(previous_handler)
    assert loop_errors == []
