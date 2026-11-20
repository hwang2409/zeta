"""One table for how a child run ends (design-child-termination-owner §6).

Every row drives a real parent ``AgentLoop`` (or serve) and checks the same
durable records: D1 child lifecycle state, D2 cancel marker, D4 parent marker,
D5 parent receipt, plus no running owner entry, open child store, or pending task.
Rows that fail today are strict xfails that must fail at the D1/D5 check
(``TerminalMismatch``); any other failure is a real failure.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from tests.support.gated_store import gate_store
from tests.support.owned import await_or_fail
from zeta.agent.presets import AGENT_PRESETS
from zeta.core.agent_progress import register_child_run_sink
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from zeta.providers.completion import owned_completion
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog

PROMPTS = {"start": "parent", "child prompt": "child", "grandchild prompt": "grandchild"}
STATUS = {"completed": "completed", "canceled": "canceled", "error": "failed", "failed": "failed"}


class TerminalMismatch(AssertionError):
    """D1 or D5 differs from the row's expected outcome."""


def _text(text: str) -> list:
    return [TextContent(text)]


def _call(name: str, call_id: str, **arguments: object) -> list:
    return [ToolUseContent(ToolCall(call_id, name, dict(arguments)))]


@dataclass
class Actor:
    turns: list[list | BaseException]
    gate_call: int | None = None
    calls: int = 0
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class Script(CompletionBackend):
    """Route each model call to its actor by the actor's prompt."""

    def __init__(self, actors: dict[str, Actor]) -> None:
        self.actors = actors

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        name = next(
            PROMPTS[block.text]
            for message in messages
            if message.role is MessageRole.USER
            for block in message.content
            if isinstance(block, TextContent) and block.text in PROMPTS
        )
        actor = self.actors[name]
        index, actor.calls = actor.calls, actor.calls + 1
        if index == actor.gate_call:
            actor.started.set()
            # wait_for, not await_or_fail: a canceled call must not leave the wait behind.
            await asyncio.wait_for(actor.release.wait(), 10)
        blocks = actor.turns[index] if index < len(actor.turns) else _text(f"{name} done")
        if isinstance(blocks, BaseException):
            raise blocks
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END, message=Message(MessageRole.ASSISTANT, blocks),
        )


@dataclass
class Run:
    tmp_path: Path
    mode: str  # fg | bg | explicit
    store: ConversationStore
    loop: AgentLoop
    backend: Script
    cleanup: AsyncExitStack
    turn: asyncio.Task | None = None
    ended: list[tuple[str, str]] = field(default_factory=list)
    baseline: set[asyncio.Task] = field(default_factory=set)

    @property
    def child(self) -> Actor:
        return self.backend.actors["child"]

    @property
    def grandchild(self) -> Actor:
        return self.backend.actors["grandchild"]

    @property
    def handle(self) -> str:
        return f"{self.store.session_id}:1"

    @property
    def child_dir(self) -> Path:
        return self.store.session_dir / "agents" / "1"

    def start(self) -> asyncio.Task:
        async def turn() -> None:
            async for _ in self.loop.run_turn("start", origin=MessageOrigin.USER):
                pass

        self.turn = asyncio.create_task(turn())
        return self.turn

    async def agent_cancel(self, handle: str | None = None) -> str:
        result = await self.loop.tool_registry.execute(
            ToolCall("cancel", "agent_cancel", {"handle": handle or self.handle})
        )
        return result["structuredContent"]["status"]

    async def finish_turn(self) -> BaseException | None:
        """Await the parent turn; return its exception (a cancel is not a test failure)."""
        assert self.turn is not None
        results = await await_or_fail(
            asyncio.gather(self.turn, return_exceptions=True), what="parent turn ends",
        )
        return results[0] if isinstance(results[0], BaseException) else None

    async def settle(self) -> None:
        await await_or_fail(self.loop._background_owner.wait(), what="child tree settles")


def _agent_arguments(mode: str, **extra: object) -> dict[str, object]:
    arguments: dict[str, object] = {"prompt": "child prompt", "description": "child run"}
    if mode == "bg":
        arguments["background"] = True
    if mode == "explicit":
        arguments["agent_type"] = "run"
    return {**arguments, **extra}


async def _open_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    child: Actor,
    *,
    grandchild: Actor | None = None,
    approval: bool = False,
    **agent_extra: object,
) -> Run:
    if mode == "explicit":
        monkeypatch.setitem(
            AGENT_PRESETS, "run", replace(AGENT_PRESETS["run"], explicit_finish=True),
        )
    parent = Actor([_call("agent", "child", **_agent_arguments(mode, **agent_extra))])
    backend = Script({
        "parent": parent, "child": child,
        "grandchild": grandchild or Actor([_text("grandchild done")]),
    })
    cleanup = AsyncExitStack()
    store = ConversationStore(tmp_path, session_id="parent")
    cleanup.callback(store.close)
    loop = AgentLoop(
        backend, store, skill_catalog=SkillCatalog.empty(),
        approval_policy=ApprovalPolicy(store=store) if approval else None,
    )
    loop.explicit_finish = mode == "explicit"
    cleanup.push_async_callback(loop.close)
    run = Run(tmp_path, mode, store, loop, backend, cleanup, baseline=set(asyncio.all_tasks()))
    cleanup.callback(register_child_run_sink(
        lambda kind, run_id, row, *_: kind == "ended" and run.ended.append((run_id, row["status"]))
    ))
    return run


def _child_store(path: Path) -> ConversationStore:
    return ConversationStore(path.parent, session_id=path.name, _read_only=True)


def _d1(path: Path) -> str | None:
    if not path.exists():
        return None
    child = _child_store(path)
    try:
        lifecycle = child.agent_lifecycle()
        return None if lifecycle is None else lifecycle["state"]
    finally:
        child.close()


def _d5(store: ConversationStore, mode: str, handle: str) -> str | None:
    if mode == "fg":
        result = store.tool_result("child")
        if result is None:
            return None
        if result.is_canceled:
            return "canceled"
        return "failed" if result.is_error else "completed"
    statuses = [
        entry.data["status"]
        for entry in store.agent_notifications(pending_only=False)
        if entry.data.get("kind") != "child_question"
        and entry.data.get("status") is not None
    ]
    return STATUS[statuses[-1]] if statuses else None


def _expect_terminal(d1: str | None, d5: str | None, expected: str) -> None:
    if d1 != expected:
        raise TerminalMismatch(f"D1 is {d1!r}, expected {expected!r} (D5 {d5!r})")
    if d5 != expected:
        raise TerminalMismatch(f"D5 is {d5!r}, expected {expected!r} (D1 {d1!r})")


async def _verify(run: Run, expected: str, *, child_exists: bool = True) -> None:
    """Close the parent, then check every durable record of the child's end."""
    await await_or_fail(run.loop.close(), what="parent close")
    owner = run.loop._background_owner
    assert not owner.running, "owner still runs a child"
    open_stores = [store.session_dir for store in owner._stores if store.directory_fd != -1]
    assert not open_stores, f"child stores still open: {open_stores}"
    leaked = [
        task for task in asyncio.all_tasks()
        if task not in run.baseline and not task.done() and task is not asyncio.current_task()
    ]
    assert not leaked, f"pending tasks: {leaked}"
    if not child_exists:
        assert not run.child_dir.exists() or _d1(run.child_dir) is None
        assert not run.store.agent_children(), "D4 left for a child that never started"
        return
    _expect_terminal(_d1(run.child_dir), _d5(run.store, run.mode, run.handle), expected)
    child = _child_store(run.child_dir)
    try:
        assert (child.agent_canceled() is not None) == (expected == "canceled"), "D2"
    finally:
        child.close()
    assert not run.store.agent_children(), "D4 not cleared"


async def _restart(run: Run) -> tuple[str | None, str | None]:
    """Resume the parent session in a fresh loop; return (D1, D5) after recovery."""
    store = ConversationStore(run.tmp_path, session_id="parent")
    loop = AgentLoop(Script({}), store, skill_catalog=SkillCatalog.empty())
    try:
        await await_or_fail(loop.activate(), what="recovery")
        return _d1(run.child_dir), _d5(store, run.mode, run.handle)
    finally:
        await await_or_fail(loop.close(), what="close recovered parent")
        store.close()


# rows ------------------------------------------------------------------------

Row = Callable[[Path, pytest.MonkeyPatch, str], Awaitable[None]]
ROWS: dict[str, Row] = {}


def row(name: str) -> Callable[[Row], Row]:
    def register(fn: Row) -> Row:
        ROWS[name] = fn
        return fn

    return register


@row("success")
async def _success(tmp_path, monkeypatch, mode):
    if mode == "explicit":
        child = Actor([_call("finish", "finish", status="done", summary="child finished")])
    else:
        child = Actor([_text("child complete")])
    run = await _open_run(tmp_path, monkeypatch, mode, child)
    async with run.cleanup:
        run.start()
        assert await run.finish_turn() is None
        await run.settle()
        await _verify(run, "completed")


@row("fail-child")
async def _fail_child(tmp_path, monkeypatch, mode):
    run = await _open_run(tmp_path, monkeypatch, mode, Actor([RuntimeError("model down")]))
    async with run.cleanup:
        run.start()
        assert await run.finish_turn() is None
        await run.settle()
        await _verify(run, "failed")


@row("cancel@model_turn")
async def _cancel_model_turn(tmp_path, monkeypatch, mode):
    run = await _open_run(tmp_path, monkeypatch, mode, Actor([_text("late")], gate_call=0))
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child provider call")
        assert await run.agent_cancel() == "cancellation_confirmed"
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")


@row("close@running")
async def _close_running(tmp_path, monkeypatch, mode):
    child = Actor([
        _call("agent", "grandchild", prompt="grandchild prompt", description="gc", background=True),
        _text("late"),
    ], gate_call=1)
    grandchild = Actor([_text("late")], gate_call=0)
    run = await _open_run(tmp_path, monkeypatch, mode, child, grandchild=grandchild)
    async with run.cleanup:
        run.start()
        await await_or_fail(run.child.started.wait(), what="child second call")
        await await_or_fail(run.grandchild.started.wait(), what="grandchild call")
        await await_or_fail(run.loop.close(), what="close with a running tree")
        await run.finish_turn()
        await _verify(run, "canceled")
        assert _d1(run.child_dir / "agents" / "1") == "canceled"


@row("cancel@finish_gate")
async def _cancel_finish_gate(tmp_path, monkeypatch, mode):
    """Bug 4: the grandchild's cancel write frees the gate before the child's lands."""
    child = Actor([
        _call("agent", "grandchild", prompt="grandchild prompt", description="gc", background=True),
        _text("child complete"),
    ])
    grandchild = Actor([_text("late")], gate_call=0)
    run = await _open_run(tmp_path, monkeypatch, mode, child, grandchild=grandchild)
    child_d1 = gate_store(
        monkeypatch, "finish_agent_lifecycle",
        when=lambda store, *_: store.session_dir == run.child_dir,
    )
    grandchild_cancel = gate_store(
        monkeypatch, "mark_agent_canceled_async",
        when=lambda store, *_: store.session_dir == run.child_dir / "agents" / "1",
    )
    # The grandchild's cancel write lands first; in fg the child also writes its
    # own D1 first (the RCA order). bg has no child D1 writer before the cancel.
    gate_store(
        monkeypatch, "mark_agent_canceled_async", hold=True,
        when=lambda store, *_: store.session_dir == run.child_dir,
    ).released = child_d1.done if mode == "fg" else grandchild_cancel.done
    async with run.cleanup:
        run.start()
        await await_or_fail(run.grandchild.started.wait(), what="grandchild call")
        run.loop._background_owner.cancel_subtree(run.handle)
        await run.finish_turn()
        await run.settle()
        await _verify(run, "canceled")
        assert _d1(run.child_dir / "agents" / "1") == "canceled"


# table -----------------------------------------------------------------------

def _xfail(reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=TerminalMismatch, reason=reason)


CASES = [
    ("success", "fg"),
    ("success", "bg"),
    ("success", "explicit"),
    ("fail-child", "fg"),
    ("fail-child", "bg"),
    ("cancel@model_turn", "fg"),
    ("cancel@model_turn", "bg"),
    ("close@running", "fg"),
    ("close@running", "bg"),
    pytest.param("cancel@finish_gate", "fg", marks=_xfail(
        "bug 4: the child finishes after the grandchild's cancel write and before its own",
    )),
    ("cancel@finish_gate", "bg"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "mode"), CASES)
async def test_child_run_termination(tmp_path, monkeypatch, name, mode):
    await ROWS[name](tmp_path, monkeypatch, mode)
